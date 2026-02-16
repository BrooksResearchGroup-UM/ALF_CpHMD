"""pH-replica exchange for CpHMD simulations.

Implements Python-based combined pH + lambda replica exchange, replacing
the legacy CHARMM ``REPD EXLM MSPH`` command. After each dynamics segment,
adjacent pH replicas attempt coordinate/velocity/lambda swaps via restart
file exchange.

Acceptance criterion (no energy re-evaluation needed):
    Δ = Σ_k [sign_k · ln(10) · δpKa] · (λ_k^i - λ_k^j)
    Accept if Δ ≤ 0, else with probability exp(−Δ)

where sign_k = +1 (UPOS), −1 (UNEG), 0 (NONE) from the TAG field.

Requires pyCHARMM >= 0.3.0 (checked at ReplicaExchanger init).
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReplicaExchangeConfig:
    """Configuration for pH-replica exchange.

    Attributes:
        enabled: Whether replica exchange is active.
        exchange_freq: MD steps between exchange attempts.
    """

    enabled: bool = False
    exchange_freq: int = 1000


@dataclass
class ExchangeState:
    """Persistent state tracking exchange attempt/acceptance statistics.

    Tracks per-pair (adjacent replicas) and total counts across runs.
    JSON-persistent following the same pattern as EWBSState.
    """

    attempted: list[int] = field(default_factory=list)
    accepted: list[int] = field(default_factory=list)
    total_attempted: int = 0
    total_accepted: int = 0
    # Tracks which original configuration is at which rank after exchanges.
    # permutation[rank] = original_config_index. Identity at start.
    permutation: list[int] = field(default_factory=list)

    def ensure_size(self, npairs: int) -> None:
        """Ensure attempt/acceptance lists and permutation have the right length."""
        while len(self.attempted) < npairs:
            self.attempted.append(0)
        while len(self.accepted) < npairs:
            self.accepted.append(0)
        if not self.permutation:
            self.permutation = list(range(npairs + 1))

    def record(self, pair_idx: int, accepted: bool) -> None:
        """Record an exchange attempt result."""
        self.attempted[pair_idx] += 1
        self.total_attempted += 1
        if accepted:
            self.accepted[pair_idx] += 1
            self.total_accepted += 1

    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate."""
        if self.total_attempted == 0:
            return 0.0
        return self.total_accepted / self.total_attempted

    def pair_acceptance_rates(self) -> list[float]:
        """Per-pair acceptance rates."""
        rates = []
        for a, t in zip(self.accepted, self.attempted):
            rates.append(a / t if t > 0 else 0.0)
        return rates

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        data = {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "total_attempted": self.total_attempted,
            "total_accepted": self.total_accepted,
            "permutation": self.permutation,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ExchangeState":
        """Load state from JSON file."""
        with open(path) as f:
            data = json.load(f)
        state = cls()
        state.attempted = data.get("attempted", [])
        state.accepted = data.get("accepted", [])
        state.total_attempted = data.get("total_attempted", 0)
        state.total_accepted = data.get("total_accepted", 0)
        state.permutation = data.get("permutation", [])
        return state


class ReplicaExchanger:
    """Manages pH-replica exchange attempts between dynamics segments.

    All ranks compute the same exchange decisions deterministically
    (shared seed + even/odd pair alternation), so no broadcast is needed.
    Only ``MPI_Allgather`` of lambda arrays is required.

    Args:
        config: ReplicaExchangeConfig instance.
        comm: MPI communicator (``MPI.COMM_WORLD``).
        rank: This process's MPI rank.
        size: Total number of MPI processes.
    """

    MIN_PYCHARMM_VERSION = (0, 3, 0)

    def __init__(self, config: ReplicaExchangeConfig, comm, rank: int, size: int):
        self.config = config
        self.comm = comm
        self.rank = rank
        self.size = size
        self.state = ExchangeState()

        self._check_pycharmm_version()

    def _check_pycharmm_version(self) -> None:
        """Validate pyCHARMM version meets minimum requirement."""
        try:
            import pycharmm

            version_str = getattr(pycharmm, "__version__", "0.0.0")
            parts = version_str.split(".")[:3]
            version = tuple(int(x) for x in parts)
        except (ImportError, ValueError):
            version = (0, 0, 0)
            version_str = "unknown"

        if version < self.MIN_PYCHARMM_VERSION:
            min_ver = ".".join(str(v) for v in self.MIN_PYCHARMM_VERSION)
            raise RuntimeError(
                f"Replica exchange requires pyCHARMM >= {min_ver} "
                f"(found {version_str}). Upgrade: pip install pycharmm>={min_ver}"
            )

    def compute_n_segments(self, nsteps_prod: int) -> int:
        """Compute the number of dynamics segments for a production run.

        Args:
            nsteps_prod: Total production steps.

        Returns:
            Number of segments (nsteps_prod // exchange_freq).
        """
        return nsteps_prod // self.config.exchange_freq

    def attempt_exchange(
        self,
        segment_idx: int,
        run_idx: int,
        lmd_path: Path,
        rst_path: Path,
        nblocks: int,
        patch_info,
        delta_pKa: float,
        temperature: float,
    ) -> Path | None:
        """Attempt replica exchange after a dynamics segment.

        All ranks participate: each reads its own lambdas, Allgather
        collects them, then all ranks compute the same deterministic
        decisions.

        Args:
            segment_idx: Current segment index (used for seed + even/odd).
            run_idx: Current ALF run index (used for seed).
            lmd_path: Path to this segment's LMD file.
            rst_path: Path to this segment's restart file.
            nblocks: Number of MSLD blocks.
            patch_info: DataFrame with TAG column for sign determination.
            delta_pKa: pH spacing between adjacent replicas.
            temperature: Simulation temperature in Kelvin.

        Returns:
            Partner's restart path if exchange accepted, None otherwise.
        """
        nreps = self.size

        if nreps < 2:
            return None

        npairs = nreps - 1
        self.state.ensure_size(npairs)

        # Read last-frame lambdas from this rank's LMD
        my_lambdas = self._read_last_frame_lambdas(lmd_path, nblocks)

        # Allgather: all ranks collect the full lambda matrix [nreps, nblocks-1]
        # (nblocks-1 because env block is stripped by read_lambda_binary)
        all_lambdas = np.zeros((nreps, len(my_lambdas)), dtype=np.float64)
        self.comm.Allgather(
            np.ascontiguousarray(my_lambdas, dtype=np.float64),
            all_lambdas,
        )

        # Build sign array from patch_info
        sign_array = self._build_sign_array(patch_info)

        # Compute beta = 1 / (kB * T) in CHARMM units (kcal/mol)
        KB = 0.001987191  # kcal/(mol·K)
        beta = 1.0 / (KB * temperature)

        # Deterministic RNG: same seed on all ranks → same decisions
        seed = run_idx * 100000 + segment_idx
        rng = np.random.RandomState(seed)

        # Even/odd alternation for detailed balance:
        # even segments try pairs (0,1), (2,3), (4,5), ...
        # odd segments try pairs (1,2), (3,4), (5,6), ...
        parity = segment_idx % 2

        partner_rank = None

        for pair_idx in range(parity, npairs, 2):
            rank_i = pair_idx
            rank_j = pair_idx + 1

            # Compute exchange criterion
            delta = self._compute_delta(
                all_lambdas[rank_i],
                all_lambdas[rank_j],
                sign_array,
                delta_pKa,
                beta,
            )

            # Accept/reject
            rand_val = rng.random()
            swap = delta <= 0.0 or rand_val < math.exp(-delta)

            self.state.record(pair_idx, swap)

            if swap:
                logger.debug(
                    "Segment %d: swap pair (%d,%d) Δ=%.4f accepted",
                    segment_idx, rank_i, rank_j, delta,
                )
                # Update permutation to reflect the exchange
                self.state.permutation[rank_i], self.state.permutation[rank_j] = (
                    self.state.permutation[rank_j],
                    self.state.permutation[rank_i],
                )
                # Track if this rank is involved
                if self.rank == rank_i:
                    partner_rank = rank_j
                elif self.rank == rank_j:
                    partner_rank = rank_i
            else:
                # Advance RNG state consistently even if pair not attempted
                pass

        if partner_rank is None:
            return None

        # Construct partner's restart path by positional parsing.
        # Filename format: {name}_{sim_type}.{k}.{replica_idx}.{seg_tag}.rst
        # parts[-3] is always the replica_idx field.
        parts = rst_path.name.split(".")
        parts[-3] = str(partner_rank)
        partner_rst_name = ".".join(parts)
        partner_rst = rst_path.parent / partner_rst_name

        if partner_rst.exists():
            return partner_rst
        else:
            logger.warning(
                "Partner restart file not found: %s — skipping swap", partner_rst
            )
            return None

    def _read_last_frame_lambdas(self, lmd_path: Path, nblocks: int) -> np.ndarray:
        """Read lambda values from the last frame of an LMD file.

        Uses fast seek-to-end rather than reading the entire file.

        Args:
            lmd_path: Path to binary .lmd file.
            nblocks: Number of MSLD blocks (including env).

        Returns:
            1D array of lambda values (nblocks - 1), env block stripped.
        """
        from cphmd.utils.lambda_io import read_lambda_binary

        data, _ = read_lambda_binary(lmd_path)
        if len(data) == 0:
            logger.warning("Empty LMD file: %s — returning zeros", lmd_path)
            return np.zeros(nblocks - 1)

        # data shape: (nsteps, 1+nblocks-1) — col 0 is time, cols 1: are lambdas
        return data[-1, 1:]

    @staticmethod
    def _compute_delta(
        lambda_i: np.ndarray,
        lambda_j: np.ndarray,
        sign_array: np.ndarray,
        delta_pKa: float,
        beta: float,
    ) -> float:
        """Compute the exchange criterion Δ.

        Derived from the Metropolis criterion for same-temperature pH-REMD:

            Δ = β · [V(pH_i, λ^j) + V(pH_j, λ^i) − V(pH_i, λ^i) − V(pH_j, λ^j)]

        where V_pH = kBT · ln(10) · Σ sign_k · (pKa_k − pH) · λ_k.

        After β · kBT cancellation:

            Δ = ln(10) · δpKa · Σ_k sign_k · (λ_k^j − λ_k^i)

        Accept if Δ ≤ 0, else with probability exp(−Δ).

        Args:
            lambda_i: Lambda values for replica i (lower pH).
            lambda_j: Lambda values for replica j (higher pH).
            sign_array: Sign per block (+1 UPOS, −1 UNEG, 0 NONE).
            delta_pKa: pH spacing (pH_j − pH_i > 0).
            beta: Unused (kept for API compatibility).

        Returns:
            Exchange criterion value (dimensionless).
        """
        ln10 = np.log(10.0)
        diff = lambda_j - lambda_i
        return float(ln10 * delta_pKa * np.dot(sign_array, diff))

    @staticmethod
    def _build_sign_array(patch_info) -> np.ndarray:
        """Build per-block sign array from patch_info TAG column.

        UPOS → +1, UNEG → −1, NONE → 0.

        Args:
            patch_info: DataFrame with TAG column.

        Returns:
            1D array of signs, one per non-env block.
        """
        from cphmd.core.cphmd_params import parse_tag_value

        signs = []
        for _, row in patch_info.iterrows():
            tag_type, _ = parse_tag_value(row["TAG"])
            if tag_type == "UPOS":
                signs.append(1.0)
            elif tag_type == "UNEG":
                signs.append(-1.0)
            else:
                signs.append(0.0)
        return np.array(signs)

    def write_exchange_log(self, path: Path, run_idx: int) -> None:
        """Write exchange statistics to a human-readable log file.

        Args:
            path: Output file path.
            run_idx: Current run index for the header.
        """
        rates = self.state.pair_acceptance_rates()
        with open(path, "w") as f:
            f.write(f"# Replica Exchange Statistics — Run {run_idx}\n")
            f.write(f"# Overall: {self.state.total_accepted}/{self.state.total_attempted} "
                    f"({self.state.acceptance_rate:.1%})\n")
            f.write("#\n")
            f.write("# Pair  Attempted  Accepted  Rate\n")
            for i, (att, acc, rate) in enumerate(
                zip(self.state.attempted, self.state.accepted, rates)
            ):
                f.write(f"  {i}↔{i+1}  {att:9d}  {acc:8d}  {rate:.3f}\n")
            if self.state.permutation:
                f.write(f"# Current permutation (config→rank): {self.state.permutation}\n")

    def load_state(self, path: Path) -> None:
        """Load exchange state from disk."""
        if path.exists():
            self.state = ExchangeState.load(path)

    def save_state(self, path: Path) -> None:
        """Save exchange state to disk."""
        self.state.save(path)
