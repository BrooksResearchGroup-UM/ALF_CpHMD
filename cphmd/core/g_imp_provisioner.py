"""G_imp entropy profile provisioning for WHAM analysis.

Manages the 3-tier fallback for G_imp data:
1. Existing G_imp/ in input folder (validate bins if configured)
2. Bundled cphmd/data/G_imp/ (validate bins if configured)
3. Compute via Monte Carlo using cphmd.core.entropy
"""

import shutil
from pathlib import Path

import numpy as np


class GImpProvisioner:
    """Provisions G_imp entropy profiles for WHAM/LMALF analysis.

    Stateless — all configuration provided via constructor. No CHARMM or MPI
    dependency; safe to use from any context.

    Args:
        input_folder: Path to the prepared system folder
        nsubs: Array of subsite counts per site
        fnex: FNEX softmax parameter (default 5.5)
        cutlsum: G12 conditional threshold (default 0.8)
        g_imp_bins: Expected bin count, or None for no validation
    """

    def __init__(
        self,
        input_folder: Path,
        nsubs: list[int] | np.ndarray,
        fnex: float = 5.5,
        cutlsum: float = 0.8,
        g_imp_bins: int | None = None,
    ):
        self.input_folder = Path(input_folder)
        self.nsubs = list(nsubs) if nsubs is not None else []
        self.fnex = fnex
        self.cutlsum = cutlsum
        self.g_imp_bins = g_imp_bins

    def ensure_available(self) -> None:
        """Ensure G_imp entropy data is available for WHAM.

        Priority:
        1. Use existing G_imp/ in input folder (validate bins if configured)
        2. Copy from bundled cphmd/data/G_imp/ (validate bins if configured)
        3. Compute via Monte Carlo using cphmd.core.entropy
        """
        dst = self.input_folder / "G_imp"
        expected_bins = self.g_imp_bins

        # --- Tier 1: use existing G_imp/ ---
        if dst.exists():
            actual_bins = self.detect_bins(dst)
            if expected_bins is not None and actual_bins != expected_bins:
                print(
                    f"G_imp: existing has bins={actual_bins}, "
                    f"need {expected_bins}. Regenerating..."
                )
                if dst.is_symlink():
                    dst.unlink()
                else:
                    shutil.rmtree(dst)
            else:
                if len(self.nsubs) > 0:
                    self.supplement(dst, actual_bins)
                print(
                    f"G_imp: using existing ({dst}, bins={actual_bins})"
                )
                return

        # --- Tier 2: copy from bundled data ---
        from cphmd import PACKAGE_DIR

        bundled = PACKAGE_DIR / "data" / "G_imp"
        if bundled.exists() and any(bundled.glob("G1_*.dat")):
            bundled_bins = self.detect_bins(bundled)
            if expected_bins is not None and bundled_bins != expected_bins:
                print(
                    f"G_imp: bundled has bins={bundled_bins}, "
                    f"need {expected_bins}. Computing fresh..."
                )
            else:
                shutil.copytree(bundled, dst)
                if len(self.nsubs) > 0:
                    self.supplement(dst, bundled_bins)
                print(
                    f"G_imp: copied bundled data ({dst}, "
                    f"bins={bundled_bins})"
                )
                return

        # --- Tier 3: compute via Monte Carlo ---
        from cphmd.core.entropy import ensure_g_imp_available

        bins = expected_bins if expected_bins is not None else 32
        if len(self.nsubs) == 0:
            raise ValueError(
                "Cannot compute G_imp: nsubs not yet initialized. "
                "Run init_vars() first."
            )
        print(
            f"G_imp: computing via Monte Carlo "
            f"(bins={bins}, fnex={self.fnex}, "
            f"nsubs={list(self.nsubs)})..."
        )
        g_imp_dir = ensure_g_imp_available(
            constraint_type="fnex",
            nsubs=self.nsubs,
            bins=bins,
            fnex=self.fnex,
            cutlsum=self.cutlsum,
        )
        try:
            dst.symlink_to(g_imp_dir)
            print(f"G_imp: cached at {g_imp_dir}, linked to {dst}")
        except OSError:
            shutil.copytree(g_imp_dir, dst, dirs_exist_ok=True)
            print(f"G_imp: cached at {g_imp_dir}, copied to {dst}")

    def regenerate_if_needed(self, old_phase: int, new_phase: int,
                             resolve_bins, alf_info: dict) -> None:
        """Regenerate G_imp if the new phase requires different bins.

        Args:
            old_phase: Previous simulation phase
            new_phase: New simulation phase
            resolve_bins: Callable(g_imp_bins_config, phase) → int|None
            alf_info: ALF info dict (updated with new bins value)
        """
        old_bins = resolve_bins(old_phase)
        new_bins = resolve_bins(new_phase)
        if old_bins == new_bins:
            return
        print(
            f"G_imp: bins {old_bins} → {new_bins} for phase {new_phase}, "
            f"regenerating..."
        )
        dst = self.input_folder / "G_imp"
        if dst.exists():
            if dst.is_symlink():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        # Update bins for the new phase
        self.g_imp_bins = new_bins
        self.ensure_available()
        alf_info["g_imp_bins"] = new_bins or 32

    def supplement(self, g_imp_dir: Path, bins: int | None) -> None:
        """Ensure G1, G12 and cross-site files exist, computing if missing.

        Args:
            g_imp_dir: Path to G_imp directory
            bins: Bin count for computation (None → 32)
        """
        from cphmd.core.entropy import compute_g1_cross, compute_g12, compute_g_imp

        actual_bins = bins if bins is not None else 32
        unique_ndims = sorted(set(self.nsubs)) if len(self.nsubs) > 0 else []

        for ndim in unique_ndims:
            g1_path = g_imp_dir / f"G1_{ndim}.dat"
            g2_path = g_imp_dir / f"G2_{ndim}.dat"
            if not g1_path.exists() or not g2_path.exists():
                print(f"G_imp: computing missing G1_{ndim}/G2_{ndim}.dat...")
                G1, G2 = compute_g_imp("fnex", ndim, bins=actual_bins)
                if not g1_path.exists():
                    np.savetxt(g1_path, G1)
                if not g2_path.exists():
                    np.savetxt(g2_path, G2)

            g12_path = g_imp_dir / f"G12_{ndim}.dat"
            if not g12_path.exists():
                print(f"G_imp: computing missing G12_{ndim}.dat...")
                G12 = compute_g12("fnex", ndim, bins=actual_bins, cutlsum=self.cutlsum)
                np.savetxt(g12_path, G12)

        for ndim_i in unique_ndims:
            for ndim_j in unique_ndims:
                cross_path = g_imp_dir / f"G1_{ndim_i}_{ndim_j}.dat"
                if not cross_path.exists():
                    print(f"G_imp: computing missing G1_{ndim_i}_{ndim_j}.dat...")
                    G_cross = compute_g1_cross(
                        "fnex", ndim_i, ndim_j, bins=actual_bins,
                    )
                    np.savetxt(cross_path, G_cross)

    @staticmethod
    def detect_bins(g_imp_dir: Path) -> int | None:
        """Detect bin count from existing G_imp files.

        Reads the first G2_*.dat file and counts lines (= 2D bins).
        Returns None if no G2 files found.
        """
        g2_files = sorted(g_imp_dir.glob("G2_[0-9]*.dat"))
        if not g2_files:
            return None
        with open(g2_files[0]) as f:
            return sum(1 for line in f if line.strip())
