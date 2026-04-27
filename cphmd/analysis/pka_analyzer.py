"""pKa analysis orchestrator for production CpHMD data.

Discovers production parquet files, builds a site map from patches.dat,
loads and processes lambda data, performs bootstrap pKa fitting, generates
static and convergence plots, and writes summary files.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .pka_fitting import FitResult, MultiStateFitResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PKaAnalysisConfig:
    """Configuration for pKa analysis."""

    input_folder: str | Path | Sequence[str | Path]
    output_dir: str | Path = "analysis"
    analysis_name: str | None = None
    lambda_cutoff: float = 0.97
    bin_size_ps: float = 1000.0
    fix_hill: bool = False
    skip_ps: float = 0.0
    n_bootstrap: int = 1000
    n_bootstrap_bin: int = 500
    n_jobs: int = 1
    transition_width: float = 2.0
    exp_pka: dict[str, float] | None = None
    min_ph_points: int = 3
    input_folders: tuple[Path, ...] = field(init=False)

    def __post_init__(self):
        if isinstance(self.input_folder, (str, Path)):
            input_folders = (Path(self.input_folder),)
        else:
            input_folders = tuple(Path(folder) for folder in self.input_folder)
        if not input_folders:
            raise ValueError("input_folder must contain at least one production directory")
        self.input_folders = input_folders
        self.input_folder = input_folders[0]
        self.output_dir = Path(self.output_dir)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SitePKaResult:
    """Results for one titratable site."""

    resid: str
    segid: str
    resname: str
    site_index: int
    n_states: int
    fit_result: "FitResult | MultiStateFitResult | None"
    pka_plot: Path | None = None
    bin_plot: Path | None = None
    acc_plot: Path | None = None


@dataclass
class PKaResults:
    """Complete pKa analysis results."""

    sites: list[SitePKaResult] = field(default_factory=list)
    pka_summary_file: Path | None = None
    exp_comparison_file: Path | None = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class PKaAnalyzer:
    """Orchestrates pKa analysis from production parquets."""

    def __init__(self, config: PKaAnalysisConfig):
        self.config = config

    def run(self) -> PKaResults:
        """Execute full pKa analysis pipeline.

        1. Setup output directories.
        2. Discover parquet files and extract metadata.
        3. Build site map from patches.dat.
        4. Parse LDIN block.str for multi-state fitting parameters.
        5. For each titratable site:
           a. Load lambda data, apply cutoff, skip equilibration.
           b. Compute total populations and per-bin populations.
           c. Perform bootstrap pKa fitting (2-state or multi-state).
           d. Generate static pKa plot and convergence plots.
        6. Write summary files and return results.
        """
        from .ldin_parser import parse_block_str
        from .pka_data import (
            build_site_map,
            discover_parquets,
            get_site_columns,
            resolve_site_columns,
        )

        cfg = self.config
        results = PKaResults()

        # ---- 1. Setup output dirs ----
        data_dir = cfg.output_dir / "data"
        plot_dir = cfg.output_dir / "plots"
        data_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # ---- 2. Discover parquets ----
        if len(cfg.input_folders) > 1:
            metadata_df = discover_parquets(
                cfg.input_folders,
                n_jobs=cfg.n_jobs,
                simulation_from_folder=True,
            )
        else:
            metadata_df = discover_parquets(cfg.input_folder, n_jobs=cfg.n_jobs)
        if metadata_df.empty:
            raise FileNotFoundError(
                f"No parquet files found in {', '.join(str(path) for path in cfg.input_folders)}"
            )
        logger.info("Discovered %d parquet files", len(metadata_df))

        # ---- 3. Build site map ----
        patches_path = self._find_file("patches.dat")
        if patches_path is None:
            raise FileNotFoundError(
                f"patches.dat not found in {', '.join(str(path) for path in cfg.input_folders)}"
            )
        site_map = build_site_map(patches_path)
        logger.info("Site map: %d entries, %d sites", len(site_map), site_map["site"].nunique())

        # ---- 4. Parse LDIN block.str ----
        block_path = self._find_file("block.str")
        if block_path is not None:
            ldin_data = parse_block_str(block_path)
        else:
            logger.warning("block.str not found; using default fitting parameters")
            ldin_data = {}

        # ---- 5. Extract metadata ----
        # Read timestep from parquet metadata
        if "Time Step" in metadata_df.columns:
            timestep_ps = float(metadata_df["Time Step"].iloc[0])
        else:
            timestep_ps = 1.0

        # Get unique site indices (skip site 0 which is typically the environment)
        site_indices = sorted(site_map["site"].unique())

        folder_name = cfg.analysis_name or (
            cfg.input_folder.name if len(cfg.input_folders) == 1 else "production"
        )
        site_errors: list[str] = []

        # ---- 6. Process each site ----
        for site_idx in site_indices:
            legacy_site_cols = get_site_columns(site_map, site_idx)
            if len(legacy_site_cols) <= 1:
                continue  # Skip single-subsite entries (e.g., environment block)
            site_cols = resolve_site_columns(site_map, site_idx, metadata_df)

            # Extract residue metadata from the site map
            site_rows = site_map[site_map["site"] == site_idx]
            resid_str = str(site_rows["resid"].iloc[0])
            segid_str = str(site_rows["segid"].iloc[0])
            # Get the 3-letter residue name from the patch column
            patch_str = str(site_rows["patch"].iloc[0])
            resn_str = patch_str[:3] if len(patch_str) >= 3 else patch_str
            n_states = len(site_cols)

            site_label = f"{resn_str} {resid_str}"
            logger.info("Processing site %d: %s (%d states)", site_idx, site_label, n_states)
            print(f"{'=' * 30}")
            print(f"Processing site {site_idx} ({site_label}) with {n_states} states")
            sys.stdout.flush()

            try:
                result = self._process_site(
                    site_idx=site_idx,
                    site_cols=site_cols,
                    resid_str=resid_str,
                    segid_str=segid_str,
                    resn_str=resn_str,
                    n_states=n_states,
                    site_label=site_label,
                    metadata_df=metadata_df,
                    ldin_data=ldin_data,
                    timestep_ps=timestep_ps,
                    folder_name=folder_name,
                    data_dir=data_dir,
                    plot_dir=plot_dir,
                )
                if result is not None:
                    results.sites.append(result)
            except Exception as exc:
                logger.error("Error processing site %s: %s", site_label, exc)
                print(f"Error processing site {site_label}: {exc}")
                sys.stdout.flush()
                site_errors.append(f"{site_label}: {exc}")
                continue

        if site_errors:
            raise RuntimeError("site analysis failures: " + "; ".join(site_errors))

        # ---- 7. Write summary files ----
        results.pka_summary_file = self._write_pka_summary(results.sites, data_dir, folder_name)
        if cfg.exp_pka:
            results.exp_comparison_file = self._write_exp_comparison(
                results.sites, data_dir, folder_name
            )

        return results

    # ------------------------------------------------------------------
    # Internal: process a single site
    # ------------------------------------------------------------------

    def _process_site(
        self,
        site_idx: int,
        site_cols: list[str],
        resid_str: str,
        segid_str: str,
        resn_str: str,
        n_states: int,
        site_label: str,
        metadata_df,
        ldin_data: dict,
        timestep_ps: float,
        folder_name: str,
        data_dir: Path,
        plot_dir: Path,
    ) -> SitePKaResult | None:
        """Process a single titratable site end-to-end."""
        from .pka_data import (
            apply_cutoff,
            compute_simulation_populations,
            compute_total_population,
            load_lambda_data,
            skip_equilibration,
        )
        from .pka_plots import plot_pka

        cfg = self.config

        # a. Load lambda data
        columns = site_cols
        all_data = load_lambda_data(metadata_df, columns, n_jobs=cfg.n_jobs)

        # b. Apply cutoff (also rearranges to {pH: {sim: df}})
        bool_data = apply_cutoff(all_data, cutoff=cfg.lambda_cutoff)

        # Load time columns separately for skip/binning
        time_data = load_lambda_data(metadata_df, ["time"], n_jobs=cfg.n_jobs)
        # Rearrange time data to match bool_data nesting: {pH: {sim: df}}
        time_rearranged: dict[str, dict[str, object]] = {}
        for sim, ph_dict in time_data.items():
            for ph, df in ph_dict.items():
                time_rearranged.setdefault(ph, {})[sim] = df

        # c. Skip equilibration
        if cfg.skip_ps > 0:
            bool_data, time_rearranged = skip_equilibration(bool_data, time_rearranged, cfg.skip_ps)

        # d. Check pH coverage
        ph_values = sorted(bool_data.keys(), key=float)
        if len(ph_values) < cfg.min_ph_points:
            logger.warning(
                "Site %s: only %d pH points (need %d), skipping",
                site_label,
                len(ph_values),
                cfg.min_ph_points,
            )
            return None

        # e. Compute total population (pooled across simulations)
        total_pops, total_errs = compute_total_population(bool_data, n_jobs=cfg.n_jobs)
        sim_pops, sim_errs = compute_simulation_populations(bool_data)

        # f. Get LDIN site info
        site_info = self._find_ldin_site(ldin_data, segid_str, resid_str)

        # g. Get state patch names for labels
        state_names = self._get_state_names(site_cols, ldin_data, segid_str, resid_str)

        if n_states >= 3 and site_info is None:
            raise RuntimeError(
                f"multi-state site requires LDIN metadata: segid={segid_str} resid={resid_str}"
            )

        ph_arr = np.array(sorted(map(float, total_pops.keys())))

        # i. Fit based on n_states
        if n_states == 2:
            fit_result = self._fit_2state(
                ph_arr=ph_arr,
                total_pops=total_pops,
                total_errs=total_errs,
                total_pops_wrapped=sim_pops,
                total_errs_wrapped=sim_errs,
                site_cols=site_cols,
                site_info=site_info,
                resid_str=resid_str,
                segid_str=segid_str,
            )
        elif n_states >= 3 and site_info is not None:
            fit_result = self._fit_multistate(
                ph_arr=ph_arr,
                total_pops=total_pops,
                total_errs=total_errs,
                total_pops_wrapped=sim_pops,
                total_errs_wrapped=sim_errs,
                site_cols=site_cols,
                site_info=site_info,
            )
        # j. Static pKa plot
        pka_plot_path = plot_dir / f"{folder_name}_{resn_str}_{resid_str}_pka.png"
        populations_for_plot, errors_for_plot = self._build_plot_arrays(
            ph_arr, total_pops, total_errs, site_cols
        )
        exp_pka_val = self._exp_pka_value(segid_str, resid_str)
        try:
            plot_pka(
                site_label=site_label,
                pH_values=ph_arr,
                populations=populations_for_plot,
                errors=errors_for_plot,
                fit_result=fit_result,
                state_names=state_names,
                site_info=site_info,
                exp_pka=exp_pka_val,
                output_path=pka_plot_path,
            )
        except Exception as exc:
            logger.warning("Failed to generate pKa plot for %s: %s", site_label, exc)
            pka_plot_path = None

        # k. Convergence plots (instantaneous and accumulated)
        bin_plot_path = plot_dir / f"{folder_name}_{resn_str}_{resid_str}_bin.png"
        acc_plot_path = plot_dir / f"{folder_name}_{resn_str}_{resid_str}_acc.png"
        bin_data_path = data_dir / f"{folder_name}_{resn_str}_{resid_str}_bin.dat"
        acc_data_path = data_dir / f"{folder_name}_{resn_str}_{resid_str}_acc.dat"

        bin_plot_path = self._convergence_plot(
            bool_data=bool_data,
            time_data=time_rearranged,
            site_cols=site_cols,
            site_info=site_info,
            site_label=site_label,
            ph_arr=ph_arr,
            accumulate=False,
            plot_path=bin_plot_path,
            data_path=bin_data_path,
            exp_pka_val=exp_pka_val,
        )

        acc_plot_path = self._convergence_plot(
            bool_data=bool_data,
            time_data=time_rearranged,
            site_cols=site_cols,
            site_info=site_info,
            site_label=site_label,
            ph_arr=ph_arr,
            accumulate=True,
            plot_path=acc_plot_path,
            data_path=acc_data_path,
            exp_pka_val=exp_pka_val,
        )

        return SitePKaResult(
            resid=resid_str,
            segid=segid_str,
            resname=resn_str,
            site_index=site_idx,
            n_states=n_states,
            fit_result=fit_result,
            pka_plot=pka_plot_path,
            bin_plot=bin_plot_path,
            acc_plot=acc_plot_path,
        )

    # ------------------------------------------------------------------
    # 2-state fitting
    # ------------------------------------------------------------------

    def _fit_2state(
        self,
        ph_arr: np.ndarray,
        total_pops: dict,
        total_errs: dict,
        total_pops_wrapped: dict,
        total_errs_wrapped: dict,
        site_cols: list[str],
        site_info,
        resid_str: str,
        segid_str: str,
    ):
        """Perform 2-state bootstrap pKa fit on the first state column."""
        from .pka_data import prepare_fit_data
        from .pka_fitting import (
            bootstrap_fit_2state,
            build_2state_guess,
            identify_transition_region,
            quick_prefit,
        )

        cfg = self.config
        exp_pka_val = self._exp_pka_value(segid_str, resid_str)

        # Prepare fit data for state 0 (NONE / main state)
        ph_fit, pop_per_ph, err_per_ph = prepare_fit_data(
            total_pops_wrapped, total_errs_wrapped, [site_cols[0]]
        )

        # Build guess
        guess, bounds = build_2state_guess(
            site_info=site_info,
            state_idx=0,
            exp_pka=exp_pka_val,
            fix_hill=cfg.fix_hill,
        )

        # Mean populations for prefit
        mean_pops = np.array([np.mean(pop_per_ph[str(p)]) for p in ph_fit])

        # Quick prefit
        try:
            pka_approx, fitted_params = quick_prefit(ph_fit, mean_pops, guess, bounds)
        except Exception as exc:
            site_segid = site_info.segid if site_info is not None else segid_str
            raise RuntimeError(
                f"quick_prefit failed for segid={site_segid} resid={resid_str}"
            ) from exc

        # Identify transition region
        all_state_pops = []
        for col in site_cols:
            state_pops = {}
            for ph_str, series in total_pops.items():
                state_pops[ph_str] = float(series[col]) if col in series.index else 0.0
            all_state_pops.append(state_pops)

        transition_mask = identify_transition_region(
            ph_fit,
            all_state_pops,
            pka_approx,
            transition_width=cfg.transition_width,
        )

        # Bootstrap fit
        fit_result = bootstrap_fit_2state(
            ph_fit,
            pop_per_ph,
            err_per_ph,
            list(fitted_params),
            bounds,
            n_samples=cfg.n_bootstrap,
            n_jobs=cfg.n_jobs,
            transition_mask=transition_mask,
        )

        return fit_result

    # ------------------------------------------------------------------
    # Multi-state fitting
    # ------------------------------------------------------------------

    def _fit_multistate(
        self,
        ph_arr: np.ndarray,
        total_pops: dict,
        total_errs: dict,
        total_pops_wrapped: dict,
        total_errs_wrapped: dict,
        site_cols: list[str],
        site_info,
    ):
        """Perform multi-state bootstrap pKa fit."""
        from .pka_fitting import (
            bootstrap_fit_multistate,
            build_multistate_guess,
            identify_transition_region,
            make_multi_sigmoid,
        )

        cfg = self.config

        # Build per-state population dicts for multi-state fitting
        # Format: list of {pH_str: [pop_rep1, ...]} per state
        all_state_pops: list[dict[str, list[float]]] = []
        all_state_errors: list[dict[str, list[float]]] = []
        mean_state_pops: list[dict[str, float]] = []

        for col in site_cols:
            state_pop_dict: dict[str, list[float]] = {}
            state_err_dict: dict[str, list[float]] = {}
            state_mean_dict: dict[str, float] = {}
            for ph_str, sim_dict in total_pops_wrapped.items():
                vals = [
                    float(series[col]) if col in series.index else 0.0
                    for series in sim_dict.values()
                ]
                errs = [
                    float(series[col]) if col in series.index else 0.0
                    for series in total_errs_wrapped.get(ph_str, {}).values()
                ]
                state_pop_dict[ph_str] = vals
                state_err_dict[ph_str] = errs
                state_mean_dict[ph_str] = float(np.mean(vals)) if vals else 0.0
            all_state_pops.append(state_pop_dict)
            all_state_errors.append(state_err_dict)
            mean_state_pops.append(state_mean_dict)

        ph_fit = np.array(sorted(map(float, total_pops.keys())))

        # Build guess
        guess, bounds = build_multistate_guess(site_info, fix_hill=cfg.fix_hill)

        # Identify transition region using mean populations
        transition_mask = identify_transition_region(
            ph_fit,
            mean_state_pops,
            pka_approx=site_info.pka_macro if site_info.pka_macro is not None else 7.0,
            transition_width=cfg.transition_width,
        )

        # Multi-sigmoid function
        func = make_multi_sigmoid(site_info.main_slope_sign)

        # Bootstrap fit
        fit_result = bootstrap_fit_multistate(
            ph_fit,
            all_state_pops,
            all_state_errors,
            func,
            guess,
            bounds,
            n_samples=cfg.n_bootstrap,
            n_jobs=cfg.n_jobs,
            transition_mask=transition_mask,
        )

        return fit_result

    # ------------------------------------------------------------------
    # Convergence plots
    # ------------------------------------------------------------------

    def _convergence_plot(
        self,
        bool_data: dict,
        time_data: dict,
        site_cols: list[str],
        site_info,
        site_label: str,
        ph_arr: np.ndarray,
        accumulate: bool,
        plot_path: Path,
        data_path: Path,
        exp_pka_val: float | None,
    ) -> Path | None:
        """Compute binned populations and generate a convergence plot."""
        from .pka_data import compute_populations
        from .pka_fitting import (
            bootstrap_fit_2state,
            build_2state_guess,
        )
        from .pka_plots import plot_pka_convergence

        cfg = self.config

        try:
            bin_pops, bin_errs = compute_populations(
                bool_data,
                time_data,
                bin_size_ps=cfg.bin_size_ps,
                accumulate=accumulate,
                n_jobs=cfg.n_jobs,
            )
        except Exception as exc:
            logger.warning("Failed to compute binned populations for %s: %s", site_label, exc)
            return None

        # Determine number of time bins from the first available simulation
        n_bins = 0
        for sim_dict in bin_pops.values():
            for df in sim_dict.values():
                n_bins = max(n_bins, len(df))

        if n_bins == 0:
            return None

        # Build guess for convergence fitting
        guess, bounds = build_2state_guess(site_info=site_info, state_idx=0, fix_hill=cfg.fix_hill)

        # Per-bin pKa fitting
        pka_per_bin = np.full(n_bins, np.nan)
        pka_err_per_bin = np.full(n_bins, np.nan)

        for bin_idx in range(n_bins):
            # Extract populations at this bin for each pH/sim
            bin_pop_dict: dict[str, list[float]] = {}
            bin_err_dict: dict[str, list[float]] = {}

            for ph in sorted(bin_pops.keys(), key=float):
                for sim, df in bin_pops[ph].items():
                    if bin_idx < len(df) and site_cols[0] in df.columns:
                        val = float(df[site_cols[0]].iloc[bin_idx])
                        bin_pop_dict.setdefault(ph, []).append(val)
                    if ph in bin_errs and sim in bin_errs[ph]:
                        err_df = bin_errs[ph][sim]
                        if bin_idx < len(err_df) and site_cols[0] in err_df.columns:
                            bin_err_dict.setdefault(ph, []).append(
                                float(err_df[site_cols[0]].iloc[bin_idx])
                            )

            ph_available = sorted(bin_pop_dict.keys(), key=float)
            if len(ph_available) < cfg.min_ph_points:
                continue

            ph_bin = np.array([float(p) for p in ph_available])
            pop_for_fit = {str(p): bin_pop_dict[str(p)] for p in ph_bin}
            err_for_fit = {str(p): bin_err_dict.get(str(p), []) for p in ph_bin}

            try:
                fit = bootstrap_fit_2state(
                    ph_bin,
                    pop_for_fit,
                    err_for_fit if any(err_for_fit.values()) else None,
                    guess,
                    bounds,
                    n_samples=cfg.n_bootstrap_bin,
                    n_jobs=cfg.n_jobs,
                )
                pka_per_bin[bin_idx] = fit.pka
                pka_err_per_bin[bin_idx] = fit.pka_err
            except Exception:
                pass  # Leave as NaN

        # Build time axis in nanoseconds
        time_bins_ns = np.arange(n_bins) * cfg.bin_size_ps / 1000.0

        try:
            plot_pka_convergence(
                site_label=site_label,
                time_bins_ns=time_bins_ns,
                pka_per_bin=pka_per_bin,
                pka_err_per_bin=pka_err_per_bin,
                exp_pka=exp_pka_val,
                accumulate=accumulate,
                output_path=plot_path,
                data_path=data_path,
            )
            return plot_path
        except Exception as exc:
            logger.warning("Failed to generate convergence plot for %s: %s", site_label, exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_file(self, filename: str) -> Path | None:
        """Search for a file in common locations under input folders."""
        candidates = []
        for input_folder in self.config.input_folders:
            candidates.extend(
                [
                    input_folder / "prep" / filename,
                    input_folder / filename,
                ]
            )
            if not input_folder.exists():
                continue
            for child in sorted(input_folder.iterdir()):
                if child.is_dir() and child.name.startswith("prod_"):
                    candidates.append(child / "prep" / filename)

        for path in candidates:
            if path.exists():
                return path
        return None

    def _find_ldin_site(self, ldin_data: dict, segid_str: str, resid_str: str):
        """Find LDIN metadata without confusing same-resid sites on different chains."""
        direct = ldin_data.get(f"{segid_str}:{resid_str}")
        if direct is not None:
            return direct

        matches = [
            site
            for site in ldin_data.values()
            if getattr(site, "segid", None) == segid_str
            and str(getattr(site, "resid", "")) == resid_str
        ]
        if len(matches) == 1:
            return matches[0]

        legacy = ldin_data.get(resid_str)
        if legacy is not None and getattr(legacy, "segid", segid_str) == segid_str:
            return legacy
        return None

    def _get_state_names(
        self,
        site_cols: list[str],
        ldin_data: dict,
        segid_str: str,
        resid_str: str,
    ) -> list[str]:
        """Get patch names for each state from LDIN data or generate defaults."""
        site_info = self._find_ldin_site(ldin_data, segid_str, resid_str)
        if site_info is not None:
            return [s.resname for s in site_info.states]
        # Fallback: use column names
        return site_cols

    def _build_plot_arrays(
        self,
        ph_arr: np.ndarray,
        total_pops: dict,
        total_errs: dict,
        site_cols: list[str],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Build per-state population arrays for plot_pka.

        Returns (populations, errors) where each is a list of arrays
        with one entry per state. Each array has shape (n_pH,).
        """
        populations = []
        errors = []
        for col in site_cols:
            pop_arr = np.array(
                [
                    (
                        float(total_pops[str(p)][col])
                        if str(p) in total_pops and col in total_pops[str(p)].index
                        else 0.0
                    )
                    for p in ph_arr
                ]
            )
            err_arr = np.array(
                [
                    (
                        float(total_errs[str(p)][col])
                        if str(p) in total_errs and col in total_errs[str(p)].index
                        else 0.0
                    )
                    for p in ph_arr
                ]
            )
            populations.append(pop_arr)
            errors.append(err_arr)
        return populations, errors

    def _exp_pka_value(self, segid: str, resid: str) -> float | None:
        if not self.config.exp_pka:
            return None
        for key in (f"{segid}:{resid}", f"{segid} {resid}", resid):
            value = self.config.exp_pka.get(key)
            if value is not None:
                return value
        return None

    def _write_pka_summary(
        self, sites: list[SitePKaResult], data_dir: Path, folder_name: str
    ) -> Path:
        """Write a tab-separated pKa summary file."""
        from .pka_fitting import FitResult, MultiStateFitResult

        summary_path = data_dir / f"pka_{folder_name}.dat"
        with open(summary_path, "w") as f:
            f.write(
                "{:<8}\t{:<8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\n".format(
                    "resid", "resname", "pka", "slope", "pka_err", "slope_err"
                )
            )
            for site in sites:
                fit = site.fit_result
                if fit is None:
                    continue
                if isinstance(fit, FitResult):
                    f.write(
                        "{:<8}\t{:<8}\t{:>8.2f}\t{:>8.2f}\t{:>8.2f}\t{:>8.2f}\n".format(
                            site.resid,
                            site.resname,
                            fit.pka_corrected,
                            fit.slope,
                            fit.pka_err,
                            fit.slope_err,
                        )
                    )
                elif isinstance(fit, MultiStateFitResult):
                    f.write(
                        "{:<8}\t{:<8}\t{:>8.2f}\t{:>8.2f}\t{:>8.2f}\t{:>8.2f}\n".format(
                            site.resid,
                            site.resname,
                            fit.pka_macro,
                            fit.hill,
                            fit.pka_macro_err,
                            fit.hill_err,
                        )
                    )
        logger.info("Summary written to %s", summary_path)
        return summary_path

    def _write_exp_comparison(
        self, sites: list[SitePKaResult], data_dir: Path, folder_name: str
    ) -> Path | None:
        """Write experimental comparison file if exp_pka is provided."""
        from .pka_fitting import FitResult, MultiStateFitResult

        if not self.config.exp_pka:
            return None

        exp_path = data_dir / f"exp_{folder_name}.dat"
        with open(exp_path, "w") as f:
            f.write(
                "{:<8}\t{:<8}\t{:<8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\n".format(
                    "resid",
                    "resname",
                    "pka",
                    "pka_err",
                    "exp_pKa",
                    "delta_pka",
                    "delta_pka_err",
                )
            )
            for site in sites:
                fit = site.fit_result
                if fit is None:
                    continue
                exp_val = self._exp_pka_value(site.segid, site.resid)
                if exp_val is None:
                    continue

                if isinstance(fit, FitResult):
                    pka_val = fit.pka_corrected
                    pka_err = fit.pka_err
                elif isinstance(fit, MultiStateFitResult):
                    pka_val = fit.pka_macro
                    pka_err = fit.pka_macro_err
                else:
                    continue

                delta = pka_val - exp_val
                f.write(
                    "{:<8}\t{:<8}\t{:<8.2f}\t{:>8.2f}\t{:>8.2f}\t{:>8.2f}\t{:>8.2f}\n".format(
                        site.resid,
                        site.resname,
                        pka_val,
                        pka_err,
                        exp_val,
                        delta,
                        pka_err,
                    )
                )
        logger.info("Experimental comparison written to %s", exp_path)
        return exp_path
