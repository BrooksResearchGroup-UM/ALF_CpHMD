"""Sigmoid models, guess builders, pKa correction, and bootstrap fitting.

Pure computation module -- takes numpy arrays, returns dataclass results. No file I/O.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .ldin_parser import SiteInfo


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """2-state pKa fit result."""

    pka: float
    pka_err: float
    slope: float  # Hill coefficient
    slope_err: float
    amplitude: float
    pka_corrected: float  # Adjusted for A != 1
    bootstrap_params: np.ndarray | None = None  # (n_samples, 3)


@dataclass
class MultiStateFitResult:
    """3-state multi-sigmoid fit result."""

    pka_macro: float
    pka_macro_err: float
    hill: float
    hill_err: float
    f_taut: float
    f_taut_err: float
    pka_micro: list[float] = field(default_factory=list)  # [pka1, pka2]
    pka_micro_err: list[float] = field(default_factory=list)
    bootstrap_params: np.ndarray | None = None  # (n_samples, 3)


# ---------------------------------------------------------------------------
# Sigmoid models
# ---------------------------------------------------------------------------


def sigmoid(pH: np.ndarray, A: float, pka: float, slope: float) -> np.ndarray:
    """Standard Henderson-Hasselbalch sigmoid.

    Parameters
    ----------
    pH : array
        pH values.
    A : float
        Amplitude (1.0 for pure 2-state, <1 for tautomer substates).
    pka : float
        Apparent pKa.
    slope : float
        Hill coefficient with sign.  Negative for acids (population decreases
        with pH), positive for bases.

    Returns
    -------
    array
        Predicted population at each pH.
    """
    return A / (1.0 + 10.0 ** ((pH - pka) * slope))


def make_multi_sigmoid(main_slope_sign: int):
    """Factory for 3-state sigmoid model.

    Parameters
    ----------
    main_slope_sign : int
        -1 for UNEG systems (ASP/GLU), +1 for UPOS systems (HIS/ARG).

    Returns
    -------
    callable
        ``func(pH_stacked, pka, hill, f_taut)`` that predicts all 3 states
        simultaneously.  *pH_stacked* is the pH array repeated 3 times
        (``np.tile(pH, 3)``); the return is the concatenation of predictions
        for state 0, state 1, and state 2.

    Notes
    -----
    State 0 (NONE tag) has amplitude 1 and slope ``main_slope_sign``.
    States 1 and 2 are the tautomers with opposite slope and amplitudes
    ``f_taut`` and ``1 - f_taut`` respectively.
    """

    def multi_sigmoid(
        x_stacked: np.ndarray,
        pka: float,
        hill: float,
        f_taut: float,
    ) -> np.ndarray:
        n = len(x_stacked) // 3
        pH = x_stacked[:n]
        s = main_slope_sign
        # State 0: NONE state (full amplitude = 1)
        state0 = 1.0 / (1.0 + 10.0 ** ((pH - pka) * s * abs(hill)))
        # States 1 and 2: tautomers (opposite slope direction)
        state1 = f_taut / (1.0 + 10.0 ** ((pH - pka) * (-s) * abs(hill)))
        state2 = (1.0 - f_taut) / (1.0 + 10.0 ** ((pH - pka) * (-s) * abs(hill)))
        return np.concatenate([state0, state1, state2])

    return multi_sigmoid


# ---------------------------------------------------------------------------
# Guess builders
# ---------------------------------------------------------------------------


def build_2state_guess(
    site_info: SiteInfo | None = None,
    state_idx: int = 0,
    exp_pka: float | None = None,
    fix_hill: bool = False,
    pka_override: float | None = None,
    pka_err: float | None = None,
) -> tuple[list, list]:
    """Build ``(initial_guess, bounds)`` for 2-state sigmoid ``[A, pka, slope]``.

    Parameters
    ----------
    site_info : SiteInfo or None
        Parsed site info from block.str.  When *None*, generic defaults are
        used (A=1, pka=7, slope=-1).
    state_idx : int
        Index of the state within the site (0 = NONE / main state).
    exp_pka : float or None
        Experimental pKa -- overrides model pKa if provided.
    fix_hill : bool
        If *True*, slope bounds are tightened to approximately +/-1.
    pka_override : float or None
        Fitted pKa from a previous state (overrides model pKa).
    pka_err : float or None
        Error on *pka_override* (constrains pKa bounds when A != 1).

    Returns
    -------
    guess : list
        ``[A, pka, slope]``
    bounds : list
        ``[lower_bounds, upper_bounds]`` each of length 3.
    """
    if site_info is None:
        # Generic defaults
        guess = [1.0, 7.0, -1.0]
        bounds = [[0.0001, -2.0, -5.0], [1.0, 16.0, 5.0]]
        if fix_hill:
            bounds[0][2] = -1.0
            bounds[1][2] = -0.999999
        return guess, bounds

    s = site_info.main_slope_sign
    model_pka = site_info.pka_macro if site_info.pka_macro is not None else 7.0

    states = site_info.states
    if state_idx < len(states):
        state = states[state_idx]
        is_none = state.tag == "NONE"
        if is_none:
            A = 1.0
            slope = float(s)
        else:
            if site_info.n_states <= 2:
                A = 1.0
            else:
                A = (
                    round(site_info.f_taut, 2)
                    if state_idx == 1
                    else round(1.0 - site_info.f_taut, 2)
                )
            slope = float(-s)
    else:
        A = 1.0
        slope = float(s)

    guess = [A, model_pka, slope]

    if exp_pka is not None:
        guess[1] = exp_pka
    if pka_override is not None:
        guess[1] = pka_override

    # Bounds: [A_lo, pKa_lo, slope_lo], [A_hi, pKa_hi, slope_hi]
    bounds: list[list[float]] = [[0.0001, -2.0, -5.0], [1.0, 16.0, 5.0]]

    # Slope bounds depend on sign direction
    if slope > 0:
        bounds[0][2] = 0.0
        bounds[1][2] = 5.0
        if fix_hill:
            bounds[0][2] = 0.999999
            bounds[1][2] = 1.0
    else:
        bounds[0][2] = -5.0
        bounds[1][2] = 0.0
        if fix_hill:
            bounds[0][2] = -1.0
            bounds[1][2] = -0.999999

    # If A == 1, lock amplitude
    if A == 1.0:
        bounds[0][0] = 0.99999
        bounds[1][0] = 1.0
    else:
        # Constrain pKa range for sub-amplitude states
        min_pka_tolerance = 0.5
        effective_err = max(pka_err if pka_err else 0.0, min_pka_tolerance)
        bounds[0][1] = guess[1] - effective_err
        bounds[1][1] = guess[1] + effective_err

    return guess, bounds


def build_multistate_guess(
    site_info: SiteInfo,
    fix_hill: bool = False,
) -> tuple[list, list]:
    """Build ``(initial_guess, bounds)`` for 3-state sigmoid ``[pka, hill, f_taut]``.

    Parameters
    ----------
    site_info : SiteInfo
        Parsed site info.  Must have ``pka_macro`` and ``f_taut``.
    fix_hill : bool
        If *True*, hill coefficient bounds are tightened to ~1.0.

    Returns
    -------
    guess : list
        ``[pka, hill, f_taut]``
    bounds : list
        ``[lower_bounds, upper_bounds]`` each of length 3.
    """
    pka_guess = site_info.pka_macro if site_info.pka_macro is not None else 7.0
    f_taut_guess = float(np.clip(site_info.f_taut, 0.05, 0.95))

    guess = [pka_guess, 1.0, f_taut_guess]

    if fix_hill:
        bounds = [[-2.0, 0.999999, 0.01], [16.0, 1.0, 0.99]]
    else:
        bounds = [[-2.0, 0.01, 0.01], [16.0, 5.0, 0.99]]

    return guess, bounds


# ---------------------------------------------------------------------------
# Quick prefit
# ---------------------------------------------------------------------------


def quick_prefit(
    pH_values: np.ndarray,
    populations: np.ndarray,
    guess: list,
    bounds: list,
) -> tuple[float, np.ndarray]:
    """Single ``curve_fit`` to get an approximate pKa for transition region identification.

    Parameters
    ----------
    pH_values : array
        pH values.
    populations : array
        Mean populations at each pH.
    guess : list
        Initial parameter guess ``[A, pka, slope]``.
    bounds : list
        ``[lower_bounds, upper_bounds]`` for curve_fit.

    Returns
    -------
    pka_approx : float
        Fitted pKa value.
    fitted_params : ndarray
        All fitted parameters ``[A, pka, slope]``.
        Use ``fitted_params[2]`` for Hill coefficient (pass to
        ``identify_transition_region(prefit_slope=...)`` for spread safeguard).
    """
    from scipy.optimize import curve_fit

    popt, _ = curve_fit(
        sigmoid,
        pH_values,
        populations,
        p0=guess,
        bounds=bounds,
        maxfev=10000,
    )
    return popt[1], popt


# ---------------------------------------------------------------------------
# pKa correction
# ---------------------------------------------------------------------------


def correct_pka(pka: float, amplitude: float, slope_sign: int) -> float:
    """Correct apparent pKa when amplitude A != 1.

    For sub-amplitude states (tautomers in a multi-state system), the fitted
    apparent pKa differs from the true micro-pKa by ``log10(A)``.

    Parameters
    ----------
    pka : float
        Fitted apparent pKa.
    amplitude : float
        Fitted amplitude A.
    slope_sign : int
        Sign of the slope (+1 or -1).

    Returns
    -------
    float
        Corrected pKa.  Returns *pka* unchanged if amplitude is out of
        range ``(0, 2)`` or exactly 1.0.
    """
    if amplitude <= 0.0 or amplitude >= 2.0:
        return pka
    if amplitude == 1.0:
        return pka
    return math.log10(amplitude) * slope_sign + pka


# ---------------------------------------------------------------------------
# Transition region identification
# ---------------------------------------------------------------------------


def identify_transition_region(
    pH_values: np.ndarray,
    all_state_pops: list[dict[str, float]],
    pka_approx: float,
    transition_width: float = 2.0,
    pop_floor: float = 0.05,
    pop_ceil: float = 0.95,
    prefit_slope: float | None = None,
) -> np.ndarray:
    """Return boolean mask of pH values in the fluctuating (transition) region.

    A pH is in the transition region if:
    - ``|pH - pka_approx| < transition_width``, OR
    - ANY state's mean population at that pH is between *pop_floor* and *pop_ceil*.

    Plateau points (outside mask) are locked at their mean values during
    bootstrap resampling.  If NO pH passes the filter, all entries are set
    to *True* as a fallback so that the full dataset is resampled.

    **Spread safeguard:** If ``prefit_slope`` is provided and
    ``|prefit_slope| < 0.3`` (very shallow transition — Hill coefficient near
    zero), the entire pH range is marked as transition (all-True) because
    importance sampling offers no benefit for spread sigmoids and a bad
    pKa_approx could exclude informative points.

    Parameters
    ----------
    pH_values : array
        pH values (sorted, length *N*).
    all_state_pops : list[dict[str, float]]
        Per-state dictionaries mapping ``str(pH)`` to mean population.
    pka_approx : float
        Approximate pKa from a prefit.
    transition_width : float
        Half-width of the pKa-centred window.
    pop_floor, pop_ceil : float
        Population thresholds for identifying plateau vs. transition.
    prefit_slope : float or None
        Hill coefficient from prefit. If ``|prefit_slope| < 0.3``, returns
        all-True (spread safeguard).

    Returns
    -------
    np.ndarray
        Boolean mask of length *N*.
    """
    n = len(pH_values)

    # Spread safeguard: shallow Hill coefficient → bootstrap everything
    if prefit_slope is not None and abs(prefit_slope) < 0.3:
        return np.ones(n, dtype=bool)

    mask = np.zeros(n, dtype=bool)
    for i, pH in enumerate(pH_values):
        # Distance criterion
        if abs(pH - pka_approx) < transition_width:
            mask[i] = True
            continue
        # Population criterion: any state between floor and ceil
        for state_pops in all_state_pops:
            pop = state_pops.get(str(pH), state_pops.get(f"{pH:.1f}"))
            if pop is not None and pop_floor < pop < pop_ceil:
                mask[i] = True
                break
    # Fallback: if nothing passes, return all-True
    if not mask.any():
        mask[:] = True
    return mask


# ---------------------------------------------------------------------------
# Bootstrap workers (module-level for pickling with ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _bootstrap_sample_2state(args: tuple, max_retries: int = 2) -> np.ndarray | None:
    """Resample 2-state data and fit sigmoid.  Returns params or None."""
    import random as _rng

    from scipy.optimize import curve_fit

    pH_values, pop_per_pH, errors, guess, bounds, transition_mask = args
    pH_keys = [str(p) for p in pH_values]

    attempts = 0
    while attempts < max_retries:
        y_resampled = []
        sigma_resampled = []

        for i, key in enumerate(pH_keys):
            sublist = pop_per_pH[key]
            if transition_mask is not None and not transition_mask[i]:
                # Plateau: use mean (no resampling)
                y_resampled.append(sum(sublist) / len(sublist))
                if errors is not None:
                    err_list = errors[key]
                    sigma_resampled.append(
                        float(np.mean(err_list)) if isinstance(err_list, (list, np.ndarray)) else float(err_list)
                    )
            else:
                max_val = max(sublist)
                min_val = min(sublist)
                if max_val - min_val >= 0.01:
                    idx = _rng.randrange(len(sublist))
                    y_resampled.append(sublist[idx])
                    if errors is not None:
                        err_list = errors[key]
                        sigma_resampled.append(
                            float(np.mean(err_list)) if isinstance(err_list, (list, np.ndarray)) else float(err_list)
                        )
                else:
                    y_resampled.append(sum(sublist) / len(sublist))
                    if errors is not None:
                        err_list = errors[key]
                        sigma_resampled.append(
                            float(np.mean(err_list)) if isinstance(err_list, (list, np.ndarray)) else float(err_list)
                        )

        y_arr = np.array(y_resampled)
        sigma_arr = np.array(sigma_resampled) if sigma_resampled else None

        if sigma_arr is not None:
            sigma_arr = np.where(sigma_arr == 0, 1e-6, sigma_arr)

        try:
            kw: dict = {"bounds": bounds, "maxfev": 10000}
            if guess is not None:
                kw["p0"] = guess
            if sigma_arr is not None:
                kw["sigma"] = sigma_arr
            popt, _ = curve_fit(sigmoid, pH_values, y_arr, **kw)
            return popt
        except (RuntimeError, ValueError):
            attempts += 1

    return None


def _bootstrap_sample_multistate(args: tuple, max_retries: int = 2) -> np.ndarray | None:
    """Resample multi-state data and fit.  Returns params or None."""
    import random as _rng

    from scipy.optimize import curve_fit

    (
        pH_values,
        all_state_pops,
        all_state_errors,
        func,
        guess,
        bounds,
        n_states,
        transition_mask,
    ) = args

    n_pH = len(pH_values)
    pH_keys = [str(p) for p in pH_values]
    attempts = 0

    while attempts < max_retries:
        y_stacked = []
        sigma_stacked = []

        for state_idx in range(n_states):
            pop_dict = all_state_pops[state_idx]
            err_dict = all_state_errors[state_idx] if all_state_errors is not None else None

            for i, key in enumerate(pH_keys):
                sublist = pop_dict[key]
                if transition_mask is not None and not transition_mask[i]:
                    y_stacked.append(sum(sublist) / len(sublist))
                    if err_dict is not None:
                        err_list = err_dict[key]
                        sigma_stacked.append(
                            float(np.mean(err_list))
                            if isinstance(err_list, (list, np.ndarray))
                            else float(err_list)
                        )
                else:
                    max_val = max(sublist)
                    min_val = min(sublist)
                    if max_val - min_val >= 0.005:
                        idx = _rng.randrange(len(sublist))
                        y_stacked.append(sublist[idx])
                        if err_dict is not None:
                            err_list = err_dict[key]
                            sigma_stacked.append(
                                float(np.mean(err_list))
                                if isinstance(err_list, (list, np.ndarray))
                                else float(err_list)
                            )
                    else:
                        y_stacked.append(sum(sublist) / len(sublist))
                        if err_dict is not None:
                            err_list = err_dict[key]
                            sigma_stacked.append(
                                float(np.mean(err_list))
                                if isinstance(err_list, (list, np.ndarray))
                                else float(err_list)
                            )

        y_arr = np.array(y_stacked)
        x_arr = np.tile(pH_values, n_states)
        sigma_arr = np.array(sigma_stacked) if sigma_stacked else None

        if sigma_arr is not None:
            sigma_arr = np.where(sigma_arr == 0, 1e-6, sigma_arr)

        try:
            kw: dict = {"bounds": bounds, "maxfev": 10000}
            if guess is not None:
                kw["p0"] = guess
            if sigma_arr is not None:
                kw["sigma"] = sigma_arr
            popt, _ = curve_fit(func, x_arr, y_arr, **kw)
            return popt
        except (RuntimeError, ValueError):
            attempts += 1

    return None


# ---------------------------------------------------------------------------
# Bootstrap orchestrators
# ---------------------------------------------------------------------------


def bootstrap_fit_2state(
    pH_values: np.ndarray,
    populations: dict[str, list[float]],
    errors: dict[str, list[float]] | None,
    guess: list,
    bounds: list,
    n_samples: int = 1000,
    n_jobs: int = 1,
    transition_mask: np.ndarray | None = None,
) -> FitResult:
    """Bootstrap 2-state pKa fit with importance sampling.

    Parameters
    ----------
    pH_values : array
        Sorted pH values.
    populations : dict
        ``{str(pH): [pop_rep1, pop_rep2, ...]}`` per-replicate populations.
    errors : dict or None
        Per-pH error values (same structure as *populations*), or *None*.
    guess : list
        Initial ``[A, pka, slope]``.
    bounds : list
        ``[lower, upper]`` for ``curve_fit``.
    n_samples : int
        Number of bootstrap resamples.
    n_jobs : int
        Workers for ``ProcessPoolExecutor``.  1 = sequential.
    transition_mask : array or None
        Boolean mask from :func:`identify_transition_region`.

    Returns
    -------
    FitResult
        With ``pka``, ``pka_err`` (95 % CI half-width), and ``bootstrap_params``.
    """
    from scipy.optimize import curve_fit as _cf

    # Single-replicate fallback: deterministic fit, no bootstrap
    n_reps = [len(v) for v in populations.values()]
    if max(n_reps) <= 1:
        mean_pops = np.array([populations[str(p)][0] for p in pH_values])
        try:
            popt, _ = _cf(sigmoid, pH_values, mean_pops, p0=guess, bounds=bounds, maxfev=10000)
            return FitResult(
                pka=popt[1],
                pka_err=0.0,
                slope=popt[2],
                slope_err=0.0,
                amplitude=popt[0],
                pka_corrected=correct_pka(popt[1], popt[0], int(np.sign(popt[2]))),
                bootstrap_params=None,
            )
        except (RuntimeError, ValueError):
            return FitResult(
                pka=float("nan"),
                pka_err=float("nan"),
                slope=float("nan"),
                slope_err=float("nan"),
                amplitude=float("nan"),
                pka_corrected=float("nan"),
                bootstrap_params=None,
            )

    # Build args for workers
    args_list = [
        (pH_values, populations, errors, guess, bounds, transition_mask)
        for _ in range(n_samples)
    ]

    # Execute bootstrap
    if n_jobs == 1:
        results = [_bootstrap_sample_2state(a) for a in args_list]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_bootstrap_sample_2state, args_list))

    # Filter failed fits
    valid = [r for r in results if r is not None]
    if len(valid) < n_samples * 0.5:
        return FitResult(
            pka=float("nan"),
            pka_err=float("nan"),
            slope=float("nan"),
            slope_err=float("nan"),
            amplitude=float("nan"),
            pka_corrected=float("nan"),
            bootstrap_params=None,
        )

    param_samples = np.array(valid)  # (n_valid, 3)
    pka_vals = param_samples[:, 1]
    slope_vals = param_samples[:, 2]
    amp_vals = param_samples[:, 0]

    pka_median = float(np.median(pka_vals))
    pka_err = float((np.percentile(pka_vals, 97.5) - np.percentile(pka_vals, 2.5)) / 2.0)
    slope_median = float(np.median(slope_vals))
    slope_err = float((np.percentile(slope_vals, 97.5) - np.percentile(slope_vals, 2.5)) / 2.0)
    amp_median = float(np.median(amp_vals))

    return FitResult(
        pka=pka_median,
        pka_err=pka_err,
        slope=slope_median,
        slope_err=slope_err,
        amplitude=amp_median,
        pka_corrected=correct_pka(pka_median, amp_median, int(np.sign(slope_median))),
        bootstrap_params=param_samples,
    )


def _derive_micro_pkas(
    pka_macro: float, f_taut: float, main_slope_sign: int = -1
) -> list[float]:
    """Derive micro-pKa values from macro pKa and tautomer fraction.

    For a 3-state system with tautomer fraction *f_taut*:
    - pKa_micro1 = pKa_macro - log10(f_taut)      (for the f_taut state)
    - pKa_micro2 = pKa_macro - log10(1 - f_taut)   (for the 1 - f_taut state)
    """
    f_taut_safe = np.clip(f_taut, 1e-6, 1.0 - 1e-6)
    pka1 = pka_macro - math.log10(f_taut_safe)
    pka2 = pka_macro - math.log10(1.0 - f_taut_safe)
    return [pka1, pka2]


def bootstrap_fit_multistate(
    pH_values: np.ndarray,
    all_state_pops: list[dict[str, list[float]]],
    all_state_errors: list[dict[str, list[float]]] | None,
    func: callable,
    guess: list,
    bounds: list,
    n_samples: int = 1000,
    n_jobs: int = 1,
    transition_mask: np.ndarray | None = None,
) -> MultiStateFitResult:
    """Bootstrap 3-state pKa fit with importance sampling.

    Parameters
    ----------
    pH_values : array
        Sorted pH values.
    all_state_pops : list[dict]
        Per-state ``{str(pH): [pop_rep1, ...]}`` dictionaries.
    all_state_errors : list[dict] or None
        Per-state error dictionaries.
    func : callable
        Multi-sigmoid function from :func:`make_multi_sigmoid`.
    guess : list
        ``[pka, hill, f_taut]`` initial guess.
    bounds : list
        ``[lower, upper]`` for ``curve_fit``.
    n_samples : int
        Number of bootstrap resamples.
    n_jobs : int
        Workers for ``ProcessPoolExecutor``.  1 = sequential.
    transition_mask : array or None
        Boolean mask from :func:`identify_transition_region`.

    Returns
    -------
    MultiStateFitResult
    """
    n_states = len(all_state_pops)

    args_list = [
        (
            pH_values,
            all_state_pops,
            all_state_errors,
            func,
            guess,
            bounds,
            n_states,
            transition_mask,
        )
        for _ in range(n_samples)
    ]

    if n_jobs == 1:
        results = [_bootstrap_sample_multistate(a) for a in args_list]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_bootstrap_sample_multistate, args_list))

    valid = [r for r in results if r is not None]
    if len(valid) < n_samples * 0.5:
        return MultiStateFitResult(
            pka_macro=float("nan"),
            pka_macro_err=float("nan"),
            hill=float("nan"),
            hill_err=float("nan"),
            f_taut=float("nan"),
            f_taut_err=float("nan"),
            bootstrap_params=None,
        )

    param_samples = np.array(valid)  # (n_valid, 3): [pka, hill, f_taut]
    pka_vals = param_samples[:, 0]
    hill_vals = param_samples[:, 1]
    ftaut_vals = param_samples[:, 2]

    pka_median = float(np.median(pka_vals))
    pka_err = float((np.percentile(pka_vals, 97.5) - np.percentile(pka_vals, 2.5)) / 2.0)
    hill_median = float(np.median(hill_vals))
    hill_err = float((np.percentile(hill_vals, 97.5) - np.percentile(hill_vals, 2.5)) / 2.0)
    ftaut_median = float(np.median(ftaut_vals))
    ftaut_err = float(
        (np.percentile(ftaut_vals, 97.5) - np.percentile(ftaut_vals, 2.5)) / 2.0
    )

    # Derive micro-pKas for each bootstrap sample
    micro_pka_samples = np.array(
        [_derive_micro_pkas(p[0], p[2]) for p in param_samples]
    )  # (n_valid, 2)
    micro_median = [float(np.median(micro_pka_samples[:, j])) for j in range(2)]
    micro_err = [
        float(
            (np.percentile(micro_pka_samples[:, j], 97.5) - np.percentile(micro_pka_samples[:, j], 2.5)) / 2.0
        )
        for j in range(2)
    ]

    return MultiStateFitResult(
        pka_macro=pka_median,
        pka_macro_err=pka_err,
        hill=hill_median,
        hill_err=hill_err,
        f_taut=ftaut_median,
        f_taut_err=ftaut_err,
        pka_micro=micro_median,
        pka_micro_err=micro_err,
        bootstrap_params=param_samples,
    )
