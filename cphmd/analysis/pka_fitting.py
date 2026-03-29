"""Sigmoid models, guess builders, and pKa correction for Henderson-Hasselbalch analysis.

Pure computation module -- takes numpy arrays, returns dataclass results. No file I/O.
Bootstrap fitting functions will be added in a later task.
"""

from __future__ import annotations

import math
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
