"""
Statistical analysis tools for quantum measurement outcomes.
"""

from __future__ import annotations

import math
from typing import Dict, Union
import numpy as np


def calculate_counts(outcomes: np.ndarray) -> Dict[int, int]:
    """Calculate the observed occurrence counts of outcomes 0 and 1.

    Parameters
    ----------
    outcomes : np.ndarray
        1D array of binary measurement outcomes (0 and 1).

    Returns
    -------
    Dict[int, int]
        Dictionary mapping outcomes 0 and 1 to their respective integer counts.

    Raises
    ------
    TypeError
        If outcomes is not a NumPy array or sequence of integers.
    ValueError
        If outcomes is empty or contains values other than 0 and 1.
    """
    arr = np.asarray(outcomes)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("outcomes must be a non-empty 1D array")

    # Verify binary domain {0, 1}
    valid_mask = (arr == 0) | (arr == 1)
    if not np.all(valid_mask):
        raise ValueError("outcomes must contain only binary values 0 and 1")

    count_1 = int(np.sum(arr == 1))
    count_0 = int(arr.size - count_1)
    return {0: count_0, 1: count_1}


def calculate_frequencies(outcomes: np.ndarray) -> Dict[int, float]:
    """Calculate the observed frequencies (relative proportions) of outcomes 0 and 1.

    Parameters
    ----------
    outcomes : np.ndarray
        1D array of binary measurement outcomes (0 and 1).

    Returns
    -------
    Dict[int, float]
        Dictionary mapping outcomes 0 and 1 to their observed relative frequencies in [0.0, 1.0].
    """
    counts = calculate_counts(outcomes)
    total = counts[0] + counts[1]
    return {0: counts[0] / total, 1: counts[1] / total}


def calculate_absolute_deviations(
    observed_frequencies: Dict[int, float],
    theoretical_probabilities: Dict[int, float],
) -> Dict[int, float]:
    """Calculate absolute deviations between observed frequencies and theoretical probabilities.

    Parameters
    ----------
    observed_frequencies : Dict[int, float]
        Observed frequencies for outcomes 0 and 1.
    theoretical_probabilities : Dict[int, float]
        Theoretical Born-rule probabilities for outcomes 0 and 1.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping outcomes 0 and 1 to absolute deviations |f(x) - P(x)|.
    """
    for outcome in (0, 1):
        if outcome not in observed_frequencies:
            raise KeyError(f"Missing outcome {outcome} in observed_frequencies")
        if outcome not in theoretical_probabilities:
            raise KeyError(f"Missing outcome {outcome} in theoretical_probabilities")

    return {
        0: abs(observed_frequencies[0] - theoretical_probabilities[0]),
        1: abs(observed_frequencies[1] - theoretical_probabilities[1]),
    }


def theoretical_standard_error(p: float, shots: int) -> float:
    """Calculate the theoretical standard error of the observed frequency of an outcome.

    Under independent Bernoulli trials, the standard error of the sample proportion is:
        SE = sqrt(p * (1 - p) / N)

    For deterministic outcomes (p = 0 or p = 1), SE is exactly 0.0.

    Parameters
    ----------
    p : float
        Theoretical probability of the target outcome (0.0 <= p <= 1.0).
    shots : int
        Number of independent measurements (N >= 1).

    Returns
    -------
    float
        Theoretical standard error.

    Raises
    ------
    TypeError
        If p or shots have invalid types.
    ValueError
        If p is outside [0.0, 1.0] or shots < 1.
    """
    if not isinstance(p, (float, int)) or isinstance(p, bool):
        raise TypeError(f"p must be a float or int, got {type(p).__name__}")
    if not isinstance(shots, (int, np.integer)) or isinstance(shots, bool):
        raise TypeError(f"shots must be an integer, got {type(shots).__name__}")

    p_val = float(p)
    if not (0.0 <= p_val <= 1.0):
        raise ValueError(f"Probability p must be in [0.0, 1.0], got {p_val}")
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")

    variance = p_val * (1.0 - p_val)
    if variance <= 0.0:
        return 0.0
    return math.sqrt(variance / shots)


def calculate_cumulative_frequencies(
    outcomes: np.ndarray,
    target_outcome: int = 1,
) -> np.ndarray:
    """Calculate the running cumulative frequency of a given outcome across sequential measurements.

    Parameters
    ----------
    outcomes : np.ndarray
        1D array of binary measurement outcomes.
    target_outcome : int, optional
        Target outcome (0 or 1) whose cumulative frequency is tracked. Default is 1.

    Returns
    -------
    np.ndarray
        1D array of cumulative frequencies, where element n-1 is the proportion
        of `target_outcome` observed in the first n shots.

    Raises
    ------
    ValueError
        If outcomes is empty or target_outcome is not in {0, 1}.
    """
    if target_outcome not in (0, 1):
        raise ValueError(f"target_outcome must be 0 or 1, got {target_outcome}")

    arr = np.asarray(outcomes)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("outcomes must be a non-empty 1D array")

    valid_mask = (arr == 0) | (arr == 1)
    if not np.all(valid_mask):
        raise ValueError("outcomes must contain only binary values 0 and 1")

    cumulative_hits = np.cumsum(arr == target_outcome)
    shot_indices = np.arange(1, arr.size + 1, dtype=float)
    return cumulative_hits / shot_indices
