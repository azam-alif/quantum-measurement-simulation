"""
Computational-basis measurement sampling for single-qubit quantum states.
"""

from __future__ import annotations

from typing import Optional, Union
import numpy as np

from src.quantum_state import QuantumState


def measure_state(
    state: QuantumState,
    shots: int,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[Union[int, np.random.SeedSequence]] = None,
) -> np.ndarray:
    """Simulate computational-basis measurement outcomes using the Born rule.

    Note
    ----
    This function performs an ideal numerical simulation of quantum measurement
    statistics based on the theoretical probabilities P(0) = |alpha|^2 and
    P(1) = |beta|^2. It does not measure physical quantum hardware.

    Parameters
    ----------
    state : QuantumState
        A valid, normalized pure single-qubit state.
    shots : int
        Number of independent measurements to simulate. Must be an integer >= 1.
    rng : Optional[np.random.Generator], optional
        NumPy Generator instance to ensure isolated, explicit random state.
        If None, a new Generator is created using `seed`.
    seed : Optional[Union[int, np.random.SeedSequence]], optional
        Random seed for reproducibility when `rng` is not provided.

    Returns
    -------
    np.ndarray
        1D NumPy array of length `shots` containing binary measurement outcomes (0 or 1).

    Raises
    ------
    TypeError
        If `state` is not a QuantumState, `shots` is not an integer, or `rng` is invalid.
    ValueError
        If `shots` < 1.
    """
    if not isinstance(state, QuantumState):
        raise TypeError(f"state must be an instance of QuantumState, got {type(state).__name__}")

    if not isinstance(shots, (int, np.integer)) or isinstance(shots, bool):
        raise TypeError(f"shots must be an integer, got {type(shots).__name__}")

    if shots < 1:
        raise ValueError(f"shots must be a positive integer (>= 1), got {shots}")

    if rng is None:
        generator = np.random.default_rng(seed)
    else:
        if not isinstance(rng, np.random.Generator):
            raise TypeError(f"rng must be an instance of np.random.Generator, got {type(rng).__name__}")
        generator = rng

    # Theoretical probabilities from the Born rule
    p0 = state.prob_0
    p1 = state.prob_1

    # Floating-point safety: normalize sum strictly to 1.0 for np.random.Generator.choice
    p_sum = p0 + p1
    probabilities = [p0 / p_sum, p1 / p_sum]

    # Sample binary outcomes strictly restricted to {0, 1}
    outcomes = generator.choice([0, 1], size=shots, p=probabilities)
    return outcomes
