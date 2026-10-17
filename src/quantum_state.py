"""
Pure single-qubit quantum state representation.
"""

from __future__ import annotations

import math
from typing import Union

ComplexLike = Union[complex, float, int]


class QuantumState:
    """Represents a pure single-qubit state: |psi> = alpha|0> + beta|1>.

    Parameters
    ----------
    alpha : complex, float, or int
        Probability amplitude associated with computational basis state |0>.
    beta : complex, float, or int
        Probability amplitude associated with computational basis state |1>.
    tolerance : float, optional
        Numerical tolerance for normalization check (|alpha|^2 + |beta|^2 == 1).
        Default is 1e-9.

    Raises
    ------
    TypeError
        If alpha or beta are not numeric (complex, float, or int), or tolerance is not a float/int.
    ValueError
        If amplitudes are non-finite (NaN or Inf), tolerance is non-positive,
        or the state is not normalized within the specified tolerance.
        Silent normalization is strictly disallowed.
    """

    def __init__(
        self,
        alpha: ComplexLike,
        beta: ComplexLike,
        tolerance: float = 1e-9,
    ) -> None:
        if not isinstance(alpha, (complex, float, int)) or isinstance(alpha, bool):
            raise TypeError(f"alpha must be numeric (complex, float, int), got {type(alpha).__name__}")
        if not isinstance(beta, (complex, float, int)) or isinstance(beta, bool):
            raise TypeError(f"beta must be numeric (complex, float, int), got {type(beta).__name__}")
        if not isinstance(tolerance, (float, int)) or isinstance(tolerance, bool):
            raise TypeError(f"tolerance must be numeric, got {type(tolerance).__name__}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")

        self._alpha = complex(alpha)
        self._beta = complex(beta)
        self._tolerance = float(tolerance)

        # Validate that amplitudes are finite (reject NaN / inf)
        if not (math.isfinite(self._alpha.real) and math.isfinite(self._alpha.imag)):
            raise ValueError(f"alpha amplitude must be finite, got {alpha}")
        if not (math.isfinite(self._beta.real) and math.isfinite(self._beta.imag)):
            raise ValueError(f"beta amplitude must be finite, got {beta}")

        # Compute Born-rule theoretical probabilities
        # P(0) = |alpha|^2 and P(1) = |beta|^2
        self._prob_0 = self._alpha.real**2 + self._alpha.imag**2
        self._prob_1 = self._beta.real**2 + self._beta.imag**2

        norm = self._prob_0 + self._prob_1
        deviation = abs(norm - 1.0)
        if deviation > self._tolerance:
            raise ValueError(
                f"Quantum state is not normalized: |alpha|^2 + |beta|^2 = {norm:.10f} "
                f"(deviation {deviation:.2e} exceeds tolerance {self._tolerance:.2e}). "
                "Silent normalization is disallowed."
            )

    @property
    def alpha(self) -> complex:
        """Probability amplitude for |0>."""
        return self._alpha

    @property
    def beta(self) -> complex:
        """Probability amplitude for |1>."""
        return self._beta

    @property
    def prob_0(self) -> float:
        """Born-rule theoretical probability P(0) = |alpha|^2."""
        return self._prob_0

    @property
    def prob_1(self) -> float:
        """Born-rule theoretical probability P(1) = |beta|^2."""
        return self._prob_1

    @property
    def tolerance(self) -> float:
        """Normalization tolerance used for state validation."""
        return self._tolerance

    def __repr__(self) -> str:
        return f"QuantumState(alpha={self._alpha}, beta={self._beta}, tolerance={self._tolerance})"

    def __str__(self) -> str:
        return f"({self._alpha})|0> + ({self._beta})|1>"
