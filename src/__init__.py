"""
Quantum measurement simulation package.

Provides pure single-qubit representation, Born-rule measurement simulation,
and statistical analysis tools.
"""

from src.quantum_state import QuantumState
from src.measurement import measure_state
from src.statistics import (
    calculate_counts,
    calculate_frequencies,
    calculate_absolute_deviations,
    theoretical_standard_error,
    calculate_cumulative_frequencies,
)

__all__ = [
    "QuantumState",
    "measure_state",
    "calculate_counts",
    "calculate_frequencies",
    "calculate_absolute_deviations",
    "theoretical_standard_error",
    "calculate_cumulative_frequencies",
]
