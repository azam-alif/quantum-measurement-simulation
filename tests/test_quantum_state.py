"""
Automated test suite for single-qubit quantum state representation,
measurement simulation, and statistical analysis.
"""

from __future__ import annotations

import math
import unittest
import numpy as np

from src.quantum_state import QuantumState
from src.measurement import measure_state
from src.statistics import (
    calculate_counts,
    calculate_frequencies,
    calculate_absolute_deviations,
    theoretical_standard_error,
    calculate_cumulative_frequencies,
)


class TestQuantumState(unittest.TestCase):
    """Tests covering quantum state representation and Born-rule probabilities."""

    def test_01_valid_state_zero(self) -> None:
        """Test 1: Valid computational basis state |0>."""
        state = QuantumState(1.0, 0.0)
        self.assertAlmostEqual(state.prob_0, 1.0, places=9)
        self.assertAlmostEqual(state.prob_1, 0.0, places=9)
        self.assertEqual(state.alpha, 1.0 + 0j)
        self.assertEqual(state.beta, 0j)

    def test_02_valid_state_one(self) -> None:
        """Test 2: Valid computational basis state |1>."""
        state = QuantumState(0.0, 1.0)
        self.assertAlmostEqual(state.prob_0, 0.0, places=9)
        self.assertAlmostEqual(state.prob_1, 1.0, places=9)
        self.assertEqual(state.alpha, 0j)
        self.assertEqual(state.beta, 1.0 + 0j)

    def test_03_valid_balanced_superposition(self) -> None:
        """Test 3: Valid balanced superposition (|0> + |1>) / sqrt(2)."""
        amplitude = 1.0 / math.sqrt(2.0)
        state = QuantumState(amplitude, amplitude)
        self.assertAlmostEqual(state.prob_0, 0.5, places=9)
        self.assertAlmostEqual(state.prob_1, 0.5, places=9)

    def test_04_correct_born_rule_probabilities(self) -> None:
        """Test 4: Correct Born-rule probabilities for real and complex amplitudes."""
        # Asymmetric state: sqrt(0.36) |0> + sqrt(0.64) |1> = 0.6 |0> + 0.8 |1>
        state_asym = QuantumState(0.6, 0.8)
        self.assertAlmostEqual(state_asym.prob_0, 0.36, places=9)
        self.assertAlmostEqual(state_asym.prob_1, 0.64, places=9)

        # Complex amplitudes: (i / sqrt(2)) |0> + (1 / sqrt(2)) |1>
        c_alpha = 1j / math.sqrt(2.0)
        c_beta = 1.0 / math.sqrt(2.0)
        state_complex = QuantumState(c_alpha, c_beta)
        self.assertAlmostEqual(state_complex.prob_0, 0.5, places=9)
        self.assertAlmostEqual(state_complex.prob_1, 0.5, places=9)

    def test_05_rejection_of_unnormalized_amplitudes(self) -> None:
        """Test 5: Strict rejection of unnormalized amplitudes (no silent normalization)."""
        with self.assertRaises(ValueError):
            QuantumState(1.0, 1.0)

        with self.assertRaises(ValueError):
            QuantumState(0.5, 0.5)

        with self.assertRaises(ValueError):
            QuantumState(0.0, 0.0)

    def test_06_rejection_of_non_finite_amplitudes(self) -> None:
        """Test 6: Strict rejection of non-finite amplitudes (NaN, Inf)."""
        with self.assertRaises(ValueError):
            QuantumState(float("nan"), 0.0)

        with self.assertRaises(ValueError):
            QuantumState(0.0, float("nan"))

        with self.assertRaises(ValueError):
            QuantumState(float("inf"), 0.0)

        with self.assertRaises(ValueError):
            QuantumState(0.0, float("-inf"))

    def test_invalid_types_raise_type_error(self) -> None:
        """Test that non-numeric types raise TypeError."""
        with self.assertRaises(TypeError):
            QuantumState("1.0", 0.0)  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            QuantumState(0.0, [1.0])  # type: ignore[arg-type]


class TestMeasurement(unittest.TestCase):
    """Tests covering measurement simulation in the computational basis."""

    def setUp(self) -> None:
        self.balanced_state = QuantumState(1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
        self.state_zero = QuantumState(1.0, 0.0)
        self.state_one = QuantumState(0.0, 1.0)

    def test_07_invalid_shot_counts(self) -> None:
        """Test 7: Clear handling of invalid shot counts."""
        with self.assertRaises(ValueError):
            measure_state(self.balanced_state, shots=0)

        with self.assertRaises(ValueError):
            measure_state(self.balanced_state, shots=-5)

        with self.assertRaises(TypeError):
            measure_state(self.balanced_state, shots=10.5)  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            measure_state(self.balanced_state, shots="100")  # type: ignore[arg-type]

    def test_08_correct_number_of_outcomes(self) -> None:
        """Test 8: Output array length strictly matches requested shot count."""
        for shots in [1, 10, 100, 1000]:
            outcomes = measure_state(self.balanced_state, shots=shots, seed=42)
            self.assertEqual(len(outcomes), shots)
            self.assertEqual(outcomes.shape, (shots,))

    def test_09_outcomes_restricted_to_zero_and_one(self) -> None:
        """Test 9: Outcomes are strictly binary (elements of {0, 1})."""
        outcomes = measure_state(self.balanced_state, shots=500, seed=123)
        unique_vals = set(np.unique(outcomes))
        self.assertTrue(unique_vals.issubset({0, 1}))

    def test_10_reproducibility_with_fixed_seed(self) -> None:
        """Test 10: Fixed seeds produce identical outcomes; differing seeds vary."""
        outcomes_1 = measure_state(self.balanced_state, shots=200, seed=999)
        outcomes_2 = measure_state(self.balanced_state, shots=200, seed=999)
        outcomes_3 = measure_state(self.balanced_state, shots=200, seed=1000)

        np.testing.assert_array_equal(outcomes_1, outcomes_2)
        self.assertFalse(np.array_equal(outcomes_1, outcomes_3))


class TestStatistics(unittest.TestCase):
    """Tests covering statistical calculations on measurement outcomes."""

    def test_11_outcome_counts_and_frequencies(self) -> None:
        """Test 11: Correct outcome counts, frequencies, and deviations."""
        sample = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1])  # 4 zeros, 6 ones
        counts = calculate_counts(sample)
        self.assertEqual(counts[0], 4)
        self.assertEqual(counts[1], 6)

        freqs = calculate_frequencies(sample)
        self.assertAlmostEqual(freqs[0], 0.4)
        self.assertAlmostEqual(freqs[1], 0.6)

        deviations = calculate_absolute_deviations(freqs, {0: 0.5, 1: 0.5})
        self.assertAlmostEqual(deviations[0], 0.1)
        self.assertAlmostEqual(deviations[1], 0.1)

    def test_12_cumulative_frequency_calculation(self) -> None:
        """Test 12: Correct running cumulative frequency progression."""
        sample = np.array([1, 0, 1, 1])
        # Cumulative ones: [1, 1, 2, 3]
        # Cumulative indices: [1, 2, 3, 4]
        # Cumulative frequencies: [1.0, 0.5, 2/3, 0.75]
        cum_freqs = calculate_cumulative_frequencies(sample, target_outcome=1)
        expected = np.array([1.0, 0.5, 2.0 / 3.0, 0.75])
        np.testing.assert_allclose(cum_freqs, expected)

        # Check cumulative frequencies for outcome 0
        cum_freqs_0 = calculate_cumulative_frequencies(sample, target_outcome=0)
        expected_0 = np.array([0.0, 0.5, 1.0 / 3.0, 0.25])
        np.testing.assert_allclose(cum_freqs_0, expected_0)

    def test_13_theoretical_standard_error(self) -> None:
        """Test 13: Correct theoretical standard error calculation SE = sqrt(p(1-p)/N)."""
        # p = 0.5, N = 100 -> sqrt(0.25 / 100) = 0.05
        se_100 = theoretical_standard_error(0.5, 100)
        self.assertAlmostEqual(se_100, 0.05, places=9)

        # p = 0.2, N = 1000 -> sqrt(0.16 / 1000)
        se_asym = theoretical_standard_error(0.2, 1000)
        self.assertAlmostEqual(se_asym, math.sqrt(0.16 / 1000), places=9)

        # Invalid probability or shots
        with self.assertRaises(ValueError):
            theoretical_standard_error(-0.1, 100)
        with self.assertRaises(ValueError):
            theoretical_standard_error(1.1, 100)
        with self.assertRaises(ValueError):
            theoretical_standard_error(0.5, 0)

    def test_14_deterministic_probabilities_and_eigenstates(self) -> None:
        """Test 14: Correct handling of deterministic probabilities (p=0 and p=1)."""
        # Theoretical SE for p=0 and p=1 is exactly 0.0
        self.assertEqual(theoretical_standard_error(0.0, 500), 0.0)
        self.assertEqual(theoretical_standard_error(1.0, 500), 0.0)

        # Eigenstate |0> produces strictly 0 on every measurement
        state_0 = QuantumState(1.0, 0.0)
        outcomes_0 = measure_state(state_0, shots=200, seed=42)
        counts_0 = calculate_counts(outcomes_0)
        self.assertEqual(counts_0[0], 200)
        self.assertEqual(counts_0[1], 0)

        freqs_0 = calculate_frequencies(outcomes_0)
        self.assertEqual(freqs_0[0], 1.0)
        self.assertEqual(freqs_0[1], 0.0)

        dev_0 = calculate_absolute_deviations(freqs_0, {0: 1.0, 1: 0.0})
        self.assertEqual(dev_0[0], 0.0)
        self.assertEqual(dev_0[1], 0.0)

        # Eigenstate |1> produces strictly 1 on every measurement
        state_1 = QuantumState(0.0, 1.0)
        outcomes_1 = measure_state(state_1, shots=200, seed=42)
        counts_1 = calculate_counts(outcomes_1)
        self.assertEqual(counts_1[0], 0)
        self.assertEqual(counts_1[1], 200)

        freqs_1 = calculate_frequencies(outcomes_1)
        self.assertEqual(freqs_1[0], 0.0)
        self.assertEqual(freqs_1[1], 1.0)

        dev_1 = calculate_absolute_deviations(freqs_1, {0: 0.0, 1: 1.0})
        self.assertEqual(dev_1[0], 0.0)
        self.assertEqual(dev_1[1], 0.0)


if __name__ == "__main__":
    unittest.main()
