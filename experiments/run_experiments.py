"""
Command-line experiment runner for quantum measurement simulations.

Executes Experiments A, B, and C:
- Experiment A: Measurement distribution for a balanced state (N = 1,000).
- Experiment B: Cumulative frequency convergence toward the theoretical probability (N = 10,000).
- Experiment C: Statistical deviation (RMSE) vs. sample size comparing simulated outcomes with the theoretical standard error (SE = sqrt(p(1-p)/N)).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Ensure non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quantum_state import QuantumState
from src.measurement import measure_state
from src.statistics import (
    calculate_counts,
    calculate_frequencies,
    calculate_absolute_deviations,
    theoretical_standard_error,
    calculate_cumulative_frequencies,
)

RESULTS_DIR = PROJECT_ROOT / "results"


def run_experiment_a() -> dict:
    """Experiment A: Measurement distribution for a balanced single-qubit state."""
    print("=" * 70)
    print("EXPERIMENT A: Measurement Distribution (N = 1,000 shots)")
    print("=" * 70)

    # Prepare balanced superposition: (|0> + |1>) / sqrt(2)
    state = QuantumState(1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
    shots = 1000
    seed = 42

    outcomes = measure_state(state, shots=shots, seed=seed)
    counts = calculate_counts(outcomes)
    frequencies = calculate_frequencies(outcomes)
    theoretical = {0: state.prob_0, 1: state.prob_1}
    deviations = calculate_absolute_deviations(frequencies, theoretical)

    print(f"State: ({state.alpha:.4f})|0> + ({state.beta:.4f})|1>")
    print(f"Theoretical Probabilities: P(0) = {theoretical[0]:.4f}, P(1) = {theoretical[1]:.4f}")
    print(f"Simulated Shots:           {shots}")
    print("-" * 70)
    print(f"{'Outcome':<10}{'Count':<10}{'Observed Freq':<16}{'Theoretical P':<16}{'Abs Deviation':<16}")
    print("-" * 70)
    for outcome in (0, 1):
        print(
            f"{outcome:<10}{counts[outcome]:<10}{frequencies[outcome]:<16.4f}"
            f"{theoretical[outcome]:<16.4f}{deviations[outcome]:<16.4f}"
        )
    print("=" * 70 + "\n")

    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    x_positions = np.array([0, 1])
    bar_width = 0.32

    bars_theo = ax.bar(
        x_positions - bar_width / 2,
        [theoretical[0], theoretical[1]],
        width=bar_width,
        color="#2b5c8f",
        alpha=0.85,
        label="Theoretical Born Probability $P(x)$",
    )
    bars_obs = ax.bar(
        x_positions + bar_width / 2,
        [frequencies[0], frequencies[1]],
        width=bar_width,
        color="#e07a5f",
        alpha=0.85,
        label=f"Observed Frequency ($N={shots}$)",
    )

    # Bar annotations
    for bar in bars_theo:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    for bar in bars_obs:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Single-Qubit Measurement Distribution (Born Rule vs. Observation)", fontsize=13, pad=12)
    ax.set_xlabel("Computational Basis Outcome", fontsize=11)
    ax.set_ylabel("Probability / Relative Frequency", fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["$|0\\rangle$", "$|1\\rangle$"], fontsize=12)
    ax.set_ylim(0, 0.65)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    fig.tight_layout()
    plot_path = RESULTS_DIR / "measurement_distribution.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot_path}")

    return {
        "counts": counts,
        "frequencies": frequencies,
        "deviations": deviations,
    }


def run_experiment_b() -> np.ndarray:
    """Experiment B: Cumulative frequency convergence toward theoretical probability."""
    print("=" * 70)
    print("EXPERIMENT B: Cumulative-Frequency Convergence (N = 10,000 shots)")
    print("=" * 70)

    state = QuantumState(1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
    shots = 10000
    seed = 101

    outcomes = measure_state(state, shots=shots, seed=seed)
    cumulative_freq_1 = calculate_cumulative_frequencies(outcomes, target_outcome=1)

    final_freq = cumulative_freq_1[-1]
    final_dev = abs(final_freq - state.prob_1)
    print(f"Total simulated measurements: {shots}")
    print(f"Theoretical P(1):             {state.prob_1:.4f}")
    print(f"Final cumulative freq f(1):  {final_freq:.4f}")
    print(f"Final absolute deviation:     {final_dev:.4f}")
    print("=" * 70 + "\n")

    # Plot convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    shot_indices = np.arange(1, shots + 1)

    ax.plot(
        shot_indices,
        cumulative_freq_1,
        color="#1f77b4",
        linewidth=1.2,
        label=r"Observed Cumulative Frequency $f_n(1) = \frac{1}{n}\sum_{i=1}^n x_i$",
    )
    ax.axhline(
        state.prob_1,
        color="#d62728",
        linestyle="--",
        linewidth=1.6,
        label=f"Theoretical Probability $P(1) = {state.prob_1:.1f}$",
    )

    ax.set_title("Convergence of Observed Frequency to Theoretical Born Probability", fontsize=13, pad=12)
    ax.set_xlabel("Number of Measurements ($n$)", fontsize=11)
    ax.set_ylabel("Cumulative Observed Frequency of Outcome 1", fontsize=11)
    ax.set_xlim(1, shots)
    ax.set_ylim(0.40, 0.60)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="upper right", fontsize=10)

    fig.tight_layout()
    plot_path = RESULTS_DIR / "convergence.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot_path}")

    return cumulative_freq_1


def run_experiment_c() -> dict:
    """Experiment C: Statistical deviation (RMSE) vs. sample size."""
    print("=" * 70)
    print("EXPERIMENT C: Statistical Deviation Scaling (RMSE vs. Sample Size N)")
    print("=" * 70)

    state = QuantumState(1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
    p = state.prob_1
    sample_sizes = [10, 100, 1000, 10000]
    repetitions = 500
    base_seed = 2024

    rng = np.random.default_rng(base_seed)

    simulated_rmse = []
    theoretical_se = []

    print(f"Repetitions per sample size M: {repetitions}")
    print(f"Theoretical probability p:     {p:.4f}")
    print("-" * 70)
    print(f"{'Sample Size (N)':<18}{'Theoretical SE':<18}{'Simulated RMSE':<18}{'Ratio (RMSE/SE)':<16}")
    print("-" * 70)

    for n in sample_sizes:
        # Generate M independent simulated experiments of size N
        # Outcomes matrix: shape (repetitions, n)
        batch_outcomes = rng.choice([0, 1], size=(repetitions, n), p=[state.prob_0, state.prob_1])
        # Observed frequencies of outcome 1 for each repetition: shape (repetitions,)
        observed_frequencies_1 = np.mean(batch_outcomes == 1, axis=1)

        # Root-mean-square deviation from theoretical probability p
        rmse = float(np.sqrt(np.mean((observed_frequencies_1 - p) ** 2)))
        se = theoretical_standard_error(p, n)

        simulated_rmse.append(rmse)
        theoretical_se.append(se)

        ratio = rmse / se if se > 0 else 1.0
        print(f"{n:<18}{se:<18.5f}{rmse:<18.5f}{ratio:<16.4f}")

    print("=" * 70 + "\n")

    # Plot RMSE vs SE on log-log axes
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        sample_sizes,
        theoretical_se,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=r"Theoretical Standard Error $\mathrm{SE} = \sqrt{\frac{p(1-p)}{N}}$",
    )
    ax.plot(
        sample_sizes,
        simulated_rmse,
        color="#1f77b4",
        marker="o",
        markersize=7,
        linewidth=1.5,
        label=f"Simulated RMSE ($M={repetitions}$ trials)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"Statistical Deviation vs. Sample Size ($1/\sqrt{N}$ Scaling)", fontsize=13, pad=12)
    ax.set_xlabel("Sample Size $N$ (Number of Measurements)", fontsize=11)
    ax.set_ylabel(r"Deviation from Theoretical Probability ($p=0.5$)", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    fig.tight_layout()
    plot_path = RESULTS_DIR / "statistical_deviation.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot_path}")

    return {
        "sample_sizes": sample_sizes,
        "theoretical_se": theoretical_se,
        "simulated_rmse": simulated_rmse,
    }


def main() -> None:
    """Run all experiments and ensure results directory exists."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\nStarting Quantum Measurement Simulation Experiments...\n")
    run_experiment_a()
    run_experiment_b()
    run_experiment_c()
    print("All experiments completed successfully. Figures saved in results/.\n")


if __name__ == "__main__":
    main()
