import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


N_SAMPLES_TRAIN = 2000
N_SAMPLES_TEST = 20000
N_BINS = 20
NOISE_BLEND_FACTOR = 0.4

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_OUTPUT_PATH = SCRIPT_DIR / "synthetic_classification_data.csv"
TEST_OUTPUT_PATH = SCRIPT_DIR / "test_data.csv"


def generate_data(
    n_samples: int,
    alpha: float,
    linear_mode: bool = False,
    noise_mode: bool = False,
) -> pd.DataFrame:
    if noise_mode:
        print("--- Running in PURE NOISE mode ---")
        var1 = np.random.uniform(-3, 3, n_samples)
        var2 = np.random.uniform(-1.5, 1.5, n_samples)
        clean_logit = np.zeros(n_samples)
    elif linear_mode:
        print("--- Running in LINEAR mode ---")
        var1 = np.random.uniform(-3, 3, n_samples)
        var2 = np.random.uniform(-1.5, 1.5, n_samples)
        clean_logit = var1
    else:
        print("--- Running in NON-LINEAR mode (default) ---")
        var1 = np.random.uniform(0, 2 * np.pi, n_samples)
        var2 = np.random.uniform(-1.5, 1.5, n_samples)
        clean_logit = np.sin(var1) + var2

    clean_probability = 1 / (1 + np.exp(-clean_logit))
    final_probability = (1 - alpha) * clean_probability + alpha * 0.5
    outcome = (np.random.uniform(0, 1, n_samples) < final_probability).astype(int)

    return pd.DataFrame(
        {
            "variable_one": var1,
            "variable_two": var2,
            "clean_probability": clean_probability,
            "final_probability": final_probability,
            "outcome": outcome,
        }
    )


def create_binned_plots(
    df: pd.DataFrame,
    alpha: float,
    linear_mode: bool = False,
    noise_mode: bool = False,
) -> None:
    mode = "Pure Noise" if noise_mode else "Linear" if linear_mode else "Non-Linear"
    print("\nGenerating plots from the test set...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    fig.suptitle(f"Binned Probability ({mode} Mode, alpha={alpha})", fontsize=16)

    for ax, variable in zip(axes, ["variable_one", "variable_two"], strict=True):
        bins = pd.cut(df[variable], bins=N_BINS)
        empirical = df.groupby(bins, observed=True)["outcome"].mean()
        expected = df.groupby(bins, observed=True)["final_probability"].mean()
        centers = np.array([interval.mid for interval in empirical.index])

        ax.plot(centers, empirical.values, "o-", label="Binned Empirical P(1)")
        ax.plot(centers, expected.values, "r--", label="True Final Probability")
        ax.set_title(f"{variable} vs. P(1)")
        ax.set_xlabel(variable)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    axes[0].set_ylabel('Proportion of "1"s in Bin')
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training and testing datasets for binary classification.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--linear", action="store_true")
    mode_group.add_argument("--noise", action="store_true")
    args = parser.parse_args()

    training_data = generate_data(
        N_SAMPLES_TRAIN,
        NOISE_BLEND_FACTOR,
        linear_mode=args.linear,
        noise_mode=args.noise,
    )
    training_data.to_csv(TRAIN_OUTPUT_PATH, index=False)
    print(f"\nTraining data ({N_SAMPLES_TRAIN} rows) saved to {TRAIN_OUTPUT_PATH}")
    print(training_data.head())

    test_data = generate_data(
        N_SAMPLES_TEST,
        NOISE_BLEND_FACTOR,
        linear_mode=args.linear,
        noise_mode=args.noise,
    )
    test_data.to_csv(TEST_OUTPUT_PATH, index=False)
    print(f"\nTest data ({N_SAMPLES_TEST} rows) saved to {TEST_OUTPUT_PATH}")
    print(test_data["outcome"].value_counts(normalize=True))

    create_binned_plots(
        test_data,
        alpha=NOISE_BLEND_FACTOR,
        linear_mode=args.linear,
        noise_mode=args.noise,
    )


if __name__ == "__main__":
    main()
