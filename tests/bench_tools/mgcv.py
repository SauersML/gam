import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# --- Constants for Data Generation ---
N_SAMPLES_TRAIN = 2000
N_SAMPLES_TEST = 20000
TRAIN_OUTPUT_FILENAME = 'synthetic_classification_data.csv'
TEST_OUTPUT_FILENAME = 'test_data.csv'
N_BINS = 20

# The noise level (alpha) from 0.0 (pure signal) to 1.0 (pure noise).
# A value of 0.4 means the final probability is a blend of 60% signal
# and a 40% pull towards the 50/50 baseline.
NOISE_BLEND_FACTOR = 0.4


def generate_data(n_samples, alpha, linear_mode=False, noise_mode=False):
    """
    Generates data by explicitly blending a clean signal with a 50/50
    noise probability. This provides a structured, interpretable way to
    control the signal-to-noise ratio.

    Args:
        n_samples (int): Number of samples to generate.
        alpha (float): The noise blend factor (0.0 to 1.0).
        linear_mode (bool): If True, use a linear signal.
        noise_mode (bool): If True, use a null signal.
    """
    # --- Stage 1: Calculate the perfect, "Clean" Signal and Probability ---
    if noise_mode:
        print("--- Running in PURE NOISE mode ---")
        # In pure noise mode, the clean signal is flat zero.
        # This means the clean probability is exactly 0.5 everywhere.
        clean_logit = np.zeros(n_samples)
        # Use standard variable ranges, they just don't affect the outcome.
        var1 = np.random.uniform(-3, 3, n_samples)
        var2 = np.random.uniform(-1.5, 1.5, n_samples)
    elif linear_mode:
        print("--- Running in LINEAR mode ---")
        var1 = np.random.uniform(-3, 3, n_samples)
        var2 = np.random.uniform(-1.5, 1.5, n_samples)
        clean_logit = var1  # var2 is intentionally omitted from the signal
    else: # Default non-linear mode
        print("--- Running in NON-LINEAR mode (default) ---")
        var1 = np.random.uniform(0, 2 * np.pi, n_samples)
        var2 = np.random.uniform(-1.5, 1.5, n_samples)
        clean_logit = np.sin(var1) + var2

    # The clean probability is the direct result of the signal, no noise.
    clean_probability = 1 / (1 + np.exp(-clean_logit))

    # --- Stage 2: Apply Noise via Probability Blending ---
    # The final probability is a direct, linear interpolation towards 0.5.
    # This is the "ground truth" the model must attempt to learn.
    final_probability = (1 - alpha) * clean_probability + (alpha * 0.5)

    # --- Stage 3: Generate the Binary Outcome ---
    # Perform a single random draw (Bernoulli trial) based on the FINAL probability.
    random_chance = np.random.uniform(0, 1, n_samples)
    outcome = (random_chance < final_probability).astype(int)

    # --- Stage 4: Create the DataFrame with all columns ---
    df = pd.DataFrame({
        'variable_one': var1,
        'variable_two': var2,
        'clean_probability': clean_probability, # The "ideal world" probability
        'final_probability': final_probability, # The "real world" ground truth
        'outcome': outcome                      # The binary result from the final probability
    })
    return df


def create_binned_plots(df, alpha, linear_mode=False, noise_mode=False):
    """
    Creates plots showing the binned probability of outcome=1 against each
    independent variable, adapted for the generation mode. It now also plots
    the true, blended probability.
    """
    print("\nGenerating plots (based on the large test set)...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # Determine the main title based on the mode
    if noise_mode:
        fig.suptitle(f'Binned Probability (Pure Noise Mode, alpha={alpha})', fontsize=16)
    elif linear_mode:
        fig.suptitle(f'Binned Probability (Linear Mode, alpha={alpha})', fontsize=16)
    else:
        fig.suptitle(f'Binned Probability (Non-Linear Mode, alpha={alpha})', fontsize=16)

    # --- Subplot 1: variable 1 vs. P(1) ---
    df['v1_bin'] = pd.cut(df['variable_one'], bins=N_BINS)
    binned_empirical = df.groupby('v1_bin', observed=True)['outcome'].mean()
    binned_true = df.groupby('v1_bin', observed=True)['final_probability'].mean()
    bin_centers_v1 = np.array([interval.mid for interval in binned_empirical.index])
    axes[0].plot(bin_centers_v1, binned_empirical.values, 'o-', label='Binned Empirical P(1)')
    axes[0].plot(bin_centers_v1, binned_true.values, 'r--', label='True Final Probability')

    axes[0].set_title('variable 1 vs. P(1)')
    axes[0].set_xlabel('variable 1' if not linear_mode else 'variable 1 (angle in radians)')
    axes[0].set_ylabel('Proportion of "1"s in Bin')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # --- Subplot 2: variable 2 vs. P(1) ---
    df['v2_bin'] = pd.cut(df['variable_two'], bins=N_BINS)
    binned_empirical_v2 = df.groupby('v2_bin', observed=True)['outcome'].mean()
    binned_true_v2 = df.groupby('v2_bin', observed=True)['final_probability'].mean()
    bin_centers_v2 = np.array([interval.mid for interval in binned_empirical_v2.index])
    axes[1].plot(bin_centers_v2, binned_empirical_v2.values, 'o-', label='Binned Empirical P(1)')
    axes[1].plot(bin_centers_v2, binned_true_v2.values, 'r--', label='True Final Probability')

    axes[1].set_title('variable 2 vs. P(1)')
    axes[1].set_xlabel('variable 2')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.show()


def main():
    """
    Parses command-line arguments and runs the data generation and plotting.
    """
    parser = argparse.ArgumentParser(
        description="Generate separate TRAINING and TESTING synthetic datasets for binary classification.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Group for mutually exclusive modes. Default is non-linear.
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--linear',
        action='store_true',
        help="""Run in 100%% linear mode:
  - 'variable_one' is linearly related to the log-odds of the outcome.
  - 'variable_two' is pure noise with no relationship to the outcome."""
    )
    mode_group.add_argument(
        '--noise',
        action='store_true',
        help="""Run in 100%% pure noise mode:
  - 'variable_one' has NO relationship to the outcome.
  - 'variable_two' has NO relationship to the outcome.
  - Both variables are pure noise (zero signal)."""
    )
    args = parser.parse_args()

    # --- 1. Generate TRAINING Data ---
    print("\n" + "="*50)
    print("### GENERATING TRAINING DATA ###")
    print("="*50)
    training_data = generate_data(
        N_SAMPLES_TRAIN,
        NOISE_BLEND_FACTOR,
        linear_mode=args.linear,
        noise_mode=args.noise
    )
    print(f"\nTraining data ({N_SAMPLES_TRAIN} rows) saved to '{TRAIN_OUTPUT_FILENAME}'")
    print("\nTraining Data Head (with new probability columns):")
    print(training_data.head())

    # --- 2. Generate TESTING Data ---
    print("\n" + "="*50)
    print("### GENERATING TESTING DATA ###")
    print("="*50)
    test_data = generate_data(
        N_SAMPLES_TEST,
        NOISE_BLEND_FACTOR,
        linear_mode=args.linear,
        noise_mode=args.noise
    )
    print(f"\nTest data ({N_SAMPLES_TEST} rows) saved to '{TEST_OUTPUT_FILENAME}'")
    print("\nTest Data Outcome Distribution (should be ~50/50):")
    print(test_data['outcome'].value_counts(normalize=True))

    # --- 3. Create Plots from the large, clean test set ---
    create_binned_plots(
        test_data.copy(), # Pass a copy to avoid SettingWithCopyWarning
        alpha=NOISE_BLEND_FACTOR,
        linear_mode=args.linear,
        noise_mode=args.noise
    )


if __name__ == "__main__":
    main()
