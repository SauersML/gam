# compare.py
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(r_csv, rust_csv):
    """Load and merge benchmark results"""
    r_df = pd.read_csv(r_csv)
    rust_df = pd.read_csv(rust_csv)

    # Rename columns for clarity
    r_df = r_df.rename(columns={"median_us": "r_median_us"})
    rust_df = rust_df.rename(columns={"time_us": "rust_median_us"})

    # Merge results
    comparison = pd.merge(r_df, rust_df, on=["n", "b", "c"], how="inner")

    # Calculate speedup
    comparison["speedup"] = comparison["r_median_us"] / comparison["rust_median_us"]

    return comparison


def plot_comparison(comparison, output_dir):
    """Generate comparison plots with reference lines and improved gridlines"""
    sns.set_theme(style="whitegrid", rc={"grid.linestyle": ":", "grid.alpha": 0.5})
    os.makedirs(output_dir, exist_ok=True)

    def add_reference_line(ax):
        """Helper to add a reference line at y=1"""
        ax.axhline(1, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.3)
        ax.minorticks_on()
        ax.grid(which="minor", linestyle=":", alpha=0.2)

    # Plot 1: Speedup by parameter b
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x="b", y="speedup", data=comparison)
    plt.title("Speedup vs Shape Parameter (b) for n=10000", pad=20)
    plt.yscale("log")
    plt.ylim(0.5, 100)
    plt.ylabel("Speedup (R / Rust)", labelpad=10)
    plt.xlabel("Shape Parameter (b)", labelpad=10)
    add_reference_line(ax)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/speedup_vs_b.png", bbox_inches="tight", dpi=300)
    plt.close()

    # Plot 2: Speedup by parameter c
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x="c", y="speedup", data=comparison)
    plt.title("Speedup vs Tilt Parameter (c) for n=10000", pad=20)
    plt.yscale("log")
    plt.ylim(0.5, 100)
    plt.ylabel("Speedup (R / Rust)", labelpad=10)
    plt.xlabel("Tilt Parameter (c)", labelpad=10)
    add_reference_line(ax)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/speedup_vs_c.png", bbox_inches="tight", dpi=300)
    plt.close()

    # Detailed summary
    with open(f"{output_dir}/summary.txt", "w") as f:
        # Header
        f.write("POLYA-GAMMA BENCHMARK SUMMARY\n")
        f.write("===========================\n\n")

        # Basic statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-----------------\n")
        f.write(f"Total comparisons: {len(comparison):,}\n")
        f.write(f"Average speedup: {comparison.speedup.mean():.2f}x\n")
        f.write(f"Median speedup: {comparison.speedup.median():.2f}x\n")
        f.write(f"Standard deviation: {comparison.speedup.std():.2f}x\n")
        f.write(f"Minimum speedup: {comparison.speedup.min():.2f}x\n")
        f.write(f"25th percentile: {comparison.speedup.quantile(0.25):.2f}x\n")
        f.write(f"75th percentile: {comparison.speedup.quantile(0.75):.2f}x\n")
        f.write(f"Maximum speedup: {comparison.speedup.max():.2f}x\n\n")

        # Breakdown by parameter 'b'
        f.write("BREAKDOWN BY SHAPE PARAMETER (b)\n")
        f.write("-------------------------------\n")
        for b_val in sorted(comparison["b"].unique()):
            subset = comparison[comparison["b"] == b_val]
            f.write(f"b = {b_val:.1f}: ")
            f.write(f"{subset.speedup.median():.2f}x median ")
            f.write(f"({subset.speedup.min():.2f}x - {subset.speedup.max():.2f}x)\n")
        f.write("\n")

        # Breakdown by parameter 'c'
        f.write("BREAKDOWN BY TILT PARAMETER (c)\n")
        f.write("-------------------------------\n")
        for c_val in sorted(comparison["c"].unique()):
            subset = comparison[comparison["c"] == c_val]
            f.write(f"c = {c_val:.1f}: ")
            f.write(f"{subset.speedup.median():.2f}x median ")
            f.write(f"({subset.speedup.min():.2f}x - {subset.speedup.max():.2f}x)\n")
        f.write("\n")

        # Performance summary
        f.write("PERFORMANCE SUMMARY\n")
        f.write("------------------\n")
        faster = len(comparison[comparison["speedup"] > 1])
        slower = len(comparison[comparison["speedup"] < 1])
        same = len(comparison[comparison["speedup"] == 1])
        f.write(
            f"Rust faster than R: {faster} cases ({faster / len(comparison):.1%})\n"
        )
        f.write(
            f"Rust slower than R: {slower} cases ({slower / len(comparison):.1%})\n"
        )
        f.write(f"Equal performance: {same} cases ({same / len(comparison):.1%})\n\n")

        # Best and worst cases
        best = comparison.loc[comparison["speedup"].idxmax()]
        worst = comparison.loc[comparison["speedup"].idxmin()]
        f.write("BEST CASE (for Rust)\n")
        f.write(f"- Parameters: b={best['b']}, c={best['c']}\n")
        f.write(f"- R time: {best['r_median_us']:.2f} µs\n")
        f.write(f"- Rust time: {best['rust_median_us']:.2f} µs\n")
        f.write(f"- Speedup: {best['speedup']:.2f}x\n\n")

        f.write("WORST CASE (for Rust)\n")
        f.write(f"- Parameters: b={worst['b']}, c={worst['c']}\n")
        f.write(f"- R time: {worst['r_median_us']:.2f} µs\n")
        f.write(f"- Rust time: {worst['rust_median_us']:.2f} µs\n")
        f.write(f"- Speedup: {worst['speedup']:.2f}x\n")

        # Add timestamp
        from datetime import datetime

        f.write(f"\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python compare.py <r_results.csv> <rust_results.csv> <output_dir>"
        )
        sys.exit(1)

    r_csv = sys.argv[1]
    rust_csv = sys.argv[2]
    output_dir = sys.argv[3]

    print("🔍 Loading benchmark results...")
    comparison = load_data(r_csv, rust_csv)

    print("📊 Generating comparison visualizations...")
    plot_comparison(comparison, output_dir)

    print(f"✅ Comparison complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
