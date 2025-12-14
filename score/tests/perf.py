import subprocess
import sys
import os
import shutil
from pathlib import Path
from bench import RealisticDataGenerator # Re-use the data generator
import pandas as pd
import re

# --- Configuration ---
WORKDIR = Path("./perf_workdir")
# The binary should have profiling symbols. The CI workflow correctly uses a binary
# from `target/profiling/gnomon`.
GNOMON_BINARY = Path("./target/profiling/gnomon").resolve()
# All profiling runs are executed on a fixed-size random subset of individuals
PROFILING_SUBSET_PCT = 0.10

# Define a variety of workloads to profile the program under different conditions.
WORKLOADS = {
    # A small, clean workload.
    "small_clean": {
        "test_name": "small_clean",
        "n_individuals": 1_000,
        "genome_variants": 100_000,
        "target_variants": 40_000,
        "score_files": [
            {
                "name": "high_quality_score", "n_scores": 5, "gwas_source_variants": 40_000,
                "overlap_pct": 0.95, "flip_pct": 0.01, "missing_weight_pct": 0.0, "score_sparsity": 1.0,
            }
        ],
    },
    # A larger, more complex workload with multiple, heterogeneous score files.
    "large_complex": {
        "test_name": "large_complex",
        "n_individuals": 5_000,
        "genome_variants": 1_000_000,
        "target_variants": 200_000,
        "score_files": [
            {
                "name": "large_score_1", "n_scores": 10, "gwas_source_variants": 150_000,
                "overlap_pct": 0.90, "flip_pct": 0.05, "missing_weight_pct": 0.01, "score_sparsity": 1.0,
            },
            {
                "name": "large_score_2", "n_scores": 8, "gwas_source_variants": 120_000,
                "overlap_pct": 0.80, "flip_pct": 0.02, "missing_weight_pct": 0.05, "score_sparsity": 0.9,
            },
        ],
    },
    # A workload specifically designed to test performance on "messy" data.
    "messy_data": {
        "test_name": "messy_data",
        "n_individuals": 2_000,
        "genome_variants": 200_000,
        "target_variants": 100_000,
        "score_files": [
            {
                "name": "messy_score", "n_scores": 4, "gwas_source_variants": 50_000,
                "overlap_pct": 0.50, "flip_pct": 0.15, "missing_weight_pct": 0.20, "score_sparsity": 0.7,
            }
        ],
    },
}

def print_header(title: str, char: str = "="):
    """Prints a formatted header to the console for better log readability."""
    width = 80
    print("\n" + char * width, flush=True)
    print(f"{char*4} {title} {char*(width - len(title) - 6)}", flush=True)
    print(char * width, flush=True)

def run_command(cmd, title, capture_output=False, **kwargs):
    """
    Runs a command, streams its output, and on failure, prints the last 20 lines
    of output for easier debugging.
    Returns (success: bool, output_lines: list) if capture_output=True
    Returns success: bool if capture_output=False
    """
    print_header(title, char="-")
    cmd_str_list = [str(c) for c in cmd]
    print(f"Executing: {' '.join(cmd_str_list)}", flush=True)

    output_lines = []
    try:
        proc = subprocess.Popen(
            cmd_str_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', **kwargs
        )
        with proc.stdout:
            for line in iter(proc.stdout.readline, ''):
                sys.stdout.write(f"  > {line.rstrip()}\n")
                sys.stdout.flush()
                output_lines.append(line)
        retcode = proc.wait()

        if retcode != 0:
            print(f"\n❌ Command failed with exit code {retcode}", flush=True)
            print("--- Last 20 lines of output for debugging ---", flush=True)
            for l in output_lines[-20:]:
                print(f"  | {l.rstrip()}", flush=True)
            return (False, output_lines) if capture_output else False

        print("\n✅ Command successful.", flush=True)
        return (True, output_lines) if capture_output else True
    except Exception as e:
        print(f"❌ Failed to execute command: {e}", flush=True)
        return (False, output_lines) if capture_output else False

def find_hot_worker_functions(perf_output, n=2):
    """
    Parses perf report output to find the top N hottest 'self-time' functions
    in the gnomon:: namespace, excluding common non-worker functions.
    Handles multiple `perf report` formats for compatibility.
    """
    candidates = []
    # Patterns that indicate boilerplate or standard library functions to ignore
    skip_patterns = [
        '::main', '::new', '::from', '::into', '::drop',
        '::fmt', '::clone', '::default', '<impl', '::{{closure}}',
        '::__rust', 'std::', 'core::', 'alloc::', '::lang_start'
    ]

    for line in perf_output:
        pct, raw_symbol = None, None

        # Try matching different perf report formats, from most to least specific.
        # Format 1: Modern perf with children and self time
        match = re.match(r'\s*(\d+\.\d+)%\s+(\d+\.\d+)%\s+\S+\s+\S+\s+\[.\]\s+(.+)', line)
        if match:
            pct = float(match.group(2)) # Use self time
            raw_symbol = match.group(3)
        else:
            # Format 2: Older perf with command/object columns
            match = re.match(r'\s*(\d+\.\d+)%\s+\S+\s+\S+\s+\[.\]\s+(.+)', line)
            if match:
                pct = float(match.group(1))
                raw_symbol = match.group(2)
            else:
                # Format 3: Minimal perf report format
                match = re.match(r'\s*(\d+\.\d+)%\s+\[.\]\s+(.+)', line)
                if match:
                    pct = float(match.group(1))
                    raw_symbol = match.group(2)
                else:
                    continue # No format matched

        # The greedy regex (.+) captures trailing columns (e.g., IPC stats).
        # We clean the symbol by splitting at the first long space, which
        # robustly separates the symbol from other columns.
        symbol = raw_symbol.split('  ', 1)[0].strip()

        # Filter for relevant gnomon functions with meaningful self-time
        if 'gnomon::' in symbol and pct > 0.5 and not any(pat in symbol for pat in skip_patterns):
            candidates.append((pct, symbol))

    # Sort by percentage and return top N symbols
    candidates.sort(reverse=True, key=lambda x: x[0])
    return [sym for pct, sym in candidates[:n]]

def create_smart_annotation(raw_output, hot_threshold=0.5):
    """
    Filters annotation output to show only hot spots with context.
    Returns a list of filtered lines with better formatting.
    """
    result = []
    context_buffer = []
    last_hot_line = -1
    in_header = True

    for i, line in enumerate(raw_output):
        # Keep header lines
        if in_header:
            if line.strip() and not line.strip().startswith(':') and not re.match(r'\s*\d+\.\d+\s*:', line):
                result.append(line.rstrip())
                continue
            else:
                in_header = False

        # Source code lines (format: ": line_num  source_code")
        if match := re.match(r'\s*:\s*(\d+)\s+(.+)', line):
            # Add spacing before source lines for readability
            if result and not result[-1].strip().startswith(':'):
                result.append("")
            result.append(line.rstrip())
            context_buffer = []  # Reset context after source
            continue

        # Assembly lines with percentages
        if match := re.match(r'\s*(\d+\.\d+)\s*:\s*([0-9a-f]+):\s*(.+)', line):
            pct = float(match.group(1))

            if pct >= hot_threshold:
                # Hot line - show with context
                if last_hot_line >= 0 and i - last_hot_line > 5:
                    result.append("        [... cold instructions omitted ...]")

                # Add up to 2 lines of context before if not already shown
                start_context = max(0, len(context_buffer) - 2)
                for ctx_line in context_buffer[start_context:]:
                    if ctx_line not in result[-3:]:  # Avoid duplicates
                        result.append(ctx_line.rstrip())

                # Highlight hot line
                result.append(f">>> {line.rstrip()}")
                last_hot_line = i
                context_buffer = []
            else:
                # Cold line - just buffer it
                context_buffer.append(line)
                # Keep buffer size reasonable
                if len(context_buffer) > 5:
                    context_buffer.pop(0)
        else:
            # Other lines (empty lines, etc)
            if line.strip():
                context_buffer.append(line)

    return result

def analyze_hot_spots(annotation_lines):
    """
    Generates a summary of hot spots from annotation output.
    """
    hot_regions = []
    current_region = {'lines': [], 'total_pct': 0.0, 'source': None}

    for line in annotation_lines:
        # Source line
        if match := re.match(r'\s*:\s*(\d+)\s+(.+)', line):
            if current_region['total_pct'] > 1.0:
                hot_regions.append(current_region)
            current_region = {'lines': [match.group(1)], 'total_pct': 0.0,
                            'source': match.group(2).strip()[:50]}
        # Hot assembly line
        elif line.startswith('>>>'):
            if match := re.match(r'>>>\s*(\d+\.\d+)', line):
                current_region['total_pct'] += float(match.group(1))

    # Don't forget the last region
    if current_region['total_pct'] > 1.0:
        hot_regions.append(current_region)

    # Sort by total percentage
    hot_regions.sort(key=lambda x: x['total_pct'], reverse=True)

    summary = ["\n=== HOT SPOT SUMMARY ==="]
    for region in hot_regions[:5]:  # Top 5 hot regions
        summary.append(f"  {region['total_pct']:.1f}% in line {region['lines'][0]}: {region['source']}")

    return summary

def main():
    """
    Main function to orchestrate data generation, profiling, and reporting.
    It generates data for all workloads, then runs them sequentially under a
    single `perf record` session for a combined report.
    """
    WORKDIR.mkdir(exist_ok=True)

    commands_to_profile = []
    generators_and_paths = []

    try:
        for name, params in WORKLOADS.items():
            print_header(f"Generating data for workload: {name}", char="*")
            run_workdir = WORKDIR / name
            run_workdir.mkdir(exist_ok=True)

            generator = RealisticDataGenerator(workload_params=params, workdir=run_workdir)
            data_prefix, score_files = generator.generate_all_files()
            generators_and_paths.append((generator, run_workdir))

            print(f"    > Generating a {PROFILING_SUBSET_PCT:.0%} random subset for profiling...")
            fam_path = data_prefix.with_suffix(".fam")
            keep_file_path = run_workdir / f"keep_{name}.txt"
            fam_df = pd.read_csv(fam_path, sep='\s+', header=None, usecols=[1], names=['IID'], engine='python')
            sample_size = int(len(fam_df) * PROFILING_SUBSET_PCT)
            fam_df.sample(n=sample_size, random_state=42).to_csv(keep_file_path, header=False, index=False)
            print(f"      ... wrote {sample_size} individuals to '{keep_file_path.name}'")

            score_dir = run_workdir / f"scores_{name}"
            score_dir.mkdir(exist_ok=True)
            for sf_path in score_files:
                shutil.move(str(sf_path), score_dir)

            # The 'score' subcommand is added here as a string.
            command_str = (f'echo "--- Running workload: {name} on {PROFILING_SUBSET_PCT:.0%} subset ---" && '
                           f'"{GNOMON_BINARY}" score "{score_dir}" "{data_prefix}" --keep "{keep_file_path}"')
            commands_to_profile.append(command_str)

        runner_script_path = WORKDIR / "run_all_workloads.sh"
        with open(runner_script_path, "w") as f:
            f.write("#!/bin/bash\nset -e\n")
            f.write('\n'.join(commands_to_profile))
        os.chmod(runner_script_path, 0o755)

        perf_data_file = Path("./perf.data").resolve()
        perf_record_cmd = [
            "perf", "record", "-g", "-o", str(perf_data_file),
            "--", str(runner_script_path)
        ]

        if not run_command(perf_record_cmd, title="Running ALL workloads under `perf record`"):
            sys.exit("Profiling run failed.")

        if perf_data_file.exists():
            # First, show the normal perf report for context
            perf_report_cmd = [
                "perf", "report", "--stdio", "--call-graph=graph",
                "--percent-limit=2", "-i", str(perf_data_file)
            ]
            run_command(perf_report_cmd, title="Combined Granular Profile Report")

            # Get a flat profile report for easier parsing of hot functions
            perf_simple_report_cmd = [
                "perf", "report", "--stdio", "--no-children",
                "--sort=symbol", "--percent-limit=0.5",
                "-i", str(perf_data_file)
            ]
            success, output = run_command(perf_simple_report_cmd,
                                        title="Analyzing hot functions",
                                        capture_output=True)

            if success:
                hot_functions = find_hot_worker_functions(output, n=2)

                if hot_functions:
                    print(f"\n🔥 Found top hot functions: {', '.join(hot_functions)}")

                    for symbol in hot_functions:
                        print_header(f"Analyzing hot spots in: {symbol}", char="*")
                        perf_annotate_cmd = [
                            "perf", "annotate", "--stdio", "-l",
                            "-i", str(perf_data_file), "--symbol", symbol,
                        ]

                        proc = subprocess.Popen(
                            [str(c) for c in perf_annotate_cmd],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding='utf-8', errors='replace'
                        )
                        stdout, stderr = proc.communicate()

                        if proc.returncode == 0 and stdout:
                            annotation_lines = stdout.split('\n')
                            filtered_lines = create_smart_annotation(annotation_lines)

                            print("\n--- FILTERED ANNOTATION (showing hot spots with context) ---")
                            for line in filtered_lines:
                                print(f"  {line}")

                            summary = analyze_hot_spots(filtered_lines)
                            for line in summary:
                                print(line)
                        else:
                            print(f"⚠️  Could not annotate {symbol}")
                            if stderr:
                                print(f"   Error: {stderr.strip()}")
                            else:
                                print("   perf annotate produced no output.")
                else:
                    print("\n⚠️  No hot worker functions found in gnomon:: namespace")
        else:
            print(f"Combined perf data file not found at '{perf_data_file}'. Profiling may have failed.")
            sys.exit(1)

    finally:
        print_header("Cleaning up all generated data", char="~")
        for gen, path in generators_and_paths:
            if path.exists():
                shutil.rmtree(path)
                print(f"Removed workload directory: {path}")

        runner_script = WORKDIR / "run_all_workloads.sh"
        if runner_script.exists():
            runner_script.unlink()
            print(f"Removed temporary script: {runner_script}")

    print("\n✅ All profiling workloads completed successfully.")

if __name__ == "__main__":
    if not GNOMON_BINARY.exists():
        print(f"❌ Error: Compiled profiling binary not found at {GNOMON_BINARY}")
        print("Please ensure you have built the project with: cargo build --profile profiling")
        sys.exit(1)
    main()
