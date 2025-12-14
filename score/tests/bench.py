import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Any
import threading
import queue

import numpy as np
import pandas as pd
import psutil
import requests

# ==============================================================================
#                              BENCHMARK CONFIGURATION
# ==============================================================================

# --- Workspace and Paths ---
WORKDIR = Path("./benchmark_workdir")
GNOMON_BINARY_REL = Path("./target/release/gnomon")
PLINK2_BINARY_REL = WORKDIR / "plink2"
PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"


# --- The Crucible: Scaled & Focused Benchmark Scenarios ---
# All benchmarks run on a random subset of a larger generated cohort.
REALISTIC_DIMENSIONS = [
    {
        "test_name": "ACAF_Dense_Scores_SUBSET",
        "n_individuals": 2_000,
        "genome_variants": 800_000,
        "target_variants": 400_000,
        "af_profile": 'acaf',
        "score_files": [
            {
                "name": "acaf_dense_panel_large", "n_scores": 100, "gwas_source_variants": 350_000,
                "overlap_pct": 0.95, "flip_pct": 0.15, "missing_weight_pct": 0.0, "score_sparsity": 1.0,
            }
        ],
        "subset_pct": 0.02,
    },
    {
        "test_name": "WGS_Standard_Rare_SUBSET",
        "n_individuals": 1_000,
        "genome_variants": 1_000_000,
        "target_variants": 600_000,
        "af_profile": 'standard',
        "score_files": [
            {
                "name": "gwas_discovery_large", "n_scores": 20, "gwas_source_variants": 500_000,
                "overlap_pct": 0.80, "flip_pct": 0.10, "missing_weight_pct": 0.01, "score_sparsity": 0.1,
            }
        ],
        "subset_pct": 0.10,
    },
    {
        "test_name": "Multi_Panel_Large_SUBSET",
        "n_individuals": 3_000,
        "genome_variants": 500_000,
        "target_variants": 250_000,
        "af_profile": 'standard',
        "score_files": [
            {"name": "panel_A_lg", "n_scores": 10, "gwas_source_variants": 15_000, "overlap_pct": 0.80, "flip_pct": 0.10, "missing_weight_pct": 0.02},
            {"name": "panel_B_lg", "n_scores": 15, "gwas_source_variants": 30_000, "overlap_pct": 0.85, "flip_pct": 0.20, "missing_weight_pct": 0.05},
            {"name": "panel_C_lg", "n_scores": 5,  "gwas_source_variants": 20_000, "overlap_pct": 0.90, "flip_pct": 0.05, "missing_weight_pct": 0.0},
        ],
        "subset_pct": 0.05,
    },
    {
        "test_name": "VeryLargeN_MediumK_ACAF_SUBSET",
        "n_individuals": 10_000,
        "genome_variants": 750_000,
        "target_variants": 350_000,
        "af_profile": 'acaf',
        "score_files": [
            {
                "name": "v_largeN_panel", "n_scores": 30, "gwas_source_variants": 100_000,
                "overlap_pct": 0.90, "flip_pct": 0.15, "missing_weight_pct": 0.01, "score_sparsity": 0.7,
            }
        ],
        "subset_pct": 0.03,
    },
    {
        "test_name": "Test_HighAF_6pct_SUBSET",
        "n_individuals": 2_000,
        "genome_variants": 800_000,
        "target_variants": 400_000,
        "af_profile": 'high_af',
        "score_files": [
            {
                "name": "high_af_panel", "n_scores": 50, "gwas_source_variants": 200_000,
                "overlap_pct": 0.90, "flip_pct": 0.10, "missing_weight_pct": 0.01, "score_sparsity": 0.8,
            }
        ],
        "subset_pct": 0.05,
    },
    {
        "test_name": "Test_LowAF_2pct_SUBSET",
        "n_individuals": 2_000,
        "genome_variants": 800_000,
        "target_variants": 400_000,
        "af_profile": 'low_af',
        "score_files": [
            {
                "name": "low_af_panel", "n_scores": 50, "gwas_source_variants": 200_000,
                "overlap_pct": 0.90, "flip_pct": 0.10, "missing_weight_pct": 0.01, "score_sparsity": 0.8,
            }
        ],
        "subset_pct": 0.05,
    },
    {
        "test_name": "HugeN_50kVars_75pct_SUBSET",
        "n_individuals": 300_000,
        "genome_variants": 100_000,
        "target_variants": 50_000,
        "af_profile": 'standard',
        "score_files": [
            {
                "name": "hugeN_smallK_panel", "n_scores": 90, "gwas_source_variants": 25_000,
                "overlap_pct": 0.90, "flip_pct": 0.10, "missing_weight_pct": 0.01, "score_sparsity": 0.5,
            }
        ],
        "subset_pct": 0.75,
    },
]



# --- Data Realism & Variety Parameters ---
EFFECT_DISTRIBUTIONS = [('normal', 0, 0.001), ('laplace', 0, 0.05)]

# ==============================================================================
#                               HELPER FUNCTIONS
# ==============================================================================

def print_header(title: str, char: str = "="):
    """Prints a formatted header to the console."""
    width = 80
    print("\n" + char * width, flush=True)
    print(f"{char*4} {title} {char*(width - len(title) - 6)}", flush=True)
    print(char * width, flush=True)

def setup_environment(gnomon_path, plink_path):
    print_header("Benchmark Setup", "-")
    WORKDIR.mkdir(exist_ok=True)
    if not gnomon_path.exists():
        print(f"❌ ERROR: Gnomon binary not found at '{gnomon_path}'. Please build with 'cargo build --release'.")
        return False
    if not plink_path.exists():
        print(f"  > PLINK2 not found. Downloading from {PLINK2_URL}...")
        zip_path = WORKDIR / "plink.zip"
        try:
            with requests.get(PLINK2_URL, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f: shutil.copyfileobj(r.raw, f)
            with zipfile.ZipFile(zip_path, 'r') as z:
                for member in z.infolist():
                    if member.filename.endswith('plink2') and not member.is_dir():
                        with z.open(member) as source, open(plink_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        break
            plink_path.chmod(0o755)
            zip_path.unlink()
            print(f"  > Successfully downloaded and configured PLINK2 at '{plink_path}'.")
        except Exception as e:
            print(f"❌ ERROR: Failed to download or extract PLINK2: {e}")
            return False
    else:
        print(f"  > Found existing PLINK2 binary at '{plink_path}'.")
    return True

# ==============================================================================
#                         REALISTIC DATA GENERATOR
# ==============================================================================

class RealisticDataGenerator:
    """Generates messy, realistic, decoupled genomic data for benchmarking."""
    def __init__(self, workload_params: Dict[str, Any], workdir: Path):
        self.params = workload_params
        self.workdir = workdir
        self.run_prefix = workdir / f"data_{workload_params['test_name']}"
        self.target_variants_df: pd.DataFrame = None
        self.gwas_universe_df: pd.DataFrame = None

    def _generate_variant_universes(self):
        print("    > Generating variant universes...")
        n_genome = self.params['genome_variants']
        positions = np.sort(np.random.choice(np.arange(1, n_genome * 10), n_genome, replace=False))
        base_alleles = np.array(['A', 'C', 'G', 'T'])
        ref_indices = np.random.randint(0, 4, n_genome)
        alt_indices = (ref_indices + np.random.randint(1, 4, n_genome)) % 4
        genome_df = pd.DataFrame({
            'chr': 1, 'pos': positions, 'id': [f"1:{p}" for p in positions],
            'a1': base_alleles[ref_indices], 'a2': base_alleles[alt_indices]
        }).set_index('id', drop=False)

        self.target_variants_df = genome_df.sample(n=self.params['target_variants'], random_state=42).sort_values('pos').reset_index(drop=True)
        self.gwas_universe_df = genome_df.copy()

    def _generate_bim_and_fam(self):
        print(f"    > Generating .bim for {len(self.target_variants_df)} variants and .fam for {self.params['n_individuals']} individuals...")
        bim_df = self.target_variants_df.copy()
        bim_df['cm'] = 0
        bim_df[['chr', 'id', 'cm', 'pos', 'a1', 'a2']].to_csv(self.run_prefix.with_suffix(".bim"), sep='\t', header=False, index=False)
        with open(self.run_prefix.with_suffix(".fam"), 'w') as f:
            for i in range(self.params['n_individuals']):
                f.write(f"0\tsample_{i}\t0\t0\t0\t-9\n")

    def _generate_bed(self):
        print(f"    > Generating .bed file (memory-efficient, row-by-row)...")
        n_variants, n_individuals = len(self.target_variants_df), self.params['n_individuals']

        af_profile = self.params.get('af_profile', 'standard')
        if af_profile == 'acaf':
            print("        (Using ACAF allele frequency profile: common variants only, MAF > 1%)")
            af = np.random.uniform(0.01, 0.5, n_variants)
        elif af_profile == 'high_af':
            print("        (Using high allele frequency profile: MAF >= 6%)")
            af = np.random.uniform(0.06, 0.5, n_variants)
        elif af_profile == 'low_af':
            print("        (Using low allele frequency profile: MAF <= 2%)")
            af = np.random.uniform(0.001, 0.02, n_variants)
        else: # 'standard'
            print("        (Using standard allele frequency profile: includes rare variants)")
            af = np.random.beta(0.2, 0.2, n_variants)
        
        code_map = {0: 0b00, 1: 0b10, 2: 0b11, -1: 0b01}
        mapping_array = np.array([code_map[key] for key in sorted(code_map)], dtype=np.uint8)
        
        with open(self.run_prefix.with_suffix(".bed"), 'wb') as f:
            f.write(bytes([0x6c, 0x1b, 0x01]))
            padded_n_individuals = (n_individuals + 3) // 4 * 4
            bytes_per_row = padded_n_individuals // 4

            for i in range(n_variants):
                if (i + 1) % 50000 == 0:
                    print(f"      ... wrote {i+1}/{n_variants} variants to .bed", flush=True)
                p = af[i]
                hwe_probs = np.array([(1-p)**2, 2*p*(1-p), p**2])
                rand_draws = np.random.rand(n_individuals)
                genotypes = (rand_draws > hwe_probs[0]) + (rand_draws > hwe_probs[0] + hwe_probs[1])
                genotypes[(np.random.rand(n_individuals) < 0.01)] = -1
                
                codes = mapping_array[genotypes + 1]
                padded_codes = np.full(padded_n_individuals, code_map[-1], dtype=np.uint8)
                padded_codes[:n_individuals] = codes
                
                packed_bytes = np.zeros(bytes_per_row, dtype=np.uint8)
                packed_bytes = (padded_codes[0::4]) | (padded_codes[1::4] << 2) | \
                               (padded_codes[2::4] << 4) | (padded_codes[3::4] << 6)
                f.write(packed_bytes.tobytes())

    def _generate_score_files(self) -> List[Path]:
        print("    > Generating score files with realistic overlap and ambiguity...")
        generated_paths = []
        target_variant_ids = set(self.target_variants_df['id'])
        
        for sf_config in self.params['score_files']:
            n_overlap = int(sf_config['gwas_source_variants'] * sf_config['overlap_pct'])
            n_non_overlap = sf_config['gwas_source_variants'] - n_overlap
            
            if n_overlap > len(target_variant_ids):
                raise ValueError(f"Cannot sample {n_overlap} overlapping variants from a pool of {len(target_variant_ids)}")
            
            overlapping_ids = np.random.choice(list(target_variant_ids), n_overlap, replace=False)
            
            gwas_non_target_ids = self.gwas_universe_df[~self.gwas_universe_df.index.isin(target_variant_ids)].index
            if n_non_overlap > len(gwas_non_target_ids):
                 raise ValueError(f"Cannot sample {n_non_overlap} non-overlapping variants from a pool of {len(gwas_non_target_ids)}")
            
            non_overlapping_ids = np.random.choice(gwas_non_target_ids, n_non_overlap, replace=False)
            
            score_variant_ids = np.concatenate([overlapping_ids, non_overlapping_ids])
            score_df_source = self.gwas_universe_df.loc[score_variant_ids].copy()
            score_df_source = score_df_source.sort_values(['chr', 'pos']).reset_index(drop=True)
            
            score_df_source['effect_allele'] = score_df_source['a2']
            score_df_source['other_allele'] = score_df_source['a1']
            
            overlap_mask = score_df_source['id'].isin(overlapping_ids)
            flip_indices = score_df_source[overlap_mask].sample(frac=sf_config['flip_pct'], random_state=42).index
            
            orig_eff = score_df_source.loc[flip_indices, 'effect_allele'].copy()
            score_df_source.loc[flip_indices, 'effect_allele'] = score_df_source.loc[flip_indices, 'other_allele']
            score_df_source.loc[flip_indices, 'other_allele'] = orig_eff
            
            final_score_df = score_df_source[['id', 'effect_allele', 'other_allele']].rename(columns={'id': 'variant_id'})
            n_variants_in_score = len(final_score_df)
            for i in range(sf_config['n_scores']):
                dist_name, p1, p2 = random.choice(EFFECT_DISTRIBUTIONS)
                sparsity = sf_config.get('score_sparsity', random.uniform(0.01, 0.8))
                
                weights = np.full(n_variants_in_score, np.nan)
                n_weighted = int(sparsity * n_variants_in_score)
                indices = np.random.choice(n_variants_in_score, n_weighted, replace=False)
                
                if n_weighted > 0:
                    eff_weights = np.random.normal(p1, p2, n_weighted) if dist_name == 'normal' else np.random.laplace(p1, p2, n_weighted)
                    weights[indices] = eff_weights
                
                if sf_config.get('missing_weight_pct', 0.0) > 0 and n_weighted > 0:
                    n_missing = int(sf_config['missing_weight_pct'] * n_weighted)
                    missing_indices = np.random.choice(indices, n_missing, replace=False)
                    weights[missing_indices] = np.nan

                final_score_df[f"score_{sf_config['name']}_{i+1}"] = weights

            score_file_path = self.run_prefix.with_suffix(f".{sf_config['name']}.score")

            score_columns = [col for col in final_score_df.columns if col.startswith('score_')]
            if score_columns:
                final_score_df.dropna(subset=score_columns, how='all', inplace=True)

            final_score_df.to_csv(score_file_path, sep='\t', index=False, float_format='%.8f', na_rep='')
            generated_paths.append(score_file_path)
            
        return generated_paths

    def generate_all_files(self) -> (Path, List[Path]):
        self._generate_variant_universes()
        self._generate_bim_and_fam()
        self._generate_bed()
        score_file_paths = self._generate_score_files()
        return self.run_prefix, score_file_paths

    def cleanup(self):
        print("    > Cleaning up generated data files...")
        extensions = [".bed", ".bim", ".fam", ".log"] + [f".{sf['name']}.score" for sf in self.params['score_files']]
        for ext in extensions:
            try: self.run_prefix.with_suffix(ext).unlink(missing_ok=True)
            except IsADirectoryError: pass
        
        for f in self.workdir.glob("plink2_run*"):
            f.unlink(missing_ok=True)

        multi_score_dir = self.workdir / f"scores_{self.params['test_name']}"
        if multi_score_dir.is_dir():
            shutil.rmtree(multi_score_dir)
        
        plink_score_dir = self.workdir / f"plink_scores_{self.params['test_name']}"
        if plink_score_dir.is_dir():
            shutil.rmtree(plink_score_dir)

        keep_file = self.workdir / f"keep_{self.params['test_name']}.txt"
        if keep_file.exists():
            keep_file.unlink()

# ==============================================================================
#                       EXECUTION & MONITORING ENGINE
# ==============================================================================

def run_and_monitor_process(tool_name: str, command: List[str], cwd: Path) -> Dict[str, Any]:
    """
    Executes a command, monitors its resource usage, and streams its output in
    real-time, correctly formatted within a collapsible log section.
    """
    print(f"  > Running {tool_name}...")
    cmd_str_list = [str(c) for c in command]
    print(f"    Command: {' '.join(cmd_str_list)}")
    start_time = time.monotonic()
    
    # This queue will hold lines from stdout and stderr to be printed by the main thread.
    output_queue = queue.Queue()

    def stream_reader(stream, stream_name):
        """Reads lines from a stream and puts them onto the queue."""
        try:
            for line in iter(stream.readline, ''):
                output_queue.put((stream_name, line))
        finally:
            stream.close()

    process = None
    try:
        process = subprocess.Popen(
            cmd_str_list,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        p = psutil.Process(process.pid)
        
        # Start threads to read stdout and stderr concurrently
        stdout_thread = threading.Thread(target=stream_reader, args=(process.stdout, 'STDOUT'))
        stderr_thread = threading.Thread(target=stream_reader, args=(process.stderr, 'STDERR'))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        peak_rss_mb = 0
        
        # Print the header for the output section immediately
        print(f"\n    --- START {tool_name.upper()} OUTPUT ---")
        
        # Main loop: Monitor process and print any available output
        current_stream = None
        while process.poll() is None:
            # Monitor memory
            try:
                peak_rss_mb = max(peak_rss_mb, p.memory_info().rss / 1024 / 1024)
            except psutil.NoSuchProcess:
                break

            # Print queued output lines
            try:
                stream_name, line = output_queue.get_nowait()
                if stream_name != current_stream:
                    print(f"    [{stream_name}]", flush=True)
                    current_stream = stream_name
                print(f"      | {line.strip()}", flush=True)
            except queue.Empty:
                # No new output, sleep briefly to yield CPU
                time.sleep(0.02)
        
        # Wait for reader threads to finish
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)

        # Drain any remaining lines from the queue
        while not output_queue.empty():
            try:
                stream_name, line = output_queue.get_nowait()
                if stream_name != current_stream:
                    print(f"    [{stream_name}]", flush=True)
                    current_stream = stream_name
                print(f"      | {line.strip()}", flush=True)
            except queue.Empty:
                break
        
        print(f"    --- END {tool_name.upper()} OUTPUT ---\n")

        wall_time = time.monotonic() - start_time
        returncode = process.returncode
        
        if returncode != 0:
            print(f"  > ❌ {tool_name} FAILED with exit code {returncode}.")
        else:
            print(f"  > ✅ {tool_name} finished in {wall_time:.2f}s. Peak Memory: {peak_rss_mb:.2f} MB")

    except (FileNotFoundError, psutil.NoSuchProcess) as e:
        print(f"  > ❌ ERROR launching {tool_name}: {e}")
        wall_time, peak_rss_mb, returncode = -1, -1, -1
    finally:
        if process and process.poll() is None:
            # Ensure process is terminated if something goes wrong
            process.kill()

    return {"tool": tool_name, "time_sec": wall_time, "peak_mem_mb": peak_rss_mb, "success": returncode == 0}

# ==============================================================================
#                             VALIDATION & REPORTING
# ==============================================================================

def simplified_validation(gnomon_success: bool, plink2_success: bool):
    """A simplified validation check focusing on successful execution."""
    if gnomon_success and plink2_success:
        print("  > ✅ Simplified Validation: Both tools completed successfully.")
        return True
    else:
        print("  > ❌ VALIDATION FAILED: One or both tools failed to complete.")
        return False
    
def report_results(results: list):
    print_header("Benchmark Summary Report")
    if not results: print("No results to report."); return
    df = pd.DataFrame(results)
    
    index_cols = ['test_name']
    for col in index_cols:
        if col not in df.columns:
            df[col] = 'N/A'
            
    try:
        success_df = df[df['success'] == True].copy()
        if not success_df.empty:
            pivoted = success_df.pivot_table(
                index=index_cols, columns='tool', values=['time_sec', 'peak_mem_mb'], aggfunc='first'
            )
            pivoted.columns = [f"{val}_{tool}" for val, tool in pivoted.columns]
            
            all_test_names = [d['test_name'] for d in REALISTIC_DIMENSIONS]
            report_df = pivoted.reindex(all_test_names)

            print("---\n- Ratios (< 1.0) indicate gnomon is faster or uses less memory.\n- Time is wall-clock seconds. Memory is peak RSS in Megabytes.\n---")
            if 'time_sec_gnomon' in report_df and 'time_sec_plink2' in report_df: report_df['time_ratio_g/p'] = report_df['time_sec_gnomon'] / report_df['time_sec_plink2']
            if 'peak_mem_mb_gnomon' in report_df and 'peak_mem_mb_plink2' in report_df: report_df['mem_ratio_g/p'] = report_df['peak_mem_mb_gnomon'] / report_df['peak_mem_mb_plink2']
            print(report_df.to_markdown(floatfmt=".3f"))
        else:
            print("No successful runs to generate a comparison report.")

    except Exception as e:
        print(f"Could not generate pivot table. Printing raw results.\n{e}")
        print(df.to_markdown(index=False, floatfmt=".3f"))
    
    summary_path = WORKDIR / "benchmark_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nFull results saved to '{summary_path}'")

def main():
    np.random.seed(42); random.seed(42)
    gnomon_abs_path, plink2_abs_path = GNOMON_BINARY_REL.resolve(), PLINK2_BINARY_REL.resolve()
    if not setup_environment(gnomon_abs_path, plink2_abs_path): sys.exit(1)

    all_results, failed_runs = [], 0
    for run_id, workload_params in enumerate(REALISTIC_DIMENSIONS, 1):
        params = workload_params.copy()
        test_name = params['test_name']
        params['test_name'] = test_name.replace('_FULL', '').replace('_SUBSET', '')
        print_header(f"Benchmark Run {run_id}/{len(REALISTIC_DIMENSIONS)}: {test_name}")
        
        generator = None
        keep_file_path = None
        plink_score_dir = None
        
        try:
            generator = RealisticDataGenerator(workload_params=params, workdir=WORKDIR)
            data_prefix, score_files = generator.generate_all_files()

            subset_pct = params.get("subset_pct")
            if subset_pct and subset_pct < 1.0:
                print(f"    > Generating a {subset_pct:.0%} random subset of individuals...")
                fam_path = data_prefix.with_suffix(".fam")
                keep_file_path = WORKDIR / f"keep_{params['test_name']}.txt"
                
                fam_df = pd.read_csv(fam_path, sep=r'\s+', header=None, usecols=[1], names=['IID'], engine='python')
                sample_size = int(len(fam_df) * subset_pct)
                fam_df.sample(n=sample_size, random_state=run_id).to_csv(keep_file_path, header=False, index=False)
                print(f"      ... wrote {sample_size} individuals to '{keep_file_path.name}'")
            else:
                print("    > Running on full cohort (no subsetting).")

            gnomon_score_dir = WORKDIR / f"scores_{params['test_name']}"
            gnomon_score_dir.mkdir(exist_ok=True)
            
            moved_gnomon_score_files = []
            for sf_path in score_files:
                new_path = gnomon_score_dir / sf_path.name
                shutil.move(str(sf_path), str(new_path))
                moved_gnomon_score_files.append(new_path)
            
            # The 'score' subcommand is added here.
            gnomon_cmd = [
                gnomon_abs_path,
                "score",
                gnomon_score_dir.name,
                data_prefix.name,
            ]
            if keep_file_path:
                gnomon_cmd.extend(["--keep", keep_file_path.name])
            gnomon_res = run_and_monitor_process("gnomon", gnomon_cmd, WORKDIR)
            gnomon_res['test_name'] = test_name
            all_results.append(gnomon_res)
            
            if gnomon_res["success"]:
                print("    > Preparing PLINK2-compatible score files (filling NA with 0)...")
                plink_score_dir = WORKDIR / f"plink_scores_{params['test_name']}"
                plink_score_dir.mkdir(exist_ok=True)
                plink_score_files = []
                for gnomon_sf_path in moved_gnomon_score_files:
                    df = pd.read_csv(gnomon_sf_path, sep='\t', low_memory=False)
                    score_columns = [col for col in df.columns if col.startswith('score_')]
                    if score_columns:
                        df[score_columns] = df[score_columns].fillna(0)
                    
                    plink_sf_path = plink_score_dir / gnomon_sf_path.name
                    df.to_csv(plink_sf_path, sep='\t', index=False, float_format='%.8f')
                    plink_score_files.append(plink_sf_path)

                plink_out_prefix = WORKDIR / f"plink2_run{run_id}"
                
                if len(plink_score_files) > 1:
                    print("    > Merging multiple score files for PLINK2...")
                    merged_df = None
                    id_cols = ['variant_id', 'effect_allele', 'other_allele']
                    for sf_path in plink_score_files:
                        current_df = pd.read_csv(sf_path, sep='\t', low_memory=False)
                        if merged_df is None:
                            merged_df = current_df
                        else:
                            merged_df = pd.merge(merged_df, current_df, on=id_cols, how='outer')
                    
                    score_columns = [col for col in merged_df.columns if col.startswith('score_')]
                    if score_columns:
                        merged_df[score_columns] = merged_df[score_columns].fillna(0)
                    
                    active_plink_score_file = WORKDIR / f"plink_merged_{params['test_name']}.score"
                    merged_df.to_csv(active_plink_score_file, sep='\t', index=False, float_format='%.8f')
                else:
                    active_plink_score_file = plink_score_files[0]
                    
                plink2_cmd = [plink2_abs_path, "--bfile", data_prefix.name, "--out", plink_out_prefix.name, "--threads", str(os.cpu_count() or 1)]
                if keep_file_path:
                    plink2_cmd.extend(["--keep", keep_file_path.name])
                
                with open(active_plink_score_file, 'r') as f:
                    header = f.readline().strip()
                    n_file_scores = len(header.split('\t')) - 3
                
                score_col_range = "4" if n_file_scores == 1 else f"4-{3 + n_file_scores}"
                score_file_arg = active_plink_score_file.relative_to(WORKDIR)
                
                plink2_cmd.extend(["--score", str(score_file_arg), "1", "2", "header", "no-mean-imputation", "--score-col-nums", score_col_range])

                plink2_res = run_and_monitor_process("plink2", plink2_cmd, WORKDIR)
                plink2_res['test_name'] = test_name
                all_results.append(plink2_res)
                
                if not simplified_validation(gnomon_res["success"], plink2_res["success"]):
                    failed_runs += 1
            else:
                print("  > Skipping PLINK2 run because Gnomon failed.")
                failed_runs += 1

        except Exception as e:
            print(f"  > ❌ Test run FAILED for {test_name}: {e}")
            failed_runs += 1
        
        finally:
            if generator:
                generator.cleanup()
            if plink_score_dir and plink_score_dir.exists():
                shutil.rmtree(plink_score_dir)
            if keep_file_path and keep_file_path.exists():
                keep_file_path.unlink()

    report_results(all_results)
    if failed_runs > 0:
        print(f"\n❌ Benchmark finished with {failed_runs} failed or invalid run(s)."); sys.exit(1)
    else:
        print("\n🎉 All benchmarks completed successfully."); sys.exit(0)

if __name__ == "__main__":
    main()
