import subprocess
import requests
import zipfile
import gzip
import shutil
import time
import sys
import os
import re
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np

# ========================================================================================
#                                   CONFIGURATION
# ========================================================================================
CI_WORKDIR = Path("./ci_workdir")
GNOMON_BINARY = Path("./target/release/gnomon")
PLINK2_BINARY = CI_WORKDIR / "plink2"
PYLINK_SCRIPT = (Path(__file__).resolve().parent / "pylink.py").resolve()

# Constants for the Disagreement Bisection Algorithm
DISCOVERY_PGS_ID = "PGS004696"
CORRELATION_THRESHOLD = 0.99
BISECTION_TERMINATION_SIZE = 3

SHARED_GENOTYPE_PREFIX = CI_WORKDIR / "chr22_subset50_compat"
ORIGINAL_PLINK_PREFIX = CI_WORKDIR / "chr22_subset50"

PLINK1_URL = "https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20231211.zip"
PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"

GENOTYPE_URL_BASE = "https://github.com/SauersML/genomic_pca/blob/main/data/"
GENOTYPE_FILES = [
    "chr22_subset50.bed.zip", "chr22_subset50.bim.zip", "chr22_subset50.fam.zip"
]

PGS_SCORES = {
    "PGS004696": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS004696/ScoringFiles/Harmonized/PGS004696_hmPOS_GRCh38.txt.gz",
    "PGS003725": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS003725/ScoringFiles/Harmonized/PGS003725_hmPOS_GRCh38.txt.gz",
    "PGS001780": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS001780/ScoringFiles/Harmonized/PGS001780_hmPOS_GRCh38.txt.gz",
}

# ========================================================================================
#                                   HELPER FUNCTIONS
# ========================================================================================
def print_header(title: str, char: str = "="):
    """Prints a formatted header to the console."""
    width = 80
    print("\n" + char * width, flush=True)
    print(f"{char*4} {title} {char*(width - len(title) - 6)}", flush=True)
    print(char * width, flush=True)

def print_debug_header(title: str):
    """Prints a smaller header for debug sections."""
    print("\n" + "." * 80, flush=True)
    print(f".... {title} ....", flush=True)
    print("." * 80, flush=True)

def download_and_extract(url: str, dest_dir: Path):
    """Downloads and extracts a file, handling .zip and .gz."""
    filename = Path(url.split('?')[0].split('/')[-1])
    outpath = dest_dir / filename
    print(f"[SCRIPT] Downloading {filename}...", flush=True)
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(outpath, 'wb') as f: shutil.copyfileobj(r.raw, f)
    except Exception as e:
        print(f"❌ FAILED to download {url}: {e}", flush=True); sys.exit(1)

    if outpath.suffix == '.zip':
        with zipfile.ZipFile(outpath, 'r') as z: z.extractall(dest_dir)
        print(f"[SCRIPT] Extracted ZIP: {filename}", flush=True); outpath.unlink()
    elif outpath.suffix == '.gz':
        dest = dest_dir / filename.stem
        with gzip.open(outpath, 'rb') as f_in, open(dest, 'wb') as f_out: shutil.copyfileobj(f_in, f_out)
        print(f"[SCRIPT] Extracted GZ: {filename} -> {dest.name}", flush=True); outpath.unlink()

def run_and_measure(cmd: list, name: str, expected_out_prefix: Path):
    """Executes a command, streams its output, and captures results."""
    print_header(f"RUNNING: {name}", char='-')
    print(f"Command:\n  {' '.join(map(str, cmd))}\n", flush=True)
    start = time.perf_counter()
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
    
    output_lines = []
    print_debug_header(f"{name} Real-time Output")
    with proc.stdout:
        for line in iter(proc.stdout.readline, ''):
            print(f"  [{name}] {line.strip()}", flush=True)
            output_lines.append(line.strip())
    
    returncode = proc.wait()
    duration = time.perf_counter() - start
        
    result = {'tool': name, 'time_sec': duration, 'returncode': returncode, 'success': returncode == 0, 'stdout': "\n".join(output_lines)}
    if result['success']: print(f"✅ {name} finished in {duration:.2f}s.", flush=True)
    else: print(f"❌ {name} FAILED (Exit Code: {returncode})", flush=True)
    return result

def setup_environment():
    """Prepares the CI environment by downloading all necessary data and tools."""
    print_header("ENVIRONMENT SETUP")
    CI_WORKDIR.mkdir(exist_ok=True)
    if not PYLINK_SCRIPT.exists():
        print(f"❌ ERROR: PyLink script not found at: {PYLINK_SCRIPT}", flush=True); sys.exit(1)
    print(f"[SCRIPT] Found PyLink script at: {PYLINK_SCRIPT}", flush=True)

    # This loop correctly pairs each URL string with its destination binary path.
    for url_str, binary_path in [(PLINK1_URL, CI_WORKDIR / "plink"), (PLINK2_URL, PLINK2_BINARY)]:
        if not binary_path.exists():
            download_and_extract(url_str, CI_WORKDIR)
            for p in CI_WORKDIR.iterdir():
                if p.is_file() and p.name.startswith(binary_path.name.split('_')[0]):
                    p.rename(binary_path)
                    binary_path.chmod(0o755)
                    print(f"[SCRIPT] Configured and chmod: {binary_path}", flush=True)
                    break

    for zip_name in GENOTYPE_FILES: download_and_extract(f"{GENOTYPE_URL_BASE}{zip_name}?raw=true", CI_WORKDIR)
    for url_str in PGS_SCORES.values(): download_and_extract(url_str, CI_WORKDIR)

def create_synchronized_genotype_files(original_prefix: Path, final_prefix: Path):
    """Creates a new PLINK fileset with harmonized 'chr:pos' variant IDs."""
    print_header("SYNCHRONIZING GENOTYPE DATA", char='.')
    bim_df = pd.read_csv(original_prefix.with_suffix('.bim'), sep=r'\s+', header=None, names=['chrom', 'rsid', 'cm', 'pos', 'a1', 'a2'], dtype=str)
    bim_df['chr_pos_id'] = bim_df['chrom'] + ':' + bim_df['pos']
    
    duplicate_pos = bim_df[bim_df.duplicated(subset=['chr_pos_id'], keep=False)]
    variants_to_exclude = set(duplicate_pos['rsid'])
    variants_to_rename = bim_df[~bim_df['rsid'].isin(variants_to_exclude)]
    
    exclude_file = CI_WORKDIR / "variants_to_exclude.txt"
    pd.Series(list(variants_to_exclude)).to_csv(exclude_file, index=False, header=False)
    
    id_map_file = CI_WORKDIR / "id_update_map.txt"
    variants_to_rename[['rsid', 'chr_pos_id']].to_csv(id_map_file, sep=' ', index=False, header=False)
    
    cmd = [str(CI_WORKDIR / "plink"), "--bfile", str(original_prefix), "--exclude", str(exclude_file), "--update-name", str(id_map_file), "2", "1", "--make-bed", "--out", str(final_prefix)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("❌ FAILED to create synchronized genotype fileset.", flush=True); print(proc.stdout); print(proc.stderr); sys.exit(1)
    print(f"[SCRIPT] Successfully created synchronized fileset at: {final_prefix}", flush=True)

def create_unified_scorefile(score_file: Path) -> pd.DataFrame:
    """Creates a unified, sorted scorefile dataframe where the variant ID is 'chr:pos'."""
    pgs_id = score_file.name.split('_')[0]
    print_header(f"CREATING UNIFIED SCORE FILE FOR {pgs_id}", char='.')
    df = pd.read_csv(score_file, sep='\t', comment='#', usecols=['hm_chr', 'hm_pos', 'effect_allele', 'other_allele', 'effect_weight'], dtype=str, low_memory=False)
    df.dropna(inplace=True)
    df['effect_weight'] = pd.to_numeric(df['effect_weight'], errors='coerce')
    df.dropna(inplace=True)

    # Create numeric columns for sorting that correctly handle chromosome names.
    # This logic is designed to match the sorting behavior of gnomon's Rust backend,
    # which is a precondition for its streaming merge-join algorithm.
    df['sort_chr'] = df['hm_chr'].str.upper().replace({'X': '23', 'Y': '24', 'MT': '25'})
    # Strip any 'CHR' prefix before attempting numeric conversion.
    df['sort_chr'] = pd.to_numeric(df['sort_chr'].str.replace('CHR', ''), errors='coerce')
    df['sort_pos'] = pd.to_numeric(df['hm_pos'], errors='coerce')

    # Drop rows where sorting keys could not be created (e.g., from unparsable chromosome names).
    df.dropna(subset=['sort_chr', 'sort_pos'], inplace=True)

    # Cast the sorting keys to integer types for a clean, unambiguous sort.
    df['sort_chr'] = df['sort_chr'].astype(int)
    df['sort_pos'] = df['sort_pos'].astype(int)
    
    # Sort the dataframe by chromosome and then by position.
    df.sort_values(by=['sort_chr', 'sort_pos'], inplace=True)

    df['chr_pos_id'] = df['hm_chr'] + ':' + df['hm_pos']
    # After sorting, we can safely drop duplicates.
    df.drop_duplicates(subset=['chr_pos_id'], keep='first', inplace=True)
    
    # Select and rename the final columns for the output file.
    final_df = df[['chr_pos_id', 'effect_allele', 'other_allele', 'effect_weight']].copy()
    final_df.columns = ['variant_id', 'effect_allele', 'other_allele', 'effect_weight']
    
    out_path = CI_WORKDIR / f"{pgs_id}_unified_format.tsv"
    final_df.to_csv(out_path, sep='\t', index=False)
    print(f"[SCRIPT] Writing {len(final_df)} variants to unified score file: {out_path.name}", flush=True)
    return final_df

def _extract_variants_processed(log_text: str, tool_name: str) -> int:
    """Parses tool logs to find the number of variants used in scoring."""
    tool_name_key = tool_name.split('_')[0]
    if 'gnomon' in tool_name_key:
        # Gnomon log example: > Preparation complete in ... Found 64 individuals to score and 15880 overlapping variants
        match = re.search(r'Found \d+ individuals to score and (\d+) overlapping variants', log_text)
    elif 'plink2' in tool_name_key:
        # PLINK2 log example: --score: 15880 variants processed.
        match = re.search(r'--score: (\d+) variants processed', log_text)
    elif 'pylink' in tool_name_key:
        # PyLink log example: > Processed 15880 variants found in score file.
        match = re.search(r'> Processed (\d+) variants found in score file', log_text)
    else:
        return 0
    return int(match.group(1)) if match else 0

def analyze_pgs_results(pgs_id: str, pgs_run_results: list) -> dict:
    """
    Analyzes and compares results from multiple tool runs for a single PGS.
    This includes calculating score correlations and summarizing missingness.
    """
    summary = {'pgs_id': pgs_id, 'success': False}
    all_dfs = []

    for res in pgs_run_results:
        if not res['success']:
            continue
        
        tool_name = res['tool'].split('_')[0]
        
        # Determine the correct output file path based on the tool
        if tool_name == 'gnomon':
            path = CI_WORKDIR / f"gnomon_{pgs_id}.sscore"
        elif tool_name == 'plink2':
            path = CI_WORKDIR / f"plink2_{pgs_id}.sscore"
        elif tool_name == 'pylink':
            path = CI_WORKDIR / f"pylink_{pgs_id}.sscore"
        else:
            continue

        if not path.exists():
            continue

        try:
            # Read the raw output file, handling variable whitespace
            df_raw = pd.read_csv(path, sep=r'\s+')
            
            # Standardize the individual ID column name
            id_col = '#IID' if '#IID' in df_raw.columns else 'IID'
            df_proc = df_raw[[id_col]].rename(columns={id_col: 'IID'})

            # --- Process Score Column ---
            score_col_name = next((c for c in df_raw.columns if '_AVG' in c), None)
            if not score_col_name:
                raise KeyError(f"{tool_name.upper()} _AVG score column not found")
            
            df_proc[f'SCORE_{tool_name}'] = pd.to_numeric(df_raw[score_col_name], errors='coerce')

            # --- Process Missingness Information ---
            if tool_name == 'gnomon':
                # Gnomon provides a direct missingness percentage column
                missing_col_name = next((c for c in df_raw.columns if '_MISSING_PCT' in c), None)
                if not missing_col_name:
                    raise KeyError(f"Gnomon _MISSING_PCT column not found")
                df_proc[f'MISSING_PCT_{tool_name}'] = pd.to_numeric(df_raw[missing_col_name], errors='coerce')
            else:
                # For plink2/pylink, calculate missingness based on the available allele counts
                total_variants = _extract_variants_processed(res['stdout'], res['tool'])
                if total_variants > 0:
                    if 'ALLELE_CT' in df_raw.columns:
                        # Missingness % = (1 - (variants_observed / variants_total)) * 100
                        # variants_observed = ALLELE_CT / 2 (since ALLELE_CT is count of alleles, not loci)
                        observed_variants = pd.to_numeric(df_raw['ALLELE_CT'], errors='coerce') / 2
                        df_proc[f'MISSING_PCT_{tool_name}'] = (1 - (observed_variants / total_variants)) * 100.0
                    elif 'NMISS_ALLELE_CT' in df_raw.columns:
                        # PyLink reports the number of non-missing alleles.
                        # Convert to observed variants (divide by two) before normalising.
                        observed_variants = pd.to_numeric(df_raw['NMISS_ALLELE_CT'], errors='coerce') / 2
                        df_proc[f'MISSING_PCT_{tool_name}'] = (1 - (observed_variants / total_variants)) * 100.0
                    else:
                        df_proc[f'MISSING_PCT_{tool_name}'] = np.nan
                else:
                    # If we cannot determine how many variants were processed, mark as unavailable.
                    df_proc[f'MISSING_PCT_{tool_name}'] = np.nan
            
            all_dfs.append(df_proc)

        except Exception as e:
            print(f"  > [ANALYSIS_ERROR] Failed to parse {tool_name} for {pgs_id}: {e}", flush=True)

    # Return early if not enough data could be processed for a comparison
    if len(all_dfs) < 2:
        return summary
    
    # Merge all processed dataframes into a single report
    from functools import reduce
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='IID', how='inner'), all_dfs)
    
    # Clean up and prepare for summary
    score_cols = sorted([c for c in merged_df.columns if c.startswith('SCORE_')])
    merged_df.dropna(subset=score_cols, inplace=True)
    
    if len(score_cols) < 2 or merged_df.empty:
        return summary
    
    # Populate the summary dictionary
    summary['report_df'] = merged_df
    summary['correlation_matrix'] = merged_df[score_cols].corr()
    summary['success'] = True
    return summary
def test_variant_subset(variant_df: pd.DataFrame, iteration_name: str) -> float:
    """Helper for the bisection algorithm. Runs Gnomon & PLINK2 on a variant subset and returns their correlation."""
    temp_score_path = CI_WORKDIR / f"_temp_score_{iteration_name}.tsv"
    # Ensure columns match the expected unified format
    variant_df.columns = ['variant_id', 'effect_allele', 'other_allele', 'effect_weight']
    variant_df.to_csv(temp_score_path, sep='\t', index=False)

    # --- Run Gnomon (Corrected) ---
    # Gnomon implicitly uses the input prefix for output naming. We must give it a unique one
    # to avoid overwriting files in parallel or sequential runs.
    g_iter_prefix = CI_WORKDIR / f"_g_iter_{iteration_name}"
    # The command does NOT take an --out flag.
    # The 'score' subcommand is added here.
    g_cmd = [
        str(GNOMON_BINARY),
        "score",
        str(temp_score_path),
        str(SHARED_GENOTYPE_PREFIX),
    ]
    g_res = run_and_measure(g_cmd, f"Gnomon_Iter_{iteration_name}", SHARED_GENOTYPE_PREFIX)
    # After running, move the output to the unique name we expect
    if g_res['success']:
        try:
            shutil.move(SHARED_GENOTYPE_PREFIX.with_suffix('.sscore'), g_iter_prefix.with_suffix('.sscore'))
        except FileNotFoundError:
            print(f"  > [BISECTION_WARNING] Could not find Gnomon output for iteration {iteration_name}")
            return -1.0 # Return a value indicating failure

    # --- Run PLINK2 ---
    p2_prefix = CI_WORKDIR / f"_p2_iter_{iteration_name}"
    p2_cmd = [str(PLINK2_BINARY), "--bfile", str(SHARED_GENOTYPE_PREFIX), "--score", str(temp_score_path), "1", "2", "4", "header", "no-mean-imputation", "--out", str(p2_prefix)]
    p2_res = run_and_measure(p2_cmd, f"PLINK2_Iter_{iteration_name}", p2_prefix)
    
    # --- Analyze and return correlation ---
    if not (g_res['success'] and p2_res['success']):
        return -1.0 # Signal a failure in one of the tool runs

    try:
        g_df = pd.read_csv(g_iter_prefix.with_suffix(".sscore"), sep=r'\s+').rename(columns={'#IID': 'IID', 'effect_weight_AVG': 'SCORE_GNOMON'})
        p2_df = pd.read_csv(p2_prefix.with_suffix(".sscore"), sep=r'\s+').rename(columns={'#IID': 'IID', 'SCORE1_AVG': 'SCORE_PLINK2'})
        merged = pd.merge(g_df[['IID', 'SCORE_GNOMON']], p2_df[['IID', 'SCORE_PLINK2']], on='IID').dropna()
        if len(merged) < 2: return 1.0
        correlation = merged['SCORE_GNOMON'].corr(merged['SCORE_PLINK2'])
        return correlation if pd.notna(correlation) else 1.0
    except Exception as e:
        print(f"  > [BISECTION_ANALYSIS_ERROR] Could not analyze results for iteration {iteration_name}: {e}", flush=True)
        return -1.0 # Signal a failure

def run_disagreement_discovery(initial_score_df: pd.DataFrame):
    """Performs the bisection search to find the variant causing the most disagreement."""
    print_header("DISAGREEMENT DISCOVERY MODE", "*")

    # Pre-filter the score file to only include variants that exist in the genotype data.
    # This is the step to prevent testing partitions with zero overlapping variants
    # Shouldn't affect anything but just speeds it up. Unnecessary to do
    print("[DISCOVERY] Pre-filtering score file against genotype data...")
    bim_df = pd.read_csv(SHARED_GENOTYPE_PREFIX.with_suffix('.bim'), sep=r'\s+', header=None, usecols=[1], names=['variant_id'], dtype=str)
    valid_variant_ids = set(bim_df['variant_id'])
    
    # The list of variants to search is the intersection of the score file and the bim file.
    searchable_variants_df = initial_score_df[initial_score_df['variant_id'].isin(valid_variant_ids)].copy()
    
    if searchable_variants_df.empty:
        print("❌ Bisection failed: No overlapping variants found between score file and genotype data. Aborting discovery.", flush=True)
        return
        
    print(f"[DISCOVERY] Found {len(searchable_variants_df)} overlapping variants to search.")

    print("[DISCOVERY] Establishing baseline correlation with filtered score file...")
    baseline_corr = test_variant_subset(searchable_variants_df, "baseline")
    print(f"[DISCOVERY] Baseline Gnomon vs PLINK2 Correlation on overlapping variants: {baseline_corr:.6f}")
    if baseline_corr >= CORRELATION_THRESHOLD:
        print("✅ Baseline correlation on overlapping variants is high. No disagreement discovery needed.", flush=True)
        return

    current_variants_df = searchable_variants_df.copy()
    iteration = 0
    while len(current_variants_df) > BISECTION_TERMINATION_SIZE:
        iteration += 1
        print_header(f"Bisection Iteration {iteration}: {len(current_variants_df)} variants remaining", "-")
        
        midpoint = len(current_variants_df) // 2
        part_a_df = current_variants_df.iloc[:midpoint]
        part_b_df = current_variants_df.iloc[midpoint:]

        print(f"[DISCOVERY] Testing Part A ({len(part_a_df)} variants)...")
        corr_a = test_variant_subset(part_a_df, f"{iteration}A")
        print(f"  > Correlation for Part A: {corr_a:.6f}")

        print(f"[DISCOVERY] Testing Part B ({len(part_b_df)} variants)...")
        corr_b = test_variant_subset(part_b_df, f"{iteration}B")
        print(f"  > Correlation for Part B: {corr_b:.6f}")

        if corr_a < 0 and corr_b < 0:
            # Only abort if BOTH sub-tests fail completely.
            print("❌ Bisection failed: BOTH halves failed their sub-tests. Aborting discovery.", flush=True)
            return
        elif corr_a < 0:
            # If only Part A fails, continue the search with Part B.
            print("[DISCOVERY] Part A failed its sub-test. Proceeding with Part B.")
            current_variants_df = part_b_df
        elif corr_b < 0:
            # If only Part B fails, continue the search with Part A.
            print("[DISCOVERY] Part B failed its sub-test. Proceeding with Part A.")
            current_variants_df = part_a_df
        elif corr_a < CORRELATION_THRESHOLD and corr_b < CORRELATION_THRESHOLD:
            # If both succeed but have low correlation, pursue the half with the worse correlation.
            print("[DISCOVERY] Both halves show disagreement. Pursuing the worse one.")
            current_variants_df = part_a_df if corr_a <= corr_b else part_b_df
        elif corr_a < CORRELATION_THRESHOLD:
            # If only Part A has low correlation, it is the culprit.
            print("[DISCOVERY] Disagreement isolated to Part A.")
            current_variants_df = part_a_df
        elif corr_b < CORRELATION_THRESHOLD:
            # If only Part B has low correlation, it is the culprit.
            print("[DISCOVERY] Disagreement isolated to Part B.")
            current_variants_df = part_b_df
        else:
            # If both succeed with high correlation, the problem is likely a complex interaction.
            print("✅ Disagreement resolved. Both halves have high correlation. The issue is likely an interaction between the two halves. Halting search.", flush=True)
            return

    print_header(f"Final Pinpointing: {len(current_variants_df)} suspect variants", "-")
    suspect_results = []
    for i, row in current_variants_df.iterrows():
        variant_id = row['variant_id']
        print(f"[DISCOVERY] Testing single suspect variant: {variant_id}")
        single_variant_df = pd.DataFrame([row])
        corr = test_variant_subset(single_variant_df, f"final_{variant_id.replace(':', '_')}")
        if corr < 0: continue # Skip failed runs
        suspect_results.append({'id': variant_id, 'corr': corr, 'df': single_variant_df})
        print(f"  > Correlation for {variant_id}: {corr:.6f}")
    
    if suspect_results:
        worst_offender = min(suspect_results, key=lambda x: x['corr'])
        print_header("DISAGREEMENT ISOLATED", "*")
        print(f"The primary source of disagreement is variant: {worst_offender['id']}")
        print(f"On its own, it yields a Gnomon vs PLINK2 correlation of: {worst_offender['corr']:.6f}")
        print("\nMinimal scorefile to reproduce this disagreement:")
        print("---------------------------------------------")
        print(worst_offender['df'].to_string(index=False))
        print("---------------------------------------------")

def main():
    """Main function to run the entire benchmark and discovery pipeline."""
    print_header("GNOMON CI TEST & BENCHMARK SUITE")
    setup_environment()
    create_synchronized_genotype_files(ORIGINAL_PLINK_PREFIX, SHARED_GENOTYPE_PREFIX)

    all_results, failures, final_summary_data = [], [], []

    for pgs_id, url in PGS_SCORES.items():
        print_header(f"TESTING SCORE: {pgs_id}", "~")
        raw_score_file = CI_WORKDIR / Path(url.split('/')[-1]).stem
        unified_score_df = create_unified_scorefile(raw_score_file)
        unified_score_file_path = CI_WORKDIR / f"{pgs_id}_unified_format.tsv"

        pgs_run_results = []
        
        commands_to_run = [
            # The 'score' subcommand is added here.
            (
                "gnomon",
                [
                    str(GNOMON_BINARY),
                    "score",
                    str(unified_score_file_path),
                    str(SHARED_GENOTYPE_PREFIX),
                ],
                SHARED_GENOTYPE_PREFIX,
            ),
            ("plink2", [str(PLINK2_BINARY), "--bfile", str(SHARED_GENOTYPE_PREFIX), "--score", str(unified_score_file_path), "1", "2", "4", "header", "no-mean-imputation", "--out", str(CI_WORKDIR / f"plink2_{pgs_id}")], CI_WORKDIR / f"plink2_{pgs_id}"),
            ("pylink", ["python3", str(PYLINK_SCRIPT), "--precise", "--bfile", str(SHARED_GENOTYPE_PREFIX), "--score", str(unified_score_file_path), "--out", str(CI_WORKDIR / f"pylink_{pgs_id}"), "1", "2", "4"], CI_WORKDIR / f"pylink_{pgs_id}"),
        ]
        
        for tool, cmd, out_prefix in commands_to_run:
            res = run_and_measure(cmd, f"{tool}_{pgs_id}", out_prefix)
            if tool == 'gnomon' and res['success']:
                try:
                    shutil.move(out_prefix.with_suffix('.sscore'), CI_WORKDIR / f"gnomon_{pgs_id}.sscore")
                except FileNotFoundError:
                    print(f"Warning: Could not find expected Gnomon output at {out_prefix.with_suffix('.sscore')}")
            pgs_run_results.append(res)
            if not res['success']: failures.append(f"{pgs_id} ({tool}_failed)")
        
        pgs_summary = analyze_pgs_results(pgs_id, pgs_run_results)
        final_summary_data.append(pgs_summary)

        if pgs_id == DISCOVERY_PGS_ID and pgs_summary.get('success'):
            corr_matrix = pgs_summary.get('correlation_matrix')
            if corr_matrix is not None and 'SCORE_gnomon' in corr_matrix and 'SCORE_plink2' in corr_matrix:
                gnomon_plink2_corr = corr_matrix.loc['SCORE_gnomon', 'SCORE_plink2']
                if gnomon_plink2_corr < CORRELATION_THRESHOLD:
                    run_disagreement_discovery(unified_score_df)
    
    print_final_summary(final_summary_data)

    print_header(f"CI CHECK {'FAILED' if failures else 'PASSED'}", "*")
    if failures:
        print('❌ One or more tool executions failed:', flush=True)
        [print(f'  - {f}', flush=True) for f in sorted(list(set(failures)))]
        sys.exit(1)
    else:
        print('🎉 All tests passed successfully.', flush=True)
        sys.exit(0)

def print_final_summary(summary_data: list):
    """Prints a single, comprehensive summary report at the end of the run."""
    print_header("FINAL TEST & BENCHMARKING REPORT", "*")
    for pgs_summary in summary_data:
        pgs_id = pgs_summary['pgs_id']
        print_header(f"Analysis for {pgs_id}", "=")
        if not pgs_summary.get('success', False):
            print(f"❌ Analysis for {pgs_id} could not be completed due to lack of successful tool runs.")
            continue
        
        report_df = pgs_summary.get('report_df')
        if report_df is None or report_df.empty:
            print(f"❌ Analysis for {pgs_id} could not be completed due to empty results after processing.")
            continue
        
        # --- Correlation Report ---
        corr_matrix = pgs_summary.get('correlation_matrix')
        if corr_matrix is not None:
            num_tools = corr_matrix.shape[0]
            print(f"✅ Concordance established using {num_tools} tools.")
            print_debug_header("Score Correlation Matrix")
            print(corr_matrix.to_markdown(floatfmt=".8f"))
        
        # --- Missingness Report ---
        missing_cols = sorted([c for c in report_df.columns if c.startswith('MISSING_PCT_')])
        if missing_cols:
            missing_stats = []
            for col in missing_cols:
                tool_name = col.replace('MISSING_PCT_', '')
                # Calculate mean, handling potential NaN values gracefully
                avg_missing = report_df[col].mean()
                missing_stats.append({'Tool': tool_name, 'Average Missingness (%)': avg_missing})
            
            if missing_stats:
                print_debug_header("Missingness Report (Average across all individuals)")
                print(pd.DataFrame(missing_stats).to_markdown(index=False, floatfmt=".4f"))

if __name__ == '__main__':
    main()
