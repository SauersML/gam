import numpy as np
import pandas as pd
import sys
import os
import subprocess
import requests
import zipfile
import shutil
import time
import math
import multiprocessing
import gmpy2
from pathlib import Path
from functools import reduce

# --- Configuration Parameters ---
N_VARIANTS = 50000
N_INDIVIDUALS = 400
CHR = '22'
CHR_LENGTH = 39_005_000
ALT_EFFECT_PROB = 0.3
MISSING_RATE = 0.5

# Mixture weights for allele-frequency distributions:
FREQ_DIST_WEIGHTS = [0.25, 0.25, 0.25, 0.25]
# Mixture weights for effect-size distributions:
EFFECT_DIST_WEIGHTS = [1.0, 0.0, 0.0, 0.0]

# --- CI/Validation Configuration ---
WORKDIR = Path("./sim_workdir")
OUTPUT_PREFIX = WORKDIR / "simulated_data"
CORR_THRESHOLD = 0.99999
MAD_THRESHOLD = 0.00001

# --- HELPER FUNCTIONS ---

def sum_column_precise(col):
    """Sums a column of floats using the ultra-precise gmpy2 library."""
    gmpy2.get_context().precision = 256
    return sum(gmpy2.mpfr(f) for f in col)

def sample_allele_frequencies(n):
    """Samples allele frequencies from a mixture of distributions."""
    choices = np.random.choice(4, size=n, p=FREQ_DIST_WEIGHTS)
    p = np.empty(n)
    for i, c in enumerate(choices):
        if c == 0: p[i] = np.random.uniform(0.001, 0.05)
        elif c == 1:
            r = np.random.beta(0.5, 10)
            p[i] = min(r, 1 - r)
        elif c == 2:
            r = np.random.beta(0.2, 0.2)
            p[i] = min(r, 1 - r)
        else: p[i] = np.random.uniform(0.01, 0.99)
    return p

def sample_effect_sizes(n):
    """Samples effect sizes from a mixture of distributions."""
    choices = np.random.choice(4, size=n, p=EFFECT_DIST_WEIGHTS)
    w = np.empty(n)
    for i, c in enumerate(choices):
        if c == 0: w[i] = np.random.normal(0, 0.0001)
        elif c == 1: w[i] = np.random.laplace(0, 0.5)
        elif c == 2: w[i] = np.random.uniform(-1, 1)
        else: w[i] = np.random.standard_cauchy() * 0.5
    return w

def sample_effect_alleles(n, ref, alt, af):
    """Chooses an effect allele, biased by allele frequency."""
    alt_prob = np.clip(ALT_EFFECT_PROB + (0.5 - af), 0.1, 0.9)
    mask = np.random.rand(n) < alt_prob
    return np.where(mask, alt, ref)

def _write_plink_files(prefix, bim_df, individuals, genotypes_df):
    """Helper to write a set of PLINK files for a test case."""
    with open(prefix.with_suffix(".fam"), 'w') as f:
        for iid in individuals:
            f.write(f"{iid} {iid} 0 0 0 -9\n")
    bim_df[['chr','id','cm','pos','a1','a2']].to_csv(
        prefix.with_suffix(".bim"), sep='\t', header=False, index=False
    )
    code_map = {0:0b00, 1:0b10, 2:0b11, -1:0b01}
    n = len(individuals)
    with open(prefix.with_suffix(".bed"), 'wb') as f:
        f.write(bytes([0x6c, 0x1b, 0x01]))
        for _, row in genotypes_df.iterrows():
            for j in range(0, n, 4):
                byte = 0
                chunk = row.iloc[j:j+4]
                for k, geno in enumerate(chunk):
                    byte |= (code_map[int(geno)] << (k*2))
                f.write(byte.to_bytes(1, 'little'))
    print(f"Programmatically wrote {prefix}.bed/.bim/.fam")

def _write_score_file(filename, score_df):
    """Helper to write a score file for a test case."""
    score_df.to_csv(filename, sep='\t', index=False)
    print(f"Programmatically wrote {filename}")

def print_file_header(filepath: Path, tool_name: str):
    """Prints the header and first data line of a file for inspection."""
    print(f"\n--- {tool_name} Output File Header and First Line ---")
    print(f"  > File: {filepath}")
    try:
        with open(filepath, 'r') as f:
            header = f.readline().strip()
            first_line = f.readline().strip()
            print(f"  > Header: {header}")
            print(f"  > Line 1: {first_line}")
    except FileNotFoundError:
        print(f"  > ❌ File not found.")
    except Exception as e:
        print(f"  > ⚠️ Could not read file: {e}")

# --- SIMULATION STEPS ---

def generate_variants_and_weights():
    """Step 1: Simulate variant positions, alleles, frequencies, and effect sizes."""
    print(f"Step 1: Simulating {N_VARIANTS} variants on chr{CHR}...")
    positions = np.sort(np.random.choice(np.arange(1, CHR_LENGTH + 1), N_VARIANTS, replace=False))
    alleles = np.array(['A', 'C', 'G', 'T'])
    ref_idx = np.random.randint(0, 4, N_VARIANTS)
    alt_idx = (ref_idx + np.random.randint(1, 4, N_VARIANTS)) % 4
    variants_df = pd.DataFrame({
        'chr': CHR, 'pos': positions, 'ref': alleles[ref_idx], 'alt': alleles[alt_idx],
        'af': sample_allele_frequencies(N_VARIANTS),
        'effect_weight': sample_effect_sizes(N_VARIANTS)
    })
    variants_df['effect_allele'] = sample_effect_alleles(N_VARIANTS, variants_df['ref'], variants_df['alt'], variants_df['af'])
    print("...Variant and weight simulation complete.")
    return variants_df

def generate_genotypes(variants_df):
    """Step 2: Generate genotypes under Hardy-Weinberg equilibrium."""
    print(f"Step 2: Simulating genotypes for {N_INDIVIDUALS} individuals...")
    p = variants_df['af'].values
    q = 1 - p
    hwe_probs = np.vstack([q**2, 2*p*q, p**2]).T # Note: HWE is for alt allele count
    rand_draws = np.random.rand(N_VARIANTS, N_INDIVIDUALS)
    cum_probs = hwe_probs.cumsum(axis=1)
    genotypes = (rand_draws > cum_probs[:, [0]]) + (rand_draws > cum_probs[:, [1]])
    print("...Genotype simulation complete.")
    return genotypes.astype(int)

def introduce_missingness(genotypes):
    """Step 3: Introduce random missingness into the genotype matrix."""
    print(f"Step 4: Introducing {MISSING_RATE*100:.1f}% missingness...")
    mask = np.random.rand(*genotypes.shape) < MISSING_RATE
    g = genotypes.astype(float)
    g[mask] = -1
    print("...Missingness introduced.")
    return g.astype(int)

def calculate_ground_truth_prs(genotypes, variants_df):
    """
    Step 5: Calculate ground truth PRS, normalized by the number of scored alleles.
    """
    print("Step 5: Calculating ground truth polygenic scores (by scored alleles)...")
    effect_weights = variants_df['effect_weight'].values
    is_alt_effect = (variants_df['effect_allele'] == variants_df['alt']).values
    valid_mask = (genotypes != -1)

    dosages = np.where(is_alt_effect[:, np.newaxis], genotypes, 2 - genotypes).astype(float)
    score_components = np.where(valid_mask, dosages * effect_weights[:, np.newaxis], 0)

    print(f"    > Dispatching summations to {multiprocessing.cpu_count()} CPU cores...")
    with multiprocessing.Pool() as pool:
        score_sums = pool.map(sum_column_precise, score_components.T)

    # Denominator is the number of non-missing variants (loci) scored per person.
    variant_counts_per_person = valid_mask.sum(axis=0)
    
    score_avg = np.array([s / v if v != 0 else 0.0 for s, v in zip(score_sums, variant_counts_per_person)], dtype=float)
    return pd.DataFrame({'FID': f'sample_{i+1}', 'IID': f'sample_{i+1}', 'PRS_AVG': score_avg[i]} for i in range(N_INDIVIDUALS))


def write_output_files(prs_results, variants_df, genotypes_with_missing, prefix: Path):
    """Step 6: Write all output files for the large-scale simulation."""
    print(f"Step 6: Writing all output files to prefix '{prefix}'...")
    sscore_truth = prefix.with_suffix(".truth.sscore")
    prs_results[['FID', 'IID', 'PRS_AVG']].to_csv(sscore_truth, sep='\t', index=False, header=True, float_format='%.17g')
    print(f"...Ground truth PRS results written to {sscore_truth}")
    gnomon_scorefile = prefix.with_suffix(".gnomon.score")
    gdf = variants_df.copy()
    gdf['variant_id'] = gdf['chr'].astype(str) + ':' + gdf['pos'].astype(str)
    gdf['other_allele'] = np.where(gdf['effect_allele'] == gdf['ref'], gdf['alt'], gdf['ref'])
    gdf[['variant_id', 'effect_allele', 'other_allele', 'effect_weight']].rename(columns={'effect_weight': 'simulated_score'}).to_csv(gnomon_scorefile, sep='\t', index=False)
    print(f"...Gnomon-native scorefile written to {gnomon_scorefile}")

    bim_df = pd.DataFrame({'chr': variants_df['chr'], 'id': variants_df['chr'].astype(str) + ':' + variants_df['pos'].astype(str), 'cm': 0, 'pos': variants_df['pos'], 'a1': variants_df['ref'], 'a2': variants_df['alt']})
    _write_plink_files(prefix, bim_df, [f"sample_{i+1}" for i in range(N_INDIVIDUALS)], pd.DataFrame(genotypes_with_missing, index=bim_df['id']))

def run_simple_dosage_test(workdir: Path, gnomon_path: Path, plink_path: Path, pylink_path: Path, run_cmd_func):
    """
    Runs a comprehensive, built-in test case to validate dosage calculations,
    comparing Gnomon, Plink2, and PyLink against a ground truth.
    """
    prefix = workdir / "simple_test"
    print("\n" + "="*80)
    print("= Running Comprehensive Simple Test Case")
    print("="*80)
    print(f"Test files will be prefixed: {prefix}")

    def _generate_test_data():
        """Generates the in-memory data for the simple test case."""
        individuals = [
            'id_hom_ref', 'id_het', 'id_hom_alt', 'id_new_person',
            'id_special_case', 'id_multi_AC', 'id_multi_AG'#, 'id_multi_CG'
        ]
        bim_data = [
            {'chr':'1','id':'1:1000','cm':0,'pos':1000,'a1':'A','a2':'G'},
            {'chr':'1','id':'1:2000','cm':0,'pos':2000,'a1':'C','a2':'T'},
            {'chr':'1','id':'1:3000','cm':0,'pos':3000,'a1':'A','a2':'T'},
            *[{'chr':'1','id':f'1:{10000+i}','cm':0,'pos':10000+i,'a1':'A','a2':'T'} for i in range(50)],
            {'chr':'1','id':'1:50000:A:C','cm':0,'pos':50000,'a1':'A','a2':'C'},
            {'chr':'1','id':'1:50000:A:G','cm':0,'pos':50000,'a1':'A','a2':'G'},
            {'chr':'1','id':'1:60000:T:C','cm':0,'pos':60000,'a1':'T','a2':'C'}
        ]
        bim_df = pd.DataFrame(bim_data)
        score_data = [
            {'variant_id':'1:1000','effect_allele':'G','other_allele':'A','simple_score':0.5},
            {'variant_id':'1:2000','effect_allele':'T','other_allele':'C','simple_score':-0.2},
            {'variant_id':'1:3000','effect_allele':'A','other_allele':'T','simple_score':-0.7},
            *[{'variant_id':f'1:{10000+i}','effect_allele':'A','other_allele':'T','simple_score':0.1} for i in range(50)],
            {'variant_id':'1:50000','effect_allele':'C','other_allele':'A','simple_score':10.0},
            {'variant_id':'1:50000','effect_allele':'G','other_allele':'A','simple_score':-5.0},
            {'variant_id':'1:60000','effect_allele':'C','other_allele':'T','simple_score':1.0}
        ]
        score_df = pd.DataFrame(score_data)
        genotypes_df = pd.DataFrame(-1, index=bim_df['id'], columns=individuals)
        genotypes_df.loc['1:1000', 'id_hom_ref'] = 0; genotypes_df.loc['1:2000', 'id_hom_ref'] = 0
        genotypes_df.loc['1:1000', 'id_het'] = 1; genotypes_df.loc['1:1000', 'id_hom_alt'] = 2
        genotypes_df.loc['1:2000', 'id_hom_alt'] = 2; genotypes_df.loc['1:3000', 'id_new_person'] = 1
        special_variants = [f'1:{10000+i}' for i in range(50)]
        genotypes_df.loc[special_variants[:10], 'id_special_case'] = 2
        genotypes_df.loc[special_variants[10:40], 'id_special_case'] = 1
        genotypes_df.loc[special_variants[40:], 'id_special_case'] = 0
        genotypes_df.loc['1:50000:A:C', 'id_multi_AC'] = 1; genotypes_df.loc['1:50000:A:G', 'id_multi_AC'] = -1; genotypes_df.loc['1:60000:T:C', 'id_multi_AC'] = 1
        genotypes_df.loc['1:50000:A:C', 'id_multi_AG'] = -1; genotypes_df.loc['1:50000:A:G', 'id_multi_AG'] = 1; genotypes_df.loc['1:60000:T:C', 'id_multi_AG'] = 1
        #genotypes_df.loc['1:50000:A:C', 'id_multi_CG'] = 1; genotypes_df.loc['1:50000:A:G', 'id_multi_CG'] = 1; genotypes_df.loc['1:60000:T:C', 'id_multi_CG'] = 1
        return bim_df, score_df, individuals, genotypes_df

    def _calculate_biologically_accurate_truth(bim_df, score_df, individuals, genotypes_df):
        """
        An independent oracle that calculates the 100% correct biological outcome,
        normalized by the number of scored loci. This logic is designed to mirror
        a robust tool's handling of multiallelic and ambiguous sites.
        """
        # Group score rules by unique chromosomal locus for efficient lookup.
        score_rules_by_locus = {}
        score_df['locus'] = score_df['variant_id'].str.split(':').str[:2].str.join(':')
        for locus, group in score_df.groupby('locus'):
            score_rules_by_locus[locus] = group.to_dict('records')
        
        total_unique_score_loci = len(score_rules_by_locus)
    
        # Group bim records by unique chromosomal locus.
        bim_by_locus = {}
        bim_df['locus'] = bim_df['chr'].astype(str) + ':' + bim_df['pos'].astype(str)
        for locus, group in bim_df.groupby('locus'):
            bim_by_locus[locus] = group.to_dict('records')
    
        truth_data = []
        iids_expected_to_fail = set()
    
        for iid in individuals:
            sum_score = 0.0
            loci_scored_count = 0
            is_fatal_error = False
    
            for locus, rules in score_rules_by_locus.items():
                # Find all non-missing genotype evidence for this person at this locus.
                evidence = []
                if locus in bim_by_locus:
                    for bim_record in bim_by_locus[locus]:
                        genotype_val = genotypes_df.loc[bim_record['id'], iid]
                        if genotype_val != -1:
                            evidence.append({'bim': bim_record, 'geno': genotype_val})
                
                # Policy-based resolution based on the collected evidence.
                if not evidence:
                    # Case 1: Truly missing. No evidence found. Locus is not scored.
                    continue
                
                if len(evidence) > 1:
                    # Case 2: Contradictory Data. Multiple non-missing genotypes found
                    # for the same person at the same locus. This is a fatal data
                    # integrity error that a robust tool should fail on.
                    is_fatal_error = True
                    iids_expected_to_fail.add(iid)
                    break # Stop processing loci for this person.
    
                # Case 3: Success - One Unambiguous Interpretation.
                winning_evidence = evidence[0]
                winning_bim = winning_evidence['bim']
                winning_geno = winning_evidence['geno']
    
                # Resolve the diploid genotype based on the winning BIM record.
                # Genotype is encoded relative to (a1, a2) where 0=a1/a1, 1=a1/a2, 2=a2/a2.
                if winning_geno == 0:
                    resolved_genotype = (winning_bim['a1'], winning_bim['a1'])
                elif winning_geno == 1:
                    resolved_genotype = tuple(sorted((winning_bim['a1'], winning_bim['a2'])))
                else: # winning_geno == 2
                    resolved_genotype = (winning_bim['a2'], winning_bim['a2'])
    
                # A single locus is scored, even if multiple rules apply to it.
                loci_scored_count += 1
                
                # Apply all score rules for this locus using the single resolved genotype.
                for rule in rules:
                    effect_allele = rule['effect_allele']
                    weight = rule['simple_score']
                    
                    # The dosage is the count of the effect allele in the resolved genotype.
                    dosage = float(resolved_genotype.count(effect_allele))
                    sum_score += dosage * weight
    
            # Final aggregation for the person
            if is_fatal_error:
                truth_data.append({'IID': iid, 'SCORE_TRUTH': np.nan, 'MISSING_PCT_TRUTH': np.nan})
            else:
                # Normalize score by the number of unique loci that were successfully scored.
                avg_score = sum_score / loci_scored_count if loci_scored_count > 0 else 0.0
                loci_missed_count = total_unique_score_loci - loci_scored_count
                missing_pct = (loci_missed_count / total_unique_score_loci) * 100.0 if total_unique_score_loci > 0 else 0.0
                truth_data.append({'IID': iid, 'SCORE_TRUTH': avg_score, 'MISSING_PCT_TRUTH': missing_pct})
    
        return pd.DataFrame(truth_data), iids_expected_to_fail

    # --- Main test logic ---
    bim_df, score_df, individuals, genotypes_df = _generate_test_data()
    truth_df, iids_expected_to_fail = _calculate_biologically_accurate_truth(bim_df, score_df, individuals, genotypes_df)
    score_file = prefix.with_suffix(".score")
    _write_plink_files(prefix, bim_df, individuals, genotypes_df)
    _write_score_file(score_file, score_df)

    # --- Run all tools ---
    gnomon_res = run_cmd_func(
        [gnomon_path, "score", score_file.name, prefix.name],
        "Simple Gnomon Test",
        workdir,
    )
    gnomon_output_path = workdir / f"{prefix.name}.sscore"
    if gnomon_res and gnomon_res.returncode == 0:
        print_file_header(gnomon_output_path, "Gnomon")

    plink_cmd = [f"./{plink_path.name}", "--bfile", prefix.name, "--score", score_file.name, "1", "2", "4", "header", "no-mean-imputation", "--out", prefix.name + "_plink"]
    plink_res = run_cmd_func(plink_cmd, "Simple Plink2 Test", workdir)
    plink_output_path = workdir / f"{prefix.name}_plink.sscore"
    if plink_res and plink_res.returncode == 0:
        print_file_header(plink_output_path, "Plink2")
    
    pylink_cmd = [sys.executable, pylink_path, "--bfile", prefix.name, "--score", score_file.name, "--out", prefix.name + "_pylink", "1", "2", "4"]
    pylink_res = run_cmd_func(pylink_cmd, "Simple PyLink Test", workdir)
    pylink_output_path = workdir / f"{prefix.name}_pylink.sscore"
    if pylink_res and pylink_res.returncode == 0:
        print_file_header(pylink_output_path, "PyLink")

    # --- Analyze results ---
    print("\n--- Analyzing Simple Test Results ---")
    results_dfs = [truth_df]
    gnomon_df = plink_df = pylink_df = None
    
    if gnomon_res and gnomon_res.returncode == 0:
        try:
            gnomon_df = pd.read_csv(gnomon_output_path, sep='\t').rename(columns={'#IID':'IID', 'simple_score_AVG':'SCORE_GNOMON', 'simple_score_MISSING_PCT': 'MISSING_PCT_GNOMON'})[['IID', 'SCORE_GNOMON', 'MISSING_PCT_GNOMON']]
            results_dfs.append(gnomon_df)
        except Exception as e: print(f"⚠️ Could not parse Gnomon output: {e}")

    if plink_res and plink_res.returncode == 0:
        try:
            #  Read SCORE1_AVG directly.
            plink_df = pd.read_csv(plink_output_path, sep='\t', skipinitialspace=True).rename(columns={'#IID':'IID', 'SCORE1_AVG': 'SCORE_PLINK2'})[['IID', 'SCORE_PLINK2']]
            results_dfs.append(plink_df)
        except Exception as e: print(f"⚠️ Could not parse Plink2 output: {e}")
        
    if pylink_res and pylink_res.returncode == 0:
        try:
            #  Read SCORE1_AVG directly.
            pylink_df = pd.read_csv(pylink_output_path, sep='\t').rename(columns={'#IID': 'IID', 'SCORE1_AVG': 'SCORE_PYLINK'})[['IID', 'SCORE_PYLINK']]
            results_dfs.append(pylink_df)
        except Exception as e: print(f"⚠️ Could not parse PyLink output: {e}")

    merged = reduce(lambda left, right: pd.merge(left, right, on='IID', how='outer'), results_dfs)
    display_cols = ['IID', 'SCORE_TRUTH', 'MISSING_PCT_TRUTH']
    if gnomon_df is not None: display_cols.extend(['SCORE_GNOMON', 'MISSING_PCT_GNOMON'])
    if plink_df is not None: display_cols.append('SCORE_PLINK2')
    if pylink_df is not None: display_cols.append('SCORE_PYLINK')

    print("\n--- Simple Dosage Test: Full Results Table ---")
    print("WHO (what sample) got WHICH SCORE and by WHICH tool:")
    print(merged[display_cols].to_markdown(index=False, floatfmt=".6f"))

    is_gnomon_ok = False
    if gnomon_df is not None:
        merged_compare = merged.dropna(subset=['SCORE_TRUTH', 'SCORE_GNOMON'])
        if not merged_compare.empty:
            scores_ok = np.allclose(merged_compare['SCORE_TRUTH'], merged_compare['SCORE_GNOMON'])
            missing_ok = np.allclose(merged_compare['MISSING_PCT_TRUTH'], merged_compare['MISSING_PCT_GNOMON'])
            if scores_ok and missing_ok:
                print("\n✅ Verification successful: Gnomon scores and missingness percentages match the ground truth.")
                is_gnomon_ok = True
            else:
                print("\n❌ Verification failed: Gnomon results DO NOT MATCH the ground truth.")
                if not scores_ok: print("  - Mismatch found in scores.")
                if not missing_ok: print("  - Mismatch found in missingness percentages.")
        else: # Handle case where all samples were expected to fail
            is_gnomon_ok = iids_expected_to_fail and gnomon_res.returncode != 0
            if is_gnomon_ok:
                print("\n✅ Verification successful: Gnomon correctly failed for ambiguous samples.")
            elif not iids_expected_to_fail:
                 print("\n❌ Verification failed: Gnomon produced no valid output to compare.")
    
    if is_gnomon_ok:
        print("\n✅ Simple Dosage Test SUCCEEDED.")
        return True
    else:
        print("\n❌ Simple Dosage Test FAILED.")
        return False

# --- TEST: IMPOSSIBLE DIPLOID ---

def run_impossible_diploid_test(workdir: Path, gnomon_path: Path, run_cmd_func):
    """
    Test Goal: Exercise the "fatal data inconsistency" branch by giving one
    individual two non-missing genotypes at the same locus.
    """
    prefix = workdir / "impossible_diploid_test"
    print("\n" + "="*80)
    print("= Running Impossible Diploid Crash Test")
    print("="*80)
    print(f"Test files will be prefixed: {prefix}")

    # 1. Test Data Generation
    individuals = ['id_ok', 'id_bad', 'id_missing']
    bim_df = pd.DataFrame([
        {'chr': 1, 'id': '1:50000:A:C', 'cm': 0, 'pos': 50000, 'a1': 'A', 'a2': 'C'},
        {'chr': 1, 'id': '1:50000:A:G', 'cm': 0, 'pos': 50000, 'a1': 'A', 'a2': 'G'},
    ])
    score_df = pd.DataFrame([
        {'variant_id': '1:50000', 'effect_allele': 'C', 'other_allele': 'A', 'crash_score': 1.0},
        {'variant_id': '1:50000', 'effect_allele': 'G', 'other_allele': 'A', 'crash_score': 2.0},
    ])
    genotypes_df = pd.DataFrame(-1, index=bim_df['id'], columns=individuals)
    # id_ok has one non-missing geno, should be fine.
    genotypes_df.loc['1:50000:A:C', 'id_ok'] = 1
    # id_bad has two non-missing genos for the same locus, this is the trigger.
    genotypes_df.loc['1:50000:A:C', 'id_bad'] = 2
    genotypes_df.loc['1:50000:A:G', 'id_bad'] = 1
    # id_missing is missing both, should be fine.
    
    score_file = prefix.with_suffix(".score")
    _write_plink_files(prefix, bim_df, individuals, genotypes_df)
    _write_score_file(score_file, score_df)

    # 2. Invocation
    gnomon_res = run_cmd_func(
        [gnomon_path, "score", score_file.name, prefix.name],
        "Impossible Diploid Test",
        workdir,
    )
    
    # 3. Validation  (require new ambiguity-resolution success path)
    if gnomon_res is None:
        print("❌ Test failed: Gnomon command could not be executed.")
        return False

    stderr = (gnomon_res.stderr or "").lower()

    # New behavior: non-fatal, explicit resolution signal(s)
    resolved_signal = (
        ("ambiguity resolved" in stderr) or
        ("complex variant resolution complete" in stderr)
    )

    if gnomon_res.returncode == 0 and resolved_signal:
        out_path = workdir / f"{prefix.name}.sscore"
        if out_path.exists():
            print_file_header(out_path, "Gnomon")
        print("\n✅ Verification successful: ambiguity detected and resolved (non-fatal).")
        print("\n✅ Impossible Diploid Test SUCCEEDED.")
        return True

    # Otherwise fail with a clear hint
    if "conflicting genotype data" in stderr:
        print("   > Observed legacy fatal error. Update required to ensure ambiguity is resolved.")
    else:
        print("   > Expected explicit ambiguity resolution with exit code 0; conditions not met.")
    print("\n❌ Impossible Diploid Test FAILED.")
    return False

# --- TEST: MULTIPLE SCORE FILES ---

def run_multi_score_file_test(workdir: Path, gnomon_path: Path, run_cmd_func):
    """
    Test Goal: Verify correct behavior when passing two separate --score files,
    including overlapping and unique loci.
    """
    prefix = workdir / "multi_score_test"
    print("\n" + "="*80)
    print("= Running Multiple Score-Files Correctness Test")
    print("="*80)
    print(f"Test files will be prefixed: {prefix}")

    # 1. Test Data Generation
    individuals = ['p1', 'p2']
    bim_df = pd.DataFrame([
        {'chr': 1, 'id': '1:1000', 'cm': 0, 'pos': 1000, 'a1': 'A', 'a2': 'G'}, # Unique to A
        {'chr': 1, 'id': '1:2000', 'cm': 0, 'pos': 2000, 'a1': 'C', 'a2': 'T'}, # Overlap (different effect allele)
        {'chr': 1, 'id': '1:3000', 'cm': 0, 'pos': 3000, 'a1': 'G', 'a2': 'T'}, # Unique to B
        {'chr': 1, 'id': '1:4000', 'cm': 0, 'pos': 4000, 'a1': 'A', 'a2': 'C'}, # Unique to A
        {'chr': 1, 'id': '1:5000', 'cm': 0, 'pos': 5000, 'a1': 'G', 'a2': 'C'}, # Unique to B
        {'chr': 1, 'id': '1:6000', 'cm': 0, 'pos': 6000, 'a1': 'A', 'a2': 'C'}, # Overlap (same effect allele)
    ])
    scoreA_df = pd.DataFrame([
        {'variant_id': '1:1000', 'effect_allele': 'G', 'other_allele': 'A', 'scoreA': 1.0},
        {'variant_id': '1:2000', 'effect_allele': 'T', 'other_allele': 'C', 'scoreA': 2.0},
        {'variant_id': '1:4000', 'effect_allele': 'A', 'other_allele': 'C', 'scoreA': 3.0},
        {'variant_id': '1:6000', 'effect_allele': 'C', 'other_allele': 'A', 'scoreA': 10.0},
    ])
    scoreB_df = pd.DataFrame([
        {'variant_id': '1:2000', 'effect_allele': 'C', 'other_allele': 'T', 'scoreB': -4.0}, # Diff effect allele
        {'variant_id': '1:3000', 'effect_allele': 'T', 'other_allele': 'G', 'scoreB': -5.0},
        {'variant_id': '1:5000', 'effect_allele': 'C', 'other_allele': 'G', 'scoreB': -6.0},
        {'variant_id': '1:6000', 'effect_allele': 'C', 'other_allele': 'A', 'scoreB': -20.0},
    ])
    # Give everyone all genotypes so missingness is 0
    genotypes_df = pd.DataFrame(1, index=bim_df['id'], columns=individuals)
    
    score_file_A = prefix.with_suffix(".scoreA.txt")
    score_file_B = prefix.with_suffix(".scoreB.txt")
    _write_plink_files(prefix, bim_df, individuals, genotypes_df)
    _write_score_file(score_file_A, scoreA_df)
    _write_score_file(score_file_B, scoreB_df)

    # 2. Ground Truth Calculation
    # Truth for p1 (all het, dosage=1 for all effect alleles)
    # ScoreA: (1.0*1 + 2.0*1 + 3.0*1 + 10.0*1) / 4 variants = 16.0 / 4 = 4.0
    # ScoreB: (-4.0*1 - 5.0*1 - 6.0*1 - 20.0*1) / 4 variants = -35.0 / 4 = -8.75
    truth_df = pd.DataFrame([
        {'#IID': 'p1', 'scoreA_AVG_TRUTH': 4.0, 'scoreB_AVG_TRUTH': -8.75},
        {'#IID': 'p2', 'scoreA_AVG_TRUTH': 4.0, 'scoreB_AVG_TRUTH': -8.75},
    ])

    # 2. Invocation
    scores_dir = workdir / "multi_score_test_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(score_file_A, scores_dir / score_file_A.name)
    shutil.copy(score_file_B, scores_dir / score_file_B.name)
    cmd = [
        gnomon_path,
        "score",
        str(scores_dir.resolve()),
        str(prefix.resolve()),
    ]
    gnomon_res = run_cmd_func(cmd, "Multi-Score-File Test", workdir)

    
    # 3. Validation
    if not (gnomon_res and gnomon_res.returncode == 0):
        print("❌ Test failed: Gnomon command failed to execute successfully.")
        return False
    
    gnomon_output_path = workdir / f"{prefix.name}.sscore"
    try:
        result_df = pd.read_csv(gnomon_output_path, sep='\t')
        print_file_header(gnomon_output_path, "Gnomon (Multi-Score Test)")
        
        merged = pd.merge(result_df, truth_df, on='#IID')
        
        # Check column names
        expected_cols = {'#IID', 'scoreA_AVG', 'scoreA_MISSING_PCT', 'scoreB_AVG', 'scoreB_MISSING_PCT'}
        if not expected_cols.issubset(result_df.columns):
            print(f"❌ Verification failed: Output missing expected columns. Found: {result_df.columns.tolist()}")
            return False

        # Check values
        a_ok = np.allclose(merged['scoreA_AVG'], merged['scoreA_AVG_TRUTH'])
        b_ok = np.allclose(merged['scoreB_AVG'], merged['scoreB_AVG_TRUTH'])
        missing_ok = (merged['scoreA_MISSING_PCT'] == 0).all() and (merged['scoreB_MISSING_PCT'] == 0).all()

        print("\n--- Multi-Score Test: Results Table ---")
        print(merged.to_markdown(index=False, floatfmt=".6f"))
        
        if a_ok and b_ok and missing_ok:
            print("\n✅ Verification successful: All calculated scores and missingness percentages match ground truth.")
            print("\n✅ Multiple Score-Files Test SUCCEEDED.")
            return True
        else:
            print("\n❌ Verification failed: Mismatches found.")
            if not a_ok: print("  - Mismatch in scoreA_AVG")
            if not b_ok: print("  - Mismatch in scoreB_AVG")
            if not missing_ok: print("  - Mismatch in missing percentages (expected 0)")
            print("\n❌ Multiple Score-Files Test FAILED.")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: Could not parse or validate Gnomon output. Error: {e}")
        return False
# --- MAIN VALIDATION RUNNER ---

def run_and_validate_tools(runtimes):
    """Downloads tools, runs them, validates results, and collects runtimes."""
    PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"
    PLINK2_BINARY_PATH = WORKDIR / "plink2"
    GNOMON_BINARY_PATH = Path("./target/release/gnomon").resolve()
    PYLINK_SCRIPT_PATH = (Path(__file__).resolve().parent / "pylink.py").resolve()
    overall_success = True

    def _print_header(title: str, char: str = "-"):
        print(f"\n{char*4} {title} {'-'*(70 - len(title) - 5)}")

    def run_command(cmd: list, step_name: str, cwd: Path) -> subprocess.CompletedProcess | None:
        """Executes a command and returns the full result object, or None on critical failure."""
        _print_header(f"Executing: {step_name}")
        cmd_str = [str(c) for c in cmd]
        print(f"  > Command: {' '.join(cmd_str)}")
        print(f"  > CWD: {cwd}")
        proc_env = os.environ.copy()
        if "gnomon" in str(cmd_str[0]):
            proc_env["RUST_BACKTRACE"] = "1"
        start_time = time.monotonic()
        try:
            result = subprocess.run(cmd_str, capture_output=True, text=True, cwd=cwd, env=proc_env, timeout=600)
            duration = time.monotonic() - start_time
            print("--- OUTPUT ---", result.stdout, sep="\n")
            if result.stderr: print("--- STDERR ---", result.stderr, sep="\n")
            print(f"\n  > Process exited with code {result.returncode}. (Completed in {duration:.4f}s)")
            if "Large-Scale Gnomon" in step_name and result.returncode == 0:
                runtimes.append({"Method": "Gnomon", "Runtime (s)": duration})
            return result
        except FileNotFoundError:
            print(f"  > ❌ CRITICAL ERROR: Command '{cmd[0]}' not found.")
            return None
        except subprocess.TimeoutExpired:
            print(f"  > ❌ CRITICAL ERROR: {step_name} timed out.")
            return None

    def setup_tools():
        _print_header("Step A: Setting up tools")
        if not GNOMON_BINARY_PATH.exists():
            print(f"  > ❌ ERROR: Gnomon binary not found at '{GNOMON_BINARY_PATH}'. Please build it first with 'cargo build --release'.")
            return False
        if not PLINK2_BINARY_PATH.exists():
            print(f"  > Downloading PLINK2 to '{PLINK2_BINARY_PATH}'...")
            zip_path = WORKDIR / "plink.zip"
            try:
                with requests.get(PLINK2_URL, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(zip_path, 'wb') as f: shutil.copyfileobj(r.raw, f)
                with zipfile.ZipFile(zip_path, 'r') as z: z.extract('plink2', path=WORKDIR)
                zip_path.unlink()
                PLINK2_BINARY_PATH.chmod(0o755)
                print("  > PLINK2 downloaded and extracted successfully.")
            except Exception as e:
                print(f"  > ❌ FAILED to download or extract PLINK2: {e}")
                return False
        if not PYLINK_SCRIPT_PATH.exists():
            print(f"  > ⚠️ WARNING: PyLink script not found at '{PYLINK_SCRIPT_PATH}'. It will be skipped in the simple test.")

        return True

    def analyze_large_scale_results():
        _print_header("Step D: Analyzing and Comparing Large-Scale Simulation Results")
        try:
            gnomon_output_path = WORKDIR / f"{OUTPUT_PREFIX.name}.sscore"
            truth_df = pd.read_csv(OUTPUT_PREFIX.with_suffix(".truth.sscore"), sep='\t').rename(columns={'PRS_AVG':'SCORE_TRUTH'})[['IID','SCORE_TRUTH']]
            gnomon_df = pd.read_csv(gnomon_output_path, sep='\t').rename(columns={'#IID':'IID','simulated_score_AVG':'SCORE_GNOMON'})[['IID','SCORE_GNOMON']]
        except (FileNotFoundError, KeyError) as e:
            print(f"  > ❌ ERROR: Failed to load or parse a result file. Error: {e}.")
            return False
        merged_df = pd.merge(truth_df, gnomon_df, on='IID')
        print("\n--- Score Correlation Matrix (Large-Scale) ---")
        corr_matrix = merged_df[['SCORE_TRUTH','SCORE_GNOMON']].corr()
        print(corr_matrix.to_markdown(floatfmt=".8f"))
        print("\n--- Mean Absolute Difference (MAD) vs. Ground Truth (Large-Scale) ---")
        mad_g = (merged_df['SCORE_TRUTH'] - merged_df['SCORE_GNOMON']).abs().mean()
        print(f"Gnomon vs. Truth: {mad_g:.10f}")
        corr_g = corr_matrix.loc['SCORE_TRUTH','SCORE_GNOMON']
        gnomon_ok = corr_g > CORR_THRESHOLD and mad_g < MAD_THRESHOLD
        if gnomon_ok:
            print(f"\n✅ Large-Scale Validation SUCCESS")
            return True
        else:
            print(f"\n❌ Large-Scale Validation FAILED for Gnomon: Corr={corr_g:.8f}, MAD={mad_g:.10f}")
            return False

    if not setup_tools(): return False
    
    # --- RUN ALL VALIDATION TESTS ---
    if not run_simple_dosage_test(WORKDIR, GNOMON_BINARY_PATH, PLINK2_BINARY_PATH, PYLINK_SCRIPT_PATH, run_command):
        overall_success = False

    if overall_success:
        if not run_impossible_diploid_test(WORKDIR, GNOMON_BINARY_PATH, run_command):
            overall_success = False
    else:
        print("Skipping Impossible Diploid Test due to previous failure.")

    if overall_success:
        if not run_multi_score_file_test(WORKDIR, GNOMON_BINARY_PATH, run_command):
            overall_success = False
    else:
        print("Skipping Multi-Score-File Test due to previous failure.")

    print("\n" + "="*80)
    print("= Running Large-Scale Simulation and Validation")
    print("="*80)
    
    if overall_success:
        gnomon_res = run_command(
            [
                GNOMON_BINARY_PATH,
                "score",
                f"{OUTPUT_PREFIX.name}.gnomon.score",
                OUTPUT_PREFIX.name,
            ],
            "Large-Scale Gnomon",
            WORKDIR,
        )
        if not (gnomon_res and gnomon_res.returncode == 0):
            print("❌ Gnomon failed to run on the large-scale dataset.")
            overall_success = False
        
        if overall_success and not analyze_large_scale_results():
            overall_success = False
    else:
        print("Skipping Large-Scale Simulation due to failure in a preceding test.")

    return overall_success

def print_runtime_summary(runtimes):
    """Prints mean runtimes for large-scale methods."""
    print("\n" + "="*50 + "\n= Mean Runtime for Large-Scale Methods\n" + "="*50)
    if runtimes:
        print(pd.DataFrame(runtimes).to_markdown(index=False, floatfmt=".4f"))
    else:
        print("No large-scale runtimes were recorded.")

def cleanup():
    """Cleans up the workspace directory."""
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
        print(f"Cleaned up {WORKDIR}")

def main():
    """Main pipeline: simulate, write files, validate."""
    np.random.seed(42)
    exit_code = 0
    runtimes = []
    if WORKDIR.exists(): shutil.rmtree(WORKDIR)
    WORKDIR.mkdir()
    try:
        print("--- Starting Full Simulation and File Writing Pipeline ---")
        variants_df = generate_variants_and_weights()
        genotypes_pristine = generate_genotypes(variants_df)
        genotypes_with_missing = introduce_missingness(genotypes_pristine)
        prs_df = calculate_ground_truth_prs(genotypes_with_missing, variants_df)
        write_output_files(prs_df, variants_df, genotypes_with_missing, OUTPUT_PREFIX)
        print("\n--- Simulation and File Writing Finished Successfully ---")
        if not run_and_validate_tools(runtimes):
            print("\n" + "!"*80)
            print("! OVERALL VALIDATION FAILED: One or more tests did not pass.")
            print("!"*80)
            exit_code = 1
        else:
            print("\n" + "✓"*80)
            print("✓ OVERALL VALIDATION SUCCEEDED: All tests passed and results were verified.")
            print("✓"*80)

    except Exception as e:
        import traceback
        print(f"\n--- An unexpected error occurred: {e} ---", file=sys.stderr)
        traceback.print_exc()
        exit_code = 1
    finally:
        print_runtime_summary(runtimes)
        cleanup()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
