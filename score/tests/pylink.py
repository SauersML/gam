import sys
import math
import argparse
import polars as pl
import numpy as np
import pandas as pd
import pyarrow as pa # Explicitly import to ensure it's installed for polars.to_pandas()


def read_fam_file(filepath: str) -> pl.DataFrame:
    """
    Reads a .fam file using Polars, robustly handling space or tab delimiters.
    """
    print(f"Reading sample info file (.fam): {filepath}...")
    try:
        df = pl.read_csv(filepath, has_header=False, separator='\n', new_columns=["data"])
        
        # .fam files are usually space-delimited, so try that first.
        struct_df = df.with_columns(
            pl.col("data").str.strip_chars().str.split_exact(" ", 5).alias("struct_data")
        )

        # If the split failed (e.g., file was tab-delimited), the struct's fields will be null.
        if struct_df.select(pl.col("struct_data").struct.field("field_0").is_null()).to_series().all():
            struct_df = df.with_columns(
                pl.col("data").str.strip_chars().str.split_exact("\t", 5).alias("struct_data")
            )

        df_final = struct_df.select(
            pl.col("struct_data").struct.field("field_0").alias("FID"),
            pl.col("struct_data").struct.field("field_1").alias("IID"),
            pl.col("struct_data").struct.field("field_2").alias("PAT"),
            pl.col("struct_data").struct.field("field_3").alias("MAT"),
            pl.col("struct_data").struct.field("field_4").alias("SEX"),
            pl.col("struct_data").struct.field("field_5").alias("PHEN"),
        )
        print(f"  > Loaded info for {df_final.height} samples.")
        return df_final
    except Exception as e:
        print(f"Error reading .fam file: {e}", file=sys.stderr)
        sys.exit(1)


def read_bim_file(filepath: str) -> pl.DataFrame:
    """
    Reads a .bim file using Polars, robustly handling tab or space delimiters.
    """
    print(f"Reading variant info file (.bim): {filepath}...")
    try:
        df = pl.read_csv(filepath, has_header=False, separator='\n', new_columns=["data"])
        
        # .bim files are usually tab-delimited, so try that first.
        struct_df = df.with_columns(
            pl.col("data").str.strip_chars().str.split_exact("\t", 5).alias("struct_data")
        )

        # If tab split failed for all rows, fallback to space.
        if struct_df.select(pl.col("struct_data").struct.field("field_0").is_null()).to_series().all():
            struct_df = df.with_columns(
                pl.col("data").str.strip_chars().str.split_exact(" ", 5).alias("struct_data")
            )
        
        df_final = struct_df.select(
            pl.col("struct_data").struct.field("field_0").cast(pl.Utf8).alias("CHR"),
            pl.col("struct_data").struct.field("field_1").cast(pl.Utf8).alias("ID"),
            pl.col("struct_data").struct.field("field_2").cast(pl.Utf8).alias("CM"),
            pl.col("struct_data").struct.field("field_3").cast(pl.Int64).alias("POS"),
            pl.col("struct_data").struct.field("field_4").cast(pl.Utf8).alias("A1"),
            pl.col("struct_data").struct.field("field_5").cast(pl.Utf8).alias("A2"),
        )
        print(f"  > Loaded info for {df_final.height} variants.")
        return df_final
    except Exception as e:
        print(f"Error reading .bim file: {e}", file=sys.stderr)
        sys.exit(1)


def read_score_file(filepath: str, id_col: int, allele_col: int, score_col: int) -> dict:
    """
    Reads the scoring file efficiently using Polars, robustly handling any whitespace delimiter.
    """
    print(f"Reading score file: {filepath}...")
    try:
        df = pl.read_csv(filepath, has_header=True, separator='\n', comment_prefix='#', new_columns=["data"])
        
        # Use a regex to extract all sequences of non-whitespace characters.
        list_df = df.with_columns(
            pl.col("data").str.extract_all(r"\S+").alias("list_data")
        )
        
        # Extract the required columns from the list based on user-provided indices.
        id_series = list_df.select(pl.col("list_data").list.get(id_col - 1).cast(pl.Utf8)).to_series()
        allele_series = list_df.select(pl.col("list_data").list.get(allele_col - 1).cast(pl.Utf8)).to_series()
        score_series = list_df.select(pl.col("list_data").list.get(score_col - 1).cast(pl.Float64)).to_series()

        score_data = {
            (vid, allele): score
            for vid, allele, score in zip(id_series, allele_series, score_series)
            if vid is not None and allele is not None
        }
        
        print(f"  > Loaded {len(score_data)} variant scores.")
        return score_data
    except Exception as e:
        print(f"Error reading score file: {e}", file=sys.stderr)
        sys.exit(1)


def find_ambiguous_samples(bed_filepath: str, num_samples: int, bim_df: pl.DataFrame) -> set:
    """
    Pass 1: Identifies samples with non-missing genotypes for multiple variants
    at the same genomic position using a fully vectorized approach.
    """
    print("\n--- Pass 1: Identifying ambiguous samples ---")
    
    # Use .len() which is the current Polars API for group counts
    ambiguous_pos_df = bim_df.group_by(['CHR', 'POS']).len().filter(pl.col('len') > 1)
    if ambiguous_pos_df.height == 0:
        print("  > No variants at duplicated positions found.")
        return set()
        
    variants_to_check_df = bim_df.with_row_index().join(ambiguous_pos_df, on=['CHR', 'POS'])
    indices_to_check = set(variants_to_check_df['index'].to_list())
    
    genotype_counts_at_pos = {}
    bytes_per_variant = math.ceil(num_samples / 4)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)

    with open(bed_filepath, "rb") as f:
        magic = f.read(3)
        if magic != b'\x6c\x1b\x01':
            raise ValueError("Invalid .bed file: incorrect magic number.")
            
        for i, variant_row in enumerate(bim_df.iter_rows(named=True)):
            variant_bytes = f.read(bytes_per_variant)
            if not variant_bytes: break
            
            if i in indices_to_check:
                byte_array = np.frombuffer(variant_bytes, dtype=np.uint8)
                unpacked = (byte_array[:, np.newaxis] >> shifts) & 0b11
                genotypes = unpacked.flatten()[:num_samples]
                non_missing_mask = (genotypes != 0b01)
                
                pos_tuple = (variant_row['CHR'], variant_row['POS'])
                if pos_tuple not in genotype_counts_at_pos:
                    genotype_counts_at_pos[pos_tuple] = non_missing_mask.astype(np.uint8)
                else:
                    genotype_counts_at_pos[pos_tuple] += non_missing_mask

    is_ambiguous = np.zeros(num_samples, dtype=bool)
    for pos_counts in genotype_counts_at_pos.values():
        is_ambiguous |= (pos_counts > 1)
        
    nan_indices = set(np.where(is_ambiguous)[0])
    
    if nan_indices:
        print(f"  > Identified {len(nan_indices)} sample(s) with ambiguous genotypes at single positions.")
    else:
        print("  > No ambiguous samples found.")
    return nan_indices


def calculate_scores(bed_filepath: str, num_samples: int, bim_df: pl.DataFrame, score_data: dict, precise: bool = False) -> tuple:
    """
    Pass 2: Calculates scores for all samples using a fully vectorized approach.
    Uses high-precision float128 for calculations if 'precise' is True.
    """
    print("\n--- Pass 2: Calculating scores ---")
    
    # Determine the float type based on the precision flag
    float_dtype = np.float128 if precise else np.float64
    if precise:
        print("  > Running in HIGH PRECISION mode (numpy.float128).")

    weighted_score_sum = np.zeros(num_samples, dtype=float_dtype)
    denominator_ct = np.zeros(num_samples, dtype=np.int64)
    variants_processed_count = 0
    bytes_per_variant = math.ceil(num_samples / 4)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    geno_to_a2_dosage = np.array([0, -1, 1, 2], dtype=np.int8)
    bim_iter = zip(bim_df['ID'], bim_df['A1'], bim_df['A2'])

    with open(bed_filepath, "rb") as f:
        f.seek(3)
        for variant_id, a1, a2 in bim_iter:
            score_info = None
            if (variant_id, a2) in score_data:
                score_info = {'weight': score_data[(variant_id, a2)], 'effect_is_a2': True}
            elif (variant_id, a1) in score_data:
                score_info = {'weight': score_data[(variant_id, a1)], 'effect_is_a2': False}

            variant_bytes = f.read(bytes_per_variant)
            if not variant_bytes: break

            if score_info:
                variants_processed_count += 1
                byte_array = np.frombuffer(variant_bytes, dtype=np.uint8)
                unpacked_codes = (byte_array[:, np.newaxis] >> shifts) & 0b11
                genotype_codes = unpacked_codes.flatten()[:num_samples]
                dosages_a2 = geno_to_a2_dosage[genotype_codes]
                
                non_missing_mask = dosages_a2 != -1
                if not np.any(non_missing_mask): continue
                
                effect_dosages = 2 - dosages_a2 if not score_info['effect_is_a2'] else dosages_a2
                
                weight = float_dtype(score_info['weight']) # Cast weight to the target precision
                weighted_score_sum[non_missing_mask] += effect_dosages[non_missing_mask] * weight
                denominator_ct[non_missing_mask] += 2

    print(f"  > Processed {variants_processed_count} variants found in score file.")
    return weighted_score_sum, denominator_ct


def main():
    """
    Main execution function.
    """
    parser = argparse.ArgumentParser(
        description="A high-performance Python script to replicate 'plink --score'.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--bfile', required=True, help='Prefix for PLINK .bed, .bim, and .fam fileset.')
    parser.add_argument('--score', required=True, help='Path to the scoring file.')
    parser.add_argument('--out', required=True, help='Prefix for the output file.')
    parser.add_argument(
        '--precise',
        action='store_true',
        help='Switch to maximum precision for calculations and output (numpy.float128).'
    )
    # This preserves the original, required positional arguments for column numbers.
    parser.add_argument('columns', nargs=3, type=int, help="1-based columns for: Variant_ID Effect_Allele Score")
    args = parser.parse_args()

    fam_df = read_fam_file(f"{args.bfile}.fam")
    bim_df = read_bim_file(f"{args.bfile}.bim")
    score_data = read_score_file(args.score, args.columns[0], args.columns[1], args.columns[2])
    
    num_samples = fam_df.height
    bed_filepath = f"{args.bfile}.bed"

    nan_indices = find_ambiguous_samples(bed_filepath, num_samples, bim_df)
    weighted_sum, denom_ct = calculate_scores(bed_filepath, num_samples, bim_df, score_data, precise=args.precise)

    print("\n--- Finalizing results ---")
    final_scores_avg = np.divide(
        weighted_sum, denom_ct,
        out=np.full(num_samples, np.nan, dtype=float),
        where=(denom_ct != 0)
    )

    if nan_indices:
        final_scores_avg[list(nan_indices)] = np.nan
        print(f"  > Applied NaN to {len(nan_indices)} sample(s).")
    
    output_df_pl = pl.DataFrame({
        '#FID': fam_df['FID'], 'IID': fam_df['IID'],
        'NMISS_ALLELE_CT': denom_ct, 'SCORE1_AVG': final_scores_avg
    })

    output_filepath = f"{args.out}.sscore"
    try:
        output_df_pd = output_df_pl.to_pandas()
        
        # Use full precision for output if the flag is set, otherwise use the original format
        float_format = None if args.precise else '%.6g'

        output_df_pd.to_csv(
            output_filepath, sep='\t', index=False,
            float_format=float_format, na_rep='nan'
        )
        print(f"Success! Results written to {output_filepath}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
