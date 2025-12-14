import gzip
import json
import os
import sys
import urllib.request
import pandas as pd
from pyliftover import LiftOver

# --- Configuration ---
CHUNK_SIZE = 50000
DATAVERSE_DOI = "doi:10.7910/DVN/7RVV9N"
DATAVERSE_BASE = "https://dataverse.harvard.edu"
TARGET_FILENAME = "Selection_Summary_Statistics_01OCT2025.tsv.gz"
CHAIN_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"
CHAIN_FILENAME = "hg19ToHg38.over.chain.gz"

# --- Helpers ---

def print_progress(current, total, prefix="Progress"):
    """Text-based progress bar."""
    if total:
        percent = 100 * (current / float(total))
        bar = '█' * int(percent // 2) + '-' * (50 - int(percent // 2))
        sys.stdout.write(f"\r{prefix} |{bar}| {percent:.1f}%")
    else:
        sys.stdout.write(f"\r{prefix}: {current / 1024 / 1024:.2f} MB")
    sys.stdout.flush()

def download_with_resume(url, dest_path):
    """Downloads file if missing, with progress bar."""
    if os.path.exists(dest_path):
        print(f"File exists: {dest_path}")
        return

    print(f"Downloading {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response:
        total_size = response.getheader('Content-Length')
        if total_size: total_size = int(total_size)
        
        with open(dest_path, 'wb') as out_file:
            downloaded = 0
            while True:
                buffer = response.read(8192)
                if not buffer: break
                downloaded += len(buffer)
                out_file.write(buffer)
                print_progress(downloaded, total_size, prefix="Downloading")
    print()

def get_dataverse_file():
    """Gets the main selection statistics file."""
    dest = os.path.join(os.getcwd(), TARGET_FILENAME)
    if os.path.exists(dest):
        print(f"Dataset exists: {dest}")
        return dest

    print("Querying Dataverse API...")
    url = f"{DATAVERSE_BASE}/api/datasets/:persistentId/?persistentId={DATAVERSE_DOI}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req) as response:
        metadata = json.load(response)
    
    # Strict lookup, crash if not found
    files = metadata['data']['latestVersion']['files']
    target_data = next(f for f in files if f['dataFile']['filename'] == TARGET_FILENAME)
    file_id = target_data['dataFile']['id']
    
    download_link = f"{DATAVERSE_BASE}/api/access/datafile/{file_id}"
    download_with_resume(download_link, dest)
    return dest

def init_pgs_file(filename, build, direction):
    """Writes PGS header."""
    with open(filename, 'w') as f:
        f.write("#format=PGS_2.0\n")
        f.write(f"#genome_build={build}\n")
        f.write(f"#selection_direction={direction}\n")
        cols = ["rsID", "chr_name", "chr_position", "effect_allele", "other_allele", "effect_weight", "weight_type"]
        f.write("\t".join(cols) + "\n")
    return filename

def write_pgs_chunk(df, f_pos, f_neg):
    """Appends PGS rows."""
    # Prepare standard columns
    subset = pd.DataFrame()
    subset['rsID'] = df['RSID'].fillna('.')
    subset['chr_name'] = df['CHROM']
    subset['chr_position'] = df['POS']
    subset['effect_allele'] = df['ALT']
    subset['other_allele'] = df['REF']
    subset['weight_type'] = 'selection_coefficient_magnitude'
    subset['S_raw'] = df['S']

    # Positive
    pos = subset[subset['S_raw'] > 0].copy()
    if not pos.empty:
        pos['effect_weight'] = pos['S_raw'].abs()
        pos.drop(columns=['S_raw'], inplace=True)
        pos.to_csv(f_pos, sep='\t', index=False, header=False, mode='a')

    # Negative
    neg = subset[subset['S_raw'] < 0].copy()
    if not neg.empty:
        neg['effect_weight'] = neg['S_raw'].abs()
        neg.drop(columns=['S_raw'], inplace=True)
        neg.to_csv(f_neg, sep='\t', index=False, header=False, mode='a')

def lift_row(lo, chrom, pos):
    """
    Lifts a single coordinate using pyliftover.
    Input: 1-based. Chain: 0-based. Output: 1-based.
    """
    # Convert 1-based input to 0-based for UCSC chain
    res = lo.convert_coordinate(chrom, pos - 1)
    if res:
        # Return new Chrom, new Pos (converted back to 1-based)
        return res[0][0].replace('chr', ''), res[0][1] + 1
    return None, None

def process_data(input_file, chain_file):
    print("Initializing LiftOver chain (this loads ~10MB into RAM)...")
    lo = LiftOver(chain_file)
    
    # Output Files
    f_hg19_pos = "ancient_dna_positive_selection_hg19.txt"
    f_hg19_neg = "ancient_dna_negative_selection_hg19.txt"
    f_hg38_pos = "ancient_dna_positive_selection_hg38.txt"
    f_hg38_neg = "ancient_dna_negative_selection_hg38.txt"
    f_hg38_dataset = "Selection_Summary_Statistics_hg38.tsv.gz"
    
    # Initialize Headers
    init_pgs_file(f_hg19_pos, "hg19", "positive")
    init_pgs_file(f_hg19_neg, "hg19", "negative")
    init_pgs_file(f_hg38_pos, "hg38", "positive")
    init_pgs_file(f_hg38_neg, "hg38", "negative")
    
    # Clear dataset if exists
    if os.path.exists(f_hg38_dataset): os.remove(f_hg38_dataset)

    print("Processing Data Streams (hg19 scoring -> LiftOver -> hg38 scoring)...")
    
    # Open all file handles
    with open(f_hg19_pos, 'a') as h19_p, open(f_hg19_neg, 'a') as h19_n, \
         open(f_hg38_pos, 'a') as h38_p, open(f_hg38_neg, 'a') as h38_n, \
         gzip.open(f_hg38_dataset, 'wt') as h38_full:
        
        iterator = pd.read_csv(input_file, sep='\t', compression='gzip', comment='#', chunksize=CHUNK_SIZE)
        
        total_rows = 0
        dataset_header_written = False
        
        for chunk in iterator:
            # 1. Write hg19 PGS
            write_pgs_chunk(chunk, h19_p, h19_n)
            
            # 2. LiftOver to hg38
            # Prepare vectors for map
            chroms = 'chr' + chunk['CHROM'].astype(str).str.lstrip('chr')
            positions = chunk['POS']
            
            # Apply liftover row-by-row
            # Note: pyliftover is fast C++ under the hood usually, but logic is python loop here
            # We use zip to iterate faster
            new_chroms = []
            new_pos = []
            valid_mask = []
            
            for c, p in zip(chroms, positions):
                nc, np = lift_row(lo, c, p)
                if nc:
                    new_chroms.append(nc)
                    new_pos.append(np)
                    valid_mask.append(True)
                else:
                    valid_mask.append(False)
            
            # 3. Create hg38 DataFrame
            # Filter chunk to only rows that mapped
            hg38_chunk = chunk[valid_mask].copy()
            if not hg38_chunk.empty:
                hg38_chunk['CHROM'] = new_chroms
                hg38_chunk['POS'] = new_pos
                
                # 4. Write hg38 Dataset
                if not dataset_header_written:
                    hg38_chunk.to_csv(h38_full, sep='\t', index=False)
                    dataset_header_written = True
                else:
                    hg38_chunk.to_csv(h38_full, sep='\t', index=False, header=False)
                
                # 5. Write hg38 PGS
                write_pgs_chunk(hg38_chunk, h38_p, h38_n)

            total_rows += len(chunk)
            sys.stdout.write(f"\rRows processed: {total_rows:,}")
            sys.stdout.flush()

    print(f"\nCompleted. Generated 4 PGS files and 1 Dataset: {f_hg38_dataset}")

def main():
    # 1. Downloads
    data_file = get_dataverse_file()
    chain_dest = os.path.join(os.getcwd(), CHAIN_FILENAME)
    download_with_resume(CHAIN_URL, chain_dest)
    
    # 2. Process
    process_data(data_file, chain_dest)

if __name__ == "__main__":
    main()
