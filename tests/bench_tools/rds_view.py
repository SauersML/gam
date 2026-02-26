from pathlib import Path
import shutil
import subprocess
import sys
import re

# --- Configuration ---
INPUT = Path("gam_model_fit.rds")
OUTPUT = Path("gam_model_fit.R")
DECIMAL_PLACES = 3
TRUNCATE_THRESHOLD = 300  # Truncate lists longer than this
TRUNCATE_KEEP = 100       # Keep this many items at the start and end

def main():
    """Converts an RDS file to a rounded and truncated R script."""
    # Stage: Pre-flight checks for the input file and Rscript
    if not INPUT.exists():
        sys.stderr.write(f"Error: Input file '{INPUT}' not found in {Path('.').resolve()}\n")
        sys.exit(1)

    rscript = shutil.which("Rscript")
    if not rscript:
        sys.stderr.write(
            "Error: Rscript not found. Please install R (https://cran.r-project.org/) "
            "and ensure `Rscript` is on your PATH.\n"
        )
        sys.exit(2)

    # Stage: Generate the full-precision R file using Rscript.
    print(f"1. Generating full-precision R code from '{INPUT}'...")
    generate_r_file()
    print(f"   Successfully generated intermediate file '{OUTPUT}'.")

    # Stage: Read the generated file and round all floating-point numbers.
    print(f"2. Rounding all floating-point numbers to {DECIMAL_PLACES} decimal places...")
    content = OUTPUT.read_text()
    rounded_content = round_floats_in_text(content)
    print("   Rounding complete.")

    # Stage: Find and truncate very long lists of numbers in the rounded content.
    print(f"3. Truncating number lists longer than {TRUNCATE_THRESHOLD} elements...")
    final_content, num_truncated = truncate_long_lists_in_text(rounded_content)
    if num_truncated > 0:
        print(f"   Truncated {num_truncated} long list(s).")
    else:
        print("   No lists were long enough to require truncation.")

    # Final Step: Write the processed content back to the file.
    OUTPUT.write_text(final_content)

    print("\n--- Success! ---")
    print(f"Wrote final, processed R code to '{OUTPUT}'.")
    print("You can open it in a text editor or run `source('gam_model_fit.R')` in R.")


def generate_r_file():
    """Calls Rscript to convert the RDS file to a .R file."""
    r_code = f"""
options(digits = 17, width = 10000)
obj <- readRDS("{INPUT}")
sink("{OUTPUT}")
cat("gam_model_fit <- ")
dput(obj)
cat("\\n")
sink()
"""
    try:
        subprocess.run(
            [shutil.which("Rscript"), "-e", r_code],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write("Error: Rscript failed to produce dput output.\n")
        sys.stderr.write(f"--- Rscript stderr ---\n{e.stderr}\n")
        sys.exit(3)
    if not OUTPUT.exists() or OUTPUT.stat().st_size == 0:
        sys.stderr.write(f"Error: Rscript failed to write '{OUTPUT}'.\n")
        sys.exit(4)

def round_floats_in_text(content: str) -> str:
    """Finds all float-like numbers in a string and rounds them."""
    def round_match(match: re.Match) -> str:
        number_str = match.group(0)
        rounded_num = round(float(number_str), DECIMAL_PLACES)
        return f"{rounded_num:.{DECIMAL_PLACES}f}"

    # This regex is broad to catch scientific notation and simple floats
    float_pattern = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")
    return float_pattern.sub(round_match, content)

def truncate_long_lists_in_text(content: str) -> tuple[str, int]:
    """Finds and truncates long, comma-separated lists of pure numbers."""
    number_pat = r"[-+]?\d+(?:\.\d+)?"  # Matches an integer or a float

    # This pattern looks for a block of text that consists of a number followed
    # by a comma and/or whitespace, repeated at least TRUNCATE_THRESHOLD times.
    # The non-capturing group (?:...) is used for the repeating element.
    # The outer group (...) captures the entire list for processing.
    long_list_pattern = re.compile(
        f"((?:{number_pat}[\\s,]+){{{TRUNCATE_THRESHOLD},}})"
    )

    truncation_count = 0

    def truncate_match(match: re.Match) -> str:
        """Callback function to process and truncate a matched list."""
        nonlocal truncation_count
        
        full_list_str = match.group(1)
        # Extract all numbers from the matched block
        numbers = re.findall(number_pat, full_list_str)

        # Double-check length, though the regex should already ensure it
        if len(numbers) <= TRUNCATE_THRESHOLD:
            return full_list_str # Should not happen, but safe to have

        truncation_count += 1
        
        first_part = numbers[:TRUNCATE_KEEP]
        last_part = numbers[-TRUNCATE_KEEP:]

        # Reconstruct the list with a truncation marker
        # The marker is an R comment to maintain syntactical validity.
        return (
            ", ".join(first_part)
            + f",\n    ... # [TRUNCATED {len(numbers) - 2 * TRUNCATE_KEEP} ITEMS] ...\n    "
            + ", ".join(last_part)
            + "," # Add a trailing comma for syntactical safety
        )

    processed_content = long_list_pattern.sub(truncate_match, content)
    return processed_content, truncation_count

if __name__ == "__main__":
    main()
