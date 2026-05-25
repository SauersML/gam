import re
import shutil
import subprocess
import sys
from pathlib import Path

INPUT = Path("gam_model_fit.rds")
OUTPUT = Path("gam_model_fit.R")
DECIMAL_PLACES = 3
TRUNCATE_THRESHOLD = 300
TRUNCATE_KEEP = 100
FLOAT_PATTERN = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")
NUMBER_PATTERN = r"[-+]?\d+(?:\.\d+)?"
LONG_LIST_PATTERN = re.compile(
    f"((?:{NUMBER_PATTERN}[\\s,]+){{{TRUNCATE_THRESHOLD},}})"
)


def main() -> None:
    if not INPUT.exists():
        sys.stderr.write(f"Error: Input file '{INPUT}' not found in {Path('.').resolve()}\n")
        raise SystemExit(1)

    rscript = shutil.which("Rscript")
    if not rscript:
        sys.stderr.write("Error: Rscript not found on PATH.\n")
        raise SystemExit(2)

    print(f"1. Generating full-precision R code from '{INPUT}'...")
    generate_r_file(rscript)
    print(f"   Successfully generated intermediate file '{OUTPUT}'.")

    print(f"2. Rounding all floating-point numbers to {DECIMAL_PLACES} decimal places...")
    rounded_content = round_floats_in_text(OUTPUT.read_text())
    print("   Rounding complete.")

    print(f"3. Truncating number lists longer than {TRUNCATE_THRESHOLD} elements...")
    final_content, truncated = truncate_long_lists_in_text(rounded_content)
    print(
        f"   Truncated {truncated} long list(s)."
        if truncated
        else "   No lists were long enough to require truncation."
    )

    OUTPUT.write_text(final_content)
    print("\n--- Success! ---")
    print(f"Wrote final, processed R code to '{OUTPUT}'.")


def generate_r_file(rscript: str) -> None:
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
            [rscript, "-e", r_code],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write("Error: Rscript failed to produce dput output.\n")
        sys.stderr.write(f"--- Rscript stderr ---\n{e.stderr}\n")
        raise SystemExit(3)
    if not OUTPUT.exists() or OUTPUT.stat().st_size == 0:
        sys.stderr.write(f"Error: Rscript failed to write '{OUTPUT}'.\n")
        raise SystemExit(4)


def round_floats_in_text(content: str) -> str:
    def round_match(match: re.Match[str]) -> str:
        rounded_num = round(float(match.group(0)), DECIMAL_PLACES)
        return f"{rounded_num:.{DECIMAL_PLACES}f}"

    return FLOAT_PATTERN.sub(round_match, content)


def truncate_long_lists_in_text(content: str) -> tuple[str, int]:
    truncation_count = 0

    def truncate_match(match: re.Match[str]) -> str:
        nonlocal truncation_count

        full_list_str = match.group(1)
        numbers = re.findall(NUMBER_PATTERN, full_list_str)
        if len(numbers) <= TRUNCATE_THRESHOLD:
            return full_list_str

        truncation_count += 1
        return (
            ", ".join(numbers[:TRUNCATE_KEEP])
            + f",\n    ... # [TRUNCATED {len(numbers) - 2 * TRUNCATE_KEEP} ITEMS] ...\n    "
            + ", ".join(numbers[-TRUNCATE_KEEP:])
            + ","
        )

    return LONG_LIST_PATTERN.sub(truncate_match, content), truncation_count


if __name__ == "__main__":
    main()
