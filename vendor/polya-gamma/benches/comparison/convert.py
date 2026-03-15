import re
import sys

import pandas as pd

if len(sys.argv) < 2:
    print("❌ Usage: python convert.py <RESULTS_DIR>")
    sys.exit(1)

RESULTS_DIR = sys.argv[1]
RUST_CSV = f"{RESULTS_DIR}/rust_bench_output.csv"
bench_file = f"{RESULTS_DIR}/rust_bench_output.txt"

# Parse the benchmark output
with open(bench_file) as f:
    lines = f.readlines()

data = []
for line in lines:
    line = line.strip()
    if line.startswith("test draw_n") and "bench:" in line:
        match = re.match(
            r"test (\w+_n\d+_b[\d\.]+_c[\d\.]+)\s+\.\.\. bench:\s+([\d,]+) ns/iter \(\+/- [\d,]+\)",
            line,
        )
        if match:
            name = match.group(1)
            time_ns = float(match.group(2).replace(",", ""))

            params = {}
            for part in name.split("_"):
                if part.startswith("n"):
                    params["n"] = int(part[1:])
                elif part.startswith("b"):
                    params["b"] = float(part[1:])
                elif part.startswith("c"):
                    params["c"] = float(part[1:])

            data.append(
                {
                    "n": params["n"],
                    "b": params["b"],
                    "c": params["c"],
                    "time_ns": time_ns,
                    "time_us": time_ns / 1000,
                }
            )

if data:
    df = pd.DataFrame(data)
    df.to_csv(RUST_CSV, index=False)
    print(f"✅ Successfully saved {len(df)} benchmark results to {RUST_CSV}")
else:
    print("❌ No benchmark data found in the output file")
    print("\nOutput file contents:")
    with open(bench_file) as f:
        print(f.read())
