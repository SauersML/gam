"""Writes gha_inputs.tsv (#2951): sha256, encoding and path (relative to ~/mpd-data) of every published
input under the given prefixes, from rp-run's publish caches (~/mpd-data/runpod/published.tsv and
encodings.tsv), for gha_e2e.sh on GitHub Actions runners. Publish first: bench/runpod/rp-run publish PATH.

  gha_inputs.py engine/vpd4l graph_oracle/behaviors/vpd4l ...
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNPOD = Path.home() / "mpd-data/runpod"


def main() -> None:
    prefixes = sys.argv[1:]
    data = Path.home() / "mpd-data"
    latest = {}
    for line in (RUNPOD / "published.tsv").read_text().splitlines():
        sha, size, mtime, rel = line.split("\t")
        f = data / rel
        # the cache line that matches the file as it is now (size and modification time)
        if any(rel == p or rel.startswith(p.rstrip("/") + "/") for p in prefixes) and f.exists() \
                and str(f.stat().st_size) == size and str(int(f.stat().st_mtime)) == mtime:
            latest[rel] = sha
    encodings = dict(line.split("\t") for line in (RUNPOD / "encodings.tsv").read_text().splitlines() if line)
    rows = [f"{sha}\t{encodings.get(sha, 'raw')}\t{rel}" for rel, sha in sorted(latest.items())]
    old = (HERE / "gha_inputs.tsv").read_text().splitlines() if (HERE / "gha_inputs.tsv").exists() else []
    keep = [r for r in old if r.split("\t")[2] not in latest]
    (HERE / "gha_inputs.tsv").write_text("".join(r + "\n" for r in sorted(keep + rows, key=lambda r: r.split("\t")[2])))
    print(f"{len(rows)} inputs under {prefixes}; {len(keep)} kept from before")


if __name__ == "__main__":
    main()
