#!/usr/bin/env python3
"""
Calibrate end-to-end benchmark + perf harness (no arguments).

What it does:
- Generates one non-linear dataset via mgcv.py.
- Builds the profiling binary (`cargo build --profile profiling`).
- Trains the GAM via `gnomon train` and profiles with `perf record -g`.
- Produces a single HTML report with a flamegraph (if inferno is installed)
  and a filtered raw-text perf call-graph report under it.

Outputs are written under tests/bench_tools and tests/bench_tools/bench_workdir.
Run simply as:  python3 tests/bench_tools/bench.py
"""

import os
import subprocess
import sys
from pathlib import Path
import shutil as _sh
import webbrowser
from html import escape

import pandas as pd

# Import data-generation utilities from the sibling mgcv.py module
import mgcv  # same directory import


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
# repo root from tests/bench_tools/
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
WORKDIR = SCRIPT_DIR / "bench_workdir"
WORKDIR.mkdir(exist_ok=True)

EXECUTABLE_PATH = WORKSPACE_ROOT / "target" / "profiling" / "gnomon"
# Minimum percent to include in perf text reports.
PERF_PERCENT_LIMIT = 5


def build_profiling_binary():
    env = os.environ.copy()
    # Always rebuild to ensure latest changes are profiled
    print("--- Building profiling binary (symbols kept) ---")
    # Build from workspace root so cargo.toml resolves correctly
    run_or_die(["cargo", "build", "--profile", "profiling"], cwd=WORKSPACE_ROOT, env=env)


def run_or_die(cmd, cwd=None, env=None, stream=True):
    print(f"$ {' '.join(map(str, cmd))}")
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace"
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if stream:
            sys.stdout.write(line)
    ret = proc.wait()
    if ret != 0:
        sys.exit(ret)
    return ret


def run_capture(cmd, cwd=None, env=None):
    """Run a command and capture stdout into a list of lines without exiting on non-zero."""
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace"
    )
    assert proc.stdout is not None
    lines = proc.stdout.read().splitlines()
    ret = proc.wait()
    return ret, lines


def _filter_tree(lines: list[str]) -> list[str]:
    """Filter perf tree output by stanza and drop 0-children stanzas.

    A stanza is a header line with two percentage columns (Overhead, Children)
    followed by its label-only block until the next header. We drop the header
    and its block when Children == 0.00. We also remove address-only and
    '(inlined)' label lines. No name-specific filtering.
    """
    import re as _re
    header_re = _re.compile(r"^\s*(\d+(?:\.\d+)?)%\s+(\d+(?:\.\d+)?)%\s+")
    addr_line_pat = _re.compile(r"0x[0-9A-Fa-f]{6,}(\s+0x[0-9A-Fa-f]{6,})*$")
    addr_token_pat = _re.compile(r"^0x[0-9A-Fa-f]{6,}$")

    def is_addr_only(text: str) -> bool:
        content = text.replace('|', ' ').replace('-', ' ').replace('`', ' ')
        content = ' '.join(content.split())
        if addr_line_pat.match(content):
            return True
        toks = content.split()
        return bool(toks and addr_token_pat.match(toks[-1]))

    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        s = ln.strip()
        # Pass through boilerplate lines
        if not s or s.startswith('#') or 'Percent' in ln or 'Overhead' in ln:
            out.append(ln)
            i += 1
            continue
        m = header_re.match(ln)
        if not m:
            # Not a stanza header: keep unless it's address-only or inlined
            if '(inlined)' not in ln and not is_addr_only(ln):
                out.append(ln)
            i += 1
            continue
        # Header line with Overhead and Children
        try:
            children_pct = float(m.group(2))
        except Exception:
            children_pct = 0.0
        keep = (children_pct != 0.0)
        if keep:
            out.append(ln)
        # Consume label block for this stanza
        i += 1
        while i < n:
            ln2 = lines[i]
            s2 = ln2.strip()
            if not s2 or s2.startswith('#') or 'Percent' in ln2 or 'Overhead' in ln2:
                if keep:
                    out.append(ln2)
                i += 1
                break
            if header_re.match(ln2):
                # Next stanza begins
                break
            if keep and '(inlined)' not in ln2 and not is_addr_only(ln2):
                out.append(ln2)
            i += 1
        # loop continues; do not increment i here as we either broke to next header or already advanced
    return out


def _dedup_label_blocks(lines: list[str]) -> list[str]:
    """Collapse repeated non-percent label blocks that perf prints multiple times.

    A "label block" is a run of lines without the two leading percentage columns.
    We keep the first occurrence and drop subsequent identical blocks until it changes.
    """
    import re as _re
    pct_pat = _re.compile(r"^\s*(\d+(?:\.\d+)?)%\s+(\d+(?:\.\d+)?)%\b")
    out: list[str] = []
    buf: list[str] = []
    last_norm: str | None = None

    def normalize(block_lines: list[str]) -> str:
        # Remove empty lines and collapse consecutive duplicates
        normed: list[str] = []
        for ln in block_lines:
            s = ln.rstrip()
            if not s:
                continue
            if not normed or normed[-1] != s:
                normed.append(s)
        return "\n".join(normed)

    def flush():
        nonlocal buf, last_norm
        if not buf:
            return
        norm = normalize(buf)
        if norm and norm != last_norm:
            out.extend(buf)
            last_norm = norm
        buf = []

    for ln in lines:
        s = ln.strip()
        if not s or s.startswith('#') or 'Percent' in ln or 'Overhead' in ln or ln.startswith('---') or pct_pat.match(ln):
            flush()
            out.append(ln)
            continue
        # Non-percent label line -> part of current block
        buf.append(ln)
    flush()
    return out


def prepare_training_tsv_from_df(df: pd.DataFrame, out_path: Path):
    # Map mgcv columns -> calibrate training schema
    mapping = {
        "variable_two": "PC1",     # penalized PC-like
        "outcome": "phenotype",
    }
    required = set(mapping.keys())
    if not required.issubset(df.columns):
        raise RuntimeError(f"Input DF missing required columns: {sorted(required - set(df.columns))}")
    df2 = df.rename(columns=mapping)
    df2.to_csv(out_path, sep="\t", index=False)


def _run_perf_record(app_cmd: list[str], perf_data: Path, env: dict) -> float:
    """Run perf record with sensible defaults and automatic fallback for mmap limit.
    Returns wall time in seconds. Exits on non-recoverable errors.
    """
    import time as _time
    # Candidate mmap sizes: fixed attempts, then let perf choose
    candidates: list[int | None] = [256, 128, 64, None]
    # Lower frequency to reduce overflow and file size (fixed)
    freq = "700"
    t0 = None
    last_err = None
    for mmap_pages in candidates:
        record = [
            "perf", "record",
            "-e", "cycles:u",
            "--call-graph", "dwarf,16384",
            "-F", freq,
            "-o", str(perf_data),
        ]
        if mmap_pages is not None:
            record += ["-m", str(mmap_pages)]
        record += ["--"] + app_cmd

        print(f"$ {' '.join(map(str, record))}")
        t0 = _time.perf_counter()
        rc, lines = run_capture(record, cwd=WORKSPACE_ROOT, env=env)
        dt = _time.perf_counter() - t0
        # Echo output for transparency
        if lines:
            sys.stdout.write("\n".join(lines) + ("\n" if lines and not lines[-1].endswith("\n") else ""))
        if rc == 0:
            return dt
        text = "\n".join(lines) if lines else ""
        last_err = (rc, text)
        if "Permission error mapping pages" in text or "mmap" in text.lower():
            print("perf record failed due to mmap/lock limits; retrying with smaller -m…")
            continue
        # Non-recoverable error
        sys.exit(rc)
    # Exhausted candidates
    if last_err is not None:
        rc, _ = last_err
        sys.exit(rc)
    return 0.0


def train_with_perf(train_tsv: Path, tag: str):
    # Compose the train command with realistic defaults
    cmd = [
        str(EXECUTABLE_PATH), "train",
        "--num-pcs", "1",
        "--pc-knots", "8", "--pc-degree", "3",
        str(train_tsv),
    ]

    perf_data = WORKDIR / f"perf_{tag}.data"

    # Use fixed threads for stability
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_WAIT_POLICY"] = "PASSIVE"

    print(f"\n=== Training [{tag}] with perf ===")
    # Run perf record with automatic fallback for mmap limits
    dt = _run_perf_record(cmd, perf_data, env)

    # Collect perf report as a call-graph tree (default, always on)
    graph_cmd = [
        "perf", "report", "--stdio", "--call-graph=graph", "--percent-limit", str(PERF_PERCENT_LIMIT),
        "--max-stack", "1024",
        "-i", str(perf_data)
    ]
    rc_graph, graph_lines = run_capture(graph_cmd, cwd=WORKSPACE_ROOT)
    if rc_graph == 0 and graph_lines:
        graph_lines = _filter_tree(graph_lines)
        graph_lines = _dedup_label_blocks(graph_lines)
    merge_ok, merge_text = merged_hot_subpaths(perf_data)
    condensed_ok, condensed_text = condensed_hot_paths(perf_data)

    return {
        "tag": tag,
        "perf_data": perf_data,
        "runtime_sec": dt,
        "graph_ok": (rc_graph == 0),
        "graph": "\n".join(graph_lines),
        "merge_ok": merge_ok,
        "merge": merge_text,
        "condensed_ok": condensed_ok,
        "condensed": condensed_text,
        # No env snapshot in report (no env flags)
    }


def generate_flamegraph(perf_data_path: Path, tag: str) -> Path | None:
    """Generate a flamegraph SVG via inferno toolchain.
    Returns SVG path or None if tools are not installed.
    """
    svg_path = WORKDIR / f"flame_{tag}.svg"
    has_collapse = _sh.which("inferno-collapse-perf") is not None
    has_flame = _sh.which("inferno-flamegraph") is not None
    if has_collapse and has_flame:
        collapsed = WORKDIR / f"collapsed_{tag}.txt"
        shell_cmd = f"perf script -i {perf_data_path} | inferno-collapse-perf > {collapsed} && inferno-flamegraph {collapsed} > {svg_path}"
        rc, _ = run_capture(["bash", "-lc", shell_cmd], cwd=WORKSPACE_ROOT)
        if rc == 0 and svg_path.exists():
            return svg_path
    return None


def merged_hot_subpaths(perf_data_path: Path) -> tuple[bool, str]:
    """Compute merged hot leaf-suffix subpaths from collapsed stacks.

    Uses `perf script | inferno-collapse-perf` to obtain folded stacks of the form
    "frame1;frame2;...;leaf count". We then aggregate counts by the last K frames
    (leafward suffix), merging identical subpaths that occur under different callers.

    No environment flags: uses fixed depth and top limits.
    """
    has_collapse = _sh.which("inferno-collapse-perf") is not None
    if not has_collapse:
        return False, ""
    rc, lines = run_capture([
        "bash", "-lc",
        f"perf script -i {perf_data_path} | inferno-collapse-perf"
    ], cwd=WORKSPACE_ROOT)
    if rc != 0 or not lines:
        return False, ""
    total = 0
    from collections import defaultdict
    agg: dict[str, int] = defaultdict(int)
    depth = 6
    # Parse folded format: frame1;frame2;...;leaf count
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            stack, cnt_str = ln.rsplit(" ", 1)
            cnt = int(cnt_str)
        except Exception:
            continue
        total += cnt
        frames = stack.split(";")
        # Remove raw address-only frames from the suffix
        frames = [f for f in frames if not f.startswith("0x") or not all(c in "0123456789abcdefABCDEFx" for c in f)]
        # Use leafward suffix of length K
        suffix = frames[-depth:]
        key = ";".join(suffix)
        agg[key] += cnt

    if not agg or total <= 0:
        return False, ""
    # Prepare top-N merged suffixes
    top_n = 20
    items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    # Build a compact text block
    out_lines = []
    out_lines.append(f"Merged by leaf-suffix (depth={depth}), top {len(items)} of {len(agg)} unique suffixes")
    out_lines.append("")
    for key, cnt in items:
        pct = 100.0 * cnt / total
        out_lines.append(f"{pct:6.2f}%  {cnt:>8}  {key}")
    return True, "\n".join(out_lines)


def condensed_hot_paths(perf_data_path: Path) -> tuple[bool, str]:
    """Condense hot paths to a human-meaningful spine per stack.

    Rule per stack (frames are root->leaf):
      1) Origin (first frame)
      2) First frame containing 'gnomon::' (if any)
      3) Last frame containing 'gnomon::' (if any, not duplicating first)
      4) First non-gnomon frame immediately after the last gnomon (if any)
      5) A few useful extras after that whose global aggregate counts are high

    Aggregates identical condensed paths and ranks by sample count.
    Avoids raw address-only frames and removes " (inlined)" markers.
    """
    has_collapse = _sh.which("inferno-collapse-perf") is not None
    if not has_collapse:
        return False, ""
    rc, lines = run_capture([
        "bash", "-lc",
        f"perf script -i {perf_data_path} | inferno-collapse-perf"
    ], cwd=WORKSPACE_ROOT)
    if rc != 0 or not lines:
        return False, ""

    from collections import defaultdict
    total = 0
    parsed: list[tuple[list[str], int]] = []
    frame_count: dict[str, int] = defaultdict(int)

    def is_addr(s: str) -> bool:
        s = s.strip()
        return s.startswith("0x") and all(c in "0123456789abcdefABCDEFx" for c in s)

    def norm(s: str) -> str:
        s = s.replace(" (inlined)", "").strip()
        return s

    # Parse and collect global frame counts
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            stack, cnt_str = ln.rsplit(" ", 1)
            cnt = int(cnt_str)
        except Exception:
            continue
        frames = [norm(f) for f in stack.split(";") if not is_addr(f)]
        if not frames:
            continue
        parsed.append((frames, cnt))
        total += cnt
        # Aggregate simple inclusive count per frame name
        for f in set(frames):
            frame_count[f] += cnt

    if not parsed or total <= 0:
        return False, ""

    # Build condensed path per stack
    agg_paths: dict[tuple[str, ...], int] = defaultdict(int)
    EXTRA_LIMIT = 3
    # Set a fixed usefulness threshold at ~2% of total samples
    EXTRA_MIN = max(1, int(0.02 * total))

    for frames, cnt in parsed:
        origin = frames[0]
        g_idx_first = next((i for i, f in enumerate(frames) if "gnomon::" in f), None)
        g_idx_last = None
        if g_idx_first is not None:
            for i, f in enumerate(frames):
                if "gnomon::" in f:
                    g_idx_last = i
        path: list[str] = [origin]
        if g_idx_first is not None:
            if frames[g_idx_first] not in path:
                path.append(frames[g_idx_first])
        if g_idx_last is not None and frames[g_idx_last] not in path:
            path.append(frames[g_idx_last])
        # First non-gnomon after last gnomon
        start_extra = (g_idx_last + 1) if g_idx_last is not None else 1
        first_non_g_after = None
        for j in range(start_extra, len(frames)):
            if "gnomon::" not in frames[j]:
                first_non_g_after = frames[j]
                break
        if first_non_g_after is not None and first_non_g_after not in path:
            path.append(first_non_g_after)

        # Useful extras: subsequent non-duplicate frames with high global weight
        extras_added = 0
        for j in range(start_extra, len(frames)):
            f = frames[j]
            if f in path:
                continue
            if frame_count.get(f, 0) >= EXTRA_MIN:
                path.append(f)
                extras_added += 1
                if extras_added >= EXTRA_LIMIT:
                    break

        agg_paths[tuple(path)] += cnt

    # Format top-N condensed paths
    items = sorted(agg_paths.items(), key=lambda kv: kv[1], reverse=True)[:25]
    lines_out = ["Condensed hot paths (origin → first gnomon → last gnomon → next non-gnomon → extras)", ""]
    for path, cnt in items:
        pct = 100.0 * cnt / total
        lines_out.append(f"{pct:6.2f}%  {cnt:>8}  {' → '.join(path)}")
    return True, "\n".join(lines_out)


def main():
    # 1) Build profiling binary
    build_profiling_binary()
    # No DSO priming needed (no annotate section)

    # 2) Generate a single dataset (default non-linear) to keep report minimal
    datasets = [
        ("nonlinear", False, False, mgcv.generate_data(mgcv.N_SAMPLES_TRAIN, mgcv.NOISE_BLEND_FACTOR, linear_mode=False, noise_mode=False)),
    ]

    # 3) Convert to the Rust training TSV schema and run training under perf
    report_sections = []
    for tag, _linear_mode, _noise_mode, df in datasets:
        train_tsv = SCRIPT_DIR / f"rust_train_{tag}.tsv"
        prepare_training_tsv_from_df(df, train_tsv)
        section = train_with_perf(train_tsv, tag)
        # Generate a flamegraph SVG if possible
        section["flame_svg"] = generate_flamegraph(section["perf_data"], tag)
        report_sections.append(section)

    # 4) Build a simple HTML report and open it automatically
    html_path = WORKDIR / "report.html"
    with html_path.open("w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'>\n")
        f.write("<title>Calibrate Flamegraph</title>\n")
        f.write("<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:18px} h1{margin:0 0 8px} .meta{color:#444;margin:4px 0 16px} .frame{border:1px solid #ddd} .flame svg{width:100%;height:auto;display:block}</style>\n")
        f.write("<h1>Calibrate Flamegraph</h1>\n")
        # only one section
        sec = report_sections[0]
        f.write(f"<div class='meta'>Runtime: {sec['runtime_sec']:.3f} s</div>\n")
        # No environment snapshot (no env flags)
        if sec.get('flame_svg') and sec['flame_svg']:
            # Inline the SVG content so it always renders
            try:
                svg_text = Path(sec['flame_svg']).read_text(encoding='utf-8')
                f.write("<div class='flame'>")
                f.write(svg_text)
                f.write("</div>")
            except Exception as e:
                f.write(f"<p>Could not inline flamegraph SVG: {escape(str(e))}</p>")
        else:
            f.write("<p>Flamegraph not available (missing tooling). Install inferno-collapse-perf and inferno-flamegraph.</p>")

        # Text perf report (call graph tree)
        f.write("<h2 style='margin-top:18px'>Perf Text Report</h2>\n")
        f.write(f"<div class='meta'>Showing entries with ≥{PERF_PERCENT_LIMIT}%</div>\n")
        f.write("<h3>Call Graph</h3>\n")
        if sec.get('graph_ok') and sec.get('graph'):
            f.write("<pre style='white-space:pre-wrap;max-height:500px;overflow:auto;border:1px solid #eee;padding:8px;background:#fafafa'>")
            f.write(escape(sec['graph']))
            f.write("</pre>")
        else:
            f.write("<p>Call-graph report unavailable.</p>")

        # Condensed and merged views to reduce duplication
        f.write("<h3>Condensed Hot Paths</h3>\n")
        if sec.get('condensed_ok') and sec.get('condensed'):
            f.write("<pre style='white-space:pre-wrap;max-height:360px;overflow:auto;border:1px solid #eee;padding:8px;background:#fafafa'>")
            f.write(escape(sec['condensed']))
            f.write("</pre>")
        else:
            f.write("<p>Condensed paths unavailable (install inferno-collapse-perf).</p>")

        # Merged identical leaf-suffix subpaths
        f.write("<h3>Merged Hot Subpaths</h3>\n")
        if sec.get('merge_ok') and sec.get('merge'):
            f.write("<pre style='white-space:pre-wrap;max-height:360px;overflow:auto;border:1px solid #eee;padding:8px;background:#fafafa'>")
            f.write(escape(sec['merge']))
            f.write("</pre>")
        else:
            f.write("<p>Subpath merge unavailable (install inferno-collapse-perf).</p>")

    print(f"\nHTML report -> {html_path}")
    try:
        webbrowser.open(html_path.resolve().as_uri())
        print("Opened report in default browser.")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")

    print("\nAll profiling runs complete.")


if __name__ == "__main__":
    main()
