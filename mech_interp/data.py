"""Samply (Firefox profile) JSON -> tokenized event stream.

Pipeline:
  1. Load samply profile (json or json.gz).
  2. For each thread, walk stackTable to materialize each sample as a list of
     frame names (root -> leaf).
  3. Strip Rust generics, normalize.
  4. Run-length-encode consecutive identical stacks per thread to get
     (stack, dwell_ms) events.
  5. Quantile-bucket log(dwell_ms) into N_DWELL buckets.
  6. Build frame vocab (top-K by dwell-weighted frequency).
  7. Emit interleaved token stream:
        <BOS> f0 f1 ... fL <EOS> <DWELL_b> <BOS> ... <SEP> ...

Saves: vocab.json, events.npz (token_ids: int32[N], dwell_id: int32[E],
       stack_lens: int32[E], thread_ids: int32[E]).
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------- vocab ----------

# Reserved token IDs (must be contiguous from 0).
TOK_PAD = 0
TOK_BOS = 1
TOK_EOS = 2
TOK_SEP = 3   # between threads / between profiles
TOK_UNK = 4
N_RESERVED = 5
N_DWELL = 16  # number of dwell-time buckets
# Dwell tokens occupy IDs [N_RESERVED, N_RESERVED + N_DWELL).
# Frame tokens occupy IDs [N_RESERVED + N_DWELL, ...).

DWELL_BASE = N_RESERVED
FRAME_BASE = N_RESERVED + N_DWELL


# ---------- frame normalization ----------

# Collapse Rust generic angle-bracket parameters: `Vec<f64, A>::push` -> `Vec::push`.
# Only strip `<...>` immediately following an identifier (i.e. generic args),
# NOT bare `<Type as Trait>` qualifiers, which preserve dispatch info we want.
_GENERIC_RE = re.compile(r"(?<=[\w_])<[^<>]*>")


def normalize_frame(name: str) -> str:
    if not name:
        return "<unknown>"
    s = name
    for _ in range(8):  # iterate to handle nested generics
        new = _GENERIC_RE.sub("", s)
        if new == s:
            break
        s = new
    # Strip trailing hash suffix like ::h1234abcd from un-demangled symbols.
    s = re.sub(r"::h[0-9a-f]{8,}$", "", s)
    # Strip closure suffixes that bloat vocab without changing semantics much.
    s = re.sub(r"::\{\{closure\}\}(::\d+)?", "::{closure}", s)
    return s.strip() or "<unknown>"


# ---------- profile loading ----------

def load_profile(path: Path) -> dict:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def _string_array(thread: dict) -> list[str]:
    # Firefox profile format: either stringArray (samply) or stringTable.
    if "stringArray" in thread:
        return thread["stringArray"]
    st = thread["stringTable"]
    if isinstance(st, list):
        return st
    # Some variants store as {"strings": [...]}.
    return st.get("strings", st)


@dataclass
class Event:
    thread_id: int
    frames: list[str]   # root -> leaf
    dwell_ms: float


def materialize_events(profile: dict) -> list[Event]:
    events: list[Event] = []
    for tid, thread in enumerate(profile["threads"]):
        sa = _string_array(thread)
        ft = thread["frameTable"]
        st = thread["stackTable"]
        fn = thread["funcTable"]
        samples = thread["samples"]

        # Resolve frame_id -> demangled normalized name.
        # frameTable.func -> funcTable.name -> stringArray
        frame_func = ft["func"]
        func_name = fn["name"]

        def frame_name(frame_id: int) -> str:
            func_id = frame_func[frame_id]
            name_id = func_name[func_id]
            raw = sa[name_id] if name_id is not None else "<unknown>"
            return normalize_frame(raw)

        # stackTable: prefix (parent stack or None), frame.
        stack_prefix = st["prefix"]
        stack_frame = st["frame"]

        stack_cache: dict[int, list[str]] = {}

        def resolve_stack(stack_id: int | None) -> list[str]:
            if stack_id is None:
                return []
            if stack_id in stack_cache:
                return stack_cache[stack_id]
            chain: list[str] = []
            cur = stack_id
            seen = set()
            while cur is not None and cur not in seen:
                seen.add(cur)
                chain.append(frame_name(stack_frame[cur]))
                cur = stack_prefix[cur]
            chain.reverse()  # root -> leaf
            stack_cache[stack_id] = chain
            return chain

        sample_stacks = samples["stack"]
        # samply format uses timeDeltas (ms since previous sample); fall back to absolute time.
        if "time" in samples:
            sample_times = samples["time"]
        else:
            deltas = samples["timeDeltas"]
            cum = 0.0
            sample_times = []
            for d in deltas:
                cum += d
                sample_times.append(cum)
        n = len(sample_stacks)
        if n < 2:
            continue

        # RLE consecutive identical stack_ids into (stack_id, dwell_ms).
        # Dwell of an event = time of next-different-sample - time of first-of-run.
        i = 0
        while i < n:
            sid = sample_stacks[i]
            if sid is None:
                i += 1
                continue
            j = i + 1
            while j < n and sample_stacks[j] == sid:
                j += 1
            t_start = sample_times[i]
            t_end = sample_times[j] if j < n else sample_times[-1]
            dwell = max(t_end - t_start, 1e-3)  # at least 1us so log is finite
            chain = resolve_stack(sid)
            if chain:
                events.append(Event(thread_id=tid, frames=chain, dwell_ms=dwell))
            i = j
    return events


# ---------- vocab construction ----------

def build_vocab(events: list[Event], max_frames: int = 4096) -> dict[str, int]:
    """Top-K frames by dwell-weighted count get a real ID, rest -> UNK."""
    weights: dict[str, float] = {}
    for ev in events:
        # dwell-weighted: contributes proportional to how much time the frame
        # was on the call chain. So heavy code paths dominate vocab.
        for f in ev.frames:
            weights[f] = weights.get(f, 0.0) + ev.dwell_ms
    most = sorted(weights.items(), key=lambda kv: -kv[1])[:max_frames]
    vocab = {
        "<PAD>": TOK_PAD,
        "<BOS>": TOK_BOS,
        "<EOS>": TOK_EOS,
        "<SEP>": TOK_SEP,
        "<UNK>": TOK_UNK,
    }
    for k in range(N_DWELL):
        vocab[f"<DWELL_{k}>"] = DWELL_BASE + k
    for i, (frame, _w) in enumerate(most):
        vocab[frame] = FRAME_BASE + i
    return vocab


def dwell_buckets(events: list[Event], n_buckets: int = N_DWELL) -> np.ndarray:
    """Quantile-based bucket edges for log(dwell_ms). Returns edges (n_buckets-1)."""
    log_dwells = np.log(np.array([e.dwell_ms for e in events]) + 1e-6)
    qs = np.linspace(0, 1, n_buckets + 1)[1:-1]
    edges = np.quantile(log_dwells, qs)
    return edges


def encode_dwell(dwell_ms: float, edges: np.ndarray) -> int:
    return int(np.searchsorted(edges, np.log(dwell_ms + 1e-6)))


# ---------- token stream emission ----------

def encode_events(
    events: list[Event],
    vocab: dict[str, int],
    edges: np.ndarray,
) -> np.ndarray:
    """Emit interleaved token sequence: <BOS> frames... <EOS> <DWELL_k> ...
    Threads separated by <SEP>."""
    out: list[int] = []
    last_tid: int | None = None
    unk = vocab["<UNK>"]
    for ev in events:
        if last_tid is not None and ev.thread_id != last_tid:
            out.append(vocab["<SEP>"])
        last_tid = ev.thread_id
        out.append(vocab["<BOS>"])
        for frame in ev.frames:
            out.append(vocab.get(frame, unk))
        out.append(vocab["<EOS>"])
        b = encode_dwell(ev.dwell_ms, edges)
        out.append(DWELL_BASE + b)
    return np.array(out, dtype=np.int32)


# ---------- full pipeline ----------

def process_profile(
    profile_path: Path,
    out_dir: Path,
    max_frames: int = 4096,
) -> dict:
    profile = load_profile(profile_path)
    events = materialize_events(profile)
    if not events:
        raise RuntimeError(f"no events parsed from {profile_path}")
    vocab = build_vocab(events, max_frames=max_frames)
    edges = dwell_buckets(events)
    tokens = encode_events(events, vocab, edges)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=1)
    np.savez(
        out_dir / "stream.npz",
        tokens=tokens,
        dwell_edges=edges,
        n_events=np.int64(len(events)),
    )
    summary = {
        "n_events": len(events),
        "n_tokens": int(tokens.size),
        "vocab_size": len(vocab),
        "n_threads": len({e.thread_id for e in events}),
        "total_dwell_ms": float(sum(e.dwell_ms for e in events)),
        "median_stack_depth": float(np.median([len(e.frames) for e in events])),
        "p99_stack_depth": float(np.percentile([len(e.frames) for e in events], 99)),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import sys
    in_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else in_path.parent / "tokenized"
    s = process_profile(in_path, out_dir)
    print(json.dumps(s, indent=2))
