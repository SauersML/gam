"""Re-parse the profile and save per-event metadata + per-token offsets,
so we can correlate SAE activations with wall-clock time and detect
sub-benchmark boundaries.

Saves events_meta.npz with:
  thread_id (E,)        - thread of each event
  start_time_ms (E,)    - wall-clock time of event start (ms)
  dwell_ms (E,)         - same as the dwell encoded in the dwell-bucket token
  stack_len (E,)        - number of frames in this event's stack
  token_start (E,)      - index into stream.tokens where this event's <BOS> sits
  token_end (E,)        - index of the dwell token (last in the event)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from data import N_DWELL, N_RESERVED, load_profile, normalize_frame, _string_array


def main(profile_path: Path, tok_dir: Path):
    profile = load_profile(profile_path)
    vocab = json.loads((tok_dir / "vocab.json").read_text())
    unk_id = vocab["<UNK>"]
    bos_id = vocab["<BOS>"]
    eos_id = vocab["<EOS>"]
    sep_id = vocab["<SEP>"]
    DWELL_BASE = N_RESERVED

    # Re-derive event order EXACTLY the way data.py does (per-thread, in
    # thread order; events appended sequentially) so token offsets match.
    thread_id_list: list[int] = []
    start_time_list: list[float] = []
    dwell_list: list[float] = []
    stack_len_list: list[int] = []
    tok_start_list: list[int] = []
    tok_end_list: list[int] = []

    cumulative_token = 0  # mirrors encode_events output position
    last_tid: int | None = None

    for tid, thread in enumerate(profile["threads"]):
        sa = _string_array(thread)
        ft = thread["frameTable"]
        st = thread["stackTable"]
        fn = thread["funcTable"]
        samples = thread["samples"]

        frame_func = ft["func"]
        func_name = fn["name"]

        def frame_name(frame_id: int) -> str:
            func_id = frame_func[frame_id]
            name_id = func_name[func_id]
            raw = sa[name_id] if name_id is not None else "<unknown>"
            return normalize_frame(raw)

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
            chain.reverse()
            stack_cache[stack_id] = chain
            return chain

        sample_stacks = samples["stack"]
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

        i = 0
        while i < n:
            sid = sample_stacks[i]
            if sid is None:
                i += 1
                continue
            j = i + 1
            while j < n and sample_stacks[j] == sid:
                j += 1
            t_start = float(sample_times[i])
            t_end = float(sample_times[j] if j < n else sample_times[-1])
            dwell = max(t_end - t_start, 1e-3)
            chain = resolve_stack(sid)
            if chain:
                # Mirror encode_events emission: optional <SEP>, then <BOS>+chain+<EOS>+<DWELL_*>
                if last_tid is not None and tid != last_tid:
                    cumulative_token += 1  # <SEP>
                last_tid = tid
                tok_start = cumulative_token  # position of <BOS>
                # <BOS> + len(chain) frames + <EOS> + <DWELL_*>
                event_len = 1 + len(chain) + 1 + 1
                tok_end = cumulative_token + event_len - 1
                cumulative_token += event_len

                thread_id_list.append(tid)
                start_time_list.append(t_start)
                dwell_list.append(dwell)
                stack_len_list.append(len(chain))
                tok_start_list.append(tok_start)
                tok_end_list.append(tok_end)
            i = j

    out = tok_dir / "events_meta.npz"
    np.savez(
        out,
        thread_id=np.array(thread_id_list, dtype=np.int32),
        start_time_ms=np.array(start_time_list, dtype=np.float64),
        dwell_ms=np.array(dwell_list, dtype=np.float64),
        stack_len=np.array(stack_len_list, dtype=np.int32),
        token_start=np.array(tok_start_list, dtype=np.int64),
        token_end=np.array(tok_end_list, dtype=np.int64),
    )

    # Sanity: stream length should match cumulative_token (last event has no <SEP> after).
    stream = np.load(tok_dir / "stream.npz")
    expected = cumulative_token
    actual = int(stream["tokens"].size)
    print(f"events: {len(thread_id_list)}  cumulative_tokens(expected): {expected}  stream.tokens: {actual}")
    print(f"saved {out}")


if __name__ == "__main__":
    profile_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/profile.json")
    tok_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/tok")
    main(profile_path, tok_dir)
