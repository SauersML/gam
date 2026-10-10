"""The oracle's questions as a chat dataset (#2951 graph oracle), for an RL stack that takes prompts as text (prime-rl's
verifiers environment) and for its SFT: one JSONL row per question of a split, {"prompt": [user message], "info":
{"task", "split"}}, the user message the question as train.py asks it (prompt.render with VPD's list and where the
prediction responds, with_vpd), and with --answers DIR "completion": [assistant message], the search answer of the
question in DIR as train.py's SFT reads it (train.read_answers; never a held-out search directory) cut to --max-tokens
(train.cut). Questions whose lists are not computed yet are left out.

  questions.py OUT.jsonl --split train [--root TEXTS] [--vpd-list 512] [--responses 32] [--answers DIR] [--max-tokens 4096]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--split", default="train", choices=("train", "heldout", "hard"))
    ap.add_argument("--root", type=Path, default=Path.home() / "mpd-data/graph_oracle/texts")
    ap.add_argument("--model", default="vpd4l")
    ap.add_argument("--vpd-list", type=int, default=512)
    ap.add_argument("--responses", type=int, default=32)
    ap.add_argument("--answers", type=Path)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B", help="counts the answer's tokens for the cut (part tokens counted as their pieces: a cut on the safe side)")
    a = ap.parse_args()
    import train
    from prompt import render, with_vpd

    encode, answers = None, {}
    if a.answers:
        from transformers import AutoTokenizer

        train.refuse_heldout([a.answers])
        answers = train.read_answers(a.answers)
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        encode = lambda t: tok.encode(t, add_special_tokens=False)  # noqa: E731
    lists = [d for d, k in (("vpd_ranked", a.vpd_list), ("vpd_responses", a.responses)) if k]
    rows = 0
    with open(a.out, "w") as out:
        for b in train.behaviors(a.root, a.model, a.split):
            if not all((a.root / d / f"{b['id']}.json").exists() for d in lists):
                continue
            row = {"prompt": [{"role": "user", "content": render(with_vpd(b, a.root, a.vpd_list, a.responses))}], "info": {"task": b["id"], "split": a.split}}
            if a.answers:
                if b["id"] not in answers:
                    continue
                row["completion"] = [{"role": "assistant", "content": train.cut(answers[b["id"]].strip(), a.max_tokens, encode)}]
            out.write(json.dumps(row) + "\n")
            rows += 1
    print(f"{a.out}: {rows} questions of {a.split}" + (f" with answers from {a.answers}" if a.answers else ""))


if __name__ == "__main__":
    main()
