"""Tokenize the subject-verb agreement pairs of Marks et al. (2025) (the feature-circuits repo's
data: simple, rc, within_rc, nounpp) once for the Rust circuit curves (#2951,
gam_mpd::explanation_battery::circuit_curves). Tokenization needs the model's own tokenizer, so it
runs here; the curves are computed in Rust.

Per task, `TRAIN` and `TEST` pairs drawn uniformly without replacement (seeds 0 and 1) among those
whose clean and counterfactual prefixes have equal token counts and whose verbs are one token
each: {clean, counterfactual, target, foil}, `target` the clean prefix's verb and `foil` the
counterfactual's.

usage: sva_export.py DATA_DIR TOKENIZER.json OUT.json [TRAIN TEST]   (default 300 and 40)
"""

import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

TASKS = ("simple", "rc", "within_rc", "nounpp")


def pairs(tokenizer: Tokenizer, path: Path, n: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in open(path)]
    out = []
    for i in np.random.default_rng(seed).permutation(len(rows)):
        r = rows[i]
        a, b = tokenizer.encode(r["clean_prefix"]).ids, tokenizer.encode(r["patch_prefix"]).ids
        y, y2 = tokenizer.encode(r["clean_answer"]).ids, tokenizer.encode(r["patch_answer"]).ids
        if len(a) == len(b) and len(y) == 1 and len(y2) == 1:
            out.append({"clean": a, "counterfactual": b, "target": y[0], "foil": y2[0]})
            if len(out) == n:
                break
    return out


def main():
    data, tokenizer_path, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    n_train, n_test = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (300, 40)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tasks = {t: {"train": pairs(tokenizer, data / f"{t}_train.json", n_train, 0), "test": pairs(tokenizer, data / f"{t}_test.json", n_test, 1)} for t in TASKS}
    json.dump({"source": str(data), "tokenizer": tokenizer_path, "tasks": tasks}, open(out, "w"))
    print(out, {t: (len(v["train"]), len(v["test"])) for t, v in tasks.items()})


if __name__ == "__main__":
    main()
