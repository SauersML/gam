"""Per-variable counterfactual items (#2951 graph oracle): a behavior's counterfactual should vary each intermediate
variable of its family algorithm (algorithms/), not only the one its builder changed, or the score never asks a
program for the parts that compute the others (induction_random's counterfactual swaps the copied word in both
copies, so the attention pattern is the same on both prompts and the previous-token step is never needed).

What an item varies: the algorithm's variables whose value at the item's target changes from the prompt to its
counterfactual, among those the answer reads (directly or through other variables); "tokens" when none does (the
answer reads a changed token itself, as when induction copies a different word). A variable is varied alone when
every variable it does not feed keeps its value at the target.

New items: for each prompt of one target, each edit of its tokens before the query context (two tokens swapped,
or one token replaced by the token another prompt of the same length has there) is evaluated with the algorithm.
The query context is the longest stretch ending at the target that occurred earlier in the prompt (at least the
target token): it stays as it is, so a variable is varied by what the prompt stored before, not by the query
(induction: the first copy is reordered and the repeat is untouched, so the matched position moves). For each
variable not yet varied by the prompt's own counterfactual, the first edit (in a seeded order) that varies it alone
and changes the answer becomes an item: the edited prompt with the original prompt as its counterfactual (the
checker keys counterfactuals by the prompt's tokens, so the edited text is the new item's prompt), marked
"varies": variable. An item is kept when the model's top token is the answer on both prompts. Families whose
answers are sets of tokens (greater-than) get their items labelled only.

  mem-lease 4 ~/mpd-data/venv/bin/python bench/oracle/graph/behaviors/vary.py BEHAVIOR.json... [--write]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import family  # noqa: E402
import mech  # noqa: E402


def varies(alg, base: dict, other: dict) -> list[str]:
    """The variables the answer reads whose value differs between two evaluations at one target."""
    return [v for v in sorted(alg.upstream["answer"]) if base[v] != other[v]] or ["tokens"]


def alone(alg, v: str, base: dict, other: dict) -> bool:
    """v varies, and so does the answer, while every variable v does not feed keeps its value."""
    fed = {u for u in alg.names if v in alg.upstream[u]}
    return base[v] != other[v] and base["answer"] != other["answer"] and other["answer"] is not None and all(
        base[u] == other[u] for u in alg.upstream["answer"] if u != v and u not in fed)


_VOCABULARIES: dict = {}


def token_id(model: str, text: str) -> int | None:
    """The token an answer `text` is (the lowest id that decodes to it), else the first token it splits into."""
    tk = mech.tokenizer(model)
    if model not in _VOCABULARIES:
        vocabulary: dict[str, int] = {}
        for i, s in enumerate(tk.decode_batch([[i] for i in range(tk.get_vocab_size())], skip_special_tokens=False)):
            vocabulary.setdefault(s, i)
        _VOCABULARIES[model] = vocabulary
    if text in _VOCABULARIES[model]:
        return _VOCABULARIES[model][text]
    ids = tk.encode(text, add_special_tokens=False).ids if text else []
    return ids[0] if ids else None


def context(ids: list[int], lo: int, t: int) -> int:
    """The first position of the query context: the longest stretch ending at t that also starts earlier."""
    for n in range(t - lo, 0, -1):
        tail = ids[t - n + 1: t + 1]
        if any(ids[j: j + n] == tail for j in range(lo, t - n + 1)):
            return t - n + 1
    return t


def edits(ids: list[int], t: int, pool: list[list[int]], rng: random.Random):
    """Token edits before the query context of target t, seeded order: swaps of two positions and replacements from
    same-length prompts."""
    lo = 1 if ids and ids[0] == 0 else 0  # vpd4l's document separator stays
    end = context(ids, lo, t)
    out = [("swap", i, k) for i in range(lo, end) for k in range(i + 1, end) if ids[i] != ids[k]]
    out += [("put", i, other[i]) for other in pool for i in range(lo, end) if other[i] != ids[i]]
    rng.shuffle(out)
    for e in out:
        new = list(ids)
        if e[0] == "swap":
            new[e[1]], new[e[2]] = new[e[2]], new[e[1]]
        else:
            new[e[1]] = e[2]
        yield e, new


def vary(behavior: dict, rng: random.Random, tries: int = 4000) -> tuple[list[dict], dict]:
    """(new items, counts): items varying each variable the prompts' own counterfactuals leave fixed; labels every
    existing item's "varies" in place."""
    model = behavior["model"]
    alg = family.Algorithm(behavior["family"])
    tk = mech.tokenizer(model)
    strings = {}

    def text_of(ids: list[int]) -> list[str]:
        missing = [i for i in ids if i not in strings]
        strings.update(zip(missing, tk.decode_batch([[i] for i in missing], skip_special_tokens=True)))
        return [strings[i] for i in ids]

    prompts = behavior["prompts"]
    texts = {q["text"] for p in prompts for q in (p, p.get("counterfactual") or p)}  # the checker keys by tokens
    counts: dict[str, int] = {}
    new = []
    for k, p in enumerate(prompts):
        t = p["target_positions"][0]
        base = alg.at(text_of(p["token_ids"]), t)
        if p.get("counterfactual"):
            p["varies"] = varies(alg, base, alg.at(text_of(p["counterfactual"]["token_ids"]), t))
            for v in p["varies"]:
                counts[v] = counts.get(v, 0) + 1
        if len(p["target_positions"]) != 1 or not isinstance(base["answer"], str):
            continue  # set-valued or multi-token answers: labelled only
        wanted = [v for v in sorted(alg.upstream["answer"]) if v not in p.get("varies", [])]
        pool = [q["token_ids"] for q in prompts if q is not p and len(q["token_ids"]) == len(p["token_ids"])][:16]
        for n, (e, ids) in enumerate(edits(p["token_ids"], t, pool, rng)):
            if not wanted or n >= tries:
                break
            other = alg.at(text_of(ids), t)
            for v in [v for v in wanted if alone(alg, v, base, other)]:
                answer_id = token_id(model, other["answer"])
                if answer_id is None:
                    continue
                made = ids[: t + 1] + [answer_id] + ids[t + 2:]
                text = "".join(text_of(made))
                if text in texts:
                    continue
                texts.add(text)
                wanted.remove(v)
                new.append({"text": text, "token_ids": made, "target_positions": [t], "answer": other["answer"],
                            "varies": [v], "edit": list(e), "of": k,
                            "counterfactual": {"text": p["text"], "token_ids": list(p["token_ids"]), "answer": p["answer"]}})
                break
    return new, counts


def checked(behavior: dict, new: list[dict], device: str) -> list[dict]:
    """The new items whose prompt and counterfactual both get the answer as the model's top token (model_top and
    correct filled as build.py does)."""
    import build

    tok = build.Tok(behavior["model"])
    model = build.Model(behavior["model"], device)
    seqs, pos, want = build.readout_requests(new)
    build.score(tok, new, model.readout(seqs, pos, want))
    return [p for p in new if all(p["correct"]) and all(p["counterfactual"]["correct"])]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("behaviors", nargs="+", type=Path)
    ap.add_argument("--seed", type=int, default=2951)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--write", action="store_true", help="append the kept items to the behavior files")
    a = ap.parse_args()
    for path in a.behaviors:
        behavior = json.loads(path.read_text())
        old = [p for p in behavior["prompts"] if "of" not in p]  # rerunning replaces earlier generated items
        behavior["prompts"] = old
        new, counts = vary(behavior, random.Random(f"{a.seed}:{behavior['id']}"))
        kept = checked(behavior, new, a.device) if new else []
        made: dict[str, list[int]] = {}
        for p in new:
            made.setdefault(p["varies"][0], [0, 0])[0] += 1
        for p in kept:
            made[p["varies"][0]][1] += 1
        print(f"{behavior['id']}: own counterfactuals vary {counts}; new items made/kept per variable "
              f"{ {v: tuple(c) for v, c in made.items()} }", flush=True)
        if a.write:
            behavior["prompts"] = old + kept
            clean = [c for p in behavior["prompts"] for c in p.get("correct", [])]
            cf = [c for p in behavior["prompts"] for c in (p.get("counterfactual") or {}).get("correct", [])]
            behavior["model_accuracy"] = round(sum(clean) / len(clean), 4) if clean else None
            behavior["counterfactual_accuracy"] = round(sum(cf) / len(cf), 4) if cf else None
            behavior["targets"] = len(clean)
            behavior["varies"] = {}
            for p in behavior["prompts"]:
                for v in p.get("varies", []):
                    behavior["varies"][v] = behavior["varies"].get(v, 0) + 1
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(behavior))
            tmp.replace(path)


if __name__ == "__main__":
    main()
