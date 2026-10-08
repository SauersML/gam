"""The reader term in its discrete form (#2951): a frozen reader LM sees only an answer's English explanation
and answers checkable questions about the target model M; the score is how many bits the explanation saves
the reader per question.

Questions (one behavior; ground truths measured on M or computed by the family algorithm):
  next     for a prompt of the behavior, which of 4 tokens is M's most probable next token: M's top token
           (the behavior file's model_top) and the top tokens of 3 other prompts of the behavior, in a seeded
           random order;
  switch   with the answer's parts switched to their counterfactual values and every other part computing
           on the prompt (the checker's clean complement run, mpd_graph_2951 op "complement"), whether M
           gives the counterfactual's top token a higher probability than the prompt's top token; asked at
           target tokens where the two top tokens differ;
  step     whether an intermediate variable of the family algorithm (a function other than `answer`) has
           a different value at the target token on two texts: a prompt and its counterfactual, or two
           prompts of the behavior.
Score: per question the reader's log-loss -log2 q(true option), q normalized over the options' first
tokens at the start of the reader's reply. bits saved = log-loss without an explanation - log-loss with
it, on the same questions. Control: each behavior's questions read with another behavior's explanation
(a seeded derangement of the behaviors), which must save nothing.

  reader_questions.py build --manifest MANIFEST.jsonl --out QUESTIONS.jsonl [--per-type 16]
  reader_questions.py score --questions QUESTIONS.jsonl --out RESULT.json [--model Qwen/Qwen3-8B --device mps]
build runs the checker (score.Checker, GRAPH_CHECKER) for the switch questions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LN2 = math.log(2.0)
LETTERS = "ABCD"
SYSTEM = "You answer questions about a language model, the target model, from an explanation of how it produces one of its behaviours."
EXPLAINED = "An explanation of one behaviour of the target model (between <<< and >>>):\n<<<{explanation}>>>\n\n"
UNEXPLAINED = "No explanation of the target model is given.\n\n"


def q_text(q: dict) -> str:
    """The question as the reader reads it, ending with how to answer."""
    t = json.dumps
    if q["type"] == "next":
        listing = "\n".join(f"{LETTERS[j]}. {t(o, ensure_ascii=False)}" for j, o in enumerate(q["options"]))
        return (f"The target model reads this text (between <<< and >>>):\n<<<{q['text']}>>>\n\nWhich of these tokens "
                f"(each a JSON string) is its most probable next token?\n{listing}\n\nAnswer with the letter.")
    if q["type"] == "switch":
        return (f"The target model reads the text <<<{q['text']}>>>. The parts of the model that the explanation names are "
                f"set to the values they take when the model reads <<<{q['counterfactual']}>>>, and every other part keeps "
                f"computing on the first text. Does the model then give the next token {t(q['counter_token'], ensure_ascii=False)} "
                f"a higher probability than {t(q['clean_token'], ensure_ascii=False)}?\n\nAnswer Yes or No.")
    if q["type"] == "step":
        return (f"In the explanation's account of the behaviour, does the value of {q['variable']} at the last token differ "
                f"between the text <<<{q['text']}>>> and the text <<<{q['other']}>>>?\n\nAnswer Yes or No.")
    raise ValueError(q["type"])


def option_words(q: dict) -> list[str]:
    return list(LETTERS[: len(q["options"])]) if q["type"] == "next" else ["Yes", "No"]


class Encoder:
    """Token ids of the reader's prompt: prefix(explanation) is shared by a behavior's questions, then the
    question and the start of the reader's reply (the answer slot); the options' first tokens."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        marker = "\u0000U\u0000"
        rendered = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": marker}],
                                                 add_generation_prompt=True, enable_thinking=False, tokenize=False)
        self.head, self.mid = rendered.split(marker)

    def enc(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False, split_special_tokens=True)

    def prefix(self, explanation: str | None) -> list[int]:
        body = EXPLAINED.replace("{explanation}", explanation) if explanation else UNEXPLAINED
        return self.tok.encode(self.head, add_special_tokens=False) + self.enc(body)

    def question(self, q: dict) -> list[int]:
        return self.enc(q_text(q)) + self.tok.encode(self.mid, add_special_tokens=False)

    def options(self, q: dict) -> list[int]:
        ids = [self.enc(w)[0] for w in option_words(q)]
        assert len(set(ids)) == len(ids), f"options share a first token: {option_words(q)}"
        return ids


def log_loss(backend, enc: Encoder, explanation: str | None, questions: list[dict]) -> np.ndarray:
    """Bits per question: -log2 of the reader's probability of the true option, normalized over the options."""
    if not questions:
        return np.zeros(0)
    suffixes = [enc.question(q) for q in questions]
    reads = [[(len(s) - 1, enc.options(q))] for s, q in zip(suffixes, questions)]
    got = backend.read(enc.prefix(explanation), suffixes, reads)
    out = []
    for q, r in zip(questions, got):
        lp = np.asarray(r[0], dtype=np.float64)
        lp = lp - np.logaddexp.reduce(lp)
        out.append(-lp[q["answer"]] / LN2)
    return np.array(out)


def derangement(n: int, seed: int) -> list[int]:
    """A seeded permutation with no fixed point (n >= 2)."""
    rng = random.Random(seed)
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if all(i != j for i, j in enumerate(p)):
            return p


# ------------------------------------------------------------------------------------------- building

def _algorithm(program_source: str):
    """The answer program's variables (mech's evaluator), align/claim lines left out."""
    import ast
    import types

    import mech

    tree = ast.parse(mech.quote_parts(program_source))
    tree.body = [s for s in tree.body if not (isinstance(s, (ast.Import, ast.ImportFrom)) or
                                              (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                                               and getattr(s.value.func, "id", "") in ("align", "claim")))]
    namespace: dict = {}
    exec(compile(tree, "<algorithm>", "exec"), namespace)  # the teacher's own program, trusted
    # The variables: `answer` and the functions it reads by name (helpers taking other arguments are not variables).
    if not isinstance(namespace.get("answer"), types.FunctionType):
        return None, []
    algorithm = mech._Algorithm(namespace, ["answer"])
    return algorithm, [n for n in algorithm.params if n != "answer"]


def build_behavior(entry: dict, per_type: int, seed: int, checker=None) -> list[dict]:
    """The questions of one teacher answer (manifest entry)."""
    import mech
    import prompt as P

    rng = random.Random(f"{seed}:{entry['behavior']}")
    path = Path.home() / "mpd-data/graph_oracle/behaviors" / entry["model"] / f"{entry['behavior']}.json"
    behavior = json.loads(path.read_text())
    tk = mech.tokenizer(entry["model"])
    decode = lambda ids: tk.decode(list(ids), skip_special_tokens=True)  # noqa: E731
    base = {"behavior": entry["behavior"], "family": entry["family"]}
    prompts = behavior["prompts"]
    out = []
    # next: M's top token against 3 other prompts' top tokens.
    tops = [(i, k, p["model_top"][k][0][0]) for i, p in enumerate(prompts) for k, _ in enumerate(p["target_positions"]) if p.get("model_top")]
    for i, k, top in rng.sample(tops, min(per_type, len(tops))):
        others = sorted({t for _, _, t in tops if t != top})
        if len(others) < 3:
            continue
        options = rng.sample(others, 3) + [top]
        rng.shuffle(options)
        pos = prompts[i]["target_positions"][k]
        out.append({**base, "type": "next", "text": decode(prompts[i]["token_ids"][: pos + 1]), "options": options, "answer": options.index(top)})
    # switch: the checker's clean complement run of the answer's parts.
    if checker is not None:
        source = P.program_of(Path(entry["answer"]).read_text())
        checker.behavior(str(path))
        reply = checker.request({"op": "complement", "program": checker.ir(source), "stand_in": "counterfactual"})
        rows = [r for r in reply["tokens"] if r["clean_top"] != r["counterfactual_top"] and r["prompt"] < len(prompts)]
        for r in rng.sample(rows, min(per_type, len(rows))):
            p = prompts[r["prompt"]]
            shift = len(p["counterfactual"]["token_ids"]) - len(p["token_ids"])
            out.append({**base, "type": "switch", "text": decode(p["token_ids"][: r["position"] + 1]),
                        "counterfactual": decode(p["counterfactual"]["token_ids"][: r["position"] + shift + 1]),
                        "clean_token": decode([r["clean_top"]]), "counter_token": decode([r["counterfactual_top"]]),
                        "answer": 0 if r["complement"][1] > r["complement"][0] else 1, "complement": r["complement"]})
    # step: intermediate variables of the answer's algorithm on two texts.
    algorithm, steps = _algorithm(P.program_of(Path(entry["answer"]).read_text()))
    if steps:
        toks, _ = mech.behavior_tokens(behavior, entry["model"])
        cands = []
        for i, p in enumerate(prompts):
            for pos in p["target_positions"]:
                if toks["counterfactuals"]:
                    shift = len(p["counterfactual"]["token_ids"]) - len(p["token_ids"])
                    cands.append((("prompt", i, pos), ("counterfactual", i, pos + shift)))
                j = rng.randrange(len(prompts))
                if j != i and prompts[j]["target_positions"]:
                    cands.append((("prompt", i, pos), ("prompt", j, prompts[j]["target_positions"][0])))
        for (ka, ia, pa), (kb, ib, pb) in rng.sample(cands, min(per_type, len(cands))):
            var = rng.choice(steps)
            seq = lambda kind, i: toks["prompts" if kind == "prompt" else "counterfactuals"][i]  # noqa: E731
            ids = lambda kind, i: prompts[i]["token_ids"] if kind == "prompt" else prompts[i]["counterfactual"]["token_ids"]  # noqa: E731
            try:
                va = algorithm.values(seq(ka, ia), [var])[var][pa]
                vb = algorithm.values(seq(kb, ib), [var])[var][pb]
            except Exception:  # noqa: BLE001 - a variable the algorithm cannot compute on this text asks nothing
                continue
            out.append({**base, "type": "step", "variable": var, "text": decode(ids(ka, ia)[: pa + 1]), "other": decode(ids(kb, ib)[: pb + 1]),
                        "answer": 0 if repr(va) != repr(vb) else 1, "values": [repr(va)[:80], repr(vb)[:80]]})
    return out


# -------------------------------------------------------------------------------------------- scoring

def score(backend, questions: list[dict], explanations: dict[str, str], seed: int = 0) -> dict:
    """Bits per question with the behavior's own explanation, with none, and with another behavior's."""
    enc = Encoder(backend.tokenizer)
    names = sorted({q["behavior"] for q in questions})
    swap = dict(zip(names, (names[j] for j in derangement(len(names), seed)))) if len(names) > 1 else {}
    rows = []
    for b in names:
        qs = [q for q in questions if q["behavior"] == b]
        t0 = time.time()
        own = log_loss(backend, enc, explanations.get(b), qs)
        none = log_loss(backend, enc, None, qs)
        other = log_loss(backend, enc, explanations.get(swap.get(b, b)), qs) if swap else none
        print(f"reader questions: {b} {len(qs)} questions {time.time() - t0:.1f} s", file=sys.stderr, flush=True)
        for q, a, n, o in zip(qs, own, none, other):
            rows.append({"behavior": b, "family": q["family"], "type": q["type"], "own": float(a), "none": float(n), "shuffled": float(o),
                         "shuffled_from": swap.get(b)})
    return {"rows": rows, "summary": summarize(rows)}


def summarize(rows: list[dict]) -> dict:
    """Bits saved per question (none - own, and none - shuffled) by question type: mean, SE over questions,
    and SE over behaviors (behavior means as the units); and by family."""
    out = {}
    for key in sorted({r["type"] for r in rows}) + ["all"]:
        sel = [r for r in rows if key == "all" or r["type"] == key]
        if not sel:
            continue
        entry = {"questions": len(sel)}
        for arm in ("own", "shuffled"):
            s = np.array([r["none"] - r[arm] for r in sel])
            by_b = {}
            for r, v in zip(sel, s):
                by_b.setdefault(r["behavior"], []).append(v)
            means = np.array([np.mean(v) for v in by_b.values()])
            entry[f"saved_{arm}"] = float(s.mean())
            entry[f"se_questions_{arm}"] = float(s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 1 else float("nan")
            entry[f"se_behaviors_{arm}"] = float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) > 1 else float("nan")
        entry["bits_none"] = float(np.mean([r["none"] for r in sel]))
        entry["bits_own"] = float(np.mean([r["own"] for r in sel]))
        out[key] = entry
    out["families"] = {f: {t: float(np.mean([r["none"] - r["own"] for r in rows if r["family"] == f and r["type"] == t]))
                           for t in sorted({r["type"] for r in rows if r["family"] == f})} for f in sorted({r["family"] for r in rows})}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "score"])
    ap.add_argument("--manifest", default=str(Path.home() / "mpd-data/graph_oracle/teacher/manifest.jsonl"))
    ap.add_argument("--questions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-type", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-checker", action="store_true", help="build: skip the switch questions")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"])
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--max-batch", type=int, default=8)
    args = ap.parse_args()
    entries = [json.loads(line) for line in open(args.manifest) if line.strip()]
    if args.command == "build":
        checker = None
        if not args.no_checker:
            import score as S

            checker = S.Checker("vpd4l", views={"vpd": str(Path.home() / "mpd-data/engine/vpd4l_decomposition")}, device="gpu")
        with open(args.out, "w") as f:
            for e in entries:
                qs = build_behavior(e, args.per_type, args.seed, checker)
                print(e["behavior"], {t: sum(q["type"] == t for q in qs) for t in ("next", "switch", "step")}, flush=True)
                for q in qs:
                    f.write(json.dumps(q) + "\n")
        if checker is not None:
            checker.close()
        return
    import prompt as P
    from reader_score import CachedReader

    questions = [json.loads(line) for line in open(args.questions) if line.strip()]
    explanations = {e["behavior"]: P.explanation_of(Path(e["answer"]).read_text()) for e in entries}
    backend = CachedReader(args.model, args.batch_tokens, args.max_batch, args.seed, args.device, args.dtype)
    start = time.time()
    result = score(backend, questions, explanations, args.seed)
    result.update({"reader": backend.describe(), "seconds": time.time() - start, "questions": len(questions)})
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
