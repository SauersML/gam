"""Build and verify the behavior suite for one model (MPD #2951 graph oracle).

    mem-lease 6 ~/mpd-data/venv/bin/python bench/oracle/graph/behaviors/build.py --model qwen3-0.6b
    mem-lease 4 ~/mpd-data/venv/bin/python bench/oracle/graph/behaviors/build.py --model vpd4l

For every family in families.py and every variant, this tokenizes each item with the model's tokenizer,
pairs it with an aligned counterfactual, runs the model on clean and counterfactual prompts, and writes
~/mpd-data/graph_oracle/behaviors/<model>/<family>.<variant>.json (design.txt section 5). A behavior is kept
when the answer is the model's top-1 token on at least half of its targets ("kept_top1"), or when the model ranks
the answer above the counterfactual's answer on at least nine prompts in ten, counting both prompts of each pair
("kept_contrast": agreement and similar tasks, where the verb competes with other continuations for top-1); the
others go to <model>/dropped/.
The summary of both models is ~/mpd-data/graph_oracle/behaviors/summary.tsv.

Prompt conventions. `token_ids` is the whole text including the answer. A target position t is a position
whose next-token distribution is scored; the correct token is token_ids[t + 1]. The targets are the tokens
that hold a non-space character of the answer (the tokenizer may merge the answer with the end of the prefix,
as in "))"). A counterfactual has the same token length and target positions; its token_ids hold its own
answer. `accepted_token_ids[i]` (only for families with a set of correct answers, such as greater-than) lists
the tokens counted correct at target i. vpd4l prompts start with <|endoftext|> (id 0), the document separator
of its Pile training stream; Qwen3 prompts have no prefix token.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from families import FAMILIES, MIN_PROMPTS, NOVEL, Item  # noqa: E402

OUT = Path.home() / "mpd-data/graph_oracle/behaviors"
KEEP_ACCURACY = 0.5  # kept when the answer is the model's top-1 token on most targets ("top1"),
KEEP_PAIR = 0.9  # or when the model ranks the answer above the counterfactual's answer on nine pairs in ten ("contrast")
TOP = 5
MAX_PROMPTS = 128  # enough prompts per behavior for the score's sampled experiments; keeps files small


def split_of(fam: str) -> str:
    """Held-out families: every family in NOVEL (written for this suite) and one in five of the others by a hash of the
    family name, the same for every model."""
    return "heldout" if fam in NOVEL or int(hashlib.sha1(fam.encode()).hexdigest(), 16) % 5 == 0 else "train"


class Tok:
    def __init__(self, model: str):
        self.model = model
        if model == "qwen3-0.6b":
            from transformers import AutoTokenizer
            self.hf = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            self.prefix_ids = []
        else:
            import tokenizers
            self.tk = tokenizers.Tokenizer.from_file(str(Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"))
            self.prefix_ids = [0]

    def encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        if self.model == "qwen3-0.6b":
            e = self.hf(text, return_offsets_mapping=True, add_special_tokens=False)
            ids, offs = e["input_ids"], [tuple(o) for o in e["offset_mapping"]]
        else:
            e = self.tk.encode(text)
            ids, offs = e.ids, [tuple(o) for o in e.offsets]
        return [*self.prefix_ids, *ids], [(0, 0)] * len(self.prefix_ids) + offs

    def count(self, s: str) -> int:
        return len(self.encode(s)[0]) - len(self.prefix_ids)

    def decode(self, i: int) -> str:
        return self.hf.decode([i]) if self.model == "qwen3-0.6b" else self.tk.decode([i])


def encode_item(tok: Tok, prefix: str, answer: str):
    """(token_ids, target_positions) or None when the answer has no token of its own."""
    text = prefix + answer
    ids, offs = tok.encode(text)
    lo = len(prefix)
    tgt = [j for j, (s, e) in enumerate(offs) if j >= 1 and e > lo and any(not c.isspace() for c in text[max(s, lo):e])]
    if not tgt or tgt != list(range(tgt[0], tgt[-1] + 1)):
        return None
    return text, ids, [j - 1 for j in tgt]


def accepted(tok: Tok, prefix: str, alts: list[str], ids: list[int], pos: int) -> list[int] | None:
    """Tokens counted correct at the first target: the first answer token of each accepted answer."""
    out = set()
    for a in alts:
        r = encode_item(tok, prefix, a)
        if r is None or r[2][0] != pos or r[1][:pos + 1] != ids[:pos + 1]:
            continue
        out.add(r[1][pos + 1])
    return sorted(out) or None


def build_prompts(tok: Tok, items: list[Item], rng: random.Random) -> list[dict]:
    enc = []
    for it in items:
        r = encode_item(tok, it.prefix, it.answer)
        if r is not None:
            enc.append((it, r))
    prompts, seen = [], set()
    for it, (text, ids, tp) in enc:
        if text in seen:
            continue
        if it.accept:
            tp = tp[:1]  # a set-valued answer is decided at its first token
        cf = None
        if it.cf_prefix is not None:
            r = encode_item(tok, it.cf_prefix, it.cf_answer)
            if r is not None and len(r[1]) == len(ids) and (r[2][:len(tp)] == tp) and (it.accept or r[2] == tp):
                cf = (it.cf_prefix, it.cf_answer, it.cf_accept, r)
        else:  # pair with another entity of the same template and token shape
            mates = [(o, ro) for o, ro in enc if o.tmpl == it.tmpl and o.answer != it.answer and len(ro[1]) == len(ids) and ro[2] == tp]
            if mates:
                o, ro = rng.choice(mates)
                cf = (o.prefix, o.answer, o.accept, ro)
        if cf is None:
            continue
        seen.add(text)
        cfp, cfa, cfacc, (cftext, cfids, _) = cf
        p = {"text": text, "token_ids": ids, "target_positions": tp, "answer": it.answer,
             "counterfactual": {"text": cftext, "token_ids": cfids, "answer": cfa}}
        if it.accept:
            p["accepted_token_ids"] = [accepted(tok, it.prefix, it.accept, ids, tp[0]) or [ids[tp[0] + 1]]]
            p["counterfactual"]["accepted_token_ids"] = [accepted(tok, cfp, cfacc, cfids, tp[0]) or [cfids[tp[0] + 1]]]
        prompts.append(p)
    return prompts


class Model:
    def __init__(self, model: str, device: str = "mps"):
        self.model, self.device = model, device
        if model == "qwen3-0.6b":
            from transformers import AutoModelForCausalLM
            self.m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).to(device).eval()
        else:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vpd_2951"))
            from vpd_model import load_target
            self.m = load_target(device)

    @torch.no_grad()
    def logprobs(self, seqs: list[list[int]], positions: list[list[int]], batch: int = 32) -> list[torch.Tensor]:
        """Log-probabilities [len(positions_i), vocab] at the given positions of each sequence (right padding:
        causal attention leaves the real positions unchanged)."""
        out = []
        for b in range(0, len(seqs), batch):
            ss = seqs[b:b + batch]
            T = max(map(len, ss))
            ids = torch.zeros(len(ss), T, dtype=torch.long)
            mask = torch.zeros(len(ss), T, dtype=torch.long)
            for i, s in enumerate(ss):
                ids[i, :len(s)] = torch.tensor(s)
                mask[i, :len(s)] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            if self.model == "qwen3-0.6b":  # the head only at the scored positions
                h, head = self.m.model(input_ids=ids, attention_mask=mask).last_hidden_state, self.m.lm_head
            else:
                h, head = self.m.hidden(ids), lambda x: x @ self.m.wte.T
            pos = positions[b:b + batch]
            bi = torch.tensor([i for i, ps in enumerate(pos) for _ in ps], device=h.device)
            ti = torch.tensor([t for ps in pos for t in ps], device=h.device)
            lp = torch.log_softmax(head(h[bi, ti]).float(), -1).cpu()  # one transfer per batch
            out += list(lp.split([len(ps) for ps in pos]))
        return out


def score(tok: Tok, model: Model, prompts: list[dict]) -> dict:
    """Fills each prompt's model_top and correct (clean and counterfactual); returns the behavior's accuracies."""
    seqs = [p["token_ids"] for p in prompts] + [p["counterfactual"]["token_ids"] for p in prompts]
    pos = [p["target_positions"] for p in prompts] * 2
    lps = model.logprobs(seqs, pos)
    n = len(prompts)
    hits = {"clean": [], "cf": []}
    pair = []
    for k, p in enumerate(prompts):
        for which, d, lp in (("clean", p, lps[k]), ("cf", p["counterfactual"], lps[n + k])):
            ids, tp = d["token_ids"], p["target_positions"]
            acc = d.get("accepted_token_ids")
            correct = []
            for i, t in enumerate(tp):
                top1 = int(lp[i].argmax())
                ok = top1 in acc[i] if acc and i < len(acc) else top1 == ids[t + 1]
                correct.append(bool(ok))
            d["correct"] = correct
            v, ix = lp.exp().topk(TOP, -1)
            d["model_top"] = [[[tok.decode(int(j)), round(float(q), 4)] for q, j in zip(vr, ir)] for vr, ir in zip(v, ix)]
            hits[which] += correct
        # the answer outranks the counterfactual's answer at the first target, on both prompts
        t0 = p["target_positions"][0]
        a, b = p["token_ids"][t0 + 1], p["counterfactual"]["token_ids"][t0 + 1]
        if a != b:
            pair.append(float(lps[k][0, a] > lps[k][0, b]))
            pair.append(float(lps[n + k][0, b] > lps[n + k][0, a]))
    mean = lambda x: round(sum(x) / len(x), 4) if x else None
    return {"model_accuracy": mean(hits["clean"]), "counterfactual_accuracy": mean(hits["cf"]), "pair_accuracy": mean(pair),
            "targets": len(hits["clean"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3-0.6b", "vpd4l"])
    ap.add_argument("--families", default="", help="comma-separated subset (default all)")
    ap.add_argument("--seed", type=int, default=2951)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    tok = Tok(a.model)
    model = Model(a.model, a.device)
    root = OUT / a.model
    (root / "dropped").mkdir(parents=True, exist_ok=True)
    rows = []
    fams = [f for f in FAMILIES if not a.families or f in a.families.split(",")]
    for fam in fams:
        rng = random.Random(f"{a.seed}:{fam}")
        for v in FAMILIES[fam](tok, rng):
            bid = f"{fam}.{v.name}"
            prompts = build_prompts(tok, v.items, random.Random(f"{a.seed}:{bid}"))
            random.Random(f"{a.seed}:{bid}:order").shuffle(prompts)
            prompts = prompts[:MAX_PROMPTS]
            row = {"model": a.model, "id": bid, "family": fam, "variant": v.name, "prompts": len(prompts), "split": split_of(fam)}
            if not prompts:
                rows.append({**row, "status": "no_prompts"})
                print(f"{bid:40s} no prompts", flush=True)
                continue
            acc = score(tok, model, prompts)
            reason = "top1" if acc["model_accuracy"] >= KEEP_ACCURACY else "contrast" if (acc["pair_accuracy"] or 0) >= KEEP_PAIR else ""
            status = "too_few" if len(prompts) < MIN_PROMPTS else f"kept_{reason}" if reason else "dropped"
            beh = {"id": bid, "model": a.model, "family": fam, "variant": v.name, "description": v.description, "frequency": None, "novel": fam in NOVEL,
                   "prompts": prompts, "split": row["split"], **acc, "keep": status}
            dest = root / f"{bid}.json" if reason and status != "too_few" else root / "dropped" / f"{bid}.json"
            for stale in (root / f"{bid}.json", root / "dropped" / f"{bid}.json"):
                stale.unlink(missing_ok=True)
            dest.write_text(json.dumps(beh))
            rows.append({**row, **acc, "status": status})
            print(f"{bid:40s} n={len(prompts):4d} acc={acc['model_accuracy']:.3f} cf={acc['counterfactual_accuracy']:.3f} "
                  f"pair={acc['pair_accuracy']} {status}", flush=True)
    write_summary(a.model, fams, rows)


def write_summary(model: str, fams: list[str], rows: list[dict]):
    """Replace this model's rows for these families in summary.tsv."""
    cols = ["model", "family", "variant", "id", "prompts", "targets", "model_accuracy", "counterfactual_accuracy", "pair_accuracy", "split", "status"]
    path = OUT / "summary.tsv"
    old = []
    if path.exists():
        with path.open() as f:
            old = [r for r in csv.DictReader(f, delimiter="\t") if not (r["model"] == model and r["family"] in fams)]
    with path.open("w") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in sorted(old + rows, key=lambda r: (r["model"], r["family"], r["variant"])):
            w.writerow(r)


if __name__ == "__main__":
    main()
