"""Retained reply state behaviors (MPD #2951 graph oracle), from the introspection study in ~/retained-reply-state.

Turn 1 asks the model to hold one of 50 animals in its thinking and reply "I understand."; turn 2 asks for the animal.
The thinking texts are that study's runs (results/qwen3_<size>.json, wording A, forced choice: the animal is written into
the start of the thinking). Two families, both held out:
  retained_reply_state   the turn-2 tokens (question, empty thinking block, "Animal:", the answer) cannot attend to the
                         turn-1 thinking; the reply tokens were computed with the thinking present and can. Encoded as
                         design.txt's "attention_block". This is the study's "reply cache kept" condition without moving
                         the reply's keys back by the thinking's length (positions stay unshifted, so rotary distances
                         from the turn-2 tokens to the prompt are longer by the thinking's length).
  retained_thinking_kept the same sequences with no mask: the thinking stays visible, so recall is in-context copying
                         (the control).
The target is the first token of the answer " <Animal>" after "Animal:". The counterfactual writes another animal into
the thinking (every mention replaced; the other animal is drawn among those that keep the sequence length and start with
another token). The effect check scores all 50 animals (log-sum-exp over " Name" and " name", every token of the name)
under the mask and reports the study's animal-level statistic (stats.animal_level: mean own-animal raise over animals,
double-centred, null re-pairs animals) with its z.

    mem-lease 8 ~/mpd-data/venv/bin/python bench/oracle/graph/behaviors/retained_state.py --model qwen3-0.6b
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

import os  # noqa: E402

# read-only; its templates and statistics are imported as they are (MATS jobs pass a copy through RETAINED_STUDY)
STUDY = Path(os.environ.get("RETAINED_STUDY", Path.home() / "retained-reply-state"))
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hidden_choice import ANIMALS, CLOSE, OPEN, WORDINGS  # noqa: E402
from stats import animal_level, raises  # noqa: E402
import build  # noqa: E402
from build import HF, TOP, write_summary  # noqa: E402


def substitute(thinking: str, a: str, b: str) -> str:
    """Every whole-word mention of animal a replaced by b, keeping capitalization."""
    return re.sub(r"\b" + a + r"\b", lambda m: b.title() if m.group(0)[0].isupper() else b, thinking, flags=re.I)


class Runs:
    """Token layout of one study run: prompt | thinking | reply | turn-2 suffix | answer."""

    def __init__(self, tok, wording="A"):
        self.tok = tok
        turn1, recall = WORDINGS[wording]
        chat = lambda msgs: tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True),
                                       add_special_tokens=False)
        self.prompt = chat([{"role": "user", "content": turn1}])
        self.reply = tok.encode("I understand.<|im_end|>", add_special_tokens=False)
        full = chat([{"role": "user", "content": turn1}, {"role": "assistant", "content": "I understand."}, {"role": "user", "content": recall}])
        P, R = len(self.prompt), len(self.reply)
        assert full[:P] == self.prompt and full[P:P + R] == self.reply, "chat template does not keep the turn-1 prefix"
        self.suffix = full[P + R:] + tok.encode("<think>\n\n</think>\n\nAnimal:", add_special_tokens=False)
        self.forms = [(c, tok.encode(f, add_special_tokens=False)) for c in ANIMALS for f in (" " + c.title(), " " + c)]

    def context(self, thinking: str):
        think = self.tok.encode("<think>\n" + thinking + "\n</think>\n\n", add_special_tokens=False)
        ids = self.prompt + think + self.reply + self.suffix
        P = len(self.prompt)
        return ids, (P, P + len(think)), P + len(think) + len(self.reply)  # ids, thinking span, turn-2 start

    def answer(self, animal: str) -> int:
        """First token of the lowercase answer: the list and the thinking write the names in lowercase, and the models
        answer in lowercase (Qwen3-0.6B with the thinking visible: ' o' for ocelot at p 0.995)."""
        return self.tok.encode(" " + animal, add_special_tokens=False)[0]

    def accepted(self, animal: str) -> list[int]:
        """First tokens of both forms the readout sums over (" name" and " Name")."""
        return sorted({self.tok.encode(" " + f, add_special_tokens=False)[0] for f in (animal, animal.title())})


def mask_for(n_ctx: int, cand: list[list[int]], think: tuple[int, int], s2: int, block: bool, dtype) -> tuple[torch.Tensor, list[int]]:
    """[1, 1, N + T, N + T] additive mask and position ids for a context of N tokens followed by packed candidates.
    Candidates attend to the context and to their own earlier tokens; with `block`, the turn-2 tokens and the candidates
    cannot attend to the thinking span."""
    T = sum(map(len, cand))
    L = n_ctx + T
    neg = torch.finfo(dtype).min
    m = torch.full((L, L), neg, dtype=dtype)
    m[:n_ctx, :n_ctx] = torch.triu(torch.full((n_ctx, n_ctx), neg, dtype=dtype), 1)
    pos = list(range(n_ctx))
    q = n_ctx
    for toks in cand:
        for t in range(len(toks)):
            m[q + t, :n_ctx] = 0
            m[q + t, q:q + t + 1] = 0
            pos.append(n_ctx + t)
        q += len(toks)
    if block:
        m[s2:, think[0]:think[1]] = neg
    return m[None, None], pos


@torch.no_grad()
def score(model, runs: Runs, ids, think, s2, block: bool, device) -> tuple[np.ndarray, torch.Tensor]:
    """log P of each of the 50 animals as the answer (all tokens, log-sum-exp over forms) and the next-token
    log-probabilities after "Animal:"."""
    cand = [t for _, t in runs.forms]
    dtype = next(model.parameters()).dtype
    m, pos = mask_for(len(ids), cand, think, s2, block, dtype)
    x = torch.tensor([ids + [t for c in cand for t in c]], device=device)
    h = model.model(input_ids=x, attention_mask=m.to(device), position_ids=torch.tensor([pos], device=device)).last_hidden_state[0]
    # the head only where a log-probability is read: after "Animal:" and before each later candidate token
    rows, q = [len(ids) - 1], len(ids)
    for toks in cand:
        rows += [q + t - 1 for t in range(1, len(toks))]
        q += len(toks)
    lp = torch.log_softmax(model.lm_head(h[rows]).float(), -1).cpu()
    first = lp[0]
    best = np.full(len(ANIMALS), -np.inf)
    r = 1
    for (c, toks) in runs.forms:
        s = float(first[toks[0]]) + sum(float(lp[r + t - 1, toks[t]]) for t in range(1, len(toks)))
        r += len(toks) - 1
        k = ANIMALS.index(c)
        best[k] = np.logaddexp(best[k], s)
    return best, first


@torch.no_grad()
def generate_runs(model, tok, wording: str, n: int, device, seed: int, batch: int = 32, max_new: int = 600) -> dict:
    """Turn-1 runs as the study's hidden_choice.py --choice forced makes them (its sampling settings and thinking
    openings): the animal is drawn uniformly and written into the start of the thinking; runs whose visible reply is
    exactly "I understand." are kept. Only the turn-1 generation, without the study's arms."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    turn1, _ = WORDINGS[wording]
    prompt = tok.encode(tok.apply_chat_template([{"role": "user", "content": turn1}], tokenize=False, add_generation_prompt=True,
                                                enable_thinking=True), add_special_tokens=False)
    chosen, thinking = [], []
    pad = tok.pad_token_id
    for s0 in range(0, n, batch):
        B = min(batch, n - s0)
        picks = [ANIMALS[k] for k in rng.integers(0, len(ANIMALS), B)]
        starts = [f"<think>\n{OPEN[rng.integers(len(OPEN))]} Draw #{rng.integers(100, 1000)}: {c}. "
                  f"{CLOSE[rng.integers(len(CLOSE))]} I'll keep {c} in mind." for c in picks]
        seqs = [prompt + tok.encode(x, add_special_tokens=False) for x in starts]
        W = max(map(len, seqs))
        inp = torch.tensor([[pad] * (W - len(x)) + x for x in seqs], device=device)
        att = torch.tensor([[0] * (W - len(x)) + [1] * len(x) for x in seqs], device=device)
        out = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=max_new, do_sample=True, temperature=0.6,
                             top_p=0.95, top_k=20, pad_token_id=pad)
        for j in range(B):
            text = starts[j] + tok.decode(out[j, W:], skip_special_tokens=False).split("<|im_end|>")[0]
            if "</think>" not in text:
                continue
            th, visible = text.split("</think>", 1)
            if visible.strip() == "I understand.":
                chosen.append(picks[j])
                thinking.append(th.replace("<think>", "", 1).strip())
        print(f"turn 1: {s0 + B} generated, {len(chosen)} kept", flush=True)
    return {"wording": wording, "choice": "forced", "chosen": chosen, "thinking": thinking, "seed": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-0.6b", choices=sorted(HF))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=2951)
    ap.add_argument("--out", default=str(build.OUT))
    ap.add_argument("--study-file", default="", help="a hidden_choice.py result (default: the study's results/<model>.json)")
    ap.add_argument("--generate", type=int, default=0, help="make this many turn-1 runs first (written to --study-file)")
    ap.add_argument("--wording", default="A", choices=sorted(WORDINGS))
    a = ap.parse_args()
    build.OUT = Path(a.out)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF[a.model])
    model = AutoModelForCausalLM.from_pretrained(HF[a.model], dtype=getattr(torch, a.dtype), attn_implementation="sdpa").to(a.device).eval()
    src = Path(a.study_file) if a.study_file else STUDY / f"results/{a.model.replace('-', '_')}.json"
    if a.generate:
        src.write_text(json.dumps(generate_runs(model, tok, a.wording, a.generate, a.device, a.seed)))
    study = json.load(open(src))
    wording = study.get("wording", "A")
    runs = Runs(tok, wording)
    rng = random.Random(f"{a.seed}:retained")
    prompts = {"retained_reply_state": [], "retained_thinking_kept": []}
    L = {k: [] for k in prompts}
    chosen = []
    for animal, thinking in zip(study["chosen"], study["thinking"]):
        ids, think, s2 = runs.context(thinking)
        ans = runs.answer(animal)
        others = [b for b in ANIMALS if b != animal and not set(runs.accepted(b)) & set(runs.accepted(animal))]
        rng.shuffle(others)
        cf = None
        for b in others:
            cids, cthink, cs2 = runs.context(substitute(thinking, animal, b))
            if len(cids) == len(ids) and cthink == think and cs2 == s2:
                cf = (b, cids)
                break
        if cf is None:
            continue
        chosen.append(ANIMALS.index(animal))
        for fam, block in (("retained_reply_state", True), ("retained_thinking_kept", False)):
            best, first = score(model, runs, ids, think, s2, block, a.device)
            _, cfirst = score(model, runs, cf[1], think, s2, block, a.device)
            L[fam].append(best)
            toks, cans = ids + [ans], cf[1] + [runs.answer(cf[0])]
            p = {"text": tok.decode(toks), "token_ids": toks, "target_positions": [len(ids) - 1], "answer": " " + animal,
                 "accepted_token_ids": [runs.accepted(animal)], "hidden": animal,
                 "counterfactual": {"text": tok.decode(cans), "token_ids": cans, "answer": " " + cf[0],
                                    "accepted_token_ids": [runs.accepted(cf[0])], "hidden": cf[0]}}
            if block:
                blk = [[s2, len(toks), think[0], think[1]]]
                p["attention_block"] = blk
                p["counterfactual"]["attention_block"] = blk
            for d, f in ((p, first), (p["counterfactual"], cfirst)):
                v, ix = f.exp().topk(TOP)
                d["model_top"] = [[[tok.decode([int(j)]), round(float(q), 4)] for q, j in zip(v, ix)]]
                d["correct"] = [int(f.argmax()) in d["accepted_token_ids"][0]]
            mass = lambda f, ids_: float(torch.logsumexp(f[ids_], 0))  # first-token mass of a name's two forms
            pa = float(mass(first, runs.accepted(animal)) > mass(first, runs.accepted(cf[0])))
            pb = float(mass(cfirst, runs.accepted(cf[0])) > mass(cfirst, runs.accepted(animal)))
            p["pair"] = [pa, pb]
            prompts[fam].append(p)
        if len(chosen) % 100 == 0:
            print(f"{len(chosen)} runs", flush=True)
    c = np.array(chosen)
    root = build.OUT / a.model
    root.mkdir(parents=True, exist_ok=True)
    nrng = np.random.default_rng(0)
    rows = []
    for fam, desc in (("retained_reply_state", "Retained reply state: the model held a hidden animal in its turn-1 thinking and replied \"I understand.\"; asked for the animal in turn 2 with the thinking masked from the turn-2 tokens (the reply tokens computed with it are kept), it answers. Study: in Qwen3-1.7B three layer-21 key/value heads read the reply tokens' entries (21:0 raises the hidden animal, 21:5 and 21:6 lower it); the net sign depends on the model and wording."),
                      ("retained_thinking_kept", "Retained reply state control: the same turn-2 recall with the turn-1 thinking visible, so the hidden animal can be copied from context.")):
        Lf = np.array(L[fam])
        z, pval = animal_level(Lf, c, nrng, 5000, two_sided=True)
        ps = prompts[fam]
        acc = float(np.mean([p["correct"][0] for p in ps]))
        beh = {"id": f"{fam}.{wording}", "model": a.model, "family": fam, "variant": wording, "description": desc, "frequency": None,
               "prompts": ps, "split": "heldout", "model_accuracy": round(acc, 4),
               "counterfactual_accuracy": round(float(np.mean([p["counterfactual"]["correct"][0] for p in ps])), 4),
               "pair_accuracy": round(float(np.mean([x for p in ps for x in p["pair"]])), 4), "targets": len(ps),
               "effect": {"own_animal_raise_nats": float(raises(Lf, c).mean()), "animal_level_z": float(z), "p_two_sided": float(pval),
                          "runs": len(ps), "top1_of_50": float(np.mean(Lf.argmax(1) == c)), "source": str(src)},
               "keep": "kept_effect"}
        (root / f"{fam}.{wording}.json").write_text(json.dumps(beh))
        rows.append({"model": a.model, "family": fam, "variant": wording, "id": beh["id"], "prompts": len(ps), "targets": len(ps),
                     "model_accuracy": beh["model_accuracy"], "counterfactual_accuracy": beh["counterfactual_accuracy"],
                     "pair_accuracy": beh["pair_accuracy"], "split": "heldout", "status": f"kept_effect z={z:+.2f}"})
        print(f"{fam}.{wording}: runs {len(ps)}  raise {beh['effect']['own_animal_raise_nats']:+.4f} nats  animal-level z {z:+.2f}  p {pval:.3g}  "
              f"top-1 of 50 {beh['effect']['top1_of_50']:.3f}  first-token top-1 {acc:.3f}", flush=True)
    write_summary(a.model, list(prompts), rows, by_id=True)  # one wording at a time


if __name__ == "__main__":
    main()
