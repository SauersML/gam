"""#2951 manifold-native addition, phase A: harvest a pretrained LM's residual stream on two-digit addition.

Analysis under SPEC 8's exception (torch/transformers execution only; the fits are gamfit's, in phase B).

Problems ``AB+CD=EF`` with ``10 <= AB + CD <= 99`` so every prompt has one length, after a fixed few-shot
prefix, with the answer teacher-forced. Qwen3 tokenizes digit by digit. At every decoder layer (and the
embedding output) the residual is saved at six positions: A's tens and units, B's tens and units, ``=``
(which predicts E) and E (which predicts F), as float16, with the problem's variables and the model's
answer distributions at ``=`` and E over the ten digit tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

FEW_SHOT = "12+35=47\n40+19=59\n23+64=87\n51+38=89\n"
POSITIONS = ("A_tens", "A_units", "B_tens", "B_units", "equals", "E")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--problems", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    seen, rows = set(), []
    while len(rows) < args.problems:
        a, b = rng.randint(10, 89), rng.randint(10, 89)
        if not (10 <= a + b <= 99) or (a, b) in seen:
            continue
        seen.add((a, b))
        rows.append((a, b))
    texts = [FEW_SHOT + f"{a}+{b}={a + b}" for a, b in rows]
    ids = torch.tensor(tok(texts).input_ids, device=args.device)
    L = ids.shape[1]
    pos = {"A_tens": L - 8, "A_units": L - 7, "B_tens": L - 5, "B_units": L - 4, "equals": L - 3, "E": L - 2}
    for i, (a, b) in enumerate(rows[:50]):
        want = [a // 10, a % 10, b // 10, b % 10]
        got = [ids[i, pos[p]].item() for p in POSITIONS[:4]]
        assert got == [digit_ids[w] for w in want], (texts[i], got)
    n_layers = len(model.model.layers)
    width = model.config.hidden_size
    store = np.zeros((len(rows), n_layers + 1, len(POSITIONS), width), dtype=np.float16)
    answers = np.zeros((len(rows), 2, 10), dtype=np.float32)
    columns = [pos[p] for p in POSITIONS]
    with torch.inference_mode():
        for s in range(0, len(rows), 64):
            out = model(input_ids=ids[s:s + 64], output_hidden_states=True)
            hs = torch.stack(out.hidden_states, 1)  # batch x layers+1 x L x width
            store[s:s + 64] = hs[:, :, columns].half().cpu().numpy()
            logits = out.logits[:, [pos["equals"], pos["E"]]][..., digit_ids].float()
            answers[s:s + 64] = torch.log_softmax(logits, -1).cpu().numpy()
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "residuals.npy"), store)
    np.save(os.path.join(args.out_dir, "answer_logp.npy"), answers)
    a = np.array([r[0] for r in rows])
    b = np.array([r[1] for r in rows])
    s = a + b
    variables = {"A": a, "B": b, "A_tens": a // 10, "A_units": a % 10, "B_tens": b // 10, "B_units": b % 10,
                 "sum": s, "sum_tens": s // 10, "sum_units": s % 10, "carry": ((a % 10) + (b % 10) >= 10).astype(int)}
    np.savez(os.path.join(args.out_dir, "variables.npz"), **variables)
    acc_e = float((answers[:, 0].argmax(-1) == s // 10).mean())
    acc_f = float((answers[:, 1].argmax(-1) == s % 10).mean())
    meta = {"model": args.model, "problems": len(rows), "layers": n_layers, "width": width, "positions": list(POSITIONS),
            "prompt_length": L, "few_shot": FEW_SHOT, "accuracy_tens": acc_e, "accuracy_units": acc_f, "seed": args.seed}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=1)
    print(f"[harvest] {meta}", flush=True)


if __name__ == "__main__":
    main()
