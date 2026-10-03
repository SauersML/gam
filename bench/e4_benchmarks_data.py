"""E4 on standard benchmarks (#2951): does the VPD paper's one-subcomponent emoticon edit, or a LoRA of equal edit
success, change what the model does on BLiMP, LAMBADA, HellaSwag, ARC-Easy and PIQA?

Scored as lm-eval-harness scores a base LM (log-likelihood ranking, no prompts beyond the harness's own):
  hellaswag  validation (10,042): query = preprocess(activity_label + ": " + ctx_a + " " + ctx_b.capitalize()),
             choices " " + preprocess(ending); acc (sum log p) and acc_norm (log p / len(ending) in characters)
  arc_easy   test (2,376): "Question: {question}\\nAnswer:", choices " {text}"; 3-5 choices
  piqa       validation (1,838): "Question: {goal}\\nAnswer:", choices " {sol1}" / " {sol2}"
  blimp      67 tasks x 1,000 pairs: whole-sentence log p after <|endoftext|> (the harness's empty context), good vs bad
  lambada    openai test (5,153): context = all but the last word, continuation " " + last word; greedy acc, log p
Continuation tokens are the tail of encode(context + continuation) beyond encode(context), as the harness does.
Contexts longer than the model's 512 positions keep their last tokens.

The measure of change is continuous and paired per item: the correct choice's log-probability share,
log p(correct) - log sum_choices p, edited minus original (for LAMBADA, the log p of the true last word).

Speed: the edit is in h.2.mlp.down_proj, so everything before it is shared. Each length-sorted batch runs the
shared part once, then every variant (base, VPD sweep, 8 LoRAs, 8 strength-matched VPD edits) finishes the network;
vocab-sized log-softmaxes are only taken at continuation positions.

usage: mem-lease 4 venv/python e4_benchmarks_data.py score [base]      MPD_MEM_GIB=2 ... summarize
outputs in ~/mpd-data/frontier/e4_side/bench/
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e4_side_effects_data as E  # noqa: E402

OUTD = E.FR / "e4_side/bench"
OUTD.mkdir(parents=True, exist_ok=True)
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
CTX = 512


def hs_pre(text):
    """lm-eval-harness hellaswag preprocess."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def build_requests():
    """Every (context, continuation) request of every benchmark, tokenized the harness way, cached to requests.npz.
    Items: task, gold index, request range, choice character lengths."""
    path = OUTD / "requests.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    from datasets import load_dataset
    tok = E.tokenizer()
    eot = tok.token_to_id("<|endoftext|>")
    enc = lambda s: tok.encode(s).ids
    items = []  # (task, gold, [(context, continuation)], [choice char lengths])
    for d in load_dataset("Rowan/hellaswag", split="validation"):
        q = hs_pre(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize())
        ch = [hs_pre(e) for e in d["endings"]]
        items.append(("hellaswag", int(d["label"]), [(q, " " + c) for c in ch], [len(c) for c in ch]))
    for d in load_dataset("allenai/ai2_arc", "ARC-Easy", split="test"):
        q = "Question: " + d["question"] + "\nAnswer:"
        ch = d["choices"]["text"]
        items.append(("arc_easy", d["choices"]["label"].index(d["answerKey"]), [(q, " " + c) for c in ch], [len(c) for c in ch]))
    for d in load_dataset("baber/piqa", split="validation"):
        q = "Question: " + d["goal"] + "\nAnswer:"
        ch = [d["sol1"], d["sol2"]]
        items.append(("piqa", int(d["label"]), [(q, " " + c) for c in ch], [len(c) for c in ch]))
    for d in load_dataset("EleutherAI/lambada_openai", "default", split="test"):
        words = d["text"].split(" ")
        items.append(("lambada", 0, [(" ".join(words[:-1]), " " + words[-1])], [len(words[-1])]))
    blimp = sorted(p.name for p in (Path.home() / ".cache/huggingface/datasets/nyu-mll___blimp").iterdir() if p.is_dir())
    assert len(blimp) == 67, len(blimp)
    for cfg in blimp:
        for d in load_dataset("nyu-mll/blimp", cfg, split="train"):
            ch = [d["sentence_good"], d["sentence_bad"]]
            items.append(("blimp:" + cfg, 0, [("", c) for c in ch], [len(c) for c in ch]))
    seqs, n_cont, req_lo, req_n = [], [], [], []
    for _, _, reqs, _ in items:
        req_lo.append(len(seqs))
        req_n.append(len(reqs))
        for ctx, cont in reqs:
            n_sp = len(ctx) - len(ctx.rstrip())  # the harness moves trailing context whitespace onto the continuation
            if n_sp:
                ctx, cont = ctx[:-n_sp], ctx[-n_sp:] + cont
            if ctx:
                whole, c_enc = enc(ctx + cont), enc(ctx)
                cont_enc, ctx_enc = whole[len(c_enc):], whole[:len(c_enc)]
            else:
                ctx_enc, cont_enc = [eot], enc(cont)
            full = (ctx_enc + cont_enc)[-(CTX + 1):]
            seqs.append(np.array(full, np.int32))
            n_cont.append(len(cont_enc))
    lens = np.array([len(s) for s in seqs])
    flat = np.concatenate(seqs)
    out = {"task": np.array([i[0] for i in items]), "gold": np.array([i[1] for i in items]), "req_lo": np.array(req_lo),
           "req_n": np.array(req_n), "chars": np.array([np.array(i[3], np.int32) for i in items], dtype=object),
           "flat": flat, "lens": lens, "n_cont": np.array(n_cont)}
    np.savez(path, **out)
    log(f"{len(items)} items, {len(seqs)} requests, {len(flat):,} tokens")
    return dict(np.load(path, allow_pickle=True))


def stage_score(base_only=False):
    """Per request and variant: the continuation's summed log-probability and whether greedy decoding reproduces it."""
    import torch
    rq = build_requests()
    lens, n_cont = rq["lens"], rq["n_cont"]
    offs = np.concatenate([[0], np.cumsum(lens)])
    if base_only:
        target, _, _ = E.load()
        W0 = target.site(E.SITE).W.clone()
        models, meta = {}, {}
    else:
        target, W0, models, meta = E.edit_variants()
    names = ["base"] + list(models)
    resid, final, head = E.split_forward(target)
    nR = len(lens)
    lp = np.zeros((nR, len(names)), np.float64)
    greedy = np.zeros((nR, len(names)), bool)
    order = np.argsort(lens, kind="stable")
    TOK = 3000  # tokens per batch (keeps the job inside 2 GiB)
    i = 0
    nb = 0
    with torch.no_grad():
        while i < nR:
            n = max(1, TOK // int(lens[order[i]]))
            while n > 1 and n * int(lens[order[min(nR, i + n) - 1]]) > TOK:  # lengths ascend: the last is the longest
                n = (n + 1) // 2
            j = min(nR, i + n)
            idx = order[i:j]
            Lm = int(lens[idx].max())
            ids = np.zeros((len(idx), Lm), np.int64)
            for r, q in enumerate(idx):
                ids[r, :lens[q]] = rq["flat"][offs[q]:offs[q + 1]]
            ids_t = torch.from_numpy(ids).to("mps")
            # scored positions: predicting token t+1 for the last n_cont tokens of each request
            rr = np.concatenate([np.full(n_cont[q], r) for r, q in enumerate(idx)])
            pp = np.concatenate([np.arange(lens[q] - n_cont[q] - 1, lens[q] - 1) for q in idx])
            tgt = ids_t[torch.from_numpy(rr).to("mps"), torch.from_numpy(pp + 1).to("mps")]
            rr_t, pp_t = torch.from_numpy(rr).to("mps"), torch.from_numpy(pp).to("mps")
            seg = torch.from_numpy(np.repeat(np.arange(len(idx)), n_cont[idx])).to("mps")
            xmid, g2 = resid(ids_t[:, :-1] if Lm > 1 else ids_t)
            for k, nm in enumerate(names):
                x = final(xmid, g2, W0 if nm == "base" else W0 + models[nm])[rr_t, pp_t]
                tl, gr = [], []
                for c in range(0, len(x), 512):
                    h = head(x[c:c + 512])
                    tl.append(h.gather(-1, tgt[c:c + 512, None])[:, 0])
                    gr.append(h.argmax(-1) == tgt[c:c + 512])
                tl, gr = torch.cat(tl), torch.cat(gr).float()  # MPS has no float64; sums of <= 512 terms are fine in fp32
                s = torch.zeros(len(idx), device="mps").index_add_(0, seg, tl)
                g = torch.zeros(len(idx), device="mps").index_add_(0, seg, gr)
                lp[idx, k] = s.cpu().numpy()
                greedy[idx, k] = (g.cpu().numpy() == n_cont[idx])
            i = j
            nb += 1
            if nb % 20 == 0:
                log(f"{i}/{nR} requests")
                torch.mps.empty_cache()
    np.savez(OUTD / ("scores_base.npz" if base_only else "scores.npz"), lp=lp, greedy=greedy, names=np.array(names))
    json.dump({"models": names, "meta": meta}, open(OUTD / "models.json", "w"), indent=1)
    log("scored")


def item_stats(rq, lp, greedy):
    """Per item and variant: correct (acc), correct by length-normalised score (acc_norm), and the continuous margin
    log p(correct) - logsumexp over choices (LAMBADA: the true word's log p)."""
    n_it = len(rq["task"])
    V = lp.shape[1]
    acc = np.zeros((n_it, V))
    accn = np.zeros((n_it, V))
    marg = np.zeros((n_it, V))
    for it in range(n_it):
        lo, n, gold = rq["req_lo"][it], rq["req_n"][it], rq["gold"][it]
        s = lp[lo:lo + n]
        if n == 1:
            acc[it] = accn[it] = greedy[lo]
            marg[it] = s[0]
            continue
        acc[it] = s.argmax(0) == gold
        accn[it] = (s / np.asarray(rq["chars"][it], float)[:, None]).argmax(0) == gold
        mx = s.max(0)
        marg[it] = s[gold] - (mx + np.log(np.exp(s - mx).sum(0)))
    return acc, accn, marg


def boot(x, B=2000, seed=0):
    """Mean and 95% interval over items (columns are variants)."""
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, len(x), len(x)), minlength=len(x)) for _ in range(B)]).astype(float)
    bs = (w @ x) / len(x)
    return x.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0), bs


def stage_summarize():
    rq = build_requests()
    f = OUTD / "scores.npz"
    z = np.load(f if f.exists() else OUTD / "scores_base.npz")
    lp, greedy, names = z["lp"], z["greedy"], list(z["names"])
    meta = json.load(open(OUTD / "models.json"))["meta"]
    acc, accn, marg = item_stats(rq, lp, greedy)
    task = rq["task"]
    chance_item = np.array([1.0 / n if n > 1 else 0.0 for n in rq["req_n"]])
    groups = {"hellaswag": task == "hellaswag", "arc_easy": task == "arc_easy", "piqa": task == "piqa",
              "lambada": task == "lambada", "blimp": np.char.startswith(task.astype(str), "blimp:")}
    for t in sorted(set(task)):
        if t.startswith("blimp:"):
            groups[t] = task == t
    pairs = [(nm, "vpd_match_" + nm) for nm in names if nm.startswith("lora") and "vpd_match_" + nm in names]
    res = {"description": __doc__, "models": names, "meta": meta, "pairs": pairs, "tasks": {}}
    for g, m in groups.items():
        e = {"n": int(m.sum()), "chance": float(chance_item[m].mean())}
        for key, arr in (("acc", acc), ("acc_norm", accn), ("margin", marg)):
            mean, lo, hi, _ = boot(arr[m][:, :1])
            e["base_" + key] = [float(mean[0]), float(lo[0]), float(hi[0])]
        if len(names) > 1:
            for key, arr in (("acc", acc), ("acc_norm", accn), ("margin", marg)):
                d = arr[m][:, 1:] - arr[m][:, :1]
                mean, lo, hi, bs = boot(d)
                e["d_" + key] = {nm: [float(mean[k]), float(lo[k]), float(hi[k])] for k, nm in enumerate(names[1:])}
                if key == "margin":
                    ad = np.abs(d)
                    am, alo, ahi, abs_bs = boot(ad, seed=1)
                    e["abs_d_margin"] = {nm: [float(am[k]), float(alo[k]), float(ahi[k])] for k, nm in enumerate(names[1:])}
                    e["pairs"] = {}
                    for L, Vm in pairs:
                        jl, jv = names.index(L) - 1, names.index(Vm) - 1
                        r = abs_bs[:, jl] / abs_bs[:, jv]
                        dd = bs[:, jl] - bs[:, jv]
                        e["pairs"][L] = {"abs_ratio": [float(am[jl] / am[jv]), *map(float, np.percentile(r, [2.5, 97.5]))],
                                         "d_margin_diff": [float(mean[jl] - mean[jv]), *map(float, np.percentile(dd, [2.5, 97.5]))]}
        res["tasks"][g] = e
    json.dump(res, open(OUTD / "e4_benchmarks.json", "w"), indent=1)
    for g in ("hellaswag", "arc_easy", "piqa", "lambada", "blimp"):
        e = res["tasks"][g]
        print(f"{g:10s} n={e['n']:6d} chance {e['chance']:.3f}  acc {e['base_acc'][0]:.3f} [{e['base_acc'][1]:.3f}, "
              f"{e['base_acc'][2]:.3f}]  acc_norm {e['base_acc_norm'][0]:.3f} [{e['base_acc_norm'][1]:.3f}, {e['base_acc_norm'][2]:.3f}]")


if __name__ == "__main__":
    {"score": lambda: stage_score(sys.argv[2:3] == ["base"]), "summarize": stage_summarize}[sys.argv[1]]()
