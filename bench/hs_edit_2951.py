"""HellaSwag through a fitted decomposition (#2951): a pre-registered search for one edit of VPD's 4-layer target
that raises HellaSwag, made by scaling one subcomponent of the fitted library of h.2.mlp.down_proj, with the same
search run on random subcomponents and random directions to show what a best-of-N gain looks like by chance, and a
LoRA trained on the same data as the baseline.

PROTOCOL (pre-registered: this docstring and the constants below, at the commit that adds them, are fixed before the
validation set is touched; the validation set is scored once, by the final stage):
  data     HellaSwag train (39,905 items) shuffled once by random.Random(2951): fit = the first 4000 items, select =
           the next 2000, confirm = the next 2000. Each item tokenized as e4_benchmarks_data.py tokenizes the
           validation set (query = preprocess(activity_label + ": " + ctx_a + " " + ctx_b.capitalize()), choices
           " " + preprocess(ending), the continuation = encode(query + choice) beyond encode(query)).
  score    per item, the correct ending's log-probability share lp(gold) - logsumexp_j lp(j), lp the summed
           log-probability of an ending's tokens (the harness's HellaSwag margin); J = its mean over items.
  library  the fitted decomposition of h.2.mlp.down_proj, methods/decomp/lib/blocks.2.down_proj.{v,u}.f64 (3840
           subcomponents, gam_mpd::site_fit on vpd4l_e2e_train; sha256 in LIBRARY_SHA256).
  family   scale one subcomponent: W' = W + (lam - 1) u_c v_c^T, lam in {0, 0.5, 1.5, 2}.
  screen   G = dJ_fit/dW at the native weights. Subcomponent c's first-order gain at |lam - 1| = 1 is s_c = u_c^T G v_c.
           The K = 8 subcomponents of largest |s_c| go on, each with the two grid values on the side sign(s_c) points
           to ({1.5, 2} if s_c > 0, else {0.5, 0}): 16 candidates.
  guard    (during selection) ordinary-text KL(edited || native) on Pile validation rows 3000-3063
           (pile_val_4096x513, every position) <= 1e-3 nats per token.
  select   the candidate with the largest mean change in J on select among those passing the guard.
  confirm  that candidate's mean change in J on confirm, with its 95% interval (reported; nothing is chosen on it).
  null     the same screen, guard, select and confirm with (a) K subcomponents drawn uniformly at random instead of the
           K of largest |s_c|, and (b) a random library: per subcomponent a Gaussian rank-one pair (u', v') with
           |u'| = |u_c| and |v'| = |v_c|, screened by |u'^T G v'|; 10 seeds each.
  baseline a rank-one LoRA on h.2.mlp.down_proj (3840 parameters; the edit has one scale and one choice of
           subcomponent) trained on fit to maximise J: AdamW lr 1e-3, 32 items per step, 300 steps, the step kept the
           one of largest J on select, checked every 50 steps.
  final    the chosen edit and the LoRA, once, on the harness: e4_benchmarks_data.py with E4_VARIANTS=hs_edit
           (HellaSwag validation acc, acc_norm and margin with 95% intervals; guards ARC-Easy, PIQA, LAMBADA, BLiMP)
           and e4_side_effects_data.py with E4_VARIANTS=hs_edit (ordinary-text KL on 4096 held-out Pile rows).
           Guards on the final evaluation: every guard benchmark's margin change has its 95% interval's upper end
           >= -0.01 nats, and the ordinary-text KL <= 1e-3 nats per token; an edit violating one is reported failed.

Stages (outputs in ~/mpd-data/frontier/hs_edit/):
  search   the fit gradient, the real search, both nulls, the confirm split, the LoRA; writes search.json and the
           harness set methods/hs_edit.{pt,json}
  explain  what the chosen subcomponent does: where the library turns it on in its training text and what it reads
           there, what its write promotes through the unembedding, and its validation effect by HellaSwag source and
           category and against ending length (after the final harness run)

usage: MPD_MEM_GIB=6 venv/python hs_edit_2951.py search | explain
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e4_side_effects_data as E  # noqa: E402

OUT = E.FR / "hs_edit"
OUT.mkdir(parents=True, exist_ok=True)
LIB = E.FR / "e4_side/methods/decomp/lib"
LIBRARY_SHA256 = {"v": "49ffea3d7faa4900d2dfae0ad79d29db802f50e37410bab3dddea77feff6e872",
                  "u": "a513097e7af6e4298102910014f05c2c4c3f7e189ffb033c60e8bfdfd88f3b97"}
SEED, N_FIT, N_SELECT, N_CONFIRM = 2951, 4000, 2000, 2000
GRID, K, KL_GUARD, NULL_SEEDS = (0.0, 0.5, 1.5, 2.0), 8, 1e-3, 10
LORA_LR, LORA_ITEMS, LORA_STEPS, LORA_EVERY = 1e-3, 32, 300, 50
KL_ROWS = range(3000, 3064)
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)


def items():
    """The fit, select and confirm splits of HellaSwag train: per item (requests [(ids, n_cont)], gold, row)."""
    from datasets import load_dataset
    from e4_benchmarks_data import CTX, hs_pre
    tok = E.tokenizer()
    enc = lambda s: tok.encode(s).ids
    ds = load_dataset("Rowan/hellaswag", split="train")
    order = list(range(len(ds)))
    random.Random(SEED).shuffle(order)
    out = []
    for i in order[:N_FIT + N_SELECT + N_CONFIRM]:
        d = ds[i]
        q = hs_pre(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize())
        reqs = []
        for e in d["endings"]:
            ctx, cont = q, " " + hs_pre(e)
            n_sp = len(ctx) - len(ctx.rstrip())
            if n_sp:
                ctx, cont = ctx[:-n_sp], ctx[-n_sp:] + cont
            whole, c_enc = enc(ctx + cont), enc(ctx)
            reqs.append((np.array(whole[-(CTX + 1):], np.int64), len(whole) - len(c_enc)))
        out.append((reqs, int(d["label"]), i))
    return out[:N_FIT], out[N_FIT:N_FIT + N_SELECT], out[N_FIT + N_SELECT:]


def batches(split, per=16):
    """Length-sorted batches of `per` items: (ids [4 per, L], continuation positions and tokens, item golds, index)."""
    import torch
    order = np.argsort([max(len(r[0]) for r in it[0]) for it in split])
    for b in range(0, len(order), per):
        idx = order[b:b + per]
        reqs = [r for i in idx for r in split[i][0]]
        L = -(-max(len(r[0]) for r in reqs) // 32) * 32  # few distinct shapes: MPS caches a graph per shape
        ids = torch.zeros(len(reqs), L, dtype=torch.long)
        rows, pos, toks = [], [], []
        for j, (t, n) in enumerate(reqs):
            ids[j, :len(t)] = torch.from_numpy(t)
            rows += [j] * n
            pos += list(range(len(t) - n - 1, len(t) - 1))
            toks += t[len(t) - n:].tolist()
        yield ids, torch.tensor(rows), torch.tensor(pos), torch.tensor(toks), torch.tensor([split[i][1] for i in idx]), idx


class Model:
    """VPD's target split at h.2.mlp.down_proj: the shared part once per batch, every edit of that site after it."""

    def __init__(self):
        import torch
        self.torch = torch
        self.target, _, _ = E.load()
        self.W0 = self.target.site(E.SITE).W.clone()
        self.resid, self.final, self.head = E.split_forward(self.target)

    def shares(self, x, rows, pos, toks, golds):
        """Per item, the gold ending's log-probability share, from the last residual x [4 items, L, d]."""
        torch = self.torch
        lp = self.head(x[rows, pos]).gather(-1, toks[:, None].to(x.device))[:, 0]
        per = torch.zeros(x.shape[0], device=x.device).index_add_(0, rows.to(x.device), lp).view(-1, 4)
        return per.gather(1, golds[:, None].to(x.device))[:, 0] - torch.logsumexp(per, 1)

    def evaluate(self, split, edits):
        """Per item and edit (column 0 the native model), the score; each edit a function of (y0, g2) -> site
        output change."""
        torch = self.torch
        out = np.zeros((len(split), 1 + len(edits)))
        with torch.no_grad():
            for ids, rows, pos, toks, golds, idx in batches(split):
                xmid, g2 = self.resid(ids.to(E.DEVICE))
                y0 = xmid + g2 @ self.W0.T
                out[idx, 0] = self.shares(self.final.after(y0), rows, pos, toks, golds).cpu().numpy()
                for k, edit in enumerate(edits):
                    out[idx, 1 + k] = self.shares(self.final.after(y0 + edit(g2)), rows, pos, toks, golds).cpu().numpy()
                E.empty_cache()
        return out

    def gradient(self, split):
        """dJ/dW at the native weights over `split`."""
        torch = self.torch
        W = self.W0.clone().requires_grad_(True)
        for ids, rows, pos, toks, golds, idx in batches(split, per=4):
            with torch.no_grad():
                xmid, g2 = self.resid(ids.to(E.DEVICE))
            s = self.shares(self.final.after(xmid + g2 @ W.T), rows, pos, toks, golds)
            (s.sum() / len(split)).backward()
            E.empty_cache()
        return W.grad.cpu().double().numpy()

    def kl(self, edit):
        """Mean KL(edited || native) per token over the guard's Pile rows."""
        torch = self.torch
        rows = np.asarray(np.load(E.VD / "pile_val_4096x513.npy", mmap_mode="r")[KL_ROWS.start:KL_ROWS.stop, :512]).astype(np.int64)
        tot = cnt = 0.0
        with torch.no_grad():
            for i in range(0, len(rows), 2):
                xmid, g2 = self.resid(torch.from_numpy(rows[i:i + 2]).to(E.DEVICE))
                y0 = xmid + g2 @ self.W0.T
                xb, xe = self.final.after(y0), self.final.after(y0 + edit(g2))
                for p0 in range(0, 512, 128):
                    lb, le = self.head(xb[:, p0:p0 + 128]), self.head(xe[:, p0:p0 + 128])
                    tot += float((le.exp() * (le - lb)).sum())
                    cnt += lb.shape[0] * lb.shape[1]
                E.empty_cache()
        return tot / cnt


def rank_one(m, u, v, a):
    """The site-output change of W + a u v^T."""
    torch = m.torch
    uu, vv = torch.from_numpy(u.astype(np.float32)).to(E.DEVICE), torch.from_numpy(v.astype(np.float32)).to(E.DEVICE)
    return lambda g2: a * (g2 @ vv)[..., None] * uu


def interval(d, B=1000):
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return [float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]


def search_one(m, select, confirm, U, V, scores, chosen):
    """The protocol's selection over the subcomponents `chosen` of (U, V): two grid values each on the side its
    first-order gain points to; the best select gain passing the guard, and its confirm gain."""
    cands = [(c, lam) for c in chosen for lam in (GRID[2:] if scores[c] > 0 else GRID[:2])]
    sel = m.evaluate(select, [rank_one(m, U[c], V[c], lam - 1) for c, lam in cands])
    gains = sel[:, 1:].mean(0) - sel[:, 0].mean()
    for k in np.argsort(-gains):
        c, lam = cands[k]
        kl = m.kl(rank_one(m, U[c], V[c], lam - 1))
        if kl <= KL_GUARD:
            conf = m.evaluate(confirm, [rank_one(m, U[c], V[c], lam - 1)])
            return {"subcomponent": int(c), "lam": lam, "first_order": float(scores[c]), "select_gain": float(gains[k]),
                    "kl": kl, "confirm_gain": interval(conf[:, 1] - conf[:, 0]),
                    "candidates": [{"c": int(c), "lam": lam, "select_gain": float(g)} for (c, lam), g in zip(cands, gains)]}
    return {"subcomponent": None, "candidates": [{"c": int(c), "lam": lam, "select_gain": float(g)} for (c, lam), g in zip(cands, gains)]}


def train_lora(m, fit, select):
    """The baseline: a rank-one LoRA on the site maximising J on fit, kept at its best select J."""
    torch = m.torch
    torch.manual_seed(0)
    A = (torch.randn(1, m.W0.shape[1]) * 0.01).to(E.DEVICE).requires_grad_(True)
    B = torch.zeros(m.W0.shape[0], 1, device=E.DEVICE, requires_grad=True)
    opt = torch.optim.AdamW([A, B], lr=LORA_LR)
    rng = np.random.default_rng(SEED)
    best, kept, trace = -np.inf, None, []
    for step in range(LORA_STEPS + 1):
        if step % LORA_EVERY == 0:
            a, b = A.detach().clone(), B.detach().clone()
            sel = m.evaluate(select, [lambda g2, a=a, b=b: (g2 @ a.T) @ b.T])
            gain = float(sel[:, 1].mean() - sel[:, 0].mean())
            trace.append({"step": step, "select_gain": gain})
            log(f"LoRA step {step}: select gain {gain:+.4f}")
            if gain > best:
                best, kept = gain, (a, b, step)
        if step == LORA_STEPS:
            break
        part = [fit[i] for i in rng.choice(len(fit), LORA_ITEMS, replace=False)]
        opt.zero_grad()
        for ids, rows, pos, toks, golds, idx in batches(part, per=4):
            with torch.no_grad():
                xmid, g2 = m.resid(ids.to(E.DEVICE))
            s = m.shares(m.final.after(xmid + g2 @ m.W0.T + (g2 @ A.T) @ B.T), rows, pos, toks, golds)
            (-s.sum() / LORA_ITEMS).backward()
        opt.step()
        E.empty_cache()
    return kept, trace


def stage_search():
    import torch
    lib_sha = {s: hashlib.sha256(open(LIB / f"blocks.2.down_proj.{s}.f64", "rb").read()).hexdigest() for s in ("v", "u")}
    assert lib_sha == LIBRARY_SHA256, lib_sha
    V = np.fromfile(LIB / "blocks.2.down_proj.v.f64").reshape(-1, 3072)
    U = np.fromfile(LIB / "blocks.2.down_proj.u.f64").reshape(-1, 768)
    fit, select, confirm = items()
    log(f"{len(fit)} fit, {len(select)} select, {len(confirm)} confirm items")
    m = Model()
    G = m.gradient(fit)
    s = np.einsum("cd,de,ce->c", U, G, V)
    log(f"gradient: |G| {np.linalg.norm(G):.4g}; first-order gains: max {s.max():+.4f}, min {s.min():+.4f}")
    res = {"protocol": __doc__, "n": [len(fit), len(select), len(confirm)]}
    res["real"] = search_one(m, select, confirm, U, V, s, np.argsort(-np.abs(s))[:K])
    log(f"real: {json.dumps({k: v for k, v in res['real'].items() if k != 'candidates'})}")
    json.dump(res, open(OUT / "search.json", "w"), indent=1)
    res["null_random_subcomponents"], res["null_random_directions"] = [], []
    nu, nv = np.linalg.norm(U, axis=1), np.linalg.norm(V, axis=1)
    for seed in range(NULL_SEEDS):
        rng = np.random.default_rng(seed)
        res["null_random_subcomponents"].append(search_one(m, select, confirm, U, V, s, rng.choice(len(U), K, replace=False)))
        Ur, Vr = rng.standard_normal(U.shape), rng.standard_normal(V.shape)
        Ur *= (nu / np.linalg.norm(Ur, axis=1))[:, None]
        Vr *= (nv / np.linalg.norm(Vr, axis=1))[:, None]
        sr = np.einsum("cd,de,ce->c", Ur, G, Vr)
        res["null_random_directions"].append(search_one(m, select, confirm, Ur, Vr, sr, np.argsort(-np.abs(sr))[:K]))
        log(f"null seed {seed}: random subcomponents confirm {res['null_random_subcomponents'][-1].get('confirm_gain')}, "
            f"random directions confirm {res['null_random_directions'][-1].get('confirm_gain')}")
        json.dump(res, open(OUT / "search.json", "w"), indent=1)
    (a, b, step), trace = train_lora(m, fit, select)
    conf = m.evaluate(confirm, [lambda g2: (g2 @ a.T) @ b.T])
    res["lora"] = {"step": step, "trace": trace, "kl": m.kl(lambda g2: (g2 @ a.T) @ b.T),
                   "confirm_gain": interval(conf[:, 1] - conf[:, 0])}
    log(f"LoRA: step {step}, confirm {res['lora']['confirm_gain']}, kl {res['lora']['kl']:.3g}")
    json.dump(res, open(OUT / "search.json", "w"), indent=1)
    edits, meta = {}, {}
    if res["real"]["subcomponent"] is not None:
        c, lam = res["real"]["subcomponent"], res["real"]["lam"]
        edits["hs_subcomponent"] = torch.from_numpy(((lam - 1) * np.outer(U[c], V[c])).astype(np.float32))
        meta["hs_subcomponent"] = {"method": "scale one subcomponent", "subcomponent": c, "lam": lam, "p_fire": float("nan")}
    edits["hs_lora"] = (b @ a).cpu()
    meta["hs_lora"] = {"method": "rank-one LoRA", "step": step, "p_fire": float("nan")}
    torch.save(edits, E.FR / "e4_side/methods/hs_edit.pt")
    json.dump({"meta": meta, "pairs": []}, open(E.FR / "e4_side/methods/hs_edit.json", "w"), indent=1)
    log(f"wrote the harness set: {list(edits)}")


def stage_explain():
    """What the chosen subcomponent c does (descriptive; after the single validation evaluation): where the library
    turns it on in its 16 training rows and what it reads there, which hidden units its read weighs, which tokens
    its write raises and lowers on the direct path to the unembedding (final-norm gain applied, layer 3 skipped), and
    its validation effect on the score by HellaSwag source and activity, and against ending length."""
    import torch
    from datasets import load_dataset
    from e4_benchmarks_data import BENCH, build_requests
    res = json.load(open(OUT / "search.json"))
    c, lam = res["real"]["subcomponent"], res["real"]["lam"]
    V = np.fromfile(LIB / "blocks.2.down_proj.v.f64").reshape(-1, 3072)
    U = np.fromfile(LIB / "blocks.2.down_proj.u.f64").reshape(-1, 768)
    tok = E.tokenizer()
    txt = lambda ids: tok.decode([int(i) for i in ids])
    out = {"subcomponent": c, "lam": lam}
    w = V[c] ** 2 / (V[c] ** 2).sum()
    top = np.argsort(-w)[:8]
    out["reads_units"] = [{"unit": int(j), "weight": float(V[c, j]), "share": float(w[j])} for j in top]
    m = Model()
    gain = m.target.ln_f.detach().cpu().double().numpy()
    logit = m.target.wte.detach().cpu().double().numpy() @ (gain * U[c])
    out["write_raises"] = [[txt([i]), float(logit[i])] for i in np.argsort(-logit)[:15]]
    out["write_lowers"] = [[txt([i]), float(logit[i])] for i in np.argsort(logit)[:15]]
    rows = np.asarray(np.load(E.VD / "pile_val_4096x513.npy", mmap_mode="r")[2048:2064, :512]).astype(np.int64)
    sets = json.load(open(LIB / "blocks.2.down_proj.train.sets.json"))["sets"]
    on = np.array([c in set(st) for st in sets]).reshape(16, 512)
    with torch.no_grad():
        a = np.concatenate([(m.resid(torch.from_numpy(rows[i:i + 2]).to(E.DEVICE))[1] @ torch.from_numpy(
            V[c].astype(np.float32)).to(E.DEVICE)).cpu().numpy() for i in range(0, 16, 2)])
    r, q = np.nonzero(on)
    order = np.argsort(-np.abs(a[r, q]))
    out["on_rate"] = float(on.mean())
    out["on_tokens"] = sorted(((txt([t]), int(n)) for t, n in zip(*np.unique(rows[r, q], return_counts=True))), key=lambda x: -x[1])[:15]
    out["on_contexts"] = [{"context": txt(rows[i, max(0, p - 12):p]), "token": txt([rows[i, p]]), "next": txt([rows[i, p + 1]]),
                           "read": float(a[i, p])} for i, p in zip(r[order[:20]], q[order[:20]])]
    out["read_on_vs_off"] = [float(np.abs(a[on]).mean()), float(np.abs(a[~on]).mean())]
    # the validation effect, from the harness's per-request scores
    rq = build_requests()
    hs = np.nonzero(rq["task"] == "hellaswag")[0]
    base = np.load(BENCH / "scores_base.npz")["lp"][:, 0]
    z = np.load(E.FR / "e4_side/methods/hs_edit/bench/scores.npz")
    ed = z["lp"][:, list(z["names"]).index("hs_subcomponent")]
    ds = load_dataset("Rowan/hellaswag", split="validation")
    margin = lambda lp, it: lp[it][int(rq["gold"][k])] - np.logaddexp.reduce(lp[it])
    d, src, act, dlen, dlp, nlen = [], [], [], [], [], []
    for k, item in zip(hs, ds):
        lo, n = int(rq["req_lo"][k]), int(rq["req_n"][k])
        it = slice(lo, lo + n)
        d.append(margin(ed, it) - margin(base, it))
        src.append(item["source_id"].split("~")[0])
        act.append(item["activity_label"])
        nlen += rq["n_cont"][it].tolist()
        dlp += (ed[it] - base[it]).tolist()
        dlen.append(int(rq["n_cont"][lo + np.argmax(ed[it])]) - int(rq["n_cont"][lo + np.argmax(base[it])]))
    d, src, act = np.array(d), np.array(src), np.array(act)
    out["val_by_source"] = {s_: interval(d[src == s_]) + [int((src == s_).sum())] for s_ in sorted(set(src))}
    cats = [x for x in set(act) if (act == x).sum() >= 40]
    by = {x: interval(d[act == x]) + [int((act == x).sum())] for x in cats}
    out["val_by_activity"] = dict(sorted(by.items(), key=lambda kv: -kv[1][0]))
    nlen, dlp = np.array(nlen), np.array(dlp)
    out["ending_length"] = {"corr_dlp_vs_tokens": float(np.corrcoef(nlen, dlp)[0, 1]),
                            "dlp_per_token": float(np.polyfit(nlen, dlp, 1)[0]),
                            "chosen_length_change": interval(np.array(dlen, float))}
    json.dump(out, open(OUT / "explain.json", "w"), indent=1)
    print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    {"search": stage_search, "explain": stage_explain}[sys.argv[1]]()
