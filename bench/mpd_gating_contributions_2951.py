"""E5b (#2951): is VPD's gating a short circuit at the level of CONTRIBUTIONS?

A rank-1 piece c at site s reads a = V_c . in_s(t). Given the RMSNorm scales, the attention pattern and the
MLP's GELU gates (s(z) = gelu(z)/z), in_s(t) is exactly linear in the residual stream, and the residual stream
at a position is exactly the embedding plus every upstream residual-writing piece's write U_c' (V_c' . in_s'(t))
plus each writing site's delta (W - (V U)^T) in_s'(t). So a decomposes exactly into per-upstream-piece terms:
  q, k, v (layer l)   a = sum_c' ((V_c . g1) . U_c' / r1(t)) coef_c'(t)            writers of layers < l
  c_fc (layer l)      a = sum_c' ((V_c . g2) . U_c' / r2(t)) coef_c'(t)            writers < l, and o of layer l
  down (layer l)      a = sum_c' ((W_fc^T (V_c . s(z_t))) . g2 . U_c' / r2(t)) coef_c'(t)   (same writers)
  o (layer l)         a = sum_c' sum_h (P_h . U_c') sum_j att_h(t, j) coef_c'(j) / r1(j),  P_h = (W_V^h^T V_c^h) . g1
plus embedding and delta terms; the sum is checked against the actual read value.
Per piece, on its firing tokens in the fit rows (rows 0-63 of masks_vpd4l.npz, <= 16 sampled per piece):
  (a) fan-in: the fewest largest-|term| terms whose sum is within 10% of a, and the fewest carrying 90% of sum |term|
  (b) stability: how often each upstream piece is among a token's fan-in terms; the share of a explained by the
      piece's 10 most frequent upstream pieces (1 - |a - their sum| / |a|), median over tokens
  (c) on a subsample of 24 pieces per site: a logistic threshold law over the terms of the k most frequent
      upstream pieces (MDL-selected, identities log2 #writers bits, weights 1/2 log2 N bits), fitted on rows 0-63
      (all positives + 8192 weighted negatives) and coded on rows 64-127, vs the base rate and vs the full read
      value a as a single feature.
Writes e5b_contributions.json. usage: MPD_MEM_GIB=4 venv python mpd_gating_contributions_2951.py   (E5B_PART=c: only part (c), from the saved json)
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "vpd_2951"))
from vpd_model import VPD_PTH, gelu_tanh, load_target, site_names  # noqa: E402

DEV = "mps"
F_ = Path.home() / "mpd-data/frontier"
Z = np.load(F_ / "masks_vpd4l.npz", mmap_mode="r")
ids = torch.tensor(np.asarray(Z["ids"]))
R, S = ids.shape
half_rows = R // 2
indptr = np.asarray(Z["vpd_indptr"]).astype(np.int64)
IDX = np.asarray(Z["vpd_indices"]).astype(np.int64)
offs = np.asarray(Z["vpd_offsets"]).astype(np.int64)
names = site_names()
L = 4
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
rng = np.random.default_rng(0)

target = load_target(DEV)
raw = torch.load(str(VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
UV = {}
for n in names:
    k = "_components." + n.replace(".", "-")
    UV[n] = (raw[k + ".U"].float().to(DEV), raw[k + ".V"].float().to(DEV))  # U [C, d_out], V [d_in, C]
del raw
H, HD, eps = target.n_head, target.hd, target.eps
g1 = [target.norms[2 * l].detach() for l in range(L)]
g2 = [target.norms[2 * l + 1].detach() for l in range(L)]
site_index = {n: i for i, n in enumerate(names)}
writer_sites = [n for n in names if n.endswith(("o_proj", "down_proj"))]
# writer table: rows = residual-writing pieces, in site order; plus per-site delta and the embedding (extra terms)
w_off = np.cumsum([0] + [UV[n][0].shape[0] for n in writer_sites])
U_all = torch.cat([UV[n][0] for n in writer_sites])  # [Wn, 768]
DELTA = {n: (target.site(n).W - (UV[n][1] @ UV[n][0]).T) for n in writer_sites}
NEXTRA = 1 + len(writer_sites)  # embedding, then one delta per writer site
WN = int(w_off[-1])


def upstream_writers(n):
    """Writer sites feeding piece site n (and the extra-term mask)."""
    l = int(n.split(".")[1])
    kind = n.split(".")[-1]
    if kind in ("q_proj", "k_proj", "v_proj", "o_proj"):
        ok = [w for w in writer_sites if int(w.split(".")[1]) < l]
    else:
        ok = [w for w in writer_sites if int(w.split(".")[1]) < l or w == f"h.{l}.attn.o_proj"]
    return ok


def writer_cols(ws):
    return np.concatenate([np.arange(w_off[writer_sites.index(w)], w_off[writer_sites.index(w) + 1]) for w in ws]) if ws else np.zeros(0, np.int64)


@torch.no_grad()
def row_state(r):
    """Everything per row: site inputs, residuals, norm scales, attention patterns, GELU gates, writer coefs."""
    b = ids[r:r + 1].to(DEV)
    for n in names:
        target.site(n).cache_input = True
    target(b)
    inp = {n: target.site(n).last_input[0] for n in names}
    for n in names:
        target.site(n).cache_input = False
        target.site(n).last_input = None
    x = target.wte[b[0]]
    st = {"in": inp, "emb": x.clone(), "r1": [], "r2": [], "att": [], "gate": []}
    for l in range(L):
        st["r1"].append(torch.sqrt(x.pow(2).mean(-1) + eps))
        q = inp[f"h.{l}.attn.q_proj"] @ target.site(f"h.{l}.attn.q_proj").W.T
        k = inp[f"h.{l}.attn.k_proj"] @ target.site(f"h.{l}.attn.k_proj").W.T
        st["att"].append(target.attention_pattern(q[None], k[None])[0])  # [H, S, S]
        x = x + inp[f"h.{l}.attn.o_proj"] @ target.site(f"h.{l}.attn.o_proj").W.T
        st["r2"].append(torch.sqrt(x.pow(2).mean(-1) + eps))
        z = inp[f"h.{l}.mlp.c_fc"] @ target.site(f"h.{l}.mlp.c_fc").W.T
        st["gate"].append(torch.where(z.abs() > 1e-12, gelu_tanh(z) / z, torch.full_like(z, 0.5)))
        x = x + inp[f"h.{l}.mlp.down_proj"] @ target.site(f"h.{l}.mlp.down_proj").W.T
    st["coef"] = torch.cat([inp[w] @ UV[w][1] for w in writer_sites], 1)  # [S, Wn]
    st["extra"] = torch.stack([st["emb"]] + [inp[w] @ DELTA[w].T for w in writer_sites], 1)  # [S, NEXTRA, 768]
    return st


@torch.no_grad()
def terms(st, n, c, ts, only=None):
    """Exact per-contributor terms of piece c (site n) at tokens ts: (writer terms [T, Wn_up], extra [T, NEXTRA], read a [T], cols).
    `only`: restrict the writer columns to these global writer indices (features for the threshold laws)."""
    l = int(n.split(".")[1])
    kind = n.split(".")[-1]
    U, Vv = UV[n]
    vc = Vv[:, c]
    ws = upstream_writers(n)
    cn = writer_cols(ws)
    if only is not None:
        cn = np.intersect1d(cn, np.asarray(only, dtype=np.int64))
    cols = torch.tensor(cn, device=DEV, dtype=torch.long)
    extra_mask = torch.zeros(NEXTRA, device=DEV)
    extra_mask[0] = 1
    for w in ws:
        extra_mask[1 + writer_sites.index(w)] = 1
    tt = torch.tensor(ts, device=DEV, dtype=torch.long)
    a = st["in"][n][tt] @ vc
    if kind == "o_proj":
        # a = sum_h sum_j att_h(t, j) (P_h . x_j) / r1(j),  P_h = (W_V^h^T V_c^h) . g1
        Wv = target.site(f"h.{l}.attn.v_proj").W  # [768(out: H*HD), 768(in)]
        P = torch.stack([(Wv[h * HD:(h + 1) * HD].T @ vc[h * HD:(h + 1) * HD]) * g1[l] for h in range(H)])  # [H, 768]
        att = st["att"][l][:, tt, :]  # [H, T, S]
        src = st["coef"][:, cols] / st["r1"][l][:, None]  # [S, Wup]
        PU = P @ U_all[cols].T  # [H, Wup]
        wt = torch.einsum("hts,sw,hw->tw", att, src, PU)
        ex_src = st["extra"] / st["r1"][l][:, None, None]  # [S, NEXTRA, 768]
        et = torch.einsum("hts,sed,hd->te", att, ex_src, P) * extra_mask
        return wt, et, a, cols
    if kind in ("q_proj", "k_proj", "v_proj"):
        Rt = (vc * g1[l])[None, :] / st["r1"][l][tt][:, None]  # [T, 768]
    elif kind == "c_fc":
        Rt = (vc * g2[l])[None, :] / st["r2"][l][tt][:, None]
    else:  # down: through the GELU gates of layer l
        Wfc = target.site(f"h.{l}.mlp.c_fc").W  # [3072, 768]
        Rt = ((vc[None, :] * st["gate"][l][tt]) @ Wfc) * g2[l][None, :] / st["r2"][l][tt][:, None]
    wt = (Rt @ U_all[cols].T) * st["coef"][tt][:, cols]
    et = torch.einsum("td,ted->te", Rt, st["extra"][tt]) * extra_mask
    return wt, et, a, cols


def fanin(wt, et, a):
    """(k to within 10% of a, k carrying 90% of |mass|, top contributor ids) per token; extras get ids -1-e."""
    allt = torch.cat([wt, et], 1)
    absd = allt.abs()
    order = absd.argsort(1, descending=True)
    srt = allt.gather(1, order)
    cs = srt.cumsum(1)
    k10 = ((cs - a[:, None]).abs() <= 0.1 * a.abs()[:, None]).float().argmax(1) + 1
    cm = absd.gather(1, order).cumsum(1)
    k90 = (cm >= 0.9 * cm[:, -1:]).float().argmax(1) + 1
    return k10.cpu().numpy(), k90.cpu().numpy(), order.cpu().numpy()


# ---------------------------------------------------------------- sample firing tokens per piece (fit rows)
N_fit = half_rows * S
lens = np.diff(indptr)
pos_of = np.repeat(np.arange(len(lens)), lens)
fit_e = pos_of < N_fit
cnt_fit = np.bincount(IDX[fit_e], minlength=int(offs[-1]))
pieces = np.nonzero(cnt_fit >= 20)[0]
log(f"{len(pieces)} pieces with >= 20 firing tokens in the fit rows")
order_e = np.lexsort((pos_of[fit_e], IDX[fit_e]))
p_fit, c_fit = pos_of[fit_e][order_e], IDX[fit_e][order_e]
starts = np.searchsorted(c_fit, np.arange(int(offs[-1]) + 1))
samples = {}  # row -> list of (pos, global piece)
for gc in pieces:
    ps = p_fit[starts[gc]:starts[gc + 1]]
    for p in rng.choice(ps, size=min(16, len(ps)), replace=False):
        samples.setdefault(int(p // S), []).append((int(p % S), int(gc)))

PART_C = __import__("os").environ.get("E5B_PART") == "c"  # rerun only (c) from the saved (a)/(b) results
if PART_C:
    res = json.load(open(F_ / "e5b_contributions.json"))
    stats = {int(k): {"site": v["site"], "top10": v["top10"]} for k, v in res["pieces"].items()}
else:
    stats = {}  # piece -> dict
    sanity = []
    for r in sorted(samples):
        st = row_state(r)
        by_piece = {}
        for p, gc in samples[r]:
            by_piece.setdefault(gc, []).append(p)
        for gc, ts in by_piece.items():
            si = int(np.searchsorted(offs, gc, side="right") - 1)
            n, c = names[si], int(gc - offs[si])
            wt, et, a, cols = terms(st, n, c, ts)
            tot = wt.sum(1) + et.sum(1)
            sanity.append(float(((tot - a).abs() / a.abs().clamp_min(1e-6)).max()))
            k10, k90, order = fanin(wt, et, a)
            Wup = wt.shape[1]
            d = stats.setdefault(gc, {"site": n, "k10": [], "k90": [], "freq": {}, "terms": []})
            d["k10"] += k10.tolist()
            d["k90"] += k90.tolist()
            colsn = cols.cpu().numpy()
            allt = torch.cat([wt, et], 1).cpu().numpy()
            an = a.cpu().numpy()
            keyof = lambda j: int(colsn[j]) if j < Wup else -1 - int(j - Wup)
            for i in range(len(ts)):
                for j in order[i, :k10[i]]:
                    key = keyof(j)
                    d["freq"][key] = d["freq"].get(key, 0) + 1
                # keep each token's 64 largest terms (sparse) for the coverage computation
                top = order[i, :64]
                d["terms"].append((np.array([keyof(j) for j in top], dtype=np.int32), allt[i, top].astype(np.float32), float(an[i])))
        del st
        torch.mps.empty_cache()
        if r % 8 == 0:
            log(f"row {r}: {len(stats)} pieces so far, max decomposition rel. error {max(sanity):.2e}")

    # (b) stability: share of a explained by the piece's 10 most frequent upstream contributors
    per_site = {}
    for gc, d in stats.items():
        top10 = [k for k, _ in sorted(d["freq"].items(), key=lambda kv: -kv[1])[:10]]
        covers = []
        t10 = np.array(top10, dtype=np.int32)
        for keys, vals, a in d["terms"]:  # terms outside a token's 64 largest count as 0 (they are the smallest)
            s10 = float(vals[np.isin(keys, t10)].sum())
            covers.append(1 - abs(a - s10) / max(abs(a), 1e-9))
        d["cover10_median"] = float(np.median(covers))
        d["top10"] = top10
        nsamp = len(d["k10"])
        d["top1_share"] = max(d["freq"].values()) / nsamp
        del d["terms"]
        ps = per_site.setdefault(d["site"], {"k10": [], "k90": [], "cover10": [], "top1_share": []})
        ps["k10"] += d["k10"]
        ps["k90"] += d["k90"]
        ps["cover10"].append(d["cover10_median"])
        ps["top1_share"].append(d["top1_share"])

    summary = {}
    for n in names:
        if n not in per_site:
            continue
        ps = per_site[n]
        k10 = np.array(ps["k10"])
        summary[n] = {"tokens": int(len(k10)), "pieces": len(ps["cover10"]),
                      "fanin_k_within10pct": {"median": float(np.median(k10)), "p90": float(np.percentile(k10, 90)),
                                              "share<=3": float((k10 <= 3).mean()), "share<=10": float((k10 <= 10).mean())},
                      "fanin_k_90pct_mass": {"median": float(np.median(ps["k90"])), "p90": float(np.percentile(ps["k90"], 90))},
                      "cover_by_10_most_frequent_median": float(np.median(ps["cover10"])),
                      "most_frequent_contributor_share_median": float(np.median(ps["top1_share"]))}
        log(f"{n}: fan-in median {summary[n]['fanin_k_within10pct']['median']:.0f} (p90 {summary[n]['fanin_k_within10pct']['p90']:.0f}), "
            f"top-10-frequent cover {summary[n]['cover_by_10_most_frequent_median']:.2f}")
    res = {"description": __doc__.split("Writes")[0], "rows": int(R), "fit_rows": "0-63", "coded_rows": "64-127",
           "decomposition_max_rel_error": float(max(sanity)), "per_site": summary,
           "per_layer": {}, "pieces": {int(k): {kk: v for kk, v in d.items() if kk in ("site", "top10", "cover10_median", "top1_share")}
                                      | {"k10_median": float(np.median(d["k10"]))} for k, d in stats.items()}}
    for l in range(L):
        k10 = np.concatenate([per_site[n]["k10"] for n in per_site if n.startswith(f"h.{l}.")]) if any(n.startswith(f"h.{l}.") for n in per_site) else np.zeros(0)
        if len(k10):
            res["per_layer"][l] = {"fanin_histogram": {str(b): int(c) for b, c in zip(*np.unique(np.minimum(k10, 50), return_counts=True))},
                                   "median": float(np.median(k10))}
    json.dump(res, open(F_ / "e5b_contributions.json", "w"), indent=1)
    log("(a)/(b) written")

# ---------------------------------------------------------------- (c) threshold laws on a subsample
law_pieces = []
for n in names:
    cand = [gc for gc, d in stats.items() if d["site"] == n]
    law_pieces += list(rng.choice(cand, size=min(24, len(cand)), replace=False)) if cand else []
law_pieces = [int(g) for g in law_pieces]
KMAX = 12
feat = {gc: [k for k in stats[gc]["top10"]][:KMAX] for gc in law_pieces}
# positives per piece (all rows), from one piece-sorted copy of the entries
order_all = np.lexsort((pos_of, IDX))
p_all, c_all = pos_of[order_all], IDX[order_all]
st_all = np.searchsorted(c_all, np.arange(int(offs[-1]) + 1))
pos_rows = {gc: np.sort(p_all[st_all[gc]:st_all[gc + 1]]) for gc in law_pieces}  # sorted arrays, not sets (memory)
del order_all, p_all, c_all
NEG = 8192
neg_fit = {gc: np.sort(rng.choice(N_fit, size=NEG, replace=False)) for gc in law_pieces}
fitX = {gc: [] for gc in law_pieces}
fitY = {gc: [] for gc in law_pieces}
fitW = {gc: [] for gc in law_pieces}
law = {}


@torch.no_grad()
def features(st, gc, ts):
    """[T, k+1]: the terms of the piece's most frequent upstream contributors, then the read value a."""
    si = int(np.searchsorted(offs, gc, side="right") - 1)
    n, c = names[si], int(gc - offs[si])
    wt, et, a, cols = terms(st, n, c, ts, only=[k for k in feat[gc] if k >= 0])
    colsn = cols.cpu().numpy()
    out = []
    for k in feat[gc]:
        if k < 0:
            out.append(et[:, -1 - k])
        else:
            j = np.nonzero(colsn == k)[0]
            out.append(wt[:, int(j[0])] if len(j) else torch.zeros_like(a))
    return torch.stack(out + [a], 1).cpu().numpy()


def logistic(X, y, w, ridge=1e-2, iters=60):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    beta = np.zeros(Xb.shape[1])
    Rg = ridge * np.eye(Xb.shape[1])
    Rg[-1, -1] = 0
    for _ in range(iters):
        z = np.clip(Xb @ beta, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = Xb.T @ (w * (p - y)) + Rg @ beta
        Hh = (Xb * (w * p * (1 - p))[:, None]).T @ Xb + Rg + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(Hh, g)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def nll_bits(X, y, w, beta):
    z = np.clip(np.hstack([X, np.ones((len(X), 1))]) @ beta, -30, 30)
    return float((w * (np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y))).sum() / math.log(2))


for r in range(half_rows):  # fit rows: all positives + sampled negatives (weighted)
    st = row_state(r)
    for gc in law_pieces:
        P, Ng = pos_rows[gc], neg_fit[gc]
        pp = P[(P >= r * S) & (P < (r + 1) * S)] - r * S
        nn = Ng[(Ng >= r * S) & (Ng < (r + 1) * S)] - r * S
        ts = np.union1d(pp, nn)
        if len(ts) == 0:
            continue
        X = features(st, gc, ts.tolist())
        y = np.isin(ts, pp).astype(np.float64)
        wneg = (N_fit - np.count_nonzero(P < N_fit)) / NEG
        fitX[gc].append(X)
        fitY[gc].append(y)
        fitW[gc].append(np.where(y > 0, 1.0, wneg))
    del st
    torch.mps.empty_cache()
log("fit features collected")
LOG2N = math.log2(N_fit)
for gc in law_pieces:
    X, y, w = np.concatenate(fitX[gc]), np.concatenate(fitY[gc]), np.concatenate(fitW[gc])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    nterm = Xs.shape[1] - 1
    # greedy MDL over the k most frequent contributors' terms
    chosen, beta = [], logistic(Xs[:, []], y, w)
    best = nll_bits(Xs[:, []], y, w, beta) + 0.5 * LOG2N
    while True:
        trial = []
        for j in range(nterm):
            if j in chosen:
                continue
            b_ = logistic(Xs[:, chosen + [j]], y, w)
            trial.append((nll_bits(Xs[:, chosen + [j]], y, w, b_) + (len(chosen) + 1) * math.log2(WN + NEXTRA) + (len(chosen) + 2) * 0.5 * LOG2N, j, b_))
        if not trial:
            break
        mdl, j, b_ = min(trial)
        if mdl >= best:
            break
        best, beta = mdl, b_
        chosen.append(j)
    beta_a = logistic(Xs[:, [nterm]], y, w)
    law[gc] = {"site": stats[gc]["site"], "chosen": chosen, "beta": beta, "beta_a": beta_a, "mu": mu, "sd": sd,
               "p_base": (y * w).sum() / w.sum(), "test": {"base": 0.0, "law": 0.0, "read_value": 0.0},
               "model_bits": len(chosen) * math.log2(WN + NEXTRA) + (len(chosen) + 1) * 0.5 * LOG2N}
for r in range(half_rows, R):  # coded rows: every token
    st = row_state(r)
    ts = list(range(S))
    for gc in law_pieces:
        d = law[gc]
        Xs = (features(st, gc, ts) - d["mu"]) / d["sd"]
        P = pos_rows[gc]
        y = np.isin(np.arange(S), P[(P >= r * S) & (P < (r + 1) * S)] - r * S).astype(np.float64)
        w = np.ones(S)
        pb = d["p_base"]
        d["test"]["base"] += float(-(y * math.log2(pb) + (1 - y) * math.log2(1 - pb)).sum())
        d["test"]["law"] += nll_bits(Xs[:, d["chosen"]], y, w, d["beta"])
        d["test"]["read_value"] += nll_bits(Xs[:, [Xs.shape[1] - 1]], y, w, d["beta_a"])
    del st
    torch.mps.empty_cache()
Nte = (R - half_rows) * S
laws_out = {}
for gc, d in law.items():
    laws_out[gc] = {"site": d["site"], "k": len(d["chosen"]), "model_bits": d["model_bits"],
                    "test_bits_per_token": {k: v / Nte for k, v in d["test"].items()},
                    "saving_vs_base": 1 - d["test"]["law"] / max(d["test"]["base"], 1e-9),
                    "read_value_saving_vs_base": 1 - d["test"]["read_value"] / max(d["test"]["base"], 1e-9)}
ks = np.array([v["k"] for v in laws_out.values()])
sv = np.array([v["saving_vs_base"] for v in laws_out.values()])
sa = np.array([v["read_value_saving_vs_base"] for v in laws_out.values()])
res["threshold_laws"] = {
    "pieces": len(laws_out), "k_histogram": {str(a): int(b) for a, b in zip(*np.unique(ks, return_counts=True))},
    "median_saving_vs_base": float(np.median(sv)), "share_saving>=50%": float((sv >= 0.5).mean()),
    "share_saving>=50%_with_k<=3": float(((sv >= 0.5) & (ks <= 3)).mean()),
    "read_value_median_saving_vs_base": float(np.median(sa)),
    "per_site_median_saving": {n: float(np.median([v["saving_vs_base"] for v in laws_out.values() if v["site"] == n]))
                               for n in names if any(v["site"] == n for v in laws_out.values())},
    "records": laws_out}
json.dump(res, open(F_ / "e5b_contributions.json", "w"), indent=1)
log("done: " + json.dumps({k: v for k, v in res["threshold_laws"].items() if k != "records"}))
