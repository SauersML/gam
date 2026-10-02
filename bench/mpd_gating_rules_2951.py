"""E5 (#2951): is VPD's gating a short function of the decomposition itself?

For every VPD piece c (masks_vpd4l.npz, rounded gates, 128 rows: fit on rows 0-63, code rows 64-127), predict
"c on at position t" from
  now-features   pieces on at t at upstream sites (site index below c's: lower layers, and earlier sites of
                 c's own layer in the order q, k, v, o, c_fc, down)
  prev-features  any piece on at t-1 (attention carries earlier positions)
  token-features the token id at t
with a sparse logistic rule chosen by two-part MDL: greedy forward selection over the 12 candidates with the
highest single-feature mutual information, each feature charged log2(#features) bits and each coefficient
(1/2) log2(N_train) bits, stopping when the total stops falling. Pieces with < 20 training positives keep
the marginal rate. Every rule is then scored on the held-out rows.

Writes e5_gating_rules.json: bits/token for the active sets (rules vs marginal vs VPD's independent code),
rule-size histogram, how many pieces have a short rule, fan-in, and per-piece records.
usage: MPD_MEM_GIB=8 venv python mpd_gating_rules_2951.py
"""

import json
import math
import time
from pathlib import Path

import numpy as np

F = Path.home() / "mpd-data/frontier"
Z = np.load(F / "masks_vpd4l.npz")
ids = Z["ids"]
R, S = ids.shape
N = R * S
half = (R // 2) * S
Ntr, Nte = half, N - half
tok = ids.reshape(-1).astype(np.int64)
pos_in_row = np.tile(np.arange(S), R)
indptr, indices, offs = Z["vpd_indptr"], Z["vpd_indices"].astype(np.int64), Z["vpd_offsets"]
C = int(offs[-1])
V = int(tok.max()) + 1
NF = 2 * C + V  # feature alphabet: now pieces, previous pieces, tokens
LOG2N = math.log2(Ntr)
lens = np.diff(indptr)
pos_of = np.repeat(np.arange(N), lens)
site_of_piece = np.searchsorted(offs, np.arange(C), side="right") - 1
t0 = time.time()

# postings: positions where each piece is on (sorted), and where each token occurs
order = np.lexsort((pos_of, indices))
p_sorted, c_sorted = pos_of[order], indices[order]
p_start = np.searchsorted(c_sorted, np.arange(C + 1))
t_order = np.argsort(tok, kind="stable")
t_start = np.searchsorted(tok[t_order], np.arange(V + 1))


def posting(f):
    """Positions where feature f is on."""
    if f < C:
        return p_sorted[p_start[f]:p_start[f + 1]]
    if f < 2 * C:
        p = p_sorted[p_start[f - C]:p_start[f - C + 1]] + 1
        return p[(p < N) & (pos_in_row[np.minimum(p, N - 1)] != 0)]
    v = f - 2 * C
    return t_order[t_start[v]:t_start[v + 1]]


n_feat_tr = np.concatenate([np.diff(p_start).astype(np.float64), np.zeros(C), np.bincount(tok[:half], minlength=V).astype(np.float64)])
for f in range(C):  # piece counts restricted to the training half
    pp = posting(f)
    n_feat_tr[f] = np.count_nonzero(pp < half)
    n_feat_tr[C + f] = np.count_nonzero(pp[pos_in_row[pp] != S - 1] < half)


def h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def mi(n11, n1, nf, n):
    """Mutual information (bits per position) between y (n1 on) and feature (nf on), n11 joint."""
    p = np.array([[n - n1 - nf + n11, nf - n11], [n1 - n11, n11]], dtype=np.float64) / n
    py, pf = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log2(p / (py * pf)), 0.0)
    return t.sum()


def patterns(feats, y_pos, lo, hi):
    """Aggregated (pattern -> [n_total, n_on]) over positions [lo, hi) for the feature list."""
    code = np.zeros(hi - lo, dtype=np.int64)
    for j, f in enumerate(feats):
        pp = posting(f)
        pp = pp[(pp >= lo) & (pp < hi)]
        code[pp - lo] |= 1 << j
    yy = y_pos[(y_pos >= lo) & (y_pos < hi)]
    yv = np.zeros(hi - lo, dtype=np.int64)
    yv[yy - lo] = 1
    keys, inv = np.unique(code, return_inverse=True)
    n = np.bincount(inv, minlength=len(keys)).astype(np.float64)
    k = np.bincount(inv, weights=yv, minlength=len(keys))
    X = ((keys[:, None] >> np.arange(len(feats))[None, :]) & 1).astype(np.float64)
    return X, n, k


def fit(X, n, k, ridge=1e-2, iters=50):
    """Weighted logistic by Newton: returns weights incl. intercept (last)."""
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    w = np.zeros(Xb.shape[1])
    w[-1] = math.log((k.sum() + 0.5) / (n.sum() - k.sum() + 0.5))
    R_ = ridge * np.eye(Xb.shape[1])
    R_[-1, -1] = 0
    for _ in range(iters):
        z = Xb @ w
        p = 1 / (1 + np.exp(-z))
        g = Xb.T @ (n * p - k) + R_ @ w
        H = (Xb * (n * p * (1 - p))[:, None]).T @ Xb + R_ + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def nll_bits(X, n, k, w):
    z = np.hstack([X, np.ones((X.shape[0], 1))]) @ w
    lp1 = -np.logaddexp(0, -z) / math.log(2)
    lp0 = -np.logaddexp(0, z) / math.log(2)
    return float(-(k * lp1 + (n - k) * lp0).sum())


records = []
total = {"marginal_test": 0.0, "rule_test": 0.0, "rule_model": 0.0, "marginal_model": 0.0}
n_on_tr = np.array([np.count_nonzero(posting(c) < half) for c in range(C)])
for c in range(C):
    yp = posting(c)
    ntr = int(np.count_nonzero(yp < half))
    nte = len(yp) - ntr
    p_marg = (ntr + 0.5) / (Ntr + 1)
    marg_te = -(nte * math.log2(p_marg) + (Nte - nte) * math.log2(1 - p_marg))
    total["marginal_test"] += marg_te
    total["marginal_model"] += 0.5 * LOG2N
    if ntr < 20 or ntr > Ntr - 20:
        total["rule_test"] += marg_te
        total["rule_model"] += 0.5 * LOG2N
        continue
    # candidates: co-occurrence with the training positives
    ytr = yp[yp < half]
    s_c = site_of_piece[c]
    rows_ = [indices[indptr[q]:indptr[q + 1]] for q in ytr.tolist()]
    now = np.concatenate([r[r < offs[s_c]] for r in rows_]) if rows_ else np.zeros(0, np.int64)
    prev = np.concatenate([indices[indptr[q - 1]:indptr[q]] for q in ytr.tolist() if pos_in_row[q] > 0] or [np.zeros(0, np.int64)])
    cand_f = np.concatenate([now, C + prev, 2 * C + tok[ytr]])
    uf, k11 = np.unique(cand_f, return_counts=True)
    keep = k11 >= 2
    uf, k11 = uf[keep], k11[keep]
    scores = np.array([mi(a, ntr, max(n_feat_tr[f], a), Ntr) for f, a in zip(uf, k11)])
    cands = uf[np.argsort(-scores)[:12]].tolist()
    # greedy two-part MDL
    chosen = []
    X, n, k = patterns([], yp, 0, half)
    w = fit(X, n, k)
    best = nll_bits(X, n, k, w) + 0.5 * LOG2N
    while cands:
        trial = []
        for f in cands:
            X, n, k = patterns(chosen + [f], yp, 0, half)
            w_ = fit(X, n, k)
            mdl = nll_bits(X, n, k, w_) + (len(chosen) + 1) * math.log2(NF) + (len(chosen) + 2) * 0.5 * LOG2N
            trial.append((mdl, f, w_))
        mdl, f, w_ = min(trial, key=lambda x: x[0])
        if mdl >= best:
            break
        best, w = mdl, w_
        chosen.append(f)
        cands.remove(f)
    X, n, k = patterns(chosen, yp, half, N)
    rule_te = nll_bits(X, n, k, w)
    model = len(chosen) * math.log2(NF) + (len(chosen) + 1) * 0.5 * LOG2N
    total["rule_test"] += rule_te
    total["rule_model"] += model
    kind = lambda f: "now" if f < C else ("prev" if f < 2 * C else "token")
    records.append({"piece": c, "site": str(Z["site_names"][s_c]), "train_on": ntr, "test_on": nte,
                    "features": [[kind(f), int(f % C if f < 2 * C else f - 2 * C)] for f in chosen],
                    "weights": [float(x) for x in w], "test_bits_rule": rule_te, "test_bits_marginal": marg_te,
                    "model_bits": model})
    if len(records) % 500 == 0:
        print(f"[{time.time() - t0:6.0f}s] {len(records)} pieces fitted (piece {c}/{C})", flush=True)

ks = np.array([len(r["features"]) for r in records])
gain = np.array([1 - r["test_bits_rule"] / max(r["test_bits_marginal"], 1e-9) for r in records])
res = {"description": "E5: per-piece sparse-logistic gating rules over upstream pieces (same position), all pieces "
                      "at the previous position, and the token id; two-part MDL selection; fit rows 0-63, coded "
                      "rows 64-127 of the 128 dumped val rows (offset 1024). Bits are for sending every piece's "
                      "on/off at every coded position (the active sets), model bits charged once.",
       "pieces_total": C, "pieces_with_rules_fitted": len(records), "coded_tokens": Nte,
       "bits_per_token": {
           "vpd_independent_code": None,
           "marginal": (total["marginal_test"] + total["marginal_model"]) / Nte,
           "marginal_data_only": total["marginal_test"] / Nte,
           "rules": (total["rule_test"] + total["rule_model"]) / Nte,
           "rules_data_only": total["rule_test"] / Nte,
           "rules_model_bits_total": total["rule_model"]},
       "fan_in_histogram": {int(a): int(b) for a, b in zip(*np.unique(ks, return_counts=True))},
       "pieces_with_short_rule": {
           "k<=3 and >=50% fewer test bits than marginal": int(((ks <= 3) & (gain >= 0.5)).sum()),
           "k<=3 and >=90% fewer": int(((ks <= 3) & (gain >= 0.9)).sum()),
           "any k and >=50% fewer": int((gain >= 0.5).sum())},
       "fan_in_mean_over_ruled_pieces": float(ks[ks > 0].mean()) if (ks > 0).any() else 0.0,
       "feature_kind_counts": {k: int(sum(f[0] == k for r in records for f in r["features"])) for k in ("now", "prev", "token")},
       "records": records}
e1 = F / "e123_vpd4l.json"
if e1.exists():
    res["bits_per_token"]["vpd_independent_code"] = json.load(open(e1))["vpd"]["independent"]["data_bits_per_token"]
json.dump(res, open(F / "e5_gating_rules.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "records"}, indent=1))
