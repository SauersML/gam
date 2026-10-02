"""E1 context-conditional coding, E2 always-on core, E3 expensive tokens (#2951), from masks_vpd4l.npz.
Streams positions in chunks over the sparse (CSR) masks with integer counters; peak memory < 4 GiB.

E1. Bits/token to send each position's active set, coded on the second half of the rows with models fitted
    on the first half, each model's own code length charged:
      independent   VPD's per-token code: per site omega(k+1) + log2 C(C_s, k) (no learning)
      marginal      one Bernoulli rate per piece (KT), counts sent as Elias-delta(n_c + 1)
      previous      rate per piece given whether the same piece was on at the previous position
      token         rate per (current token id, piece), backed off to the marginal with weight beta
      token_and_previous  per (token, piece) given the previous state, backed off to 'previous'
    beta from {1/4, 1, 4, 16}, chosen on the coded half and charged 2 bits. Reported: data bits/token,
    model bits, model bits amortised over the coded tokens and over 1e6 tokens.
E2. Core = pieces on at >= 90% of positions; bits for the conditional set and the core's exceptions; KL with
    only the core on (from the dump). VPD's tiny core is an artifact of its loss (every active piece is
    charged on every token); lead with bits/token.
E3. Pieces/token, bits/token and KL/token by token type.
Writes e123_vpd4l.json (rewritten after each part).
"""

import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from tokenizers import Tokenizer

F = Path.home() / "mpd-data/frontier"
Z = np.load(F / "masks_vpd4l.npz", mmap_mode="r")
ids = np.asarray(Z["ids"])
R, S = ids.shape
tok_all = ids.reshape(-1).astype(np.int64)
pos_all = np.tile(np.arange(S), R)
V = int(tok_all.max()) + 1
LOG2 = math.log(2)
CH = 2048  # positions per chunk


def omega_len(m):
    bits, k = 1, int(m)
    while k > 1:
        w = k.bit_length()
        bits += w
        k = w - 1
    return bits


def omega_table(n):
    out = np.ones(n + 2, dtype=np.int64)
    for m in range(2, n + 2):
        bits, k = 1, m
        while k > 1:
            w = k.bit_length()
            bits += w
            k = w - 1
        out[m] = bits
    return out  # out[m] = omega(m)


def delta_len_arr(n):
    n = np.asarray(n, dtype=np.int64)
    L = np.floor(np.log2(np.maximum(n, 1))).astype(np.int64)
    return L + 2 * np.floor(np.log2(L + 1)).astype(np.int64) + 1


lb = lambda n, k: (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)) / LOG2
h = lambda p: -np.log2(np.clip(p, 1e-300, 1.0))


def merge_counts(parts):
    """Merge a list of (unique keys, counts) into one sorted table."""
    if not parts:
        return np.zeros(0, np.int64), np.zeros(0, np.float64)
    k = np.concatenate([p[0] for p in parts])
    c = np.concatenate([p[1] for p in parts])
    u, inv = np.unique(k, return_inverse=True)
    return u, np.bincount(inv, weights=c).astype(np.float64)


def lookup(keys, table_k, table_v):
    j = np.minimum(np.searchsorted(table_k, keys), max(len(table_k) - 1, 0))
    hit = (table_k[j] == keys) if len(table_k) else np.zeros(len(keys), bool)
    return np.where(hit, table_v[j] if len(table_k) else 0.0, 0.0)


def analyse(kind, rows, token_models=True):
    N, half = rows * S, (rows // 2) * S
    indptr = np.asarray(Z[kind + "_indptr"][:N + 1]).astype(np.int64)
    IDX = Z[kind + "_indices"]
    offs = np.asarray(Z[kind + "_offsets"]).astype(np.int64)
    C, nsite = int(offs[-1]), len(offs) - 1
    csz = np.diff(offs)
    OM = omega_table(int(csz.max()) + 2)
    lens = np.diff(indptr)
    CH = int(max(64, 1.5e6 / max(lens[:N].mean(), 1)))  # ~1.5M active entries per chunk
    Ntr, Nte = half, N - half

    def chunk(a, b):
        """Entries of positions [a, b): (position, piece, site)."""
        idx = np.asarray(IDX[indptr[a]:indptr[b]]).astype(np.int64)
        pos = np.repeat(np.arange(a, b), lens[a:b])
        return pos, idx, np.searchsorted(offs, idx, side="right") - 1

    def prev_on(pos, idx):
        """Was piece idx on at pos-1 (same row)? via the CSR of pos-1."""
        out = np.zeros(len(idx), bool)
        ok = pos_all[pos] > 0
        if not ok.any():
            return out
        p = pos[ok]
        # sorted keys (position * C + piece) of the block of positions p.min()-1 .. p.max()-1
        block = np.asarray(IDX[indptr[p.min() - 1]:indptr[p.max()]]).astype(np.int64)
        bpos = np.repeat(np.arange(p.min() - 1, p.max()), lens[p.min() - 1:p.max()])
        bkey = bpos * C + block
        if len(bkey) == 0:
            return out
        qkey = (p - 1) * C + idx[ok]
        j = np.minimum(np.searchsorted(bkey, qkey), len(bkey) - 1)
        out[ok] = bkey[j] == qkey
        return out

    out = {"pieces_total": C, "mean_active_per_token": float(lens[:N].mean())}
    # ---------------- pass 1: training counts
    n_c = np.zeros(C)
    a_c = np.zeros(C)
    b_c = np.zeros(C)
    n_c_np = np.zeros(C)
    tok_parts, k11_parts, kb1_parts = [], [], []
    for a in range(0, half, CH):
        b = min(a + CH, half)
        pos, idx, _ = chunk(a, b)
        n_c += np.bincount(idx, minlength=C)
        po = prev_on(pos, idx)
        a_c += np.bincount(idx[po], minlength=C)
        hp = pos_all[pos] > 0
        n_c_np += np.bincount(idx[hp], minlength=C)
        nl = pos_all[pos] != S - 1
        b_c += np.bincount(idx[nl & (pos + 1 < half)], minlength=C)
        keys = tok_all[pos] * C + idx
        if token_models:  # (token, piece) tables; off for the 4.5k-pieces/token basis (memory)
            tok_parts.append(np.unique(keys, return_counts=True))
            k11_parts.append(np.unique(keys[po], return_counts=True))
            sel = nl & (pos + 1 < half)
            kb1_parts.append(np.unique(tok_all[pos[sel] + 1] * C + idx[sel], return_counts=True))
    uk, ucnt = merge_counts(tok_parts)
    k11 = merge_counts(k11_parts)
    kb1 = merge_counts(kb1_parts)
    del tok_parts, k11_parts, kb1_parts
    N_v = np.bincount(tok_all[:half], minlength=V).astype(np.float64)
    Np = Ntr - Ntr // S
    p_c = (n_c + 0.5) / (Ntr + 1)
    S0 = h(1 - p_c).sum()
    model_marg = float(delta_len_arr(n_c + 1).sum())
    model_prev = model_marg + float(delta_len_arr(a_c + 1).sum() + delta_len_arr(b_c + 1).sum())
    seen_v = np.nonzero(N_v)[0]
    om_nv = sum(omega_len(int(N_v[v]) + 1) for v in seen_v)
    vk = uk // C
    nz_per_v = np.bincount(vk, minlength=V)
    nzv = nz_per_v[nz_per_v > 0]
    model_tok = model_marg + om_nv + float(sum(omega_len(int(k) + 1) for k in nzv) + lb(C, nzv).sum()) + float(delta_len_arr(ucnt).sum())
    allk = np.union1d(np.union1d(uk, k11[0]), kb1[0])
    N1, A11, B1 = lookup(allk, uk, ucnt), lookup(allk, *k11), lookup(allk, *kb1)
    kv, kc = allk // C, allk % C
    nzb = np.bincount(kv, minlength=V)
    nzb = nzb[nzb > 0]
    model_both = model_prev + om_nv + float(sum(omega_len(int(k) + 1) for k in nzb) + lb(C, nzb).sum()) + float(
        (delta_len_arr(N1 + 1) + delta_len_arr(A11 + 1) + delta_len_arr(B1 + 1)).sum())

    # ---------------- pass 2: coding the test half
    betas = (0.25, 1.0, 4.0, 16.0)
    cost = {k: np.zeros(Nte) for k in ("independent", "marginal")}
    for name in ("token", "previous", "token_and_previous"):
        for be in betas:
            cost[(name, be)] = np.zeros(Nte)
    q = {be: ((a_c + be * p_c) / (b_c + be), (n_c_np - a_c + be * p_c) / (Np - b_c + be)) for be in betas}  # (q1, q0)
    te_tok = tok_all[half:N]
    te_first = pos_all[half:N] == 0
    Nv_te = N_v[te_tok]
    # base sums (inactive terms over all pieces) depend on the token only through N_v
    for be in betas:
        base_t = {nv: h(1 - be * p_c / (nv + be)).sum() for nv in np.unique(Nv_te)}
        cv = np.bincount(vk, weights=h(1 - (ucnt + be * p_c[uk % C]) / (N_v[vk] + be)) - h(1 - be * p_c[uk % C] / (N_v[vk] + be)), minlength=V)
        cost[("token", be)] += np.array([base_t[nv] for nv in Nv_te]) + cv[te_tok]
        q1, q0 = q[be]
        cost[("previous", be)] += np.where(te_first, S0, h(1 - q0).sum())
        base_b = {nv: h(1 - be * q0 / (nv + be)).sum() for nv in np.unique(Nv_te)}
        p0 = (N1 - A11 + be * q0[kc]) / (N_v[kv] - B1 + be)
        pb0 = be * q0[kc] / (N_v[kv] + be)
        cvb = np.bincount(kv, weights=h(1 - p0) - h(1 - pb0), minlength=V)
        cost[("token_and_previous", be)] += np.where(te_first, S0, np.array([base_b[nv] for nv in Nv_te]) + cvb[te_tok])
    corr = h(p_c) - h(1 - p_c)
    cost["marginal"] += S0
    k_site_all = np.zeros((N, nsite), dtype=np.int64)
    for a in range(0, N, CH):
        b = min(a + CH, N)
        pos, idx, site = chunk(a, b)
        np.add.at(k_site_all, (pos, site), 1)
        if b <= half:
            continue
        m = pos >= half
        pos, idx = pos[m], idx[m]
        t = pos - half
        cost["marginal"] += np.bincount(t, weights=corr[idx], minlength=Nte)
        po = prev_on(pos, idx)
        first = pos_all[pos] == 0
        # previous-on pieces condition the NEXT position: switch its inactive term
        nl = (pos_all[pos] != S - 1) & (pos + 1 < N)
        tn = pos[nl] + 1 - half
        cn = idx[nl]
        kp = tok_all[pos] * C + idx
        ptok, nvp = lookup(kp, uk, ucnt), N_v[tok_all[pos]]
        cp = (lookup(kp, allk, N1), lookup(kp, allk, A11), lookup(kp, allk, B1))
        kn = tok_all[tn + half] * C + cn
        cnn = (lookup(kn, allk, N1), lookup(kn, allk, A11), lookup(kn, allk, B1))
        nvn = N_v[tok_all[tn + half]]
        for be in betas:
            q1, q0 = q[be]
            p_t = (ptok + be * p_c[idx]) / (nvp + be)
            cost[("token", be)] += np.bincount(t, weights=h(p_t) - h(1 - p_t), minlength=Nte)
            cost[("previous", be)] += np.bincount(tn, weights=h(1 - q1[cn]) - h(1 - q0[cn]), minlength=Nte)
            pp = np.where(first, p_c[idx], np.where(po, q1[idx], q0[idx]))
            cost[("previous", be)] += np.bincount(t, weights=h(pp) - h(1 - pp), minlength=Nte)

            def p_both(cnt, nv, cc_, on_):
                n1, a11, b1 = cnt
                return np.where(on_, (a11 + be * q1[cc_]) / (b1 + be), (n1 - a11 + be * q0[cc_]) / (nv - b1 + be))

            cost[("token_and_previous", be)] += np.bincount(
                tn, weights=h(1 - p_both(cnn, nvn, cn, True)) - h(1 - p_both(cnn, nvn, cn, False)), minlength=Nte)
            pb = np.where(first, p_c[idx], p_both(cp, nvp, idx, po))
            cost[("token_and_previous", be)] += np.bincount(t, weights=h(pb) - h(1 - pb), minlength=Nte)
    indep = (OM[np.minimum(k_site_all + 1, len(OM) - 1)] + lb(csz[None, :], k_site_all)).sum(1)
    cost["independent"] = indep[half:]
    models = {"independent": 0.0, "marginal": model_marg, "previous": model_prev}
    if token_models:
        models.update({"token": model_tok, "token_and_previous": model_both})
    else:
        out["token_models"] = "not computed for this basis: (token, piece) tables do not fit the 4 GiB budget"
    for name in models:
        if name in ("independent", "marginal"):
            c, be = cost[name], None
        else:
            be = min(betas, key=lambda x: cost[(name, x)].mean())
            c = cost[(name, be)]
        mb = models[name] + (2 if be is not None else 0)
        out[name] = {"data_bits_per_token": float(c.mean()), "model_bits": float(mb), "beta": be,
                     "model_bits_per_coded_token": mb / Nte, "total_bits_per_token": float(c.mean()) + mb / Nte,
                     "total_bits_per_token_at_1e6_tokens": float(c.mean()) + mb / 1e6}

    # ---------------- E2 core
    core = np.asarray(Z["core_" + kind]).astype(np.int64)
    is_core = np.zeros(C, bool)
    is_core[core] = True
    core_site = np.bincount(np.searchsorted(offs, core, side="right") - 1, minlength=nsite)
    kc_site = np.zeros((N, nsite), dtype=np.int64)
    for a in range(0, N, CH):
        pos, idx, site = chunk(a, min(a + CH, N))
        m = is_core[idx]
        np.add.at(kc_site, (pos[m], site[m]), 1)
    cond = k_site_all - kc_site
    miss = core_site[None, :] - kc_site
    bits_cond = (OM[np.minimum(cond + 1, len(OM) - 1)] + lb((csz - core_site)[None, :], cond)).sum(1)
    bits_exc = (OM[np.minimum(miss + 1, len(OM) - 1)] + lb(core_site[None, :], miss)).sum(1)
    kl = np.asarray(Z["kl_" + kind]).reshape(-1)[:N]
    klc = np.asarray(Z["kl_" + kind + "_core"]).reshape(-1)[:N]
    out["core"] = {"threshold": 0.9, "core_pieces": int(len(core)), "core_bits_once": float(lb(C, len(core))),
                   "core_on_per_token": float(kc_site.sum(1).mean()), "conditional_on_per_token": float(cond.sum(1).mean()),
                   "bits_conditional_per_token": float(bits_cond.mean()), "bits_core_exceptions_per_token": float(bits_exc.mean()),
                   "bits_independent_per_token": float(indep.mean()), "kl_full": float(np.nanmean(kl)), "kl_core_only": float(np.nanmean(klc)),
                   "core_per_site": {str(n): int(c) for n, c in zip(Z["site_names"], core_site)}}
    return out, lens[:N], indep, kl


tk = Tokenizer.from_file(str(Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"))
STOP = set("""a about above after again against all am an and any are as at be because been before being below between
both but by can did do does doing down during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or
other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom why will with
you your yours yourself yourselves would could also may might must shall""".split())


def category(t, pos):
    if pos == 0:
        return "first position"
    s = tk.id_to_token(int(t)) or ""
    lead = s.startswith("Ġ")
    w = s.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
    st = w.strip()
    if not st:
        return "whitespace/newline"
    if re.fullmatch(r"[\d.,]+", st) and any(ch.isdigit() for ch in st):
        return "number"
    if all(not ch.isalnum() for ch in st):
        return "punctuation"
    if st.isalpha():
        if not lead:
            return "word continuation"
        return "function word" if st.lower() in STOP else "content word"
    return "other"


res = {"rows": int(R), "offset": 1024, "seq": int(S), "fit_rows": "first half of the rows used", "coded_rows": "second half",
       "description": "E1 context-conditional coding (bits/token by context model), E2 always-on core, E3 by token type; "
                      "vpd = VPD rounded masks (gate > 0), all 128 rows and the 32 frontier rows; wsvd = per-matrix "
                      "Fisher-metric SVD basis at the threshold matching VPD's KL (mean L0 4451), 32 frontier rows",
       "framing": "Compare codes by bits/token at each code's own rates (E1), not by L0. VPD's tiny always-on core "
                  "(E2) is an artifact of its training loss, which charges every active piece on every token, so "
                  "it pushes always-needed structure into conditionally-active pieces; it is not evidence that the "
                  "computation has no fixed part."}
cats = np.array([category(t, p) for t, p in zip(tok_all, pos_all)])
res["E3"] = {}
for kind, rows, key in (("vpd", R, "vpd"), ("vpd", 32, "vpd_32rows"), ("wsvd", 32, "wsvd")):
    out, lens, indep, kl = analyse(kind, rows, token_models=(kind == "vpd"))
    out["rows"] = rows
    res[key] = out
    if key != "vpd_32rows":
        for c in np.unique(cats[:len(lens)]):
            m = cats[:len(lens)] == c
            res["E3"].setdefault(c, {})[key + "_tokens"] = int(m.sum())
            res["E3"][c][key] = {"pieces_per_token": float(lens[m].mean()), "bits_per_token_independent": float(indep[m].mean()),
                                 "kl_per_token": float(np.nanmean(kl[m]))}
    json.dump(res, open(F / "e123_vpd4l.json", "w"), indent=1)
    print(key, json.dumps({k: v for k, v in out.items() if k != "core"}), flush=True)
    print(key, "core", json.dumps({k: v for k, v in out["core"].items() if k != "core_per_site"}), flush=True)
for c, d in sorted(res["E3"].items(), key=lambda kv: -kv[1].get("vpd_tokens", 0)):
    w = d.get("wsvd", {})
    print(f"{c:20s} n {d.get('vpd_tokens', 0):6d}  vpd L0 {d['vpd']['pieces_per_token']:6.1f} bits {d['vpd']['bits_per_token_independent']:7.1f} "
          f"KL {d['vpd']['kl_per_token']:.3f}  wsvd L0 {w.get('pieces_per_token', float('nan')):7.1f} KL {w.get('kl_per_token', float('nan')):.3f}")
