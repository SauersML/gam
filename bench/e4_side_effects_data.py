"""E4 side effects (#2951): what the VPD paper's one-subcomponent emoticon edit and its LoRA baselines break,
measured on every token of 4096 held-out Pile validation rows plus synthetic copying probes.

The edits are rebuilt exactly as frontier's e4_replicate.py makes them (~/mpd-data/frontier, E4_CI_MIN=0.5):
harvest-style windows around firings of h.2.mlp.down_proj:2359 with CI > 0.5, eval = first 50 after
random.seed(42) shuffling, train = the next 282; VPD sets U_c -> -alpha u_o/|u_o| (bf16); LoRA is rank 1 on
the same matrix, AdamW 1e-3, 300 steps, loss = CE('o' at fire positions) + lambda * KL over the rest.

Stages (each resumable; outputs in ~/mpd-data/frontier/e4_side/):
  lora       retrain the 8 LoRA edits (282 and 10 training windows x lambda in {0.1, 1, 10, 100}); save A, B.
  eval       build every edited weight: the 7-strength VPD sweep, the 8 LoRAs, and for each LoRA a VPD edit whose
             strength is SOLVED (bisection on alpha) to give exactly that LoRA's edit success on the 50 eval
             windows. Then, on rows 48638-52733 of the val shard (row group 1, never scanned, so disjoint from
             every training/eval window) plus 128 rows of repeated random tokens, store per token and per model:
             KL(edited || original), change in loss on the true next token, the edited top-1 token and its
             probability, and log-probabilities of the contrast token (is/are, was/were, has/have, a/an, ...).
  summarize  mine the capability battery from the token text, break damage down along unchosen axes, cluster-
             bootstrap by document, pick worst non-emoticon contexts; writes e4_side_effects.json for
             bench/figures_2951/e4_side_effects_fig.py.

usage: mem-lease 2 venv/python e4_side_effects_data.py lora [NAMES...] | eval;   MPD_MEM_GIB=4 ... summarize
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
VD = HOME / "mpd-data/vpd"
FR = HOME / "mpd-data/frontier"
DEV = int(__import__("os").environ.get("E4_DEV_ROWS", "0"))  # > 0: a quick run on that many rows with the LoRAs trained so far
OUTD = FR / "e4_side" / ("dev" if DEV else "")
OUTD.mkdir(parents=True, exist_ok=True)
SITE, COMP, O_TOK, SIDE, CI_MIN = "h.2.mlp.down_proj", 2359, 80, 20, 0.5
REPL = json.load(open(FR / "e4_replication_ci0.5.json"))
LAMS = (0.1, 1.0, 10.0, 100.0)
LORAS = [f"lora{n}_lam{lam:g}" for n in (282, 10) for lam in LAMS]
ALPHAS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
HELD0, N_HELD, SYN_D, SYN_PER = 48638, 4096, (8, 32, 128, 250), 32
CONTRAST = {"Ġis": "Ġare", "Ġare": "Ġis", "Ġwas": "Ġwere", "Ġwere": "Ġwas", "Ġhas": "Ġhave", "Ġhave": "Ġhas",
            "Ġdoes": "Ġdo", "Ġdo": "Ġdoes", "Ġa": "Ġan", "Ġan": "Ġa", "Ġthis": "Ġthese", "Ġthese": "Ġthis"}
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(VD / "t-9d2b8f02/tokenizer.json"))


# ---------------------------------------------------------------- model side (lora, eval)
def load():
    import torch
    sys.path.insert(0, str(VD))
    from vpd_model import VPD_PTH, load_target
    target = load_target("mps")
    raw = torch.load(str(VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    key = "_components." + SITE.replace(".", "-")
    U = raw[key + ".U"].float()[COMP].to("mps")
    V = raw[key + ".V"].float()[:, COMP].to("mps")
    del raw
    return target, U, V


def harvest():
    """The harvest-style windows of e4_replicate.py, verbatim: (eval, train) lists of (tokens, fire offsets)."""
    import torch
    fires = {}
    for scan in ("scan_0_4096.pt", "scan_edits_4096_28672.pt"):
        s = torch.load(VD / scan)
        for r, p, ci in s["fires"][(SITE, COMP)].tolist():
            if ci / 1e6 <= CI_MIN:
                continue
            fires.setdefault(r, set()).add(p)
    rz = np.load(FR / "e4_rows.npz")
    row_ids = {int(r): torch.tensor(t, dtype=torch.long) for r, t in zip(rz["rows"], rz["tokens"])}
    examples = []
    for r in sorted(fires):
        for p in sorted(fires[r]):
            a, b = max(0, p - SIDE), min(512, p + SIDE + 1)
            toks = row_ids[r][a:b]
            f = [q - a for q in sorted(fires[r]) if a <= q < b]
            if any(q + 1 < len(toks) for q in f):
                examples.append((toks, f))
    random.seed(42)
    random.shuffle(examples)
    return examples[:50], examples[50:50 + 947]


def train_lora(target, pool, lam, steps=300, batch=256, micro=8):
    """e4_replicate.py's train_lora (the paper's LoRATrainer), verbatim."""
    import torch
    import torch.nn.functional as F
    site = target.site(SITE)
    W0 = site.W.clone()
    torch.manual_seed(0)
    A = (torch.randn(1, W0.shape[1]) * 0.01).to("mps").requires_grad_(True)
    B = torch.zeros(W0.shape[0], 1, device="mps", requires_grad=True)
    opt = torch.optim.AdamW([A, B], lr=1e-3)
    seqs = []
    for toks, f in pool:
        t = toks.clone()
        fp = [q for q in f if q + 1 < len(t)]
        for q in fp:
            t[q + 1] = O_TOK
        seqs.append((t, fp))
    L = max(len(t) for t, _ in seqs)
    T = torch.zeros(len(seqs), L, dtype=torch.long)
    fire = torch.zeros(len(seqs), L, dtype=torch.bool)
    pad = torch.zeros(len(seqs), L, dtype=torch.bool)
    for i, (t, fp) in enumerate(seqs):
        T[i, :len(t)] = t
        fire[i, fp] = True
        pad[i, :len(t)] = True
    n = len(seqs)
    for step in range(steps):
        idx = torch.randint(n, (min(batch, n),))
        n_fire = int(fire[idx].sum())
        n_kl = int((pad[idx] & ~fire[idx]).sum())
        opt.zero_grad()
        for j in range(0, len(idx), micro):
            mi = idx[j:j + micro]
            tb, fb, pb = T[mi].to("mps"), fire[mi].to("mps"), pad[mi].to("mps")
            with torch.no_grad():
                site.W = W0
                lb = F.log_softmax(target(tb), -1)
            site.W = W0 + B @ A
            lg = target(tb)
            fi = fb.nonzero()
            ce = F.cross_entropy(lg[fi[:, 0], fi[:, 1]], tb[fi[:, 0], fi[:, 1] + 1], reduction="sum") / max(n_fire, 1)
            le = F.log_softmax(lg, -1)
            kl = (le.exp() * (le - lb))[pb & ~fb].sum() / max(n_kl, 1)
            (ce + lam * kl).backward()
            del lg, le, lb
        opt.step()
    site.W = W0
    return A.detach().cpu(), B.detach().cpu()


def p_fire_fn(target, ev):
    """Mean P('o') at the eval fire positions under weights W (the protocol's p_fire); windows in batches of 10."""
    import torch
    L = max(len(t) for t, _ in ev)
    T = torch.zeros(len(ev), L, dtype=torch.long)
    for i, (t, _) in enumerate(ev):
        T[i, :len(t)] = t  # right padding; attention is causal, so real positions never see it
    pos = [(i, q) for i, (t, f) in enumerate(ev) for q in f if q + 1 < len(t)]
    site = target.site(SITE)
    W0 = site.W.clone()

    @torch.no_grad()
    def f(W):
        site.W = W
        ps = []
        for b in range(0, len(ev), 10):
            ii = [i - b for i, q in pos if b <= i < b + 10]
            qq = [q for i, q in pos if b <= i < b + 10]
            ps.append(torch.softmax(target(T[b:b + 10].to("mps"))[ii, qq], -1)[:, O_TOK].cpu())
        site.W = W0
        return float(torch.cat(ps).mean())
    return f


def lora_deltas():
    """Every LoRA trained so far: lora_deltas.pt plus the per-name files of parallel runs."""
    import torch
    have = {}
    for f in sorted((FR / "e4_side").glob("lora_deltas*.pt")):
        have.update(torch.load(f))
    return have


def stage_lora(only=None):
    """Train the LoRAs not yet saved (or just the named ones, each into its own file, so runs can go in parallel)."""
    import torch
    target, _, _ = load()
    ev, train = harvest()
    assert len(train) == REPL["n_train"] == 282, len(train)
    pf = p_fire_fn(target, ev)
    W0 = target.site(SITE).W.clone()
    for n, pool in ((10, train[:10]), (282, train)):
        for lam in LAMS:
            name = f"lora{n}_lam{lam:g}"
            if name in lora_deltas() or (only and name not in only):
                continue
            A, B = train_lora(target, pool, lam)
            torch.save({name: {"A": A, "B": B}}, FR / f"e4_side/lora_deltas_{name}.pt")
            ref = REPL["lora" if n == 282 else "lora_low"][str(lam)]["p_fire"]
            log(f"{name}: p_fire {pf(W0 + (B @ A).to('mps')):.4f} (e4_replicate {ref:.4f})")


# ---------------------------------------------------------------- eval
def heldout_tokens():
    """[N_HELD + synthetic, 512] int32 rows and a synthetic-copy-distance vector (0 = natural row)."""
    path = FR / "e4_side/rows.npz"
    if path.exists():
        z = np.load(path)
        return z["tokens"], z["syn_d"]
    import pyarrow.parquet as pq
    sys.path.insert(0, str(VD))
    from vpd_model import VAL_PARQUET
    col = pq.ParquetFile(VAL_PARQUET).read_row_group(1, columns=["input_ids"]).column("input_ids")
    sl = col.slice(0, N_HELD).combine_chunks()
    nat = sl.flatten().to_numpy().reshape(N_HELD, -1)[:, :512].astype(np.int32)
    del col, sl
    rng = np.random.default_rng(0)
    syn, syn_d = [], []
    for d in SYN_D:
        for _ in range(SYN_PER):
            filler = rng.integers(1000, 30000, 512 - 2 * d)
            s = rng.integers(1000, 30000, d)
            syn.append(np.concatenate([filler, s, s]).astype(np.int32))
            syn_d.append(d)
    tokens = np.concatenate([nat, np.stack(syn)])
    syn_d = np.concatenate([np.zeros(N_HELD, np.int32), np.array(syn_d, np.int32)])
    np.savez(path, tokens=tokens, syn_d=syn_d)
    return tokens, syn_d


def solve_alpha(pf, W0, U, V, u_o, p_target, lo=0.5, hi=12.0):
    """The VPD strength whose edit success equals p_target (p_fire rises with alpha; bisection to 1e-4)."""
    import torch
    W = lambda a: W0 + torch.outer((-a * u_o).to(torch.bfloat16).float() - U, V)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if pf(W(mid)) < p_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break
    return 0.5 * (lo + hi)


def edit_variants():
    """The model and every edit: the VPD strength sweep, each LoRA, and for each LoRA the VPD edit whose strength is
    solved to give exactly that LoRA's edit success. Returns (target, W0, {name: delta W}, {name: meta})."""
    import torch
    target, U, V = load()
    W0 = target.site(SITE).W.clone()
    u_o = target.wte[O_TOK] / target.wte[O_TOK].norm()
    ev, _ = harvest()
    pf = p_fire_fn(target, ev)
    deltas = lora_deltas()
    assert DEV or set(deltas) == set(LORAS), sorted(deltas)
    vpd_dw = lambda a: torch.outer((-a * u_o).to(torch.bfloat16).float() - U, V)
    models, meta = {}, {}
    for a in ALPHAS:
        models[f"vpd_a{a:g}"] = vpd_dw(a)
        meta[f"vpd_a{a:g}"] = {"method": "vpd", "alpha": a, "p_fire": pf(W0 + vpd_dw(a))}
    for name in [n for n in LORAS if n in deltas]:
        dw = (deltas[name]["B"] @ deltas[name]["A"]).to("mps")
        p = pf(W0 + dw)
        models[name] = dw
        n, lam = name[4:].split("_lam")
        meta[name] = {"method": "lora", "n_train": int(n), "lambda": float(lam), "p_fire": p}
        a = solve_alpha(pf, W0, U, V, u_o, p)
        models["vpd_match_" + name] = vpd_dw(a)
        meta["vpd_match_" + name] = {"method": "vpd", "alpha": a, "p_fire": pf(W0 + vpd_dw(a)), "matches": name}
        log(f"{name}: p_fire {p:.4f}; VPD alpha {a:.4f} gives {meta['vpd_match_' + name]['p_fire']:.4f}")
    return target, W0, models, meta


def split_forward(target):
    """(resid, final, head): resid(ids) runs the shared part once (everything before h.2.mlp.down_proj) and returns
    the residual entering that MLP output and the MLP hidden activation; final(xmid, g2, W) finishes the network with
    h.2.mlp.down_proj = W and returns the last residual; head(x) gives log-probabilities."""
    import torch.nn.functional as F
    sys.path.insert(0, str(VD))
    from vpd_model import gelu_tanh, rms

    def block(x, i, T):
        s = lambda k: target.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")
        Bn = x.shape[0]
        h = rms(x, target.norms[2 * i], target.eps)
        q = s("q_proj")(h).view(Bn, T, target.n_head, target.hd).transpose(1, 2)
        k = s("k_proj")(h).view(Bn, T, target.n_head, target.hd).transpose(1, 2)
        v = s("v_proj")(h).view(Bn, T, target.n_head, target.hd).transpose(1, 2)
        q, k = target._rope(q, T), target._rope(k, T)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s("o_proj")(y.transpose(1, 2).reshape(Bn, T, -1))
        g = gelu_tanh(s("c_fc")(rms(x, target.norms[2 * i + 1], target.eps)))
        return x, g
    down3 = target.site("h.3.mlp.down_proj")

    def resid(ids):
        x = target.wte[ids]
        for i in range(2):
            x, g = block(x, i, ids.shape[1])
            x = x + target.site(f"h.{i}.mlp.down_proj")(g)
        return block(x, 2, ids.shape[1])

    def final(xmid, g2, W):
        x3, g3 = block(xmid + g2 @ W.T, 3, xmid.shape[1])
        return x3 + down3(g3)

    def head(x):
        return F.log_softmax(rms(x, target.ln_f, target.eps) @ target.wte.T, -1)
    return resid, final, head


def stage_eval():
    import torch
    import torch.nn.functional as F
    target, W0, models, meta = edit_variants()
    names = list(models)
    json.dump({"models": names, "meta": meta}, open(OUTD / "models.json", "w"), indent=1)

    tok = tokenizer()
    tokens, syn_d = heldout_tokens()
    if DEV:
        keep = np.r_[0:DEV, N_HELD:N_HELD + 8]
        tokens, syn_d = tokens[keep], syn_d[keep]
    np.savez(OUTD / "tokens.npz", tokens=tokens, syn_d=syn_d)
    R, P = tokens.shape[0], 511
    cpos = {tok.token_to_id(k): tok.token_to_id(v) for k, v in CONTRAST.items()}
    ctab = torch.full((50277,), -1, dtype=torch.long)
    for k, v in cpos.items():
        ctab[k] = v
    ctab = ctab.to("mps")
    mm = lambda nm, dt, shape: np.lib.format.open_memmap(OUTD / f"{nm}.npy", mode="r+" if (OUTD / f"{nm}.npy").exists() else "w+", dtype=dt, shape=shape)
    base = {k: mm("base_" + k, dt, (R, P)) for k, dt in (("ce", np.float32), ("top", np.uint16), ("ptop", np.float16),
                                                             ("ent", np.float16), ("lpc", np.float32))}
    per = {nm: {k: mm(f"{nm}_{k}", dt, (R, P)) for k, dt in (("kl", np.float32), ("dce", np.float32), ("top", np.uint16),
                                                             ("ptop", np.float16), ("lpc", np.float32))} for nm in names}
    done_path = OUTD / "eval_done.txt"
    done = int(done_path.read_text()) if done_path.exists() else 0
    B, PC = 2, 128  # rows per forward; positions per vocab-sized block (keeps the job inside 2 GiB)
    resid, final, head = split_forward(target)

    with torch.no_grad():
        ids = torch.from_numpy(tokens[:1].astype(np.int64)).to("mps")  # the split forward is the model's forward
        err = (head(final(*resid(ids), W0)) - F.log_softmax(target(ids), -1)).abs().max().item()
        assert err < 1e-3, err
        for r0 in range(done, R, B):
            ids = torch.from_numpy(tokens[r0:r0 + B].astype(np.int64)).to("mps")
            xmid, g2 = resid(ids)
            xb = final(xmid, g2, W0)
            xe = {nm: final(xmid, g2, W0 + models[nm]) for nm in names}
            rs = slice(r0, r0 + ids.shape[0])
            for p0 in range(0, P, PC):
                ps = slice(p0, min(p0 + PC, P))
                nxt = ids[:, ps.start + 1:ps.stop + 1]
                lb = head(xb[:, ps])
                ce_b = -lb.gather(-1, nxt[..., None])[..., 0]
                pt, tp = lb.max(-1)
                ct = ctab[nxt]
                has_c = ct >= 0
                base["ce"][rs, ps] = ce_b.cpu().numpy()
                base["top"][rs, ps] = tp.cpu().numpy().astype(np.uint16)
                base["ptop"][rs, ps] = pt.exp().cpu().numpy().astype(np.float16)
                base["ent"][rs, ps] = (-(lb.exp() * lb).sum(-1)).cpu().numpy().astype(np.float16)
                base["lpc"][rs, ps] = torch.where(has_c, lb.gather(-1, ct.clamp(min=0)[..., None])[..., 0], torch.nan).cpu().numpy()
                for nm in names:
                    le = head(xe[nm][:, ps])
                    o = per[nm]
                    o["kl"][rs, ps] = (le.exp() * (le - lb)).sum(-1).cpu().numpy()
                    o["dce"][rs, ps] = (-le.gather(-1, nxt[..., None])[..., 0] - ce_b).cpu().numpy()
                    pe, te = le.max(-1)
                    o["top"][rs, ps] = te.cpu().numpy().astype(np.uint16)
                    o["ptop"][rs, ps] = pe.exp().cpu().numpy().astype(np.float16)
                    o["lpc"][rs, ps] = torch.where(has_c, le.gather(-1, ct.clamp(min=0)[..., None])[..., 0], torch.nan).cpu().numpy()
                    del le
                del lb
            if (r0 + B) % 64 == 0 or r0 + B >= R:
                for d in [base, *per.values()]:
                    for a in d.values():
                        a.flush()
                done_path.write_text(str(min(r0 + B, R)))
                log(f"rows {min(r0 + B, R)}/{R}")
                torch.mps.empty_cache()


# ---------------------------------------------------------------- summarize
EMO_LEFT = {":", ";", "=", ":-", ";-", "=-", ">:", ">;", ":'"}
MOUTH = set(")(DPpSOo|][/\\3xXd")


def is_mouth(t, eq=False):
    """A token that can complete an emoticon written right after ':' / ';' (':D', ':-)', ';P', ':/'); after '=' only
    the unambiguous mouths D P p ) (."""
    if not t or t[0].isspace():
        return False
    if t[0] == "-" and len(t) > 1:
        t = t[1:]
    return t[0] in ("DPp)(" if eq else MOUTH) and (len(t) == 1 or not t[1].isalpha() or t in ("DD", "PP", "pp"))


def emoticon_positions(tokens, txt, ends_space):
    """[R, 511] positions whose input token is the colon of an emoticon (the edit's intended target): an emoticon
    eye (':', ';', '=', ':-', ...) written after a space or at a line start, followed by a mouth token."""
    left = np.array([t.lstrip() in EMO_LEFT for t in txt])
    eq = np.array([t.lstrip().startswith("=") for t in txt])
    spaced = np.array([t[:1] == " " for t in txt])
    mouth = np.array([is_mouth(t) for t in txt])
    mouth_eq = np.array([is_mouth(t, True) for t in txt])
    three = np.array([t[:1] == "3" for t in txt])
    cur, nxt = tokens[:, :-1], tokens[:, 1:]
    prv = np.concatenate([np.full((len(tokens), 1), len(txt) - 1), tokens[:, :-2]], 1)
    sep = spaced[cur] | ends_space[prv]
    return left[cur] & np.where(eq[cur], mouth_eq[nxt], mouth[nxt]) & sep & ~(three[nxt] & ~spaced[cur])


def boot_means(S, N, wb):
    """Point estimate and 95% interval of sum(S)/sum(N) under document (row) resampling; S [R, K], N [R, K]."""
    est = S.sum(0) / np.maximum(N.sum(0), 1)
    bs = (wb @ S) / np.maximum(wb @ N, 1)
    return est, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0), bs


def stage_summarize():
    import torch
    tok = tokenizer()
    V = tok.get_vocab_size()
    txt = [tok.decode([i]) for i in range(V)] + [""]  # id V = padding
    PAD = V
    z = np.load(OUTD / "tokens.npz")
    tokens, syn_d = z["tokens"].astype(np.int64), z["syn_d"]
    mj = json.load(open(OUTD / "models.json"))
    names, meta = mj["models"], mj["meta"]
    assert int((OUTD / "eval_done.txt").read_text()) >= len(tokens), "eval has not finished"
    R, P = tokens.shape[0], 511
    nat = (syn_d == 0)[:, None] & np.ones((1, P), bool)
    cur, nxt = tokens[:, :-1], tokens[:, 1:]
    nx2 = np.concatenate([tokens[:, 2:], np.full((R, 1), PAD)], 1)
    prv = np.concatenate([np.full((R, 1), PAD), tokens[:, :-2]], 1)
    prop = lambda f: np.array([bool(f(t)) for t in txt])
    stx = [t.strip() for t in txt]
    ends_space = prop(lambda t: t[-1:].isspace())
    starts_nl = prop(lambda t: t[:1] == "\n")
    is_digits = np.array([s.isdigit() and s.isascii() for s in stx])
    num_val = np.array([int(s) if s.isdigit() and s.isascii() and len(s) <= 4 else -10 for s in stx])
    has_colon, has_semi = prop(lambda t: ":" in t), prop(lambda t: ";" in t)
    word_start = prop(lambda t: len(t) > 1 and t[0] == " " and t[1].isalpha())
    word_piece = prop(lambda t: t[:1].isalpha())
    cap_word = prop(lambda t: len(t) > 2 and t[0] == " " and t[1].isupper() and t[2:].isalpha() and t[2:].islower())
    cnt = lambda ch: np.array([t.count(ch) for t in txt], np.int64)
    emo = emoticon_positions(tokens, txt, ends_space) & nat
    near_emo = emo | np.pad(emo, ((0, 0), (1, 0)))[:, :P] | np.pad(emo, ((0, 0), (2, 0)))[:, :P]
    ok = nat & ~emo  # every damage check: natural text, never the emoticon colon itself

    # reference statistics from rows 0-4095 of the shard (disjoint from the held-out rows)
    ref = np.load(VD / "pile_val_4096x513.npy")[:, :512].astype(np.int64)
    uni = np.bincount(ref.ravel(), minlength=V + 1).astype(np.float64)
    bk, bc = np.unique((ref[:, :-1] * (V + 1) + ref[:, 1:]).ravel(), return_counts=True)

    def bigram_count(a, b):
        q = a * (V + 1) + b
        i = np.clip(np.searchsorted(bk, q), 0, len(bk) - 1)
        return np.where(bk[i] == q, bc[i], 0)
    big = bigram_count(cur, nxt)

    # does the text rule match where the edited subcomponent actually fires? (VPD's CI scan of rows 0-4095)
    sc = torch.load(VD / "scan_0_4096.pt")["fires"][(SITE, COMP)].numpy()
    fire = np.zeros((4096, P), bool)
    sel = (sc[:, 2] / 1e6 > CI_MIN) & (sc[:, 1] < P)
    fire[sc[sel, 0], sc[sel, 1]] = True
    emo_ref = emoticon_positions(ref, txt, ends_space)
    rule = {"n_fire": int(fire.sum()), "n_rule": int(emo_ref.sum()), "both": int((fire & emo_ref).sum()),
            "fire_tokens": {}}
    for t in ref[:, :P][fire]:
        rule["fire_tokens"][txt[t]] = rule["fire_tokens"].get(txt[t], 0) + 1
    log(f"emoticon rule vs CI>{CI_MIN} firings on rows 0-4095: {rule['both']} of {rule['n_fire']} firings, "
        f"{rule['both']} of {rule['n_rule']} rule positions")

    # context state along each row
    depth = {k: np.cumsum(cnt(o)[cur] - cnt(c)[cur], 1) for k, (o, c) in
             {"paren": ("(", ")"), "square": ("[", "]"), "curly": ("{", "}")}.items()}
    quote_odd = np.cumsum(cnt('"')[cur], 1) % 2 == 1
    ar = np.arange(P)[None, :]
    last_cs = np.maximum.accumulate(np.where((has_colon | has_semi)[cur], ar, -1), 1)
    dist_cs = np.where(last_cs >= 0, ar - last_cs, 10 ** 6)
    emo_before = np.pad(np.cumsum(emo, 1), ((0, 0), (1, 0)))[:, :P] > 0
    kw = prop(lambda t: t.strip() in ("def", "import", "return", "self", "elif", "const", "var", "function", "void",
                                      "public", "private", "static", "#include", "struct", "func", "fn", "let"))
    n_line = cnt("\n")[tokens].sum(1)
    latex = cnt("\\")[tokens].sum(1) > 10  # LaTeX is brace-heavy but is not code
    line_end = (prop(lambda t: t.rstrip().endswith((";", "{", "}")) and t.strip() != "")[cur] & starts_nl[nxt]).sum(1)
    code_row = (((line_end >= 6) | (kw[tokens].sum(1) >= 6)) & (n_line >= 10) & ~latex
                & (syn_d == 0))[:, None] & np.ones((1, P), bool)
    copy_d = np.full((R, P), -1)
    name_seen = np.zeros((R, P), bool)
    count_up = np.zeros((R, P), bool)
    in_url = np.zeros((R, P), bool)
    urlish = prop(lambda t: "http" in t or "www." in t)
    for r in np.nonzero(syn_d == 0)[0]:
        row = tokens[r].tolist()
        last, seen, url, nums = {}, set(), False, []
        for t in range(P):
            x, y = row[t], row[t + 1]
            j = last.get(x)
            if j is not None and row[j + 1] == y:
                copy_d[r, t] = t - j
            last[x] = t
            seen.add(x)
            name_seen[r, t] = y in seen
            if urlish[x]:
                url = True
            elif txt[x][:1].isspace() or txt[x][-1:].isspace():
                url = False
            in_url[r, t] = url and not txt[y][:1].isspace()
            v = num_val[y]
            count_up[r, t] = 0 < v <= 1000 and any(n == v - 1 for n in nums[-8:])
            nums.append(num_val[x] if num_val[x] >= 0 else -10)
    syn_copy = np.zeros((R, P), bool)
    for r in np.nonzero(syn_d > 0)[0]:
        syn_copy[r, 512 - syn_d[r]:] = True

    T = lambda s: tok.token_to_id(s.replace(" ", "Ġ").replace("\n", "Ċ"))
    tset = lambda *ss: np.isin(nxt, [T(s) for s in ss])
    nt = lambda f: prop(f)[nxt]
    ct = lambda f: prop(f)[cur]
    short_num = (prop(lambda t: t.isdigit() and len(t) <= 2))
    two_dig = prop(lambda t: t.isdigit() and len(t) == 2)
    colon_bare = ct(lambda t: t == ":")
    after_num_colon = colon_bare & short_num[prv]
    C = []  # (key, label, family, mask, kind)
    add = lambda key, label, fam, m, kind="ce": C.append((key, label, fam, m & ok if kind != "syn" else m, kind))
    F1, F2, F3, F4, F5, F6, F7, F8 = ("Colons and semicolons that are not emoticons", "The letter o elsewhere",
                                      "Closing brackets and quotes", "Copying from earlier in the text", "Numbers",
                                      "Grammar and common words", "Code, web addresses and lists", "Everything")
    add("spaced_colon", "After ' :' (space then colon) that is not an emoticon", F1, ct(lambda t: t == " :"))
    add("spaced_semi", "After ' ;' that is not an emoticon", F1, ct(lambda t: t == " ;"))
    add("spaced_eq", "After ' =' (assignment, not an emoticon)", F1, ct(lambda t: t == " ="))
    add("time_colon", "The colon of a clock time (10 → ':' in 10:30)", F1, short_num[cur] & tset(":") & two_dig[nx2])
    add("time_minutes", "The minutes of a clock time (10: → '30')", F1, after_num_colon & two_dig[nxt])
    add("ratio", "The number after a ratio or verse colon (3:1, John 3:16)", F1,
        after_num_colon & is_digits[nxt] & ~two_dig[nxt])
    add("label_colon_next", "The word after a label colon ('Note: the')", F1,
        colon_bare & (word_piece[prv] | word_start[prv]) & word_start[nxt])
    add("word_colon", "The colon after a word ('as follows' → ':')", F1, (word_piece[cur] | word_start[cur]) & tset(":"))
    add("colon_newline_prose", "The line break after a colon, outside code files", F1,
        ct(lambda t: t.rstrip(" ").endswith(":")) & starts_nl[nxt] & ~code_row)
    add("colon_newline_code", "The line break after a colon, in code files ('def f():')", F1,
        ct(lambda t: t.rstrip(" ").endswith(":")) & starts_nl[nxt] & code_row)
    add("dict_key", "After a quoted key and its colon ('\"key\":' → ' 1')", F1,
        ct(lambda t: t in ('":', "':", '":"', " :")) & (prop(lambda t: t.endswith(('"', "'")))[prv] | ct(lambda t: t in ('":', "':", '":"'))))
    add("url_scheme", "'://' after 'http' or 'https'", F1, ct(lambda t: t.strip() in ("http", "https", "ftp")) & nt(lambda t: t.startswith(":")))
    add("code_colon_mid", "After a colon inside code, not at a line end", F1, has_colon[cur] & ~starts_nl[nxt] & code_row)
    add("predict_semi_code", "Predicting the ';' that ends a code statement", F1, has_semi[nxt] & code_row)
    add("after_semi_code", "After a ';' in code files", F1, has_semi[cur] & code_row)
    add("after_semi_prose", "After a ';' outside code files", F1, has_semi[cur] & ~code_row)
    add("predict_colon", "Predicting any colon", F1, has_colon[nxt])
    add("o_bare", "Predicting a lone 'o' token", F2, nxt == O_TOK)
    add("o_piece", "Predicting a word piece that starts with 'o' ('o' in 'hello', 'ology')", F2,
        nt(lambda t: t[:1] == "o" and t != "o"))
    add("o_word", "Predicting a word that starts with 'o' (' of', ' on', ' or')", F2, nt(lambda t: t[:2] == " o"))
    add("O_cap", "Predicting a token that starts with 'O'", F2, nt(lambda t: t.strip()[:1] == "O"))
    add("close_paren", "Closing an open parenthesis ')'", F3, nt(lambda t: t.lstrip()[:1] == ")") & (depth["paren"] >= 1))
    add("close_square", "Closing an open square bracket ']'", F3, nt(lambda t: t.lstrip()[:1] == "]") & (depth["square"] >= 1))
    add("close_curly", "Closing an open curly brace '}'", F3, nt(lambda t: t.lstrip()[:1] == "}") & (depth["curly"] >= 1))
    add("close_quote", "Closing an open double quote", F3, nt(lambda t: t.lstrip()[:1] == '"') & quote_odd)
    add("close_tag", "Starting a closing markup tag '</'", F3, nt(lambda t: t.lstrip().startswith("</")))
    for d in SYN_D:
        add(f"syn_copy_{d}", f"Copying random tokens seen {d} tokens earlier", F4, syn_copy & (syn_d[:, None] == d), "syn")
    rare = big <= 2  # the pair (current, next) is rare in the reference text, so only the context can supply it
    for lo, hi in ((1, 16), (17, 64), (65, 256), (257, 511)):
        add(f"copy_{lo}_{hi}", f"Repeating a rare word pair seen {lo}–{hi} tokens earlier", F4,
            (copy_d >= lo) & (copy_d <= hi) & rare)
    add("name_repeat", "Repeating a capitalized name seen earlier", F4,
        cap_word[nxt] & name_seen & (uni[nxt] / uni.sum() < 2e-5))
    add("digit_cont", "Continuing a number with more digits", F5, is_digits[cur] & nt(lambda t: t[:1].isdigit()))
    add("count_up", "The next number of a count (… 3 → 4)", F5, count_up)
    add("year", "The last two digits of a year ('19' → '87')", F5, ct(lambda t: t.strip() in ("19", "20")) & two_dig[nxt])
    add("any_number", "Predicting any number", F5, is_digits[nxt])
    add("capital_after_period", "A capitalized word after a sentence ends", F6,
        ct(lambda t: t.strip() in (".", "!", "?", '."', '!"', '?"')) & nt(lambda t: len(t) > 1 and t[0] == " " and t[1].isupper()))
    add("agreement", "Verb agreement: is/are, was/were, has/have, does/do", F6,
        tset(" is", " are", " was", " were", " has", " have", " does", " do"), "pair")
    add("article", "Choosing 'a' or 'an'", F6, tset(" a", " an"), "pair")
    add("this_these", "Choosing 'this' or 'these'", F6, tset(" this", " these"), "pair")
    add("function_words", "Function words: the, of, and, to, in, a", F6, tset(" the", " of", " and", " to", " in", " a"))
    add("period", "A period ending a sentence", F6, (word_piece[cur] | word_start[cur]) & tset("."))
    add("comma", "A comma", F6, tset(","))
    add("newline", "A line break", F6, starts_nl[nxt])
    add("collocation", "Fixed phrases ('carried' → ' out', 'compared' → ' to')", F6,
        (big >= 20) & (big / np.maximum(uni[cur], 1) >= 0.5) & word_start[cur])
    add("word_piece", "Finishing a word split into pieces", F6, (word_start[cur] | word_piece[cur]) & word_piece[nxt])
    add("code_any", "Every token of code files", F7, code_row)
    add("code_call", "'(' after a name in code", F7, code_row & (word_piece[cur] | word_start[cur]) & nt(lambda t: t[:1] == "("))
    add("indent", "Indentation after a line break in code", F7, code_row & starts_nl[cur] & nt(lambda t: t != "" and t.isspace() and "\n" not in t))
    add("url", "Inside a web address", F7, in_url)
    add("list_marker", "A list marker at a line start ('-', '*', '1.')", F7,
        prop(lambda t: t.endswith("\n"))[cur] & (nt(lambda t: t.strip() in ("-", "*", "•", "+")) | (is_digits[nxt] & prop(lambda t: t.startswith("."))[nx2])))
    add("all", "Every held-out word (not an emoticon colon)", F8, np.ones((R, P), bool))
    emo_check = ("emoticon", "The emoticon colons themselves (the edit's target)", "Target", emo, "ce")

    # breakdown bins (natural text, every non-emoticon position)
    base_ptop = np.load(OUTD / "base_ptop.npy").astype(np.float32)
    fr = uni[nxt] / uni.sum() * 1e6
    bins = {
        "distance": [("on the ':' or ';'", dist_cs == 0)] + [
            (lab, (dist_cs >= a) & (dist_cs <= b)) for lab, a, b in
            (("1 after", 1, 1), ("2 after", 2, 2), ("3 after", 3, 3), ("4–7 after", 4, 7), ("8–15 after", 8, 15),
             ("16–63 after", 16, 63), ("64+ after", 64, 10 ** 5))] + [("no ':' or ';' earlier", dist_cs >= 10 ** 6)],
        "emoticon_context": [("no emoticon earlier in the document", ~emo_before), ("an emoticon earlier in the document", emo_before)],
        "confidence": [(f"{a:.0%}–{a + 0.1:.0%}", (base_ptop >= a) & (base_ptop < (a + 0.1 if a < 0.85 else 2)))
                       for a in np.round(np.arange(0, 1, 0.1), 1)],
        "frequency": [(lab, (fr >= a) & (fr < b)) for lab, a, b in
                      (("< 1 per million", 0, 1), ("1–10", 1, 10), ("10–100", 10, 100), ("100–1,000", 100, 1000),
                       ("1,000–10,000", 1000, 10 ** 4), ("> 10,000 per million", 10 ** 4, 10 ** 9))],
        "input_token": [(lab, m[cur]) for lab, m in (
            ("colon or semicolon", has_colon | has_semi), ("'=' sign", prop(lambda t: "=" in t) & ~has_colon & ~has_semi),
            ("other punctuation", prop(lambda t: t.strip() != "" and not any(c.isalnum() for c in t)) & ~has_colon & ~has_semi & ~prop(lambda t: "=" in t)),
            ("number", is_digits), ("line break or spaces", prop(lambda t: t != "" and t.isspace())),
            ("start of a word", word_start), ("middle of a word", word_piece))],
    }
    # per-row sums for every check and bin, for every model (+ base for reference)
    groups = [(k, lab, fam, m, kind) for k, lab, fam, m, kind in C] + [emo_check] + [
        (f"{ax}:{i}", lab, ax, m & ok, "ce") for ax, bl in bins.items() for i, (lab, m) in enumerate(bl)]
    flat = [np.flatnonzero(m) for _, _, _, m, _ in groups]
    rows_of = [f // P for f in flat]
    K, M = len(groups), len(names)
    N = np.stack([np.bincount(r, minlength=R) for r in rows_of], 1).astype(np.float64)
    ld = lambda nm: np.load(OUTD / f"{nm}.npy", mmap_mode="r")
    ce_b = np.asarray(ld("base_ce")).ravel().astype(np.float64)
    lpc_b = np.asarray(ld("base_lpc")).ravel().astype(np.float64)
    hit_b = (np.asarray(ld("base_top")).astype(np.int64) == nxt).ravel()
    softplus = lambda x: np.logaddexp(0, x)
    marg_b = -ce_b - lpc_b
    S = {k: np.zeros((R, K, M)) for k in ("kl", "loss", "dacc")}
    SB = {k: np.zeros((R, K)) for k in ("loss", "acc")}
    for g, (f, ro, grp) in enumerate(zip(flat, rows_of, groups)):
        pair = grp[4] == "pair"
        SB["loss"][:, g] = np.bincount(ro, softplus(-marg_b[f]) if pair else ce_b[f], minlength=R)
        SB["acc"][:, g] = np.bincount(ro, (marg_b[f] > 0) if pair else hit_b[f], minlength=R)
    allkl = {}
    for j, nm in enumerate(names):
        kl = np.asarray(ld(nm + "_kl")).ravel().astype(np.float64)
        dce = np.asarray(ld(nm + "_dce")).ravel().astype(np.float64)
        hit = (np.asarray(ld(nm + "_top")).astype(np.int64) == nxt).ravel()
        marg = -(ce_b + dce) - np.asarray(ld(nm + "_lpc")).ravel().astype(np.float64)
        for g, (f, ro, grp) in enumerate(zip(flat, rows_of, groups)):
            pair = grp[4] == "pair"
            S["kl"][:, g, j] = np.bincount(ro, kl[f], minlength=R)
            S["loss"][:, g, j] = np.bincount(ro, (softplus(-marg[f]) - softplus(-marg_b[f])) if pair else dce[f], minlength=R)
            S["dacc"][:, g, j] = np.bincount(ro, ((marg[f] > 0).astype(float) - (marg_b[f] > 0)) if pair else hit[f].astype(float) - hit_b[f], minlength=R)
        allkl[nm] = kl
        log(f"sums {nm}")
    rng = np.random.default_rng(0)
    wb = np.stack([np.bincount(rng.integers(0, R, R), minlength=R) for _ in range(1000)]).astype(np.float64)
    NM = np.repeat(N[:, :, None], M, 2).reshape(R, -1)
    stats, boots = {}, {}
    for k in S:
        est, lo, hi, bs = boot_means(S[k].reshape(R, -1), NM, wb)
        stats[k] = np.stack([est, lo, hi], -1).reshape(K, M, 3)
        boots[k] = bs.reshape(-1, K, M)
    base_stats = {k: boot_means(SB[k], N, wb)[0] for k in SB}
    pairs = [(nm, "vpd_match_" + nm) for nm in names if nm.startswith("lora") and "vpd_match_" + nm in names]
    out_groups = []
    for g, (key, lab, fam, m, kind) in enumerate(groups):
        pick = np.random.default_rng(g).choice(flat[g], min(4, len(flat[g])), replace=False) if len(flat[g]) else []
        e = {"key": key, "label": lab, "family": fam, "kind": kind, "n": int(N[:, g].sum()),
             "samples": [[tok.decode(tokens[f // P, max(0, f % P - 12):f % P + 1].tolist()), txt[tokens[f // P, f % P + 1]]]
                         for f in pick],
             "n_docs": int((N[:, g] > 0).sum()), "base_loss": float(base_stats["loss"][g]), "base_acc": float(base_stats["acc"][g]),
             "models": {nm: {k: [float(x) for x in stats[k][g, j]] for k in S} for j, nm in enumerate(names)}, "pairs": {}}
        tot = S["kl"][:, g, :].sum(0)
        for j, nm in enumerate(names):
            e["models"][nm]["kl_sum"] = float(tot[j])
        for L, Vm in pairs:
            jl, jv = names.index(L), names.index(Vm)
            with np.errstate(divide="ignore", invalid="ignore"):
                rb = boots["kl"][:, g, jl] / boots["kl"][:, g, jv]
                r0 = stats["kl"][g, jl, 0] / stats["kl"][g, jv, 0]
            db = boots["loss"][:, g, jl] - boots["loss"][:, g, jv]
            e["pairs"][L] = {"kl_ratio": [float(r0), *map(float, np.nanpercentile(rb, [2.5, 97.5]))],
                             "loss_diff": [float(stats["loss"][g, jl, 0] - stats["loss"][g, jv, 0]), *map(float, np.percentile(db, [2.5, 97.5]))]}
        out_groups.append(e)
    checks = [e for e in out_groups if ":" not in e["key"]]
    breakdowns = {ax: [e for e in out_groups if e["key"].startswith(ax + ":")] for ax in bins}

    # where the damage sits: by true next token, by input token, and how concentrated it is
    valid = ok.ravel()
    nflat, cflat = nxt.ravel(), cur.ravel()
    tot_valid = {nm: allkl[nm][valid].sum() for nm in names}
    top_tok, lorenz = {}, {}
    fracs = np.logspace(-6, 0, 61)
    nvalid = int(valid.sum())
    for nm in names:
        kv = allkl[nm][valid]
        top_tok[nm] = {}
        for axis, ids in (("next", nflat[valid]), ("input", cflat[valid])):
            s = np.bincount(ids, kv, minlength=V + 1)
            c = np.bincount(ids, minlength=V + 1)
            order = np.argsort(-s)[:15]
            top_tok[nm][axis] = [{"token": txt[i], "n": int(c[i]), "kl_share": float(s[i] / tot_valid[nm]),
                                  "mean_kl": float(s[i] / max(c[i], 1))} for i in order]
        srt = np.sort(kv)[::-1]
        cs = np.cumsum(srt) / srt.sum()
        lorenz[nm] = {"frac": fracs.tolist(), "share": [float(cs[max(int(f * nvalid) - 1, 0)]) for f in fracs]}

    # the most-disturbed contexts that are not emoticons, at most one per document
    ex_ok = (ok & ~near_emo).ravel()
    ptop_b = base_ptop.ravel()
    top_b = np.asarray(ld("base_top")).ravel()
    dec = lambda ids: tok.decode([int(i) for i in ids])
    examples = {}
    for nm in [n for p in pairs for n in p]:
        kl = np.where(ex_ok, allkl[nm], -1)
        dce = np.asarray(ld(nm + "_dce")).ravel()
        te, pe = np.asarray(ld(nm + "_top")).ravel(), np.asarray(ld(nm + "_ptop")).ravel()
        got, used = [], set()
        for f in np.argsort(-kl)[:5000]:
            r, t = divmod(int(f), P)
            if r in used:
                continue
            used.add(r)
            got.append({"row": r, "pos": t, "context": dec(tokens[r, max(0, t - 24):t]), "token": txt[tokens[r, t]],
                        "true_next": txt[tokens[r, t + 1]], "orig_top": txt[top_b[f]], "orig_p": float(ptop_b[f]),
                        "edit_top": txt[te[f]], "edit_p": float(pe[f]), "kl": float(allkl[nm][f]),
                        "p_true_orig": float(np.exp(-ce_b[f])), "p_true_edit": float(np.exp(-ce_b[f] - dce[f]))})
            if len(got) == 20:
                break
        examples[nm] = got

    res = {"description": __doc__, "models": names, "meta": meta, "pairs": pairs, "n_docs": int((syn_d == 0).sum()),
           "n_positions": nvalid, "n_emoticon_positions": int(emo.sum()), "emoticon_rule_vs_ci": rule,
           "checks": checks, "breakdowns": breakdowns, "top_tokens": top_tok, "lorenz": lorenz, "examples": examples,
           "kl_total_mean": {nm: float(tot_valid[nm] / nvalid) for nm in names}}
    json.dump(res, open(OUTD / "e4_side_effects.json", "w"), indent=1)
    log(f"wrote {OUTD / 'e4_side_effects.json'}: {len(checks)} checks, {nvalid} positions")
    for e in checks:
        print(f"{e['n']:8d}  {e['label']}")


if __name__ == "__main__":
    {"lora": lambda: stage_lora(sys.argv[2:]), "eval": stage_eval, "summarize": stage_summarize}[sys.argv[1]]()
