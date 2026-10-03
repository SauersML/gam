"""E4 edit methods (#2951): ways to write "o" after an emoticon colon with fewer off-target effects than VPD's
one-subcomponent edit or LoRA, all on h.2.mlp.down_proj of VPD's 4-layer target, compared at equal edit success.

Every closed-form method is a rank-one edit dW = s * w u^T: a READ u (which inputs the edit responds to) and a WRITE
w (what it adds to the output), with the strength s solved by bisection to the edit success being matched.
  reads   rome         u = C^-1 k / (k^T C^-1 k): the least mean-square change on general text (C = second moment of
                       down_proj inputs over general Pile text) with unit response on the mean fire key k (Meng et al.)
          memit(l)     u = (l C + K^T K / n)^-1 k: every fire key, ridge-traded against C
          null(r)      u = P_r k, P_r the projector off the top-r eigenvectors of C (AlphaEdit-style null space)
          contrast(b)  u = (C + b C_neg)^-1 k, C_neg the second moment at hard negatives: non-emoticon ':' ';' '='
                       positions and positions before a word piece (where VPD's edit does its damage)
          compiled     the native edit compiler (gam_mpd::compile::linear) with every fire key required to respond,
                       the general-text moment as metric, optionally a basis of preserved keys held fixed
          vpd / spec   a VPD subcomponent's own read V_c: 2359 (the paper's), or the most emoticon-specific one
  writes  fisher       w = H^-1 g / |H^-1 g|: H the output Fisher of general text pulled back to the site output,
                       g the mean gradient of log p(o) at the fire keys (largest gain per unit off-target KL)
          grad         w = g / |g|
LoRA with hard negatives: the paper's LoRA plus a KL term on windows around hard negatives (gamma), trained like
e4_replicate.py otherwise.

Stages (outputs in ~/mpd-data/frontier/e4_side/methods/):
  prep        fire keys and target gradients (282 training windows), C, C_neg, H on general text (val rows 2048-2303
              of pile_val_4096x513, emoticon positions excluded), VPD subcomponent specificity
  lora_neg G  train the hard-negative LoRA with gamma = G
  screen      every method's frontier: KL(edited || original) on 64 screening rows (3000-3063) against edit success
              on the 50 eval windows; methods matched at each success level by solving s
usage: mem-lease 2 venv/python e4_edit_methods.py prep|screen|lora_neg G
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e4_side_effects_data as E  # noqa: E402

OUT = E.FR / "e4_side/methods"
OUT.mkdir(parents=True, exist_ok=True)
REF = np.load(E.VD / "pile_val_4096x513.npy", mmap_mode="r")
MOMENT_ROWS, SCREEN_ROWS, NEG_ROWS = range(2048, 2304), range(3000, 3064), range(0, 2048)
SUCCESS = (0.8, 0.9, 0.95, 0.97, 0.985, 0.992)
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)


def vocab():
    tok = E.tokenizer()
    txt = [tok.decode([i]) for i in range(tok.get_vocab_size())] + [""]
    return tok, txt


def masks(tokens, txt):
    """Per [R, 511] position: emoticon colon (excluded), hard negative ':' ';' '=' use, next token a word piece."""
    ends_space = np.array([t[-1:].isspace() for t in txt])
    emo = E.emoticon_positions(tokens, txt, ends_space)
    colon = np.array([t.strip() in (":", ";", "=", '":', "':", ":-") or t.rstrip(" ").endswith(":") for t in txt])
    piece = np.array([t[:1].isalpha() for t in txt])
    cur, nxt = tokens[:, :-1], tokens[:, 1:]
    return emo, colon[cur] & ~emo, piece[nxt] & ~emo


def fire_batches(windows, B=16):
    """Right-padded batches of (tokens, fire offsets) windows."""
    for i in range(0, len(windows), B):
        chunk = windows[i:i + B]
        L = max(len(t) for t, _ in chunk)
        ids = np.zeros((len(chunk), L), np.int64)
        for j, (t, _) in enumerate(chunk):
            ids[j, :len(t)] = t.numpy()
        fire = [(j, q) for j, (t, f) in enumerate(chunk) for q in f if q + 1 < len(t)]
        yield ids, fire


def stage_prep():
    import torch
    target, U, V = E.load()
    W0 = target.site(E.SITE).W.clone()
    resid, final, head = E.split_forward(target)
    ev, train = E.harvest()
    tok, txt = vocab()
    # fire keys and the gradient of log p(o) with respect to the site output there
    keys, grads = [], []
    for ids, fire in fire_batches(train):
        ids_t = torch.from_numpy(ids).to("mps")
        with torch.no_grad():
            xmid, g2 = resid(ids_t)
        y = (g2 @ W0.T).detach().requires_grad_(True)
        lp = head(final.after(xmid + y))
        b = torch.tensor([j for j, _ in fire], device="mps")
        q = torch.tensor([q for _, q in fire], device="mps")
        (gy,) = torch.autograd.grad(lp[b, q, E.O_TOK].sum(), y)
        keys.append(g2[b, q].detach().cpu().double())
        grads.append(gy[b, q].detach().cpu().double())
    K, Gk = torch.cat(keys).numpy(), torch.cat(grads).numpy()
    ev_keys = []
    with torch.no_grad():
        for ids, fire in fire_batches(ev):
            _, g2 = resid(torch.from_numpy(ids).to("mps"))
            ev_keys.append(g2[[j for j, _ in fire], [q for _, q in fire]].cpu().double())
    Ke = torch.cat(ev_keys).numpy()
    log(f"{len(K)} training fire keys, {len(Ke)} eval fire keys")
    sample = []
    # general-text moments and the output Fisher at the site output
    C = np.zeros((3072, 3072))
    Cc, Cp = np.zeros((3072, 3072)), np.zeros((3072, 3072))
    H = np.zeros((768, 768))
    n = nc = npc = nh = 0
    rows = np.asarray(REF[MOMENT_ROWS.start:MOMENT_ROWS.stop, :512]).astype(np.int64)
    emo, colon, piece = masks(rows, txt)
    Vall = torch.load(str(E.VD / "s-55ea3f9b/model_400000.pth"), map_location="cpu", weights_only=True, mmap=True)[
        "_components." + E.SITE.replace(".", "-") + ".V"].float()
    Vm = Vall.to("mps")
    spec = {k: torch.zeros(Vall.shape[1], dtype=torch.float64) for k in ("general", "colon", "piece")}
    torch.manual_seed(0)
    for i in range(0, len(rows), 2):
        ids_t = torch.from_numpy(rows[i:i + 2]).to("mps")
        with torch.no_grad():
            xmid, g2 = resid(ids_t)
        g = g2[:, :-1]
        keep = torch.from_numpy(~emo[i:i + 2]).to("mps")
        if i % 16 == 0:
            sample.append(g[keep][::64].cpu().double())
        for mask, acc, key in ((keep, C, "general"), (torch.from_numpy(colon[i:i + 2]).to("mps"), Cc, "colon"),
                               (torch.from_numpy(piece[i:i + 2]).to("mps"), Cp, "piece")):
            x = g[mask]
            acc += (x.T @ x).cpu().double().numpy()
            spec[key] += ((x @ Vm) ** 2).sum(0).cpu().double()
            if key == "general":
                n += len(x)
            elif key == "colon":
                nc += len(x)
            else:
                npc += len(x)
        for j in range(2) if i < 32 else ():  # the Fisher from 32 rows, one at a time: labels sampled from the
            y = (g2[j:j + 1] @ W0.T).detach().requires_grad_(True)  # model, so E[g g^T] is the Fisher
            lp = head(final.after(xmid[j:j + 1] + y))[:, :-1]
            lab = (lp.detach() - torch.log(-torch.log(torch.rand_like(lp)))).argmax(-1)
            (gy,) = torch.autograd.grad(lp.gather(-1, lab[..., None]).sum(), y)
            gy = gy[:, :-1][keep[j:j + 1]]
            H += (gy.T @ gy).cpu().double().numpy()
            nh += len(gy)
            del lp, lab, y
        torch.mps.empty_cache()
        if i % 32 == 0:
            log(f"moments: rows {i}")
            torch.mps.empty_cache()
    kf = torch.from_numpy(K).float().to("mps")
    spec_fire = ((kf @ Vm) ** 2).mean(0).cpu().double().numpy()
    np.savez(OUT / "prep.npz", K=K, Gk=Gk, K_eval=Ke, X_general=torch.cat(sample).numpy(), C=C / n, C_colon=Cc / nc, C_piece=Cp / npc, H=H / nh, n=n, n_colon=nc,
             n_piece=npc, spec_fire=spec_fire, spec_general=(spec["general"] / n).numpy(),
             spec_colon=(spec["colon"] / nc).numpy(), spec_piece=(spec["piece"] / npc).numpy())
    log(f"moments over {n} general, {nc} colon, {npc} word-piece positions; Fisher over {nh}")


def train_lora_neg(target, pool, negs, lam, gamma, steps=300, batch=256, micro=8):
    """e4_replicate.py's LoRA plus gamma * mean KL(edited || original) over hard-negative windows."""
    import torch
    import torch.nn.functional as F
    site = target.site(E.SITE)
    W0 = site.W.clone()
    torch.manual_seed(0)
    A = (torch.randn(1, W0.shape[1]) * 0.01).to("mps").requires_grad_(True)
    B = torch.zeros(W0.shape[0], 1, device="mps", requires_grad=True)
    opt = torch.optim.AdamW([A, B], lr=1e-3)

    def pack(seqs):
        L = max(len(t) for t, _ in seqs)
        T = torch.zeros(len(seqs), L, dtype=torch.long)
        fire = torch.zeros(len(seqs), L, dtype=torch.bool)
        pad = torch.zeros(len(seqs), L, dtype=torch.bool)
        for i, (t, fp) in enumerate(seqs):
            T[i, :len(t)] = t
            fire[i, fp] = True
            pad[i, :len(t)] = True
        return T, fire, pad
    seqs = []
    for toks, f in pool:
        t = toks.clone()
        fp = [q for q in f if q + 1 < len(t)]
        for q in fp:
            t[q + 1] = E.O_TOK
        seqs.append((t, fp))
    T, fire, pad = pack(seqs)
    Tn, _, padn = pack([(t, []) for t in negs])
    n, nn_ = len(seqs), len(negs)
    for step in range(steps):
        idx = torch.randint(n, (min(batch, n),))
        jdx = torch.randint(nn_, (min(batch, nn_),))
        n_fire = int(fire[idx].sum())
        n_kl = int((pad[idx] & ~fire[idx]).sum())
        n_neg = int(padn[jdx].sum())
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
        for j in range(0, len(jdx), micro):
            mj = jdx[j:j + micro]
            tb, pb = Tn[mj].to("mps"), padn[mj].to("mps")
            with torch.no_grad():
                site.W = W0
                lb = F.log_softmax(target(tb), -1)
            site.W = W0 + B @ A
            le = F.log_softmax(target(tb), -1)
            (gamma * (le.exp() * (le - lb))[pb].sum() / max(n_neg, 1)).backward()
            del le, lb
        opt.step()
        torch.mps.empty_cache()  # keeps the footprint inside the 2 GiB reservation
    site.W = W0
    return A.detach().cpu(), B.detach().cpu()


def negative_windows(txt, per_kind=200):
    """Hard-negative windows (20 tokens either side) from val rows 0-2047: centred on a non-emoticon ':' ';' '='
    use, or on a position before a word piece."""
    import torch
    rows = np.asarray(REF[NEG_ROWS.start:NEG_ROWS.stop, :512]).astype(np.int64)
    _, colon, piece = masks(rows, txt)
    rng = np.random.default_rng(0)
    out = []
    for m in (colon, piece):
        r, p = np.nonzero(m)
        for k in rng.choice(len(r), per_kind, replace=False):
            a = max(0, p[k] - E.SIDE)
            out.append(torch.from_numpy(rows[r[k], a:p[k] + E.SIDE + 1].copy()))
    return out


def stage_lora_neg(gamma):
    import torch
    target, _, _ = E.load()
    ev, train = E.harvest()
    _, txt = vocab()
    negs = negative_windows(txt)
    A, B = train_lora_neg(target, train, negs, lam=10.0, gamma=gamma)
    name = f"loraneg_g{gamma:g}"
    torch.save({name: {"A": A, "B": B}}, OUT / f"lora_{name}.pt")
    pf = E.p_fire_fn(target, ev)
    log(f"{name}: p_fire {pf(target.site(E.SITE).W + (B @ A).to('mps')):.4f}")


def reads_and_writes(z, V):
    """The candidate reads u (unit response on the mean fire key) and writes w (unit norm), from prep.npz."""
    import scipy.linalg as sl
    K, C, Cneg, H = z["K"], z["C"], z["C_colon"] + z["C_piece"], z["H"]
    kbar, ghat = K.mean(0), z["Gk"].mean(0)
    ridge = 1e-6 * np.trace(C) / len(C)
    Ci = lambda M, v: sl.solve(M + ridge * np.eye(len(M)), v, assume_a="pos")
    reads = {"rome": Ci(C, kbar)}
    for lam in (0.1, 1.0, 10.0):
        reads[f"memit_l{lam:g}"] = Ci(lam * C + K.T @ K / len(K), kbar)
    evals, evecs = sl.eigh(C)
    evecs = evecs[:, ::-1]
    for r in (0, 64, 256, 1024, 2048, 2816):
        reads[f"null_r{r}"] = kbar - evecs[:, :r] @ (evecs[:, :r].T @ kbar)
    for beta in (1.0, 10.0, 100.0):
        reads[f"contrast_b{beta:g}"] = Ci(C + beta * Cneg, kbar)
    reads["vpd2359_read"] = V[:, E.COMP].copy()
    sf, sg = z["spec_fire"], z["spec_general"]
    ok = sf >= 0.01 * sf.max()
    spec_c = int(np.argmax(np.where(ok, sf / np.maximum(sg, 1e-12), -1)))
    reads[f"spec{spec_c}_read"] = V[:, spec_c].copy()
    reads = {k: u / (u @ kbar) for k, u in reads.items()}  # unit response on the mean fire key
    hv, hV = sl.eigh(H)
    hv = np.maximum(hv, 1e-6 * hv.max())
    wf = hV @ ((hV.T @ ghat) / hv)
    writes = {"fisher": wf / np.linalg.norm(wf), "grad": ghat / np.linalg.norm(ghat)}
    return reads, writes, spec_c


def stage_screen():
    import torch
    z = np.load(OUT / "prep.npz")
    target, U, V = E.load()
    W0 = target.site(E.SITE).W.clone()
    u_o = target.wte[E.O_TOK] / target.wte[E.O_TOK].norm()
    ev, _ = E.harvest()
    pf = E.p_fire_fn(target, ev)
    resid, final, head = E.split_forward(target)
    _, txt = vocab()
    raw = torch.load(str(E.VD / "s-55ea3f9b/model_400000.pth"), map_location="cpu", weights_only=True, mmap=True)
    Vall = raw["_components." + E.SITE.replace(".", "-") + ".V"].double().numpy()
    del raw
    reads, writes, spec_c = reads_and_writes(z, Vall)
    log(f"most emoticon-specific subcomponent read: {spec_c}")
    rows = np.asarray(REF[SCREEN_ROWS.start:SCREEN_ROWS.start + 32, :512]).astype(np.int64)
    emo, _, _ = masks(rows, txt)
    keep = torch.from_numpy(~emo)
    cache = []
    with torch.no_grad():
        for i in range(0, len(rows), 2):
            xmid, g2 = resid(torch.from_numpy(rows[i:i + 2]).to("mps"))
            cache.append((xmid.cpu(), g2.cpu()))

    @torch.no_grad()
    def screen_kl(dW):
        tot, cnt = 0.0, 0
        for c, (xmid, g2) in enumerate(cache):
            xmid, g2 = xmid.to("mps"), g2.to("mps")
            xb, xe = final(xmid, g2, W0)[:, :-1], final(xmid, g2, W0 + dW)[:, :-1]
            k = keep[2 * c:2 * c + 2].to("mps")
            for p0 in range(0, 511, 128):
                lb, le = head(xb[:, p0:p0 + 128]), head(xe[:, p0:p0 + 128])
                kl = (le.exp() * (le - lb)).sum(-1)
                tot += float(kl[k[:, p0:p0 + 128]].sum())
                cnt += int(k[:, p0:p0 + 128].sum())
        torch.mps.empty_cache()
        return tot / cnt

    def solve(family, p_target, lo=1e-3, hi=1e4):
        """The strength giving edit success p_target (bisection in log strength), or None if out of reach."""
        if pf(W0 + family(hi)) < p_target:
            return None
        for _ in range(30):
            mid = (lo * hi) ** 0.5
            lo, hi = (mid, hi) if pf(W0 + family(mid)) < p_target else (lo, mid)
            if hi / lo < 1.001:
                break
        return (lo * hi) ** 0.5

    path = OUT / "screen.json"
    res = json.load(open(path)) if path.exists() else {}
    res["spec_component"] = spec_c
    t32 = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to("mps")
    fams = {}
    for rn, u in reads.items():
        for wn in (("fisher", "grad") if rn == "rome" else ("fisher",)):
            w, uu = t32(writes[wn]), t32(u)
            fams[f"{rn}+{wn}"] = lambda s, w=w, uu=uu: s * torch.outer(w, uu)
    for plan in sorted((OUT / "compile").glob("plan_*.left.npy")):
        name = plan.name[len("plan_"):-len(".left.npy")]
        unit = t32(np.load(plan) @ np.load(plan.with_name(plan.name.replace(".left.", ".right."))).T)
        fams[name] = lambda s, unit=unit: s * unit
    vpd_dw = lambda a: torch.outer((-a * u_o).to(torch.bfloat16).float() - U, V)
    fams["vpd"] = vpd_dw
    for fname, fam in fams.items():
        if fname in res:
            continue
        pts = []
        for p in SUCCESS:
            s = solve(fam, p, *((0.3, 20.0) if fname == "vpd" else (1e-3, 1e4)))
            if s is None:
                pts.append({"target": p, "strength": None})
                continue
            dW = fam(s)
            pts.append({"target": p, "strength": s, "p_fire": pf(W0 + dW), "kl": screen_kl(dW),
                        "norm": float(dW.norm())})
        res[fname] = pts
        log(f"{fname}: " + "  ".join(f"{q['target']}:{q['kl']:.2e}" if q.get("kl") is not None else f"{q['target']}:--" for q in pts))
        json.dump(res, open(path, "w"), indent=1)
    deltas = E.lora_deltas()
    for f in OUT.glob("lora_loraneg_*.pt"):
        deltas.update(torch.load(f))
    for nm, d in deltas.items():
        if nm in res:
            continue
        dW = (d["B"] @ d["A"]).to("mps")
        res[nm] = [{"target": None, "p_fire": pf(W0 + dW), "kl": screen_kl(dW), "norm": float(dW.norm())}]
        log(f"{nm}: p {res[nm][0]['p_fire']:.4f} kl {res[nm][0]['kl']:.2e}")
        json.dump(res, open(path, "w"), indent=1)


def stage_compile_export():
    """The native edit compiler's problems: every training fire key must respond with the Fisher write (unit
    strength), metric = the general-text input moment, claim about the span of the held-out eval fire keys,
    optionally the top-r general-text eigendirections preserved exactly; off-target damage on general inputs."""
    import scipy.linalg as sl
    import torch
    z = np.load(OUT / "prep.npz")
    raw = torch.load(str(E.VD / "s-55ea3f9b/model_400000.pth"), map_location="cpu", weights_only=True, mmap=True)
    Vall = raw["_components." + E.SITE.replace(".", "-") + ".V"].double().numpy()
    del raw
    from safetensors.torch import load_file
    W0 = load_file(str(E.VD / "t-9d2b8f02/model_step_99999.safetensors"))[E.SITE + ".weight"].double().numpy()
    _, writes, _ = reads_and_writes(z, Vall)
    d = OUT / "compile"
    d.mkdir(exist_ok=True)
    C = z["C"]
    C = C + 1e-6 * np.trace(C) / len(C) * np.eye(len(C))
    K = z["K"]
    files = {"native": W0, "inputs": K, "targets": K @ W0.T + writes["fisher"][None, :], "moment": C,
             "class_span": z["K_eval"], "off_target": z["X_general"]}
    evecs = sl.eigh(C)[1][:, ::-1]
    for r in (64, 256, 1024):
        files[f"preserve_r{r}"] = evecs[:, :r].T.copy()
    for k, v in files.items():
        np.save(d / f"{k}.npy", np.ascontiguousarray(v, dtype="<f8"))
    base = {"storage": E.SITE + ".weight", "native": "native.npy", "inputs": "inputs.npy", "targets": "targets.npy",
            "class": "span_of", "class_span": "class_span.npy", "moment": "moment.npy", "off_target": "off_target.npy"}
    problems = [dict(base, name="compiled", out="plan_compiled")]
    for r in (64, 256, 1024):
        problems.append(dict(base, name=f"compiled_keep{r}", preserve=f"preserve_r{r}.npy", out=f"plan_compiled_keep{r}"))
    json.dump({"problems": problems}, open(d / "manifest.json", "w"), indent=1)
    log(f"wrote {d / 'manifest.json'}: {len(problems)} problems, {len(K)} fire keys")


def family_deltas():
    """Every closed-form family as strength -> delta W (torch, mps), from prep.npz and the compiled plans."""
    import torch
    z = np.load(OUT / "prep.npz")
    target, U, V = E.load()
    u_o = target.wte[E.O_TOK] / target.wte[E.O_TOK].norm()
    raw = torch.load(str(E.VD / "s-55ea3f9b/model_400000.pth"), map_location="cpu", weights_only=True, mmap=True)
    Vall = raw["_components." + E.SITE.replace(".", "-") + ".V"].double().numpy()
    del raw
    reads, writes, spec_c = reads_and_writes(z, Vall)
    t32 = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to("mps")
    fams = {}
    for rn, u in reads.items():
        for wn in (("fisher", "grad") if rn == "rome" else ("fisher",)):
            w, uu = t32(writes[wn]), t32(u)
            fams[f"{rn}+{wn}"] = lambda s, w=w, uu=uu: s * torch.outer(w, uu)
    for plan in sorted((OUT / "compile").glob("plan_*.left.npy")):
        name = plan.name[len("plan_"):-len(".left.npy")]
        unit = t32(np.load(plan) @ np.load(plan.with_name(plan.name.replace(".left.", ".right."))).T)
        fams[name] = lambda s, unit=unit: s * unit
    fams["vpd"] = lambda a: torch.outer((-a * u_o).to(torch.bfloat16).float() - U, V)
    return target, fams, spec_c


def stage_assemble():
    """The final edit set: from every family, the configuration with the least screening KL at 98.5% success,
    re-solved to the headline LoRA's exact success; the headline LoRA and the VPD edit at that success; the best
    hard-negative LoRA with a VPD edit at its own success. Writes methods/final.pt and final.json."""
    import torch
    scr = json.load(open(OUT / "screen.json"))
    target, fams, _ = family_deltas()
    W0 = target.site(E.SITE).W.clone()
    ev, _ = E.harvest()
    pf = E.p_fire_fn(target, ev)
    head = json.load(open(E.FR / "e4_side/models.json"))["meta"]
    p_star = head["lora282_lam10"]["p_fire"]

    def at(fam, p, lo, hi):
        for _ in range(40):
            mid = (lo * hi) ** 0.5
            lo, hi = (mid, hi) if pf(W0 + fam(mid)) < p else (lo, mid)
        s = (lo * hi) ** 0.5
        return s, fam(s)
    groups = {"rome": ["rome+fisher"], "rome_gradwrite": ["rome+grad"], "memit": [k for k in fams if k.startswith("memit")],
              "nullspace": [k for k in fams if k.startswith("null")], "contrast": [k for k in fams if k.startswith("contrast")],
              "vpd_read_fisher_write": ["vpd2359_read+fisher"], "specific_subcomponent": [k for k in fams if k.startswith("spec")],
              "compiled": [k for k in fams if k.startswith("compiled")]}
    kl_at = lambda k: next((q["kl"] for q in scr.get(k, []) if q.get("target") == 0.985 and q.get("kl") is not None), np.inf)
    out, meta = {}, {}
    for g, ks in groups.items():
        ks = [k for k in ks if np.isfinite(kl_at(k))]
        if not ks:
            continue
        best = min(ks, key=kl_at)
        s, dW = at(fams[best], p_star, 1e-3, 1e4)
        out[g] = dW.cpu()
        meta[g] = {"method": g, "config": best, "strength": s, "p_fire": pf(W0 + dW)}
        log(f"{g}: {best} strength {s:.4g} p_fire {meta[g]['p_fire']:.4f}")
    s, dW = at(fams["vpd"], p_star, 0.3, 20.0)
    out["vpd"], meta["vpd"] = dW.cpu(), {"method": "vpd", "alpha": s, "p_fire": pf(W0 + dW)}
    deltas = E.lora_deltas()
    out["lora"] = (deltas["lora282_lam10"]["B"] @ deltas["lora282_lam10"]["A"])
    meta["lora"] = {"method": "lora", "config": "lora282_lam10", "p_fire": pf(W0 + out["lora"].to("mps"))}
    negs = {}
    for f in OUT.glob("lora_loraneg_*.pt"):
        negs.update(torch.load(f))
    pairs = [[g, "vpd"] for g in out if g not in ("vpd",)]
    if negs:
        best = min(negs, key=lambda k: scr.get(k, [{"kl": np.inf}])[0]["kl"])
        dW = negs[best]["B"] @ negs[best]["A"]
        p = pf(W0 + dW.to("mps"))
        out["lora_hardneg"], meta["lora_hardneg"] = dW, {"method": "lora_hardneg", "config": best, "p_fire": p}
        s, dv = at(fams["vpd"], p, 0.3, 20.0)
        out["vpd_at_hardneg"], meta["vpd_at_hardneg"] = dv.cpu(), {"method": "vpd", "alpha": s, "p_fire": pf(W0 + dv)}
        pairs = [q for q in pairs if q[0] not in ("lora_hardneg", "vpd_at_hardneg")] + [["lora_hardneg", "vpd_at_hardneg"]]
    torch.save(out, OUT / "final.pt")
    json.dump({"meta": meta, "pairs": pairs}, open(OUT / "final.json", "w"), indent=1)
    log(f"final set: {list(out)}")


if __name__ == "__main__":
    {"prep": stage_prep, "screen": stage_screen, "compile_export": stage_compile_export, "assemble": stage_assemble, "lora_neg": lambda: stage_lora_neg(float(sys.argv[2]))}[sys.argv[1]]()
