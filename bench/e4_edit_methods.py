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
        ids_t = torch.from_numpy(ids).to(E.DEVICE)
        with torch.no_grad():
            xmid, g2 = resid(ids_t)
        y = (g2 @ W0.T).detach().requires_grad_(True)
        lp = head(final.after(xmid + y))
        b = torch.tensor([j for j, _ in fire], device=E.DEVICE)
        q = torch.tensor([q for _, q in fire], device=E.DEVICE)
        (gy,) = torch.autograd.grad(lp[b, q, E.O_TOK].sum(), y)
        keys.append(g2[b, q].detach().cpu().double())
        grads.append(gy[b, q].detach().cpu().double())
    K, Gk = torch.cat(keys).numpy(), torch.cat(grads).numpy()
    ev_keys = []
    with torch.no_grad():
        for ids, fire in fire_batches(ev):
            _, g2 = resid(torch.from_numpy(ids).to(E.DEVICE))
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
    Vm = Vall.to(E.DEVICE)
    spec = {k: torch.zeros(Vall.shape[1], dtype=torch.float64) for k in ("general", "colon", "piece")}
    torch.manual_seed(0)
    for i in range(0, len(rows), 2):
        ids_t = torch.from_numpy(rows[i:i + 2]).to(E.DEVICE)
        with torch.no_grad():
            xmid, g2 = resid(ids_t)
        g = g2[:, :-1]
        keep = torch.from_numpy(~emo[i:i + 2]).to(E.DEVICE)
        if i % 16 == 0:
            sample.append(g[keep][::64].cpu().double())
        for mask, acc, key in ((keep, C, "general"), (torch.from_numpy(colon[i:i + 2]).to(E.DEVICE), Cc, "colon"),
                               (torch.from_numpy(piece[i:i + 2]).to(E.DEVICE), Cp, "piece")):
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
            x3 = final.after(xmid[j:j + 1] + y)
            gy = torch.zeros_like(y)
            for p0 in range(0, 511, 128):  # the vocab-sized log-softmax in blocks of positions
                lp = head(x3[:, p0:min(p0 + 128, 511)])
                lab = (lp.detach() - torch.log(-torch.log(torch.rand_like(lp)))).argmax(-1)
                gy += torch.autograd.grad(lp.gather(-1, lab[..., None]).sum(), y, retain_graph=True)[0]
                del lp, lab
            gy = gy[:, :-1][keep[j:j + 1]]
            H += (gy.T @ gy).cpu().double().numpy()
            nh += len(gy)
            del x3, y
        E.empty_cache()
        if i % 32 == 0:
            log(f"moments: rows {i}")
            E.empty_cache()
    kf = torch.from_numpy(K).float().to(E.DEVICE)
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
    A = (torch.randn(1, W0.shape[1]) * 0.01).to(E.DEVICE).requires_grad_(True)
    B = torch.zeros(W0.shape[0], 1, device=E.DEVICE, requires_grad=True)
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
            tb, fb, pb = T[mi].to(E.DEVICE), fire[mi].to(E.DEVICE), pad[mi].to(E.DEVICE)
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
            tb, pb = Tn[mj].to(E.DEVICE), padn[mj].to(E.DEVICE)
            with torch.no_grad():
                site.W = W0
                lb = F.log_softmax(target(tb), -1)
            site.W = W0 + B @ A
            le = F.log_softmax(target(tb), -1)
            (gamma * (le.exp() * (le - lb))[pb].sum() / max(n_neg, 1)).backward()
            del le, lb
        opt.step()
        E.empty_cache()  # keeps the footprint inside the 2 GiB reservation
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
            xmid, g2 = resid(torch.from_numpy(rows[i:i + 2]).to(E.DEVICE))
            cache.append((xmid.cpu(), g2.cpu()))

    @torch.no_grad()
    def screen_kl(dW):
        tot, cnt = 0.0, 0
        for c, (xmid, g2) in enumerate(cache):
            xmid, g2 = xmid.to(E.DEVICE), g2.to(E.DEVICE)
            xb, xe = final(xmid, g2, W0)[:, :-1], final(xmid, g2, W0 + dW)[:, :-1]
            k = keep[2 * c:2 * c + 2].to(E.DEVICE)
            for p0 in range(0, 511, 128):
                lb, le = head(xb[:, p0:p0 + 128]), head(xe[:, p0:p0 + 128])
                kl = (le.exp() * (le - lb)).sum(-1)
                tot += float(kl[k[:, p0:p0 + 128]].sum())
                cnt += int(k[:, p0:p0 + 128].sum())
        E.empty_cache()
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
    t32 = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(E.DEVICE)
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
        dW = (d["B"] @ d["A"]).to(E.DEVICE)
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


def family_deltas(only=None):
    """Every closed-form family as strength -> delta W (torch, on the device), from prep.npz and the compiled plans;
    with `only`, just the compiled plans named there (the others are cheap and always built)."""
    import torch
    z = np.load(OUT / "prep.npz")
    target, U, V = E.load()
    u_o = target.wte[E.O_TOK] / target.wte[E.O_TOK].norm()
    raw = torch.load(str(E.VD / "s-55ea3f9b/model_400000.pth"), map_location="cpu", weights_only=True, mmap=True)
    Vall = raw["_components." + E.SITE.replace(".", "-") + ".V"].double().numpy()
    del raw
    reads, writes, spec_c = reads_and_writes(z, Vall)
    t32 = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(E.DEVICE)
    fams = {}
    for rn, u in reads.items():
        for wn in (("fisher", "grad") if rn == "rome" else ("fisher",)):
            w, uu = t32(writes[wn]), t32(u)
            fams[f"{rn}+{wn}"] = lambda s, w=w, uu=uu: s * torch.outer(w, uu)
    for plan in sorted((OUT / "compile").glob("plan_*.left.npy")):
        name = plan.name[len("plan_"):-len(".left.npy")]
        if only is not None and name not in only:
            continue
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
    groups = {"rome": ["rome+fisher"], "memit": [k for k in fams if k.startswith("memit")],
              "nullspace": [k for k in fams if k.startswith("null")], "contrast": [k for k in fams if k.startswith("contrast")],
              "specific_subcomponent": [k for k in fams if k.startswith("spec")],
              "compiled_every_key": [k for k in fams if k.startswith("compiled_old")]}
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
    meta["lora"] = {"method": "lora", "config": "lora282_lam10", "p_fire": pf(W0 + out["lora"].to(E.DEVICE))}
    negs = {}
    for f in OUT.glob("lora_loraneg_*.pt"):
        negs.update(torch.load(f))
    pairs = [[g, "vpd"] for g in out if g not in ("vpd",)]
    if negs:
        best = min(negs, key=lambda k: scr.get(k, [{"kl": np.inf}])[0]["kl"])
        dW = negs[best]["B"] @ negs[best]["A"]
        p = pf(W0 + dW.to(E.DEVICE))
        out["lora_hardneg"], meta["lora_hardneg"] = dW, {"method": "lora_hardneg", "config": best, "p_fire": p}
        s, dv = at(fams["vpd"], p, 0.3, 20.0)
        out["vpd_at_hardneg"], meta["vpd_at_hardneg"] = dv.cpu(), {"method": "vpd", "alpha": s, "p_fire": pf(W0 + dv)}
        pairs = [q for q in pairs if q[0] not in ("lora_hardneg", "vpd_at_hardneg")] + [["lora_hardneg", "vpd_at_hardneg"]]
    torch.save(out, OUT / "final.pt")
    json.dump({"meta": meta, "pairs": pairs}, open(OUT / "final.json", "w"), indent=1)
    log(f"final set: {list(out)}")


def stage_assemble_extra(name, fam_names):
    """A further edit set NAME: the named families at the headline LoRA's exact success, with the VPD edit there."""
    import torch
    target, fams, _ = family_deltas()
    W0 = target.site(E.SITE).W.clone()
    ev, _ = E.harvest()
    pf = E.p_fire_fn(target, ev)
    p_star = json.load(open(E.FR / "e4_side/models.json"))["meta"]["lora282_lam10"]["p_fire"]
    out, meta = {}, {}
    for f in fam_names + ["vpd"]:
        lo, hi = (0.3, 20.0) if f == "vpd" else (1e-3, 1e4)
        for _ in range(40):
            mid = (lo * hi) ** 0.5
            lo, hi = (mid, hi) if pf(W0 + fams[f](mid)) < p_star else (lo, mid)
        dW = fams[f]((lo * hi) ** 0.5)
        out[f], meta[f] = dW.cpu(), {"method": f, "strength": (lo * hi) ** 0.5, "p_fire": pf(W0 + dW)}
        log(f"{f}: p_fire {meta[f]['p_fire']:.4f}")
    torch.save(out, OUT / f"{name}.pt")
    json.dump({"meta": meta, "pairs": [[f, "vpd"] for f in fam_names]}, open(OUT / f"{name}.json", "w"), indent=1)


def stage_compile_span():
    """Compiler problems exact only on the k-dimensional span of fire-key combinations that the required response
    needs most: the top-k eigenvectors a_j of the whitened key Gram K C^-1 K^T, with inputs b_j = K^T a_j and
    targets W0 b_j + (a_j . 1) w, so each b_j gets exactly the response its keys ask for. k = 1, 2, 4, ..., 64."""
    import scipy.linalg as sl
    d = OUT / "compile"
    K, C, W0 = np.load(d / "inputs.npy"), np.load(d / "moment.npy"), np.load(d / "native.npy")
    w = (np.load(d / "targets.npy") - K @ W0.T).mean(0)
    M = K @ sl.solve(C, K.T, assume_a="pos")
    vals, vecs = sl.eigh((M + M.T) / 2)
    vecs = vecs[:, ::-1]
    base = json.load(open(d / "manifest.json"))["problems"][0]
    problems = []
    for k in (1, 2, 4, 8, 16, 32, 64):
        A = vecs[:, :k]
        B = A.T @ K
        np.save(d / f"inputs_span{k}.npy", np.ascontiguousarray(B))
        np.save(d / f"targets_span{k}.npy", np.ascontiguousarray(B @ W0.T + np.outer(A.sum(0), w)))
        problems.append(dict(base, name=f"compiled_span{k}", inputs=f"inputs_span{k}.npy", targets=f"targets_span{k}.npy",
                             class_="sample", out=f"plan_compiled_span{k}"))
    for q in problems:
        q["class"] = q.pop("class_")
        q.pop("class_span", None)
    json.dump({"problems": problems}, open(d / "manifest_span.json", "w"), indent=1)
    log(f"whitened key Gram eigenvalues (top 8): {np.round(vals[::-1][:8], 3).tolist()}")


def stage_heldout():
    """Held-out emoticons: every emoticon (by the text rule) in rows 0-19999 of the val shard's row group 2 (never
    scanned, harvested or trained on), as a window of 20 tokens either side like the edit's own windows. P('o') at the
    emoticon colon under every edit of the final set (E4_VARIANTS=final), with 95% intervals over emoticons."""
    import pyarrow.parquet as pq
    import torch
    sys.path.insert(0, str(E.VD))
    from vpd_model import VAL_PARQUET
    path = OUT / "heldout_emoticons.npz"
    _, txt = vocab()
    if not path.exists():
        parts = []
        for batch in pq.ParquetFile(VAL_PARQUET).iter_batches(batch_size=2000, row_groups=[2], columns=["input_ids"]):
            col = batch.column(0)
            parts.append(col.flatten().to_numpy().reshape(len(col), -1)[:, :512].astype(np.int32))
            if sum(len(x) for x in parts) >= 20000:
                break
        rows = np.concatenate(parts)[:20000].astype(np.int64)
        del parts
        ends_space = np.array([t[-1:].isspace() for t in txt])
        r, q = np.nonzero(E.emoticon_positions(rows, txt, ends_space))
        wins = np.zeros((len(r), 2 * E.SIDE + 1), np.int64)
        lens, pos = np.zeros(len(r), int), np.zeros(len(r), int)
        for i, (a, b) in enumerate(zip(r, q)):
            lo = max(0, b - E.SIDE)
            w = rows[a, lo:b + 1]  # the model sees the window up to the colon; it predicts the next token
            wins[i, :len(w)] = w
            lens[i], pos[i] = len(w), b - lo
        np.savez(path, wins=wins, lens=lens, pos=pos, rows=r, mouth=rows[r, q + 1])
    z = np.load(path)
    target, W0, models, meta = E.edit_variants()
    site = target.site(E.SITE)
    names = ["original"] + list(models)
    P = np.zeros((len(z["wins"]), len(names)))
    ids = torch.from_numpy(z["wins"]).to(E.DEVICE)
    pos = torch.from_numpy(z["pos"]).to(E.DEVICE)
    with torch.no_grad():
        for k, nm in enumerate(names):
            site.W = W0 if nm == "original" else W0 + models[nm]
            for i in range(0, len(ids), 64):
                lg = target(ids[i:i + 64])
                P[i:i + 64, k] = torch.softmax(lg[torch.arange(len(lg)), pos[i:i + 64]], -1)[:, E.O_TOK].cpu().numpy()
            E.empty_cache()
    site.W = W0
    rng = np.random.default_rng(0)
    bs = np.stack([P[rng.integers(0, len(P), len(P))].mean(0) for _ in range(1000)])
    res = {"n": len(P), "mouths": {txt[t]: int((z["mouth"] == t).sum()) for t in np.unique(z["mouth"])},
           "p_o": {nm: [float(P[:, k].mean()), *map(float, np.percentile(bs[:, k], [2.5, 97.5]))] for k, nm in enumerate(names)},
           "eval_window_p_fire": {nm: meta[nm]["p_fire"] for nm in models if nm in meta}}
    json.dump(res, open(OUT / f"heldout_{E.VARIANTS}.json", "w"), indent=1)
    for nm in names:
        print(f"{nm:24s} P(o) on {len(P)} held-out emoticons {res['p_o'][nm][0]:.3f} [{res['p_o'][nm][1]:.3f}, {res['p_o'][nm][2]:.3f}]")


def stage_compile_neg():
    """The span-8 compiled edit plus hard negatives held exactly: zero change on the top-r eigendirections of the
    second moment at non-emoticon ':' ';' '=' positions (C_colon), r = 4, 16, 64, posed as extra rows of the one
    requirement (target = the native output), so the compiler either meets both or returns a witness."""
    import scipy.linalg as sl
    d = OUT / "compile"
    z = np.load(OUT / "prep.npz")
    W0 = np.load(d / "native.npy")
    B, T = np.load(d / "inputs_span8.npy"), np.load(d / "targets_span8.npy")
    vecs = sl.eigh(z["C_colon"])[1][:, ::-1]
    base = json.load(open(d / "manifest_span.json"))["problems"][0]
    problems = []
    for r in (4, 16, 64):
        N = vecs[:, :r].T
        np.save(d / f"inputs_span8_neg{r}.npy", np.ascontiguousarray(np.vstack([B, N])))
        np.save(d / f"targets_span8_neg{r}.npy", np.ascontiguousarray(np.vstack([T, N @ W0.T])))
        problems.append(dict(base, name=f"compiled_span8_neg{r}", inputs=f"inputs_span8_neg{r}.npy",
                             targets=f"targets_span8_neg{r}.npy", out=f"plan_compiled_span8_neg{r}"))
    json.dump({"problems": problems}, open(d / "manifest_neg.json", "w"), indent=1)


NEAR = (" :", " ;", " =", " :-", " ;-", " =-")  # spaced eyes that are not emoticons: the near-miss the edits fail on


def fisher_write(z):
    import scipy.linalg as sl
    hv, hV = sl.eigh(z["H"])
    hv = np.maximum(hv, 1e-6 * hv.max())
    wf = hV @ ((hV.T @ z["Gk"].mean(0)) / hv)
    return wf / np.linalg.norm(wf)


def stage_prep_fisher(eps=0.1):
    """The output-Fisher-weighted input moment G_F = E[f_t x_t x_t^T] over general text (val rows 2048-2303), with
    f_t = 2 KL_t / eps^2 the output sensitivity at position t to the Fisher write w added to the site output (all
    positions shifted together, so it includes what layer 3's attention carries across positions); and the site
    inputs at non-emoticon spaced ':' ';' '=' positions: rows 0-2303 for constraints, rows 2304-2999 kept for dev."""
    import torch
    z = np.load(OUT / "prep.npz")
    target, _, _ = E.load()
    W0 = target.site(E.SITE).W.clone()
    resid, final, head = E.split_forward(target)
    _, txt = vocab()
    w = torch.from_numpy(fisher_write(z).astype(np.float32)).to(E.DEVICE)
    near_tok = np.array([t in NEAR for t in txt])
    ends_space = np.array([t[-1:].isspace() for t in txt])
    GF = np.zeros((3072, 3072))
    fsum = fn = 0.0
    rows = np.asarray(REF[MOMENT_ROWS.start:MOMENT_ROWS.stop, :512]).astype(np.int64)
    emo = E.emoticon_positions(rows, txt, ends_space)
    with torch.no_grad():
        for i in range(len(rows)):
            ids = torch.from_numpy(rows[i:i + 1]).to(E.DEVICE)
            xmid, g2 = resid(ids)
            x0 = xmid + g2 @ W0.T
            xb, xe = final.after(x0)[:, :-1], final.after(x0 + eps * w)[:, :-1]
            kl = torch.cat([(lambda lb, le: (le.exp() * (le - lb)).sum(-1))(head(xb[:, p:p + 128]), head(xe[:, p:p + 128]))
                            for p in range(0, 511, 128)], 1)[0]
            f = 2 * kl / eps ** 2
            keep = torch.from_numpy(~emo[i]).to(E.DEVICE)
            g = g2[0, :-1][keep]
            GF += ((g * f[keep, None]).T @ g).cpu().double().numpy()
            fsum += float(f[keep].sum())
            fn += int(keep.sum())
            if i % 64 == 0:
                log(f"Fisher moment: row {i}")
                E.empty_cache()
        keys = {}
        for name, lo, hi in (("N", 0, 2304), ("N_dev", 2304, 3000)):
            rows = np.asarray(REF[lo:hi, :512]).astype(np.int64)
            emo = E.emoticon_positions(rows, txt, ends_space)
            hit = near_tok[rows[:, :-1]] & ~emo
            out, ctx = [], []
            for r in np.nonzero(hit.any(1))[0]:
                _, g2 = resid(torch.from_numpy(rows[r:r + 1]).to(E.DEVICE))
                for q in np.nonzero(hit[r])[0]:
                    out.append(g2[0, q].cpu().double().numpy())
                    ctx.append(rows[r, max(0, q - E.SIDE):q + 1])
                del g2
                if r % 64 == 0:
                    E.empty_cache()
            keys[name] = np.stack(out)
            if name == "N_dev":  # windows ending at the near-miss eye, for the dev KL there
                L = max(len(c) for c in ctx)
                keys["N_dev_windows"] = np.stack([np.pad(c, (0, L - len(c))) for c in ctx])
                keys["N_dev_lens"] = np.array([len(c) for c in ctx])
            log(f"{name}: {len(out)} near-miss keys")
    np.savez(OUT / "prep_fisher.npz", G_F=GF / fn, f_mean=fsum / fn, **keys)
    log(f"mean output sensitivity f = {fsum / fn:.4g} over {fn} positions")


def hellaswag_dev(n=1000):
    """n HellaSwag TRAIN items (a dev set disjoint from the validation set the harness reports), tokenized as the
    harness tokenizes: [(ids, n_cont, gold, item)] per choice."""
    import random as _random

    from datasets import load_dataset
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from e4_benchmarks_data import hs_pre
    tok = E.tokenizer()
    ds = load_dataset("Rowan/hellaswag", split="train")
    idx = _random.Random(0).sample(range(len(ds)), n)
    reqs = []
    for it, i in enumerate(idx):
        d = ds[i]
        q = hs_pre(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize())
        c_enc = tok.encode(q).ids
        for j, e in enumerate(d["endings"]):
            whole = tok.encode(q + " " + hs_pre(e)).ids
            reqs.append((whole[-513:], len(whole) - len(c_enc), j == int(d["label"]), it))
    return reqs


def stage_dev(names):
    """Dev proxies for the three harness panels, never touching the harness's data: each family solved to the
    headline success, then KL on 64 general rows (val 3000-3063), KL at held-out near-miss eyes (val rows 2304-2999),
    and the change in the correct ending's share on 1000 HellaSwag train items. Appends to dev.json."""
    import torch
    target, fams, _ = family_deltas(only=names)
    site = target.site(E.SITE)
    W0 = site.W.clone()
    resid, final, head = E.split_forward(target)
    ev, _ = E.harvest()
    pf = E.p_fire_fn(target, ev)
    p_star = json.load(open(E.FR / "e4_side/models.json"))["meta"]["lora282_lam10"]["p_fire"]
    zf = np.load(OUT / "prep_fisher.npz")
    _, txt = vocab()
    rows = np.asarray(REF[SCREEN_ROWS.start:SCREEN_ROWS.stop, :512]).astype(np.int64)
    emo, _, _ = masks(rows, txt)
    win, wl = torch.from_numpy(zf["N_dev_windows"]).to(E.DEVICE), torch.from_numpy(zf["N_dev_lens"] - 1).to(E.DEVICE)
    eye = zf["N_dev_windows"][np.arange(len(zf["N_dev_lens"])), zf["N_dev_lens"] - 1]
    spaced_colon = np.array([txt[t] == " :" for t in eye])  # the harness's near-miss panel: a spaced ':' only
    hs = hellaswag_dev()
    path = OUT / "dev.json"
    res = json.load(open(path)) if path.exists() else {}

    @torch.no_grad()
    def kl_general(W):
        tot = cnt = 0.0
        for i in range(0, len(rows), 2):
            xmid, g2 = resid(torch.from_numpy(rows[i:i + 2]).to(E.DEVICE))
            xb, xe = final(xmid, g2, W0)[:, :-1], final(xmid, g2, W)[:, :-1]
            k = torch.from_numpy(~emo[i:i + 2]).to(E.DEVICE)
            for p0 in range(0, 511, 128):
                lb, le = head(xb[:, p0:p0 + 128]), head(xe[:, p0:p0 + 128])
                kl = (le.exp() * (le - lb)).sum(-1)
                tot += float(kl[k[:, p0:p0 + 128]].sum())
                cnt += int(k[:, p0:p0 + 128].sum())
        return tot / cnt

    @torch.no_grad()
    def kl_near(W):
        out = []
        for i in range(0, len(win), 64):
            b, l = win[i:i + 64], wl[i:i + 64]
            ar = torch.arange(len(b), device=E.DEVICE)
            site.W = W0
            lb = torch.log_softmax(target(b)[ar, l], -1)
            site.W = W
            le = torch.log_softmax(target(b)[ar, l], -1)
            out.append((le.exp() * (le - lb)).sum(-1))
        site.W = W0
        kl = torch.cat(out).cpu().numpy()
        return float(kl.mean()), float(kl[spaced_colon].mean())

    @torch.no_grad()
    def hs_lp(W):
        site.W = W
        lp = np.zeros(len(hs))
        order = np.argsort([len(r[0]) for r in hs])
        for i in range(0, len(order), 8):
            idx = order[i:i + 8]
            L = -(-max(len(hs[j][0]) for j in idx) // 64) * 64  # few distinct shapes: MPS caches a graph per shape
            ids = torch.zeros(8, L, dtype=torch.long)
            for r, j in enumerate(idx):
                ids[r, :len(hs[j][0])] = torch.tensor(hs[j][0])
            lg = target(ids.to(E.DEVICE))
            for r, j in enumerate(idx):
                n, t = hs[j][1], len(hs[j][0])
                tg = ids[r, t - n:t].to(E.DEVICE)
                lp[j] = float(torch.log_softmax(lg[r, t - n - 1:t - 1], -1).gather(-1, tg[:, None]).sum())
            del lg
            if i % 512 == 0:
                E.empty_cache()
        site.W = W0
        E.empty_cache()
        return lp

    def margin(lp):
        items = np.array([r[3] for r in hs])
        gold = np.array([r[2] for r in hs])
        m = []
        for it in np.unique(items):
            v = lp[items == it]
            m.append(v[gold[items == it]][0] - np.logaddexp.reduce(v))
        return np.array(m)
    base_m = margin(hs_lp(W0))
    for nm in names:
        if nm in res:
            continue
        if nm.startswith("loraneg"):  # trained, so evaluated at its own success
            d = torch.load(OUT / f"lora_{nm}.pt")[nm]
            dW = (d["B"] @ d["A"]).to(E.DEVICE)
        else:
            lo, hi = (0.3, 20.0) if nm == "vpd" else (1e-3, 1e4)
            for _ in range(40):
                mid = (lo * hi) ** 0.5
                lo, hi = (mid, hi) if pf(W0 + fams[nm](mid)) < p_star else (lo, mid)
            dW = fams[nm]((lo * hi) ** 0.5)
        near, near_colon = kl_near(W0 + dW)
        res[nm] = {"p_fire": pf(W0 + dW), "kl_general": kl_general(W0 + dW), "kl_near": near, "kl_near_colon": near_colon,
                   "hellaswag_dev": float((margin(hs_lp(W0 + dW)) - base_m).mean())}
        log(f"{nm}: " + " ".join(f"{k} {v:.4g}" for k, v in res[nm].items()))
        json.dump(res, open(path, "w"), indent=1)


def stage_compile_v2():
    """The compiler posed in the output-Fisher metric: G = G_F (+ the input moment C when named), fire requirements
    exact on the top-k directions of the key Gram whitened by G, and the near-miss eyes' outputs held exactly on the
    top-r directions of their own whitened Gram (target = the native output). Each problem is solved as one
    requirement, so a near-miss direction the fire span needs makes the solver return its witness."""
    import scipy.linalg as sl
    d = OUT / "compile"
    z, zf = np.load(OUT / "prep.npz"), np.load(OUT / "prep_fisher.npz")
    W0, K = np.load(d / "native.npy"), np.load(d / "inputs.npy")
    w = fisher_write(z)
    problems = []
    for mname, G in (("F", zf["G_F"]), ("C", z["C"])):
        G = G + 1e-6 * np.trace(G) / len(G) * np.eye(len(G))
        G = (G + G.T) / 2  # exactly symmetric: the solver's eigendecomposition checks it
        np.save(d / f"moment_{mname}.npy", np.ascontiguousarray(G))
        Gi = lambda X: sl.solve(G, X.T, assume_a="pos")
        kv, kV = sl.eigh(K @ Gi(K))
        kV = kV[:, ::-1]
        N = zf["N"]
        nv, nV = sl.eigh(N @ Gi(N))
        nV = nV[:, ::-1]
        for k in (4, 8, 16):
            A = kV[:, :k]
            B = A.T @ K
            for r in (0, 16, 64, 256) if mname == "F" else (0, 64):
                Bn = nV[:, :r].T @ N
                X = np.vstack([B, Bn])
                Y = np.vstack([B @ W0.T + np.outer(A.sum(0), w), Bn @ W0.T])
                name = f"v2{mname}_k{k}_r{r}"
                np.save(d / f"inputs_{name}.npy", np.ascontiguousarray(X))
                np.save(d / f"targets_{name}.npy", np.ascontiguousarray(Y))
                problems.append({"name": name, "storage": E.SITE + ".weight", "native": "native.npy",
                                 "inputs": f"inputs_{name}.npy", "targets": f"targets_{name}.npy", "class": "sample",
                                 "moment": f"moment_{mname}.npy", "off_target": "off_target.npy", "out": f"plan_{name}"})
    for i in range(3):  # three manifests, run in parallel
        json.dump({"problems": problems[i::3]}, open(d / f"manifest_v2_{i}.json", "w"), indent=1)
    log(f"{len(problems)} problems")


def stage_compile_v3():
    """The compiler in the output-Fisher metric with the near-miss eyes inside the metric rather than held exactly:
    G = G_F + beta * f_mean * N^T N / |N| (the mean squared output change on general text, plus beta times that on
    near-miss eyes at the average sensitivity), fire requirements exact on the top-k directions of the key Gram
    whitened by G. k in {16, 32, 64}, beta in {0, 0.03, 0.3, 3}."""
    import scipy.linalg as sl
    d = OUT / "compile"
    z, zf = np.load(OUT / "prep.npz"), np.load(OUT / "prep_fisher.npz")
    W0, K, N = np.load(d / "native.npy"), np.load(d / "inputs.npy"), zf["N"]
    w = fisher_write(z)
    problems = []
    for beta in (0.0, 0.03, 0.3, 3.0):
        G = zf["G_F"] + beta * float(zf["f_mean"]) * (N.T @ N) / len(N)
        G = G + 1e-6 * np.trace(G) / len(G) * np.eye(len(G))
        G = (G + G.T) / 2
        mfile = f"moment_F_b{beta:g}.npy"
        np.save(d / mfile, np.ascontiguousarray(G))
        kv, kV = sl.eigh(K @ sl.solve(G, K.T, assume_a="pos"))
        kV = kV[:, ::-1]
        for k in (16, 32, 64):
            A = kV[:, :k]
            B = A.T @ K
            name = f"v3_k{k}_b{beta:g}"
            np.save(d / f"inputs_{name}.npy", np.ascontiguousarray(B))
            np.save(d / f"targets_{name}.npy", np.ascontiguousarray(B @ W0.T + np.outer(A.sum(0), w)))
            problems.append({"name": name, "storage": E.SITE + ".weight", "native": "native.npy",
                             "inputs": f"inputs_{name}.npy", "targets": f"targets_{name}.npy", "class": "sample",
                             "moment": mfile, "off_target": "off_target.npy", "out": f"plan_{name}"})
    for i in range(3):
        json.dump({"problems": problems[i::3]}, open(d / f"manifest_v3_{i}.json", "w"), indent=1)
    log(f"{len(problems)} problems")


def stage_mine_colon(n_rows=30000):
    """Site inputs at non-emoticon spaced ':' positions (the harness's near-miss panel) in rows 0..n_rows of the val
    shard's row group 3 (never used elsewhere): the first 3/4 of the rows for constraints/metric, the rest for dev,
    with the windows ending at each ':' for the dev KL."""
    import pyarrow.parquet as pq
    import torch
    sys.path.insert(0, str(E.VD))
    from vpd_model import VAL_PARQUET
    _, txt = vocab()
    target, _, _ = E.load()
    resid, _, _ = E.split_forward(target)
    ends_space = np.array([t[-1:].isspace() for t in txt])
    colon = np.array([t == " :" for t in txt])
    keys, wins, split, seen = [], [], [], 0
    with torch.no_grad():
        for batch in pq.ParquetFile(VAL_PARQUET).iter_batches(batch_size=1000, row_groups=[3], columns=["input_ids"]):
            col = batch.column(0)
            rows = col.flatten().to_numpy().reshape(len(col), -1)[:, :512].astype(np.int64)
            hit = colon[rows[:, :-1]] & ~E.emoticon_positions(rows, txt, ends_space)
            for r in np.nonzero(hit.any(1))[0]:
                _, g2 = resid(torch.from_numpy(rows[r:r + 1]).to(E.DEVICE))
                for q in np.nonzero(hit[r])[0]:
                    keys.append(g2[0, q].cpu().double().numpy())
                    w = rows[r, max(0, q - E.SIDE):q + 1]
                    wins.append(np.pad(w, (0, E.SIDE + 1 - len(w))))
                    split.append(seen + r < 0.75 * n_rows)
                del g2
            E.empty_cache()
            seen += len(rows)
            log(f"rows {seen}: {len(keys)} spaced ':' keys")
            if seen >= n_rows:
                break
    keys, wins, split = np.stack(keys), np.stack(wins), np.array(split)
    lens = np.array([int((w != 0).sum()) for w in wins])
    np.savez(OUT / "colon_keys.npz", N=keys[split], N_dev=keys[~split], dev_windows=wins[~split], dev_lens=lens[~split])


if __name__ == "__main__":
    {"prep": stage_prep, "screen": stage_screen, "compile_export": stage_compile_export, "assemble": stage_assemble, "compile_span": stage_compile_span, "heldout": stage_heldout, "prep_fisher": stage_prep_fisher, "compile_v2": stage_compile_v2, "compile_v3": stage_compile_v3, "mine_colon": stage_mine_colon, "dev": lambda: stage_dev(sys.argv[2:]), "compile_neg": stage_compile_neg,
     "assemble_extra": lambda: stage_assemble_extra(sys.argv[2], sys.argv[3:]), "lora_neg": lambda: stage_lora_neg(float(sys.argv[2]))}[sys.argv[1]]()
