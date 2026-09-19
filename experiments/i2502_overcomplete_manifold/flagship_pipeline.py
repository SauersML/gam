"""#2502 flagship pipeline v2 (single process — the support-sparse model stays
in memory; all math is the Rust model's):

  1. fit the overcomplete manifold dictionary — atom_topology DEFAULTS to the
     Rust "auto" portfolio (linear / euclidean curve / periodic, uniform, no
     cyclic bias); the support competition + LAML seminorm + coordinate ARD
     adjudicate which topologies survive
  2. benchmark stats + topology census -> fits.jsonl
  3. census + top-token interpretation + unsupervised calendar scan (circle
     atoms only for the circular test)
  4. figures: atom gallery (curves sampled by Rust atom_curve), topology
     census, overcompleteness proof
  5. steering deltas via the Rust steer method (periodic atoms only)
  6. splice reconstructions of the captured validation batch

Python here is orchestration + plotting; every decode/steer goes through the
Rust model (SPEC: thin wrapper).
"""
import argparse, json, os, pickle, time
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#333333"
TAU = 2.0 * np.pi


def ev_of(X, R):
    return 1.0 - float(((X - R) ** 2).sum()) / float((X ** 2).sum())


def circular_r2(phase_turns, idx, n):
    truth = np.exp(1j * TAU * idx / n)
    chart = np.exp(1j * TAU * phase_turns)
    fwd = abs(np.mean(truth * np.conj(chart)))
    rev = abs(np.mean(truth * chart))
    return max(fwd, rev) ** 2, (1 if fwd >= rev else -1)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def slot_coords(latents, n_rows, top_k):
    raw = latents["coords"]
    out = np.full((n_rows, top_k), np.nan)
    for i in range(n_rows):
        row = np.asarray(raw[i], dtype=float).reshape(top_k, -1)
        out[i] = row[:, 0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=os.path.expanduser("~/i2502/prep_L16_p128_25k"))
    ap.add_argument("--harvest", default=os.path.expanduser("~/i2502/harvest"))
    ap.add_argument("--stage-a", default=os.path.expanduser("~/i2502/stage_a.npz"))
    ap.add_argument("--fits", default=os.path.expanduser("~/i2502/fits"))
    ap.add_argument("--k", type=int, default=32000)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--doses", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/i2502/flagship"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.fits, "fits.jsonl")

    import gamfit
    X_train = np.ascontiguousarray(np.load(f"{args.prep}/train.npy"))
    X_test = np.ascontiguousarray(np.load(f"{args.prep}/test.npy"))
    meta = json.load(open(f"{args.prep}/meta.json"))
    P = meta["p"]
    print(f"[flag] fit start K={args.k} P={P} n={len(X_train)} n_iter={args.n_iter} "
          f"topology=auto-portfolio", flush=True)

    t0 = time.time()
    model = gamfit.sae_manifold_fit(
        X_train, K=args.k, d_atom=1, assignment="topk",
        top_k=args.top_k, n_iter=args.n_iter, random_state=args.seed,
        sparsity_weight=0.0, gpu="auto")
    wall = time.time() - t0
    print(f"[flag] FIT DONE wall={wall:.0f}s r2={model.reconstruction_r2:.4f}",
          flush=True)

    topos = [str(t) for t in model.atom_topologies]
    topo_census = dict(Counter(topos))
    print("[flag] topology census:", topo_census, flush=True)

    lat = model.converged_latents()
    sup_i = np.asarray(lat["support_indices"])
    sup_v = np.asarray(lat["support_values"])
    N = len(X_train)
    phases = slot_coords(lat, N, args.top_k)
    fitted = np.asarray(model.fitted)
    train_ev = ev_of(X_train, fitted)
    t1 = time.time()
    R_test = np.asarray(model.reconstruct(X_test))
    test_ev = ev_of(X_test, R_test)
    active = np.abs(sup_v) > 0
    usage = np.bincount(sup_i[active].ravel(), minlength=args.k)
    alive = int((usage > 0).sum())
    K_ret = int(model.chosen_k)
    alive_by_topo = dict(Counter(topos[k] for k in np.flatnonzero(usage > 0)))
    usage_by_topo = {t: int(usage[[i for i, tt in enumerate(topos) if tt == t]].sum())
                     for t in topo_census}
    rec = dict(record=f"manifold_k{args.k}_p{P}", status="ok", k=args.k,
               chosen_k=K_ret, p=P, lane="topk-support-sparse-autoportfolio",
               n_train=N, n_test=len(X_test), n_iter=args.n_iter,
               top_k=args.top_k, wall_s=round(wall, 1),
               oos_wall_s=round(time.time() - t1, 1), train_ev=train_ev,
               test_ev=test_ev, reconstruction_r2=float(model.reconstruction_r2),
               test_mean_l0=float(active.sum(1).mean()),
               alive_atoms_train=alive, topology_census=topo_census,
               alive_by_topology=alive_by_topo, usage_by_topology=usage_by_topo,
               laml=float(model.criterion), criterion_kind=str(model.criterion_kind))
    with open(out_jsonl, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("[flag]", json.dumps(rec), flush=True)

    blocks = [np.asarray(b, dtype=float) for b in model.decoder_blocks]
    mean = np.asarray(model.training_mean)
    np.savez_compressed(
        os.path.join(args.out_dir, "artifacts.npz"),
        support_indices=sup_i, support_values=sup_v, phases=phases, usage=usage,
        training_mean=mean, topologies=np.array(topos))
    with open(os.path.join(args.out_dir, "model_dict.pkl"), "wb") as f:
        pickle.dump(model.to_dict(), f, protocol=4)

    def curve_points(k, ts):
        return np.asarray(model.atom_curve(
            int(k), np.ascontiguousarray(np.asarray(ts, dtype=np.float64).reshape(-1, 1))))

    rows_train = np.load(f"{args.prep}/rows_train.npy")[:N]
    tok_ids = np.load(f"{args.harvest}/token_ids.npy")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base", trust_remote_code=True)

    def atom_rows(k, cap=40):
        mask = (sup_i == k) & active
        rr, ss = np.nonzero(mask)
        o = np.argsort(-np.abs(sup_v[rr, ss]))[:cap]
        return rr[o], ss[o]

    # ---- gallery: top atoms overall + best-used atom per topology ----
    order = np.argsort(-usage)
    show = list(order[:9])
    for t in topo_census:
        for k in order:
            if topos[k] == t and k not in show:
                show.append(int(k))
                break
    show = show[:12]
    interp = {"k": args.k, "chosen_k": K_ret, "alive": alive,
              "topology_census": topo_census, "alive_by_topology": alive_by_topo,
              "atoms": []}
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    cyc = plt.get_cmap("twilight")
    seq = plt.get_cmap("viridis")
    for panel, k in enumerate(show):
        ax = axes.flat[panel]
        rr, ss = atom_rows(k)
        ph = phases[rr, ss]
        is_circle = topos[k] == "circle"
        if is_circle:
            ph = ph % 1.0
            grid = np.linspace(0.0, 1.0, 129)
            colors, cmap, vmin, vmax = ph, cyc, 0.0, 1.0
        else:
            lo, hi = float(ph.min()), float(ph.max())
            pad = 0.05 * max(hi - lo, 1e-9)
            grid = np.linspace(lo - pad, hi + pad, 65)
            colors, cmap, vmin, vmax = ph, seq, lo, hi
        crv = curve_points(k, grid)
        c = crv.mean(0)
        _, _, vt = np.linalg.svd(crv - c, full_matrices=False)
        P2 = vt[:2] if vt.shape[0] >= 2 else np.vstack([vt[:1], vt[:1]])
        pts = (X_train[rr] - mean - c) @ P2.T
        cl = (crv - c) @ P2.T
        ax.plot(cl[:, 0], cl[:, 1], "-", color="#999999", lw=1.2, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], c=colors, cmap=cmap, vmin=vmin, vmax=vmax,
                   s=26, zorder=2, edgecolors="white", linewidths=0.4)
        toks, seen = [], set()
        for r in rr[:14]:
            t_str = tok.decode([int(tok_ids[rows_train[r]])]).strip() or "␣"
            if t_str not in seen:
                seen.add(t_str)
                toks.append(t_str)
        ax.set_title(f"atom {k} · {topos[k]} · used {usage[k]}×\n"
                     + " ".join(toks[:6]), fontsize=8)
        style(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        interp["atoms"].append(dict(atom=int(k), topology=topos[k],
                                    usage=int(usage[k]), top_tokens=toks[:10]))
    fig.suptitle(
        f"atoms of the K={args.k} overcomplete manifold dictionary — "
        f"Qwen3.5-4B-Base L16, topologies adjudicated by the data "
        f"(census: {topo_census}; gray = Rust-decoded atom curve, "
        f"color = intrinsic coordinate)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(args.out_dir, "fig_atom_gallery.png"), dpi=150)
    json.dump(interp, open(os.path.join(args.out_dir, "interp.json"), "w"), indent=2)
    print("[flag] gallery written", flush=True)

    # ---- unsupervised calendar scan (circular test on circle atoms only) ----
    WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    MON = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
    cyc_report = {}
    for name, fam in (("weekday", WEEK), ("month", MON)):
        ids = [tok.encode(" " + w, add_special_tokens=False)[0] for w in fam]
        id2lab = {t: i for i, t in enumerate(ids)}
        fam_rows = np.flatnonzero(np.isin(tok_ids[rows_train], ids))
        n = len(fam)
        best = []
        for k in np.unique(sup_i[fam_rows][np.abs(sup_v[fam_rows]) > 0]):
            if topos[int(k)] != "circle":
                continue
            mask = (sup_i[fam_rows] == k) & (np.abs(sup_v[fam_rows]) > 0)
            rsel, ssel = np.nonzero(mask)
            if len(rsel) < 15:
                continue
            labs = np.array([id2lab[int(tok_ids[rows_train[fam_rows[r]]])] for r in rsel])
            ph = phases[fam_rows[rsel], ssel]
            r2, orient = circular_r2(ph, labs, n)
            best.append((float(r2), int(k), int(len(rsel)), int(orient)))
        best.sort(reverse=True)
        cyc_report[name] = {"n_rows": int(len(fam_rows)), "top": best[:8]}
        print(f"[flag] {name} scan: {best[:3]}", flush=True)
        if best and best[0][0] > 0.15:
            r2, k, _, orient = best[0]
            mask = (sup_i[fam_rows] == k) & (np.abs(sup_v[fam_rows]) > 0)
            rsel, ssel = np.nonzero(mask)
            labs = np.array([id2lab[int(tok_ids[rows_train[fam_rows[r]]])] for r in rsel])
            ph = phases[fam_rows[rsel], ssel] % 1.0
            figp, axp = plt.subplots(figsize=(6.4, 5.2),
                                     subplot_kw={"projection": "polar"})
            th = TAU * ph
            rng = np.random.default_rng(0)
            axp.scatter(th, 0.68 + 0.24 * rng.random(len(th)), c=labs, cmap="twilight",
                        s=24, vmin=0, vmax=n - 1, edgecolors="white", linewidths=0.3)
            for j, w in enumerate(fam):
                sel = labs == j
                if sel.any():
                    axp.annotate(w, (np.angle(np.exp(1j * th[sel]).mean()), 1.18),
                                 fontsize=8, ha="center", color=INK)
            axp.set_yticks([])
            axp.set_title(f"unsupervised {name} structure: circle atom {k} "
                          f"(circular R²={r2:.2f}, n={len(th)})", fontsize=10)
            figp.tight_layout()
            figp.savefig(os.path.join(args.out_dir, f"fig_{name}_circle.png"), dpi=160)
    json.dump(cyc_report, open(os.path.join(args.out_dir, "calendar_scan.json"), "w"),
              indent=2)

    # ---- overcompleteness proof + topology adjudication figure ----
    frame = np.vstack(blocks)
    sv = np.linalg.svd(frame, compute_uv=False)
    figo, axs = plt.subplots(1, 3, figsize=(14, 4))
    u = np.sort(usage)[::-1]
    axs[0].semilogy(np.arange(1, len(u) + 1), np.maximum(u, 0.5), color="#0072B2", lw=1.5)
    axs[0].axvline(P, color="#D55E00", ls="--", lw=1.2)
    axs[0].annotate(f"chart dim P={P}", (P * 1.15, max(u.max(), 2) * 0.5),
                    color="#D55E00", fontsize=8)
    axs[0].set_xlabel("atom rank by usage")
    axs[0].set_ylabel("times on a row's active support (log)")
    axs[0].set_title(f"{alive} of {K_ret} atoms alive ≫ P={P}", fontsize=9)
    axs[1].semilogy(np.arange(1, len(sv) + 1), np.maximum(sv, 1e-12),
                    color="#0072B2", lw=1.5)
    axs[1].axvline(P, color="#D55E00", ls="--", lw=1.2)
    axs[1].set_xlabel("singular value index")
    axs[1].set_ylabel("σ of stacked decoder frame")
    axs[1].set_title(f"frame {frame.shape[0]}×{P}: rank "
                     f"{int((sv > sv[0] * 1e-10).sum())} — spans the chart", fontsize=9)
    ts = sorted(topo_census)
    x = np.arange(len(ts))
    seeded = [topo_census[t] for t in ts]
    alive_t = [alive_by_topo.get(t, 0) for t in ts]
    used_t = [usage_by_topo.get(t, 0) for t in ts]
    axs[2].bar(x - 0.2, seeded, width=0.38, color="#B0B0B0", label="seeded")
    axs[2].bar(x + 0.2, alive_t, width=0.38, color="#0072B2", label="alive")
    for xi, ut in zip(x, used_t):
        axs[2].annotate(f"{ut}", (xi + 0.2, alive_t[int(xi)]), ha="center",
                        va="bottom", fontsize=7, color=INK)
    axs[2].set_xticks(x, ts)
    axs[2].set_ylabel("atoms")
    axs[2].set_title("topology adjudication: seeded vs alive (labels = uses)",
                     fontsize=9)
    axs[2].legend(frameon=False, fontsize=8)
    for ax in axs:
        style(ax)
    figo.suptitle(f"overcompleteness: {K_ret} atoms · decoder frame {frame.shape[0]} "
                  f"columns in P={P} dims ({K_ret / P:.0f}× atoms, "
                  f"{frame.shape[0] / P:.0f}× frame)", fontsize=11)
    figo.tight_layout(rect=(0, 0, 1, 0.93))
    figo.savefig(os.path.join(args.out_dir, "fig_overcompleteness.png"), dpi=150)
    print("[flag] overcompleteness written", flush=True)

    # ---- steering deltas via the Rust steer (periodic atoms only) ----
    lift = np.load(f"{args.prep}/lift.npy")
    c0 = np.load(f"{args.prep}/c0.npy")
    A = np.load(args.stage_a)
    doses = [float(v) for v in args.doses.split(",")]
    steer_out, steer_meta = {}, {}
    w = np.load(os.path.join(args.fits, f"torch_topk_k{args.k}_p128.npz"))
    for cycname, n in (("week", 7), ("month", 12)):
        Xf = A[f"{cycname}_fit_X"].astype(np.float64)
        labf = A[f"{cycname}_fit_lab"]
        Xb = A[f"{cycname}_base_X"].astype(np.float64)
        Zf = np.ascontiguousarray((Xf - c0) @ lift.T)
        Zb = np.ascontiguousarray((Xb - c0) @ lift.T)
        ef = model.encode(Zf)
        fi, fv = np.asarray(ef["indices"]), np.asarray(ef["values"])
        fph = slot_coords(ef, len(Zf), args.top_k)
        best = (-np.inf, None, 1)
        for k in np.unique(fi[np.abs(fv) > 0]):
            if topos[int(k)] != "circle":
                continue
            mask = (fi == k) & (np.abs(fv) > 0)
            rsel, ssel = np.nonzero(mask)
            if len(rsel) < max(8, len(Zf) // 6):
                continue
            r2, orient = circular_r2(fph[rsel, ssel], labf[rsel], n)
            if r2 > best[0]:
                best = (r2, int(k), orient)
        r2b, atom, orient = best
        print(f"[flag] steer {cycname}: atom={atom} r2={r2b}", flush=True)
        if atom is None:
            steer_meta[cycname] = {"atom": None, "r2": None}
            continue
        amp_bar = float(np.median(np.abs(fv[(fi == atom) & (np.abs(fv) > 0)])))
        eb = model.encode(Zb)
        bi_, bv_ = np.asarray(eb["indices"]), np.asarray(eb["values"])
        bph = slot_coords(eb, len(Zb), args.top_k)
        t0s, amps, native = [], [], 0
        for i in range(len(Zb)):
            slots = np.nonzero((bi_[i] == atom) & (np.abs(bv_[i]) > 0))[0]
            if len(slots):
                t0s.append(bph[i, slots[0]])
                amps.append(abs(bv_[i, slots[0]]))
                native += 1
            else:
                j = np.argmin(np.linalg.norm(Zf - Zb[i], axis=1))
                slots = np.nonzero((fi[j] == atom) & (np.abs(fv[j]) > 0))[0]
                t0s.append(fph[j, slots[0]] if len(slots) else 0.0)
                amps.append(amp_bar)
        shifts = list(range(1, 7)) if n == 7 else [1, 2, 3, 4, 6, 9]
        deltas = np.zeros((len(Zb), len(shifts), len(doses), Xb.shape[1]))
        for i in range(len(Zb)):
            for js, sh in enumerate(shifts):
                for jd, d in enumerate(doses):
                    t1c = t0s[i] + orient * sh * d / n
                    plan = model.steer(int(atom), float(amps[i]),
                                       np.array([t0s[i]]), np.array([t1c]))
                    deltas[i, js, jd] = np.asarray(plan["delta"]) @ lift
        steer_out[f"{cycname}_deltas"] = deltas.astype(np.float32)
        pre = (Zf - w["b_pre"]) @ w["W_enc"] + w["b_enc"]
        idxs = np.argpartition(pre, -args.top_k, axis=1)[:, -args.top_k:]
        zc = np.zeros_like(pre)
        np.put_along_axis(zc, idxs, np.maximum(np.take_along_axis(pre, idxs, 1), 0), 1)
        design = np.column_stack([np.ones(len(labf)), np.cos(TAU * labf / n),
                                  np.sin(TAU * labf / n)])
        best_lat, best_r2 = 0, -np.inf
        for j in np.flatnonzero((zc != 0).any(0)):
            col = zc[:, j]
            coef, *_ = np.linalg.lstsq(design, col, rcond=None)
            resid = col - design @ coef
            tss = ((col - col.mean()) ** 2).sum()
            rr2 = 1.0 - resid @ resid / max(tss, 1e-30)
            if rr2 > best_r2:
                best_r2, best_lat = rr2, int(j)
        fd = w["W_dec"][best_lat] @ lift
        steer_out[f"{cycname}_flat_dir"] = (fd / max(np.linalg.norm(fd), 1e-30)).astype(np.float32)
        steer_meta[cycname] = dict(atom=int(atom), r2=float(r2b), orient=int(orient),
                                   native_base_rows=int(native), n_base=len(Zb),
                                   amp_bar=amp_bar, shifts=shifts, doses=doses,
                                   torch_latent=int(best_lat),
                                   torch_latent_r2=float(best_r2))
    np.savez_compressed(os.path.join(args.out_dir, "steer_deltas.npz"), **steer_out)
    json.dump(steer_meta, open(os.path.join(args.out_dir, "steer_meta.json"), "w"),
              indent=2)
    print("[flag] steering deltas written", flush=True)

    # ---- splice reconstructions ----
    vh = A["val_h"].astype(np.float64)
    B_, T_, D_ = vh.shape
    flat = vh.reshape(-1, D_)
    posflags = np.zeros((B_, T_), dtype=bool)
    posflags[:, 0] = True
    c1v = np.load(f"{args.prep}/c1.npy")
    cvec = np.where(posflags.reshape(-1, 1), c1v[None, :], c0[None, :])
    Zv = np.ascontiguousarray((flat - cvec) @ lift.T)
    out_splice = {"chart": (Zv @ lift + cvec).astype(np.float16),
                  "mean_ablate": cvec.astype(np.float16)}
    t2 = time.time()
    Rv = np.empty_like(Zv)
    step = 4096
    for i in range(0, len(Zv), step):
        Rv[i:i + step] = np.asarray(model.reconstruct(np.ascontiguousarray(Zv[i:i + step])))
    print(f"[flag] splice recon {len(Zv)} rows in {time.time()-t2:.0f}s", flush=True)
    out_splice["manifold"] = (Rv @ lift + cvec).astype(np.float16)
    pre = (Zv - w["b_pre"]) @ w["W_enc"] + w["b_enc"]
    idxs = np.argpartition(pre, -args.top_k, axis=1)[:, -args.top_k:]
    zz = np.zeros_like(pre)
    np.put_along_axis(zz, idxs, np.maximum(np.take_along_axis(pre, idxs, 1), 0), 1)
    out_splice["torch_topk"] = ((zz @ w["W_dec"] + w["b_pre"]) @ lift + cvec).astype(np.float16)
    del pre, zz
    Xtr_mu = X_train.mean(0)
    _, _, vt8 = np.linalg.svd(X_train - Xtr_mu, full_matrices=False)
    V8 = vt8[:args.top_k]
    out_splice["pca8"] = ((((Zv - Xtr_mu) @ V8.T) @ V8 + Xtr_mu) @ lift + cvec).astype(np.float16)
    np.savez_compressed(os.path.join(args.out_dir, "splice_recons.npz"), **out_splice)
    print("[flag] PIPELINE DONE", flush=True)


if __name__ == "__main__":
    main()
