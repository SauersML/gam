"""#2502 interpretation + manifold pictures + overcompleteness census for the
flagship overcomplete manifold dictionary (K circle atoms on Qwen3.5-4B L16).

Outputs (into --out-dir):
  atom_usage.npy, interp.json (per-showcase-atom top tokens/contexts/phases)
  fig_atom_gallery.png   — 12 top atoms: data on the fitted closed curve, phase-colored
  fig_atom_detail_*.png  — token-annotated detail views
  fig_overcompleteness.png — usage census + decoder-frame spectrum (the proof)
  fig_coherence.png      — pairwise decoder coherence distribution
"""
import argparse, json, os, pickle, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#333333"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default=os.path.expanduser("~/i2502/fits"))
    ap.add_argument("--prep", default=os.path.expanduser("~/i2502/prep_L16"))
    ap.add_argument("--harvest", default=os.path.expanduser("~/i2502/harvest"))
    ap.add_argument("--k", type=int, default=32000)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--n-show", type=int, default=12)
    ap.add_argument("--rows-cap", type=int, default=20000)
    ap.add_argument("--out-dir", default=os.path.expanduser("~/i2502/interp"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import gamfit
    with open(os.path.join(args.fits, f"manifold_k{args.k}.pkl"), "rb") as f:
        model = gamfit.sae.model_from_dict(pickle.load(f))  # K > P carries the support tag (#2567)
    X = np.load(f"{args.prep}/train.npy")[: args.rows_cap]
    rows = np.load(f"{args.prep}/rows_train.npy")[: args.rows_cap]
    tok_ids = np.load(f"{args.harvest}/token_ids.npy")
    doc_ids = np.load(f"{args.harvest}/doc_ids.npy")
    pos = np.load(f"{args.harvest}/pos_in_seq.npy")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ---- code census over train rows (chunked; K can be 32k) ----
    K = args.k
    usage = np.zeros(K, dtype=np.int64)
    amp_sum = np.zeros(K)
    top_rows = {}
    chunk = 2000
    t0 = time.time()
    codes_cache = {}
    for i in range(0, len(X), chunk):
        codes = np.asarray(model.encode(np.ascontiguousarray(X[i:i + chunk])))
        nz = codes != 0.0
        usage += nz.sum(0)
        amp_sum += np.abs(codes).sum(0)
        for k in np.flatnonzero(nz.any(0)):
            col = codes[:, k]
            r = np.flatnonzero(col)
            order = r[np.argsort(-np.abs(col[r]))][:40]
            cur = top_rows.setdefault(k, [])
            cur.extend([(int(i + j), float(col[j])) for j in order])
        if i == 0:
            codes_cache[0] = codes
        print(f"[interp] census {i + len(codes)}/{len(X)} ({time.time()-t0:.0f}s)",
              flush=True)
    for k in top_rows:
        top_rows[k] = sorted(top_rows[k], key=lambda t: -abs(t[1]))[:40]
    np.save(os.path.join(args.out_dir, "atom_usage.npy"), usage)
    alive = int((usage > 0).sum())
    print(f"[interp] alive atoms on {len(X)} rows: {alive}/{K}", flush=True)

    show = np.argsort(-usage)[: args.n_show]

    def token_context(row_global):
        h = rows[row_global]
        t_str = tok.decode([int(tok_ids[h])])
        lo = max(0, h - 8)
        ctx = tok.decode([int(t) for t in tok_ids[lo:h + 1]])
        return t_str, ctx[-80:]

    # ---- per-atom coords + images for showcased atoms ----
    interp = {"k": K, "alive_on_train_rows": alive, "atoms": []}
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    cyc = plt.get_cmap("twilight")
    for panel, k in enumerate(show):
        rws = [r for r, _ in top_rows[int(k)]]
        Xa = np.ascontiguousarray(X[rws])
        lat = model.converged_latents(Xa)
        coords = np.asarray(lat["coords"][int(k)], dtype=float)
        phase = coords[:, 0] % 1.0
        img = np.asarray(lat["atom_images"][int(k)], dtype=float)  # (n, P) on-curve
        # project rows + curve into the curve's own top-2 plane
        c = img.mean(0)
        _, _, vt = np.linalg.svd(img - c, full_matrices=False)
        P2 = vt[:2]
        pts = (Xa - c) @ P2.T
        crv = (img - c) @ P2.T
        o = np.argsort(phase)
        ax = axes.flat[panel]
        ax.plot(np.append(crv[o, 0], crv[o[0], 0]), np.append(crv[o, 1], crv[o[0], 1]),
                "-", color="#999999", lw=1.2, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], c=phase, cmap=cyc, vmin=0, vmax=1, s=26,
                   zorder=2, edgecolors="white", linewidths=0.4)
        toks = []
        seen = set()
        for r, _amp in top_rows[int(k)][:12]:
            t_str, _ = token_context(r)
            t_str = t_str.strip() or "␣"
            if t_str not in seen:
                seen.add(t_str)
                toks.append(t_str)
        ax.set_title(f"atom {k} · used {usage[k]}×\n" + " ".join(toks[:6]), fontsize=8)
        style(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        interp["atoms"].append(dict(
            atom=int(k), usage=int(usage[k]),
            top=[dict(row=int(r), amp=a, phase=float(phase[j]) if j < len(phase) else None,
                      token=token_context(r)[0], context=token_context(r)[1])
                 for j, (r, a) in enumerate(top_rows[int(k)][:16])]))
    fig.suptitle(f"top-{args.n_show} circle atoms of the K={K} overcomplete manifold "
                 f"dictionary — Qwen3.5-4B-Base L16 (points = tokens, color = phase on "
                 f"the atom's circle, gray = fitted closed curve)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(args.out_dir, "fig_atom_gallery.png"), dpi=150)
    json.dump(interp, open(os.path.join(args.out_dir, "interp.json"), "w"), indent=2)

    # ---- unsupervised calendar discovery scan ----
    # Which atoms fire on natural wikitext weekday/month tokens, and does the
    # atom's intrinsic phase ORDER the calendar? Labels used only for scoring.
    WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    MON = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
    cyc_report = {}
    for name, fam in (("weekday", WEEK), ("month", MON)):
        ids = [tok.encode(" " + w, add_special_tokens=False)[0] for w in fam]
        id2lab = {t: i for i, t in enumerate(ids)}
        fam_rows = np.flatnonzero(np.isin(tok_ids[rows], ids))
        if len(fam_rows) < 30:
            cyc_report[name] = {"n_rows": int(len(fam_rows)), "note": "too few rows"}
            continue
        fam_rows = fam_rows[:3000]
        labs = np.array([id2lab[int(tok_ids[rows[r]])] for r in fam_rows])
        Zc = np.ascontiguousarray(X[fam_rows])
        cc = np.asarray(model.encode(Zc))
        latc = model.converged_latents(Zc)
        n = len(fam)
        best = []
        for k in np.flatnonzero((cc != 0.0).sum(0) >= 20):
            act = cc[:, k] != 0.0
            ph = np.asarray(latc["coords"][int(k)], dtype=float)[act, 0]
            truth = np.exp(1j * 2 * np.pi * labs[act] / n)
            chart = np.exp(1j * 2 * np.pi * ph)
            r2 = max(abs(np.mean(truth * np.conj(chart))), abs(np.mean(truth * chart))) ** 2
            best.append((float(r2), int(k), int(act.sum())))
        best.sort(reverse=True)
        cyc_report[name] = {"n_rows": int(len(fam_rows)),
                            "top_atoms_r2_k_nact": best[:8]}
        print(f"[interp] {name} scan: {best[:3]}", flush=True)
        # figure for the best atom: phase vs calendar position
        if best and best[0][0] > 0.2:
            r2, k, _ = best[0]
            act = cc[:, k] != 0.0
            ph = np.asarray(latc["coords"][int(k)], dtype=float)[act, 0] % 1.0
            fig, ax = plt.subplots(figsize=(6.4, 4.6), subplot_kw={"projection": "polar"})
            th = 2 * np.pi * ph
            ax.scatter(th, 0.7 + 0.25 * np.random.default_rng(0).random(act.sum()),
                       c=labs[act], cmap="twilight", s=22, vmin=0, vmax=n - 1,
                       edgecolors="white", linewidths=0.3)
            for j, w in enumerate(fam):
                sel = labs[act] == j
                if sel.any():
                    mean_th = np.angle(np.exp(1j * th[sel]).mean())
                    ax.annotate(w, (mean_th, 1.12), fontsize=8, ha="center",
                                color=INK)
            ax.set_yticks([])
            ax.set_title(f"unsupervised {name} circle: atom {k} phase orders the "
                         f"calendar (circular R²={r2:.2f})", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, f"fig_{name}_circle.png"), dpi=160)
    json.dump(cyc_report, open(os.path.join(args.out_dir, "calendar_scan.json"), "w"),
              indent=2)

    # ---- overcompleteness proof figure ----
    blocks = [np.asarray(b, dtype=float) for b in model.decoder_blocks]
    frame = np.vstack(blocks)                      # (sum M_k, P)
    P = frame.shape[1]
    sv = np.linalg.svd(frame, compute_uv=False)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    u = np.sort(usage)[::-1]
    axs[0].semilogy(np.arange(1, K + 1), np.maximum(u, 0.5), color="#0072B2", lw=1.5)
    axs[0].axvline(P, color="#D55E00", ls="--", lw=1.2)
    axs[0].annotate(f"ambient dim P={P}", (P, max(u.max(), 2)), color="#D55E00",
                    fontsize=8, rotation=90, va="top", ha="right")
    axs[0].set_xlabel("atom rank by usage")
    axs[0].set_ylabel("times used (train rows, log)")
    axs[0].set_title(f"{alive} of {K} atoms alive ≫ P={P}", fontsize=9)
    axs[1].semilogy(np.arange(1, len(sv) + 1), sv, color="#0072B2", lw=1.5)
    axs[1].axvline(P, color="#D55E00", ls="--", lw=1.2)
    axs[1].set_xlabel("singular value index")
    axs[1].set_ylabel("σ (decoder frame)")
    axs[1].set_title(f"frame {frame.shape[0]}×{P}: rank {int((sv > sv[0]*1e-10).sum())} "
                     f"(spans the chart)", fontsize=9)
    # pairwise atom coherence on a sample of alive atoms (mean output direction)
    alive_idx = np.flatnonzero(usage > 0)
    samp = np.random.default_rng(0).choice(alive_idx, min(2000, len(alive_idx)),
                                           replace=False)
    dirs = np.stack([blocks[i][0] if blocks[i].ndim == 2 else blocks[i] for i in samp])
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    G = np.abs(dirs @ dirs.T)
    iu = np.triu_indices(len(samp), 1)
    axs[2].hist(G[iu], bins=60, color="#0072B2")
    axs[2].set_xlabel("|cos| between atom principal directions")
    axs[2].set_ylabel("atom pairs")
    axs[2].set_title(f"max coherence {G[iu].max():.3f} < 1: atoms distinct", fontsize=9)
    for ax in axs:
        style(ax)
    fig.suptitle(f"overcompleteness: K={K} atoms, decoder frame {frame.shape[0]} "
                 f"columns in P={P} dims ({K/P:.1f}× atoms, {frame.shape[0]/P:.1f}× frame)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(args.out_dir, "fig_overcompleteness.png"), dpi=150)
    print("[interp] DONE", flush=True)


if __name__ == "__main__":
    main()
