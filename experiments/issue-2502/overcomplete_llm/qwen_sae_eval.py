"""Score Qwen's own published SAE on this issue's held-out split (#2502, criterion 4).

`Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50` is the model vendor's own
interpretability module for `Qwen/Qwen3.5-2B-Base`: a TopK SAE, `d_sae = 32768`
(16x expansion over `d_model = 2048`), `k = 50`, hook point `resid_post`,
released 2026-04-27 with a technical report (arXiv:2605.11887). It is the
standard method this issue's criterion 4 asks to be measured against, trained by
the people who trained the model.

Two arms are scored, both on the SAME held-out rows and the SAME FVU
denominator the Rust arms use:

* `qwen_sae` — the vendor's published inference, verbatim from their model card
  (`pre = x @ W_enc.T + b_enc`, keep the top 50, `x_hat = acts @ W_dec.T + b_dec`);
* `qwen_sae_ls` — the same support, with the 50 amplitudes re-solved by least
  squares. This can only improve their reconstruction, and is included so the
  comparison is against the strongest form of their dictionary rather than
  against their encoder's amortisation error.

Reconstructions are written centred (`x_hat - train_mean`) so the splice judge
can treat this arm exactly like a Rust arm.
"""

import argparse
import json
import os

import numpy as np
import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True, help="layer<L>.sae.pt")
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--train-mean", default="",
                    help="train_mean.f32 from a Rust arm; recomputed from train.npy if absent")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--dump-recon", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sae = torch.load(args.sae, map_location="cpu")
    w_enc = sae["W_enc"].float().cuda()  # (d_sae, d_model)
    b_enc = sae["b_enc"].float().cuda()
    w_dec = sae["W_dec"].float().cuda()  # (d_model, d_sae)
    b_dec = sae["b_dec"].float().cuda()
    d_sae, d_model = w_enc.shape
    print(f"[qwen-sae] d_sae={d_sae} d_model={d_model} k={args.topk}", flush=True)

    ev = np.load(os.path.join(args.acts_dir, "eval.npy"), mmap_mode="r")
    n, p = ev.shape
    if p != d_model:
        raise SystemExit(f"activation width {p} != SAE d_model {d_model}")
    if args.train_mean:
        mean = np.fromfile(args.train_mean, dtype=np.float32)
    else:
        tr = np.load(os.path.join(args.acts_dir, "train.npy"), mmap_mode="r")
        acc = np.zeros(p, dtype=np.float64)
        for s in range(0, tr.shape[0], args.chunk):
            acc += np.asarray(tr[s : s + args.chunk], dtype=np.float64).sum(0)
        mean = (acc / tr.shape[0]).astype(np.float32)
    if mean.size != p:
        raise SystemExit("train_mean width mismatch")
    mean_t = torch.from_numpy(mean).cuda()

    eval_mean = np.zeros(p, dtype=np.float64)
    for s in range(0, n, args.chunk):
        eval_mean += np.asarray(ev[s : s + args.chunk], dtype=np.float64).sum(0)
    eval_mean /= n
    eval_mean_t = torch.from_numpy(eval_mean.astype(np.float32)).cuda()

    files = {}
    if args.dump_recon:
        for arm in ("qwen_sae", "qwen_sae_ls"):
            os.makedirs(os.path.join(args.out_dir, arm), exist_ok=True)
            files[arm] = open(os.path.join(args.out_dir, arm, "eval_recon.f32"), "wb")
            mean.tofile(os.path.join(args.out_dir, arm, "train_mean.f32"))

    rss = {"qwen_sae": 0.0, "qwen_sae_ls": 0.0}
    tss = 0.0
    used = torch.zeros(d_sae, dtype=torch.bool, device="cuda")
    l0_total = 0
    n_pinv = 0
    with torch.no_grad():
        for s in range(0, n, args.chunk):
            x = torch.from_numpy(np.asarray(ev[s : s + args.chunk])).cuda().float()
            pre = x @ w_enc.T + b_enc
            vals, idx = pre.topk(args.topk, dim=-1)
            used[idx.reshape(-1)] = True
            l0_total += int((vals != 0).sum().item())

            atoms = w_dec.T[idx]  # (rows, k, d_model)
            recon = torch.einsum("rk,rkd->rd", vals, atoms) + b_dec
            # Least-squares amplitudes on the SAME support: the minimum-norm
            # solution of A^T c = x - b_dec over the k selected decoder columns.
            # `lstsq` needs no ridge, so no regularisation constant enters the
            # comparator. CUDA only offers the `gels` driver, which assumes full
            # column rank; 50 decoder columns in 2048 dimensions satisfy that, and
            # a non-finite solution falls back to the pseudo-inverse.
            target = (x - b_dec).unsqueeze(2)  # (rows, d_model, 1)
            a_t = atoms.transpose(1, 2)
            coef = torch.linalg.lstsq(a_t, target).solution.squeeze(2)
            bad = ~torch.isfinite(coef).all(dim=1)
            if bool(bad.any()):
                coef[bad] = (
                    torch.linalg.pinv(a_t[bad]) @ target[bad]
                ).squeeze(2)
                n_pinv += int(bad.sum().item())
            recon_ls = torch.einsum("rk,rkd->rd", coef, atoms) + b_dec

            centred = x - eval_mean_t
            tss += float((centred * centred).sum().item())
            for arm, r in (("qwen_sae", recon), ("qwen_sae_ls", recon_ls)):
                d = x - r
                rss[arm] += float((d * d).sum().item())
                if arm in files:
                    (r - mean_t).cpu().numpy().astype(np.float32).tofile(files[arm])
    for f in files.values():
        f.close()

    report = {
        "sae": args.sae,
        "d_sae": int(d_sae),
        "d_model": int(d_model),
        "topk": args.topk,
        "rows_scored": int(n),
        "measured_mean_l0": l0_total / n,
        "least_squares_pinv_fallback_rows": n_pinv,
        "features_used_on_heldout": int(used.sum().item()),
        "decoder_parameters": int(d_model * d_sae),
        "encoder_parameters": int(d_model * d_sae),
        "rate": {
            "active_scalars_per_token": args.topk,
            "selection_bits_per_token": args.topk * float(np.log2(d_sae)),
        },
        "arms": {
            arm: {
                "heldout_fvu": rss[arm] / tss,
                "heldout_explained_variance": 1.0 - rss[arm] / tss,
                "rss": rss[arm],
            }
            for arm in rss
        },
        "tss_about_eval_mean": tss,
    }
    with open(os.path.join(args.out_dir, "qwen_sae.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
