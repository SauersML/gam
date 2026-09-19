"""Standard-method comparator for issue #2502 (criterion 4): a TopK sparse
autoencoder (Gao et al. 2024, arXiv:2406.04093) trained with Adam on the SAME
train rows the Rust arms fit, and scored on the SAME held-out rows against the
SAME FVU denominator.

This is the comparator, not the method under test, so like `pca_reference.py`
it runs outside the Rust engine. Its recipe is the published one:

* pre-activations `(x - b_pre) W_enc + b_enc`, the top `k` kept, then ReLU;
* reconstruction `z W_dec + b_pre`, with every decoder row held at unit norm
  (the gradient component along a row is removed before each step and the row
  renormalised after it), and `W_dec` initialised to `W_encᵀ`;
* the AuxK loss: latents that have not fired for `--dead-epochs` epochs
  reconstruct the main residual with their own top `k_aux`, weighted `1/32`.

At matched active scalars per token `m = k·b`, this arm keeps `k = m` latents.
The rows are the Rust arms' own: the first `--rows` of `train.npy`, centred by the
Rust fit's `train_mean.f32`, so both see identical inputs.

Two held-out decodes on the SAME support are reported:

* `sae` -- the trained encoder's amplitudes, as the method deploys;
* `sae_ls` -- the amplitudes re-solved by least squares on that support, the
  strongest form of the dictionary.

Every learning rate in `--lrs` is trained and the one with the best held-out FVU
is reported. Selecting on the held-out split can only flatter the comparator.
"""

import argparse
import json
import math
import os
import shutil

import numpy as np
import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--rust-fit-dir", required=True,
                    help="a Rust arm's out dir: its train_mean.f32 centres the rows and its "
                         "numbers.json supplies the held-out FVU denominator")
    ap.add_argument("--latents", type=int, required=True)
    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--rows", type=int, default=100000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lrs", default="1e-4,3e-4,1e-3")
    ap.add_argument("--k-aux", type=int, default=512)
    ap.add_argument("--dead-epochs", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()


def centred(rows, mean):
    # The Rust reader subtracts the f32 train mean from each f32 row in f32; doing
    # the same here makes both arms see bit-identical inputs.
    return np.asarray(rows, dtype=np.float32) - mean


class TopKSae:
    def __init__(self, p, latents, k, generator):
        w = torch.randn(p, latents, generator=generator, device="cuda") / math.sqrt(p)
        self.w_enc = w.requires_grad_()
        self.w_dec = (w.detach().T / w.detach().T.norm(dim=1, keepdim=True)).clone().requires_grad_()
        self.b_pre = torch.zeros(p, device="cuda", requires_grad=True)
        self.b_enc = torch.zeros(latents, device="cuda", requires_grad=True)
        self.k = k

    def params(self):
        return [self.w_enc, self.w_dec, self.b_pre, self.b_enc]

    def encode(self, x):
        pre = (x - self.b_pre) @ self.w_enc + self.b_enc
        vals, idx = pre.topk(self.k, dim=1)
        return pre, torch.relu(vals), idx

    def decode(self, vals, idx):
        return torch.einsum("rk,rkp->rp", vals, self.w_dec[idx]) + self.b_pre


def train_one(x_train, x_eval, rss_denominator, args, lr):
    n, p = x_train.shape
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    sae = TopKSae(p, args.latents, args.topk, generator)
    opt = torch.optim.Adam(sae.params(), lr=lr)
    steps_per_epoch = math.ceil(n / args.batch)
    total = steps_per_epoch * args.epochs
    # Linear decay over the last fifth of training, as in the published recipe.
    decay_from = int(0.8 * total)
    last_fired = torch.zeros(args.latents, dtype=torch.long, device="cuda")
    curve = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n, device="cuda", generator=generator)
        for start in range(0, n, args.batch):
            xb = x_train[perm[start : start + args.batch]]
            pre, vals, idx = sae.encode(xb)
            recon = sae.decode(vals, idx)
            residual = xb - recon
            loss = residual.pow(2).sum(1).mean() / xb.pow(2).sum(1).mean()
            fired = torch.zeros(args.latents, dtype=torch.bool, device="cuda")
            fired[idx[vals > 0]] = True
            last_fired[fired] = step
            dead = (step - last_fired) >= args.dead_epochs * steps_per_epoch
            n_dead = int(dead.sum().item())
            if n_dead > 0:
                k_aux = min(args.k_aux, n_dead)
                dead_pre = pre.masked_fill(~dead, float("-inf"))
                aux_vals, aux_idx = dead_pre.topk(k_aux, dim=1)
                # Dense: gathering k_aux decoder rows per row would hold
                # batch x k_aux x p floats (21 GB at 4096 x 512 x 2560).
                aux_code = torch.zeros_like(pre).scatter(1, aux_idx, torch.relu(aux_vals))
                aux_recon = aux_code @ sae.w_dec
                target = residual.detach()
                aux = (target - aux_recon).pow(2).sum(1).mean() / target.pow(2).sum(1).mean()
                loss = loss + aux / 32.0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                g = sae.w_dec.grad
                g -= (g * sae.w_dec).sum(1, keepdim=True) * sae.w_dec
            if step >= decay_from:
                for group in opt.param_groups:
                    group["lr"] = lr * (total - step) / (total - decay_from)
            opt.step()
            with torch.no_grad():
                sae.w_dec /= sae.w_dec.norm(dim=1, keepdim=True)
            step += 1
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            rss, _rss_ls, alive = score(sae, x_eval, args.chunk, None)
            curve.append({"epoch": epoch, "heldout_fvu": rss / rss_denominator,
                          "alive_on_heldout": alive, "dead_in_training": n_dead})
            print(f"[topk-sae] lr={lr:g} epoch {epoch}/{args.epochs} "
                  f"heldout_fvu={rss / rss_denominator:.6f} alive={alive} dead={n_dead}",
                  flush=True)
    return sae, curve


def score(sae, x_eval, chunk, files):
    """Held-out RSS of the deployed and the least-squares decode, in f64."""
    rss = 0.0
    rss_ls = 0.0
    used = torch.zeros(sae.w_dec.shape[0], dtype=torch.bool, device="cuda")
    with torch.no_grad():
        for start in range(0, x_eval.shape[0], chunk):
            x = x_eval[start : start + chunk]
            _pre, vals, idx = sae.encode(x)
            used[idx[vals > 0]] = True
            recon = sae.decode(vals, idx)
            # Least-squares amplitudes on the same support, signed like the Rust
            # arms' codes: the orthogonal projection of x - b_pre onto the span of
            # the k selected decoder rows.
            atoms = sae.w_dec[idx].transpose(1, 2)  # (rows, p, k)
            coef = torch.linalg.lstsq(atoms, (x - sae.b_pre).unsqueeze(2)).solution
            if not bool(torch.isfinite(coef).all()):
                raise SystemExit(f"rows {start}..{start + x.shape[0]}: a selected support is "
                                 "rank deficient, so its least-squares amplitudes are not finite")
            recon_ls = (atoms @ coef).squeeze(2) + sae.b_pre
            rss += float((x - recon).double().pow(2).sum().item())
            rss_ls += float((x - recon_ls).double().pow(2).sum().item())
            if files is not None:
                files["sae"].write(recon.cpu().numpy().astype(np.float32).tobytes())
                files["sae_ls"].write(recon_ls.cpu().numpy().astype(np.float32).tobytes())
    return rss, rss_ls, int(used.sum().item())


def main():
    args = parse_args()
    # Full f32 matmuls: TF32 would round the held-out decode to 10 mantissa bits.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(args.out_dir, exist_ok=True)
    rust = json.load(open(os.path.join(args.rust_fit_dir, "numbers.json")))
    mean_path = os.path.join(args.rust_fit_dir, "train_mean.f32")
    mean = np.fromfile(mean_path, dtype=np.float32)
    train = np.load(os.path.join(args.acts_dir, "train.npy"), mmap_mode="r")
    ev = np.load(os.path.join(args.acts_dir, "eval.npy"), mmap_mode="r")
    n = min(args.rows, train.shape[0])
    p = train.shape[1]
    if mean.size != p:
        raise SystemExit(f"{mean_path}: {mean.size} floats, activations have p={p}")
    if rust["inputs"]["n_train"] != n or rust["inputs"]["n_eval"] != ev.shape[0]:
        raise SystemExit(f"rows differ from the Rust arm's: n_train={n} n_eval={ev.shape[0]} vs "
                         f"{rust['inputs']['n_train']} {rust['inputs']['n_eval']}")
    denominator = rust["heldout"]["tss_about_eval_mean"]
    # Positive control on the denominator: recompute it from the same rows.
    eval_c = np.concatenate([centred(ev[s : s + args.chunk], mean)
                             for s in range(0, ev.shape[0], args.chunk)])
    eval_mean = eval_c.astype(np.float64).mean(0)
    own = float(((eval_c.astype(np.float64) - eval_mean) ** 2).sum())
    print(f"[topk-sae] rows train={n} eval={eval_c.shape[0]} p={p} latents={args.latents} "
          f"k={args.topk} denominator rust={denominator:.9e} recomputed={own:.9e} "
          f"rel={abs(own - denominator) / denominator:.2e}", flush=True)
    x_train = torch.from_numpy(centred(train[:n], mean)).cuda()
    x_eval = torch.from_numpy(eval_c).cuda()

    runs = []
    best = None
    for lr in [float(v) for v in args.lrs.split(",")]:
        sae, curve = train_one(x_train, x_eval, denominator, args, lr)
        final = curve[-1]["heldout_fvu"]
        runs.append({"lr": lr, "curve": curve, "final_heldout_fvu": final})
        if best is None or final < best[0]:
            best = (final, lr, sae)
    _final, lr, sae = best

    files = {}
    for arm in ("sae", "sae_ls"):
        d = os.path.join(args.out_dir, arm)
        os.makedirs(d, exist_ok=True)
        files[arm] = open(os.path.join(d, "eval_recon.f32"), "wb")
        shutil.copyfile(mean_path, os.path.join(d, "train_mean.f32"))
    rss, rss_ls, alive = score(sae, x_eval, args.chunk, files)
    for fh in files.values():
        fh.close()
    params = 2 * args.latents * p + args.latents + p
    for arm, value in (("sae", rss), ("sae_ls", rss_ls)):
        report = {
            "arm": arm,
            "engine": "torch TopK SAE (Gao et al. 2024), the comparator",
            "inputs": {"n_train": n, "n_eval": int(eval_c.shape[0]), "p": p,
                       "centred_by": mean_path},
            "dictionary": {"K_atoms": args.latents, "block_size": 1, "block_topk": args.topk,
                           "overcomplete_ratio_K_over_p": args.latents / p,
                           "parameters": params},
            "fit": {"optimizer": "Adam", "selected_lr": lr, "epochs": args.epochs,
                    "batch": args.batch, "k_aux": args.k_aux, "aux_weight": 1.0 / 32.0,
                    "dead_epochs": args.dead_epochs, "seed": args.seed, "lr_sweep": runs},
            "heldout": {"rows_scored": int(eval_c.shape[0]), "rss": value,
                        "tss_about_eval_mean": denominator, "fvu": value / denominator,
                        "explained_variance": 1.0 - value / denominator,
                        "latents_used_on_heldout": alive},
            "rate": {"active_scalars_per_token": args.topk,
                     "selection_bits_per_token": args.topk * math.log2(args.latents),
                     "firings_per_token": args.topk},
        }
        with open(os.path.join(args.out_dir, arm, "numbers.json"), "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[topk-sae] {arm}: heldout_fvu={value / denominator:.6f} lr={lr:g} "
              f"latents_used={alive}/{args.latents} params={params}", flush=True)


if __name__ == "__main__":
    main()
