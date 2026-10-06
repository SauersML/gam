"""Label table for the oracle on VPD's decomposition of vpd4l (#2951): the measured behaviour of each of
VPD's 38,912 rank-one subcomponents u v^T (run s-55ea3f9b of the 4-layer Pile model t-9d2b8f02), the
labels the oracle is trained and scored on.

A subcomponent c of a site (a linear map y = W x of one layer: q, k, v, o, c_fc or down_proj) reads its
activity v . x from the site's input x and writes (v . x) u. Its native edit is W(alpha) = W + (alpha - 1)
u v^T: the site's output gains (alpha - 1)(v . x) u at every position, exactly, and every later
computation is run again (no linear approximation). alpha = 0 removes the subcomponent, 1.5 amplifies it.

Contexts. Rows of the Pile validation split (fresh text; VPD trained on the train split), the first 128
tokens of each: a pool of `--pool` rows starting at `--offset` (training labels: offset 0; held-out
labels: a disjoint offset). For each subcomponent: its `--top` contexts of largest max_t |v . x_t| in the
pool and `--random` more drawn uniformly from the rest (seeded), so both its strong and its ordinary
behaviour are measured.

Per subcomponent and context (at p, the context's position of largest |v . x|):
  activity [T]          v . x at every position (float16)
  position              p
  for alpha in (0, 1.5): kl (KL(clean || edited) of the next-token distribution at p, nats), next (the
                        change of log p of the actual next token), and the 10 tokens whose probability
                        rises most and the 10 that fall most (ids, change of probability, change of
                        log-probability)
and per subcomponent a greedy continuation of S tokens after p in its first top context, with alpha =
AMPLIFY, beside the clean continuation of the same prefix.

Execution. The clean pass over the pool keeps the residual stream entering each layer and the final
normed stream; an edited row starts from the stream entering its site's layer. Float32, TF32 off;
log-probabilities normalized in float64 where the device has it. The model is the paper code's
(bench/vpd_2951/vpd_model.py, the same forward as the VPD repository's LlamaSimpleMLP).

Output under OUT: site_<layer>_<kind>.safetensors and .json per site, contexts.safetensors (the pool's
row ids and tokens).

  vpd_labels.py --out DIR [--offset 0] [--pool 2048] [--top 16] [--random 16] [--sites h.0.mlp.c_fc,...]
                [--limit N] [--rows 256] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vpd_2951"))
import vpd_model as VM  # noqa: E402

ALPHAS = (("ablate", 0.0), ("amplify", 1.5))
AMPLIFY = 1.5
T, S, TOP = 128, 12, 10


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_uv(device: torch.device, path: Path | None = None) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Per site name, U [C, d_out] and V [d_in, C] of the published decomposition (no CI network): from
    the run's checkpoint, or from `path`, a file `export_uv` wrote (the subcomponents alone, 0.15 GB)."""
    if path is not None:
        from safetensors.torch import load_file

        d = load_file(str(path))
        return {n: (d[f"{n}.U"].to(device), d[f"{n}.V"].to(device)) for n in VM.site_names()}
    raw = torch.load(str(VM.VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    uv: dict[str, list] = {}
    for k, v in raw.items():
        if k.startswith("_components."):
            site, which = k[len("_components.") :].rsplit(".", 1)
            uv.setdefault(site.replace("-", "."), [None, None])["UV".index(which)] = v.float().to(device)
    return {n: (uv[n][0], uv[n][1]) for n in VM.site_names()}


def export_uv(path: Path):
    uv = load_uv(torch.device("cpu"))
    save_file({f"{n}.{w}": t.contiguous() for n, (U, V) in uv.items() for w, t in (("U", U), ("V", V))}, str(path))


class Model:
    """vpd4l run layer by layer (the paper code's Target.hidden), with an optional per-row edit
    out += coef_r (x . v_r) u_r at one site, and the site's input recorded."""

    def __init__(self, dev: torch.device):
        torch.backends.cuda.matmul.allow_tf32 = False
        self.dev = dev
        self.t = VM.load_target(str(dev))
        self.wide = torch.float32 if dev.type == "mps" else torch.float64

    def layer(self, i: int, x: torch.Tensor, edit=None):
        """Layer i on the stream x; edit = (site name, v [R, d_in], u [R, d_out], coef [R]). Returns the
        stream after the layer and, with an edit, the edited site's input."""
        t = self.t
        B, L, _ = x.shape
        seen = None

        def site(kind, h):
            nonlocal seen
            name = f"h.{i}.{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}.{kind}"
            out = t.site(name)(h)
            if edit is not None and edit[0] == name:
                _, v, u, coef = edit
                seen = h
                out = out + (coef[:, None] * torch.einsum("rtd,rd->rt", h, v))[..., None] * u[:, None, :]
            return out

        h = VM.rms(x, t.norms[2 * i], t.eps)
        q = site("q_proj", h).view(B, L, t.n_head, t.hd).transpose(1, 2)
        k = site("k_proj", h).view(B, L, t.n_head, t.hd).transpose(1, 2)
        v = site("v_proj", h).view(B, L, t.n_head, t.hd).transpose(1, 2)
        q, k = t._rope(q, L), t._rope(k, L)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + site("o_proj", y.transpose(1, 2).reshape(B, L, -1))
        h = VM.rms(x, t.norms[2 * i + 1], t.eps)
        x = x + site("down_proj", VM.gelu_tanh(site("c_fc", h)))
        return x, seen

    @torch.no_grad()
    def clean(self, ids: torch.Tensor):
        """The stream entering each layer and the final normed stream."""
        x = self.t.wte[ids]
        entering = []
        for i in range(self.t.n_layer):
            entering.append(x)
            x, _ = self.layer(i, x)
        return entering, VM.rms(x, self.t.ln_f, self.t.eps)

    @torch.no_grad()
    def from_layer(self, l: int, x: torch.Tensor, edit):
        seen = None
        for i in range(l, self.t.n_layer):
            x, s = self.layer(i, x, edit if i == l else None)
            seen = s if i == l else seen
        return VM.rms(x, self.t.ln_f, self.t.eps), seen

    def log_probs(self, h: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax((h @ self.t.wte.T).to(self.wide), dim=-1)


@torch.no_grad()
def site_inputs(model: Model, ids: torch.Tensor, chunk: int):
    """Per site, max_t |x . V| over each pool row ([pool, C]), by forward passes recording every site's input."""
    t = model.t
    names = VM.site_names(t.n_layer)
    out = {}
    for s in range(0, len(ids), chunk):
        for n in names:
            t.site(n).cache_input = True
        t.hidden(ids[s : s + chunk])
        for n in names:
            st = t.site(n)
            act = (st.last_input @ UV[n][1]).abs().amax(1)
            out.setdefault(n, []).append(act)
            st.last_input, st.cache_input = None, False
    return {n: torch.cat(v) for n, v in out.items()}


@torch.no_grad()
def greedy(model: Model, prefixes: list[torch.Tensor], edit, steps: int) -> torch.Tensor:
    """Greedy continuations of right-padded prefixes (causal attention: padding after a row's tokens
    never reaches them), every step a full forward with the rows' edit (a weight edit acts everywhere)."""
    R = len(prefixes)
    lengths = torch.tensor([len(p) for p in prefixes], device=model.dev)
    width = int(lengths.max()) + steps
    ids = torch.zeros(R, width, dtype=torch.long, device=model.dev)
    for r, p in enumerate(prefixes):
        ids[r, : len(p)] = p
    rows = torch.arange(R, device=model.dev)
    out = torch.empty(R, steps, dtype=torch.long, device=model.dev)
    for s in range(steps):
        cur = int(lengths.max())
        x_ = model.t.wte[ids[:, :cur]]
        for i in range(model.t.n_layer):
            x_, _ = model.layer(i, x_, edit if edit is not None and int(edit[0].split(".")[1]) == i else None)
        final = VM.rms(x_, model.t.ln_f, model.t.eps)
        nxt = (final[rows, lengths - 1] @ model.t.wte.T).argmax(-1)
        out[:, s] = nxt
        ids[rows, lengths] = nxt
        lengths = lengths + 1
    return out


UV: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", help="the target run's directory (default: the paper code's ~/mpd-data/vpd/t-9d2b8f02)")
    ap.add_argument("--uv", help="the subcomponents file export_uv wrote (default: read the decomposition's checkpoint)")
    ap.add_argument("--data", help="the Pile validation rows (default: the paper code's ~/mpd-data/vpd/pile_val_4096x513.npy)")
    ap.add_argument("--export-uv", action="store_true", help="write the subcomponents to OUT/uv.safetensors and stop")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--pool", type=int, default=2048)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--random", type=int, default=16)
    ap.add_argument("--sites", default="")
    ap.add_argument("--limit", type=int, default=0, help="only the first N subcomponents of each site (a smoke test)")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.export_uv:
        export_uv(out / "uv.safetensors")
        return
    if args.target:
        VM.TARGET_DIR = Path(args.target)
    if args.data:
        VM.DATA = Path(args.data)
    dev = device()
    model = Model(dev)
    UV.update(load_uv(dev, Path(args.uv) if args.uv else None))
    ids = VM.val_tokens(args.pool, T, args.offset).to(dev)
    save_file({"rows": torch.arange(args.offset, args.offset + args.pool, dtype=torch.int64), "tokens": ids.to(torch.int32).cpu()}, str(out / "contexts.safetensors"))
    t0 = time.time()
    peaks = site_inputs(model, ids, 128)
    entering, final = model.clean(ids)
    print(json.dumps({"pool": args.pool, "clean_seconds": round(time.time() - t0, 1)}), flush=True)
    rng = np.random.default_rng(args.seed)
    names = [s for s in args.sites.split(",") if s] or VM.site_names(model.t.n_layer)
    K = args.top + args.random
    for name in names:
        tag = name.replace("h.", "").replace(".attn.", "_").replace(".mlp.", "_")
        if (out / f"site_{tag}.json").exists():
            continue
        started = time.time()
        U, V = UV[name]
        layer = int(name.split(".")[1])
        C = U.shape[0] if not args.limit else min(args.limit, U.shape[0])
        # Contexts per subcomponent: the top ones by peak |activity|, then uniform others.
        order = peaks[name][:, :C].T.argsort(dim=-1, descending=True).cpu().numpy()  # [C, pool]
        ctx = np.empty((C, K), dtype=np.int64)
        for c in range(C):
            rest = order[c, args.top :]
            ctx[c] = np.concatenate([order[c, : args.top], rng.choice(rest, size=args.random, replace=False)])
        ctx_t = torch.from_numpy(ctx).to(dev)
        activity = torch.empty(C, K, T, dtype=torch.float32, device=dev)
        position = torch.empty(C, K, dtype=torch.int64, device=dev)
        rec = {}
        for alpha_name, alpha in ALPHAS:
            kl = torch.empty(C, K, dtype=model.wide, device=dev)
            nxt = torch.empty(C, K, dtype=model.wide, device=dev)
            up_ids = torch.empty(C, K, TOP, dtype=torch.int64, device=dev)
            down_ids = torch.empty(C, K, TOP, dtype=torch.int64, device=dev)
            up_dp, up_dl, down_dp, down_dl = (torch.empty(C, K, TOP, dtype=model.wide, device=dev) for _ in range(4))
            pairs = torch.cartesian_prod(torch.arange(C, device=dev), torch.arange(K, device=dev))
            for s in range(0, len(pairs), args.rows):
                cs, ks = pairs[s : s + args.rows, 0], pairs[s : s + args.rows, 1]
                R = len(cs)
                rows_ctx = ctx_t[cs, ks]
                edit = (name, V[:, cs].T, U[cs], torch.full((R,), alpha - 1.0, device=dev))
                h, seen = model.from_layer(layer, entering[layer][rows_ctx], edit)
                act = torch.einsum("rtd,rd->rt", seen, V[:, cs].T)
                if alpha_name == "ablate":
                    activity[cs, ks] = act
                    position[cs, ks] = act.abs().argmax(-1)
                pos = position[cs, ks]
                at = torch.arange(R, device=dev)
                lp_e = model.log_probs(h[at, pos])
                lp_c = model.log_probs(final[rows_ctx, pos])
                delta = lp_e - lp_c
                dp = lp_e.exp() - lp_c.exp()
                kl[cs, ks] = (lp_c.exp() * -delta).sum(-1)
                upi, dni = dp.topk(TOP, dim=-1).indices, (-dp).topk(TOP, dim=-1).indices
                up_ids[cs, ks], down_ids[cs, ks] = upi, dni
                up_dp[cs, ks], up_dl[cs, ks] = dp.gather(-1, upi), delta.gather(-1, upi)
                down_dp[cs, ks], down_dl[cs, ks] = dp.gather(-1, dni), delta.gather(-1, dni)
                nt = ids[rows_ctx, (pos + 1).clamp(max=T - 1)]
                nd = delta.gather(-1, nt[:, None])[:, 0]
                nxt[cs, ks] = torch.where(pos + 1 < T, nd, torch.full_like(nd, float("nan")))
            f32 = lambda x: x.to(torch.float32).cpu()  # noqa: E731
            rec.update({f"kl_{alpha_name}": f32(kl), f"next_{alpha_name}": f32(nxt), f"up_ids_{alpha_name}": up_ids.to(torch.int32).cpu(), f"up_dp_{alpha_name}": f32(up_dp),
                        f"up_dlogp_{alpha_name}": f32(up_dl), f"down_ids_{alpha_name}": down_ids.to(torch.int32).cpu(), f"down_dp_{alpha_name}": f32(down_dp), f"down_dlogp_{alpha_name}": f32(down_dl)})
        # Continuations after the first top context's peak position, amplified and clean.
        clean_cont = torch.empty(C, S, dtype=torch.int32)
        amp_cont = torch.empty(C, S, dtype=torch.int32)
        for s in range(0, C, args.rows // 4 or 1):
            cs = torch.arange(s, min(C, s + (args.rows // 4 or 1)), device=dev)
            prefixes = [ids[int(ctx[c, 0]), : int(position[c, 0]) + 1] for c in cs.tolist()]
            amp_cont[cs.cpu()] = greedy(model, prefixes, (name, V[:, cs].T, U[cs], torch.full((len(cs),), AMPLIFY - 1.0, device=dev)), S).to(torch.int32).cpu()
            clean_cont[cs.cpu()] = greedy(model, prefixes, None, S).to(torch.int32).cpu()
        rec.update({"contexts": torch.from_numpy(ctx).to(torch.int32), "activity": activity.to(torch.float16).cpu(), "position": position.to(torch.int16).cpu(),
                    "clean": clean_cont, "amplified": amp_cont})
        save_file({k: v.contiguous() for k, v in rec.items()}, str(out / f"site_{tag}.safetensors"))
        meta = {"site": name, "layer": layer, "subcomponents": C, "top": args.top, "random": args.random, "pool_offset": args.offset, "pool": args.pool,
                "alphas": dict(ALPHAS), "amplify": AMPLIFY, "continuation_steps": S, "seconds": time.time() - started, "seed": args.seed,
                "decomposition": str(VM.VPD_PTH), "target": str(VM.TARGET_DIR)}
        (out / f"site_{tag}.json").write_text(json.dumps(meta))
        print(json.dumps({"site": name, "subcomponents": C, "seconds": round(time.time() - started, 1),
                          "median_kl_ablate": float(rec["kl_ablate"].median()), "median_kl_amplify": float(rec["kl_amplify"].median()),
                          "continuations_changed": float((amp_cont != clean_cont).any(-1).float().mean())}), flush=True)
    print(json.dumps({"total_seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
