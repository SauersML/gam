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

Row edits (`--edit row`, meta "edit": "row"): the same weight edit W + (alpha - 1) u v^T applied to M at
chosen rows only, alpha in (0, 2) (interchange's FACTORS' removal and doubling), every `--stride`-th
subcomponent of each site. At p, the next-token distribution under the edit at row p alone (the same as
from row p on: later rows do not reach p): kl (KL(M || M_e), nats), effect (KL(M_e || M), bits), next,
the 10 most raised and lowered tokens, as above, and the activity at every position. Per subcomponent and
its first `--continue` top contexts, greedy continuations of S tokens after p: clean (cont_clean), with
the edit at row p only (cont_row_<alpha>) and with it at every row from p on, the generated ones
included (cont_from_<alpha>). Effects of either size, zero included, are kept: predicting no change is
part of the task. The next-token fields also come for three larger edits (suffixes): _from, the edit at
every row from the first token after the sink on (at p, every earlier row's edit reaches it too); _group,
the subcomponent with the other GROUP - 1 subcomponents of its site most active at p (|v . x|), removed or
doubled together at row p (their numbers in `group`, their activities at p in `group_act`); _groupfrom,
the group from the first token on; _sites, a multi-site edit: the subcomponent and, at the site of its kind
in every other layer, the subcomponent most active at p in the clean run (`sites_member` [C, K, layers]:
each layer's number, the subcomponent's own in its layer; `sites_act` their activities at p), edited
together at row p; _sitesfrom, the same from the first token on. Continuations also come for the group
from p on (cont_groupfrom_<alpha>), the size of edit whose continuation changes often enough to ask about.

  vpd_labels.py --out DIR [--offset 0] [--pool 2048] [--top 16] [--random 16] [--sites h.0.mlp.c_fc,...]
                [--limit N] [--rows 256] [--seed 0] [--edit all|row] [--stride 1] [--continue 1]
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
ROW_ALPHAS = (("ablate", 0.0), ("amplify", 2.0))
# Row-edit variants (field suffix, edited rows, edited subcomponents): the subcomponent at row p only; from
# the first token after the sink on (rows 1..T-1, so at p every earlier row's edit reaches it too); and the
# same two for its group, the GROUP subcomponents of its site most active (|v . x|) at p, itself included.
# The multi-site variants: the subcomponent with, at its kind's site in every other layer, the subcomponent
# most active at p (clean run), together.
VARIANTS = (("", "row", "one"), ("_from", "from", "one"), ("_group", "row", "group"), ("_groupfrom", "from", "group"),
            ("_sites", "row", "sites"), ("_sitesfrom", "from", "sites"))
GROUP = 8
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

    def layer(self, i: int, x: torch.Tensor, edit=None, record: dict | None = None):
        """Layer i on the stream x; edit = (site name, v [R, d_in], u [R, d_out], coef [R][, start [R][, stop
        [R]]]): the edit at every row, or at rows start_r <= t (< stop_r) only; or a list of such edits at
        several sites (each acts at its own site; others are ignored). Returns the stream after the layer
        and, with an edit, the edited site's input; `record` collects every site's input by name."""
        t = self.t
        B, L, _ = x.shape
        seen = None
        edits = [] if edit is None else (edit if isinstance(edit, list) else [edit])

        def site(kind, h):
            nonlocal seen
            name = f"h.{i}.{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}.{kind}"
            out = t.site(name)(h)
            if record is not None:
                record[name] = h
            for edit in (e for e in edits if e[0] == name):
                _, v, u, coef = edit[:4]
                seen = h
                if v.dim() == 2:  # one subcomponent per row, or k of them: v [R, k, d_in], u [R, k, d_out]
                    v, u = v[:, None], u[:, None]
                g = coef[:, None, None] * torch.einsum("rtd,rkd->rkt", h, v)
                if len(edit) > 4:  # rows from start_r on (before stop_r)
                    t_ = torch.arange(h.shape[1], device=h.device)[None, None]
                    g = g * ((t_ >= edit[4][:, None, None]) & ((t_ < edit[5][:, None, None]) if len(edit) > 5 else True))
                out = out + torch.einsum("rkt,rko->rto", g, u)
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
        """Layers l on from the stream entering layer l (edits name their sites, so each acts in its own
        layer); the final normed stream and the input of the edited site of layer l."""
        seen = None
        for i in range(l, self.t.n_layer):
            x, s = self.layer(i, x, edit)
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
    never reaches them), every step a full forward with the rows' edit (a weight edit acts everywhere,
    or at the rows its start and stop give)."""
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
            x_, _ = model.layer(i, x_, edit)  # an edit (or list of them) acts at its own sites
        final = VM.rms(x_, model.t.ln_f, model.t.eps)
        nxt = (final[rows, lengths - 1] @ model.t.wte.T).argmax(-1)
        out[:, s] = nxt
        ids[rows, lengths] = nxt
        lengths = lengths + 1
    return out


UV: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


@torch.no_grad()
def row_labels(args, model: Model, ids: torch.Tensor, entering, final, peaks, names, out: Path, rng):
    """The row-edit table (module note, `--edit row`)."""
    dev, T = model.dev, ids.shape[1]
    K = args.top + args.random
    for name in names:
        tag = name.replace("h.", "").replace(".attn.", "_").replace(".mlp.", "_")
        if (out / f"site_{tag}.json").exists():
            continue
        started = time.time()
        U, V = UV[name]
        layer = int(name.split(".")[1])
        chosen = torch.arange(0, U.shape[0] if not args.limit else min(args.limit, U.shape[0]), args.stride)
        C = len(chosen)
        order = peaks[name][:, chosen].T.argsort(dim=-1, descending=True).cpu().numpy()
        ctx = np.empty((C, K), dtype=np.int64)
        for c in range(C):
            ctx[c] = np.concatenate([order[c, : args.top], rng.choice(order[c, args.top :], size=args.random, replace=False)])
        ctx_t = torch.from_numpy(ctx).to(dev)
        activity = torch.empty(C, K, T, device=dev)
        position = torch.empty(C, K, dtype=torch.int64, device=dev)
        fields = {a: {} for a, _ in ROW_ALPHAS}
        groups, group_acts, site_members, site_acts = [], [], [], []
        kind = name.split(".")[-1]
        others = [f"h.{l}.{name.split('.')[2]}.{kind}" for l in range(model.t.n_layer)]  # its kind's site in every layer
        pairs = torch.cartesian_prod(torch.arange(C, device=dev), torch.arange(K, device=dev))
        for s in range(0, len(pairs), args.rows):
            cs, ks = pairs[s : s + args.rows, 0], pairs[s : s + args.rows, 1]
            R = len(cs)
            at = torch.arange(R, device=dev)
            rows_ctx = ctx_t[cs, ks]
            sub = chosen.to(dev)[cs]
            v, u = V[:, sub].T, U[sub]
            _, seen = model.from_layer(layer, entering[layer][rows_ctx], (name, v, u, torch.zeros(R, device=dev)))
            act = torch.einsum("rtd,rd->rt", seen, v)
            pos = act[:, 1:].abs().argmax(-1) + 1  # after the first token
            activity[cs, ks], position[cs, ks] = act, pos
            lp_c = model.log_probs(final[rows_ctx, pos])
            # Its group: the GROUP subcomponents of the site most active at p, itself first.
            signed = seen[at, pos] @ V  # [R, all of the site's subcomponents]
            every = signed.abs()
            every[at, sub] = float("inf")
            group = every.topk(GROUP, dim=-1).indices  # [R, GROUP]
            groups.append(group.to(torch.int32).cpu())
            group_acts.append(signed.gather(-1, group).float().cpu())
            # Its sites: in every other layer, the subcomponent of its kind's site most active at p in the clean run.
            inputs = {}
            x0 = entering[0][rows_ctx]
            for i in range(model.t.n_layer):
                x0, _ = model.layer(i, x0, None, inputs)
            members, member_acts = [], []
            for n_ in others:
                if n_ == name:
                    members.append(sub); member_acts.append(signed[at, sub])
                    continue
                a_ = inputs[n_][at, pos] @ UV[n_][1]  # [R, the site's subcomponents]
                m_ = a_.abs().argmax(-1)
                members.append(m_); member_acts.append(a_[at, m_])
            site_members.append(torch.stack(members, -1).to(torch.int32).cpu())
            site_acts.append(torch.stack(member_acts, -1).float().cpu())
            for (alpha_name, alpha), (suffix, span, what) in [(a, v_) for a in ROW_ALPHAS for v_ in VARIANTS]:
                rows_ = (pos, pos + 1) if span == "row" else (torch.ones_like(pos),)
                coef = torch.full((R,), alpha - 1.0, device=dev)
                if what == "sites":  # one subcomponent per layer, from the first layer on
                    edits = [(n_, UV[n_][1][:, m_].T, UV[n_][0][m_], coef, *rows_) for n_, m_ in zip(others, members)]
                    h, _ = model.from_layer(0, entering[0][rows_ctx], edits)
                else:
                    ev, eu = (V.T[group], U[group]) if what == "group" else (v, u)
                    h, _ = model.from_layer(layer, entering[layer][rows_ctx], (name, ev, eu, coef, *rows_))
                lp_e = model.log_probs(h[at, pos])
                delta, dp = lp_e - lp_c, lp_e.exp() - lp_c.exp()
                upi, dni = dp.topk(TOP, dim=-1).indices, (-dp).topk(TOP, dim=-1).indices
                nt = ids[rows_ctx, (pos + 1).clamp(max=T - 1)]
                nd = delta.gather(-1, nt[:, None])[:, 0]
                vals = {"kl": (lp_c.exp() * -delta).sum(-1), "effect": (lp_e.exp() * delta).sum(-1) / math.log(2),
                        "next": torch.where(pos + 1 < T, nd, torch.full_like(nd, float("nan"))),
                        "up_ids": upi, "up_dp": dp.gather(-1, upi), "up_dlogp": delta.gather(-1, upi),
                        "down_ids": dni, "down_dp": dp.gather(-1, dni), "down_dlogp": delta.gather(-1, dni)}
                for key, val in vals.items():
                    fields[alpha_name].setdefault((key, suffix), []).append(val.cpu())
        rec = {}
        for alpha_name, _ in ROW_ALPHAS:
            for (key, suffix), parts in fields[alpha_name].items():
                val = torch.cat(parts).reshape(C, K, *parts[0].shape[1:])
                rec[f"{key}_{alpha_name}{suffix}"] = val.to(torch.int32) if key.endswith("_ids") else val.to(torch.float32)
        rec["group"] = torch.cat(groups).reshape(C, K, GROUP)
        rec["group_act"] = torch.cat(group_acts).reshape(C, K, GROUP)  # their activities v . x at p
        rec["sites_member"] = torch.cat(site_members).reshape(C, K, -1)  # per layer, the edited subcomponent of its kind's site
        rec["sites_act"] = torch.cat(site_acts).reshape(C, K, -1)
        # Continuations after p in each subcomponent's first `--continue` top contexts.
        J = args.cont
        conts = {"cont_clean": torch.empty(C, J, S, dtype=torch.int32)}
        for alpha_name, _ in ROW_ALPHAS:
            conts[f"cont_row_{alpha_name}"] = torch.empty(C, J, S, dtype=torch.int32)
            conts[f"cont_from_{alpha_name}"] = torch.empty(C, J, S, dtype=torch.int32)
            conts[f"cont_groupfrom_{alpha_name}"] = torch.empty(C, J, S, dtype=torch.int32)
        cj = torch.cartesian_prod(torch.arange(C), torch.arange(J))
        step = max(1, args.rows // 4)
        for s in range(0, len(cj), step):
            cs, js = cj[s : s + step, 0], cj[s : s + step, 1]
            pos = position[cs.to(dev), js.to(dev)]
            prefixes = [ids[int(ctx[c, j]), : int(pos[r]) + 1] for r, (c, j) in enumerate(zip(cs.tolist(), js.tolist()))]
            sub = chosen.to(dev)[cs.to(dev)]
            v, u = V[:, sub].T, U[sub]
            conts["cont_clean"][cs, js] = greedy(model, prefixes, None, S).to(torch.int32).cpu()
            for alpha_name, alpha in ROW_ALPHAS:
                coef = torch.full((len(cs),), alpha - 1.0, device=dev)
                conts[f"cont_row_{alpha_name}"][cs, js] = greedy(model, prefixes, (name, v, u, coef, pos, pos + 1), S).to(torch.int32).cpu()
                conts[f"cont_from_{alpha_name}"][cs, js] = greedy(model, prefixes, (name, v, u, coef, pos), S).to(torch.int32).cpu()
                g_ = rec["group"][cs, js].long().to(dev)  # the group most active at p in this context
                conts[f"cont_groupfrom_{alpha_name}"][cs, js] = greedy(model, prefixes, (name, V.T[g_], U[g_], coef, pos), S).to(torch.int32).cpu()
        rec.update(conts)
        rec.update({"subcomponents": chosen.to(torch.int32), "contexts": torch.from_numpy(ctx).to(torch.int32), "activity": activity.to(torch.float16).cpu(),
                    "position": position.to(torch.int16).cpu()})
        save_file({k: val.contiguous() for k, val in rec.items()}, str(out / f"site_{tag}.safetensors"))
        meta = {"site": name, "layer": layer, "subcomponents": C, "stride": args.stride, "top": args.top, "random": args.random, "pool_offset": args.offset, "pool": args.pool,
                "alphas": dict(ROW_ALPHAS), "amplify": dict(ROW_ALPHAS)["amplify"], "edit": "row", "continuation_steps": S, "continue": J,
                "seconds": time.time() - started, "seed": args.seed, "decomposition": str(VM.VPD_PTH), "target": str(VM.TARGET_DIR)}
        (out / f"site_{tag}.json").write_text(json.dumps(meta))
        print(json.dumps({"site": name, "subcomponents": C, "seconds": round(meta["seconds"], 1), "median_effect_bits_ablate": float(rec["effect_ablate"].median()),
                          "share_above_0.1_bits": float((rec["effect_ablate"] > 0.1).float().mean())}), flush=True)


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
    ap.add_argument("--edit", default="all", choices=("all", "row"), help="row: the row-edit table (module note)")
    ap.add_argument("--stride", type=int, default=1, help="row edits: every stride-th subcomponent of each site")
    ap.add_argument("--continue", dest="cont", type=int, default=1, help="row edits: continuations in each subcomponent's first N top contexts")
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
    if args.edit == "row":
        return row_labels(args, model, ids, entering, final, peaks, names, out, rng)
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
