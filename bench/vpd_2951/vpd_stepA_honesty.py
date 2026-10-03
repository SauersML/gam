"""Like-for-like checks of per-word sets over VPD's subcomponents against VPD's own (#2951).

Two families of per-word sets on the same 38,912 rank-one subcomponents of the 4L Pile model are
scored in VPD's own harness (vpd_model / vpd_eval): VPD's causal importances g from its CI network,
and given sets (the masked driver's `OUT.pass{P}.{indptr,indices,offsets}.npy`, pieces numbered in
the driver's site order). A family is a lower bound per (word, subcomponent): VPD's g, VPD's g
rounded (g > 0), or a given set's indicator. Every mask lies in VPD's box [g, 1].

  box     the robustness claim over the whole box: the KL with the mask at its lower bound; VPD's
          threshold curve (rounded, g > 0.1, g > 0.5, g itself); 64 uniform draws of every off
          entry per word (m = g + (1 - g) U, the same U for every family); VPD's eval adversary
          (one source per subcomponent shared over every word, sign steps of 0.1) at 20/40/80
          steps. Each with the residual (delta) off, as the sets are scored, and with `--delta` also
          VPD's own delta semantics (uniform per word for draws, one adversarial coordinate for PGD).
  layers  cross-layer compensation: the KL with only layer l masked (the rest native), and the
          hybrids H_0 (native) .. H_4 (every layer masked), H_l masking layers < l, in total
          variation between consecutive hybrids and end to end.

Results go under `OUT[key][family]` (the file is read and rewritten, so the modes share it).
usage: vpd_stepA_honesty.py {box|layers} SETS_STEM OUT.json [--rows N] [--offset OFF] [--draws D]
       [--steps 20,40,80] [--key NAME] [--families vpd_ci,vpd_rounded,given] [--name NAME]
SETS_STEM is the driver's `OUT.pass{P}` path prefix (unused without `given`); `--offset` is the
export's first val row; the given family is stored under `--name`. The threshold curve and the
all-on point with `--all-on`.
"""

import argparse
import fcntl
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("VPD_MB", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from vpd_eval import MB, gates_and_l0, kl_per_pos, pgd_recon  # noqa: E402
from vpd_model import VPD, VPD_PTH, load_target, load_vpd, site_names, val_tokens  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["box", "layers"])
parser.add_argument("sets", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--rows", type=int, default=32)
parser.add_argument("--offset", type=int, default=1024)
parser.add_argument("--draws", type=int, default=64)
parser.add_argument("--steps", default="20,40,80")
parser.add_argument("--key", default=None)
parser.add_argument("--families", default="vpd_ci,vpd_rounded,given")
parser.add_argument("--all-on", action="store_true", help="also every subcomponent on (box)")
parser.add_argument("--parts", default="fixed,draws,pgd", help="which parts of box to run")
parser.add_argument("--delta", action="store_true", help="also VPD's residual semantics (box; it moves the KL by <1%%)")
parser.add_argument("--name", default="ours", help="the given sets' family name")
parser.add_argument("--library", type=Path, default=Path.home() / "mpd-data/pieces/vpd4l_library")
parser.add_argument("--gates", type=Path, default=None,
                    help="VPD's gates on these rows (written by the first run, read by the rest: the CI network then never loads)")
args = parser.parse_args()
DEV = "mps"
t0 = time.time()


def log(msg: str):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    print(f"[{time.time() - t0:6.0f}s rss {rss:.1f}G] {msg}", flush=True)


target = load_target(DEV)
if args.gates is None:
    args.gates = Path.home() / f"mpd-data/frontier/vpd4l_gates_{args.offset}_{args.rows}.npz"


def subcomponents_only() -> VPD:
    """VPD's subcomponents installed in the target, without the CI network."""
    raw = torch.load(str(VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    uv = {}
    for k, v in raw.items():
        if k.startswith("_components."):
            site, which = k[len("_components."):].rsplit(".", 1)
            uv.setdefault(site.replace("-", "."), {})[which] = v.float().to(DEV)
    del raw
    return VPD(target, None, {n: (uv[n]["U"], uv[n]["V"]) for n in site_names()})


wanted = args.families.split(",")
need_gates = "vpd_ci" in wanted or "vpd_rounded" in wanted
vpd = load_vpd(target, DEV) if need_gates and not args.gates.exists() else subcomponents_only()
names = site_names()
ids = val_tokens(args.rows, offset=args.offset).to(DEV)
S = ids.shape[1]
n_mb = args.rows // MB
assert n_mb * MB == args.rows, "rows must be a multiple of VPD_MB"


class Family:
    """Per microbatch, per site, the lower bound g as (flat index, value) of its nonzeros."""

    def __init__(self, mb):
        self.mb = mb

    def dense(self, i: int) -> dict[str, torch.Tensor]:
        out = {}
        for n, (idx, val, shape) in self.mb[i].items():
            z = torch.zeros(int(np.prod(shape)), device=DEV)
            z[idx] = val
            out[n] = z.view(shape)
        return out

    def l0(self) -> float:
        return sum(int(idx.numel()) for d in self.mb for idx, _, _ in d.values()) / (args.rows * S)


def given_family() -> Family:
    """The driver's CSR sets in VPD's site order."""
    indptr = np.load(f"{args.sets}.indptr.npy")
    indices = np.load(f"{args.sets}.indices.npy")
    offsets = np.load(f"{args.sets}.offsets.npy")
    manifest = json.load(open(args.library / "manifest.json"))
    driver_sites = sorted(manifest)
    assert len(offsets) == len(driver_sites) + 1, "offsets do not number the library's sites"
    assert len(indptr) - 1 >= args.rows * S, f"{len(indptr) - 1} positions for {args.rows} rows"
    vpd_of = {n: manifest[n]["vpd"] for n in driver_sites}
    mbs = []
    for i in range(n_mb):
        lo, hi = indptr[i * MB * S], indptr[(i + 1) * MB * S]
        flat = indices[lo:hi]
        pos = np.repeat(np.arange(MB * S), np.diff(indptr[i * MB * S:(i + 1) * MB * S + 1]))
        site = np.searchsorted(offsets, flat, side="right") - 1
        d = {}
        for k, dn in enumerate(driver_sites):
            n = vpd_of[dn]
            C = vpd.C[n]
            assert offsets[k + 1] - offsets[k] == C, f"{dn}: {offsets[k + 1] - offsets[k]} pieces, VPD {C}"
            sel = site == k
            idx = torch.tensor(pos[sel] * C + (flat[sel] - offsets[k]), dtype=torch.long, device=DEV)
            d[n] = (idx, torch.ones(idx.numel(), device=DEV), torch.Size((MB, S, C)))
        mbs.append(d)
    return Family(mbs)


def transformed(fam: Family, f) -> Family:
    """A family whose nonzero values are f(values), zeros dropped."""
    mbs = []
    for d in fam.mb:
        e = {}
        for n, (idx, val, shape) in d.items():
            v = f(val)
            keep = v > 0
            e[n] = (idx[keep], v[keep], shape)
        mbs.append(e)
    return Family(mbs)


if not need_gates:
    ci_mb = None
elif args.gates.exists():
    stored = np.load(args.gates)
    ci_mb = []
    for i in range(n_mb):
        d = {}
        for n in names:
            flat, val = stored[f"{n}:index"], stored[f"{n}:value"]
            span = MB * S * vpd.C[n]
            sel = (flat >= i * span) & (flat < (i + 1) * span)
            d[n] = (torch.tensor(flat[sel] - i * span, device=DEV), torch.tensor(val[sel], device=DEV),
                    torch.Size((MB, S, vpd.C[n])))
        ci_mb.append(d)
else:
    ci, _ = gates_and_l0(vpd, ids)
    ci_mb = ci.mb
    # Only the gates are needed: the CI network (0.54B parameters) leaves memory.
    del vpd.ci_fn
    torch.mps.empty_cache()
    np.savez(args.gates, **{
        f"{n}:{part}": np.concatenate([(d[n][0].cpu().numpy() + i * MB * S * vpd.C[n]) if part == "index" else d[n][1].cpu().numpy()
                                       for i, d in enumerate(ci_mb)])
        for n in names for part in ("index", "value")})
    log(f"gates written to {args.gates}")
families = {}
if "vpd_ci" in wanted:
    families["vpd_ci"] = Family(ci_mb)
if "vpd_rounded" in wanted:
    families["vpd_rounded"] = transformed(Family(ci_mb), lambda v: (v > 0).float())
if "given" in wanted:
    families[args.name] = given_family()
log("families: " + ", ".join(f"{k} L0 {f.l0():.1f}" for k, f in families.items()))


class Targets:
    """Native logits of one microbatch at a time (every microbatch's would be 3 GB on 32 rows)."""

    def __init__(self):
        self.i, self.logits = None, None

    def __getitem__(self, i: int) -> torch.Tensor:
        if self.i != i:
            self.logits = None
            with torch.no_grad():
                self.i, self.logits = i, vpd.target_forward(ids[i * MB:(i + 1) * MB])
        return self.logits


targets = Targets()


def set_masks(masks: dict, delta: dict | None):
    """Install masks; a site absent from `masks` runs native."""
    vpd.clear()
    for n in names:
        st = target.site(n)
        st.mask = masks.get(n)
        st.delta_mask = None if delta is None or n not in masks else delta[n]


@torch.no_grad()
def logits_with(i: int, masks: dict, delta: dict | None = None) -> torch.Tensor:
    set_masks(masks, delta)
    try:
        return target(ids[i * MB:(i + 1) * MB])
    finally:
        vpd.clear()


def kl_rows(i: int, masks: dict, delta: dict | None = None) -> torch.Tensor:
    kl = kl_per_pos(logits_with(i, masks, delta), targets[i])
    torch.mps.empty_cache()
    return kl


def zero_delta(g: dict) -> dict:
    return {n: torch.zeros(v.shape[:-1], device=DEV) for n, v in g.items()}


def stats(kl: np.ndarray) -> dict:
    return {"mean": float(kl.mean()), "median": float(np.median(kl)), "p95": float(np.quantile(kl, 0.95)),
            "max": float(kl.max())}


def update_out(key: str, value: dict):
    # Runs of other modes and families share the file: read-modify-write under a lock.
    with open(args.out.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        merge_out(key, value)


def merge_out(key: str, value: dict):
    data = json.load(open(args.out)) if args.out.exists() else {}
    # One level deep, so runs over other families add to what is there.
    for k, v in value.items():
        if isinstance(v, dict) and isinstance(data.setdefault(key, {}).get(k), dict):
            data[key][k].update(v)
        else:
            data.setdefault(key, {})[k] = v
    data[key]["rows"], data[key]["offset"] = args.rows, args.offset
    if args.name in families:
        data[key].setdefault("sets", {})[args.name] = str(args.sets)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(args.out)


def box():
    key = args.key or "box"
    steps = [int(s) for s in args.steps.split(",")]
    # The fixed points: each family at its lower bound, VPD's thresholds, and every subcomponent on.
    parts = args.parts.split(",")
    fixed = {}
    with torch.no_grad():
        for name, fam in families.items():
            fixed[name] = np.concatenate([kl_rows(i, fam.dense(i), zero_delta(fam.dense(i))).cpu().numpy() for i in range(n_mb)])
        for tau in ((0.1, 0.5) if "vpd_ci" in families else ()):
            fam = transformed(families["vpd_ci"], lambda v, tau=tau: (v > tau).float())
            fixed[f"vpd_gt_{tau}"] = np.concatenate([kl_rows(i, fam.dense(i), zero_delta(fam.dense(i))).cpu().numpy() for i in range(n_mb)])
            log(f"threshold {tau}: L0 {fam.l0():.1f} KL {fixed[f'vpd_gt_{tau}'].mean():.4f}")
        if args.all_on:
            ones = {n: torch.ones((MB, S, vpd.C[n]), device=DEV) for n in names}
            fixed["all_on"] = np.concatenate([kl_rows(i, ones, zero_delta(ones)).cpu().numpy() for i in range(n_mb)])
            del ones
    log("fixed: " + ", ".join(f"{k} {v.mean():.4f}" for k, v in fixed.items()))
    update_out(key, {"fixed_kl": {k: stats(v) for k, v in fixed.items()},
                     "l0": {k: f.l0() for k, f in families.items()}})
    # Uniform draws over the box, the same U per (draw, microbatch) for every family.
    draws = {}
    for delta_mode in () if "draws" not in parts else (("off", "uniform") if args.delta else ("off",)):
        for name, fam in families.items():
            kl = np.zeros((args.draws, args.rows * S))
            with torch.no_grad():
                for i in range(n_mb):
                    g = fam.dense(i)
                    for d in range(args.draws):
                        gen = torch.Generator(device=DEV).manual_seed(1_000_003 * d + i)
                        # Sites in one fixed order, so every family sees the same draws.
                        m = {n: g[n] + (1 - g[n]) * torch.rand(g[n].shape, generator=gen, device=DEV) for n in names}
                        delta = (zero_delta(g) if delta_mode == "off"
                                 else {n: torch.rand(g[n].shape[:-1], generator=gen, device=DEV) for n in names})
                        kl[d, i * MB * S:(i + 1) * MB * S] = kl_rows(i, m, delta).cpu().numpy().ravel()
                        del m, delta
                    del g
                    torch.mps.empty_cache()
            worst = kl.max(0)
            draws[f"{name}/delta_{delta_mode}"] = {
                "all_draws": stats(kl.ravel()), "worst_of_draws_per_word": stats(worst),
                "worst_over_lower_bound_mean": float(worst.mean() / max(fixed[name].mean(), 1e-12)),
            }
            log(f"draws {name} delta {delta_mode}: mean {kl.mean():.4f} worst-of-{args.draws} mean {worst.mean():.4f}")
            update_out(key, {"draws": draws, "draws_per_word": args.draws})
    # VPD's eval adversary: one shared source per subcomponent (plus one delta coordinate).
    pgd = {}
    for with_delta in () if "pgd" not in parts else ((False, True) if args.delta else (False,)):
        for name, fam in families.items():
            ladder = pgd_recon(vpd, ids, fam, steps, step_size=0.1, with_delta=with_delta, seed=0)
            pgd[f"{name}/delta_{'adversarial' if with_delta else 'off'}"] = {str(k): v for k, v in ladder.items()}
            log(f"pgd {name} delta {with_delta}: {ladder}")
            update_out(key, {"pgd_shared": pgd, "pgd_step_size": 0.1})
            torch.mps.empty_cache()


@torch.no_grad()
def layers():
    key = args.key or "layers"
    out = {}
    n_layer = 4
    layer_sites = [[n for n in names if n.startswith(f"h.{l}.")] for l in range(n_layer)]
    for name, fam in families.items():
        only = np.zeros((n_layer, args.rows * S))
        joint = np.zeros(args.rows * S)
        tv_step = np.zeros((n_layer, args.rows * S))
        tv_end = np.zeros(args.rows * S)
        for i in range(n_mb):
            g = fam.dense(i)
            sl = slice(i * MB * S, (i + 1) * MB * S)
            for l in range(n_layer):
                m = {n: g[n] for n in layer_sites[l]}
                only[l, sl] = kl_rows(i, m, zero_delta(m)).cpu().numpy().ravel()
            # Hybrids H_0 .. H_4: H_l masks layers < l.
            probs = [targets[i].softmax(-1)]
            for l in range(1, n_layer + 1):
                m = {n: g[n] for k in range(l) for n in layer_sites[k]}
                lg = logits_with(i, m, zero_delta(m))
                if l == n_layer:
                    joint[sl] = kl_per_pos(lg, targets[i]).cpu().numpy().ravel()
                p = lg.softmax(-1)
                del lg
                tv_step[l - 1, sl] = (0.5 * (p - probs[-1]).abs().sum(-1)).cpu().numpy().ravel()
                probs = [probs[0], p]
            tv_end[sl] = (0.5 * (probs[-1] - probs[0]).abs().sum(-1)).cpu().numpy().ravel()
            del probs, g
            torch.mps.empty_cache()
        step_sum = tv_step.sum(0)
        out[name] = {
            "kl_only_layer": [float(x) for x in only.mean(1)],
            "kl_joint": float(joint.mean()),
            "kl_joint_to_sum": float(joint.mean() / only.mean(1).sum()),
            "kl_joint_to_sum_per_word_median": float(np.median(joint / np.maximum(only.sum(0), 1e-12))),
            "tv_hybrid_step": [float(x) for x in tv_step.mean(1)],
            "tv_end_to_end": float(tv_end.mean()),
            "tv_end_to_sum": float(tv_end.mean() / step_sum.mean()),
            "tv_end_to_sum_per_word_median": float(np.median(tv_end / np.maximum(step_sum, 1e-12))),
        }
        log(f"layers {name}: {out[name]}")
        update_out(key, out)


box() if args.mode == "box" else layers()
log("done")
