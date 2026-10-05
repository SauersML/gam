"""The common evaluation battery (#2951) applied to the VPD decomposition of the 4-layer Pile model
(goodfire/spd/runs/s-55ea3f9b) in its intended setting: the causal-importance network reads M's own
activations on the input and its values g in [0, 1] set the masks. The battery for library
explanations is examples/mpd_battery_2951.rs; both score the same held-out rows with the same
definitions.

Every divergence is KL(M || E) of the next-token distributions per token, in nats, E the explanation
run in place of M. Per-token quantiles are over all tokens of the rows; cross-entropy is over the
511 predicted positions of each 512-token row.

Masks (VPD's Table 3; only the stochastic masks route the delta component):
    ci          mask = g
    rounded     mask = 1 where g > 0
    stochastic  mask = g + (1 - g) u, delta mask u, u ~ U(0, 1) per entry
    unmasked    mask = 1

Protocols (VPD's appendix B.1), with the CI and the rounded masks, at layer granularity:
    error-propagating  every layer replaced, each reading the explanation's own stream
    clean-input        every layer replaced, each reading M's stream entering it; the final stream
                       is M's embedding plus every replaced layer's increment on M's stream
    single-layer       one layer replaced, M elsewhere
    subsets k          every subset of k layers replaced (the exact mean over a uniform subset)

usage: vpd_battery.py OUT.json FIRST:END [PAPER_ROWS PGD_STEPS]
    FIRST:END    held-out val rows of the matched table (vpd4l_frontier32 is 1024:1056)
    PAPER_ROWS   rows 0:PAPER_ROWS for the check against VPD's reported numbers (CE, L0, and
                 PGD_STEPS steps of shared-across-batch PGD, step 0.1, as in VPD's eval)
"""

import itertools
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from vpd_model import gelu_tanh, load_target, load_vpd, rms, val_tokens

MB = 8  # rows per microbatch: the CI values of 8 x 512 tokens and the logits fit in MPS memory
QUANTILES = (0.5, 0.9, 0.99, 1.0)
DEVICE = "mps"


def kl_tokens(logits: Tensor, target: Tensor) -> Tensor:
    """KL(softmax(target) || softmax(logits)) per token [B, T], two rows at a time."""
    out = []
    for i in range(0, logits.shape[0], 2):
        lp = F.log_softmax(target[i:i + 2], -1)
        out.append((lp.exp() * (lp - F.log_softmax(logits[i:i + 2], -1))).sum(-1))
    return torch.cat(out)


def ce_tokens(logits: Tensor, ids: Tensor) -> Tensor:
    """Next-token cross-entropy per predicted position [B, T-1]."""
    return F.cross_entropy(logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten(), reduction="none").view(ids.shape[0], -1)


def summary(values: list[Tensor]) -> dict:
    v = torch.cat([x.flatten().cpu().double() for x in values]).numpy()
    out = {"mean": float(v.mean()), "tokens": int(v.size)}
    out.update({f"q{int(round(q * 100))}": float(np.quantile(v, q)) for q in QUANTILES})
    return out


class Runner:
    """M and the decomposition, run layer by layer so that any set of layers takes the masks."""

    def __init__(self, vpd):
        self.vpd, self.t = vpd, vpd.target

    def layer_sites(self, i: int) -> list[str]:
        return [n for n in self.vpd.names if n.startswith(f"h.{i}.")]

    def layer(self, x: Tensor, i: int, masks: dict | None, deltas: dict | None) -> Tensor:
        """Layer i on the stream x entering it, with the masks on its sites (M's weights when none)."""
        t = self.t
        B, T, _ = x.shape
        if masks is not None:
            for n in self.layer_sites(i):
                st = t.site(n)
                st.mask = masks[n]
                st.delta_mask = None if deltas is None else deltas[n]
        try:
            s = lambda k: t.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")
            h = rms(x, t.norms[2 * i], t.eps)
            q = s("q_proj")(h).view(B, T, t.n_head, t.hd).transpose(1, 2)
            k = s("k_proj")(h).view(B, T, t.n_head, t.hd).transpose(1, 2)
            v = s("v_proj")(h).view(B, T, t.n_head, t.hd).transpose(1, 2)
            q, k = t._rope(q, T), t._rope(k, T)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + s("o_proj")(y.transpose(1, 2).reshape(B, T, -1))
            h = rms(x, t.norms[2 * i + 1], t.eps)
            return x + s("down_proj")(gelu_tanh(s("c_fc")(h)))
        finally:
            self.vpd.clear()

    def logits(self, x: Tensor) -> Tensor:
        return rms(x, self.t.ln_f, self.t.eps) @ self.t.wte.T

    def streams(self, ids: Tensor) -> list[Tensor]:
        """M's stream entering each layer and its final stream."""
        x = self.t.wte[ids]
        out = [x]
        for i in range(self.t.n_layer):
            x = self.layer(x, i, None, None)
            out.append(x)
        return out

    def replaced(self, ids: Tensor, layers: set[int], masks: dict, deltas: dict | None) -> Tensor:
        """Error-propagating with the layers `layers` replaced."""
        x = self.t.wte[ids]
        for i in range(self.t.n_layer):
            x = self.layer(x, i, masks if i in layers else None, deltas)
        return self.logits(x)

    def clean_input(self, m_streams: list[Tensor], masks: dict, deltas: dict | None) -> Tensor:
        x = m_streams[0].clone()
        for i in range(self.t.n_layer):
            x = x + self.layer(m_streams[i], i, masks, deltas) - m_streams[i]
        return self.logits(x)


def mask_sets(g: dict[str, Tensor], gen: torch.Generator) -> dict[str, tuple[dict, dict | None]]:
    rand = lambda v: torch.rand(v.shape, generator=gen, device=v.device)
    zero = {n: torch.zeros(v.shape[:-1], device=v.device) for n, v in g.items()}
    return {
        "ci": (g, zero),
        "rounded": ({n: (v > 0).float() for n, v in g.items()}, zero),
        "stochastic": ({n: v + (1 - v) * rand(v) for n, v in g.items()}, {n: rand(v[..., 0]) for n, v in g.items()}),
        "unmasked": ({n: torch.ones_like(v) for n, v in g.items()}, zero),
    }


@torch.no_grad()
def behaviour_and_protocols(runner: Runner, ids: Tensor, protocols: bool, seed: int) -> dict:
    L = runner.t.n_layer
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    kl, ce, top1 = {}, {}, {}
    ce_m, l0 = [], []
    l0_layer = {i: [] for i in range(L)}
    subsets = {k: [frozenset(s) for s in itertools.combinations(range(L), k)] for k in range(1, L)}
    for b0 in range(0, ids.shape[0], MB):
        b = ids[b0:b0 + MB].to(DEVICE)
        m_logits, g = runner.vpd.target_and_ci(b)
        ce_m.append(ce_tokens(m_logits, b))
        l0.append(sum((v > 0).sum(-1) for v in g.values()))
        for i in range(L):
            l0_layer[i].append(sum((g[n] > 0).sum(-1) for n in runner.layer_sites(i)))
        m_top = m_logits.argmax(-1)
        streams = runner.streams(b) if protocols else None

        def record(key: str, logits: Tensor):
            kl.setdefault(key, []).append(kl_tokens(logits, m_logits))
            ce.setdefault(key, []).append(ce_tokens(logits, b))
            top1.setdefault(key, []).append((logits.argmax(-1) == m_top).float())

        for name, (masks, deltas) in mask_sets(g, gen).items():
            record(f"{name}/error_propagating", runner.replaced(b, set(range(L)), masks, deltas))
            if protocols and name in ("ci", "rounded"):
                record(f"{name}/clean_input", runner.clean_input(streams, masks, deltas))
                for k, sets in subsets.items():
                    for s in sets:
                        record(f"{name}/layers_{''.join(map(str, sorted(s)))}", runner.replaced(b, set(s), masks, deltas))
        del g, m_logits, streams
        torch.mps.empty_cache()
    ce_target = torch.cat([x.flatten() for x in ce_m]).mean().item()
    out = {"ce_target": ce_target, "active_subcomponents_per_token": summary(l0),
           "active_subcomponents_per_token_by_layer": {i: summary(v)["mean"] for i, v in l0_layer.items()}, "rows": {}}
    for key in kl:
        out["rows"][key] = {"kl_nats": summary(kl[key]), "ce": torch.cat([x.flatten() for x in ce[key]]).mean().item(),
                            "top1_agreement": torch.cat([x.flatten() for x in top1[key]]).mean().item()}
    if protocols:
        for name in ("ci", "rounded"):
            for k, sets in subsets.items():
                keys = [f"{name}/layers_{''.join(map(str, sorted(s)))}" for s in sets]
                out["rows"][f"{name}/subsets_{k}"] = {
                    "kl_nats": {"mean": float(np.mean([out["rows"][x]["kl_nats"]["mean"] for x in keys]))},
                    "ce": float(np.mean([out["rows"][x]["ce"] for x in keys])),
                    "top1_agreement": float(np.mean([out["rows"][x]["top1_agreement"] for x in keys])),
                }
    return out


def pgd_shared(runner: Runner, ids: Tensor, steps: int, step_size: float = 0.1, seed: int = 0) -> dict:
    """VPD's eval PGDReconLoss: one source vector per site (and its delta coordinate) shared across
    the batch and every position, random init, `steps` sign-gradient ascent steps of `step_size` on
    the batch-mean KL, clamped to [0, 1]; mask = g + (1 - g) s. Returns the batch-mean KL in nats
    after the last step."""
    vpd = runner.vpd
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    src = {n: torch.rand((vpd.C[n] + 1,), generator=gen, device=DEVICE).requires_grad_(True) for n in vpd.names}
    n_mb = ids.shape[0] // MB
    gates = []
    with torch.no_grad():
        for i in range(n_mb):
            b = ids[i * MB:(i + 1) * MB].to(DEVICE)
            m_logits, g = vpd.target_and_ci(b)
            # stored sparsely (about 205 of 38912 values per token are nonzero), the support found
            # on the host because MPS nonzero is unreliable past 2^24 entries
            sparse = {}
            for n, v in g.items():
                flat = v.reshape(-1)
                idx = flat.cpu().nonzero().squeeze(-1).to(DEVICE)
                sparse[n] = (idx, flat[idx], v.shape)
            gates.append(sparse)
            del m_logits, g

    def sweep(step: bool) -> float:
        total = 0.0
        grads = {n: torch.zeros_like(s) for n, s in src.items()}
        for i in range(n_mb):
            b = ids[i * MB:(i + 1) * MB].to(DEVICE)
            g = {n: torch.zeros(shape.numel(), device=DEVICE).index_put_((idx,), val).view(shape) for n, (idx, val, shape) in gates[i].items()}
            with torch.no_grad():
                target = vpd.target_forward(b)
            with torch.enable_grad():
                m = {n: v + (1 - v) * src[n][:vpd.C[n]] for n, v in g.items()}
                d = {n: src[n][-1].expand(v.shape[:-1]) for n, v in g.items()}
                loss = kl_tokens(vpd.masked(b, m, d), target).mean() / n_mb
                if step:
                    for n, gg in zip(src, torch.autograd.grad(loss, list(src.values())), strict=True):
                        grads[n] += gg
            total += loss.item()
            del g, target, m, d, loss
        if step:
            with torch.no_grad():
                for n in src:
                    src[n].add_(step_size * grads[n].sign()).clamp_(0.0, 1.0)
        torch.mps.empty_cache()
        return total

    for _ in range(steps):
        sweep(True)
    return {"steps": steps, "step_size": step_size, "rows": int(ids.shape[0]), "kl_nats": sweep(False)}


def main():
    out_path = sys.argv[1]
    first, end = map(int, sys.argv[2].split(":"))
    paper_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    pgd_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    t0 = time.time()
    vpd = load_vpd(load_target(DEVICE), DEVICE)
    runner = Runner(vpd)
    result = {"held_out_rows": [first, end], "units": "KL(M || E) per token in nats; CE in nats per predicted token"}

    def save():
        json.dump(result, open(out_path, "w"), indent=1)

    print(f"[{time.time() - t0:.0f}s] loaded", flush=True)
    result["held_out"] = behaviour_and_protocols(runner, val_tokens(end - first, offset=first), True, seed=first)
    save()
    print(f"[{time.time() - t0:.0f}s] held-out battery done", flush=True)
    if paper_rows:
        ids = val_tokens(paper_rows)
        result["paper_check"] = behaviour_and_protocols(runner, ids, False, seed=0)
        save()
        print(f"[{time.time() - t0:.0f}s] paper check done", flush=True)
        if pgd_steps:
            result["paper_check"]["pgd"] = pgd_shared(runner, ids, pgd_steps)
            save()
            print(f"[{time.time() - t0:.0f}s] pgd done", flush=True)


if __name__ == "__main__":
    main()
