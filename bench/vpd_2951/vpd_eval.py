"""Matched evaluation harness on the 4L Pile target, reproducing VPD's eval metrics
(param_decomp/metrics/{ce_and_kl_losses,ci_l0,pgd_utils}.py at the vpd-paper tag).

An explanation exposes, per site, rank-one factors U [C, d_out], V [d_in, C], and a gate
function giving each factor's causal importance g in [0, 1] per (batch, pos). The ablation
family is VPD's: mask = g + (1 - g) * s, with s in [0,1]^C chosen by an adversary. The eval
adversary (PGDReconLoss, init random, source_shape c) holds ONE source vector per site shared
across the whole batch and every position, and takes n_steps of sign-gradient ascent with
step_size 0.1 on the batch-mean per-position KL(target || masked), clamping s to [0, 1].

Batches of 128 x 512 do not fit on MPS at once (logits alone are 13 GB), so every quantity is
accumulated over row-microbatches: the batch-mean KL is the mean of equal-size microbatch
means, and its gradient w.r.t. the shared source is the mean of the microbatch gradients.
Gates are stored sparsely (VPD's are exactly zero outside ~200 of 38912 per position).
"""

import torch
import torch.nn.functional as F
from torch import Tensor

MB = int(__import__("os").environ.get("VPD_MB", "8"))  # rows per microbatch (VPD_MB overrides; the footprint scales with it)


def kl_per_pos(logits: Tensor, target_logits: Tensor, rows: int = 2) -> Tensor:
    """KL(softmax(target) || softmax(pred)) per position, [B, S], a few rows at a time to bound
    the vocab-sized temporaries."""
    out = []
    for i in range(0, logits.shape[0], rows):
        lp = F.log_softmax(target_logits[i:i + rows], -1)
        out.append((lp.exp() * (lp - F.log_softmax(logits[i:i + rows], -1))).sum(-1))
    return torch.cat(out)


def ce_rows(logits: Tensor, ids: Tensor) -> Tensor:
    """Within-row next-token CE summed over the S-1 predicted positions, per row."""
    return F.cross_entropy(logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten(), reduction="none").view(ids.shape[0], -1).sum(1)


class SparseGates:
    """Per-microbatch, per-site gates stored as (flat index, value) of their nonzeros. MPS
    nonzero returns out-of-range indices past 2^24 elements and a 2-D index_put of that size
    segfaults (torch 2.11), so a microbatch of one site must stay below 2^24 entries
    (8 x 512 x 3584 = 14.7M does)."""

    def __init__(self):
        self.mb: list[dict[str, tuple[Tensor, Tensor, torch.Size]]] = []

    def add(self, g: dict[str, Tensor]):
        d = {}
        for n, v in g.items():
            assert v.numel() < 1 << 24, (n, v.shape)
            flat = v.reshape(-1)
            # MPS nonzero intermittently returns out-of-range indices: take the support on CPU
            idx = flat.cpu().nonzero().squeeze(-1).to(flat.device)
            d[n] = (idx, flat[idx], v.shape)
        self.mb.append(d)

    def dense(self, i: int) -> dict[str, Tensor]:
        out = {}
        for n, (idx, val, shape) in self.mb[i].items():
            z = torch.zeros(shape.numel(), device=val.device, dtype=val.dtype)
            z[idx] = val
            out[n] = z.view(shape)
        return out


@torch.no_grad()
def gates_and_l0(expl, ids: Tensor) -> tuple[SparseGates, dict[str, float]]:
    """Gates for every microbatch plus mean L0 (count of g > 0) per site."""
    sg, l0 = SparseGates(), {n: 0.0 for n in expl.names}
    n_mb = ids.shape[0] // MB
    for i in range(n_mb):
        _, g = expl.target_and_ci(ids[i * MB:(i + 1) * MB])
        for n, v in g.items():
            l0[n] += (v > 0).float().sum(-1).mean().item() / n_mb
        sg.add(g)
        if i == 0:
            back = sg.dense(0)
            assert all(torch.equal(back[n], g[n]) for n in g), "sparse gate round-trip failed"
        del g
    return sg, l0


@torch.no_grad()
def ce_kl_battery(expl, ids: Tensor, sg: SparseGates, seed: int = 0, with_delta: bool = True,
                  strategies: tuple[str, ...] = ("ci_masked", "unmasked", "stoch_masked", "random_masked",
                                                 "rounded_masked", "zero_masked")) -> dict[str, float]:
    """VPD's CEandKLLosses: KL vs target and CE difference for six mask strategies. Only
    stoch_masked routes the delta component (U(0,1) per position); the rest set delta_mask 0.
    with_delta=False drops the delta component from the program everywhere."""
    gen = torch.Generator(device=ids.device).manual_seed(seed)
    rand = lambda t: torch.rand(t.shape, generator=gen, device=t.device)
    strategies = list(strategies)
    kl = {s: 0.0 for s in strategies}
    ce = {s: 0.0 for s in strategies + ["target"]}
    n_mb, n_pos = ids.shape[0] // MB, ids.shape[0] * (ids.shape[1] - 1)
    for i in range(n_mb):
        b = ids[i * MB:(i + 1) * MB]
        g = sg.dense(i)
        tgt = expl.target_forward(b)
        ce["target"] += ce_rows(tgt, b).sum().item() / n_pos
        z = None if not with_delta else {n: torch.zeros(v.shape[:-1], device=v.device) for n, v in g.items()}
        for s in strategies:
            if s == "ci_masked":
                m, d = g, z
            elif s == "unmasked":
                m, d = {n: torch.ones_like(v) for n, v in g.items()}, z
            elif s == "stoch_masked":
                m = {n: v + (1 - v) * rand(v) for n, v in g.items()}
                d = {n: rand(v[..., 0]) for n, v in g.items()} if with_delta else None
            elif s == "random_masked":
                m, d = {n: rand(v) for n, v in g.items()}, z
            elif s == "rounded_masked":
                m, d = {n: (v > 0).float() for n, v in g.items()}, z
            else:
                m, d = {n: torch.zeros_like(v) for n, v in g.items()}, z
            lg = expl.masked(b, m, d)
            kl[s] += kl_per_pos(lg, tgt).mean().item() / n_mb
            ce[s] += ce_rows(lg, b).sum().item() / n_pos
            del lg, m
        del g, tgt
    out = {f"kl_{s}": kl[s] for s in strategies}
    out.update({f"ce_difference_{s}": ce[s] - ce["target"] for s in strategies})
    out["ce_target"] = ce["target"]
    return out


def pgd_recon(expl, ids: Tensor, sg: SparseGates, n_steps_list: list[int], step_size: float = 0.1,
              with_delta: bool = True, seed: int = 0, on_rung=None, state_path: str | None = None) -> dict[int, float]:
    """Fresh shared-across-batch PGD (init random, source_shape c). Returns the batch-mean KL
    after each n in n_steps_list (one trajectory; the ladder shares its init as in VPD's
    SlowPGDReconLoss). with_delta=False drops the delta component from the program (its mask
    is fixed at 0 and the adversary has no delta coordinate)."""
    gen = torch.Generator(device=ids.device).manual_seed(seed)
    src = {n: torch.rand((expl.C[n] + int(with_delta),), generator=gen, device=ids.device).requires_grad_(True)
           for n in expl.names}
    n_mb = ids.shape[0] // MB

    def loss_and_grad(need_grad: bool) -> float:
        total = 0.0
        grads = {n: torch.zeros_like(s) for n, s in src.items()}
        for i in range(n_mb):
            b = ids[i * MB:(i + 1) * MB]
            g = sg.dense(i)
            with torch.no_grad():
                tgt = expl.target_forward(b)
            with torch.enable_grad():
                m = {n: v + (1 - v) * src[n][: expl.C[n]] for n, v in g.items()}
                d = ({n: src[n][-1].expand(v.shape[:-1]) for n, v in g.items()} if with_delta else None)
                loss = kl_per_pos(expl.masked(b, m, d), tgt).mean() / n_mb
                if need_grad:
                    gs = torch.autograd.grad(loss, list(src.values()))
                    for n, gg in zip(src, gs, strict=True):
                        grads[n] += gg
            total += loss.item()
            del g, tgt, m, loss
        if need_grad:
            with torch.no_grad():
                for n in src:
                    src[n].add_(step_size * grads[n].sign()).clamp_(0.0, 1.0)
        return total

    out, want, first = {}, sorted(set(n_steps_list)), 1
    if state_path is not None and __import__("os").path.exists(state_path):
        # resume a killed run: the source vector after `step` sign steps and the rungs so far
        st = torch.load(state_path)
        with torch.no_grad():
            for n in src:
                src[n].copy_(st["src"][n].to(src[n].device))
        out, first = {int(k): v for k, v in st["out"].items()}, st["step"] + 1
    elif 0 in want:
        out[0] = loss_and_grad(False)
    for step in range(first, want[-1] + 1):
        loss_and_grad(True)
        if step in want:
            out[step] = loss_and_grad(False)
            if on_rung is not None:
                on_rung(dict(out))
        if step % 5 == 0 and state_path is not None:
            torch.save({"step": step, "out": out, "src": {n: v.detach().cpu() for n, v in src.items()}},
                       state_path + ".tmp")
            __import__("os").replace(state_path + ".tmp", state_path)
        if step % 10 == 0 and ids.device.type == "mps":
            torch.mps.empty_cache()
    return out


def _masked_capture(expl, ids: Tensor, masks, delta_masks) -> tuple[Tensor, dict[str, Tensor]]:
    """Masked forward that also returns every site's output (for hidden-acts recon and the
    attention patterns). masks=None runs the target."""
    tgt = expl.target
    expl.clear()
    for n in expl.names:
        st = tgt.site(n)
        st.cache_output = True
        if masks is not None:
            st.mask = masks[n]
            st.delta_mask = None if delta_masks is None else delta_masks[n]
    try:
        logits = tgt(ids)
        outs = {n: tgt.site(n).last_output for n in expl.names}
        return logits, outs
    finally:
        expl.clear()


def _attn_kl_sum(target_model, t_outs, m_outs, n_layer: int = 4) -> tuple[list[float], int]:
    """Sum over (b, h, t_query) of KL(target pattern || masked pattern), per layer, VPD's
    CIMaskedAttnPatternsReconLoss numerator (kl_div with masked.clamp(1e-12).log())."""
    sums = []
    n = 0
    for i in range(n_layer):
        q, k = f"h.{i}.attn.q_proj", f"h.{i}.attn.k_proj"
        tot = 0.0
        for r in range(t_outs[q].shape[0]):
            pt = target_model.attention_pattern(t_outs[q][r:r + 1], t_outs[k][r:r + 1])
            pm = target_model.attention_pattern(m_outs[q][r:r + 1], m_outs[k][r:r + 1])
            tot += F.kl_div(pm.clamp(1e-12).log(), pt, reduction="sum").item()
            if i == 0:
                n += pt.shape[1] * pt.shape[2]
            del pt, pm
        sums.append(tot)
    return sums, n


@torch.no_grad()
def full_battery(expl, ids: Tensor, sg: SparseGates, seed: int = 0) -> dict:
    """VPD's standard eval table on one eval batch: per mask strategy, KL / CE difference / top-1
    agreement with the target's argmax; for ci_masked and stoch_masked also the per-site
    hidden-acts MSE (CIHiddenActsReconLoss / StochasticHiddenActsReconLoss) and the
    attention-pattern KL (CIMasked/StochasticAttnPatternsReconLoss). Delta handling as in
    CEandKLLosses: only stoch_masked routes the delta component."""
    gen = torch.Generator(device=ids.device).manual_seed(seed)
    rand = lambda t: torch.rand(t.shape, generator=gen, device=t.device)
    strategies = ["ci_masked", "unmasked", "stoch_masked", "random_masked", "rounded_masked", "zero_masked"]
    rich = {"ci_masked", "stoch_masked"}
    kl = {s: 0.0 for s in strategies}
    top1 = {s: 0.0 for s in strategies}
    ce = {s: 0.0 for s in strategies + ["target"]}
    mse_sum = {s: {n: 0.0 for n in expl.names} for s in rich}
    attn_sum = {s: [0.0] * 4 for s in rich}
    attn_n = 0
    n_mb, n_pos = ids.shape[0] // MB, ids.shape[0] * (ids.shape[1] - 1)
    for i in range(n_mb):
        b = ids[i * MB:(i + 1) * MB]
        g = sg.dense(i)
        tgt, t_outs = _masked_capture(expl, b, None, None)
        t_top = tgt.argmax(-1)
        ce["target"] += ce_rows(tgt, b).sum().item() / n_pos
        z = {n: torch.zeros(v.shape[:-1], device=v.device) for n, v in g.items()}
        for s in strategies:
            if s == "ci_masked":
                m, d = g, z
            elif s == "unmasked":
                m, d = {n: torch.ones_like(v) for n, v in g.items()}, z
            elif s == "stoch_masked":
                m = {n: v + (1 - v) * rand(v) for n, v in g.items()}
                d = {n: rand(v[..., 0]) for n, v in g.items()}
            elif s == "random_masked":
                m, d = {n: rand(v) for n, v in g.items()}, z
            elif s == "rounded_masked":
                m, d = {n: (v > 0).float() for n, v in g.items()}, z
            else:
                m, d = {n: torch.zeros_like(v) for n, v in g.items()}, z
            if s in rich:
                lg, m_outs = _masked_capture(expl, b, m, d)
                for n in expl.names:
                    mse_sum[s][n] += F.mse_loss(m_outs[n], t_outs[n]).item() / n_mb
                sums, nd = _attn_kl_sum(expl.target, t_outs, m_outs)
                for li in range(4):
                    attn_sum[s][li] += sums[li]
                if s == "ci_masked":
                    attn_n += nd
                del m_outs
            else:
                lg = expl.masked(b, m, d)
            kl[s] += kl_per_pos(lg, tgt).mean().item() / n_mb
            ce[s] += ce_rows(lg, b).sum().item() / n_pos
            top1[s] += (lg.argmax(-1) == t_top).float().mean().item() / n_mb
            del lg, m
        del g, tgt, t_outs
        torch.mps.empty_cache()
    out = {f"kl_{s}": kl[s] for s in strategies}
    out.update({f"ce_difference_{s}": ce[s] - ce["target"] for s in strategies})
    out.update({f"top1_agree_{s}": top1[s] for s in strategies})
    out["ce_target"] = ce["target"]
    for s, tag in (("ci_masked", "CI"), ("stoch_masked", "Stochastic")):
        per = mse_sum[s]
        out[f"{tag}HiddenActsReconLoss/sites"] = per
        out[f"{tag}HiddenActsReconLoss/mean_over_sites"] = sum(per.values()) / len(per)
        out[f"{tag}AttnPatternsReconLoss/layers"] = [x / attn_n for x in attn_sum[s]]
        out[f"{tag}AttnPatternsReconLoss"] = sum(attn_sum[s]) / (4 * attn_n)
    return out


@torch.no_grad()
def ci_usage(expl, ids: Tensor, rows: int = MB) -> dict[str, tuple[Tensor, Tensor]]:
    """Per site: (sum of CI over tokens, count of tokens with CI > 0) per subcomponent."""
    acc: dict[str, list[Tensor]] = {}
    for i in range(0, ids.shape[0], rows):
        _, g = expl.target_and_ci(ids[i:i + rows])
        for n, v in g.items():
            s, c = v.sum((0, 1)).cpu().double(), (v > 0).sum((0, 1)).cpu().double()
            if n in acc:
                acc[n][0] += s
                acc[n][1] += c
            else:
                acc[n] = [s, c]
        del g
    return {n: (a[0].cpu(), a[1].cpu()) for n, a in acc.items()}


def ppgd_recon(expl, ids: Tensor, n_steps_list: list[int], lr: float = 0.01, betas=(0.5, 0.99),
               eps: float = 1e-8, rows: int = 2, seed: int = 0) -> dict[int, float]:
    """PPGD-scope adversary at eval: fresh per-(batch, position, c) sources (plus a delta source
    per position), random init, VPD's persistent-PGD optimizer (Adam, lr 0.01, betas 0.5/0.99,
    projected to [0,1]) ascending the per-position KL. Sources are per row, so rows are
    independent and are processed `rows` at a time (bounded memory). Returns the mean KL after
    each n in n_steps_list."""
    want = sorted(set(n_steps_list))
    out = {n: 0.0 for n in want}
    gen = torch.Generator(device=ids.device).manual_seed(seed)
    n_chunks = ids.shape[0] // rows
    for ci in range(n_chunks):
        b = ids[ci * rows:(ci + 1) * rows]
        tgt, g = expl.target_and_ci(b)
        B, S = b.shape
        src = {n: torch.rand((B, S, expl.C[n] + 1), generator=gen, device=b.device).requires_grad_(True)
               for n in expl.names}
        m1 = {n: torch.zeros_like(s) for n, s in src.items()}
        m2 = {n: torch.zeros_like(s) for n, s in src.items()}

        def loss_fn():
            m = {n: g[n] + (1 - g[n]) * src[n][..., :-1] for n in expl.names}
            d = {n: src[n][..., -1] for n in expl.names}
            return kl_per_pos(expl.masked(b, m, d), tgt).mean()

        if 0 in want:
            with torch.no_grad():
                out[0] += loss_fn().item() / n_chunks
        for t in range(1, want[-1] + 1):
            with torch.enable_grad():
                loss = loss_fn()
                grads = torch.autograd.grad(loss, list(src.values()))
            with torch.no_grad():
                for (n, s), gr in zip(src.items(), grads, strict=True):
                    m1[n].mul_(betas[0]).add_(gr, alpha=1 - betas[0])
                    m2[n].mul_(betas[1]).addcmul_(gr, gr, value=1 - betas[1])
                    s.add_(lr * (m1[n] / (1 - betas[0] ** t)) / ((m2[n] / (1 - betas[1] ** t)).sqrt() + eps))
                    s.clamp_(0.0, 1.0)
            del loss, grads
            if t in want:
                with torch.no_grad():
                    out[t] += loss_fn().item() / n_chunks
        del src, m1, m2, g, tgt
    return out
