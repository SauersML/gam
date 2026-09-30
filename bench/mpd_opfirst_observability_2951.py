"""Residual-stream observability of a decoder as an exact linear system (#2951).

Architecture read from config (bench/mpd_opfirst_decoder_2951.py); weights one tensor at a time in float64,
OV maps kept factored (d x hd times hd x d) and densified only for the owner's time-invariant closure.
Readout C = W_U diag(g_final) (lm_head or the tied embedding, times the final RMSNorm gain; the RMSNorm
normalizer 1/rms(x) is a scalar gain that is excluded, i.e. treated as a fixed positive scale). Transitions
with attention patterns held fixed: for every layer l and routing law r, T = I + A_{l,r}, A_{l,r} the sum over the
law's heads h of
  pre-norm (Qwen3):   A_{l,h} = W_O[:, h] W_V[g(h)] diag(g_in,l)          (input RMSNorm gain folded)
  post-norm (OLMo 2): A_{l,h} = diag(g_post_attn,l) W_O[:, h] W_V[h]     (attention-output RMSNorm gain folded)
In the post-norm case the excluded normaliser 1/rms(attention output) is one positive scalar per token and
layer shared by all heads, so T = I + c A with c > 0 unknown: closures and ranks are exact (they do not depend
on c), while the Gramian weights assume c = 1. MLPs are nonlinear; their read directions [W_gate; W_up]
(times the pre-MLP RMSNorm gain in a pre-norm model; raw in OLMo 2, whose MLP reads the raw residual) are
added as extra readouts in the "+mlp" variant.

Three measurements:
1. rowspace(C) at the state.rs rank rule (the owner's resolved row space through the MPD
   surface, ``linear_state_quotient``), plus relative-threshold effective dimensions.
2. state.rs LinearStateQuotient::close itself (op ``linear_state_quotient``): the orthonormal
   chart of the time-invariant family of all routing laws' OV transports, with its measured quotient bounds (when rowspace(C)
   is already all of d_model the closure is d_model with no computation, and the owner is not called).
3. Causal backward closure O_L = O_{L+1} + sum_r O_{L+1} A_{L,r} (+ MLP reads of layer L) over the layer's
   routing laws r (A_{L,r} = sum of A_{L,h} over the heads of law r: heads with one attention pattern act as one
   transport, and one letter per head over-counts and is not invariant under their cross-head GL), kept as an
   orthonormal chart at a relative band tau, and the weighted observability Gramian
   G_L = sum over words of (R T_w)^T (R T_w) with letters {I, A_{L,r}} per layer, the attention block entering
   as the surface's ``attention`` letter (joint_operators::attention_letters groups the heads): state.rs
   WeightedObservability through the MPD surface (op ``weighted_observability``), whose per-step spectra
   give the decay of how strongly each residual direction is read, at the owner's rank band.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpd_opfirst_linalg_2951 as la  # noqa: E402
from mpd_opfirst_decoder_2951 import Decoder, compact_json  # noqa: E402


def apply(rows, t):
    """rows @ A for a transition kept factored, t = (left, right), A = left @ right."""
    return (rows @ t[0]) @ t[1]


def linear_quotient(readouts, transitions=(), chart=None):
    """state.rs ``LinearStateQuotient`` through the MPD surface (op ``linear_state_quotient``): ``close`` (the
    readouts' resolved row span closed under the transitions) or, given ``chart``, ``measure`` of that chart.
    Returns the chart Q (orthonormal rows) and the report, whose SpectralNormBounds certify the quotient."""
    from gamfit.sae import run_parameter_decomposition

    tensors = {f"readout/{i}": r for i, r in enumerate(readouts)}
    tensors.update({f"transition/{i}": t for i, t in enumerate(transitions)})
    declared = {"kind": "close"}
    if chart is not None:
        tensors["chart"] = chart
        declared = {"kind": "declared", "tensor": "chart"}
    operation = {"kind": "linear_state_quotient", "readouts": [f"readout/{i}" for i in range(len(readouts))],
                 "transitions": [f"transition/{i}" for i in range(len(transitions))], "chart": declared}
    out = run_parameter_decomposition({"schema": "gam.mpd-request", "schema_version": 1, "operation": operation},
                                      tensors)
    return out.arrays["chart"], out.report["result"]


def resolved_rows(matrix):
    """state.rs's resolved row space of one matrix (sigma > max(m, n) eps sigma_max) as orthonormal rows: the
    owner's closed chart of ``matrix`` under no transitions (op ``linear_closed_chart``,
    ``LinearStateQuotient::closed_chart``), without the quotient bounds nothing here reads.
    ``.shape[0]`` is the rank at the eps band."""
    from gamfit.sae import run_parameter_decomposition

    operation = {"kind": "linear_closed_chart", "readouts": ["readout/0"], "transitions": []}
    out = run_parameter_decomposition({"schema": "gam.mpd-request", "schema_version": 1, "operation": operation},
                                      {"readout/0": matrix})
    return out.arrays["chart"]


def relative_rows(matrix, tau):
    """Right singular vectors above tau * sigma_max: a numerical conditioning statement, never a rank."""
    if matrix.shape[0] > 2 * matrix.shape[1]:
        matrix = la.qr(matrix, mode="r")
    _, sigma, vt = la.svd(matrix)
    return vt[: int((sigma > tau * sigma[0]).sum())]


def weighted_observability(steps, candidates=()):
    """state.rs ``WeightedObservability::pull_back`` through the MPD surface (op ``weighted_observability``).

    ``steps`` run forward, each ``(letters, readouts)``: a letter is a dense linear map ``n_{l+1} x n_l`` (a
    residual stream passes the identity as a letter like any other) or an attention block as the surface's
    ``attention`` letter ``(letter, tensors)`` (``Decoder.attention_letter``): the owner groups its heads into
    routing laws and takes one letter per law, never one per head. Readouts read the step's output. Returns the
    report (whose ``attention_laws`` lists each attention letter's laws in step order), the factor ``F``
    (``F^T F = G``), the directions (``G``'s eigenvectors by decreasing weight), every step's spectrum with its
    singular values, and each candidate row space's capture (energy fraction, principal cosines)."""
    from gamfit.sae import run_parameter_decomposition

    tensors, request_steps = {}, []
    for step, (letters, readouts) in enumerate(steps):
        declared = []
        for index, letter in enumerate(letters):
            if isinstance(letter, tuple):
                request, owned = letter
                tensors.update(owned)
                declared.append(request)
            else:
                tensors[f"steps/{step}/letters/{index}"] = letter
                declared.append({"kind": "linear", "tensor": f"steps/{step}/letters/{index}"})
        for index, readout in enumerate(readouts):
            tensors[f"steps/{step}/readouts/{index}"] = readout
        request_steps.append({"letters": declared,
                              "readouts": [f"steps/{step}/readouts/{i}" for i in range(len(readouts))]})
    tensors.update({f"candidates/{i}": candidate for i, candidate in enumerate(candidates)})
    out = run_parameter_decomposition(
        {"schema": "gam.mpd-request", "schema_version": 1,
         "operation": {"kind": "weighted_observability", "steps": request_steps,
                       "candidates": [f"candidates/{i}" for i in range(len(candidates))]}}, tensors)
    report = out.report["result"]
    spectra = [spectrum | {"singular_values": out.arrays[spectrum["singular_values"]]}
               for spectrum in report["step_spectra"]]
    return report, out.arrays["factor"], out.arrays["directions"], spectra


def law_transports(ov, laws):
    """The routing laws' summed OV transports, kept factored: per law, the heads' left factors side by side and
    their right factors stacked, so ``left @ right = sum_{h in law} A_h``. ``laws`` is the owner's grouping
    (``attention_laws`` of a weighted_observability report)."""
    return [(np.hstack([ov[h][0] for h in law]), np.vstack([ov[h][1] for h in law])) for law in laws]


def svd_band_rank(sigma, width):
    """Singular values above width * eps * sigma_1, the resolved row space's SVD band (state.rs
    factor_singular_band for a factor of at most ``width`` rows): the pull-back's formation bound is NOT included,
    so this is a numerical rank of the computed factor. The owner's ``resolved_rank`` includes it (certified)."""
    return int((sigma > width * np.finfo(np.float64).eps * sigma[0]).sum()) if sigma.size else 0


def singular_values(matrix):
    if matrix.shape[0] > 2 * matrix.shape[1]:
        matrix = la.qr(matrix, mode="r")
    return la.svdvals(matrix)


def effective(sigma, taus=(1e-2, 1e-3, 1e-6)):
    out = {f"{t:g}": int((sigma > t * sigma[0]).sum()) for t in taus}
    p = sigma**2 / (sigma**2).sum()
    out["entropy_rank"] = float(np.exp(-(p * np.log(p + 1e-300)).sum()))
    out["participation_ratio"] = float((sigma**2).sum() ** 2 / (sigma**4).sum())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    started = time.time()
    D = Decoder(args.model, args.revision)
    d, n_layers = D.d, D.L

    g_final = D("model.norm.weight")
    readout = D.readout()
    ov = [D.ov(layer) for layer in range(n_layers)]
    load_s = time.time() - started

    # 1. rowspace(C) through its R factor (same Gram, row space and singular values; the owner is slow on the
    # tall C). The owner's eps band then uses R's shape, max(m, n) = d instead of the vocabulary size: the rank
    # is the same under either band whenever sigma_min / sigma_max clears the larger one (recorded).
    r_readout = la.qr(readout, mode="r")
    readout_shape = list(readout.shape)
    rank_c = int(resolved_rows(r_readout).shape[0])
    s_c = singular_values(r_readout)
    weak = np.argsort(np.abs(g_final))[:5]
    pre_norm = D.where == "pre"
    result = {
        "model": args.model,
        "architecture": D.describe(),
        "conventions": {
            "readout": "C = W_U diag(g_final), " + ("tied embeddings" if D.tied else "untied lm_head")
                       + ", RMSNorm normalizer excluded (fixed positive scalar)",
            "transition": ("A_{l,h} = W_O[:, h] W_V[g(h)] diag(g_in,l)" if pre_norm else
                           "A_{l,h} = diag(g_post_attn,l) W_O[:, h] W_V[h]; the excluded 1/rms(attention output) "
                           "is one positive scalar per token and layer shared by all heads: ranks/closures exact, "
                           "Gramian weights at scalar 1") + "; heads grouped into routing laws by their score operators "
                          "(joint_operators::attention_letters), A_{l,r} = sum of A_{l,h} over law r; T = I + A_{l,r}; "
                          "closing under T equals closing under A",
            "mlp_reads": "rows of [W_gate; W_up]" + (" diag(g_post)" if pre_norm else " (raw residual input)")
                         + " added as readouts at the post-attention residual",
            "rank_rule": "state.rs resolved_row_space and LinearStateQuotient::close, called through the MPD surface "
                         "(linear_state_quotient); 'tau' charts keep the singular directions above tau sigma_max; "
                         "section 3: rank_at_band counts the owner's singular values above d eps sigma_1 (formation "
                         "not included), certified_rank is WeightedObservability's resolved rank at its band "
                         "(factor band + the pull-back's formation bound)",
            "labels": "rank at the eps band is exact-arithmetic rank to within roundoff (numerical-exact); "
                      "tau and effective dimensions are numerical/conditioning statements",
        },
        "readout": {
            "shape": readout_shape, "rank_at_band": rank_c,
            "band_choice_irrelevant": bool(s_c[-1] / s_c[0] > max(readout_shape) * np.finfo(float).eps),
            "sigma_max": float(s_c[0]), "sigma_min": float(s_c[-1]), "sigma_min_over_max": float(s_c[-1] / s_c[0]),
            "sigma_top5": s_c[:5].tolist(), "sigma_bottom8": s_c[-8:].tolist(),
            "effective": effective(s_c),
            "smallest_final_gain_coords": {int(i): float(g_final[i]) for i in weak},
        },
    }
    del readout

    # 3. causal backward closure per layer: the owner's weighted Gramian with each layer's attention block as its
    # routing-law letters (joint_operators::attention_letters), then tau charts (numerical) under the same laws.
    # Section 3 runs first because the laws it reports are the transitions of section 2's closure.
    per_layer = {"attention_only": [], "with_mlp_reads": []}
    layer_laws = None
    for variant in ("attention_only", "with_mlp_reads"):
        steps = []
        for layer in reversed(range(n_layers)):
            readouts = [r_readout] if layer == n_layers - 1 else []
            if variant == "with_mlp_reads":
                readouts.append(D.mlp_reads(layer))
            steps.insert(0, ([np.eye(d), D.attention_letter(layer, f"layers/{layer}/")], readouts))
        report, _, directions, spectra = weighted_observability(steps)
        del steps
        layer_laws = report["attention_laws"]
        charts = {tau: relative_rows(r_readout, tau) for tau in (1e-2, 3e-2)}
        chart_ranks = {}
        for layer in reversed(range(n_layers)):
            if variant == "with_mlp_reads":
                reads = D.mlp_reads(layer)
                for tau in charts:
                    charts[tau] = relative_rows(np.vstack([charts[tau], reads / la.spectral_norm(reads)]), tau)
            laws = law_transports(ov[layer], layer_laws[layer])
            for tau in charts:
                charts[tau] = relative_rows(np.vstack([charts[tau]] + [apply(charts[tau], a) for a in laws]), tau)
            chart_ranks[layer] = {"chart_rank_tau_1e-2": int(charts[1e-2].shape[0]),
                                  "chart_rank_tau_3e-2": int(charts[3e-2].shape[0])}
        for layer, spectrum in enumerate(spectra):
            s = spectrum["singular_values"]
            per_layer[variant].append({
                "layer": layer, "rank_at_band": svd_band_rank(s, d), "certified_rank": spectrum["resolved_rank"],
                "certified_band": spectrum["band"],
                "sigma_min_over_max": float(s[-1] / s[0]), "effective": effective(s),
                "participation_ratio": spectrum["participation_ratio"], **chart_ranks[layer],
                "sigma_bottom4_over_max": (s[-4:] / s[0]).tolist(),
            })
            print(variant, layer, spectrum["resolved_rank"], f"PR={spectrum['participation_ratio']:.1f}",
                  f"{time.time() - started:.0f}s", flush=True)
        weakest = directions[-1]
        per_layer[variant + "_layer0_weakest_dir_top_coords"] = {
            int(i): float(weakest[i]) for i in np.argsort(-np.abs(weakest))[:3]}
        per_layer[variant + "_formation"] = report["formation"]
    result["routing_laws"] = {"per_layer": layer_laws,
                              "shared_heads": sum(len(law) - 1 for laws in layer_laws for law in laws)}

    # 2. state.rs's closure under all routing laws' OV transports (time-invariant family)
    all_laws = [a for layer in range(n_layers) for a in law_transports(ov[layer], layer_laws[layer])]

    def closure(readouts):
        chart, report = linear_quotient(readouts, [a[0] @ a[1] for a in all_laws])
        return {"rank_closed": int(chart.shape[0]),
                "max_quotient_bound": max(b["upper"] for b in report["quotient_bounds"]),
                "max_readout_bound": max(b["upper"] for b in report["readout_bounds"]),
                "section_bound": report["section_bounds"]["upper"]}
    if rank_c == d:  # rowspace(C) is already everything: every closure is d_model, exactly
        full = {"rank_closed": d, "owner_called": False, "reason": "rank_C = d_model"}
        result["time_invariant_closure"] = {"attention_only": {"rank_C": rank_c, **full},
                                            "with_mlp_reads": {"rank_C_plus_mlp_reads": d, **full}}
    else:
        mlp_reads = [D.mlp_reads(layer) for layer in range(n_layers)]
        result["time_invariant_closure"] = {
            "attention_only": {"rank_C": rank_c, **closure([r_readout])},
            "with_mlp_reads": {
                "rank_C_plus_mlp_reads": int(resolved_rows(np.vstack([r_readout] + mlp_reads)).shape[0]),
                **closure([r_readout] + mlp_reads)}}
        del mlp_reads
    result["causal_backward"] = per_layer
    all_ov = [a for layer in ov for a in layer]
    head_norms = [la.spectral_norm(a[0] @ a[1]) for a in all_ov]
    result["ov_head_norm_range"] = [float(min(head_norms)), float(max(head_norms))]
    result["runtime_s"] = {"load": load_s, "total": time.time() - started}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(compact_json(result))
    print(json.dumps({k: result[k] for k in ("readout", "time_invariant_closure", "runtime_s")}, indent=1))
    for variant in ("attention_only", "with_mlp_reads"):
        print(variant, per_layer[variant + "_layer0_weakest_dir_top_coords"])
        for r in per_layer[variant]:
            print(r["layer"], r["rank_at_band"], f"{r['sigma_min_over_max']:.2e}", r["effective"]["0.01"],
                  r["effective"]["0.001"], f"{r['effective']['entropy_rank']:.0f}", f"{r['participation_ratio']:.1f}",
                  r["chart_rank_tau_1e-2"], r["chart_rank_tau_3e-2"])


if __name__ == "__main__":
    main()
