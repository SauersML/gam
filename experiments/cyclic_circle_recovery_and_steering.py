#!/usr/bin/env python
"""Recover, certify, and causally steer a cyclic concept circle in a causal LM.

End-to-end recipe for weekday/month calendar circles (the measured case:
Qwen3-8B layer 20, recovery ordering r = 0.96 against a chart-rebuilt
permutation null, steering-validated):

  1. RECOVER: harvest residual activations at the concept-token position
     across varied mid-sentence templates; build the *day-signal chart*
     (per-template centering + top-2 PCs of the label-class means) and read
     each row's angle in that plane.
  2. CERTIFY: (a) circular-linear ordering of the angles against the true
     calendar phase, tested against a FULL-PIPELINE permutation null (the
     label-aware chart is rebuilt for every shuffled labeling, so chart
     construction cannot inflate the statistic); (b) the
     ``gamfit.adjudicate_atom_shape`` race on the 2-D coords, paired with the
     centroid circular-ordering diagnostic (``centroid_ordering.py`` next to
     this file). The race owns discrete cyclic structure through its
     ``ring_clusters`` class; ``ring_clusters_reporting_k`` names its all-data
     diagnostic fit, while ``ring_clusters_fold_selected_k`` records the honest
     outer-fold orders. ``circle_wins`` compares total circular stacking mass
     (smooth circle plus ring clusters) with total non-circular mass. The
     centroid test independently checks angular ordering.
  3. STEER: fit a circle atom on final-position activations of calendar
     continuation prompts (gamfit torch lane), then patch the chord of the
     fitted decoder curve into the residual stream and measure next-token
     calendar advance — against two matched-norm controls (random ambient
     direction; in-chart direction orthogonal to the local circle tangent).

Why the class-mean chart instead of plain top-2 PCA: the calendar circle is a
LOW-RELATIVE-VARIANCE structure. Template/context variance dominates raw
concept-token activations, and a top-2-variance projection loses a ring masked
by a linear factor at >= ~1x its radius (measured by injection; ordering
p = 0.16 at 1x, adjudicator verdict degrades to mixture at 2x). Per-template
centering plus class-mean PCs is the label-seeded projection that reaches the
circle. The label-awareness is exactly why the ordering null must rebuild the
chart per permutation.

Full run needs a GPU for the default 8B model (a smaller --model runs on CPU);
--smoke shrinks everything for a fast pipeline check.

Example:
  python cyclic_circle_recovery_and_steering.py --model Qwen/Qwen3-8B \
      --layer 20 --out-dir circle_out/
  python cyclic_circle_recovery_and_steering.py --model Qwen/Qwen2.5-0.5B \
      --layer 12 --smoke --allow-cpu --out-dir circle_smoke/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from centroid_ordering import centroid_circular_ordering  # noqa: E402

TAU = 2.0 * math.pi

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday")
MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")

RECOVERY_TEMPLATES_DAY = (
    "I will see you on {w}.",
    "The meeting is scheduled for {w} morning.",
    "She was born on a {w}.",
    "We usually go shopping on {w}.",
    "The package arrived last {w}.",
    "Every {w} we have a call.",
    "It happened on {w} afternoon.",
    "They are leaving next {w}.",
    "My favorite day is {w}.",
    "The store is closed on {w}.",
    "He starts work this {w}.",
    "The concert is on {w} evening.",
)
RECOVERY_TEMPLATES_MONTH = (
    "The conference takes place in {w} this year.",
    "She was born in {w}.",
    "The lease expires in {w} next year.",
    "Every {w} the village holds a festival.",
    "It rained heavily throughout {w}.",
    "The project deadline is in {w}.",
    "They got married last {w}.",
    "The new store opens in {w}.",
    "Harvest season peaks in {w} here.",
    "His visa runs out in {w}.",
    "The exam results come out in {w}.",
    "We visited Japan in {w}.",
)
STEER_FIT_TEMPLATES_DAY = (
    "Today is {w}. Tomorrow is",
    "If today is {w}, then tomorrow is",
    "The weekday after {w} is",
    "On a weekly calendar, {w} is followed by",
    "Yesterday was {w}, so today is",
    "The day that comes right after {w} is",
    "After {w} comes",
    "Counting forward from {w}, the next day is",
    "A day later than {w} is",
    "Following {w} on the calendar is",
)
STEER_BASE_TEMPLATES_DAY = (
    "Starting on {w}, the next day is",
    "Calendar note: the day after {w} is",
)
STEER_FIT_TEMPLATES_MONTH = (
    "This month is {w}. Next month is",
    "If this month is {w}, then next month is",
    "The month after {w} is",
    "On the calendar, {w} is followed by",
    "Last month was {w}, so this month is",
    "The month that comes right after {w} is",
    "After {w} comes the month of",
    "Counting forward from {w}, the next month is",
    "A month later than {w} is",
    "Following {w} on the calendar is",
)
STEER_BASE_TEMPLATES_MONTH = (
    "Starting in {w}, the next month is",
    "Calendar note: the month after {w} is",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Model plumbing (tuple-safe hooks; layer L means hidden_states[L], i.e. the
# OUTPUT of model.model.layers[L-1] — capture and patch use the same site)
# --------------------------------------------------------------------------- #
def load_model(model_name: str, dtype_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[dtype_name]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, tok


def hook_module_for_layer(model, layer: int):
    """Module whose OUTPUT is hidden_states[layer] (layer >= 1)."""
    return model.model.layers[layer - 1]


def find_token_pos(tok, prompt: str, word: str) -> tuple[list[int], int]:
    """(input_ids, index of last subtoken of ``\u0020word`` in prompt)."""
    ids = tok(prompt, add_special_tokens=False).input_ids
    wid = tok(" " + word, add_special_tokens=False).input_ids
    n, m = len(ids), len(wid)
    for i in range(n - m, -1, -1):
        if ids[i:i + m] == wid:
            return ids, i + m - 1
    raise ValueError(f"could not locate {word!r} in {prompt!r}")


def run_capture(model, tok, prompt: str, layer: int, position: int | None):
    """Forward once; return (h[pos] at hidden_states[layer], final logits, pos)."""
    import torch

    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids.to(next(model.parameters()).device)
    pos = ids.shape[1] - 1 if position is None else position
    with torch.inference_mode():
        out = model(ids, output_hidden_states=True, use_cache=False)
    act = out.hidden_states[layer][0, pos, :].detach().float().cpu().numpy()
    logits = out.logits[0, -1, :].detach().float().cpu()
    return act, logits, pos


def run_patched(model, tok, layer_mod, prompt: str, pos: int,
                delta_ambient: np.ndarray):
    """Forward with h[pos] += delta at the hook site; return final logits."""
    import torch

    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids.to(next(model.parameters()).device)
    dvec = torch.from_numpy(np.asarray(delta_ambient, dtype=np.float32))

    def hook(_m, _i, output):
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden.clone()
        edited[0, pos, :] = edited[0, pos, :] + dvec.to(device=edited.device,
                                                        dtype=edited.dtype)
        if isinstance(output, tuple):
            return (edited,) + tuple(output[1:])
        return edited

    h = layer_mod.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            out = model(ids, use_cache=False)
    finally:
        h.remove()
    return out.logits[0, -1, :].detach().float().cpu()


# --------------------------------------------------------------------------- #
# Chart + ordering statistics
# --------------------------------------------------------------------------- #
def per_template_center(X: np.ndarray, template_ids: np.ndarray) -> np.ndarray:
    """Remove per-template means (uses template IDs only, never labels).
    Kills the dominant context/template variance that swamps the circle."""
    Xc = X.astype(np.float64).copy()
    for t in np.unique(template_ids):
        m = template_ids == t
        Xc[m] -= Xc[m].mean(0, keepdims=True)
    return Xc


def day_signal_chart(X: np.ndarray, labels: np.ndarray,
                     template_ids: np.ndarray, dim: int):
    """Chart where the calendar circle lives: per-template centering, then the
    top ``dim`` PCs of the LABEL-CLASS MEANS (rank <= period-1).

    Label-aware by construction — downstream ordering tests must rebuild this
    chart under label permutation (see ``pipeline_null``).

    Returns (Z, basis, ev): rows projected into the chart, the orthonormal
    (dim, d) ambient basis (delta_z @ basis lifts exactly), and the fraction
    of class-mean variance kept."""
    Xc = per_template_center(X, template_ids)
    period = int(labels.max()) + 1
    means = np.stack([Xc[labels == u].mean(0) for u in range(period)])
    means = means - means.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(means, full_matrices=False)
    r = int(min(dim, vt.shape[0]))
    basis = np.ascontiguousarray(vt[:r])
    ev = float((s[:r] ** 2).sum() / max((s ** 2).sum(), 1e-30))
    return np.ascontiguousarray(Xc @ basis.T), basis, ev


def class_mean_plane_angles(X: np.ndarray, labels: np.ndarray,
                            template_ids: np.ndarray) -> np.ndarray:
    """Circle coordinate: per-row angle in the class-mean top-2 plane."""
    Z, _b, _ev = day_signal_chart(X, labels, template_ids, 2)
    return np.arctan2(Z[:, 1], Z[:, 0])


def circular_linear_r(angles: np.ndarray, labels: np.ndarray,
                      period: int) -> float:
    """Correlation between fitted angles and the true calendar phase,
    maximized over rotation and reflection (both gauge freedoms of a fit)."""
    tgt = TAU * labels.astype(np.float64) / period
    best = -1.0
    for refl in (1.0, -1.0):
        a = refl * angles
        for shift in np.linspace(0.0, TAU, 144, endpoint=False):
            aa = (a + shift) % TAU
            sa, st = np.sin(aa - aa.mean()), np.sin(tgt - tgt.mean())
            denom = math.sqrt(float((sa ** 2).sum() * (st ** 2).sum())) + 1e-12
            best = max(best, float((sa * st).sum()) / denom)
    return best


def pipeline_null(X: np.ndarray, labels: np.ndarray, template_ids: np.ndarray,
                  period: int, n_perm: int, seed: int) -> dict:
    """Full-pipeline permutation null: the label-aware chart is REBUILT for
    every shuffled labeling, so chart construction cannot inflate the
    ordering statistic."""
    rng = np.random.default_rng(seed)
    obs = circular_linear_r(class_mean_plane_angles(X, labels, template_ids),
                            labels, period)
    null = np.empty(n_perm)
    for i in range(n_perm):
        lp = rng.permutation(labels)
        null[i] = circular_linear_r(
            class_mean_plane_angles(X, lp, template_ids), lp, period)
    return {
        "observed_r": obs,
        "null_p95": float(np.quantile(null, 0.95)),
        "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm)),
        "n_perm": n_perm,
        "null_kind": "chart_rebuilt_per_permutation",
    }


def fixed_chart_null(angles: np.ndarray, labels: np.ndarray, period: int,
                     n_perm: int, seed: int) -> dict:
    """Weaker fixed-chart null (labels shuffled in the correlation only) for
    coordinates that were fit WITHOUT labels, e.g. the SAE circle atom."""
    rng = np.random.default_rng(seed)
    obs = circular_linear_r(angles, labels, period)
    null = np.array([circular_linear_r(angles, rng.permutation(labels), period)
                     for _ in range(n_perm)])
    return {
        "observed_r": obs,
        "null_p95": float(np.quantile(null, 0.95)),
        "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm)),
        "n_perm": n_perm,
        "null_kind": "fixed_chart",
    }


# --------------------------------------------------------------------------- #
# Native circle fit + explicit chord steering
# --------------------------------------------------------------------------- #
def circle_fit(Z: np.ndarray, *, steps: int, seed: int, lr: float = 1e-2,
               grid: int = 720) -> dict:
    """Single circle atom via the native converged manifold-SAE fit.

    Returns per-row angles (radians), the decoder curve over one period
    (grid, chart_dim), and training R^2. Native circle coordinates have
    period 1.0; angles are converted to radians."""
    import gamfit

    fit = gamfit.sae_manifold_fit(
        X=np.ascontiguousarray(Z, dtype=np.float64),
        K=1,
        d_atom=1,
        atom_topology="circle",
        assignment="softmax",
        n_iter=steps,
        learning_rate=lr,
        random_state=seed,
    )
    pos = np.asarray(fit.coords[0], dtype=np.float64)[:, 0]
    atom = fit.atoms[0]
    curve = atom.shape_band_mean
    if curve is None:
        raise RuntimeError("native circle fit did not emit a posterior shape grid")
    curve = np.asarray(curve, dtype=np.float64)
    if curve.shape[0] != grid:
        curve = np.stack([curve_point(curve, value / grid) for value in range(grid)])
    return {
        "angles": (pos % 1.0) * TAU,
        "curve": curve,
        "r2": float(fit.reconstruction_r2),
    }


def curve_point(curve: np.ndarray, t_frac: float) -> np.ndarray:
    """Periodic linear interpolation of the decoder curve at t in [0,1)."""
    g = curve.shape[0]
    x = (t_frac % 1.0) * g
    i0 = int(np.floor(x)) % g
    w = x - np.floor(x)
    return (1.0 - w) * curve[i0] + w * curve[(i0 + 1) % g]


def curve_tangent(curve: np.ndarray, t_frac: float) -> np.ndarray:
    g = curve.shape[0]
    eps = 1.0 / g
    return (curve_point(curve, t_frac + eps)
            - curve_point(curve, t_frac - eps)) / (2 * eps)


def steer_delta(curve: np.ndarray, t_from_rad: float, t_to_rad: float,
                basis: np.ndarray) -> tuple[np.ndarray, float]:
    """Chord of the decoder curve from t_from to t_to, lifted to ambient space.

    Returns (delta_ambient, off_manifold_norm) where off_manifold_norm is the
    chord's residual off the local tangent line at t_from — the on-manifold
    quality check (~0 for small steps along the fitted shape)."""
    tf, tt = t_from_rad / TAU, t_to_rad / TAU
    delta_z = curve_point(curve, tt) - curve_point(curve, tf)
    tau_v = curve_tangent(curve, tf)
    tau_hat = tau_v / max(np.linalg.norm(tau_v), 1e-30)
    off = delta_z - (delta_z @ tau_hat) * tau_hat
    return delta_z @ basis, float(np.linalg.norm(off))


# --------------------------------------------------------------------------- #
def run_concept(model, tok, gamfit_mod, concept: str, words, rec_templates,
                fit_templates, base_templates, layer: int, *, pca_dim: int,
                fit_steps: int, n_perm: int, ks, seed: int, out_dir: Path,
                results: dict) -> None:
    import torch

    period = len(words)
    layer_mod = hook_module_for_layer(model, layer)

    # ---- 1. RECOVER: concept-token harvest -> class-mean chart -------------
    log(f"=== [{concept}] recovery harvest at layer {layer} ===")
    acts, labels, template_ids = [], [], []
    for ti, tmpl in enumerate(rec_templates):
        for wi, w in enumerate(words):
            prompt = tmpl.format(w=w)
            _ids, pos = find_token_pos(tok, prompt, w)
            act, _lg, _p = run_capture(model, tok, prompt, layer, pos)
            acts.append(act)
            labels.append(wi)
            template_ids.append(ti)
    X = np.asarray(acts, dtype=np.float64)
    labels = np.asarray(labels)
    template_ids = np.asarray(template_ids)

    # ---- 2. CERTIFY: ordering vs chart-rebuilt null + shape adjudication ---
    ordering = pipeline_null(X, labels, template_ids, period, n_perm, seed)
    log(f"  ordering r={ordering['observed_r']:.3f} "
        f"(chart-rebuilt null p={ordering['p_value']:.4f})")

    Z2, _basis2, ev2 = day_signal_chart(X, labels, template_ids, 2)
    coords2 = np.ascontiguousarray(Z2)
    try:
        verdict = dict(gamfit_mod.adjudicate_atom_shape(
            coords2,
            folds=5,
            seed=seed + 11,
            matched_controls=False,
        ))
        verdict.pop("bic", None)
    except Exception as exc:  # noqa: BLE001
        verdict = {"error": f"{type(exc).__name__}: {exc}"}
    # Diagnose the class that owns the verdict. A circular class win uses the
    # constrained ring candidate's order; otherwise the free mixture's order
    # checks whether a nominally non-circular cluster winner is nevertheless
    # angularly ordered. An adjudication error or k < 3 is a typed unavailable
    # diagnostic, never a reason to substitute the known concept period or a
    # different cluster order.
    if "error" in verdict:
        tier2 = {
            "status": "not_run",
            "reason": "shape_adjudication_error",
            "candidate_class": None,
            "k": None,
        }
    else:
        if bool(verdict["circle_wins"]):
            tier2_candidate = "ring_clusters"
            k2 = int(verdict["ring_clusters_reporting_k"])
        else:
            tier2_candidate = "free_mixture"
            k2 = int(verdict["mixture_reporting_k"])
        if k2 < 3:
            tier2 = {
                "status": "not_run",
                "reason": "candidate_order_below_three",
                "candidate_class": tier2_candidate,
                "k": k2,
            }
        else:
            tier2 = centroid_circular_ordering(coords2, k2, seed=seed, n_null=n_perm)
            tier2.pop("centers", None)
            tier2["status"] = "ok"
            tier2["candidate_class"] = tier2_candidate
    if "error" in verdict:
        log(
            f"  adjudicator failed: {verdict['error']}; "
            f"centroid diagnostic unavailable: {tier2['reason']}"
        )
    elif tier2["status"] == "ok":
        log(f"  adjudicator: reporting_winner={verdict['reporting_winner']}; "
            f"winner_class={verdict['winner_class']}; "
            f"circular_class_wins={verdict['circle_wins']}; "
            f"centroid diagnostic ({tier2['candidate_class']}, k={tier2['k']}): "
            f"ordered_on_circle={tier2['ordered_on_circle']} "
            f"(radius_cv={tier2['radius_cv']:.3f}, mc_p={tier2['mc_p']:.4f})")
    else:
        log(f"  adjudicator: {verdict['reporting_winner']}; "
            f"centroid diagnostic unavailable: {tier2['reason']}")

    # ---- 3. STEER: circle fit on continuation prompts + chord patching -----
    log(f"=== [{concept}] steering harvest (final position) ===")
    examples = []
    tid = 0
    for group, tmpls in (("fit", fit_templates), ("base", base_templates)):
        for tmpl in tmpls:
            for wi, w in enumerate(words):
                prompt = tmpl.format(w=w)
                act, logits, pos = run_capture(model, tok, prompt, layer, None)
                examples.append({"prompt": prompt, "pos": pos, "label": wi,
                                 "group": group, "template": tid, "act": act,
                                 "logits": logits})
            tid += 1
    s_labels = np.asarray([e["label"] for e in examples])
    s_tids = np.asarray([e["template"] for e in examples])
    X_amb = np.stack([e["act"] for e in examples]).astype(np.float64)

    chart_dim = int(min(pca_dim, period - 1))
    Z, lift, ev = day_signal_chart(X_amb, s_labels, s_tids, chart_dim)
    tfit = circle_fit(Z, steps=fit_steps, seed=seed)
    gate = fixed_chart_null(tfit["angles"], s_labels, period, n_perm, seed + 77)
    log(f"  steering fit r2={tfit['r2']:.3f}; gate r={gate['observed_r']:.3f} "
        f"(p={gate['p_value']:.4f}); chart class-mean ev={ev:.3f}")

    ang = tfit["angles"]
    curve = tfit["curve"]
    label_phase = TAU * s_labels / period

    def _concentration(sign: float) -> float:
        d = (sign * ang - label_phase + math.pi) % TAU - math.pi
        return float(np.abs(np.exp(1j * d).mean()))
    orient = 1.0 if _concentration(1.0) >= _concentration(-1.0) else -1.0

    cand_ids = [tok(" " + w, add_special_tokens=False).input_ids[0]
                for w in words]
    base_rows = [i for i, e in enumerate(examples) if e["group"] == "base"]
    rng = np.random.default_rng(seed + 5)
    records = []
    for i in base_rows:
        e = examples[i]
        b = e["label"]
        base_top = int(np.argmax(
            torch.softmax(e["logits"], dim=-1)[cand_ids].numpy()))
        t_from = float(ang[i])
        for k in ks:
            t_to = t_from + orient * k * TAU / period
            delta, off_norm = steer_delta(curve, t_from, t_to, lift)
            dn = float(np.linalg.norm(delta))
            delta_z = delta @ lift.T

            # matched-norm controls: random ambient; in-chart orthogonal to
            # the local circle tangent (kills "any big in-chart push works")
            g = rng.standard_normal(delta.shape[0])
            ctrl_amb = g / np.linalg.norm(g) * dn
            tang = delta_z / max(np.linalg.norm(delta_z), 1e-30)
            gz = rng.standard_normal(delta_z.shape[0])
            gz -= (gz @ tang) * tang
            ctrl_chart = (gz / max(np.linalg.norm(gz), 1e-30)
                          * np.linalg.norm(delta_z)) @ lift

            for arm, dvec in (("manifold", delta),
                              ("random_ambient", ctrl_amb),
                              ("chart_orthogonal", ctrl_chart)):
                pl = run_patched(model, tok, layer_mod, e["prompt"], e["pos"],
                                 dvec)
                probs_r = torch.softmax(pl, dim=-1)[cand_ids].numpy()
                logp = torch.log_softmax(e["logits"].to(torch.float64), dim=-1)
                logq = torch.log_softmax(pl.to(torch.float64), dim=-1)
                kl = float((logp.exp() * (logp - logq)).sum())
                tgt = (b + k) % period
                records.append({
                    "arm": arm, "k": int(k), "base_label": int(b),
                    "target_label": int(tgt), "delta_norm": dn,
                    "off_manifold_norm": off_norm if arm == "manifold" else None,
                    "realized_kl": max(kl, 0.0),
                    "top_label_realized": int(np.argmax(probs_r)),
                    "already_correct": bool(base_top == tgt),
                })

    summary = {}
    for arm in ("manifold", "random_ambient", "chart_orthogonal"):
        rs = [r for r in records if r["arm"] == arm]
        nt = [r for r in rs if not r["already_correct"]]
        summary[arm] = {
            "advance_accuracy": float(np.mean(
                [r["top_label_realized"] == r["target_label"] for r in rs])),
            "nontrivial_advance_accuracy": (float(np.mean(
                [r["top_label_realized"] == r["target_label"] for r in nt]))
                if nt else float("nan")),
            "n_nontrivial": len(nt),
            "mean_realized_kl": float(np.mean([r["realized_kl"] for r in rs])),
        }
    log(f"  steer advance acc: manifold="
        f"{summary['manifold']['advance_accuracy']:.3f} vs random="
        f"{summary['random_ambient']['advance_accuracy']:.3f} vs chart-orth="
        f"{summary['chart_orthogonal']['advance_accuracy']:.3f}")

    results[concept] = {
        "period": period, "layer": layer,
        "recovery_ordering": ordering,
        "chart_class_mean_ev_2d": ev2,
        "adjudication": verdict,
        "centroid_tier2": tier2,
        "steering": {"fit_r2": tfit["r2"], "gate": gate,
                     "orientation": orient, "ks": list(ks),
                     "summary": summary},
    }
    with open(out_dir / f"steer_{concept}_records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--layer", type=int, default=20,
                    help="hidden_states index (1-based blocks)")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--pca-dim", type=int, default=6)
    ap.add_argument("--fit-steps", type=int, default=600)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--max-k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-months", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit CPU execution (small models / smoke)")
    args = ap.parse_args()

    import torch
    if not (torch.cuda.is_available() or args.allow_cpu):
        raise SystemExit("CUDA unavailable: pass --allow-cpu for small models "
                         "or run on a GPU host.")

    import gamfit
    assert hasattr(gamfit, "adjudicate_atom_shape"), "gamfit missing adjudicator"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    day_rec, month_rec = RECOVERY_TEMPLATES_DAY, RECOVERY_TEMPLATES_MONTH
    day_fit, day_base = STEER_FIT_TEMPLATES_DAY, STEER_BASE_TEMPLATES_DAY
    month_fit, month_base = STEER_FIT_TEMPLATES_MONTH, STEER_BASE_TEMPLATES_MONTH
    fit_steps, n_perm = args.fit_steps, args.n_perm
    ks = list(range(1, args.max_k + 1))
    if args.smoke:
        day_rec, month_rec = day_rec[:4], month_rec[:4]
        day_fit, day_base = day_fit[:4], day_base[:1]
        month_fit, month_base = month_fit[:4], month_base[:1]
        fit_steps, n_perm, ks = 100, 100, [1]

    log(f"loading {args.model} ({args.dtype})")
    model, tok = load_model(args.model, args.dtype)
    n_layers = model.config.num_hidden_layers
    assert 1 <= args.layer <= n_layers, \
        f"--layer {args.layer} out of range for {n_layers}-block model"

    results: dict = {"meta": {
        "model": args.model, "layer": args.layer, "pca_dim": args.pca_dim,
        "fit_steps": fit_steps, "n_perm": n_perm, "ks": ks,
        "seed": args.seed, "smoke": args.smoke,
        "layer_convention":
            "hidden_states[L] = output of model.model.layers[L-1]",
    }}
    concepts = [("weekday", WEEKDAYS, day_rec, day_fit, day_base)]
    if not args.skip_months:
        concepts.append(("month", MONTHS, month_rec, month_fit, month_base))
    for name, words, rec, fit_t, base_t in concepts:
        run_concept(model, tok, gamfit, name, words, rec, fit_t, base_t,
                    args.layer, pca_dim=args.pca_dim, fit_steps=fit_steps,
                    n_perm=n_perm, ks=ks, seed=args.seed, out_dir=out_dir,
                    results=results)

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=float))
    log(f"wrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
