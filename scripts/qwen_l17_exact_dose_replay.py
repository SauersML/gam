#!/usr/bin/env python3
"""Exact local-Fisher replay for the frozen Qwen layer-17 dose protocol.

This is an audit wrapper around the historical model-facing harvest driver.  The
Qwen fused-MoE stack has no forward-mode rule for ``torch._grouped_mm``; the old
driver therefore estimated ``J^T F J`` from sampled vocabulary-class gradients.
For every intervention this wrapper instead evaluates the exact directional
quadratic ``0.5 * delta^T J^T F J delta`` with reverse-over-reverse autodiff,
then measures the matching forward KL ``KL(p_base || p_patched)``.  No finite
differences or alternate dose definitions are used.

The wrapped driver still owns the frozen prompt bank, chart fit, amplitude-
normalised chord construction, base split, and patched forward.  Its output is
rewritten in place to contain only the exact endpoint prediction, plus a
cluster-bootstrap acceptance report over held-out base prompts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def _load_driver(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_qwen_dose_driver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen dose driver {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _score(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    x = np.asarray([row["predicted_nats"] for row in rows], dtype=np.float64)
    y = np.asarray([row["measured_kl"] for row in rows], dtype=np.float64)
    x2 = float(x @ x)
    y2 = float(y @ y)
    if not rows or not (x2 > 0.0 and y2 > 0.0):
        raise ValueError("dose score needs non-empty, non-zero prediction and measurement")
    slope = float((x @ y) / x2)
    residual = y - slope * x
    return {
        "n": len(rows),
        "slope_through_origin": slope,
        "r2_through_origin": float(1.0 - (residual @ residual) / y2),
    }


def _dose_band(row: dict[str, Any]) -> str:
    """Bucket an edit by its predicted-quadratic magnitude (in nats).

    The second-order Fisher form is exact only near the base point; even inside
    the reported validity radius the KL curvature grows with dose. Banding by the
    quadratic magnitude lets the acceptance verify calibration holds *within each
    dose regime* rather than trusting one pooled slope, which can average a
    truncation-driven over-shoot against a curvature-driven under-shoot (#2249).
    """
    q = float(row["predicted_nats"])
    if q <= 0.02:
        return "deep_local_<=0.02nat"
    if q <= 0.2:
        return "local_0.02-0.2nat"
    return "near_radius_>0.2nat"


def _stratified_report(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-dose-band and per-atom (one atom == one calendar feature) slope/R², so
    no single pooled slope can hide a regime mixture. Bands with too few points to
    fit are reported as bare counts."""
    def _by(keyfn) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(keyfn(row)), []).append(row)
        out: dict[str, dict[str, Any]] = {}
        for key in sorted(groups):
            grp = groups[key]
            x = np.asarray([r["predicted_nats"] for r in grp], dtype=np.float64)
            if len(grp) >= 3 and float(x @ x) > 0.0:
                out[key] = _score(grp)
            else:
                out[key] = {"n": len(grp), "slope_through_origin": None, "r2_through_origin": None}
        return out

    return {
        "by_dose_band": _by(_dose_band),
        "by_atom": _by(lambda r: f"atom_{int(r['atom'])}"),
    }


def _cluster_bootstrap_slope_ci(
    rows: list[dict[str, Any]], *, draws: int, seed: int
) -> list[float]:
    clusters: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        clusters.setdefault(int(row["base"]), []).append(row)
    keys = sorted(clusters)
    if len(keys) < 2:
        raise ValueError("cluster bootstrap needs at least two held-out base prompts")
    rng = np.random.Generator(np.random.PCG64(seed))
    slopes = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        replay = [row for key in sampled for row in clusters[int(key)]]
        slopes[draw] = _score(replay)["slope_through_origin"]
    lo, hi = np.quantile(slopes, [0.025, 0.975])
    return [float(lo), float(hi)]


def _install_exact_protocol(driver):
    import torch

    historical_measurer = driver.MeasuredKL
    historical_run_sweep = driver.run_sweep

    class ExactMeasuredKL(historical_measurer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.exact_calls: list[dict[str, float]] = []

        def kl(self, prompt, delta):
            ids = self._ids(prompt)
            delta64 = torch.as_tensor(
                np.asarray(delta), dtype=torch.float64, device=self.device
            )
            captured: dict[str, torch.Tensor] = {}

            def _capture(_module, _inputs, output):
                captured["activation"] = output.detach()

            handle = self.hook_module.register_forward_hook(_capture)
            try:
                with torch.no_grad():
                    self.lm.module(ids)
            finally:
                handle.remove()
            activation = captured["activation"]
            flat = activation.reshape(-1, activation.shape[-1])
            x0_native = flat[-1].detach()
            requested_delta = delta64.to(device=x0_native.device, dtype=x0_native.dtype)
            # The patched forward adds in the hook tensor's native dtype.  On
            # this model that is bf16, so the actual edit is the rounded target
            # minus the already-rounded base activation—not the f64 chart delta.
            # Price precisely that effective move.
            target_native = x0_native + requested_delta
            x0 = x0_native.float().detach().requires_grad_(True)
            delta_work = target_native.float() - x0_native.float()

            def logits_from_x(x):
                def _splice(_module, _inputs, output):
                    output_flat = output.reshape(-1, output.shape[-1])
                    rows = [output_flat[index] for index in range(output_flat.shape[0])]
                    rows[-1] = x.to(device=output.device, dtype=output.dtype)
                    return torch.stack(rows, dim=0).reshape(output.shape)

                splice = self.hook_module.register_forward_hook(_splice)
                try:
                    logits = self.lm.module(ids)
                finally:
                    splice.remove()
                return logits[0, -1].float()

            logits0 = logits_from_x(x0)
            if float(torch.linalg.vector_norm(delta_work)) == 0.0:
                predicted = 0.0
            else:
                cotangent = torch.zeros_like(logits0, requires_grad=True)
                jt_cotangent = torch.autograd.grad(
                    logits0,
                    x0,
                    grad_outputs=cotangent,
                    create_graph=True,
                )[0]
                j_delta = torch.autograd.grad(
                    jt_cotangent,
                    cotangent,
                    grad_outputs=delta_work,
                )[0].double()
                probs0 = torch.softmax(logits0.detach().double(), dim=-1)
                mean = torch.sum(probs0 * j_delta)
                predicted = float(
                    0.5 * (torch.sum(probs0 * j_delta.square()) - mean.square())
                )

            with torch.no_grad():
                logits1 = logits_from_x(target_native).double()
            log_probs0 = torch.log_softmax(logits0.detach().double(), dim=-1)
            log_probs1 = torch.log_softmax(logits1, dim=-1)
            probs0 = log_probs0.exp()
            probs1 = log_probs1.exp()
            forward_kl = float(torch.sum(probs0 * (log_probs0 - log_probs1)))
            reverse_kl = float(torch.sum(probs1 * (log_probs1 - log_probs0)))
            self.exact_calls.append(
                {
                    "requested_delta_norm": float(torch.linalg.vector_norm(delta64)),
                    "effective_delta_norm": float(torch.linalg.vector_norm(delta_work)),
                    "predicted_nats": predicted,
                    "forward_kl": forward_kl,
                    "reverse_kl": reverse_kl,
                    "symmetric_kl": 0.5 * (forward_kl + reverse_kl),
                }
            )
            return forward_kl

    def exact_run_sweep(measurer, *args, **kwargs):
        start = len(measurer.exact_calls)
        rows = historical_run_sweep(measurer, *args, **kwargs)
        calls = measurer.exact_calls[start:]
        floor_reps = int(os.environ.get("DOSE_FLOOR_REPS", "1"))
        cursor = 0
        pending_linear: dict[str, float] | None = None
        for row in rows:
            method = row["method"]
            if method == "empty":
                cursor += floor_reps
                continue
            if method == "manifold":
                call = calls[cursor]
                cursor += 1
                row["requested_delta_norm"] = call["requested_delta_norm"]
                row["delta_norm"] = call["effective_delta_norm"]
                row["predicted_nats"] = call["predicted_nats"]
                row["measured_kl"] = call["forward_kl"]
                row["measured_reverse_kl"] = call["reverse_kl"]
                row["measured_symmetric_kl"] = call["symmetric_kl"]
                row.pop("predicted_nats_pathint", None)
                row.pop("predicted_nats_tangent", None)
                dt = row.get("dt")
                radius = row.get("validity_radius")
                row["within_validity"] = bool(
                    dt is not None and radius is not None and abs(float(dt)) <= float(radius)
                )
                continue
            if method == "linear_norm":
                pending_linear = calls[cursor]
                cursor += 1
                row["requested_delta_norm"] = pending_linear["requested_delta_norm"]
                row["delta_norm"] = pending_linear["effective_delta_norm"]
                row["measured_kl"] = pending_linear["forward_kl"]
                continue
            if method == "linear_fisher":
                if pending_linear is None:
                    raise RuntimeError("linear Fisher row has no preceding measured move")
                row["requested_delta_norm"] = pending_linear["requested_delta_norm"]
                row["delta_norm"] = pending_linear["effective_delta_norm"]
                row["predicted_nats"] = pending_linear["predicted_nats"]
                row["measured_kl"] = pending_linear["forward_kl"]
                pending_linear = None
        if cursor != len(calls):
            raise RuntimeError(f"consumed {cursor} exact calls but recorded {len(calls)}")
        return rows

    driver.MeasuredKL = ExactMeasuredKL
    driver.run_sweep = exact_run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--features", default="weekday")
    parser.add_argument("--max-templates", type=int, default=6)
    parser.add_argument("--bases", type=int, default=10)
    parser.add_argument("--fit-iterations", type=int, default=40)
    parser.add_argument("--fractions", default="0.005,0.01,0.02,0.05,0.1,0.2,0.4")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    settings = {
        "DOSE_MODEL": str(args.model),
        "DOSE_MODEL_DTYPE": "bfloat16",
        "DOSE_DEVICE_MAP": "auto",
        "DOSE_DTYPE": "float32",
        "DOSE_LAYER": str(args.layer),
        "DOSE_RANK": str(args.rank),
        "DOSE_FEATURES": args.features,
        "DOSE_MAXTPL": str(args.max_templates),
        "DOSE_NBASES": str(args.bases),
        "DOSE_NITER": str(args.fit_iterations),
        "DOSE_FRACS": args.fractions,
        "DOSE_DT_PI": "",
        "DOSE_FLOOR_REPS": "1",
        "DOSE_OUT": str(args.out),
        "DOSE_SEED": str(args.seed),
    }
    os.environ.update(settings)
    driver = _load_driver(args.driver)
    _install_exact_protocol(driver)
    return_code = driver.main()
    if return_code not in (None, 0):
        raise SystemExit(return_code)

    ledger_path = args.out / "dose_calibration_real.json"
    ledger = json.loads(ledger_path.read_text())
    manifold = [row for row in ledger["rows"] if row["method"] == "manifold"]
    heldout = [row for row in manifold if row.get("heldout", False)]
    heldout_within = [row for row in heldout if row.get("within_validity", False)]
    finite = [
        row
        for row in heldout_within
        if math.isfinite(float(row["predicted_nats"]))
        and math.isfinite(float(row["measured_kl"]))
        and float(row["predicted_nats"]) >= 0.0
        and float(row["measured_kl"]) >= 0.0
    ]
    report = _score(finite)
    report["slope_cluster_bootstrap_95_ci"] = _cluster_bootstrap_slope_ci(
        finite, draws=args.bootstrap_draws, seed=args.seed + 2249
    )

    # #2249: do not trust one pooled slope. Stratify the in-radius edits by dose
    # regime and by atom so a truncation-driven over-shoot cannot average against a
    # curvature-driven under-shoot into a spuriously "calibrated" pooled number.
    report["stratified"] = _stratified_report(finite)
    band_slopes = [
        b["slope_through_origin"]
        for b in report["stratified"]["by_dose_band"].values()
        if b.get("slope_through_origin") is not None and int(b.get("n", 0)) >= 3
    ]
    all_bands_calibrated = bool(band_slopes) and all(0.9 <= s <= 1.1 for s in band_slopes)

    # Out-of-radius held-out edits are NOT scored for acceptance (the second-order
    # form is honestly an upper bound past the validity radius), but are reported as
    # a diagnostic — their slope should drift below 1 as the true KL saturates.
    heldout_out = [
        row
        for row in heldout
        if not row.get("within_validity", False)
        and math.isfinite(float(row["predicted_nats"]))
        and math.isfinite(float(row["measured_kl"]))
        and float(row["predicted_nats"]) > 0.0
        and float(row["measured_kl"]) >= 0.0
    ]
    report["out_of_radius_diagnostic"] = (
        _score(heldout_out) if len(heldout_out) >= 3 else {"n": len(heldout_out)}
    )

    report.update(
        {
            "acceptance_slope_pooled": 0.9 <= report["slope_through_origin"] <= 1.1,
            "acceptance_slope_all_dose_bands": all_bands_calibrated,
            "acceptance_r2": report["r2_through_origin"] >= 0.9,
            "ci_excludes_0_54": not (
                report["slope_cluster_bootstrap_95_ci"][0]
                <= 0.54
                <= report["slope_cluster_bootstrap_95_ci"][1]
            ),
            "ci_excludes_4_30": not (
                report["slope_cluster_bootstrap_95_ci"][0]
                <= 4.30
                <= report["slope_cluster_bootstrap_95_ci"][1]
            ),
            "prediction": "0.5 * delta^T J^T (diag(p)-p p^T) J delta",
            # The predictor here is the EXACT full output-Fisher pullback (no rank
            # truncation): the low-rank {8,32,64}-factor metric only fits the chart;
            # the dose is scored against the exact J^T F J. So the reported slope is
            # free of the truncation regime that biases a low-rank ledger's slope
            # above 1 (#2249); `chart_fit_rank` is recorded only for provenance.
            "predictor_rank_truncation": "none (exact full-rank J^T F J)",
            "chart_fit_rank": int(args.rank),
            "measurement": "KL(p_base || p_patched)",
            "validity_rule": "heldout and abs(dt) <= steer.validity_radius",
            "heldout_rule": "second half of the seeded random base-prompt draw",
            "acceptance_rule": (
                "calibrated iff pooled slope in [0.9,1.1] AND every populated dose "
                "band (n>=3) in [0.9,1.1] AND R2>=0.9; scored on in-radius held-out "
                "edits against the exact full-rank Fisher (#2249)"
            ),
        }
    )
    ledger["canonical_prediction"] = report["prediction"]
    ledger["canonical_measurement"] = report["measurement"]
    ledger["acceptance_2249"] = report
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
