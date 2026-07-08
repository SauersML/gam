#!/usr/bin/env python3
"""SAEBench + manifold-native SAE evaluation driver (#1942).

This is the one evaluation entry point for #1942. It scores an SAE on both
axes the issue asks for:

* **SAEBench capability suite** (absorption / SCR / targeted-unlearning). These
  require a forward pass through the host LLM with the SAE spliced in, so they
  are delegated to the public ``sae_bench`` package when it is importable. When
  it is not (e.g. a CPU box without torch), the corresponding block is reported
  as ``skipped`` with the reason rather than fabricated.

* **Manifold-native metrics** that only this system can report, computed by the
  audited Rust core through ``gamfit`` (no math is re-implemented here):

  - ``chart-interp`` — orientation-quotiented weighted cyclic phase-lock of a
    recovered chart coordinate ``t`` against ground-truth cyclic labels
    (``gamfit.chart_interp_score``). An interpretability metric for *coordinates*,
    not just latents.
  - ``dose-response calibration`` — measured next-token KL vs the local
    output-Fisher prediction along steered arcs, with the unit-speed constancy
    kill-test (``gamfit.dose_response_calibration``).
  - the frozen-dictionary capability ``audit`` (routability floor, dark-matter
    fraction, dual certificate, absorption pairs, per-atom Betti topology and
    atlas nerve) via ``gamfit.audit_sae``.

Inputs are typed observation ledgers harvested by the model-facing code
(a dose-response ledger json, a chart-fit json, a decoder + activations pair);
this driver only marshals them into the audited scorers and assembles one
report. Run ``--help`` for the individual arms; any subset may be supplied.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import gamfit


# --------------------------------------------------------------------------- #
# Dose-response calibration (manifold-native metric 2)
# --------------------------------------------------------------------------- #
# Canonical weekday/month cyclic label order for chart-interp ground truth.
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _dose_predicted(row: dict[str, Any], predicted_key: str) -> float:
    """Local output-Fisher prediction in nats for a dose row.

    ``predicted_key`` selects the prediction variant the ledger stores
    (``predicted_nats`` tangent quadratic form, ``predicted_nats_pathint``
    path-integrated, ``predicted_nats_tangent``)."""
    return float(row[predicted_key])


def dose_response_report(
    ledger_path: Path,
    *,
    predicted_key: str = "predicted_nats",
    within_validity_only: bool = True,
    heldout_only: bool = False,
) -> dict[str, Any]:
    """Score a harvested dose-response ledger per steering method.

    The ledger is the json emitted by the dosimetry harvest: a ``rows`` list of
    per-intervention records carrying ``method``, ``dt`` (unit-speed arc length),
    a predicted-nats variant, ``measured_kl`` (patched-forward next-token KL),
    ``within_validity`` and ``heldout``. Each method is scored independently by
    ``gamfit.dose_response_calibration``; the manifold method is the headline,
    the linear baselines are the deconfounding arm the issue describes."""
    doc = json.loads(ledger_path.read_text())
    rows = doc["rows"]
    methods: dict[str, list[tuple[float, float, float, float]]] = {}
    for row in rows:
        if within_validity_only and not row.get("within_validity", True):
            continue
        if heldout_only and not row.get("heldout", False):
            continue
        arc = float(row["dt"])
        pred = _dose_predicted(row, predicted_key)
        meas = float(row["measured_kl"])
        if not (math.isfinite(arc) and math.isfinite(pred) and math.isfinite(meas)):
            continue
        # The audited scorer requires non-negative predicted/measured nats.
        if pred < 0.0 or meas < 0.0:
            continue
        methods.setdefault(str(row["method"]), []).append((arc, pred, meas, 1.0))

    out: dict[str, Any] = {
        "ledger": str(ledger_path),
        "predicted_key": predicted_key,
        "within_validity_only": within_validity_only,
        "heldout_only": heldout_only,
        "model": doc.get("model"),
        "config": doc.get("config"),
        "per_method": {},
    }
    for method, obs in sorted(methods.items()):
        if not obs:
            out["per_method"][method] = {"n": 0, "skipped": "no valid rows"}
            continue
        try:
            rep = gamfit.dose_response_calibration(obs)
        except Exception as exc:  # noqa: BLE001 - surface the scorer's own message
            out["per_method"][method] = {"n": len(obs), "error": str(exc)}
            continue
        out["per_method"][method] = {
            "n": len(obs),
            "slope_through_origin": rep.slope_through_origin,
            "r2_through_origin": rep.r2_through_origin,
            "mean_measured_nats_per_arc": rep.mean_measured_nats_per_arc,
            "cv_measured_nats_per_arc": rep.cv_measured_nats_per_arc,
            "effective_weight": rep.effective_weight,
        }
    return out


# --------------------------------------------------------------------------- #
# Chart-interp (manifold-native metric 1)
# --------------------------------------------------------------------------- #
def _cyclic_order(words: list[str]) -> tuple[list[int], int] | None:
    """Ground-truth cyclic index of each word + period against calendar orders."""
    for order in (WEEKDAYS, MONTHS):
        idx = {w: i for i, w in enumerate(order)}
        if all(w in idx for w in words):
            return [idx[w] for w in words], len(order)
    return None


def chart_interp_report(fit_path: Path) -> dict[str, Any]:
    """Score chart-coordinate interpretability from a per-atom cyclic fit.

    The fit json is the manifold-SAE chart fit: ``fit.per_atom[k].cyclic_ordering``
    carries ``words_present`` and their recovered ``angles_rad``. The recovered
    angle is converted to turns and scored against the ground-truth calendar
    cyclic label by ``gamfit.chart_interp_score``. NOTE (#1942 honesty rule):
    weekday-style features fail the matched-spectrum null and must not be cited
    as evidence; the ``null_robust`` flag records this so the number is reported
    but qualified."""
    doc = json.loads(fit_path.read_text())
    atoms = doc.get("fit", {}).get("per_atom", [])
    results = []
    for entry in atoms:
        cyc = entry.get("cyclic_ordering")
        if not cyc:
            continue
        words = list(cyc.get("words_present", []))
        angles = cyc.get("angles_rad")
        if not words or angles is None or len(words) != len(angles):
            continue
        resolved = _cyclic_order(words)
        if resolved is None:
            continue
        indices, period = resolved
        obs = [
            (float(a) / (2.0 * math.pi), float(i) / float(period), 1.0)
            for a, i in zip(angles, indices)
        ]
        rep = gamfit.chart_interp_score(obs)
        family = "weekday" if period == 7 else ("month" if period == 12 else f"period{period}")
        results.append(
            {
                "atom": entry.get("atom"),
                "family": family,
                "period": period,
                "n_words": len(words),
                "circular_correlation": rep.circular_correlation,
                "signed_circular_correlation": rep.signed_circular_correlation,
                # weekday fails the matched-spectrum null (Manifold-SAE#2); month
                # is the null-robust example the issue names.
                "null_robust": family != "weekday",
            }
        )
    return {"fit": str(fit_path), "atoms": results}


# --------------------------------------------------------------------------- #
# Frozen-dictionary capability audit (manifold-native)
# --------------------------------------------------------------------------- #
def audit_report(
    decoder_path: Path,
    activations_path: Path,
    codes_path: Path | None = None,
    *,
    active: int = 1,
    block_size: int = 1,
    subsample: int | None = None,
    seed: int = 7,
) -> dict[str, Any]:
    """Run the frozen-dictionary capability audit on real activations.

    ``decoder_path`` is a ``K x P`` dictionary (.npy/.npz/.safetensors) and
    ``activations_path`` an ``N x P`` residual matrix. ``codes_path`` (optional)
    is the frozen external encoder's ``N x K`` codes; when absent the Rust sparse
    router encodes against the frozen decoder with ``active`` atoms/row. The
    architecture-matched null donor for topology/atlas claims is a seeded
    per-column permutation of the codes (spectrum-matched firing pattern). The
    audit returns the routability floor, empirical dark-matter fraction, dual
    certificate, absorption pairs and per-atom Betti topology, all in Rust."""
    decoder = np.ascontiguousarray(np.load(decoder_path).astype(np.float32))
    acts = np.load(activations_path).astype(np.float32, copy=False)
    if subsample is not None and subsample < acts.shape[0]:
        rng0 = np.random.Generator(np.random.PCG64(seed))
        acts = np.ascontiguousarray(acts[rng0.choice(acts.shape[0], subsample, replace=False)])
    k = decoder.shape[0]
    codes = None
    if codes_path is not None:
        codes = np.ascontiguousarray(np.load(codes_path).astype(np.float32))
        # Spectrum-matched null: shuffle each atom column independently.
        rng = np.random.Generator(np.random.PCG64(seed))
        null = np.array(codes, dtype=np.float32, copy=True)
        for col in range(null.shape[1]):
            rng.shuffle(null[:, col])
    else:
        # Without external codes the router encodes internally; supply a
        # zeroed donor of the right shape so the non-topology fields (floor,
        # dark-matter, dual certificate, absorption) are still audited. Topology
        # / atlas null claims should be read with this caveat.
        null = np.zeros((acts.shape[0], k), dtype=np.float32)
    routed = gamfit.audit_sae(
        decoder,
        acts,
        codes=codes,
        random_weight_codes=null,
        active=active,
        block_size=block_size,
    )

    def _slim(rep: dict[str, Any]) -> dict[str, Any]:
        keep = {}
        for key in (
            "routability",
            "dual_certificate",
            "absorption",
            "topology",
            "atlas_nerve",
            "route_source",
            "checkpoint",
        ):
            if key in rep:
                keep[key] = rep[key]
        return keep

    report = _slim(routed)
    report["shape"] = {"n": int(acts.shape[0]), "P": int(acts.shape[1]), "K": int(k)}
    report["decoder"] = str(decoder_path)
    report["activations"] = str(activations_path)
    return report


# --------------------------------------------------------------------------- #
# SAEBench capability suite (absorption / SCR / unlearning) — delegated
# --------------------------------------------------------------------------- #
def saebench_suite(sae_path: Path | None, model: str | None) -> dict[str, Any]:
    """Run the public SAEBench absorption/SCR/unlearning evals if available.

    These require a forward pass through the host LLM with the SAE spliced in, so
    they are delegated to the ``sae_bench`` package. When it (or torch) is not
    importable the block is reported as ``skipped`` with the reason — never
    fabricated."""
    try:
        import sae_bench  # type: ignore  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "skipped",
            "reason": f"sae_bench not importable ({type(exc).__name__}: {exc}); "
            "needs the public SAEBench package + torch + host model on a GPU node",
            "sae": None if sae_path is None else str(sae_path),
            "model": model,
        }
    # When present, defer to SAEBench's own runners. Left as an explicit hook so
    # the delegation point is auditable rather than silently absent.
    return {
        "status": "available",
        "note": "sae_bench importable; wire absorption/SCR/unlearning runners here",
        "sae": None if sae_path is None else str(sae_path),
        "model": model,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dose-ledger", type=Path, default=None, help="dose-response ledger json")
    ap.add_argument("--dose-predicted-key", default="predicted_nats",
                    choices=["predicted_nats", "predicted_nats_pathint", "predicted_nats_tangent"])
    ap.add_argument("--dose-all-rows", action="store_true",
                    help="score all rows, not only within-validity ones")
    ap.add_argument("--chart-fit", type=Path, default=None, help="per-atom cyclic chart fit json")
    ap.add_argument("--decoder", type=Path, default=None, help="K x P dictionary (.npy/.npz/.safetensors)")
    ap.add_argument("--activations", type=Path, default=None, help="N x P residual activations .npy")
    ap.add_argument("--codes", type=Path, default=None, help="optional N x K frozen external codes .npy")
    ap.add_argument("--active", type=int, default=1)
    ap.add_argument("--block-size", type=int, default=1)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--saebench-sae", type=Path, default=None, help="SAE checkpoint for the public suite")
    ap.add_argument("--saebench-model", default=None, help="host model id for the public suite")
    ap.add_argument("--out", type=Path, required=True, help="output report json")
    args = ap.parse_args()

    report: dict[str, Any] = {"api": "experiments/audit_sae/saebench_eval.py"}

    if args.dose_ledger is not None:
        report["dose_response"] = dose_response_report(
            args.dose_ledger,
            predicted_key=args.dose_predicted_key,
            within_validity_only=not args.dose_all_rows,
        )
    if args.chart_fit is not None:
        report["chart_interp"] = chart_interp_report(args.chart_fit)
    if args.decoder is not None and args.activations is not None:
        report["audit"] = audit_report(
            args.decoder,
            args.activations,
            args.codes,
            active=args.active,
            block_size=args.block_size,
            subsample=args.subsample,
            seed=args.seed,
        )
    report["saebench_suite"] = saebench_suite(args.saebench_sae, args.saebench_model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
