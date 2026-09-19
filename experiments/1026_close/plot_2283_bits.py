#!/usr/bin/env python3
"""Plot the #2283 paired Eq-4 rows against the crossover theorem's prediction.

The absolute score is thousands of bits per token and the theorem predicts a
margin of a few tens, so a plot of the two totals shows two indistinguishable
lines. The figure is therefore built around the MARGIN, which is the quantity the
theorem actually makes a claim about:

* **left** — the two arms' bits at each fixed-distortion R2 operating point, for
  scale and for the shape of the distortion sweep.
* **middle** — ``external - hybrid`` at each target against the theorem's
  closed-form prediction. That prediction is a-priori and flat in R2: the
  faithful ``k_flat`` config equalises the decoder scalar counts exactly, so
  ``Ddict = 0``; a circle-class chart has span ``s = d + 1 = 2``, so the
  theorem's code term ``(s-d-1)*0.5*log2(lambda/delta)`` is identically zero and
  its matched-recon residual delta is zero; what remains is the support term,
  which is the scorer's cardinality-then-subset code ``log2(G+1) + log2 C(G, L0)``
  of the two CONFIGURATIONS and contains no fitted quantity. The hybrid's full-linear-span re-score (every atom charged its whole
  decoder span rather than the ``d+1`` scalars the theorem's ledger names) is
  drawn as the pessimistic end of the measurement at the acceptance target.
* **right** — the four-term delta at the acceptance target, so it is visible
  which term the contest was decided on and whether ``Dcode`` really vanished.
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crossover_theorem_check import support_code_bits  # noqa: E402

TARGETS = (0.80, 0.90, 0.95, 0.99)
ACCEPTANCE_TARGET = 0.99
EXTERNAL_COLOR = "#B4413C"
HYBRID_COLOR = "#1F5C8B"
THEOREM_COLOR = "#1D7874"


def _rows(path: str, run_id: str) -> tuple[dict, dict | None]:
    """Return the arm results plus the flat-checkpoint record, if any.

    The checkpoint record carries the hybrid CONFIGURATION (`k_flat`,
    `curved_atoms`, `curved_k`, `d_atom`) even when the curved-resume stage has
    not produced its row yet, which is exactly what the theorem's closed-form
    prediction needs: the prediction is a function of the two configurations and
    the external arm's measurement, never of the hybrid fit.
    """
    found: dict[str, dict] = {}
    checkpoint: dict | None = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("run_id") != run_id:
                continue
            if record.get("record_type") == "result":
                found[record["arm"]] = record
            elif record.get("record_type") == "flat_checkpoint":
                checkpoint = record
    if "external_topk" not in found:
        raise SystemExit(f"{path}: run {run_id!r} has no external_topk result")
    return found, checkpoint


def _declared_config(text: str) -> dict:
    """`K,k_flat,curved_atoms,curved_k,d_atom` for a run with no hybrid record yet.

    The theorem's prediction is a function of the two CONFIGURATIONS and the
    external arm's measurement — it never touches the hybrid fit — so it can be
    drawn before the hybrid tier has run. Taking the configuration as an explicit
    argument keeps that visible: nothing here is read from a fit.
    """
    names = ("K", "k_flat", "curved_atoms", "curved_k", "d_atom")
    parts = text.split(",")
    if len(parts) != len(names):
        raise SystemExit(f"--hybrid-config wants {','.join(names)}, got {text!r}")
    config = {name: int(value) for name, value in zip(names, parts, strict=True)}
    config["top_k"] = None
    return config


def _series(record: dict, key: str) -> list[float]:
    return [float(record[f"bits_{key}_at_r2_{target:g}"]) for target in TARGETS]


def _theorem_support_bits(record: dict) -> tuple[float, float]:
    """Support cost of the two CONFIGURATIONS in closed form (no fitted input).

    External: ``G = K`` atoms named ``L0 = top_k`` at a time. Hybrid: the faithful
    config spends ``curved_atoms * m`` decoder rows on charts instead of flat
    atoms, and each chart firing consumes ``1 + d`` of the active-scalar budget
    where a flat firing consumes one -- so the hybrid names fewer atoms per token
    while transmitting the same number of scalars. Both are priced with the
    scorer's cardinality-then-subset code at their fixed per-row cardinality.
    """
    flat_actives = int(record["top_k"]) - int(record["curved_k"]) * (1 + int(record["d_atom"]))
    external = support_code_bits(int(record["K"]), int(record["top_k"]))
    hybrid = support_code_bits(
        int(record["k_flat"]) + int(record["curved_atoms"]),
        flat_actives + int(record["curved_k"]),
    )
    return external, hybrid


def _terms(record: dict, target: float) -> dict[str, float]:
    return {
        "dictionary": float(record["bits_dictionary_bits"]),
        "support": float(record["bits_support_bits"]),
        "code": float(record[f"bits_code_bits_at_r2_{target:g}"]),
        "residual": float(record[f"bits_resid_bits_at_r2_{target:g}"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--estimator-comparison",
        nargs=2,
        metavar=("RESULTS", "RUN_ID"),
        default=None,
        help="a second run of the SAME fit scored on a different number of "
             "estimation rows. Its external row is drawn against this one, because "
             "the movement between them is the scale a margin has to be read "
             "against",
    )
    parser.add_argument(
        "--hybrid-config",
        default=None,
        help="K,k_flat,curved_atoms,curved_k,d_atom — the hybrid CONFIGURATION to "
             "draw the theorem's prediction for, when the run has produced neither "
             "a hybrid row nor a flat checkpoint to read it from",
    )
    args = parser.parse_args()

    rows, checkpoint = _rows(args.results, args.run_id)
    external = rows["external_topk"]
    hybrid = rows.get("hybrid_rust")
    config = hybrid if hybrid is not None else checkpoint
    if config is None and args.hybrid_config is not None:
        config = _declared_config(args.hybrid_config)
        config["top_k"] = int(external["top_k"])
    if config is None:
        raise SystemExit(
            f"{args.results}: run {args.run_id!r} has neither a hybrid_rust result "
            "nor a flat checkpoint; pass --hybrid-config to draw the prediction"
        )
    external_bits = _series(external, "bits")
    support_external, support_hybrid = _theorem_support_bits(config)
    predicted_margin = support_external - support_hybrid
    predicted_bits = [value - predicted_margin for value in external_bits]
    hybrid_bits = _series(hybrid, "bits") if hybrid is not None else None
    margin = (
        None if hybrid_bits is None
        else [e - h for e, h in zip(external_bits, hybrid_bits, strict=True)]
    )
    full_span = (
        None if hybrid is None
        else hybrid.get("bits_faithfulness_audit", {}).get("bracket", {}).get("full_span_bits")
    )

    figure, (curve, gap, split) = plt.subplots(1, 3, figsize=(16.5, 5.2))

    curve.plot(TARGETS, external_bits, "o-", color=EXTERNAL_COLOR, linewidth=2,
               label=f"external TopK, K={external['K']}   (EV {external['ev']:.4f})")
    chart_label = f"{config['k_flat']} flat + {config['curved_atoms']} charts"
    if hybrid_bits is not None:
        curve.plot(TARGETS, hybrid_bits, "s--", color=HYBRID_COLOR, linewidth=2,
                   label=f"hybrid, {chart_label}   (EV {hybrid['ev']:.4f})")
    else:
        curve.plot(TARGETS, predicted_bits, ":", color=THEOREM_COLOR, linewidth=2,
                   label=f"hybrid PREDICTED by the theorem, {chart_label}")
    estimator_shift = None
    if args.estimator_comparison is not None:
        other_rows, _ = _rows(*args.estimator_comparison)
        other = other_rows["external_topk"]
        other_bits = _series(other, "bits")
        curve.plot(TARGETS, other_bits, "o:", color="#7A7A7A", linewidth=1.6,
                   markersize=4,
                   label=f"same fit, scored on {other['bits_rows']} rows")
        estimator_shift = (external_bits[-1] - other_bits[-1], int(other["bits_rows"]))
    curve.set_xlabel("fixed-distortion operating point  $R^2$")
    curve.set_ylabel("Eq-4 description length (bits / token)")
    curve.set_title(f"held-out bits at $R^2$   ({external['bits_rows']} estimation rows)")
    curve.grid(alpha=0.25)
    curve.legend(fontsize=8, loc="upper left")

    gap.axhline(predicted_margin, color=THEOREM_COLOR, linestyle="--", linewidth=1.8,
                label=f"crossover theorem: $\\Delta$support = {predicted_margin:.2f} bits")
    gap.axhline(0.0, color="#777777", linewidth=0.9)
    if margin is not None:
        gap.plot(TARGETS, margin, "s-", color=HYBRID_COLOR, linewidth=2,
                 label="measured  external $-$ hybrid")
    if full_span is not None:
        gap.plot([ACCEPTANCE_TARGET], [external_bits[-1] - full_span], "v",
                 color=HYBRID_COLOR, markersize=10, markerfacecolor="none",
                 label="measured, charts paid at full decoder span")
    gap.set_xlim(min(TARGETS) - 0.01, max(TARGETS) + 0.01)
    gap.set_xlabel("fixed-distortion operating point  $R^2$")
    gap.set_ylabel("bits / token the hybrid saves")
    gap.set_title("the margin, against the theorem's closed form")
    gap.grid(alpha=0.25)
    gap.legend(fontsize=8, loc="best")
    if estimator_shift is not None:
        # The scale the margin has to be read against: the same bit-identical fit,
        # scored on a different number of estimation rows, moves by this much.
        shift, other_rows_count = estimator_shift
        gap.annotate(
            f"for scale: the SAME fit scored on\n"
            f"{other_rows_count} vs {external['bits_rows']} estimation rows\n"
            f"moves by {abs(shift):.1f} bits "
            f"({abs(shift) / max(predicted_margin, 1e-12):.0f}× this margin)",
            xy=(0.5, 0.06), xycoords="axes fraction", ha="center", fontsize=8.5,
            color="#8A3324",
            bbox=dict(boxstyle="round,pad=0.35", fc="#FBEFE9", ec="#8A3324", lw=0.8))

    names = ("dictionary", "support", "code", "residual")
    external_terms = _terms(external, ACCEPTANCE_TARGET)
    positions = range(len(names))
    if hybrid is not None:
        hybrid_terms = _terms(hybrid, ACCEPTANCE_TARGET)
        deltas = [external_terms[name] - hybrid_terms[name] for name in names]
        bars = split.bar(
            list(positions), deltas, 0.55,
            color=[HYBRID_COLOR if value >= 0 else EXTERNAL_COLOR for value in deltas])
    else:
        deltas = [0.0, predicted_margin, 0.0, 0.0]
        bars = split.bar(list(positions), deltas, 0.55, color=THEOREM_COLOR, alpha=0.35,
                         hatch="//", edgecolor=THEOREM_COLOR)
    split.axhline(0.0, color="#333333", linewidth=1.0)
    # The theorem predicts each term's delta exactly: 0 for dictionary (equal
    # decoder scalars by construction), the closed-form support gap, 0 for code
    # (circle class), and no claim for residual (a fit outcome, not a prediction).
    split.plot([0, 1, 2], [0.0, predicted_margin, 0.0], "_", color=THEOREM_COLOR,
               markersize=30, markeredgewidth=2.5, linestyle="none",
               label="theorem's predicted delta")
    for bar, value in zip(bars, deltas, strict=True):
        split.annotate(f"{value:+.2f}", (bar.get_x() + bar.get_width() / 2, value),
                       textcoords="offset points",
                       xytext=(0, 5 if value >= 0 else -13), ha="center", fontsize=9)
    split.set_xticks(list(positions))
    split.set_xticklabels(names)
    split.set_ylabel(f"external $-$ hybrid, bits / token at $R^2$={ACCEPTANCE_TARGET}")
    split.set_title("which term carries the margin"
                    + ("" if hybrid is not None else "  (prediction only)"), pad=12)
    split.grid(alpha=0.25, axis="y")
    split.legend(fontsize=8, loc="best")

    verdict = (
        f"external {external_bits[-1]:.1f} vs hybrid {hybrid_bits[-1]:.1f} bits/token"
        if hybrid_bits is not None
        else f"external {external_bits[-1]:.1f} bits/token measured; hybrid "
             f"{predicted_bits[-1]:.1f} predicted (curved tier not yet run)"
    )
    figure.suptitle(
        f"gam #2283 — Eq-4 bits at $R^2$={ACCEPTANCE_TARGET} on creditscope L30 residual_post "
        f"(N={external['N']}, p={external['p']}, horizon={external['amortization_horizon']}): "
        + verdict,
        fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
