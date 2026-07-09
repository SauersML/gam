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
import os
from pathlib import Path
from typing import Any

import numpy as np

import gamfit


# --------------------------------------------------------------------------- #
# Deploy-skew preflight
# --------------------------------------------------------------------------- #
# node1/node2 are targeted-job-only; a job must never burn a slot to
# AttributeError minutes into a model load because the installed gamfit wheel
# predates a metric binding this driver calls. Assert the required pyffi metric
# entry points exist at entry with an actionable upgrade message.
_REQUIRED_GAMFIT_API = (
    "chart_interp_score",
    "dose_response_calibration",
    "audit_sae",
)


def _preflight_gamfit() -> None:
    import sys

    pyver = ".".join(str(v) for v in sys.version_info[:3])
    missing = [name for name in _REQUIRED_GAMFIT_API if not hasattr(gamfit, name)]
    if missing:
        raise SystemExit(
            f"[saebench_eval] python {pyver} ({sys.executable}); installed gamfit "
            f"{getattr(gamfit, '__version__', '?')} at {os.path.dirname(gamfit.__file__)} is "
            f"missing required metric entry point(s): {', '.join(missing)}. Upgrade the venv "
            f"wheel to a build that exposes the #1942 SAEBench scorers (>= the commit that "
            f"landed gamfit.chart_interp_score / dose_response_calibration / audit_sae), and "
            f"note gamfit needs python >= 3.10 (no wheel for the MSI-node default 3.6)."
        )


# --------------------------------------------------------------------------- #
# Dose-response calibration (manifold-native metric 2)
# --------------------------------------------------------------------------- #
# Canonical weekday/month cyclic label order for chart-interp ground truth.
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# --------------------------------------------------------------------------- #
# SAEBench-shaped report envelope
# --------------------------------------------------------------------------- #
# SAEBench writes every eval as one JSON object with a fixed shell: an
# ``eval_type_id`` naming the eval, the ``eval_config`` that produced it, a flat
# ``eval_result_metrics`` map of {category: {metric_name: value}}, and an
# ``eval_result_details`` list of per-item rows. The public absorption/SCR/
# unlearning runners emit exactly this; wrapping the manifold-native metrics in
# the same shell lets one downstream tool (or the public scoreboard schema)
# consume every arm of this driver uniformly instead of special-casing ours.
def _saebench_envelope(
    eval_type_id: str,
    eval_config: dict[str, Any],
    eval_result_metrics: dict[str, dict[str, Any]],
    *,
    details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "eval_type_id": eval_type_id,
        "eval_config": eval_config,
        "eval_result_metrics": eval_result_metrics,
        "eval_result_details": details if details is not None else [],
    }


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
    # SAEBench-shaped view: the headline `manifold` method's calibration R² and
    # the unit-speed constancy CV are the two scalars the dosimetry figure turns
    # on; every scored method is carried as a per-item detail row.
    headline = out["per_method"].get("manifold")
    metrics = {"dose_response_calibration": {}}
    if headline and "r2_through_origin" in headline:
        metrics["dose_response_calibration"] = {
            "r2_through_origin": headline["r2_through_origin"],
            "slope_through_origin": headline["slope_through_origin"],
            "cv_measured_nats_per_arc": headline["cv_measured_nats_per_arc"],
        }
    out["eval_output"] = _saebench_envelope(
        "dose_response_calibration",
        {
            "predicted_key": predicted_key,
            "within_validity_only": within_validity_only,
            "heldout_only": heldout_only,
            "model": out["model"],
        },
        metrics,
        details=[{"method": m, **v} for m, v in sorted(out["per_method"].items())],
    )
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
    # SAEBench-shaped view: report the mean circular correlation over the
    # null-robust atoms only (weekday-style atoms fail the matched-spectrum null
    # and must not be scored into the headline, per #1942's honesty rule).
    null_robust = [r for r in results if r["null_robust"]]
    metrics = {"chart_interp": {"n_atoms": len(results), "n_null_robust": len(null_robust)}}
    if null_robust:
        metrics["chart_interp"]["mean_circular_correlation_null_robust"] = float(
            np.mean([r["circular_correlation"] for r in null_robust])
        )
    eval_output = _saebench_envelope(
        "chart_interp", {"fit": str(fit_path)}, metrics, details=results
    )
    return {"fit": str(fit_path), "atoms": results, "eval_output": eval_output}


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
    # SAEBench-shaped view: surface the frozen-dictionary capability scalars
    # (routability floor vs empirical, dark-matter/dual-certificate coverage,
    # absorption-pair count, Betti signature) as one metric block.
    metrics: dict[str, Any] = {"frozen_dictionary_audit": {}}
    rout = report.get("routability")
    if isinstance(rout, dict):
        metrics["frozen_dictionary_audit"]["routability_floor"] = rout.get("floor")
        metrics["frozen_dictionary_audit"]["fraction_below_floor"] = rout.get(
            "fraction_below_floor"
        )
    cert = report.get("dual_certificate")
    if isinstance(cert, dict):
        metrics["frozen_dictionary_audit"]["frac_certified"] = cert.get("frac_certified")
    absorption = report.get("absorption")
    if isinstance(absorption, dict) and "pairs" in absorption:
        metrics["frozen_dictionary_audit"]["n_absorption_pairs"] = len(absorption["pairs"])
    report["eval_output"] = _saebench_envelope(
        "frozen_dictionary_audit",
        {
            "decoder": str(decoder_path),
            "activations": str(activations_path),
            "active": active,
            "block_size": block_size,
        },
        metrics,
    )
    return report


# --------------------------------------------------------------------------- #
# SAEBench capability suite (absorption / SCR / unlearning) — delegated
# --------------------------------------------------------------------------- #
# What each delegated metric is (the reference SAEBench protocol we call into,
# so the delegation point is auditable even though the math lives in sae_bench):
#
#   * absorption      — feature-splitting probe deltas. SAEBench trains a linear
#                       probe for a concept (first-letter spelling), then checks
#                       whether the SAE latent that "should" fire for that concept
#                       is absorbed into a more general latent on some tokens; the
#                       mean absorption fraction + main/split feature counts are the
#                       score. Lower is better.
#   * scr             — spurious-correlation removal (SHIFT). A probe is trained on
#                       a biased task (bias_in_bios profession vs. gender), the
#                       top-N SAE latents most aligned with the spurious attribute
#                       are ablated, and probe accuracy *recovery* on the debiased
#                       axis is measured. Higher is better. (SAEBench's scr_and_tpp
#                       runner also emits TPP, the targeted-probe-perturbation
#                       control; both come back in the same call.)
#   * unlearning      — targeted-concept accuracy drop at matched side-effect. The
#                       WMDP-bio hazardous latents are clamped off; the score is the
#                       accuracy drop on forget-set questions minus the drop on a
#                       retain set (side-effect budget). Higher is better.
#   * sparse_probing  — k-sparse concept detection: probe accuracy from the top-k
#                       most concept-discriminative latents. The disentanglement
#                       axis #1942 argues matters more than reconstruction EV.
#
# We do NOT reimplement any of that math; we build a SAEBench-compatible SAE from
# our trained dictionary and call sae_bench's own runners.
_SAEBENCH_EVALS = {
    "absorption": ("sae_bench.evals.absorption.main", "sae_bench.evals.absorption.eval_config", "AbsorptionEvalConfig"),
    "scr": ("sae_bench.evals.scr_and_tpp.main", "sae_bench.evals.scr_and_tpp.eval_config", "ScrAndTppEvalConfig"),
    "unlearning": ("sae_bench.evals.unlearning.main", "sae_bench.evals.unlearning.eval_config", "UnlearningEvalConfig"),
    "sparse_probing": ("sae_bench.evals.sparse_probing.main", "sae_bench.evals.sparse_probing.eval_config", "SparseProbingEvalConfig"),
}


def _load_full_sae_tensors(path: Path) -> dict[str, Any]:
    """Load a full encoder+decoder SAE checkpoint (numpy) for splicing.

    Accepts ``.safetensors`` or ``.npz`` carrying ``W_enc, b_enc, W_dec, b_dec``
    plus either ``threshold`` (JumpReLU) or an integer ``k`` (TopK). Orientation
    is normalised downstream from the decoder shape, not assumed here."""
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.numpy import load_file

        tensors = dict(load_file(str(path)))
    elif suffix == ".npz":
        with np.load(path) as archive:
            tensors = {key: archive[key] for key in archive.files}
    else:
        raise ValueError(f"unsupported SAE checkpoint {suffix!r}; expected .safetensors or .npz")
    return tensors


def _build_saebench_sae(
    tensors: dict[str, Any],
    *,
    arch: str,
    model_name: str,
    hook_layer: int,
    hook_name: str | None,
    device: str,
    dtype_str: str,
    topk: int | None,
):
    """Wrap our trained dictionary as a SAEBench custom SAE (torch module).

    Subclasses ``sae_bench.custom_saes.base_sae.BaseSAE`` (the officially
    supported custom-SAE path) so SAEBench sees a proper ``cfg`` and the standard
    ``encode``/``decode`` interface, and loads our ``W_enc/W_dec/b_enc/b_dec`` into
    it. ``encode`` implements the TopK or JumpReLU nonlinearity matching how the
    dictionary was trained. Constructor kwargs are signature-filtered so this
    survives ``BaseSAE`` argument drift across sae_bench versions."""
    import inspect

    import torch
    from sae_bench.custom_saes.base_sae import BaseSAE

    def _t(name: str) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(tensors[name])).to(dtype=getattr(torch, dtype_str))

    w_dec = _t("W_dec")  # SAEBench convention: (d_sae, d_in)
    if w_dec.ndim != 2:
        raise ValueError(f"W_dec must be 2-D; got shape {tuple(w_dec.shape)}")
    d_sae, d_in = int(w_dec.shape[0]), int(w_dec.shape[1])
    w_enc = _t("W_enc")
    # Normalise encoder orientation to (d_in, d_sae).
    if w_enc.shape == (d_sae, d_in):
        w_enc = w_enc.t().contiguous()
    elif w_enc.shape != (d_in, d_sae):
        raise ValueError(f"W_enc shape {tuple(w_enc.shape)} is not (d_in,d_sae)=({d_in},{d_sae}) nor its transpose")
    b_enc = _t("b_enc").reshape(-1) if "b_enc" in tensors else torch.zeros(d_sae, dtype=w_enc.dtype)
    b_dec = _t("b_dec").reshape(-1) if "b_dec" in tensors else torch.zeros(d_in, dtype=w_enc.dtype)
    threshold = _t("threshold").reshape(-1) if "threshold" in tensors else None
    if arch == "topk" and topk is None:
        topk = int(tensors["k"]) if "k" in tensors else None
    if arch == "topk" and topk is None:
        raise ValueError("TopK SAE needs --saebench-k (or a `k` entry in the checkpoint)")
    if arch == "jumprelu" and threshold is None:
        raise ValueError("JumpReLU SAE needs a `threshold` entry in the checkpoint")

    hook = hook_name or f"blocks.{hook_layer}.hook_resid_post"
    base_kwargs = {
        "d_in": d_in,
        "d_sae": d_sae,
        "model_name": model_name,
        "hook_layer": hook_layer,
        "hook_name": hook,
        "device": device,
        "dtype": getattr(torch, dtype_str),
    }
    accepted = set(inspect.signature(BaseSAE.__init__).parameters)
    base_kwargs = {k: v for k, v in base_kwargs.items() if k in accepted}

    class _GamDictionarySAE(BaseSAE):
        def __init__(self) -> None:
            super().__init__(**base_kwargs)
            with torch.no_grad():
                self.W_enc.data = w_enc.to(self.W_enc.device, self.W_enc.dtype)
                self.W_dec.data = w_dec.to(self.W_dec.device, self.W_dec.dtype)
                self.b_enc.data = b_enc.to(self.b_enc.device, self.b_enc.dtype)
                self.b_dec.data = b_dec.to(self.b_dec.device, self.b_dec.dtype)
            self._arch = arch
            self._topk = topk
            self._threshold = None if threshold is None else threshold.to(self.W_enc.device, self.W_enc.dtype)

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            pre = (x - self.b_dec) @ self.W_enc + self.b_enc
            if self._arch == "topk":
                topk_vals, topk_idx = pre.topk(self._topk, dim=-1)
                out = torch.zeros_like(pre)
                out.scatter_(-1, topk_idx, torch.relu(topk_vals))
                return out
            # JumpReLU: keep pre-activations above the per-latent threshold.
            return pre * (pre > self._threshold)

        def decode(self, feature_acts: "torch.Tensor") -> "torch.Tensor":
            return feature_acts @ self.W_dec + self.b_dec

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.decode(self.encode(x))

    sae = _GamDictionarySAE()
    return sae, {"d_in": d_in, "d_sae": d_sae, "arch": arch, "hook_name": hook, "topk": topk}


def _run_one_saebench_eval(
    eval_key: str,
    selected_saes: list,
    device: str,
    output_dir: Path,
    model_name: str,
    force_rerun: bool,
) -> dict[str, Any]:
    """Build the eval's config and call its ``run_eval`` with introspected args."""
    import importlib
    import inspect

    main_mod_name, cfg_mod_name, cfg_cls_name = _SAEBENCH_EVALS[eval_key]
    try:
        main_mod = importlib.import_module(main_mod_name)
        cfg_mod = importlib.import_module(cfg_mod_name)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": f"import {main_mod_name} failed ({type(exc).__name__}: {exc})"}
    cfg_cls = getattr(cfg_mod, cfg_cls_name, None)
    if cfg_cls is None:
        return {"status": "error", "reason": f"{cfg_mod_name} has no {cfg_cls_name}"}
    # Build the config, filtering to the fields this version actually declares.
    cfg_fields = {f.name for f in getattr(cfg_cls, "__dataclass_fields__", {}).values()}
    cfg_kwargs = {"model_name": model_name}
    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in cfg_fields}
    try:
        config = cfg_cls(**cfg_kwargs)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": f"{cfg_cls_name}(**{cfg_kwargs}) failed ({type(exc).__name__}: {exc})"}
    run_eval = getattr(main_mod, "run_eval", None)
    if run_eval is None:
        return {"status": "error", "reason": f"{main_mod_name} has no run_eval"}
    out_sub = output_dir / eval_key
    out_sub.mkdir(parents=True, exist_ok=True)
    call_kwargs = {
        "config": config,
        "selected_saes": selected_saes,
        "device": device,
        "output_path": str(out_sub),
        "output_folder": str(out_sub),
        "force_rerun": force_rerun,
    }
    accepted = set(inspect.signature(run_eval).parameters)
    call_kwargs = {k: v for k, v in call_kwargs.items() if k in accepted}
    try:
        result = run_eval(**call_kwargs)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": f"run_eval failed ({type(exc).__name__}: {exc})", "output_dir": str(out_sub)}
    # SAEBench runners variously return the metrics dict and/or write json files;
    # capture the return value (JSON-coerced so a rich object never breaks the
    # report write) and point at the on-disk artifacts.
    produced = sorted(str(p) for p in out_sub.rglob("*.json"))
    return {"status": "ran", "result": _jsonable(result), "output_dir": str(out_sub), "produced_json": produced}


def _jsonable(obj: Any) -> Any:
    """Best-effort JSON-safe view of a SAEBench runner's return value."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        pass
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    for attr in ("to_dict", "model_dump", "__dict__"):
        member = getattr(obj, attr, None)
        if callable(member):
            try:
                return _jsonable(member())
            except Exception:  # noqa: BLE001
                break
        if isinstance(member, dict):
            return _jsonable(member)
    return repr(obj)


def saebench_suite(
    sae_path: Path | None,
    model: str | None,
    *,
    evals: list[str] | None = None,
    arch: str = "topk",
    hook_layer: int = 0,
    hook_name: str | None = None,
    device: str = "cuda",
    dtype_str: str = "float32",
    topk: int | None = None,
    output_dir: Path | None = None,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Run the public SAEBench absorption/SCR/unlearning/sparse-probing evals.

    These require a forward pass through the host LLM with our SAE spliced in, so
    they are delegated to the ``sae_bench`` package: we build a SAEBench-compatible
    SAE from ``sae_path`` and call each eval's own ``run_eval``. When ``sae_bench``
    (or torch, or the checkpoint) is missing the block is reported as ``skipped``/
    ``error`` with the reason — never fabricated. All metric math is sae_bench's."""
    evals = evals or ["absorption", "scr", "unlearning", "sparse_probing"]
    unknown = [e for e in evals if e not in _SAEBENCH_EVALS]
    if unknown:
        return {"status": "error", "reason": f"unknown eval(s) {unknown}; choose from {sorted(_SAEBENCH_EVALS)}"}
    try:
        import sae_bench  # type: ignore  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "skipped",
            "reason": f"sae_bench/torch not importable ({type(exc).__name__}: {exc}); "
            "needs the public SAEBench package + torch + host model on a GPU node",
            "sae": None if sae_path is None else str(sae_path),
            "model": model,
        }
    if sae_path is None or model is None:
        return {
            "status": "skipped",
            "reason": "the public suite needs both --saebench-sae (encoder+decoder checkpoint) "
            "and --saebench-model (host model id); one or both were omitted",
            "sae": None if sae_path is None else str(sae_path),
            "model": model,
        }
    try:
        tensors = _load_full_sae_tensors(sae_path)
        sae, sae_meta = _build_saebench_sae(
            tensors, arch=arch, model_name=model, hook_layer=hook_layer,
            hook_name=hook_name, device=device, dtype_str=dtype_str, topk=topk,
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": f"building SAEBench SAE from {sae_path} failed ({type(exc).__name__}: {exc})",
                "sae": str(sae_path), "model": model}
    out_root = output_dir or (sae_path.parent / "saebench_out")
    out_root.mkdir(parents=True, exist_ok=True)
    sae_name = sae_path.stem
    selected_saes = [(sae_name, sae)]
    per_eval: dict[str, Any] = {}
    for eval_key in evals:
        per_eval[eval_key] = _run_one_saebench_eval(
            eval_key, selected_saes, device, out_root, model, force_rerun
        )
    return {
        "status": "ran",
        "sae": str(sae_path),
        "sae_meta": sae_meta,
        "model": model,
        "device": device,
        "output_dir": str(out_root),
        "per_eval": per_eval,
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
    ap.add_argument("--saebench-sae", type=Path, default=None,
                    help="encoder+decoder SAE checkpoint (.safetensors/.npz) for the public suite")
    ap.add_argument("--saebench-model", default=None, help="host model id for the public suite (e.g. gemma-2-2b)")
    ap.add_argument("--saebench-evals", default="absorption,scr,unlearning,sparse_probing",
                    help="comma list of public evals to run")
    ap.add_argument("--saebench-arch", default="topk", choices=["topk", "jumprelu"],
                    help="encoder nonlinearity of the checkpoint")
    ap.add_argument("--saebench-k", type=int, default=None, help="TopK k (if not stored in the checkpoint)")
    ap.add_argument("--saebench-hook-layer", type=int, default=0, help="host-model residual layer the SAE reads")
    ap.add_argument("--saebench-hook-name", default=None,
                    help="explicit hook name (default blocks.{layer}.hook_resid_post)")
    ap.add_argument("--saebench-device", default="cuda", help="torch device for the host-model forward pass")
    ap.add_argument("--saebench-dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                    help="dtype for the spliced SAE parameters")
    ap.add_argument("--saebench-output-dir", type=Path, default=None,
                    help="where SAEBench writes its own per-eval json (default <sae>.parent/saebench_out)")
    ap.add_argument("--saebench-force-rerun", action="store_true", help="ignore SAEBench's cached artifacts")
    ap.add_argument("--out", type=Path, required=True, help="output report json")
    args = ap.parse_args()

    # Fail-fast deploy-skew guard before any ledger read / SAE build / model load.
    _preflight_gamfit()

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
    report["saebench_suite"] = saebench_suite(
        args.saebench_sae,
        args.saebench_model,
        evals=[e.strip() for e in args.saebench_evals.split(",") if e.strip()],
        arch=args.saebench_arch,
        hook_layer=args.saebench_hook_layer,
        hook_name=args.saebench_hook_name,
        device=args.saebench_device,
        dtype_str=args.saebench_dtype,
        topk=args.saebench_k,
        output_dir=args.saebench_output_dir,
        force_rerun=args.saebench_force_rerun,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
