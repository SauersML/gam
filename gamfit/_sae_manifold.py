"""Thin public facade for Rust-backed SAE manifold fitting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ._binding import rust_module
from ._penalty_bridge import (
    GumbelTemperatureSchedule,
    validate_gumbel_schedule_fields as _validate_gumbel_schedule_fields,
)
from ._sae_trust import atom_trust_scores, coerce_sae_trust_diagnostics


_ASSIGNMENT_KINDS: dict[str, str] = {
    "ibp_map": "ibp_map",
    "softmax": "softmax",
    "jumprelu": "jumprelu",
}

_PUBLIC_ASSIGNMENT_KINDS: dict[str, str] = {
    "ibp_map": "ibp_map",
    "softmax": "softmax",
    "jumprelu": "jumprelu",
}


def _e_benjamini_hochberg(log_e_values: list[float], alpha: float) -> list[int]:
    """e-BH confirmed set, mirroring `inference::structure_evidence::e_benjamini_hochberg`.

    Sort claims by descending log e-value; confirm the prefix up to the largest
    rank `k` whose k-th-largest log e-value clears `ln(m) - ln(alpha) - ln(k)`
    (i.e. `e_(k) >= m / (alpha * k)`). FDR <= alpha over the confirmed set under
    arbitrary dependence; valid at any stopping time.
    """
    import math

    m = len(log_e_values)
    if m == 0 or not (alpha > 0.0):
        return []
    order = sorted(range(m), key=lambda i: log_e_values[i], reverse=True)
    k_star = 0
    for rank0, idx in enumerate(order):
        k = rank0 + 1
        if log_e_values[idx] >= math.log(m) - math.log(alpha) - math.log(k):
            k_star = rank0 + 1
    return order[:k_star]


def _structure_claim_label(kind: Any) -> str:
    """Human-readable label for a serialized `ClaimKind` (serde-tagged enum)."""
    if isinstance(kind, str):
        return kind
    if isinstance(kind, Mapping):
        for tag, body in kind.items():
            if tag == "AtomExists":
                return f"atom {body['atom']} exists"
            if tag == "BindingEdge":
                return f"atoms {body['a']}-{body['b']} bound"
            if tag == "GeometryKind":
                return f"atom {body['atom']} geometry={body['kind']}"
            if tag == "Custom":
                return str(body.get("label", "custom"))
            return f"{tag}:{body}"
    return str(kind)


def _structure_claim_atom_exists(kind: Any) -> int | None:
    """Return the atom index for a serialized `ClaimKind::AtomExists`."""
    if isinstance(kind, Mapping):
        body = kind.get("AtomExists")
        if isinstance(body, Mapping) and "atom" in body:
            return int(body["atom"])
    return None


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(v) for v in value]
    return value


def _canonical_assignment(value: str, label: str) -> str:
    name = str(value).strip().lower()
    canon = _ASSIGNMENT_KINDS.get(name)
    if canon is None:
        raise ValueError(
            f"{label}={value!r} is not a recognized assignment kind; "
            f"expected one of {sorted(set(_ASSIGNMENT_KINDS))}"
        )
    return canon


def _coerce_atom_inference(raw: Any) -> list[dict[str, Any]] | None:
    """Normalize the Rust ``atom_inference`` payload list (#1097 / #1103).

    Each Rust entry is a per-atom dict carrying ``atom_index`` / ``atom_name``,
    an optional ``functionals`` block, and an optional ``smooth_significance``
    block whose ``log_e_nonconstant`` is the #1103 any-n-valid split-LRT e-value.
    We pass the report through as plain Python containers (a shallow copy so the
    accessor never aliases the raw FFI object), coercing the e-value to ``float``
    when present. ``None`` for payloads predating the report.
    """
    if raw is None:
        return None
    reports: list[dict[str, Any]] = []
    for entry in raw:
        report = dict(entry)
        sig = report.get("smooth_significance")
        if sig is not None:
            sig = dict(sig)
            log_e = sig.get("log_e_nonconstant")
            sig["log_e_nonconstant"] = None if log_e is None else float(log_e)
            report["smooth_significance"] = sig
        if report.get("functionals") is not None:
            report["functionals"] = dict(report["functionals"])
        reports.append(report)
    return reports


def _coerce_cotrain_report(raw: Any) -> dict[str, Any] | None:
    """Normalize the Rust co-trained amortized-encoder report (#1154)."""
    if raw is None:
        return None
    report = dict(raw)
    required = (
        "recon_consistency",
        "uncertified_fraction",
        "n_uncertified",
        "n_encodes",
    )
    missing = [key for key in required if key not in report]
    if missing:
        raise ValueError(f"SAE cotrain report missing keys: {missing}")
    recon = float(report["recon_consistency"])
    uncert = float(report["uncertified_fraction"])
    n_uncert = int(report["n_uncertified"])
    n_encodes = int(report["n_encodes"])
    if not np.isfinite(recon) or recon < 0.0:
        raise ValueError(f"SAE cotrain recon_consistency must be finite >= 0, got {recon}")
    if not np.isfinite(uncert) or uncert < 0.0 or uncert > 1.0:
        raise ValueError(
            "SAE cotrain uncertified_fraction must be finite in [0, 1], "
            f"got {uncert}"
        )
    if n_uncert < 0 or n_encodes < 0 or n_uncert > n_encodes:
        raise ValueError(
            "SAE cotrain counts must satisfy 0 <= n_uncertified <= n_encodes; "
            f"got {n_uncert} / {n_encodes}"
        )
    return {
        "recon_consistency": recon,
        "uncertified_fraction": uncert,
        "n_uncertified": n_uncert,
        "n_encodes": n_encodes,
    }


def _canonical_public_assignment(value: str) -> str:
    name = str(value).strip().lower()
    canon = _PUBLIC_ASSIGNMENT_KINDS.get(name)
    if canon is None:
        raise ValueError(
            f"assignment={value!r} is not a recognized assignment kind; "
            f"expected one of {sorted(_PUBLIC_ASSIGNMENT_KINDS)}"
        )
    return canon


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _fit_disjoint_periodic_top1(
    x: np.ndarray,
    *,
    bases: list[str],
    dims: list[int],
    assignment: str,
    top_k: int | None,
    penalties: list[str],
    alpha: float,
    learnable_alpha: bool,
    tau: float,
    sparsity_strength: float,
    smoothness: float,
    learning_rate: float,
    max_iter: int,
    random_state: int,
    assignment_label: str,
    jumprelu_threshold: float,
) -> "ManifoldSAE | None":
    """Closed-form fit for visibly separable top-1 periodic atoms."""
    k_atoms = len(bases)
    n_obs, p_out = x.shape
    if (
        k_atoms != 2
        or p_out < 4
        or top_k != 1
        or assignment != "softmax"
        or any(b != "periodic" for b in bases)
        or any(int(d) != 1 for d in dims)
    ):
        return None

    col_profiles = np.square(x).T
    norms = np.linalg.norm(col_profiles, axis=1)
    if np.any(norms <= 1e-12):
        return None
    col_profiles = col_profiles / norms[:, None]
    dist = np.sum((col_profiles[:, None, :] - col_profiles[None, :, :]) ** 2, axis=2)
    c0, c1 = np.unravel_index(int(np.argmax(dist)), dist.shape)
    centers = col_profiles[[c0, c1]].copy()
    labels = np.zeros(p_out, dtype=int)
    for _ in range(8):
        d0 = np.sum((col_profiles - centers[0]) ** 2, axis=1)
        d1 = np.sum((col_profiles - centers[1]) ** 2, axis=1)
        labels = (d1 < d0).astype(int)
        if labels.min() == labels.max():
            return None
        for k in range(2):
            centers[k] = col_profiles[labels == k].mean(axis=0)

    row_energy = np.zeros((n_obs, 2), dtype=float)
    for k in range(2):
        cols = labels == k
        if int(np.sum(cols)) < 2:
            return None
        row_energy[:, k] = np.sum(np.square(x[:, cols]), axis=1)
    total_energy = np.sum(row_energy, axis=1)
    usable = total_energy > 1e-10
    if not np.any(usable):
        return None
    dominance = np.max(row_energy[usable], axis=1) / np.maximum(total_energy[usable], 1e-12)
    if float(np.median(dominance)) < 0.90:
        return None

    winners = np.argmax(row_energy, axis=1)
    if min(int(np.sum(winners == 0)), int(np.sum(winners == 1))) < 3:
        return None

    coords: list[np.ndarray] = []
    decoder_blocks: list[np.ndarray] = []
    fitted = np.zeros_like(x, dtype=float)
    assignments = np.zeros((n_obs, 2), dtype=float)
    assignments[np.arange(n_obs), winners] = 1.0
    for k in range(2):
        rows = winners == k
        cols = labels == k
        block = x[rows][:, cols]
        mean = block.mean(axis=0, keepdims=True)
        centered = block - mean
        try:
            _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        if vt.shape[0] < 2:
            return None
        scores_all = (x[:, cols] - mean) @ vt[:2].T
        phase = np.arctan2(scores_all[:, 1], scores_all[:, 0]) / (2.0 * np.pi)
        phase = phase - np.floor(phase)
        coords.append(np.ascontiguousarray(phase.reshape(-1, 1)))
        phi_rows = np.asarray(
            rust_module().basis_with_jet(
                "periodic",
                np.ascontiguousarray(phase[rows].reshape(-1, 1)),
                {"n_harmonics": 1},
            )[0],
            dtype=float,
        )
        try:
            block_b, *_ = np.linalg.lstsq(phi_rows, x[rows], rcond=None)
        except np.linalg.LinAlgError:
            return None
        decoder_blocks.append(np.ascontiguousarray(block_b))
        fitted[rows] = phi_rows @ block_b

    logits = np.full((n_obs, 2), -4.0, dtype=float)
    logits[np.arange(n_obs), winners] = 4.0
    payload = {
        "atoms": [
            {
                "decoder_B": decoder_blocks[k],
                "basis_kind": "periodic",
                "assignments_z": assignments[:, k],
                "on_atom_coords_t": coords[k],
                "active_dim": 1,
            }
            for k in range(2)
        ],
        "assignments_z": assignments,
        "logits": logits,
        "fitted": fitted,
        "reml_score": float(rust_module().sae_manifold_reconstruction_r2(x, fitted)),
        "chosen_k": 2,
        "atom_plans": [
            {
                "kind": "periodic",
                "latent_dim": 1,
                "n_harmonics": 1,
                "basis_size": 3,
                "duchon_centers": None,
            },
            {
                "kind": "periodic",
                "latent_dim": 1,
                "n_harmonics": 1,
                "basis_size": 3,
                "duchon_centers": None,
            },
        ],
        "dispersion": float(np.mean(np.square(x - fitted))),
        "oos_projection_top1": True,
    }
    return ManifoldSAE.from_payload(
        x,
        payload,
        _topology_for_bases(bases),
        assignment,
        penalties,
        alpha=alpha,
        learnable_alpha=learnable_alpha,
        assignment_label=assignment_label,
        tau=tau,
        sparsity_strength=sparsity_strength,
        smoothness=smoothness,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state,
        top_k=top_k,
        jumprelu_threshold=jumprelu_threshold,
    )


def _functional_basis_params(plan: Mapping[str, Any]) -> dict[str, Any] | None:
    kind = str(plan["kind"]).lower().replace("-", "_")
    if kind in {"periodic", "periodic_spline", "circle"}:
        n_harmonics = int(plan.get("n_harmonics", 0))
        if n_harmonics <= 0:
            basis_size = int(plan.get("basis_size", 0))
            n_harmonics = (basis_size - 1) // 2
        return {"n_harmonics": max(1, n_harmonics)}
    if kind in {"duchon", "euclidean", "euclidean_patch"}:
        centers = plan.get("duchon_centers")
        if centers is None:
            return None
        return {"centers": np.asarray(centers, dtype=float), "m": int(plan["basis_size"])}
    if kind == "sphere":
        return {}
    return None


def activation_statistics(X: Any) -> dict[str, float]:
    """Cheap, scale-free statistics of an activation matrix `X` (n, p) that key
    the adaptive hyperparameter default (#977 measure→improve). All three are
    properties of the centred spectrum, so they transfer across datasets rather
    than overfitting one corpus:

      * ``effective_rank`` — the spectral entropy rank
        ``exp(-Σ pᵢ log pᵢ)`` with ``pᵢ = sᵢ² / Σ sⱼ²`` (participation ratio of
        the singular spectrum). Low ⇒ the signal lives in few directions ⇒ a
        low-dimensional / low intrinsic-rank atom suffices; high ⇒ richer.
      * ``spectral_decay`` — ``s₀ / s_{k}`` at ``k = min(8, p-1)`` (how fast the
        spectrum falls). Sharp decay ⇒ a clean low-harmonic ring; slow decay ⇒
        the curve carries higher harmonics.
      * ``snr`` — ``(Σ top-d² ) / (Σ tail²)`` with ``d = 2`` (signal vs residual
        energy), a proxy for assignment sharpness ⇒ the gate temperature.
    """
    x = np.asarray(X, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = x - x.mean(axis=0, keepdims=True)
    n, p = x.shape
    if n < 2 or p < 1:
        return {"effective_rank": 1.0, "spectral_decay": 1.0, "snr": 1.0}
    s = np.linalg.svd(x, compute_uv=False)
    s2 = s ** 2
    total = float(s2.sum())
    if total <= 0.0:
        return {"effective_rank": 1.0, "spectral_decay": 1.0, "snr": 1.0}
    probs = s2 / total
    nz = probs[probs > 0]
    eff_rank = float(np.exp(-np.sum(nz * np.log(nz))))
    k = min(8, len(s) - 1)
    decay = float(s[0] / max(s[k], 1e-12)) if k >= 1 else 1.0
    d = min(2, len(s2))
    tail = float(s2[d:].sum())
    snr = float(s2[:d].sum() / tail) if tail > 1e-12 else float("inf")
    return {"effective_rank": eff_rank, "spectral_decay": decay, "snr": snr}


def recommend_sae_hyperparams(X: Any) -> dict[str, Any]:
    """Activation-statistics-keyed adaptive default for the manifold-SAE
    hyperparameters `(tau, n_harmonics, intrinsic_rank)` (#977 measure→improve).

    Calibrated against the held-out-EV optimum the on-corpus hillclimb
    (`tests/sae/olmo_research_battery.py`) finds on REAL OLMo L25 activations:
    the OLMo-fixture statistics must map to the measured optimum (asserted in
    `tests/test_sae_adaptive_defaults.py`). The mapping is monotone in the
    spectrum statistics so it generalises off that single corpus instead of
    hard-coding one dataset's argmax.

    The map (intentionally simple, each axis keyed by one statistic):
      * ``intrinsic_rank`` from ``effective_rank``: a higher participation ratio
        of the spectrum buys an extra intrinsic dimension (1 ⇒ low, 2 ⇒ high).
      * ``n_harmonics`` from ``spectral_decay``: sharp decay ⇒ a clean ring at
        1 harmonic; slow decay ⇒ admit a 2nd/3rd harmonic.
      * ``tau`` from ``snr``: high SNR ⇒ sharper gate (lower temperature).
    """
    stats = activation_statistics(X)
    eff = stats["effective_rank"]
    decay = stats["spectral_decay"]
    snr = stats["snr"]

    intrinsic_rank = 2 if eff >= 6.0 else 1
    if decay >= 12.0:
        n_harmonics = 1
    elif decay >= 4.0:
        n_harmonics = 2
    else:
        n_harmonics = 3
    # Sharper assignment when the signal stands well clear of the residual.
    if snr >= 8.0:
        tau = 0.25
    elif snr >= 2.0:
        tau = 0.5
    else:
        tau = 0.7

    return {
        "tau": tau,
        "n_harmonics": n_harmonics,
        "intrinsic_rank": intrinsic_rank,
        "statistics": stats,
    }


def ev_knee_k(
    ev_by_k: Mapping[int, float] | list[tuple[int, float]],
    *,
    mode: str = "kneedle",
    knee_slope_fraction: float = 0.10,
    complexity_penalty: float = 0.05,
    flat_span_tol: float = 1.0e-6,
    return_details: bool = False,
) -> int | dict[str, Any]:
    """Auto-K from an explained-variance-vs-K frontier (#977/#1026).

    This Python helper delegates to Rust ``gam::terms::sae::k_selection``, the
    single source of truth for knee/MDL selection and the endpoint flags
    (``knee``, ``no_knee``, ``linear``, ``flat``).
    """
    from gamfit._binding import rust_module

    items = sorted(
        (ev_by_k.items() if isinstance(ev_by_k, Mapping) else ev_by_k),
        key=lambda kv: kv[0],
    )
    result = dict(
        rust_module().sae_select_k(
            [(int(k), float(v)) for k, v in items],
            mode,
            float(knee_slope_fraction),
            float(complexity_penalty),
            float(flat_span_tol),
        )
    )
    result["k"] = int(result["k"])
    result["ev"] = float(result["ev"])
    result["score"] = float(result["score"])
    return result if return_details else result["k"]


def wager_verdict(
    manifold_ev_by_k: Mapping[int, float],
    linear_ev_by_k: Mapping[int, float],
    *,
    mode: str = "kneedle",
    knee_slope_fraction: float = 0.10,
    complexity_penalty: float = 0.05,
    flat_span_tol: float = 1.0e-6,
) -> dict[str, Any]:
    """The #977 wager, adjudicated on a real manifold-vs-linear EV-vs-K frontier
    (#1026): are curved manifold atoms parameter-efficient relative to a linear
    SAE — do they reach a target EV at strictly lower K?

    Returns a dict with:
      * ``confirmed`` — True iff at some K the manifold EV meets-or-beats the
        BEST linear EV achieved at any (>=) K (i.e. manifold ties the linear
        ceiling at fewer atoms).
      * ``manifold_k`` / ``linear_k`` — the parameter-efficiency statement
        "manifold K=manifold_k EV >= linear K=linear_k EV".
      * ``efficiency_ratio`` — linear_k / manifold_k (>1 ⇒ manifold wins).
      * ``best_linear_ev`` — the linear ceiling used as the bar.

    Honest both ways: if no manifold K reaches the linear ceiling, ``confirmed``
    is False and the verdict reports the EV gap — the wager loses measurably,
    which is itself a finding (structured minority is small; hybrid-with-linear-
    tail is the end state).
    """
    from gamfit._binding import rust_module

    manifold = sorted(manifold_ev_by_k.items(), key=lambda kv: kv[0])
    linear = sorted(linear_ev_by_k.items(), key=lambda kv: kv[0])
    result = dict(
        rust_module().sae_auto_k_recommendation(
            [(int(k), float(v)) for k, v in manifold],
            [(int(k), float(v)) for k, v in linear],
            mode,
            float(knee_slope_fraction),
            float(complexity_penalty),
            float(flat_span_tol),
        )
    )
    best_linear_ev = max(float(v) for _, v in linear)
    best_manifold_ev = max(float(v) for _, v in manifold)
    result.update(
        {
            "k": int(result["k"]),
            "ev": float(result["ev"]),
            "score": float(result["score"]),
            "target_ev": float(result["target_ev"]),
            "manifold_k": None if result["manifold_k"] is None else int(result["manifold_k"]),
            "linear_k": None if result["linear_k"] is None else int(result["linear_k"]),
            "efficiency_ratio": (
                None
                if result["efficiency_ratio"] is None
                else float(result["efficiency_ratio"])
            ),
            "confirmed": bool(result["confirmed"]),
            "best_linear_ev": best_linear_ev,
            "best_manifold_ev": best_manifold_ev,
            "ev_gap": max(0.0, best_linear_ev - best_manifold_ev),
        }
    )
    return result


def _weighted_row_mean(rows: np.ndarray, weights: np.ndarray | None) -> np.ndarray | None:
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[0] == 0 or not np.all(np.isfinite(rows)):
        return None
    if weights is None:
        return np.mean(rows, axis=0)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.shape[0] != rows.shape[0] or not np.all(np.isfinite(weights)):
        return None
    weights = np.maximum(weights, 0.0)
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return None
    return np.einsum("n,nm->m", weights / weight_sum, rows, optimize=True)


def _channel_se_from_decoder_covariance(
    gradient: np.ndarray,
    covariance: np.ndarray | None,
    output_dim: int,
) -> np.ndarray | None:
    if covariance is None:
        return None
    gradient = np.asarray(gradient, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    basis_size = gradient.shape[0]
    if covariance.shape != (basis_size * output_dim, basis_size * output_dim):
        return None
    se = np.zeros(output_dim, dtype=float)
    for channel in range(output_dim):
        idx = np.arange(channel, basis_size * output_dim, output_dim)
        sub = covariance[np.ix_(idx, idx)]
        var = float(gradient @ sub @ gradient)
        if not np.isfinite(var):
            return None
        se[channel] = np.sqrt(max(var, 0.0))
    return se


def _vector_evidence_payload(
    estimate: np.ndarray,
    se: np.ndarray | None = None,
    **extra: Any,
) -> dict[str, Any] | None:
    estimate = np.asarray(estimate, dtype=float)
    if estimate.size == 0 or not np.all(np.isfinite(estimate)):
        return None
    payload: dict[str, Any] = {
        "estimate": estimate.tolist(),
        "norm": float(np.linalg.norm(estimate)),
    }
    if se is not None:
        se = np.asarray(se, dtype=float)
        if se.shape == estimate.shape and np.all(np.isfinite(se)):
            payload["se"] = se.tolist()
    payload.update(extra)
    return payload


def _atom_functional_evidence(
    atom: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    native = atom.get("functional_evidence")
    if native is not None:
        return dict(native)

    params = _functional_basis_params(plan)
    if params is None:
        return None
    coords = np.asarray(atom["on_atom_coords_t"], dtype=float)
    decoder = np.asarray(atom["decoder_B"], dtype=float)
    assignments = np.asarray(atom.get("assignments_z"), dtype=float)
    cov = None if atom.get("decoder_covariance") is None else np.asarray(atom["decoder_covariance"], dtype=float)
    if coords.ndim != 2 or decoder.ndim != 2 or not np.all(np.isfinite(coords)):
        return None
    try:
        phi, jet, _penalty = rust_module().basis_with_jet(
            str(plan["kind"]),
            np.ascontiguousarray(coords),
            params,
        )
    except Exception:
        return None
    phi = np.asarray(phi, dtype=float)
    jet = np.asarray(jet, dtype=float)
    if (
        phi.ndim != 2
        or jet.ndim != 3
        or phi.shape[0] != coords.shape[0]
        or phi.shape[1] != decoder.shape[0]
        or jet.shape[:2] != phi.shape
        or not np.all(np.isfinite(phi))
        or not np.all(np.isfinite(jet))
    ):
        return None

    output_dim = int(decoder.shape[1])
    value_gradient = _weighted_row_mean(phi, assignments)
    if value_gradient is None:
        return None
    average_value = _vector_evidence_payload(
        value_gradient @ decoder,
        _channel_se_from_decoder_covariance(value_gradient, cov, output_dim),
    )

    derivative_estimates = []
    derivative_ses = []
    for axis in range(jet.shape[2]):
        grad = _weighted_row_mean(jet[:, :, axis], assignments)
        if grad is None:
            return None
        derivative_estimates.append(grad @ decoder)
        axis_se = _channel_se_from_decoder_covariance(grad, cov, output_dim)
        if axis_se is not None:
            derivative_ses.append(axis_se)
    derivative_est = np.vstack(derivative_estimates)
    derivative_se = np.vstack(derivative_ses) if len(derivative_ses) == derivative_est.shape[0] else None
    average_derivative = _vector_evidence_payload(derivative_est, derivative_se)

    mean = phi @ decoder
    norm = np.linalg.norm(mean, axis=1)
    peak_idx = int(np.argmax(norm))
    baseline_idx = int(np.argmin(norm))
    contrast_gradient = phi[peak_idx] - phi[baseline_idx]
    peak_contrast = _vector_evidence_payload(
        contrast_gradient @ decoder,
        _channel_se_from_decoder_covariance(contrast_gradient, cov, output_dim),
        from_coord=coords[baseline_idx].tolist(),
        to_coord=coords[peak_idx].tolist(),
    )

    out: dict[str, Any] = {"source": "decoder_covariance_plugin"}
    if average_value is not None:
        out["average_value"] = average_value
    if average_derivative is not None:
        # The conditional-on-fit average derivative E_data[∂g/∂t] of the fitted
        # decoder curve. Deliberately NOT aliased as "marginal_slope": the latent
        # coordinate is a fitted, generated regressor, so this is a descriptive
        # variation of the fitted curve, not a population marginal slope (the
        # same #1097/#1115 honesty correction the native Rust report makes by
        # naming the field `decoder_variation_norm`, never `marginal_slope`).
        out["average_derivative"] = average_derivative
    if peak_contrast is not None:
        out["peak_contrast"] = peak_contrast
    return out if len(out) > 1 else None


@dataclass(slots=True)
class SaeManifoldAtomFit:
    """Per-atom fit payload returned inside :class:`ManifoldSAE`.

    Attributes
    ----------
    basis
        Basis kind used by this atom, for example ``"periodic"``,
        ``"euclidean"``, ``"duchon"``, ``"sphere"``, or ``"torus"``.
    decoder_coefficients
        Decoder basis coefficients ``B_k`` with shape ``(M_k, p)`` where
        ``M_k`` is the atom basis size and ``p`` is the ambient/output
        dimension. Values are in the same units as ``X`` because the basis
        functions are dimensionless.
    assignments
        Per-observation assignment/gate values for this atom, shape ``(N,)``.
        For ``assignment="softmax"`` these are mixture masses; for
        ``"ibp_map"`` and ``"jumprelu"`` these are
        gate activations.
    coords
        Recovered on-atom latent coordinates ``t*`` for the training data,
        shape ``(N, d_k)``. Units are the atom's raw latent coordinate system:
        periodic/circle coordinates are normalized phase coordinates, while
        euclidean/duchon coordinates are raw chart coordinates.
    evidence
        Fit REML score copied from the full SAE result.
    active_dim
        Estimated active intrinsic coordinate dimension for this atom.
    decoder_covariance
        Optional phi-scaled posterior covariance of the flattened decoder
        coefficients, shape ``(M_k * p, M_k * p)`` in row-major
        ``(basis, channel)`` layout. Entries have squared ``X`` units. Present
        on fresh fits when the Rust payload includes posterior uncertainty.
    shape_band_coords
        Optional coordinate grid for the posterior shape band, shape
        ``(G, d_k)``, in the same latent-coordinate units as ``coords``.
    shape_band_mean
        Optional fitted ambient manifold values on ``shape_band_coords``,
        shape ``(G, p)``, in the same units as ``X``.
    shape_band_sd
        Optional per-channel posterior standard deviation of
        ``shape_band_mean``, shape ``(G, p)``, in the same units as ``X``.
    functional_evidence
        Optional per-atom decoder functional evidence. Native Rust/Riesz
        payloads are passed through as-is; otherwise fresh fits may populate a
        conservative plugin block from decoder covariance with
        ``average_value``, ``average_derivative`` (the conditional-on-fit mean
        decoder derivative — NOT a population marginal slope; the latent
        coordinate is a generated regressor), and ``peak_contrast``.
    """

    basis: str
    decoder_coefficients: np.ndarray
    assignments: np.ndarray
    coords: np.ndarray
    evidence: float
    active_dim: int
    # Posterior shape uncertainty. These fields are ``None`` only when the
    # source payload did not include uncertainty arrays. ``decoder_covariance``
    # is the phi-scaled posterior covariance of this atom's decoder
    # coefficients, shape ``(M_k*p, M_k*p)`` in row-major ``(basis, channel)``
    # flat layout. The shape band is the closed-form push-forward to ambient
    # space along the on-atom coordinates: ``shape_band_mean`` is the fitted
    # point ``(G, p)``, ``shape_band_sd`` its per-channel posterior sd
    # ``(G, p)``, at ``shape_band_coords`` ``(G, d_k)``.
    decoder_covariance: np.ndarray | None = None
    shape_band_coords: np.ndarray | None = None
    shape_band_mean: np.ndarray | None = None
    shape_band_sd: np.ndarray | None = None
    functional_evidence: dict[str, Any] | None = None


@dataclass(slots=True)
class SaeManifoldFitResult:
    atoms: list[SaeManifoldAtomFit]
    chosen_k: int
    evidence_by_candidate: dict[int, float]
    comparison: dict[str, Any]
    fitted: np.ndarray
    assignments: np.ndarray
    coords: list[np.ndarray]
    reml_score: float


@dataclass(slots=True)
class ManifoldSAE:
    """Fitted SAE-manifold model returned by :func:`sae_manifold_fit`.

    The main result arrays are ``fitted`` ``(N, p)``, ``assignments`` ``(N, K)``,
    ``coords`` as a list of per-atom ``(N, d_k)`` arrays, ``decoder_blocks`` as
    per-atom ``(M_k, p)`` decoder matrices, and ``atoms`` as detailed
    :class:`SaeManifoldAtomFit` payloads. Metadata records the resolved
    ``assignment`` kind, per-atom topology/basis information, score fields
    (``reml_score``, ``reconstruction_r2``, ``dispersion``), fit controls
    (``alpha``, ``learnable_alpha``, ``tau``, ``top_k``,
    ``jumprelu_threshold``), and cached training data used for exact
    training-set predictions.

    Public helpers include :meth:`predict`/:meth:`reconstruct`,
    :meth:`encode`, :meth:`converged_latents`, :meth:`project`,
    :meth:`shape_uncertainty`, :meth:`coordinate_range`, and
    :meth:`typical_shape`.
    """

    atoms: list[SaeManifoldAtomFit]
    atom_topology: str
    atom_topologies: list[str]
    assignment: str
    assignment_label: str
    primitive_names: list[str]
    fitted: np.ndarray
    assignments: np.ndarray
    coords: list[np.ndarray]
    decoder_blocks: list[np.ndarray]
    basis_specs: list[str]
    reml_score: float
    reconstruction_r2: float
    training_mean: np.ndarray
    training_data: np.ndarray
    low_level: SaeManifoldFitResult
    low_level_logits: np.ndarray
    diagnostics: dict[str, Any]
    _basis_kinds: list[str]
    _atom_dims: list[int]
    _basis_sizes: list[int]
    _n_harmonics: list[int]
    _duchon_centers: list[np.ndarray | None]
    _oos_projection_top1: bool = False
    alpha: float = 1.0
    learnable_alpha: bool = False
    tau: float = 0.5
    sparsity_strength: float = 1.0
    smoothness: float = 1.0
    learning_rate: float = 0.04
    max_iter: int = 50
    random_state: int = 0
    top_k: int | None = None
    jumprelu_threshold: float = 0.0
    solver_plan: dict[str, Any] | None = None
    # Gaussian reconstruction scale phi-hat that scales every per-atom decoder
    # covariance (Cov(beta_k) = phi * S_beta^{-1}[block]).
    dispersion: float = 1.0
    # Provenance of the per-row inner product the fit installed (#980):
    # ``"Euclidean"`` (no shard, bit-identical isotropic path) or
    # ``"OutputFisher"`` (a WP-D output-Fisher shard was supplied and
    # ``RowMetric::OutputFisher`` was installed for the gauge/lens). The
    # likelihood is untouched either way.
    metric_provenance: str = "Euclidean"
    # Per-row output-Fisher truncation diagnostic ``(n,)`` =
    # ``trace(G_n) - sum_{k<=r} lambda_k``, the mass that fell off the captured
    # rank-r subspace. ``None`` when no shard (or no mass_residual) was supplied.
    # Surfaced so a too-small rank ``r`` is visible, not silent.
    fisher_mass_residual: np.ndarray | None = None
    # Additive two-score per-atom lens (#980): for each atom, ``presence``
    # (representational, activation-side, Fisher-free), ``coupling`` (behavioral
    # output-Fisher mass; ``NaN`` under a Euclidean / no-harvest provenance), and
    # ``discrepancy = presence_normalized - coupling_normalized`` (the
    # "represented but not currently used" headline; ``NaN`` when coupling is
    # unavailable). A pure read of the fitted model; it feeds no loss or
    # criterion. ``None`` only for payloads predating the diagnostic.
    atom_two_lens: dict[str, Any] | None = None
    # Residual-gauge certificate (#980): which symmetry group the fit is
    # identified up to. ``group_signature`` names the surviving generator
    # families; ``generators`` lists each enumerated symmetry's pinned/unpinned
    # verdict; ``metric_provenance`` records the inner product it was computed in.
    # Pure read; ``None`` only for payloads predating the diagnostic.
    residual_gauge: dict[str, Any] | None = None
    # Empirical curved-dictionary certificate inputs (#1008): frame incoherence
    # ``mu_hat``, per-atom curvature bounds, activity floors, and an SNR proxy.
    # This is deliberately quantities-only; no global-optimality verdict exists
    # until the theorem threshold is implemented.
    incoherence_report: dict[str, Any] | None = None
    # Per-atom curvature report (#1099, rescoped under #1115).
    # ``atoms[k]`` is ``{"atom": int, "kappa_hat": float}``: the fitted empirical
    # second-fundamental-form sup-norm bound for atom k, a descriptive plug-in
    # geometry summary. A curvature bound is not an estimand with a profiled
    # criterion, so no SE/CI/flatness fields are carried.
    curvature_report: dict[str, Any] | None = None
    # Per-atom smooth-functional inference (#1097 / #1103): one entry per fitted
    # atom, ``{"atom_index": int, "atom_name": str, "functionals": {...} | None,
    # "smooth_significance": {"log_e_nonconstant": float | None} | None}``. The
    # #1103 ``smooth_significance.log_e_nonconstant`` is the any-n-valid split-LRT
    # e-value for "the atom's inner decoder smooth is non-constant" (null =
    # constant), with ``E_{H0}[E] <= 1`` — a large positive ``log_e_nonconstant``
    # is honest evidence the atom carries structure. Surfaced via
    # :meth:`atom_inference`. ``None`` only for payloads predating the report.
    atom_inference_reports: list[dict[str, Any]] | None = None
    # The unified certificate ledger (#16): ONE coherent block consolidating every
    # certificate this fit produced under a shared claim+evidence+verdict shape.
    # ``{"overall": str, "overall_certified": bool, "claims": {claim_id: {"claim":
    # str, "verdict": str, "certified": bool, "evidence": {...}}}}``. ``verdict``
    # is on the conservative ladder ``unavailable < insufficient < certified`` —
    # an absent or below-margin certificate never reads as a pass. The bespoke
    # ``residual_gauge`` / ``incoherence_report`` keys above remain populated with
    # the same values for back-compat; this is the additive canonical surface.
    # ``None`` only for payloads predating the ledger.
    certificates: dict[str, Any] | None = None
    # Anytime-valid structure certificate (#1058 / #984): the e-BH certificate
    # over the structure-search ledger's per-claim e-processes at FDR level α.
    # JSON string ``{"alpha": float, "entries": [{"kind": ..., "log_e": float,
    # "steps": int, "confirmed": bool}, ...]}`` serialized by the Rust core.
    # Surfaced via :meth:`structure_certificate`. ``None`` only for payloads
    # predating the certificate.
    structure_certificate_json: str | None = None
    # Co-trained amortized-encoder diagnostics (#1154), emitted by the Rust
    # fit payload when the Design-A REML + encoder-consistency fold is active.
    # Keys: recon_consistency, uncertified_fraction, n_uncertified, n_encodes.
    cotrain: dict[str, Any] | None = None
    # WP-D output-Fisher shard the fit installed (#980), retained so a follow-up
    # :meth:`steer` call can re-install ``RowMetric::OutputFisher`` and report the
    # path-integrated KL dose. The ``(n, p, r)`` factor stack ``U`` exactly as
    # supplied to ``sae_manifold_fit(..., fisher_factors=...)``. ``None`` under the
    # Euclidean (no-shard) path: steering still returns the geometry (delta /
    # off_manifold_norm) but ``predicted_nats`` / ``validity_radius`` are ``None``
    # (no behavioral axis to measure the dose through).
    fisher_factors: np.ndarray | None = None
    # Which output-Fisher pullback produced ``fisher_factors`` (#980):
    # ``"output_fisher"`` (same-position, the default) or
    # ``"output_fisher_downstream"`` (KV-path aggregate over future positions). A
    # follow-up :meth:`steer` re-installs the matching ``RowMetric`` so the dose
    # is measured in the same geometry the fit's gauge used.
    fisher_provenance: str = "output_fisher"

    def __repr__(self) -> str:
        d_atom = int(self.coords[0].shape[1]) if self.coords else 0
        n, p = (self.fitted.shape if self.fitted.ndim == 2 else (self.fitted.shape[0], 1))
        return (
            f"ManifoldSAE(K={len(self.atoms)}, d_atom={d_atom}, "
            f"atom_topology={self.atom_topology!r}, assignment={self.assignment!r}, "
            f"alpha={self.alpha!r}, learnable_alpha={self.learnable_alpha}, "
            f"n={n}, p={p}, r2={self.reconstruction_r2:.3f})"
        )

    @classmethod
    def from_payload(cls, x: np.ndarray, payload: Mapping[str, Any], topology: str, assignment: str, penalties: list[str], alpha: float = 1.0, learnable_alpha: bool = False, *, assignment_label: str | None = None, tau: float = 0.5, sparsity_strength: float = 1.0, smoothness: float = 1.0, learning_rate: float = 0.04, max_iter: int = 50, random_state: int = 0, top_k: int | None = None, jumprelu_threshold: float = 0.0) -> "ManifoldSAE":
        plans = list(payload["atom_plans"])
        # #977 variable-K boundary contract: the structure search may have GROWN
        # K (evidence-gated births / fissions) or shrunk routing mass; the Rust
        # producer re-derives EVERY per-atom field from the post-search dictionary
        # so each one has length == discovered K. Assert that contract here at the
        # single ingest point rather than letting a producer drift surface as an
        # opaque ``plans[atom_idx]`` IndexError (grown K) or a silent truncation
        # (shrunk lists). ``atom_plans`` is zipped positionally against
        # ``payload["atoms"]`` below, and ``chosen_k`` / ``assignments_z`` /
        # ``logits`` must agree on the same K.
        payload_atoms = list(payload["atoms"])
        k_discovered = len(payload_atoms)
        if len(plans) != k_discovered:
            raise ValueError(
                "SAE payload is inconsistent at the variable-K boundary: "
                f"{k_discovered} atoms but {len(plans)} atom_plans; every "
                "per-atom field must have length == the discovered K"
            )
        chosen_k_declared = int(payload["chosen_k"])
        if chosen_k_declared != k_discovered:
            raise ValueError(
                "SAE payload chosen_k does not match the atom count: "
                f"chosen_k={chosen_k_declared} but {k_discovered} atoms were "
                "emitted; the discovered K must thread through every field"
            )
        for field in ("assignments_z", "logits"):
            arr = np.asarray(payload[field], dtype=float)
            if arr.ndim != 2 or arr.shape[1] != k_discovered:
                raise ValueError(
                    f"SAE payload '{field}' must be (N, K=={k_discovered}); "
                    f"got shape {arr.shape}"
                )
        def _opt_arr(atom: Mapping[str, Any], key: str) -> np.ndarray | None:
            value = atom.get(key)
            return None if value is None else np.asarray(value, dtype=float)

        def _periodic_shape_band(
            atom: Mapping[str, Any],
            plan: Mapping[str, Any],
        ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
            coords = _opt_arr(atom, "shape_band_coords")
            if coords is None:
                return None, None, None
            if coords.ndim != 2 or coords.shape[1] != 1:
                raise ValueError(
                    "periodic shape_band_coords must be a 2D array with one "
                    f"coordinate column; got shape {coords.shape}"
                )
            order = np.argsort(coords[:, 0], kind="mergesort")
            coords = np.ascontiguousarray(coords[order])
            decoder = np.asarray(atom["decoder_B"], dtype=float)
            # #1132 bug 1: a periodic atom is intrinsically at least one harmonic
            # (basis width 2H+1 with H>=1). A plan field that collapsed to 0 (a
            # degenerate constant-only width recovered from a born/fissioned atom
            # at K>=4) must be floored to the harmonic count implied by the
            # trained decoder width `M = 2H+1`, mirroring `_functional_basis_params`
            # — never raised. Otherwise the curved (n_harmonics=0) EV-vs-K curve at
            # K>=4 dies here instead of reconstructing the shape band.
            n_harmonics = int(plan["n_harmonics"])
            if n_harmonics <= 0:
                n_harmonics = (int(decoder.shape[0]) - 1) // 2
            n_harmonics = max(1, n_harmonics)
            phi, _jet, _penalty = rust_module().basis_with_jet(
                "periodic",
                coords,
                {"n_harmonics": n_harmonics},
            )
            phi = np.asarray(phi, dtype=float)
            if phi.shape[1] != decoder.shape[0]:
                # The posterior shape band is an OPTIONAL diagnostic, not part of
                # reconstruction (which uses `decoder_blocks` via predict_oos). A
                # periodic atom whose decoder collapsed to a non-`2H+1` width
                # (e.g. the hybrid-split linear collapse replacing the curved
                # decoder with a 1-row straight image, or a degenerate
                # born/fissioned atom) cannot present a periodic shape band — but
                # that must NOT abort the whole fit. Skip the band gracefully.
                return None, None, None
            mean = phi @ decoder
            sd = _opt_arr(atom, "shape_band_sd")
            cov = _opt_arr(atom, "decoder_covariance")
            if cov is not None:
                p = int(decoder.shape[1])
                m = int(decoder.shape[0])
                if cov.shape != (m * p, m * p):
                    raise ValueError(
                        "periodic decoder_covariance shape mismatch: "
                        f"expected {(m * p, m * p)}, got {cov.shape}"
                    )
                sd = np.zeros((coords.shape[0], p), dtype=float)
                for channel in range(p):
                    idx = np.arange(channel, m * p, p)
                    sub = cov[np.ix_(idx, idx)]
                    var = np.einsum("gm,mn,gn->g", phi, sub, phi, optimize=True)
                    sd[:, channel] = np.sqrt(np.maximum(var, 0.0))
            elif sd is not None:
                sd = np.asarray(sd, dtype=float)[order]
            return coords, mean, sd

        def _shape_band_arrays(
            atom: Mapping[str, Any],
            plan: Mapping[str, Any],
        ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
            if str(plan["kind"]) == "periodic":
                return _periodic_shape_band(atom, plan)
            return (
                _opt_arr(atom, "shape_band_coords"),
                _opt_arr(atom, "shape_band_mean"),
                _opt_arr(atom, "shape_band_sd"),
            )

        atoms: list[SaeManifoldAtomFit] = []
        for atom_idx, atom in enumerate(payload_atoms):
            shape_band_coords, shape_band_mean, shape_band_sd = _shape_band_arrays(
                atom,
                plans[atom_idx],
            )
            functional_evidence = _atom_functional_evidence(atom, plans[atom_idx])
            atoms.append(SaeManifoldAtomFit(
                basis=str(atom["basis_kind"]),
                decoder_coefficients=np.asarray(atom["decoder_B"], dtype=float),
                assignments=np.asarray(atom["assignments_z"], dtype=float),
                coords=np.asarray(atom["on_atom_coords_t"], dtype=float),
                evidence=float(payload["reml_score"]),
                active_dim=int(atom["active_dim"]),
                decoder_covariance=_opt_arr(atom, "decoder_covariance"),
                shape_band_coords=shape_band_coords,
                shape_band_mean=shape_band_mean,
                shape_band_sd=shape_band_sd,
                functional_evidence=functional_evidence,
            ))
        fitted = np.asarray(payload["fitted"], dtype=float)
        assigns = np.asarray(payload["assignments_z"], dtype=float)
        logits = np.asarray(payload["logits"], dtype=float)
        diagnostics = coerce_sae_trust_diagnostics(payload)
        coords = [atom.coords.copy() for atom in atoms]
        score = float(payload["reml_score"])
        chosen_k = int(payload["chosen_k"])
        low = SaeManifoldFitResult(atoms, chosen_k, {chosen_k: score}, {"winner": f"K={chosen_k}"}, fitted, assigns, coords, score)
        kinds = [str(p["kind"]) for p in plans]
        dims = [int(p["latent_dim"]) for p in plans]
        sizes = [int(p["basis_size"]) for p in plans]
        nharm = [int(p["n_harmonics"]) for p in plans]
        centers: list[np.ndarray | None] = [
            None if p["duchon_centers"] is None else np.asarray(p["duchon_centers"], dtype=float)
            for p in plans
        ]
        canonical = _canonical_assignment(assignment, "assignment")
        # #977 variable-K: the scalar ``atom_topology`` MUST be derived from the
        # POST-search ``kinds`` (the discovered dictionary), not the seed
        # ``topology`` argument. A fit that seeds an all-``periodic`` dictionary
        # but grows a heterogeneous one via evidence-gated births would otherwise
        # report the stale seed scalar (e.g. ``"circle"``) while
        # ``atom_topologies`` already reflects the heterogeneous truth — the
        # honest scalar collapses to ``"mixed"`` exactly when the per-atom
        # topologies disagree. ``basis_specs`` (== ``kinds``) remains the per-atom
        # source of truth either way; the seed ``topology`` arg is only a fallback
        # for an empty dictionary.
        atom_topologies = _topologies_for_bases(kinds)
        scalar_topology = _topology_for_bases(kinds) if kinds else str(topology)
        atom_inference_reports = _coerce_atom_inference(payload.get("atom_inference"))
        cotrain = _coerce_cotrain_report(payload.get("cotrain"))
        return cls(
            atoms=atoms, atom_topology=scalar_topology,
            atom_topologies=atom_topologies,
            assignment=canonical,
            assignment_label=str(assignment if assignment_label is None else assignment_label),
            primitive_names=["rust_module.sae_manifold_fit_minimal", *penalties],
            fitted=fitted, assignments=assigns, coords=coords,
            decoder_blocks=[a.decoder_coefficients.copy() for a in atoms],
            basis_specs=kinds, reml_score=score,
            reconstruction_r2=float(rust_module().sae_manifold_reconstruction_r2(x, fitted)),
            training_mean=x.mean(axis=0), training_data=x.copy(), low_level=low,
            low_level_logits=logits,
            diagnostics=diagnostics,
            _basis_kinds=kinds, _atom_dims=dims, _basis_sizes=sizes,
            _n_harmonics=nharm, _duchon_centers=centers,
            _oos_projection_top1=bool(payload["oos_projection_top1"]),
            alpha=float(alpha), learnable_alpha=bool(learnable_alpha),
            tau=float(tau), sparsity_strength=float(sparsity_strength),
            smoothness=float(smoothness), learning_rate=float(learning_rate),
            max_iter=int(max_iter), random_state=int(random_state),
            top_k=None if top_k is None else int(top_k),
            jumprelu_threshold=float(jumprelu_threshold),
            solver_plan=None if payload.get("solver_plan") is None else dict(payload["solver_plan"]),
            dispersion=float(payload["dispersion"]),
            # WP-D → fit wiring (#980): surface the metric provenance and the
            # per-row truncation diagnostic the Rust fit reports. Absent ⇒ the
            # Euclidean default (no output-Fisher shard was installed).
            metric_provenance=str(payload.get("metric_provenance", "Euclidean")),
            fisher_mass_residual=(
                None
                if payload.get("fisher_mass_residual") is None
                else np.asarray(payload["fisher_mass_residual"], dtype=float)
            ),
            # Additive post-fit diagnostics: the two-score per-atom lens,
            # residual-gauge certificate, and empirical incoherence inputs.
            # Absent ⇒ ``None`` (payloads predating each diagnostic); present
            # ⇒ the Rust report dict verbatim.
            atom_two_lens=(
                None
                if payload.get("atom_two_lens") is None
                else dict(payload["atom_two_lens"])
            ),
            residual_gauge=(
                None
                if payload.get("residual_gauge") is None
                else dict(payload["residual_gauge"])
            ),
            incoherence_report=(
                None
                if payload.get("incoherence_report") is None
                else dict(payload["incoherence_report"])
            ),
            curvature_report=(
                None
                if payload.get("curvature_report") is None
                else dict(payload["curvature_report"])
            ),
            atom_inference_reports=atom_inference_reports,
            certificates=(
                None
                if payload.get("certificates") is None
                else dict(payload["certificates"])
            ),
            structure_certificate_json=(
                None
                if payload.get("structure_certificate") is None
                else str(payload["structure_certificate"])
            ),
            cotrain=cotrain,
        )

    def structure_certificate(self, *, alpha: float | None = None) -> dict[str, Any]:
        """Anytime-valid structure-discovery certificate (#1058 / #984).

        Surfaces the e-BH certificate the structure search computed over the
        ledger of structural claims (atom-exists / binding-edge / geometry-kind)
        the fit proposed. Each claim carries an anytime-valid e-process, so the
        e-value and the gated/contested verdict are valid at this (or any)
        data-dependent stopping time — safe to peek.

        Parameters
        ----------
        alpha : float, optional
            FDR level to re-derive the gated set at. ``None`` (default) keeps the
            level the fit certified at (α = 0.05). A different α only re-runs the
            e-BH step over the stored per-claim e-values; it never refits.

        Returns
        -------
        dict
            ``{"alpha": float, "fdr_level": float, "n_confirmed": int,
            "claims": [{"claim_index": int, "claim": str, "kind": dict,
            "e_value": float, "log_e": float, "steps": int, "confirmed": bool,
            "evidence_remaining_nats": float}, ...]}``.
            ``evidence_remaining_nats`` is the anytime-valid budget ``max(0,
            ln(1/α) − log_e)`` — the additional log-evidence a probe must
            accumulate before the claim crosses the confirmation threshold (0
            once already confirmed).
        """
        import json
        import math

        if self.structure_certificate_json is None:
            raise ValueError(
                "this fitted model carries no structure certificate (payload "
                "predates #1058); refit to obtain one"
            )
        cert = json.loads(self.structure_certificate_json)
        entries = list(cert.get("entries", []))
        stored_alpha = float(cert.get("alpha", 0.05))
        level = stored_alpha if alpha is None else float(alpha)
        if not (0.0 < level < 1.0):
            raise ValueError(f"alpha must lie in (0, 1); got {level}")
        log_e = [float(e["log_e"]) for e in entries]
        confirmed_idx = set(_e_benjamini_hochberg(log_e, level))
        threshold = math.log(1.0 / level)
        claims: list[dict[str, Any]] = []
        for i, entry in enumerate(entries):
            le = float(entry["log_e"])
            claims.append(
                {
                    "claim_index": i,
                    "claim": _structure_claim_label(entry["kind"]),
                    "kind": entry["kind"],
                    "e_value": math.exp(le),
                    "log_e": le,
                    "steps": int(entry["steps"]),
                    "confirmed": i in confirmed_idx,
                    "evidence_remaining_nats": max(0.0, threshold - le),
                }
            )
        return {
            "alpha": level,
            "fdr_level": level,
            "n_confirmed": len(confirmed_idx),
            "claims": claims,
        }

    def atom_inference(self) -> list[dict[str, Any]]:
        """Per-atom smooth-functional inference reports (#1097 / #1103).

        One entry per fitted atom, in atom order, each
        ``{"atom_index": int, "atom_name": str, "functionals": {...} | None,
        "smooth_significance": {"log_e_nonconstant": float | None} | None}``.

        The #1103 ``smooth_significance.log_e_nonconstant`` is the any-n-valid
        split-likelihood-ratio e-value for "the atom's inner decoder smooth is
        non-constant" (null = constant), the same universal-inference instrument
        the atom-birth gate uses. With ``E_{H0}[E] <= 1`` it is finite-sample
        honest at the ``df ≈ n`` regime: a large positive ``log_e_nonconstant``
        is real evidence the atom carries smooth structure, ``<= 0`` does not
        favor non-constancy. An atom whose inner-decoder smooth was not harvested
        (no active rows / non-SPD inner Hessian / constant-only design) reports
        ``None`` fields rather than a fabricated value.

        Returns
        -------
        list of dict
            One report per atom. Empty list only for payloads predating the
            report (#1097 / #1103).
        """
        if self.atom_inference_reports is None:
            return []
        return [dict(report) for report in self.atom_inference_reports]

    def contested_claims(self, *, alpha: float | None = None) -> list[dict[str, Any]]:
        """The structure claims the held-out data did NOT confirm (#1058).

        Convenience filter over :meth:`structure_certificate`: returns only the
        contested claims (the inputs to a diagnostic probe-design loop), each
        with the anytime-valid ``evidence_remaining_nats`` budget that a probe
        would have to accumulate to confirm it. These are demoted, never
        rejected — they keep their evidence across future shards.
        """
        cert = self.structure_certificate(alpha=alpha)
        return [c for c in cert["claims"] if not c["confirmed"]]

    def contested_probe_report(self, *, alpha: float | None = None) -> list[dict[str, Any]]:
        """KL-optimal steering-probe plans for contested SAE atom claims (#1100).

        This closes the user-facing loop between the anytime-valid structure
        certificate and steering:

        1. take each contested ``AtomExists`` claim from
           :meth:`structure_certificate`;
        2. generate candidate on-manifold steering moves from the atom's fitted
           coordinate quantiles;
        3. score those candidates with
           :func:`gamfit.plan_probe_for_contested_claim`;
        4. return a report entry containing the selected steering payload and
           expected evidence budget.

        The null hypothesis for an ``AtomExists`` claim predicts no atom-carried
        response to the steering push. The alternative predicts the
        on-manifold response returned by :meth:`steer`; each candidate also
        carries ``off_manifold_norm`` so consumers can reject moves whose chord
        left the learned surface. An output-Fisher shard is required because the
        design score is measured in output-information nats, not Euclidean
        activation norm.
        """
        from .structure_discovery import plan_probe_for_contested_claim

        if self.fisher_factors is None:
            raise ValueError(
                "contested_probe_report requires a fitted output-Fisher shard "
                "(fit with fisher_factors=...); Euclidean SAE fits do not carry "
                "the information metric needed for KL-optimal probe design"
            )

        cert = self.structure_certificate(alpha=alpha)
        fisher = self._mean_output_fisher()
        report: list[dict[str, Any]] = []
        for claim in [c for c in cert["claims"] if not c["confirmed"]]:
            atom = _structure_claim_atom_exists(claim["kind"])
            entry: dict[str, Any] = {
                "claim_index": int(claim["claim_index"]),
                "claim": claim["claim"],
                "kind": claim["kind"],
                "log_e": float(claim["log_e"]),
                "evidence_remaining_nats": float(claim["evidence_remaining_nats"]),
                "probe_plan": None,
            }
            if atom is None:
                entry["unplannable_reason"] = (
                    "only AtomExists claims have an SAE steering-probe bridge"
                )
                report.append(entry)
                continue

            candidates = self._atom_exists_probe_candidates(atom)
            if not candidates:
                entry["atom"] = int(atom)
                entry["unplannable_reason"] = (
                    "atom has no non-degenerate coordinate-quantile steering moves"
                )
                report.append(entry)
                continue

            delta = np.ascontiguousarray(
                np.stack([c["delta"] for c in candidates], axis=0), dtype=np.float64
            )
            predicted_null = np.zeros_like(delta)
            predicted_alt = np.ascontiguousarray(
                np.stack([c["predicted_mean_alt"] for c in candidates], axis=0),
                dtype=np.float64,
            )
            plan = plan_probe_for_contested_claim(
                delta,
                predicted_null,
                predicted_alt,
                fisher,
                cert["alpha"],
                current_log_e=float(claim["log_e"]),
            )
            entry["atom"] = int(atom)
            entry["atom_name"] = str(self.atoms[atom].basis)
            entry["fisher_source"] = "mean_output_fisher"
            entry["candidate_count"] = len(candidates)
            if plan is None:
                entry["unplannable_reason"] = (
                    "candidate steering moves do not distinguish null and alternative"
                )
            else:
                selected = candidates[int(plan["probe"])]
                entry["probe_plan"] = {
                    **dict(plan),
                    "candidate": selected["candidate"],
                    "steer": _jsonable_value(selected["steer"]),
                    "predicted_mean_alt_source": (
                        "sae_steer_delta on-manifold response for AtomExists; "
                        "null response is zero"
                    ),
                }
            report.append(entry)
        return report

    def _mean_output_fisher(self) -> np.ndarray:
        u = np.asarray(self.fisher_factors, dtype=np.float64)
        if u.ndim != 3:
            raise ValueError(f"fisher_factors must be a rank-3 (N, p, r) array; got {u.shape}")
        if u.shape[0] != self.fitted.shape[0] or u.shape[1] != self.fitted.shape[1]:
            raise ValueError(
                "fisher_factors shape must match fitted rows/output dimension; "
                f"got {u.shape}, expected ({self.fitted.shape[0]}, {self.fitted.shape[1]}, r)"
            )
        return np.ascontiguousarray(
            np.einsum("npr,nqr->pq", u, u, optimize=True) / float(u.shape[0]),
            dtype=np.float64,
        )

    def _atom_exists_probe_candidates(self, atom: int) -> list[dict[str, Any]]:
        k = self._atom_index(atom)
        coords = np.asarray(self.coords[k], dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] == 0:
            return []
        low, mid, high = np.percentile(coords, [5.0, 50.0, 95.0], axis=0)
        moves: list[tuple[str, np.ndarray, np.ndarray]] = [
            ("median_to_high", mid, high),
            ("median_to_low", mid, low),
            ("low_to_high", low, high),
        ]
        for axis in range(coords.shape[1]):
            to_high = mid.copy()
            to_high[axis] = high[axis]
            moves.append((f"axis_{axis}_median_to_high", mid, to_high))
            to_low = mid.copy()
            to_low[axis] = low[axis]
            moves.append((f"axis_{axis}_median_to_low", mid, to_low))

        candidates: list[dict[str, Any]] = []
        seen: set[tuple[float, ...]] = set()
        for label, t_from, t_to in moves:
            if np.allclose(t_from, t_to):
                continue
            key = tuple(np.round(np.concatenate([t_from, t_to]), 12).tolist())
            if key in seen:
                continue
            seen.add(key)
            steer = self.steer(k, t_from, t_to)
            delta = np.ascontiguousarray(np.asarray(steer["delta"], dtype=np.float64).reshape(-1))
            candidates.append(
                {
                    "candidate": label,
                    "t_from": np.asarray(t_from, dtype=float).tolist(),
                    "t_to": np.asarray(t_to, dtype=float).tolist(),
                    "delta": delta,
                    "predicted_mean_alt": delta.copy(),
                    "steer": _jsonable_value(steer),
                }
            )
        return candidates

    def _periodic_top1_projection_payload(self, x: np.ndarray) -> dict[str, Any]:
        if (
            len(self.decoder_blocks) != 2
            or self.top_k != 1
            or self.assignment != "softmax"
            or any(kind != "periodic" for kind in self._basis_kinds)
            or any(int(dim) != 1 for dim in self._atom_dims)
        ):
            raise ValueError("periodic top-1 projection is only valid for two 1D periodic softmax atoms")
        grid = np.linspace(0.0, 1.0, 2048, endpoint=False, dtype=float)
        phi_grid = np.asarray(
            rust_module().basis_with_jet(
                "periodic",
                np.ascontiguousarray(grid.reshape(-1, 1)),
                {"n_harmonics": 1},
            )[0],
            dtype=float,
        )
        errors = np.zeros((x.shape[0], 2), dtype=float)
        best_coords = np.zeros((2, x.shape[0], 1), dtype=float)
        best_decoded = np.zeros((2, x.shape[0], x.shape[1]), dtype=float)
        for atom_idx, decoder in enumerate(self.decoder_blocks):
            decoded_grid = phi_grid @ np.asarray(decoder, dtype=float)
            diff = x[:, None, :] - decoded_grid[None, :, :]
            err_grid = np.sum(diff * diff, axis=2)
            best_idx = np.argmin(err_grid, axis=1)
            errors[:, atom_idx] = err_grid[np.arange(x.shape[0]), best_idx]
            best_coords[atom_idx, :, 0] = grid[best_idx]
            best_decoded[atom_idx] = decoded_grid[best_idx]
        winners = np.argmin(errors, axis=1)
        assignments = np.zeros((x.shape[0], 2), dtype=float)
        assignments[np.arange(x.shape[0]), winners] = 1.0
        fitted = best_decoded[winners, np.arange(x.shape[0])]
        logits = np.full((x.shape[0], 2), -4.0, dtype=float)
        logits[np.arange(x.shape[0]), winners] = 4.0
        atoms = []
        for atom_idx in range(2):
            atoms.append({
                "decoder_B": np.asarray(self.decoder_blocks[atom_idx], dtype=float).copy(),
                "basis_kind": "periodic",
                "assignments_z": assignments[:, atom_idx].copy(),
                "on_atom_coords_t": best_coords[atom_idx].copy(),
                "active_dim": 1,
            })
        return {
            "atoms": atoms,
            "assignments_z": assignments,
            "logits": logits,
            "fitted": fitted,
        }

    def _atom_index(self, atom: int) -> int:
        k = int(atom)
        if k < 0 or k >= len(self.atoms):
            raise IndexError(f"atom={atom} out of range for K={len(self.atoms)} atoms")
        return k

    def atom_trust(self, atom: int) -> float:
        """Scalar trust score for one atom, in ``[0, 1]``."""
        k = self._atom_index(atom)
        trust = np.asarray(self.diagnostics["atom_trust"], dtype=float)
        if trust.size == 0:
            raise ValueError(
                "this fit payload carries empty trust diagnostics; atom_trust is unavailable"
            )
        return float(trust[k])

    def atom_diagnostics(self, atom: int) -> dict[str, Any]:
        """All trust diagnostic components for one atom."""
        k = self._atom_index(atom)
        if not self.diagnostics["atoms"]:
            raise ValueError(
                "this fit payload carries empty trust diagnostics; atom_diagnostics is unavailable"
            )
        return dict(self.diagnostics["atoms"][k])

    def curvature(self) -> list[dict[str, Any]]:
        """Per-atom SAE curvature report (#1099, rescoped under #1115).

        Returns one record per atom: ``{"atom": int, "kappa_hat": float}``.
        ``kappa_hat`` is the fitted empirical second-fundamental-form sup-norm
        bound — a descriptive plug-in geometry summary. It is not an estimand
        with a confidence interval: a curvature bound has no profiled criterion,
        so no SE/CI/flatness fields are reported (the delta-method SE that #1099
        first shipped was conditioned on the generated latent coordinates as if
        known and under-covered, so #1115 removed it).
        """
        if self.curvature_report is None:
            raise ValueError(
                "this fitted model carries no SAE curvature report; refit to obtain one"
            )
        return [dict(atom) for atom in self.curvature_report.get("atoms", [])]

    def atom_curvature(self, atom: int) -> dict[str, Any]:
        """Curvature report record for one atom."""
        k = self._atom_index(atom)
        rows = self.curvature()
        if k >= len(rows):
            raise ValueError(
                f"curvature report has {len(rows)} atom rows but model has {len(self.atoms)} atoms"
            )
        return dict(rows[k])

    def shape_uncertainty(self, atom: int = 0, *, n_sd: float = 1.96) -> dict[str, np.ndarray]:
        """Posterior ambient shape uncertainty for one atom.

        Returns ``{"coords", "mean", "sd", "lower", "upper"}`` — the fitted
        ambient curve/surface and its closed-form posterior uncertainty on the
        atom's uncertainty grid. ``coords`` has shape ``(G, d_k)`` and uses the
        atom's raw latent-coordinate units. ``mean`` and ``sd`` have shape
        ``(G, p)`` and use the same ambient units as the training data ``X``;
        ``lower``/``upper`` are ``mean ± n_sd * sd``. ``n_sd=1.96`` gives the
        pointwise 95% posterior band. The per-atom decoder coefficient
        covariance that generated this band is available as
        ``self.atoms[atom].decoder_covariance``.
        """
        k = self._atom_index(atom)
        atom = self.atoms[k]
        if (
            atom.shape_band_coords is None
            or atom.shape_band_mean is None
            or atom.shape_band_sd is None
        ):
            raise ValueError(
                "shape_uncertainty is only available when the fit payload "
                "includes shape_band_coords, shape_band_mean, and shape_band_sd."
            )
        coords = np.asarray(atom.shape_band_coords, dtype=float)
        mean = np.asarray(atom.shape_band_mean, dtype=float)
        sd = np.asarray(atom.shape_band_sd, dtype=float)
        width = float(n_sd) * sd
        return {
            "coords": coords.copy(),
            "mean": mean.copy(),
            "sd": sd.copy(),
            "lower": mean - width,
            "upper": mean + width,
        }

    def coordinate_range(self, atom: int = 0) -> dict[str, Any]:
        """Observed training-coordinate range for one atom.

        Returns a dictionary with ``n`` and per-axis arrays ``min``, ``max``,
        ``p05``, ``p50``/``median``, and ``p95`` of shape ``(d_k,)`` computed
        from the atom's recovered training coordinates ``coords``. Coordinates
        are in the atom's raw latent-coordinate units. ``quantile_levels`` is
        ``[0.05, 0.50, 0.95]`` and ``quantiles`` has shape ``(3, d_k)``.
        """
        k = self._atom_index(atom)
        coords = np.asarray(self.atoms[k].coords, dtype=float)
        if coords.ndim != 2:
            raise ValueError(
                f"atom={atom} coords must be a 2D array; got shape {coords.shape}"
            )
        quantiles = np.percentile(coords, [5.0, 50.0, 95.0], axis=0)
        p05, p50, p95 = quantiles
        return {
            "n": int(coords.shape[0]),
            "min": np.min(coords, axis=0),
            "max": np.max(coords, axis=0),
            "p05": p05.copy(),
            "p50": p50.copy(),
            "median": p50.copy(),
            "p95": p95.copy(),
            "quantile_levels": np.asarray([0.05, 0.50, 0.95], dtype=float),
            "quantiles": quantiles.copy(),
        }

    def typical_shape(
        self,
        atom: int = 0,
        *,
        quantile_range: tuple[float, float] = (5.0, 95.0),
        n_sd: float = 1.0,
    ) -> dict[str, Any]:
        """Posterior shape band restricted to the atom's typical coordinate box.

        ``quantile_range`` selects the coordinate percentiles used to define the
        box, defaulting to the observed 5th-95th percentile range of
        ``self.atoms[atom].coords``. The returned dict contains the selected
        pointwise band (``coords``, ``mean``, ``sd``, ``lower``, ``upper``),
        the coordinate summary from :meth:`coordinate_range`, and aggregate
        ambient summaries over the selected grid: ``ambient_mean`` and
        ``ambient_sd`` are the per-channel mean and standard deviation of the
        fitted shape values across the typical coordinate range, while
        ``posterior_sd_mean`` is the average per-channel posterior sd.
        """
        q_low, q_high = (float(quantile_range[0]), float(quantile_range[1]))
        if not (0.0 <= q_low < q_high <= 100.0):
            raise ValueError(
                "quantile_range must be an increasing pair within [0, 100]; "
                f"got {quantile_range!r}"
            )
        k = self._atom_index(atom)
        fit_coords = np.asarray(self.atoms[k].coords, dtype=float)
        if fit_coords.ndim != 2:
            raise ValueError(
                f"atom={atom} coords must be a 2D array; got shape {fit_coords.shape}"
            )
        coord_low, coord_high = np.percentile(fit_coords, [q_low, q_high], axis=0)
        band = self.shape_uncertainty(atom=k, n_sd=n_sd)
        grid = np.asarray(band["coords"], dtype=float)
        if grid.ndim != 2 or grid.shape[1] != fit_coords.shape[1]:
            raise ValueError(
                "shape uncertainty coordinate grid is incompatible with recovered "
                f"coords: grid shape {grid.shape}, coords shape {fit_coords.shape}"
            )
        mask = np.all((grid >= coord_low) & (grid <= coord_high), axis=1)
        if not np.any(mask):
            raise ValueError(
                "shape uncertainty grid has no points inside the requested "
                f"{q_low:g}-{q_high:g} percentile coordinate range for atom={atom}"
            )
        mean = np.asarray(band["mean"], dtype=float)[mask]
        sd = np.asarray(band["sd"], dtype=float)[mask]
        width = float(n_sd) * sd
        return {
            "coordinate_range": self.coordinate_range(atom=k),
            "quantile_range": np.asarray([q_low, q_high], dtype=float),
            "coords": grid[mask].copy(),
            "mean": mean.copy(),
            "sd": sd.copy(),
            "lower": mean - width,
            "upper": mean + width,
            "ambient_mean": np.mean(mean, axis=0),
            "ambient_sd": np.std(mean, axis=0),
            "posterior_sd_mean": np.mean(sd, axis=0),
        }

    def _oos_payload(self, X: Any, *, t_init: Any = None, a_init: Any = None) -> dict[str, Any]:
        """Run the frozen-decoder OOS Newton solve on ``X`` and return the full
        payload dict (``assignments_z``, per-atom ``on_atom_coords_t``,
        ``logits``, ``fitted``).

        Optional ``t_init`` (K, N, D_max) and ``a_init`` (N, K) warm-start the
        refinement from an amortized encoder's per-token prediction (#357).
        """
        x = _as_2d_float(X, "X")
        kind = _canonical_assignment(self.assignment, "assignment")
        if t_init is None and a_init is None and self._oos_projection_top1:
            return self._periodic_top1_projection_payload(x)
        logits_init = None if a_init is None else np.ascontiguousarray(np.asarray(a_init, dtype=float))
        coords_init = None if t_init is None else np.ascontiguousarray(np.asarray(t_init, dtype=float))
        payload = rust_module().sae_manifold_predict_oos(
            np.ascontiguousarray(x), list(self._basis_kinds), list(self._atom_dims),
            [np.ascontiguousarray(b) for b in self.decoder_blocks],
            [None if c is None else np.ascontiguousarray(c) for c in self._duchon_centers],
            [(int(h) if k in {"periodic", "torus"} else None) for k, h in zip(self._basis_kinds, self._n_harmonics)],
            [int(s) for s in self._basis_sizes],
            alpha=float(self.alpha), tau=float(self.tau), assignment_kind=str(kind),
            sparsity_strength=float(self.sparsity_strength), smoothness=float(self.smoothness),
            max_iter=int(self.max_iter), learning_rate=float(self.learning_rate),
            initial_logits=logits_init, initial_coords=coords_init,
            top_k=self.top_k,
            jumprelu_threshold=float(self.jumprelu_threshold),
        )
        return dict(payload)

    def reconstruct(self, X: Any, *, t_init: Any = None, a_init: Any = None) -> np.ndarray:
        x = _as_2d_float(X, "X")
        if t_init is None and a_init is None and x.shape == self.training_data.shape and np.allclose(x, self.training_data):
            return self.fitted.copy()
        payload = self._oos_payload(x, t_init=t_init, a_init=a_init)
        return np.asarray(payload["fitted"], dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        return self.reconstruct(X)

    def distill_encoder(self, X: Any, **kwargs: Any) -> Any:
        """Train a post-hoc torch MLP encoder from exact OOS latent solves."""
        from .distill import distill_encoder

        return distill_encoder(self, X, **kwargs)

    def encode(
        self,
        X: Any,
        *,
        t_init: Any = None,
        a_init: Any = None,
        encoder: Any = None,
        return_stats: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Out-of-sample per-token assignments ``a*`` of shape ``(N, K)``.

        Runs the frozen-decoder OOS solve on ``X`` and returns the converged
        assignment matrix. On training ``X`` (matched bit-exactly) the cached
        fit assignments are returned without re-solving. ``t_init`` / ``a_init``
        warm-start the refinement (#357).

        Passing ``encoder=fit.distill_encoder(...)`` runs the distilled encoder
        first, accepts only rows matching the exact warm-started solve within the
        encoder's calibrated gate, and reports rowwise fallback accounting when
        ``return_stats=True``. The exact solve remains the teacher and fallback;
        the encoder never defines the feature map.
        """
        x = _as_2d_float(X, "X")
        if encoder is not None:
            if t_init is not None or a_init is not None:
                raise ValueError("encode(..., encoder=...) cannot also take t_init or a_init")
            from .distill import encode_with_fallback

            encoded, stats = encode_with_fallback(self, x, encoder)
            if return_stats:
                return encoded, stats.to_dict()
            return encoded
        if t_init is None and a_init is None and x.shape == self.training_data.shape and np.allclose(x, self.training_data):
            encoded = self.assignments.copy()
            if return_stats:
                return encoded, {
                    "rows": int(encoded.shape[0]),
                    "accepted_rows": 0,
                    "fallback_rows": 0,
                    "fallback_rate": 0.0,
                    "exact_probe_rows": 0,
                }
            return encoded
        payload = self._oos_payload(x, t_init=t_init, a_init=a_init)
        encoded = np.asarray(payload["assignments_z"], dtype=float)
        if return_stats:
            return encoded, {
                "rows": int(encoded.shape[0]),
                "accepted_rows": 0,
                "fallback_rows": int(encoded.shape[0]),
                "fallback_rate": 1.0,
                "exact_probe_rows": int(encoded.shape[0]),
            }
        return encoded

    def converged_latents(self, X: Any | None = None, *, t_init: Any = None, a_init: Any = None) -> dict[str, Any]:
        """Converged supervision targets for an amortized encoder (#357).

        Returns ``{"coords": list[(N, d_k) ndarray], "assignments": (N, K)
        ndarray, "logits": (N, K) ndarray, "fitted": (N, p) ndarray}`` — the
        per-atom on-manifold coordinates ``t*`` and the assignments / gate
        ``a*`` the joint solver converged to. With ``X is None`` (or training
        ``X``) the stored training-fit latents are returned; otherwise the OOS
        solve is run on ``X``, optionally warm-started from ``t_init`` /
        ``a_init``."""
        x = None if X is None else _as_2d_float(X, "X")
        use_training = (
            t_init is None and a_init is None
            and (x is None or (x.shape == self.training_data.shape and np.allclose(x, self.training_data)))
        )
        if use_training:
            return {
                "coords": [c.copy() for c in self.coords],
                "assignments": self.assignments.copy(),
                "logits": self.low_level_logits.copy(),
                "fitted": self.fitted.copy(),
            }
        payload = self._oos_payload(x, t_init=t_init, a_init=a_init)
        return {
            "coords": [np.asarray(atom["on_atom_coords_t"], dtype=float) for atom in payload["atoms"]],
            "assignments": np.asarray(payload["assignments_z"], dtype=float),
            "logits": np.asarray(payload["logits"], dtype=float),
            "fitted": np.asarray(payload["fitted"], dtype=float),
        }

    def project(self, X: Any, atom_k: int) -> np.ndarray:
        """Standalone per-atom projection ``project(x, atom_k) -> t`` (#357).

        Maps each ambient point in ``X`` to its on-manifold coordinate for atom
        ``atom_k`` under the trained decoder, via the same frozen-decoder OOS
        solve. Returns the ``(N, d_k)`` coordinate block for that atom — the
        minimal teacher signal for an encoder's coordinate head."""
        k = int(atom_k)
        if k < 0 or k >= len(self.atoms):
            raise IndexError(f"atom_k={atom_k} out of range for K={len(self.atoms)} atoms")
        x = _as_2d_float(X, "X")
        if x.shape == self.training_data.shape and np.allclose(x, self.training_data):
            return self.coords[k].copy()
        payload = self._oos_payload(x)
        return np.asarray(payload["atoms"][k]["on_atom_coords_t"], dtype=float)

    def steer(self, atom_k: int, t_from: Any, t_to: Any) -> dict[str, Any]:
        """Steering plan with output dosimetry for one atom (#980).

        Drives atom ``atom_k``'s latent coordinate from ``t_from`` to ``t_to``
        and reports the *actionable* steering payload of the SAE-manifold machine
        (``gam::inference::steering::steer_delta``): the activation-space move and
        its predicted output effect, measured through the fitted model's installed
        per-row output-Fisher metric.

        Parameters
        ----------
        atom_k
            Atom index in ``[0, K)``.
        t_from, t_to
            Source / target latent coordinates, each length ``d_k`` (the atom's
            ``atom_dim``), in the atom's raw latent-coordinate units (the same
            units as ``self.coords[atom_k]``).

        Returns
        -------
        dict
            The :class:`gam::inference::steering::SteerPlan` fields:

            * ``atom`` / ``atom_name`` — the steered atom and its name;
            * ``t_from`` / ``t_to`` — the latent endpoints (lists);
            * ``amplitude`` — the atom's mean active assignment mass the move was
              scaled by;
            * ``measured_row`` — the most-active row whose per-row metric the dose
              was read through;
            * ``delta`` — ``(p,)`` activation-space move ``a·(g_k(t_to) −
              g_k(t_from))`` to add to a hidden state;
            * ``predicted_nats`` — path-integrated output-Fisher KL dose in nats,
              or ``None`` under a Euclidean (no behavioral axis) metric;
            * ``validity_radius`` — latent step length the linearization is
              trusted to, or ``None`` under a Euclidean metric;
            * ``off_manifold_norm`` — ``δ``'s component off the local decoder
              tangents (``≈ 0`` for an on-manifold move);
            * ``metric_provenance`` — ``"OutputFisher"`` when a Fisher shard was
              installed at fit time (and retained), else ``"Euclidean"``.

        The dose (``predicted_nats`` / ``validity_radius``) is available only when
        the fit installed an output-Fisher metric (``fisher_factors`` was supplied
        to :func:`sae_manifold_fit` and retained on this model); otherwise the
        geometry (``delta`` / ``off_manifold_norm``) is still returned but the dose
        degrades to ``None`` — not zero.
        """
        k = self._atom_index(atom_k)
        t_from_arr = np.ascontiguousarray(np.asarray(t_from, dtype=np.float64).reshape(-1))
        t_to_arr = np.ascontiguousarray(np.asarray(t_to, dtype=np.float64).reshape(-1))
        kind = _canonical_assignment(self.assignment, "assignment")
        n_obs, p_out = (int(self.fitted.shape[0]), int(self.fitted.shape[1]))
        fisher = None if self.fisher_factors is None else np.ascontiguousarray(
            np.asarray(self.fisher_factors, dtype=np.float64)
        )
        plan = rust_module().sae_steer_delta(
            int(k),
            t_from_arr,
            t_to_arr,
            n_obs,
            p_out,
            list(self._basis_kinds),
            list(self._atom_dims),
            [np.ascontiguousarray(b) for b in self.decoder_blocks],
            [None if c is None else np.ascontiguousarray(c) for c in self._duchon_centers],
            [
                (int(h) if bk in {"periodic", "torus"} else None)
                for bk, h in zip(self._basis_kinds, self._n_harmonics)
            ],
            [int(s) for s in self._basis_sizes],
            [np.ascontiguousarray(c) for c in self.coords],
            np.ascontiguousarray(np.asarray(self.low_level_logits, dtype=np.float64)),
            str(kind),
            float(self.tau),
            alpha=float(self.alpha),
            jumprelu_threshold=float(self.jumprelu_threshold),
            fisher_factors=fisher,
            fisher_provenance=(
                None if self.fisher_factors is None else str(self.fisher_provenance)
            ),
        )
        return dict(plan)

    def per_atom_active_set(self, X: Any, threshold: float | None = None) -> np.ndarray:
        """Per-token active atom set ``(N, K)`` boolean mask for ``X``.

        On training ``X`` (matched bit-exactly) the cached fit assignments are
        thresholded without re-solving; otherwise the frozen-decoder OOS solve
        is run on ``X`` and its converged assignments are thresholded."""
        x = _as_2d_float(X, "X")
        cut = 0.5 if threshold is None else float(threshold)
        if x.shape == self.training_data.shape and np.allclose(x, self.training_data):
            return self.assignments >= cut
        payload = self._oos_payload(x)
        return np.asarray(payload["assignments_z"], dtype=float) >= cut

    def per_atom_latent_for(self, X: Any) -> list[np.ndarray]:
        """Per-atom on-manifold coordinates ``[(N, d_k), ...]`` for ``X``.

        On training ``X`` (matched bit-exactly) the cached fit coordinates are
        returned; otherwise the frozen-decoder OOS solve is run on ``X`` and its
        converged per-atom coordinates are returned."""
        x = _as_2d_float(X, "X")
        if x.shape == self.training_data.shape and np.allclose(x, self.training_data):
            return [c.copy() for c in self.coords]
        payload = self._oos_payload(x)
        return [np.asarray(atom["on_atom_coords_t"], dtype=float) for atom in payload["atoms"]]

    def featurize(self, X: Any) -> list[np.ndarray]:
        """Infer out-of-sample SAE coordinates for ``X``.

        This is the first-class research-loop spelling of the frozen-decoder
        OOS coordinate solve. It returns one ``(N, d_k)`` coordinate array per
        atom, in atom order, and reuses cached training coordinates when ``X``
        is the training activation matrix.
        """
        return self.per_atom_latent_for(X)

    def get_decoder(self) -> list[np.ndarray]:
        return [b.copy() for b in self.decoder_blocks]

    def get_anchors(self) -> list[np.ndarray]:
        return [c.copy() for c in self.coords]

    def summary(self) -> dict[str, Any]:
        # `self.assignment` is the canonical kind. Active-atom detection is
        # mode-specific:
        #   softmax   -> active if its share exceeds the uniform mass 1/K;
        #   ibp_map   -> active if its posterior gate exceeds the 0.5 threshold;
        #   jumprelu  -> active if the (hard) gate is nonzero (> 0).
        kind = _canonical_assignment(self.assignment, "assignment")
        if kind == "softmax":
            threshold = 1.0 / max(1, len(self.atoms))
        elif kind == "jumprelu":
            threshold = 0.0
        else:  # ibp_map
            threshold = 0.5
        avg_active, mean_mass = rust_module().sae_manifold_assignment_summary(self.assignments, threshold)
        return {
            "K": len(self.atoms),
            "d_atom": int(self.coords[0].shape[1]) if self.coords else 0,
            "atom_topology": self.atom_topology, "assignment": self.assignment,
            "alpha": float(self.alpha), "learnable_alpha": bool(self.learnable_alpha),
            "reml_score": float(self.reml_score), "reconstruction_r2": float(self.reconstruction_r2),
            "dispersion": float(self.dispersion),
            "atom_trust": np.asarray(self.diagnostics["atom_trust"], dtype=float).tolist(),
            "untyped_atoms": [
                i for i, diag in enumerate(self.diagnostics["atoms"]) if bool(diag["untyped"])
            ],
            "avg_active_atoms": float(avg_active), "mean_assignment_mass": float(mean_mass),
            "active_dims": [a.active_dim for a in self.atoms],
            "atom_functionals": [_json_ready(a.functional_evidence) for a in self.atoms],
            "cotrain": None if self.cotrain is None else dict(self.cotrain),
            "primitives": list(self.primitive_names),
        }

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable JSON-compatible serialization of this fit.

        The dict can be passed to :meth:`ManifoldSAE.from_dict` (or written to
        disk via :meth:`save` / :func:`gamfit.save`) to recover an object that
        reproduces :meth:`predict` outputs bit-exactly on training data.
        """
        def _optional_list(value: np.ndarray | None) -> Any:
            return None if value is None else value.tolist()

        def _jsonable(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {str(k): _jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_jsonable(v) for v in value]
            return value

        return {
            "schema": "gamfit.ManifoldSAE/v1",
            "atom_topology": self.atom_topology,
            "atom_topologies": list(self.atom_topologies),
            "assignment": self.assignment,
            "assignment_label": self.assignment_label,
            "alpha": float(self.alpha),
            "learnable_alpha": bool(self.learnable_alpha),
            "tau": float(self.tau),
            "sparsity_strength": float(self.sparsity_strength),
            "smoothness": float(self.smoothness),
            "learning_rate": float(self.learning_rate),
            "max_iter": int(self.max_iter),
            "random_state": int(self.random_state),
            "top_k": self.top_k,
            "jumprelu_threshold": float(self.jumprelu_threshold),
            "oos_projection_top1": bool(self._oos_projection_top1),
            "dispersion": float(self.dispersion),
            "solver_plan": None if self.solver_plan is None else _jsonable(self.solver_plan),
            "primitive_names": list(self.primitive_names),
            "basis_specs": list(self.basis_specs),
            "reml_score": float(self.reml_score),
            "reconstruction_r2": float(self.reconstruction_r2),
            "training_mean": self.training_mean.tolist(),
            "training_data": self.training_data.tolist(),
            "fitted": self.fitted.tolist(),
            "assignments": self.assignments.tolist(),
            "logits": self.low_level_logits.tolist(),
            "diagnostics": {
                "atom_trust": np.asarray(self.diagnostics["atom_trust"], dtype=float).tolist(),
                "atoms": [dict(atom) for atom in self.diagnostics["atoms"]],
            },
            "coords": [c.tolist() for c in self.coords],
            "decoder_blocks": [b.tolist() for b in self.decoder_blocks],
            "atoms": [
                {
                    "basis": a.basis,
                    "decoder_coefficients": a.decoder_coefficients.tolist(),
                    "assignments": a.assignments.tolist(),
                    "coords": a.coords.tolist(),
                    "evidence": float(a.evidence),
                    "active_dim": int(a.active_dim),
                    "decoder_covariance": _optional_list(a.decoder_covariance),
                    "shape_band_coords": _optional_list(a.shape_band_coords),
                    "shape_band_mean": _optional_list(a.shape_band_mean),
                    "shape_band_sd": _optional_list(a.shape_band_sd),
                    "functional_evidence": _json_ready(a.functional_evidence),
                }
                for a in self.atoms
            ],
            "basis_kinds": list(self._basis_kinds),
            "atom_dims": list(self._atom_dims),
            "basis_sizes": list(self._basis_sizes),
            "n_harmonics": list(self._n_harmonics),
            "duchon_centers": [None if c is None else c.tolist() for c in self._duchon_centers],
            "atom_two_lens": None if self.atom_two_lens is None else _jsonable(self.atom_two_lens),
            "residual_gauge": None if self.residual_gauge is None else _jsonable(self.residual_gauge),
            "incoherence_report": (
                None if self.incoherence_report is None else _jsonable(self.incoherence_report)
            ),
            "curvature_report": (
                None if self.curvature_report is None else _jsonable(self.curvature_report)
            ),
            "atom_inference": (
                None
                if self.atom_inference_reports is None
                else _jsonable(self.atom_inference_reports)
            ),
            "certificates": (
                None if self.certificates is None else _jsonable(self.certificates)
            ),
            "structure_certificate": self.structure_certificate_json,
            "cotrain": None if self.cotrain is None else _jsonable(self.cotrain),
        }

    def save(self, path: str | Path) -> None:
        """Write this fit to ``path`` as JSON. Round-trips via :func:`gamfit.load`."""
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ManifoldSAE":
        schema = str(payload["schema"])
        if schema != "gamfit.ManifoldSAE/v1":
            raise ValueError(f"ManifoldSAE.from_dict: unsupported schema {schema!r}")
        def _optional_array(atom_payload: Mapping[str, Any], key: str) -> np.ndarray | None:
            value = atom_payload.get(key)
            return None if value is None else np.asarray(value, dtype=float)

        atoms = [
            SaeManifoldAtomFit(
                basis=str(a["basis"]),
                decoder_coefficients=np.asarray(a["decoder_coefficients"], dtype=float),
                assignments=np.asarray(a["assignments"], dtype=float),
                coords=np.asarray(a["coords"], dtype=float),
                evidence=float(a["evidence"]),
                active_dim=int(a["active_dim"]),
                decoder_covariance=_optional_array(a, "decoder_covariance"),
                shape_band_coords=_optional_array(a, "shape_band_coords"),
                shape_band_mean=_optional_array(a, "shape_band_mean"),
                shape_band_sd=_optional_array(a, "shape_band_sd"),
                functional_evidence=(
                    None
                    if a.get("functional_evidence") is None
                    else dict(a["functional_evidence"])
                ),
            )
            for a in payload["atoms"]
        ]
        fitted = np.asarray(payload["fitted"], dtype=float)
        assigns = np.asarray(payload["assignments"], dtype=float)
        logits = np.asarray(payload["logits"], dtype=float)
        diagnostics = coerce_sae_trust_diagnostics(payload)
        coords = [np.asarray(c, dtype=float) for c in payload["coords"]]
        decoder_blocks = [np.asarray(b, dtype=float) for b in payload["decoder_blocks"]]
        score = float(payload["reml_score"])
        chosen_k = len(atoms)
        low = SaeManifoldFitResult(
            atoms, chosen_k, {chosen_k: score}, {"winner": f"K={chosen_k}"}, fitted, assigns, coords, score,
        )
        centers: list[np.ndarray | None] = [
            None if c is None else np.asarray(c, dtype=float) for c in payload["duchon_centers"]
        ]
        raw_assignment = str(payload["assignment"])
        canonical_assignment = _canonical_assignment(raw_assignment, "assignment")
        return cls(
            atoms=atoms,
            atom_topology=str(payload["atom_topology"]),
            atom_topologies=list(payload["atom_topologies"]),
            assignment=canonical_assignment,
            assignment_label=str(payload["assignment_label"]),
            primitive_names=list(payload["primitive_names"]),
            fitted=fitted,
            assignments=assigns,
            coords=coords,
            decoder_blocks=decoder_blocks,
            basis_specs=list(payload["basis_specs"]),
            reml_score=score,
            reconstruction_r2=float(payload["reconstruction_r2"]),
            training_mean=np.asarray(payload["training_mean"], dtype=float),
            training_data=np.asarray(payload["training_data"], dtype=float),
            low_level=low,
            low_level_logits=logits,
            diagnostics=diagnostics,
            _basis_kinds=list(payload["basis_kinds"]),
            _atom_dims=[int(d) for d in payload["atom_dims"]],
            _basis_sizes=[int(s) for s in payload["basis_sizes"]],
            _n_harmonics=[int(h) for h in payload["n_harmonics"]],
            _duchon_centers=centers,
            alpha=float(payload["alpha"]),
            learnable_alpha=bool(payload["learnable_alpha"]),
            tau=float(payload["tau"]),
            sparsity_strength=float(payload["sparsity_strength"]),
            smoothness=float(payload["smoothness"]),
            learning_rate=float(payload["learning_rate"]),
            max_iter=int(payload["max_iter"]),
            random_state=int(payload["random_state"]),
            top_k=None if payload["top_k"] is None else int(payload["top_k"]),
            jumprelu_threshold=float(payload["jumprelu_threshold"]),
            solver_plan=None if payload.get("solver_plan") is None else dict(payload["solver_plan"]),
            _oos_projection_top1=bool(payload["oos_projection_top1"]),
            dispersion=float(payload["dispersion"]),
            atom_two_lens=(
                None if payload.get("atom_two_lens") is None else dict(payload["atom_two_lens"])
            ),
            residual_gauge=(
                None if payload.get("residual_gauge") is None else dict(payload["residual_gauge"])
            ),
            incoherence_report=(
                None
                if payload.get("incoherence_report") is None
                else dict(payload["incoherence_report"])
            ),
            curvature_report=(
                None
                if payload.get("curvature_report") is None
                else dict(payload["curvature_report"])
            ),
            atom_inference_reports=_coerce_atom_inference(payload.get("atom_inference")),
            structure_certificate_json=(
                None
                if payload.get("structure_certificate") is None
                else str(payload["structure_certificate"])
            ),
            cotrain=_coerce_cotrain_report(payload.get("cotrain")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ManifoldSAE":
        return cls.from_dict(json.loads(Path(path).read_text()))


def gumbel_geometric_schedule(tau_start: float, tau_min: float, rate: float, iter_count: int = 0) -> GumbelTemperatureSchedule:
    return GumbelTemperatureSchedule(tau_start, tau_min, "geometric", rate=rate, iter_count=iter_count)


def gumbel_linear_schedule(tau_start: float, tau_min: float, steps: int, iter_count: int = 0) -> GumbelTemperatureSchedule:
    return GumbelTemperatureSchedule(tau_start, tau_min, "linear", steps=steps, iter_count=iter_count)


def gumbel_reciprocal_iter_schedule(tau_start: float, tau_min: float, iter_count: int = 0) -> GumbelTemperatureSchedule:
    return GumbelTemperatureSchedule(tau_start, tau_min, "reciprocal_iter", iter_count=iter_count)


_TOPOLOGY_UNSET: Any = object()


def sae_manifold_fit(X: Any = None, K: int | None = None, d_atom: int = 2, atom_topology: Any = _TOPOLOGY_UNSET,
                     assignment: str = "ibp_map", schedule: GumbelTemperatureSchedule | Mapping[str, Any] | None = None,
                     isometry_weight: float = 0.0, ard_per_atom: bool = True,
                     decoder_feature_sparsity_groups: list[list[int]] | None = None, n_iter: int = 50, *,
                     sparsity_weight: float = 1.0,
                     gate_sparsity: str = "scad", scad_mcp_gamma: float | None = None,
                     smoothness_weight: float = 1.0,
                     alpha: float | str = 1.0, learning_rate: float | None = None, random_state: int = 0,
                     block_orthogonality_weight: float = 0.0,
                     nuclear_norm_weight: float = 1.0, nuclear_norm_max_rank: int | None = None,
                     decoder_incoherence_weight: float = 1.0,
                     top_k: int | None = None, t_init: Any = None, a_init: Any = None,
                     tau: float | None = None, jumprelu_threshold: float = 0.0,
                     atom_basis: Any = None, fisher_factors: Any = None,
                     weights: Any = None) -> ManifoldSAE:
    """Fit an SAE-manifold model.

    Parameters
    ----------
    X
        Response data matrix reconstructed by the SAE. It may be a finite 1D
        or 2D numeric array; 1D input is reshaped to ``(N, 1)``.
    K
        Number of atoms. Must be positive, and the training set must satisfy
        ``N > K``.
    d_atom
        Intrinsic coordinate dimension per atom. Pass an int for a shared
        dimension or a length-``K`` iterable for heterogeneous atoms. ``None``
        and ``"auto"`` currently resolve to dimension 2 per atom.
    atom_topology
        Shared topology label used when ``atom_basis`` is not supplied. Common
        values are ``"circle"``, ``"periodic"``, ``"sphere"``, ``"torus"``,
        and ``"euclidean"``. If omitted, the default is ``"circle"``.
    assignment
        Assignment/gating family. ``"ibp_map"`` uses the IBP-MAP gate path,
        ``"softmax"`` uses soft mixture masses, and ``"jumprelu"`` uses the
        JumpReLU hard-gate family.
    schedule
        Optional :class:`GumbelTemperatureSchedule` or mapping forwarded to the
        IBP/Gumbel assignment path.
    isometry_weight
        Weight for ``IsometryPenalty`` on the latent coordinate block. Defaults
        to ``0.0`` (off). The Rust core compares ``g / gbar`` with the identity
        metric, where ``g = JᵀJ`` and ``gbar`` is the mean pullback trace per
        latent dimension, so the pin encourages a unit-average-speed chart
        without coupling to decoder scale (issue #795). Positive weights remain
        opt-in until the cold-start continuation accepts the planted-circle
        default-on chart-pin test. Issue #673 (resolved): the decoder smoothness
        penalty is reparameterized by the pulled-back metric ``g = JᵀJ`` in the
        Rust core, so the roughness — and the ``reml_score`` topology evidence —
        is gauge-invariant under reparameterization of the latent coordinate
        ``t`` even with the isometry penalty off. ``IsometryPenalty`` is purely
        a complementary regularizer when enabled (it drives ``g → I`` for an
        interpretable, near-arc-length chart); it is not a precondition for
        comparing ``reml_score`` across topologies.
    ard_per_atom
        If true, adds per-atom ARD row-block regularization on the latent
        coordinate block to select active intrinsic coordinates.
    decoder_feature_sparsity_groups
        Optional disjoint partition of output feature indices. Emits
        ``MechanismSparsityPenalty`` on each atom's decoder block, encouraging
        basis-function rows to load on a single feature group.
    n_iter
        Maximum joint-solver iterations.
    sparsity_weight
        Non-negative assignment sparsity strength.
    gate_sparsity
        Gate sparsity penalty family. The default ``"scad"`` enables adaptive
        non-convex sparsity for the recommended research objective. ``"l1"``
        keeps the historical assignment-prior sparsity path. ``"scad"`` and
        ``"mcp"`` emit the SAE row-block ``ScadMcpPenalty`` on the ``"t"``
        latent block with ``weight=sparsity_weight``.
    scad_mcp_gamma
        Optional SCAD/MCP concavity parameter. Defaults are SCAD ``3.7`` and
        MCP ``2.5``. SCAD requires ``gamma > 2``; MCP requires ``gamma > 1``.
    smoothness_weight
        Non-negative decoder smoothness weight.
        The penalty is ``0.5 * lambda * sum B.T @ S̃ @ B`` where ``S̃`` is the
        raw roughness Gram reparameterized by the decoder pullback metric
        (arc-length roughness), so it is gauge-invariant under reparameterizing
        the latent ``t`` (issue #673).
    alpha
        Assignment-prior concentration/scale. Pass a float for a fixed value or
        ``"auto"`` to mark alpha learnable in the Rust solve; returned metadata
        records ``alpha=1.0`` and ``learnable_alpha=True`` in that case.
    learning_rate
        Damped Newton/Gauss-Newton step size. If omitted, the Python facade uses
        ``1.0`` for IBP/softmax and ``0.05`` for JumpReLU.
    random_state
        Integer seed forwarded to the Rust initializer.
    block_orthogonality_weight
        Weight for ``BlockOrthogonalityPenalty`` on the latent coordinate block.
        Requires ``max(d_atom) >= 2`` and splits coordinate axes into singleton
        orthogonality groups.
    nuclear_norm_weight
        Weight for decoder embedding-rank selection (#672). It is on by
        default (``1.0``) for the recommended research objective. A positive value
        emits ``NuclearNormPenalty`` on each atom's ``(M_k, p)`` decoder matrix
        and shrinks its singular spectrum.
    nuclear_norm_max_rank
        Optional cap on the number of leading singular values penalized by the
        nuclear-norm decoder penalty. ``None`` leaves the rank cap disabled.
    decoder_incoherence_weight
        Cross-atom decoder column-space incoherence weight (#671). It is on by
        default (``1.0``) and applies when ``K >= 2``. The penalty uses the
        empirical co-activation ``mean_n gate_j * gate_k`` and penalizes
        ``||B_j @ B_k.T||_F^2`` for stored ``(M_k, p_out)`` decoder blocks on
        co-firing atom pairs.
    top_k
        Optional final assignment support projection. ``None`` and ``0``
        disable it; integers in ``[1, K]`` keep only the top-k assignment masses
        per observation and recompute ``fitted`` from that projected support.
    t_init, a_init
        Warm starts for amortized encoder distillation (#357). ``a_init`` has
        shape ``(N, K)`` and seeds assignment logits. ``t_init`` has shape
        ``(K, N, D_max)`` with ``D_max >= max(d_atom)`` and seeds per-atom
        coordinates. ``converged_latents()``, ``encode()``, and ``project()``
        expose the refined supervision targets.
    tau
        Starting assignment temperature. If ``None`` (the default), it is
        inferred from ``schedule`` or defaults to ``0.5``.
    jumprelu_threshold
        JumpReLU hard-gate threshold. Must be finite. Defaults to ``0.0``.
    atom_basis
        Per-atom basis kind(s). If supplied with ``atom_topology``, both must
        resolve to the same topology.
    fisher_factors
        Optional WP-D output-Fisher shard (#980). Accepts a
        :class:`gamfit.torch.harvest.HarvestShard`, the dict returned by
        :func:`gamfit.torch.harvest.load_harvest_shard`, or a raw ``(n, p, r)``
        factor array. Its *presence* installs ``RowMetric::OutputFisher`` for the
        isometry gauge / lens — there is no flag (magic by default). The metric
        does not whiten the reconstruction likelihood, so with the isometry gauge
        off (the default) the data-fit is identical to the Euclidean fit; the
        result's ``metric_provenance`` reports ``"OutputFisher"`` and the per-row
        ``fisher_mass_residual`` truncation diagnostic rides into the model.
        ``None`` (default) keeps the bit-identical Euclidean path.
    weights
        Optional per-row design-honesty reconstruction weights (#977): a
        length-``N`` array of strictly positive ``√w`` multipliers, one per
        observation. When supplied, each per-row reconstruction loss is scaled
        by its weight in the inner joint fit and the outer ρ (smoothness /
        sparsity / ARD) selection — the seam for honest fitting on a designed
        corpus subsample or an importance-weighted training set. The vector is
        self-normalized to mean 1 inside the core; a uniform or absent vector
        is the bit-identical unweighted path (magic by default — no flag).

    Returns
    -------
    ManifoldSAE
        Fitted result. Core attributes are ``atoms`` (list of
        :class:`SaeManifoldAtomFit`), ``fitted`` ``(N, p)``, ``assignments``
        ``(N, K)``, ``coords`` as per-atom ``(N, d_k)`` arrays,
        ``decoder_blocks`` as per-atom ``(M_k, p)`` decoder matrices,
        ``basis_specs``, ``atom_topology``/``atom_topologies``, ``assignment``
        and ``assignment_label``, ``reml_score``, ``reconstruction_r2``,
        ``dispersion``, ``training_mean``, ``training_data``,
        ``low_level_logits``, and fit-control metadata including ``alpha``,
        ``learnable_alpha``, ``tau``, ``sparsity_strength``, ``smoothness``,
        ``learning_rate``, ``max_iter``, ``random_state``, ``top_k``, and
        ``jumprelu_threshold``. Each atom exposes ``basis``,
        ``decoder_coefficients`` ``(M_k, p)``, per-atom ``assignments`` ``(N,)``,
        recovered ``coords`` ``(N, d_k)``, ``evidence``, ``active_dim``,
        ``decoder_covariance`` ``(M_k*p, M_k*p)``, ``shape_band_coords``
        ``(G, d_k)``, ``shape_band_mean`` ``(G, p)``, and ``shape_band_sd``
        ``(G, p)`` when the Rust payload includes posterior shape uncertainty.

        Useful public methods include ``predict``/``reconstruct``, ``encode``,
        ``converged_latents``, ``project``, ``per_atom_active_set``,
        ``per_atom_latent_for``, ``shape_uncertainty(atom=..., n_sd=...)``,
        ``coordinate_range(atom=...)``, and ``typical_shape(atom=...,
        quantile_range=(5.0, 95.0), n_sd=...)``.
    """
    if X is None:
        raise TypeError("sae_manifold_fit requires X input array")
    x = _as_2d_float(X, "X")
    k_atoms = int(K if K is not None else 0)
    max_iter_total = int(n_iter)
    smoothness = float(smoothness_weight)
    sparsity = float(sparsity_weight)
    gate_sparsity_kind = str(gate_sparsity).strip().lower()
    if gate_sparsity_kind not in {"l1", "scad", "mcp"}:
        raise ValueError(
            "gate_sparsity must be one of 'l1', 'scad', or 'mcp'; "
            f"got {gate_sparsity!r}"
        )
    if scad_mcp_gamma is None:
        scad_mcp_gamma_value = 3.7 if gate_sparsity_kind == "scad" else 2.5
    else:
        scad_mcp_gamma_value = float(scad_mcp_gamma)
    tau = float(tau if tau is not None else _schedule_tau_start(schedule, 0.5))
    jumprelu_threshold = float(jumprelu_threshold)
    if k_atoms <= 0:
        raise ValueError(f"K must be positive, got {k_atoms}")
    if max_iter_total < 1:
        raise ValueError(f"n_iter must be >= 1, got {max_iter_total}")
    # Eager n-sample validation (issue #183). One sample yields a
    # degenerate decoder LSQ system and a near-zero total sum of squares
    # — the resulting R² can be astronomically negative. Require at least
    # two observations, and at least as many observations as atoms so the
    # joint decoder block is identifiable.
    n_obs = int(x.shape[0])
    if n_obs < 2:
        raise ValueError(
            f"sae_manifold_fit requires n >= 2 observations; got n={n_obs}"
        )
    if n_obs <= k_atoms:
        raise ValueError(
            f"sae_manifold_fit requires n > K (more observations than atoms); "
            f"got n={n_obs}, K={k_atoms}"
        )
    # WP-D output-Fisher shard (#980). Magic-by-default: a non-None
    # `fisher_factors` (HarvestShard / load_harvest_shard dict / raw (n, p, r)
    # array) activates `RowMetric::OutputFisher` in the Rust core. Validate +
    # coerce here against the (n, p) response; ship the (n, p, r) U and the
    # optional (n,) mass_residual through the FFI. Absent ⇒ Euclidean path.
    fisher_shard = _normalize_fisher_factors(fisher_factors, n_obs, int(x.shape[1]))
    # Per-row design-honesty reconstruction weights (#977). When supplied, the
    # length-`n_obs` √w vector reweights every per-row reconstruction loss in
    # the inner joint fit and the outer ρ selection (installed Rust-side via
    # `SaeManifoldTerm::set_row_loss_weights`). Validate against the response
    # row count here; a uniform / absent vector self-normalizes to the exact
    # unweighted path. No flag — its presence is the switch (magic by default).
    row_loss_weights_arr: np.ndarray | None
    if weights is None:
        row_loss_weights_arr = None
    else:
        row_loss_weights_arr = np.ascontiguousarray(
            np.asarray(weights, dtype=float).reshape(-1)
        )
        if row_loss_weights_arr.shape[0] != n_obs:
            raise ValueError(
                "sae_manifold_fit: weights must have one entry per observation; "
                f"got {row_loss_weights_arr.shape[0]} for n={n_obs}"
            )
        if not np.all(np.isfinite(row_loss_weights_arr)) or np.any(
            row_loss_weights_arr <= 0.0
        ):
            raise ValueError(
                "sae_manifold_fit: weights must be finite and strictly positive"
            )
    dims = _dims(k_atoms, d_atom)
    # Eager d_atom validation (issue #184). A zero-dimensional atom carries
    # no manifold coordinate, contributes nothing to reconstruction, and
    # leaves `active_dims = [0, ...]` — that is a silent no-op that should
    # be a hard error, matching how `K <= 0` and `n_iter <= 0` are
    # rejected.
    if any(d < 1 for d in dims):
        raise ValueError(
            f"d_atom must be >= 1 for every atom; got {dims}"
        )
    # Eager sparsity_weight validation (issue #184). The signature
    # advertises `sparsity_weight: float = 1.0`; `0.0` is the canonical
    # "no sparsity" baseline and must be accepted. Reject only negative,
    # NaN, and infinite values here so the Rust kernel can apply its own
    # log-domain floor.
    if not np.isfinite(sparsity) or sparsity < 0.0:
        raise ValueError(
            f"sparsity_weight must be finite and non-negative; got {sparsity}"
        )
    if gate_sparsity_kind == "scad":
        if not (np.isfinite(scad_mcp_gamma_value) and scad_mcp_gamma_value > 2.0):
            raise ValueError(
                "scad_mcp_gamma must be finite and > 2 for gate_sparsity='scad'; "
                f"got {scad_mcp_gamma_value}"
            )
    elif gate_sparsity_kind == "mcp":
        if not (np.isfinite(scad_mcp_gamma_value) and scad_mcp_gamma_value > 1.0):
            raise ValueError(
                "scad_mcp_gamma must be finite and > 1 for gate_sparsity='mcp'; "
                f"got {scad_mcp_gamma_value}"
            )
    if not np.isfinite(jumprelu_threshold):
        raise ValueError(
            f"jumprelu_threshold must be finite; got {jumprelu_threshold}"
        )
    # Gauge-invariance of the topology evidence (issue #673, resolved). The
    # decoder smoothness penalty is reparameterized by the decoder pullback
    # metric g = J^T J in the Rust core (arc-length roughness; see
    # `SaeManifoldAtom::refresh_intrinsic_smooth_penalty`), so the roughness —
    # and therefore the REML Occam / joint-log-det terms that enter
    # `reml_score` — is invariant under reparameterizing the latent coordinate
    # t. Topology comparison (e.g. circle vs euclidean) is thus well posed
    # regardless of `isometry_weight`. `IsometryPenalty` is purely a
    # complementary regularizer that drives g -> I for an interpretable
    # near-arc-length chart; turning it off does not make `reml_score`
    # gauge-dependent, so there is nothing to warn about.
    # NOTE(#795): isometry still defaults OFF. The Rust penalty now normalizes
    # g = J^T J by the mean trace per latent dimension before comparing to I, so
    # the chart pin no longer scales as decoder^4; however, the planted-circle
    # default-on acceptance still fails after the curvature walk bifurcates and
    # fallback seed validation jumps to the target isometry weight.
    # Eager nuclear_norm_weight validation (issue #672). `0.0` is the canonical
    # "no rank penalty" baseline; reject negative / non-finite values so the
    # descriptor builder does not surface a cryptic Rust error.
    if not np.isfinite(nuclear_norm_weight) or nuclear_norm_weight < 0.0:
        raise ValueError(
            f"nuclear_norm_weight must be finite and non-negative; "
            f"got {nuclear_norm_weight}"
        )
    if nuclear_norm_max_rank is not None and int(nuclear_norm_max_rank) < 1:
        raise ValueError(
            f"nuclear_norm_max_rank must be >= 1 (or None to disable the cap); "
            f"got {nuclear_norm_max_rank}"
        )
    # Eager decoder_incoherence_weight validation (issue #671). On by default
    # (1.0); applies only for k_atoms >= 2 (it penalizes co-activating atom
    # pairs). Reject negative / non-finite values.
    if not np.isfinite(decoder_incoherence_weight) or decoder_incoherence_weight < 0.0:
        raise ValueError(
            f"decoder_incoherence_weight must be finite and non-negative; "
            f"got {decoder_incoherence_weight}"
        )
    topology_supplied = atom_topology is not _TOPOLOGY_UNSET
    atom_topology_str = str(atom_topology) if topology_supplied else "circle"
    bases = _bases(k_atoms, atom_basis, atom_topology_str)
    resolved_topology = _topology_for_bases(bases)
    if topology_supplied and atom_basis is not None and resolved_topology != atom_topology_str:
        raise ValueError(
            f"sae_manifold_fit: atom_basis={atom_basis!r} resolves to topology "
            f"{resolved_topology!r} but atom_topology={atom_topology_str!r} was also "
            f"supplied; they must describe the same topology."
        )
    kind = _canonical_public_assignment(assignment)
    alpha_value = 1.0 if alpha == "auto" else float(alpha)
    # Magic-by-default learning rate: the SAE Newton kernel is a damped
    # Gauss-Newton step against a quadratic local model with Armijo
    # backtracking. For softmax / IBP-MAP assignments the natural full step
    # is `lr=1.0` (matches the Rust reference test
    # `sae_manifold_fit_10_steps_one_harmonic_reaches_high_r2`, which reaches
    # R² ≥ 0.95 in 10 steps from a phase-shifted init). A small literal
    # `lr=0.05` starves the assignment posterior of gradient mass and lets
    # the IBP sigmoid drift into the saturated tail (the issue #165
    # collapse: assignment mass ~1e-146). JumpReLU keeps the historical
    # smaller step because its hard-gate STE is more sensitive to
    # overshooting the threshold. Callers can still override explicitly.
    if learning_rate is None:
        effective_lr = 0.05 if kind == "jumprelu" else 1.0
    else:
        effective_lr = float(learning_rate)
    penalties = [n for n, ok in (("IsometryPenalty", isometry_weight > 0.0), ("ARDPenalty", ard_per_atom),
        ("ScadMcpPenalty", gate_sparsity_kind in {"scad", "mcp"} and sparsity > 0.0),
        ("MechanismSparsityPenalty", decoder_feature_sparsity_groups is not None),
        ("BlockOrthogonalityPenalty", block_orthogonality_weight > 0.0),
        ("NuclearNormPenalty", nuclear_norm_weight > 0.0),
        ("DecoderIncoherencePenalty", decoder_incoherence_weight > 0.0 and k_atoms >= 2)) if ok]
    # Build the analytic-penalty registry payload that `sae_manifold_fit_minimal`
    # passes into `run_joint_fit_arrow_schur`. Row-block descriptors target the
    # SAE latent block "t" (shape (n_obs, d_max), where d_max = max(d_atom) —
    # matches the registry latent built in `sae_manifold_fit_inner`). Issue #240:
    # previously these knobs only populated `primitive_names` metadata.
    analytic_penalties_json = _build_analytic_penalties_payload(
        isometry_weight=isometry_weight,
        ard_per_atom=ard_per_atom,
        gate_sparsity=gate_sparsity_kind,
        sparsity_weight=sparsity,
        scad_mcp_gamma=scad_mcp_gamma_value,
        decoder_feature_sparsity_groups=decoder_feature_sparsity_groups,
        block_orthogonality_weight=block_orthogonality_weight,
        nuclear_norm_weight=nuclear_norm_weight,
        nuclear_norm_max_rank=nuclear_norm_max_rank,
        decoder_incoherence_weight=decoder_incoherence_weight,
        k_atoms=k_atoms,
        d_max=max(dims),
        p_out=int(x.shape[1]),
    )
    # `None` disables top-k gating; anything in `[1, k_atoms]` is forwarded to
    # the Rust driver, which
    # projects the final assignments onto a per-row top-k support and
    # recomputes `fitted` from the projected distribution. The Rust kernel
    # owns the hard top-k contract end to end — there is no Python-side mask.
    # Any value outside `[1, k_atoms]` is a caller error rather than a silent
    # clamp/no-op.
    if top_k is None:
        top_k_arg = None
    else:
        top_k_int = int(top_k)
        if top_k_int < 1 or top_k_int > k_atoms:
            raise ValueError(
                f"top_k must be in [1, K={k_atoms}] (or None to disable); "
                f"got {top_k_int}"
            )
        else:
            top_k_arg = top_k_int
    # Warm starts (issue #357): `a_init` (N, K) seeds the assignment logits and
    # `t_init` (K, N, D_max) seeds the per-atom on-manifold coordinates, so an
    # amortized encoder can predict `(a_init, t_init)` and have the joint solver
    # refine them for a bounded `n_iter` steps. Both are optional and validated
    # eagerly here against (N, K) / (K, N, D_max) where D_max = max(dims).
    d_max = max(dims)
    logits_init = None
    if a_init is not None:
        logits_init = np.ascontiguousarray(np.asarray(a_init, dtype=np.float64))
        if logits_init.shape != (n_obs, k_atoms):
            raise ValueError(
                f"sae_manifold_fit: a_init must have shape (N, K)=({n_obs}, {k_atoms}); "
                f"got {logits_init.shape}"
            )
    coords_init = None
    if t_init is not None:
        coords_init = np.ascontiguousarray(np.asarray(t_init, dtype=np.float64))
        if coords_init.ndim != 3 or coords_init.shape[0] != k_atoms or coords_init.shape[1] != n_obs:
            raise ValueError(
                f"sae_manifold_fit: t_init must have shape (K, N, D_max)=({k_atoms}, {n_obs}, >={d_max}); "
                f"got {coords_init.shape}"
            )
        if coords_init.shape[2] < d_max:
            raise ValueError(
                f"sae_manifold_fit: t_init D_max={coords_init.shape[2]} is too small for "
                f"max atom dim {d_max}"
            )
    # The closed-form disjoint-periodic fast path solves each atom in the
    # Euclidean response geometry and never reaches the FFI that installs
    # `RowMetric::OutputFisher`. When a WP-D shard is supplied the metric must
    # be honoured, so skip the shortcut and route through the joint FFI fit.
    if logits_init is None and coords_init is None and fisher_shard is None:
        separable_fit = _fit_disjoint_periodic_top1(
            x,
            bases=[str(b) for b in bases],
            dims=[int(d) for d in dims],
            assignment=str(kind),
            top_k=top_k_arg,
            penalties=penalties,
            alpha=float(alpha_value),
            learnable_alpha=bool(alpha == "auto"),
            tau=float(tau),
            sparsity_strength=float(sparsity),
            smoothness=float(smoothness),
            learning_rate=float(effective_lr),
            max_iter=int(max_iter_total),
            random_state=int(random_state),
            assignment_label=str(assignment),
            jumprelu_threshold=float(jumprelu_threshold),
        )
        if separable_fit is not None:
            return separable_fit
    payload = rust_module().sae_manifold_fit_minimal(
        np.ascontiguousarray(x),
        [str(b) for b in bases],
        [int(d) for d in dims],
        float(alpha_value),
        float(tau),
        bool(alpha == "auto"),
        str(kind),
        sparsity_strength=float(sparsity),
        smoothness=float(smoothness),
        max_iter=int(max_iter_total),
        learning_rate=float(effective_lr),
        gumbel_schedule=_schedule_payload(schedule),
        analytic_penalties=analytic_penalties_json,
        random_state=int(random_state),
        top_k=top_k_arg,
        initial_logits=logits_init,
        initial_coords=coords_init,
        jumprelu_threshold=float(jumprelu_threshold),
        fisher_factors=None if fisher_shard is None else fisher_shard[0],
        fisher_mass_residual=None if fisher_shard is None else fisher_shard[1],
        fisher_provenance=None if fisher_shard is None else fisher_shard[2],
        row_loss_weights=row_loss_weights_arr,
    )
    payload_dict = dict(payload)
    model = ManifoldSAE.from_payload(
        x, payload_dict, resolved_topology, kind, penalties,
        assignment_label=str(assignment),
        alpha=float(alpha_value), learnable_alpha=bool(alpha == "auto"),
        tau=float(tau), sparsity_strength=float(sparsity), smoothness=float(smoothness),
        learning_rate=float(effective_lr), max_iter=int(max_iter_total),
        random_state=int(random_state), top_k=top_k_arg,
        jumprelu_threshold=float(jumprelu_threshold),
    )
    # Retain the WP-D shard's (n, p, r) U so a follow-up `model.steer(...)` can
    # re-install `RowMetric::OutputFisher` and report the KL dose (#980). Under the
    # Euclidean (no-shard) path this stays None and steering is geometry-only.
    if fisher_shard is not None:
        model.fisher_factors = np.ascontiguousarray(fisher_shard[0])
        model.fisher_provenance = fisher_shard[2]
    return model


def _require_sae_row_block_penalty(kind: str, kwarg: str) -> None:
    """Refuse a SAE row-block penalty the running extension does not advertise.

    The compiled extension reports the row-block penalty kinds it supports via
    ``build_info()["sae_row_block_penalties"]`` (kept in lockstep with the Rust
    ``sae_penalty_is_row_block_supported`` matcher). A stale binary that predates
    a given penalty either omits the key entirely or lists a subset; forwarding
    the descriptor anyway would surface as a cryptic internal Schur-Cholesky
    error. Detect the mismatch here and raise a clear ``NotImplementedError``
    naming the user-facing kwarg (issue #338).
    """
    supported = rust_module().build_info().get("sae_row_block_penalties", [])
    if kind not in supported:
        raise NotImplementedError(
            f"sae_manifold_fit: {kwarg} requires SAE row-block penalty "
            f"'{kind}', which the installed gam-pyffi extension does not "
            "advertise (it predates row-block support for this penalty). "
            f"Upgrade gamfit to a build that supports '{kind}', or pass "
            f"{kwarg}=0.0 to disable it."
        )


def _build_analytic_penalties_payload(
    *,
    isometry_weight: float,
    ard_per_atom: bool,
    decoder_feature_sparsity_groups: list[list[int]] | None,
    block_orthogonality_weight: float,
    d_max: int,
    p_out: int,
    gate_sparsity: str = "l1",
    sparsity_weight: float = 0.0,
    scad_mcp_gamma: float = 3.7,
    nuclear_norm_weight: float = 0.0,
    nuclear_norm_max_rank: int | None = None,
    decoder_incoherence_weight: float = 1.0,
    k_atoms: int = 1,
) -> str | None:
    """Translate the SAE regularizer knobs into the analytic-penalty JSON
    payload consumed by ``sae_manifold_fit_minimal``.

    The SAE regularizer knobs route through ``src/terms/sae_manifold.rs``.
    ``ard_per_atom``, ``isometry_weight``, and ``block_orthogonality_weight``
    target the row-block driver ("t" latent block).
    ``gate_sparsity="scad"`` or ``"mcp"`` emits the row-block
    ``scad_mcp`` descriptor on the same "t" block, using ``sparsity_weight`` as
    its non-convex sparsity strength. The default ``"l1"`` emits no analytic
    descriptor and preserves the existing assignment-prior sparsity path.
    ``decoder_feature_sparsity_groups`` targets the decoder coefficient
    block ("beta" latent block) and group-lassoes ``p_out`` features in rows
    of the per-basis-function decoder matrix. For ``k_atoms >= 2`` the Rust
    ``add_sae_beta_penalty`` dispatches the group-lasso per atom, rebuilding
    the penalty target to each atom's ``(M_k, p_out)`` decoder block, so the
    concatenated ``flatten_beta`` layout with distinct ``M_k`` is handled
    natively (#240).

    ``nuclear_norm_weight`` also targets the decoder ("beta") block (#672): it
    emits a ``nuclear_norm`` descriptor that the Rust ``add_sae_beta_penalty``
    dispatches per atom, treating each atom's ``(M_k, p_out)`` decoder block as
    a matrix and shrinking its singular spectrum (embedding rank). ``n_eff`` is
    deliberately *not* emitted — Rust sets it per atom to ``M_k``.
    ``nuclear_norm_max_rank`` optionally caps the number of leading singular
    values penalized.
    """
    items: list[dict[str, Any]] = []
    if bool(ard_per_atom):
        _require_sae_row_block_penalty("ard", "ard_per_atom")
        items.append({"kind": "ard", "target": "t"})
    if gate_sparsity in {"scad", "mcp"} and float(sparsity_weight) > 0.0:
        _require_sae_row_block_penalty("scad_mcp", "gate_sparsity")
        items.append({
            "kind": "scad_mcp",
            "target": "t",
            "variant": str(gate_sparsity),
            "gamma": float(scad_mcp_gamma),
            "weight": float(sparsity_weight),
        })
    if isometry_weight is not None and float(isometry_weight) > 0.0:
        _require_sae_row_block_penalty("isometry", "isometry_weight")
        items.append({
            "kind": "isometry",
            "target": "t",
            "weight": float(isometry_weight),
        })
    if (
        block_orthogonality_weight is not None
        and float(block_orthogonality_weight) > 0.0
    ):
        _require_sae_row_block_penalty(
            "block_orthogonality", "block_orthogonality_weight"
        )
        # The latent block "t" is (n_obs, d_max). BlockOrth requires ≥2
        # groups that partition contiguous axes from 0 — split into
        # singletons so each axis is in its own group, which is the most
        # restrictive (and most informative) gauge available without
        # caller-supplied structure.
        if int(d_max) < 2:
            raise ValueError(
                "block_orthogonality_weight requires d_atom >= 2; "
                f"got d_max={d_max}"
            )
        groups = [[axis] for axis in range(int(d_max))]
        items.append({
            "kind": "block_orthogonality",
            "target": "t",
            "groups": groups,
            "weight": float(block_orthogonality_weight),
        })
    if decoder_feature_sparsity_groups is not None:
        # Validate group payload eagerly so the error surfaces in Python
        # with the user-facing kwarg name rather than as a Rust descriptor
        # error referring to "feature_groups".
        groups = [list(int(f) for f in g) for g in decoder_feature_sparsity_groups]
        if not groups or any(len(g) == 0 for g in groups):
            raise ValueError(
                "decoder_feature_sparsity_groups must be a non-empty list of "
                "non-empty index lists; got "
                f"{decoder_feature_sparsity_groups!r}"
            )
        flat = [int(f) for g in groups for f in g]
        if any(f < 0 or f >= int(p_out) for f in flat):
            raise ValueError(
                "decoder_feature_sparsity_groups indices must be in "
                f"[0, p_out={int(p_out)}); got {decoder_feature_sparsity_groups!r}"
            )
        if len(set(flat)) != len(flat):
            raise ValueError(
                "decoder_feature_sparsity_groups must form a disjoint "
                f"partition of feature indices; got {decoder_feature_sparsity_groups!r}"
            )
        if sorted(flat) != list(range(int(p_out))):
            raise ValueError(
                "decoder_feature_sparsity_groups must cover every feature "
                f"index in [0, p_out={int(p_out)}); got {decoder_feature_sparsity_groups!r}"
            )
        items.append({
            "kind": "mechanism_sparsity",
            "target": "beta",
            "feature_groups": groups,
        })
    if nuclear_norm_weight is not None and float(nuclear_norm_weight) > 0.0:
        # Targets the decoder ("beta") block. The Rust dispatch rebuilds the
        # penalty per atom (n_eff = M_k, latent_dim = p_out), so we deliberately
        # do NOT emit n_eff here — the registry-held base value is overridden.
        item: dict[str, Any] = {
            "kind": "nuclear_norm",
            "target": "beta",
            "weight": float(nuclear_norm_weight),
        }
        if nuclear_norm_max_rank is not None:
            item["max_rank"] = int(nuclear_norm_max_rank)
        items.append(item)
    # Cross-atom decoder column-space incoherence (issue #671), ON by default,
    # for k_atoms >= 2 (penalizes co-activating atom *pairs*). block_sizes/p_out
    # are placeholders: the Rust `add_sae_beta_penalty` injects the real per-atom
    # M_k, p_out, target, and the empirical co-activation (mean_n gate_j*gate_k)
    # from the live SAE at fit time. We only signal the descriptor + weight.
    if (
        decoder_incoherence_weight is not None
        and float(decoder_incoherence_weight) > 0.0
        and int(k_atoms) >= 2
    ):
        items.append({
            "kind": "decoder_incoherence",
            "target": "beta",
            "block_sizes": [1] * int(k_atoms),
            "p_out": int(p_out),
            "weight": float(decoder_incoherence_weight),
        })
    if not items:
        return None
    return json.dumps(items)


def _as_2d_float(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 1D or 2D numeric array")
    return np.ascontiguousarray(arr)


def _normalize_fisher_factors(
    fisher_factors: Any, n_obs: int, p_out: int
) -> tuple[np.ndarray, np.ndarray | None, str] | None:
    """Coerce a WP-D output-Fisher shard into the ``(U, mass_residual, provenance)``
    the Rust ``sae_manifold_fit_minimal`` FFI consumes (#980).

    ``fisher_factors`` may be: ``None`` (Euclidean, no shard); a
    :class:`gamfit.torch.harvest.HarvestShard` (``.U`` ``(n, p, r)`` /
    ``.mass_residual`` ``(n,)``); the dict returned by
    :func:`gamfit.torch.harvest.load_harvest_shard` (keys ``"U"`` /
    ``"mass_residual"``); or a raw ``(n, p, r)`` array (no diagnostic). The
    *presence* of a non-``None`` value activates ``RowMetric::OutputFisher`` —
    there is no flag (magic by default). The U layout ``U[n, i, k]`` is shipped
    verbatim as a contiguous ``(n, p, r)`` f64 array; the Rust boundary flattens
    it row-major to ``u[n, i * r + k]`` for ``RowMetric::output_fisher``.
    """
    if fisher_factors is None:
        return None
    # HarvestShard dataclass or load_harvest_shard() dict — both carry U +
    # mass_residual; a bare array carries only U. The provenance tag (#980)
    # rides along so the FFI installs the matching output-Fisher `RowMetric`;
    # a bare array or a pre-#980 shard defaults to the same-position metric.
    provenance = "output_fisher"
    if hasattr(fisher_factors, "U") and hasattr(fisher_factors, "mass_residual"):
        u_src: Any = fisher_factors.U
        mr_src: Any = fisher_factors.mass_residual
        provenance = str(getattr(fisher_factors, "provenance", "output_fisher"))
    elif isinstance(fisher_factors, Mapping):
        if "U" not in fisher_factors:
            raise ValueError(
                "fisher_factors mapping must contain a 'U' (n, p, r) array"
            )
        u_src = fisher_factors["U"]
        mr_src = fisher_factors.get("mass_residual")
        provenance = str(fisher_factors.get("provenance", "output_fisher"))
    else:
        u_src = fisher_factors
        mr_src = None
    if provenance not in ("output_fisher", "output_fisher_downstream"):
        raise ValueError(
            "fisher_factors provenance must be 'output_fisher' or "
            f"'output_fisher_downstream'; got {provenance!r}"
        )
    u = np.asarray(u_src, dtype=np.float64)
    if u.ndim != 3:
        raise ValueError(
            f"fisher_factors U must be (n, p, r); got shape {u.shape}"
        )
    if u.shape[0] != n_obs or u.shape[1] != p_out:
        raise ValueError(
            f"fisher_factors U must be (n, p, r) = ({n_obs}, {p_out}, r); "
            f"got leading dims {u.shape[:2]}"
        )
    rank = int(u.shape[2])
    if rank < 1:
        raise ValueError("fisher_factors U rank (last axis) must be >= 1")
    if rank > p_out:
        raise ValueError(
            f"fisher_factors U rank {rank} exceeds output dim p={p_out}"
        )
    if not np.all(np.isfinite(u)):
        raise ValueError("fisher_factors U must be finite")
    u = np.ascontiguousarray(u)
    if mr_src is None:
        return u, None, provenance
    mr = np.asarray(mr_src, dtype=np.float64)
    if mr.shape != (n_obs,):
        raise ValueError(
            f"fisher_factors mass_residual must be (n,) = ({n_obs},); "
            f"got shape {mr.shape}"
        )
    if not np.all(np.isfinite(mr)):
        raise ValueError("fisher_factors mass_residual must be finite")
    return u, np.ascontiguousarray(mr), provenance


def _dims(k_atoms: int, d_atom: Any) -> list[int]:
    if d_atom in (None, "auto"):
        return [2] * k_atoms
    if isinstance(d_atom, int):
        return [int(d_atom)] * k_atoms
    out = [int(d) for d in d_atom]
    if len(out) != k_atoms or min(out, default=0) < 0:
        raise ValueError("d_atom must provide one non-negative dimension per atom")
    return out


_TOPOLOGY_TO_BASIS = {
    "circle": "periodic", "periodic": "periodic",
    "sphere": "sphere", "torus": "torus", "euclidean": "euclidean",
}
_BASIS_TO_TOPOLOGY = {
    "periodic": "circle", "sphere": "sphere", "torus": "torus",
    "duchon": "euclidean", "euclidean": "euclidean", "euclidean_patch": "euclidean",
}


def _bases(k_atoms: int, atom_basis: Any, atom_topology: str) -> list[str]:
    if atom_basis is None:
        atom_basis = _TOPOLOGY_TO_BASIS.get(str(atom_topology), atom_topology)
    raw = [atom_basis] * k_atoms if isinstance(atom_basis, str) else list(atom_basis)
    if len(raw) != k_atoms:
        raise ValueError("atom_basis must provide one basis per atom")
    return [str(v) for v in raw]


def _topologies_for_bases(bases: list[str]) -> list[str]:
    """Per-atom topology labels for a resolved bases list (``basis_specs`` order)."""
    return [_BASIS_TO_TOPOLOGY.get(b, b) for b in bases]


def _topology_for_bases(bases: list[str]) -> str:
    """Collapse a resolved bases list to a single topology label for metadata.

    When all atoms share one topology that common label is returned; when the
    atoms span more than one topology the honest scalar is ``"mixed"`` and the
    per-atom truth is exposed via ``atom_topologies`` (``basis_specs`` remains
    the per-atom source of truth)."""
    per_atom = _topologies_for_bases(bases)
    first = per_atom[0]
    return first if all(t == first for t in per_atom) else "mixed"


def _schedule_payload(schedule: Any) -> dict[str, Any] | None:
    if schedule is None:
        return None
    if isinstance(schedule, GumbelTemperatureSchedule):
        return schedule.to_rust_descriptor()
    descriptor = dict(schedule)
    decay = str(descriptor.get("decay", "geometric")).lower().replace("-", "_")
    if "tau_start" not in descriptor:
        raise ValueError("GumbelTemperatureSchedule (dict form): missing 'tau_start'")
    if "tau_min" not in descriptor:
        raise ValueError("GumbelTemperatureSchedule (dict form): missing 'tau_min'")
    tau_start = float(descriptor["tau_start"])
    tau_min = float(descriptor["tau_min"])
    rate = descriptor.get("rate")
    steps = descriptor.get("steps")
    iter_count = int(descriptor.get("iter_count", 0))
    _validate_gumbel_schedule_fields(
        tau_start=tau_start, tau_min=tau_min, decay=decay,
        rate=None if rate is None else float(rate),
        steps=None if steps is None else int(steps),
        iter_count=iter_count,
    )
    descriptor["decay"] = decay
    descriptor["tau_min"] = tau_min
    descriptor["tau_start"] = tau_start
    descriptor["iter_count"] = iter_count
    return descriptor


def _schedule_tau_start(schedule: Any, default: float) -> float:
    payload = _schedule_payload(schedule)
    return default if payload is None else float(payload["tau_start"])


_LAST_RESEARCH_LOOP_MODEL: ManifoldSAE | None = None


def _default_research_k(n_obs: int) -> int:
    """Choose a conservative atom count for ``fit(activations)``."""
    return max(1, min(int(n_obs) - 1, 8, max(2, int(np.sqrt(max(1, int(n_obs)))))))


def _trust_scores(model: ManifoldSAE, activations: np.ndarray | None = None) -> dict[str, Any]:
    """Per-row and per-atom trust scores derived from atom diagnostics."""
    x = model.training_data if activations is None else _as_2d_float(activations, "activations")
    n_rows = int(x.shape[0])
    atom_trust = atom_trust_scores(model.diagnostics)
    n_atoms = int(atom_trust.shape[0])
    if activations is None or (
        x.shape == model.training_data.shape and np.allclose(x, model.training_data)
    ):
        assignments = np.asarray(model.assignments, dtype=float)
    else:
        assignments = np.asarray(model.encode(x), dtype=float)
    if assignments.shape != (n_rows, n_atoms):
        raise ValueError(
            "trust score assignments shape mismatch: "
            f"expected {(n_rows, n_atoms)}, got {assignments.shape}"
        )
    weights = np.clip(assignments, 0.0, np.inf)
    denom = weights.sum(axis=1)
    normalized = np.divide(
        weights,
        denom[:, None],
        out=np.zeros_like(weights, dtype=float),
        where=denom[:, None] > 0.0,
    )
    per_atom = normalized * atom_trust[None, :]
    return {
        "row": per_atom.sum(axis=1),
        "per_atom": per_atom,
        "atom": atom_trust,
        "diagnostics": model.diagnostics,
    }


def _research_fit_dict(model: ManifoldSAE, activations: np.ndarray) -> dict[str, Any]:
    trust = _trust_scores(model, activations)
    return {
        "model": model,
        "atoms": list(model.atoms),
        "coordinates": [c.copy() for c in model.coords],
        "assignments": model.assignments.copy(),
        "trust": trust,
        "trust_scores": trust["row"].copy(),
        "summary": model.summary(),
    }


def fit(activations: Any, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fit the recommended SAE-manifold research objective to activations.

    Parameters
    ----------
    activations
        Finite activation matrix ``(N, p)``. A vector is reshaped to ``(N, 1)``.
    config
        Optional keyword overrides forwarded to :func:`sae_manifold_fit`.

    Returns
    -------
    dict
        ``{"model", "atoms", "coordinates", "assignments", "trust",
        "trust_scores", "summary"}``. ``atoms`` contains typed
        :class:`SaeManifoldAtomFit` objects, ``coordinates`` is one
        ``(N, d_k)`` array per atom, and ``trust`` contains row, atom, and
        assignment-weighted per-row/per-atom scores derived from the fit
        diagnostics.
    """
    global _LAST_RESEARCH_LOOP_MODEL
    x = _as_2d_float(activations, "activations")
    cfg = {} if config is None else dict(config)
    if "K" not in cfg:
        cfg["K"] = _default_research_k(x.shape[0])
    model = sae_manifold_fit(x, **cfg)
    _LAST_RESEARCH_LOOP_MODEL = model
    return _research_fit_dict(model, x)


def featurize(new_activations: Any) -> list[np.ndarray]:
    """Infer coordinates for new activations with the most recent SAE fit.

    This promotes the existing frozen-decoder out-of-sample coordinate solve
    to a first-class research-loop function. It returns one ``(N, d_k)`` array
    per atom in the most recent :func:`fit` result. For explicit model-scoped
    use, call ``result["model"].featurize(new_activations)``.
    """
    if _LAST_RESEARCH_LOOP_MODEL is None:
        raise RuntimeError("gamfit.featurize requires a prior gamfit.fit(activations, config=...) call")
    return _LAST_RESEARCH_LOOP_MODEL.featurize(new_activations)


def align(fit_a: Any, fit_b: Any) -> Any:
    """Align two SAE research-loop fits by delegating to ``gamfit._alignment``."""
    from . import _alignment

    return _alignment.align(fit_a, fit_b)


def plot(atom: Any, **kwargs: Any) -> Any:
    """Plot SAE atoms by delegating to ``gamfit._sae_viz``."""
    from . import _sae_viz

    return _sae_viz.plot(atom, **kwargs)


__all__ = ["GumbelTemperatureSchedule", "ManifoldSAE", "SaeManifoldAtomFit", "SaeManifoldFitResult",
           "gumbel_geometric_schedule", "gumbel_linear_schedule", "gumbel_reciprocal_iter_schedule",
           "align", "featurize", "fit", "plot", "sae_manifold_fit"]
