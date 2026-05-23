"""SAE-manifold user-facing fit wrapper.

This module implements the Methodspace configuration from
``proposals/sae_manifold.md``:

    Z_i ~= sum_k a_ik Phi_k(t_ik) B_k

This wrapper owns candidate K construction, topology-basis materialization,
decoder REML fits, and evidence ranking via :func:`gamfit.compare_models`.
The Rust ``src/terms/sae_manifold.rs`` formal term owns the joint
Arrow-Schur row-block assembly for the same configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np

from ._api import gaussian_reml_fit
from ._binding import rust_module
from ._compare import compare_models
from .smooth import Duchon, PeriodicSplineCurve, Smooth, Sphere
from .topology import Circle, EuclideanPatch


@dataclass
class SaeManifoldAtomFit:
    """One fitted SAE-manifold atom."""

    basis: Smooth | str
    decoder_coefficients: np.ndarray
    assignments: np.ndarray
    coords: np.ndarray
    evidence: float
    active_dim: int


@dataclass
class SaeManifoldFitResult:
    """Result returned by :func:`sae_manifold_fit`."""

    atoms: list[SaeManifoldAtomFit]
    chosen_k: int
    evidence_by_candidate: dict[int, float]
    comparison: dict[str, Any]
    fitted: np.ndarray
    assignments: np.ndarray
    coords: list[np.ndarray]
    reml_score: float


@dataclass(frozen=True)
class GumbelTemperatureSchedule:
    """Deterministic temperature schedule for SAE assignment relaxations."""

    tau_start: float
    tau_min: float
    decay: Literal["geometric", "linear", "reciprocal_iter"]
    rate: float | None = None
    steps: int | None = None
    iter_count: int = 0


def gumbel_geometric_schedule(
    tau_start: float, tau_min: float, rate: float, iter_count: int = 0
) -> GumbelTemperatureSchedule:
    return GumbelTemperatureSchedule(
        tau_start=tau_start,
        tau_min=tau_min,
        decay="geometric",
        rate=rate,
        iter_count=iter_count,
    )


def gumbel_linear_schedule(
    tau_start: float, tau_min: float, steps: int, iter_count: int = 0
) -> GumbelTemperatureSchedule:
    return GumbelTemperatureSchedule(
        tau_start=tau_start,
        tau_min=tau_min,
        decay="linear",
        steps=steps,
        iter_count=iter_count,
    )


def gumbel_reciprocal_iter_schedule(
    tau_start: float, tau_min: float, iter_count: int = 0
) -> GumbelTemperatureSchedule:
    return GumbelTemperatureSchedule(
        tau_start=tau_start,
        tau_min=tau_min,
        decay="reciprocal_iter",
        iter_count=iter_count,
    )


def _normalize_penalty_descriptors(penalties: Sequence[Any] | None) -> str | None:
    """Normalize a Python penalty list to a JSON string accepted by the Rust FFI.

    The Rust pyfunctions take ``Option<String>`` (JSON-serialized) because
    PyO3 cannot bind ``Option<serde_json::Value>`` directly; we keep the
    Python-side data structure list-of-dicts and serialize at the FFI edge.
    """
    if penalties is None:
        return None
    if isinstance(penalties, (str, bytes)) or not isinstance(penalties, Sequence):
        raise TypeError("penalties must be a sequence of analytic penalty wrappers")
    out: list[dict[str, Any]] = []
    for index, penalty in enumerate(penalties):
        if hasattr(penalty, "to_rust_descriptor"):
            out.append(penalty.to_rust_descriptor())
        elif hasattr(penalty, "_to_rust_payload"):
            out.append(penalty._to_rust_payload())
        elif isinstance(penalty, Mapping):
            out.append(dict(penalty))
        else:
            raise TypeError(
                f"penalties[{index}] must expose to_rust_descriptor() or be a mapping"
            )
    import json as _json
    return _json.dumps(out)


def sae_manifold_fit(
    Z: Any,
    n_atoms: int | Literal["auto"] = 10,
    atom_basis: str | Smooth | Sequence[str | Smooth] = "duchon",
    atom_dim: int | Sequence[int] | Literal["auto"] | None = None,
    sparsity_strength: float | Literal["auto"] = "auto",
    smoothness: float | Literal["auto"] = "auto",
    *,
    assignment_prior: Literal["softmax", "ibp_map"] = "softmax",
    alpha: float | Literal["auto"] = "auto",
    learnable_alpha: bool = False,
    tau: float = 0.5,
    gumbel_schedule: GumbelTemperatureSchedule | Mapping[str, Any] | None = None,
    max_iter: int = 12,
    learning_rate: float = 0.05,
    random_state: int = 0,
    penalties: Sequence[Any] | None = None,
) -> SaeManifoldFitResult:
    """Fit a sparse mixture of smooth manifold atoms to activations ``Z``.

    Parameters follow the proposal. ``n_atoms="auto"`` compares
    ``K in {1, 2, 4, 8, 16, 32}`` by REML evidence. ``atom_dim="auto"``
    starts from two coordinates per atom and uses the ARD normalised prior
    in the evidence score; the reported ``active_dim`` counts axes whose
    empirical variance survives the ARD precision. ``assignment_prior`` is
    ``"softmax"`` by default; ``"ibp_map"`` uses deterministic concrete
    Beta-Bernoulli active indicators with ``alpha="auto"`` evidence-ranked
    over a small truncation-prior grid. ``learnable_alpha`` mirrors the Rust
    ``IBPAssignmentPenalty`` effective-alpha convention. ``gumbel_schedule``
    accepts :class:`GumbelTemperatureSchedule` or dictionaries such as
    ``{"decay": "geometric", "tau_start": 1.0, "tau_min": 1e-3,
    "rate": 0.95}`` and advances once per outer iteration.
    """

    z = _as_2d_float(Z, "Z")
    if z.shape[0] == 0 or z.shape[1] == 0:
        raise ValueError("sae_manifold_fit requires a non-empty (N, p) matrix")
    if assignment_prior not in {"softmax", "ibp_map"}:
        raise ValueError("assignment_prior must be 'softmax' or 'ibp_map'")
    if learnable_alpha and assignment_prior != "ibp_map":
        raise ValueError("learnable_alpha only applies to assignment_prior='ibp_map'")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")
    schedule = _normalize_gumbel_schedule(gumbel_schedule)
    tau_initial = tau if schedule is None else float(schedule["tau_start"])

    candidates = [int(n_atoms)] if n_atoms != "auto" else [1, 2, 4, 8, 16, 32]
    fits: list[SaeManifoldFitResult] = []
    names: list[str] = []
    for k in candidates:
        if k <= 0:
            raise ValueError(f"n_atoms candidates must be positive; got {k}")
        fit_k = _fit_fixed_k(
            z,
            k,
            atom_basis,
            atom_dim,
            sparsity_strength,
            smoothness,
            assignment_prior,
            alpha,
            learnable_alpha,
            tau_initial,
            schedule,
            max_iter=max_iter,
            learning_rate=learning_rate,
            random_state=random_state + 1009 * k,
            penalties=penalties,
        )
        fits.append(fit_k)
        names.append(f"K={k}")

    comparison = compare_models(
        [{"reml_score": f.reml_score, "edf": _edf_proxy(f)} for f in fits],
        names=names,
    )
    chosen_name = comparison["winner"]
    chosen_idx = names.index(chosen_name)
    chosen = fits[chosen_idx]
    chosen.comparison = comparison
    chosen.evidence_by_candidate = {k: f.reml_score for k, f in zip(candidates, fits)}
    return chosen


def _fit_fixed_k(
    z: np.ndarray,
    k_atoms: int,
    atom_basis: str | Smooth | Sequence[str | Smooth],
    atom_dim: int | Sequence[int] | Literal["auto"] | None,
    sparsity_strength: float | Literal["auto"],
    smoothness: float | Literal["auto"],
    assignment_prior: Literal["softmax", "ibp_map"],
    alpha: float | Literal["auto"],
    learnable_alpha: bool,
    tau: float,
    gumbel_schedule: dict[str, Any] | None,
    *,
    max_iter: int,
    learning_rate: float,
    random_state: int,
    penalties: Sequence[Any] | None,
) -> SaeManifoldFitResult:
    if sparsity_strength == "auto" and assignment_prior == "softmax":
        lambda_grid = [0.1, 1.0, 10.0]
        fits = [
            _fit_fixed_k(
                z,
                k_atoms,
                atom_basis,
                atom_dim,
                lam,
                smoothness,
                assignment_prior,
                alpha,
                learnable_alpha,
                tau,
                gumbel_schedule,
                max_iter=max_iter,
                learning_rate=learning_rate,
                random_state=random_state + idx * 7919,
                penalties=penalties,
            )
            for idx, lam in enumerate(lambda_grid)
        ]
        labels = [f"lambda_sparse={lam:g}" for lam in lambda_grid]
        comparison = compare_models(
            [{"reml_score": f.reml_score, "edf": _edf_proxy(f)} for f in fits],
            names=labels,
        )
        winner = labels.index(comparison["winner"])
        chosen = fits[winner]
        chosen.comparison = comparison
        return chosen

    if assignment_prior == "ibp_map" and alpha == "auto":
        alpha_grid = [0.25, 0.5, 1.0, 2.0]
        fits = [
            _fit_fixed_k(
                z,
                k_atoms,
                atom_basis,
                atom_dim,
                sparsity_strength,
                smoothness,
                assignment_prior,
                candidate_alpha,
                learnable_alpha,
                tau,
                gumbel_schedule,
                max_iter=max_iter,
                learning_rate=learning_rate,
                random_state=random_state + idx * 3571,
                penalties=penalties,
            )
            for idx, candidate_alpha in enumerate(alpha_grid)
        ]
        labels = [f"alpha={candidate_alpha:g}" for candidate_alpha in alpha_grid]
        comparison = compare_models(
            [{"reml_score": f.reml_score, "edf": _edf_proxy(f)} for f in fits],
            names=labels,
        )
        winner = labels.index(comparison["winner"])
        chosen = fits[winner]
        chosen.comparison = comparison
        return chosen

    rng = np.random.default_rng(random_state)
    n = z.shape[0]
    dims = _resolve_dims(k_atoms, atom_dim)
    basis_specs = _resolve_basis_specs(k_atoms, atom_basis)

    labels = _deterministic_partition(z, k_atoms)
    logits = np.full((n, k_atoms), -2.0, dtype=float)
    logits[np.arange(n), labels] = 2.0
    coords = [_initial_coords(z, labels, atom, dims[atom], rng) for atom in range(k_atoms)]
    log_ard = [np.zeros(d, dtype=float) for d in dims]

    lambda_sparse = 1.0 if sparsity_strength == "auto" else float(sparsity_strength)
    lambda_smooth = 1.0 if smoothness == "auto" else float(smoothness)
    if lambda_sparse <= 0.0 or lambda_smooth <= 0.0:
        raise ValueError("sparsity_strength and smoothness must be positive or 'auto'")
    alpha_value = 1.0 if alpha == "auto" else float(alpha)
    if not np.isfinite(alpha_value) or alpha_value <= 0.0:
        raise ValueError("alpha must be positive, finite, or 'auto'")

    if assignment_prior == "ibp_map":
        return _fit_fixed_k_ibp_rust(
            z,
            k_atoms,
            basis_specs,
            dims,
            lambda_sparse,
            lambda_smooth,
            alpha_value,
            learnable_alpha,
            tau,
            logits,
            coords,
            gumbel_schedule,
            max_iter=max_iter,
            learning_rate=learning_rate,
            analytic_penalties=penalties,
        )

    effective_alpha = (
        alpha_value * lambda_sparse
        if assignment_prior == "ibp_map" and learnable_alpha
        else alpha_value
    )

    last_payload: dict[str, Any] | None = None
    last_designs: list[np.ndarray] = []
    last_assignments = _assignment_from_logits(
        logits, assignment_prior, _gumbel_tau_at(gumbel_schedule, 0, tau)
    )

    for iter_idx in range(int(max_iter)):
        tau_iter = _gumbel_tau_at(gumbel_schedule, iter_idx, tau)
        assignments = _assignment_from_logits(logits, assignment_prior, tau_iter)
        designs: list[np.ndarray] = []
        jets: list[np.ndarray] = []
        penalties: list[np.ndarray] = []
        blocks: list[np.ndarray] = []
        for atom in range(k_atoms):
            phi, jet, penalty = _basis_and_jacobian(basis_specs[atom], coords[atom])
            designs.append(phi)
            jets.append(jet)
            penalties.append(penalty)
            blocks.append(assignments[:, [atom]] * phi)
        X = np.concatenate(blocks, axis=1)
        S = _block_diag(penalties)
        payload = gaussian_reml_fit(X, z, S * lambda_smooth)
        B_flat = np.asarray(payload["coefficients"], dtype=float)
        fitted = np.asarray(payload["fitted"], dtype=float)
        residual = z - fitted

        decoder_blocks = _split_decoder(B_flat, [d.shape[1] for d in designs])
        decoded = [designs[a] @ decoder_blocks[a] for a in range(k_atoms)]

        grad_a = np.zeros_like(assignments)
        for atom in range(k_atoms):
            grad_a[:, atom] = -np.einsum("np,np->n", residual, decoded[atom])
        grad_logits = _assignment_jvp(assignments, grad_a, assignment_prior, tau_iter)
        prior_grad = _assignment_prior_grad_logits(
            assignments, assignment_prior, lambda_sparse, effective_alpha, tau_iter
        )
        logits -= learning_rate * (grad_logits + prior_grad)

        for atom in range(k_atoms):
            d = dims[atom]
            if d == 0:
                continue
            dg_dt = np.einsum("nmd,mp->ndp", jets[atom], decoder_blocks[atom])
            grad_t = -assignments[:, atom, None] * np.einsum(
                "np,ndp->nd", residual, dg_dt
            )
            alpha = np.exp(log_ard[atom])
            grad_t += coords[atom] * alpha[None, :]
            coords[atom] -= learning_rate * grad_t
            if atom_dim == "auto" or atom_dim is None:
                var = np.maximum(np.var(coords[atom], axis=0), 1e-8)
                log_ard[atom] = np.clip(-np.log(var), -8.0, 12.0)

        last_payload = payload
        last_designs = designs
        last_assignments = assignments

    if last_payload is None:
        raise RuntimeError("sae_manifold_fit did not execute an optimization iteration")

    B_flat = np.asarray(last_payload["coefficients"], dtype=float)
    decoder_blocks = _split_decoder(B_flat, [d.shape[1] for d in last_designs])
    fitted = np.zeros_like(z)
    atoms: list[SaeManifoldAtomFit] = []
    for atom in range(k_atoms):
        decoded = last_designs[atom] @ decoder_blocks[atom]
        fitted += last_assignments[:, [atom]] * decoded
    score = float(last_payload["reml_score"])
    score -= _assignment_prior_value(
        last_assignments, assignment_prior, lambda_sparse, effective_alpha
    )
    score -= _ard_value(coords, log_ard)

    # Fixed-K runs get a single-candidate comparison object; auto-K replaces
    # this after all candidates have been ranked.
    comparison = compare_models([{"reml_score": score}], names=[f"K={k_atoms}"])

    for atom in range(k_atoms):
        active_dim = int(np.sum(np.var(coords[atom], axis=0) > 1e-5))
        atoms.append(
            SaeManifoldAtomFit(
                basis=basis_specs[atom],
                decoder_coefficients=decoder_blocks[atom],
                assignments=last_assignments[:, atom].copy(),
                coords=coords[atom].copy(),
                evidence=score,
                active_dim=active_dim,
            )
        )
    return SaeManifoldFitResult(
        atoms=atoms,
        chosen_k=k_atoms,
        evidence_by_candidate={k_atoms: score},
        comparison=comparison,
        fitted=fitted,
        assignments=last_assignments.copy(),
        coords=[c.copy() for c in coords],
        reml_score=score,
    )


def _fit_fixed_k_ibp_rust(
    z: np.ndarray,
    k_atoms: int,
    basis_specs: Sequence[str | Smooth],
    dims: Sequence[int],
    lambda_sparse: float,
    lambda_smooth: float,
    alpha_value: float,
    learnable_alpha: bool,
    tau: float,
    logits: np.ndarray,
    coords: Sequence[np.ndarray],
    gumbel_schedule: dict[str, Any] | None,
    *,
    max_iter: int,
    learning_rate: float,
    analytic_penalties: Sequence[Any] | None,
) -> SaeManifoldFitResult:
    designs: list[np.ndarray] = []
    jets: list[np.ndarray] = []
    penalties: list[np.ndarray] = []
    for atom in range(k_atoms):
        phi, jet, penalty = _basis_and_jacobian(basis_specs[atom], coords[atom])
        designs.append(np.ascontiguousarray(phi, dtype=float))
        jets.append(np.ascontiguousarray(jet, dtype=float))
        penalties.append(np.ascontiguousarray(penalty, dtype=float))

    n, p = z.shape
    m_max = max(d.shape[1] for d in designs)
    d_max = max((c.shape[1] for c in coords), default=0)
    basis_values = np.zeros((k_atoms, n, m_max), dtype=float)
    basis_jacobian = np.zeros((k_atoms, n, m_max, d_max), dtype=float)
    decoder_coefficients = np.zeros((k_atoms, m_max, p), dtype=float)
    smooth_penalties = np.zeros((k_atoms, m_max, m_max), dtype=float)
    coords_padded = np.zeros((k_atoms, n, d_max), dtype=float)
    basis_sizes: list[int] = []
    for atom in range(k_atoms):
        m = designs[atom].shape[1]
        d = coords[atom].shape[1]
        basis_sizes.append(m)
        basis_values[atom, :, :m] = designs[atom]
        basis_jacobian[atom, :, :m, :d] = jets[atom]
        smooth_penalties[atom, :m, :m] = penalties[atom]
        coords_padded[atom, :, :d] = coords[atom]

    normalized_penalties = _normalize_penalty_descriptors(analytic_penalties)
    payload = rust_module().sae_manifold_fit_ibp(
        np.ascontiguousarray(z, dtype=float),
        [_basis_kind_name(spec) for spec in basis_specs],
        [int(d) for d in dims],
        np.ascontiguousarray(basis_values),
        np.ascontiguousarray(basis_jacobian),
        basis_sizes,
        np.ascontiguousarray(decoder_coefficients),
        np.ascontiguousarray(smooth_penalties),
        np.ascontiguousarray(logits, dtype=float),
        np.ascontiguousarray(coords_padded),
        float(alpha_value),
        float(tau),
        bool(learnable_alpha),
        float(lambda_sparse),
        float(lambda_smooth),
        int(max_iter),
        float(learning_rate),
        gumbel_schedule=gumbel_schedule,
        analytic_penalties=normalized_penalties,
    )

    assignments = np.asarray(payload["assignments_z"], dtype=float)
    fitted = np.asarray(payload["fitted"], dtype=float)
    score = float(payload["reml_score"])
    atoms: list[SaeManifoldAtomFit] = []
    out_coords: list[np.ndarray] = []
    for atom, atom_payload in enumerate(payload["atoms"]):
        atom_coords = np.asarray(atom_payload["on_atom_coords_t"], dtype=float)
        decoder = np.asarray(atom_payload["decoder_B"], dtype=float)
        out_coords.append(atom_coords.copy())
        atoms.append(
            SaeManifoldAtomFit(
                basis=basis_specs[atom],
                decoder_coefficients=decoder,
                assignments=assignments[:, atom].copy(),
                coords=atom_coords.copy(),
                evidence=score,
                active_dim=int(np.sum(np.var(atom_coords, axis=0) > 1e-5)),
            )
        )
    comparison = compare_models([{"reml_score": score}], names=[f"K={k_atoms}"])
    return SaeManifoldFitResult(
        atoms=atoms,
        chosen_k=k_atoms,
        evidence_by_candidate={k_atoms: score},
        comparison=comparison,
        fitted=fitted,
        assignments=assignments.copy(),
        coords=out_coords,
        reml_score=score,
    )


def _resolve_dims(k_atoms: int, atom_dim: int | Sequence[int] | Literal["auto"] | None) -> list[int]:
    if atom_dim == "auto" or atom_dim is None:
        return [2 for _ in range(k_atoms)]
    if isinstance(atom_dim, int):
        if atom_dim < 0:
            raise ValueError("atom_dim must be non-negative")
        return [atom_dim for _ in range(k_atoms)]
    dims = [int(d) for d in atom_dim]
    if len(dims) != k_atoms:
        raise ValueError(f"atom_dim list length {len(dims)} must equal n_atoms={k_atoms}")
    if any(d < 0 for d in dims):
        raise ValueError("atom_dim entries must be non-negative")
    return dims


def _resolve_basis_specs(k_atoms: int, atom_basis: str | Smooth | Sequence[str | Smooth]) -> list[str | Smooth]:
    if isinstance(atom_basis, (str, Smooth)):
        raw = [atom_basis for _ in range(k_atoms)]
    else:
        raw = list(atom_basis)
        if len(raw) != k_atoms:
            raise ValueError(f"atom_basis list length {len(raw)} must equal n_atoms={k_atoms}")
    return [_basis_from_name(spec) if isinstance(spec, str) else spec for spec in raw]


def _basis_from_name(name: str) -> Smooth:
    key = name.lower().replace("-", "_")
    if key in {"duchon", "euclidean", "euclidean_patch"}:
        return EuclideanPatch(d=2, centers=None)
    if key in {"periodic", "periodic_spline", "circle"}:
        return Circle(n_knots=24)
    if key == "sphere":
        return Sphere(n_centers=16)
    raise ValueError(f"unsupported atom_basis {name!r}")


def _basis_kind_name(spec: str | Smooth) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, PeriodicSplineCurve):
        return "periodic"
    if isinstance(spec, Sphere):
        return "sphere"
    if isinstance(spec, Duchon):
        return "duchon"
    if isinstance(spec, EuclideanPatch):
        return "euclidean_patch"
    return spec.__class__.__name__


def _basis_and_jacobian(spec: str | Smooth, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(spec, PeriodicSplineCurve):
        return _periodic_fourier_basis(t[:, 0], max(3, int(spec.n_knots // 2)))
    if isinstance(spec, Sphere):
        return _sphere_chart_basis(t)
    if isinstance(spec, Duchon):
        return _duchon_basis_local(t, spec)
    return _duchon_basis_local(t, Duchon(centers=None, m=2))


def _duchon_basis_local(t: np.ndarray, spec: Duchon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = t.shape
    n_centers = 12 if spec.centers is None else spec.centers
    if isinstance(n_centers, int):
        centers = _latent_grid_centers(t, int(n_centers))
    else:
        centers = _as_2d_float(n_centers, "Duchon.centers")
    m = int(spec.m)
    power = max(1, 2 * m - d)
    diff = t[:, None, :] - centers[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    if power % 2 == 0:
        phi = (r ** power) * np.log(np.maximum(r, 1e-12))
        dphi_dr = power * (r ** (power - 1)) * np.log(np.maximum(r, 1e-12)) + r ** (power - 1)
    else:
        phi = r ** power
        dphi_dr = power * r ** (power - 1)
    jet = np.zeros((n, centers.shape[0], d), dtype=float)
    scale = np.divide(dphi_dr, np.maximum(r, 1e-12), out=np.zeros_like(r), where=r > 0)
    jet[:, :, :] = scale[:, :, None] * diff
    poly = np.concatenate([np.ones((n, 1)), t], axis=1)
    poly_jet = np.zeros((n, d + 1, d), dtype=float)
    for axis in range(d):
        poly_jet[:, 1 + axis, axis] = 1.0
    Phi = np.concatenate([phi, poly], axis=1)
    Jet = np.concatenate([jet, poly_jet], axis=1)
    penalty = np.eye(Phi.shape[1]) * 1e-6
    penalty[: centers.shape[0], : centers.shape[0]] = _rbf_gram(centers, power)
    return Phi, Jet, penalty


def _periodic_fourier_basis(t: np.ndarray, n_harmonics: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.mod(t, 1.0)
    cols = [np.ones_like(x)]
    dcols = [np.zeros_like(x)]
    penalty_diag = [1e-8]
    for h in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * h * x
        cols.extend([np.sin(angle), np.cos(angle)])
        dcols.extend([2.0 * np.pi * h * np.cos(angle), -2.0 * np.pi * h * np.sin(angle)])
        penalty_diag.extend([float(h**4), float(h**4)])
    Phi = np.stack(cols, axis=1)
    Jet = np.stack(dcols, axis=1)[:, :, None]
    return Phi, Jet, np.diag(penalty_diag)


def _sphere_chart_basis(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.clip(t[:, 0], -np.pi / 2.0, np.pi / 2.0)
    lon = t[:, 1] if t.shape[1] > 1 else np.zeros_like(lat)
    clat = np.cos(lat)
    slat = np.sin(lat)
    clon = np.cos(lon)
    slon = np.sin(lon)
    x = clat * clon
    y = clat * slon
    z = slat
    Phi = np.stack([np.ones_like(lat), x, y, z, x * y, y * z, x * z], axis=1)
    Jet = np.zeros((t.shape[0], Phi.shape[1], t.shape[1]), dtype=float)
    dx_dlat = -slat * clon
    dx_dlon = -clat * slon
    dy_dlat = -slat * slon
    dy_dlon = clat * clon
    dz_dlat = clat
    Jet[:, 1, 0] = dx_dlat
    Jet[:, 2, 0] = dy_dlat
    Jet[:, 3, 0] = dz_dlat
    if t.shape[1] > 1:
        Jet[:, 1, 1] = dx_dlon
        Jet[:, 2, 1] = dy_dlon
        Jet[:, 4, 0] = dx_dlat * y + x * dy_dlat
        Jet[:, 4, 1] = dx_dlon * y + x * dy_dlon
        Jet[:, 5, 0] = dy_dlat * z + y * dz_dlat
        Jet[:, 5, 1] = dy_dlon * z
        Jet[:, 6, 0] = dx_dlat * z + x * dz_dlat
        Jet[:, 6, 1] = dx_dlon * z
    return Phi, Jet, np.diag([1e-8, 1, 1, 1, 4, 4, 4])


def _initial_coords(z: np.ndarray, labels: np.ndarray, atom: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    if dim == 0:
        return np.zeros((z.shape[0], 0), dtype=float)
    centered = z - z.mean(axis=0, keepdims=True)
    mask = labels == atom
    local = z[mask] if np.any(mask) else z
    local = local - local.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(local, full_matrices=False)
    basis = np.zeros((dim, z.shape[1]), dtype=float)
    take = min(dim, vt.shape[0])
    basis[:take] = vt[:take]
    if take < dim:
        basis[take:] = rng.normal(scale=0.01, size=(dim - take, z.shape[1]))
    coords = centered @ basis.T
    scale = np.std(coords, axis=0, keepdims=True)
    return coords / np.maximum(scale, 1e-6)


def _deterministic_partition(z: np.ndarray, k_atoms: int) -> np.ndarray:
    if k_atoms == 1:
        return np.zeros(z.shape[0], dtype=int)
    score = z @ np.linspace(1.0, 2.0, z.shape[1])
    ranks = np.argsort(np.argsort(score))
    return np.minimum(k_atoms - 1, (ranks * k_atoms) // max(1, z.shape[0]))


def _latent_grid_centers(t: np.ndarray, n_centers: int) -> np.ndarray:
    if t.shape[1] == 1:
        qs = np.linspace(0.02, 0.98, n_centers)
        return np.quantile(t[:, 0], qs).reshape(-1, 1)
    rng = np.random.default_rng(17 + n_centers + t.shape[1])
    idx = rng.choice(t.shape[0], size=min(n_centers, t.shape[0]), replace=False)
    centers = t[idx].copy()
    if centers.shape[0] < n_centers:
        extra = rng.normal(scale=0.1, size=(n_centers - centers.shape[0], t.shape[1]))
        centers = np.concatenate([centers, t.mean(axis=0, keepdims=True) + extra], axis=0)
    return centers


def _rbf_gram(centers: np.ndarray, power: int) -> np.ndarray:
    diff = centers[:, None, :] - centers[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    if power % 2 == 0:
        gram = (r ** power) * np.log(np.maximum(r, 1e-12))
    else:
        gram = r ** power
    return gram @ gram.T + np.eye(gram.shape[0]) * 1e-6


def _normalize_gumbel_schedule(
    schedule: GumbelTemperatureSchedule | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if schedule is None:
        return None
    if isinstance(schedule, GumbelTemperatureSchedule):
        schedule = asdict(schedule)
    if not isinstance(schedule, Mapping):
        raise ValueError("gumbel_schedule must be GumbelTemperatureSchedule, a mapping, or None")
    decay = str(_require_schedule_key(schedule, "decay")).lower().replace("-", "_")
    tau_start = float(_require_schedule_key(schedule, "tau_start"))
    tau_min = float(_require_schedule_key(schedule, "tau_min"))
    if not np.isfinite(tau_start) or tau_start <= 0.0:
        raise ValueError("gumbel_schedule['tau_start'] must be finite and positive")
    if not np.isfinite(tau_min) or tau_min <= 0.0:
        raise ValueError("gumbel_schedule['tau_min'] must be finite and positive")
    if tau_min > tau_start:
        raise ValueError("gumbel_schedule['tau_min'] cannot exceed tau_start")
    iter_count = int(schedule.get("iter_count", 0))
    if iter_count < 0:
        raise ValueError("gumbel_schedule['iter_count'] must be non-negative")
    out: dict[str, Any] = {
        "decay": decay,
        "tau_start": tau_start,
        "tau_min": tau_min,
        "iter_count": iter_count,
    }
    if decay == "geometric":
        rate = float(_require_schedule_key(schedule, "rate"))
        if not np.isfinite(rate) or rate <= 0.0 or rate >= 1.0:
            raise ValueError("gumbel_schedule['rate'] must be in (0, 1)")
        out["rate"] = rate
    elif decay == "linear":
        steps = int(_require_schedule_key(schedule, "steps"))
        if steps <= 0:
            raise ValueError("gumbel_schedule['steps'] must be positive")
        out["steps"] = steps
    elif decay == "reciprocal_iter":
        out["decay"] = "reciprocal_iter"
    else:
        raise ValueError(
            "gumbel_schedule['decay'] must be 'geometric', 'linear', or 'reciprocal_iter'"
        )
    return out


def _require_schedule_key(schedule: Mapping[str, Any], key: str) -> Any:
    if key not in schedule:
        raise ValueError(f"gumbel_schedule is missing required key {key!r}")
    return schedule[key]


def _gumbel_tau_at(schedule: dict[str, Any] | None, iter_idx: int, tau: float) -> float:
    if schedule is None:
        return tau
    decay = schedule["decay"]
    tau_start = float(schedule["tau_start"])
    tau_min = float(schedule["tau_min"])
    iter_at = int(schedule.get("iter_count", 0)) + iter_idx
    if decay == "geometric":
        raw = tau_start * (float(schedule["rate"]) ** iter_at)
    elif decay == "linear":
        steps = int(schedule["steps"])
        if iter_at >= steps:
            raw = tau_min
        else:
            raw = tau_start + (iter_at / steps) * (tau_min - tau_start)
    else:
        raw = tau_start / (1.0 + iter_at)
    return max(tau_min, raw)


def _assignment_from_logits(
    logits: np.ndarray, assignment_prior: Literal["softmax", "ibp_map"], tau: float
) -> np.ndarray:
    if assignment_prior == "softmax":
        return _softmax(logits, tau)
    return _sigmoid(logits / tau)


def _softmax(logits: np.ndarray, tau: float) -> np.ndarray:
    scaled = logits / tau
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    ex = np.exp(shifted)
    return ex / ex.sum(axis=1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _assignment_jvp(
    assignments: np.ndarray,
    grad_a: np.ndarray,
    assignment_prior: Literal["softmax", "ibp_map"],
    tau: float,
) -> np.ndarray:
    if assignment_prior == "softmax":
        return _softmax_jvp(assignments, grad_a, tau)
    return grad_a * assignments * (1.0 - assignments) / tau


def _softmax_jvp(assignments: np.ndarray, grad_a: np.ndarray, tau: float) -> np.ndarray:
    mean = np.sum(assignments * grad_a, axis=1, keepdims=True)
    return assignments * (grad_a - mean) / tau


def _assignment_prior_value(
    assignments: np.ndarray,
    assignment_prior: Literal["softmax", "ibp_map"],
    lambda_sparse: float,
    alpha: float,
) -> float:
    if assignment_prior == "softmax":
        a = np.clip(assignments, 1e-300, 1.0)
        return float(lambda_sparse * np.sum(-a * np.log(a)))
    pi = _ibp_pi_map(assignments, alpha)
    z = np.clip(assignments, 1e-12, 1.0 - 1e-12)
    p = np.clip(pi, 1e-12, 1.0 - 1e-12)
    nll = -np.sum(z * np.log(p)[None, :] + (1.0 - z) * np.log(1.0 - p)[None, :])
    nll += np.sum((alpha / assignments.shape[1] - 1.0) * np.log(p))
    return float(nll)


def _assignment_prior_grad_logits(
    assignments: np.ndarray,
    assignment_prior: Literal["softmax", "ibp_map"],
    lambda_sparse: float,
    alpha: float,
    tau: float,
) -> np.ndarray:
    if assignment_prior == "softmax":
        d_h_da = -lambda_sparse * (np.log(np.clip(assignments, 1e-300, 1.0)) + 1.0)
        mean = np.sum(assignments * d_h_da, axis=1, keepdims=True)
        return assignments * (d_h_da - mean) / tau
    pi = np.clip(_ibp_pi_map(assignments, alpha), 1e-12, 1.0 - 1e-12)
    d_p_d_z = np.log((1.0 - pi) / pi)[None, :]
    return d_p_d_z * assignments * (1.0 - assignments) / tau


def _ibp_pi_map(assignments: np.ndarray, alpha: float) -> np.ndarray:
    n, k = assignments.shape
    a = alpha / k
    denom = max(float(n) + a - 1.0, 1e-9)
    raw = (assignments.sum(axis=0) + a - 1.0) / denom
    return np.clip(raw, 1e-9, 1.0 - 1e-9)


def _ard_value(coords: list[np.ndarray], log_ard: list[np.ndarray]) -> float:
    total = 0.0
    for t, lp in zip(coords, log_ard):
        for axis in range(t.shape[1]):
            alpha = float(np.exp(lp[axis]))
            total += 0.5 * alpha * float(np.sum(t[:, axis] ** 2))
            total -= 0.5 * t.shape[0] * float(lp[axis])
    return total


def _split_decoder(B: np.ndarray, widths: Sequence[int]) -> list[np.ndarray]:
    out = []
    cursor = 0
    for width in widths:
        out.append(B[cursor : cursor + width, :])
        cursor += width
    return out


def _block_diag(blocks: Sequence[np.ndarray]) -> np.ndarray:
    total = sum(b.shape[0] for b in blocks)
    out = np.zeros((total, total), dtype=float)
    cursor = 0
    for block in blocks:
        n = block.shape[0]
        out[cursor : cursor + n, cursor : cursor + n] = block
        cursor += n
    return out


def _edf_proxy(fit: SaeManifoldFitResult) -> float:
    return float(sum(atom.decoder_coefficients.size + atom.coords.shape[1] for atom in fit.atoms))


def _as_2d_float(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D numeric array; got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


__all__ = [
    "GumbelTemperatureSchedule",
    "SaeManifoldAtomFit",
    "SaeManifoldFitResult",
    "gumbel_geometric_schedule",
    "gumbel_linear_schedule",
    "gumbel_reciprocal_iter_schedule",
    "sae_manifold_fit",
]
