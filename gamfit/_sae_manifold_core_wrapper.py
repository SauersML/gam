"""STAGING ARTIFACT (#2091) — the thin `ManifoldSAE` wrapper that composes a
Rust-owned `ManifoldSaeCore`, replacing the legacy `@dataclass ManifoldSAE`.

DO NOT IMPORT THIS AT RUNTIME YET. This file is authored out-of-tree so it does
NOT collide with the concurrent #2011 edits in `gamfit/_sae_manifold.py`. When
(a) the fresh gamfit wheel lands (so pytest can run the pyclass parity oracle)
and (b) #2011 commits `_sae_manifold.py`, the coordinator performs the SWAP
described in the DIFF-PLAN block below and runs the oracle.

──────────────────────────────────────────────────────────────────────────────
DESIGN (direction W, blessed by team-lead)
──────────────────────────────────────────────────────────────────────────────
STATE lives in Rust: the wrapper holds exactly one `ManifoldSaeCore` (`self._core`)
plus a Python-side `training_data` overlay handle (the core never retains X).
Every state field the old dataclass exposed is read straight off the core's
getters via `__getattr__` (single source of truth — satisfies #2091's title).

ORCHESTRATION stays in Python: predict / reconstruct / encode / steer / project /
converged_latents / shape_uncertainty / summary / description_length /
structure_certificate / atom_inference / curvature / … remain unchanged Python
methods — they are numpy / Rust-FFI / torch-interop orchestration (NOT state), so
per the "Python is a thin wrapper (torch interop only)" rule they belong here.
Their bodies are UNCHANGED: they read `self.<field>`, which now resolves through
`__getattr__` to the core getter, so they keep working verbatim.

KEY CONSEQUENCE FOR THE SWAP: the ~40 orchestration method bodies (currently
`_sae_manifold.py` lines ~1212–2315, i.e. everything between the class scaffolding
and `to_dict`) DO NOT CHANGE. The swap replaces ONLY the dataclass scaffolding and
the construction/serialization paths. Re-transcribing the unchanged methods would
only add transcription risk, so this staging file provides the COMPLETE new
scaffolding (below) and the DIFF-PLAN pins exactly which method bodies stay put.

──────────────────────────────────────────────────────────────────────────────
DIFF-PLAN — exact edits to `gamfit/_sae_manifold.py` at swap time
──────────────────────────────────────────────────────────────────────────────
Line numbers are against the current HEAD of `_sae_manifold.py` (verify before
applying — #2011 may shift them by a few lines around `_trust_scores`).

  1. REPLACE the dataclass scaffolding of class `ManifoldSAE`:
       - DELETE `@dataclass(slots=True)` decorator (line ~673) and ALL field
         declarations + defaults (lines ~696–874, `atoms:` … `_lazy_artifact:`).
       - DELETE `__post_init__` (876–892), `__getattribute__` (894–895),
         `__setattr__` (897–898).
       - KEEP the `chosen_k` property (900–909) but repoint its body to
         `int(self._core.chosen_k)` (see WRAPPER below).
       - KEEP the `reml_score` property (911–920) → `self._core.reml_score`.
       - KEEP `__repr__` (922–936) VERBATIM (reads self.fitted/self.coords/
         description_length — all resolve via __getattr__).
       - REPLACE `from_payload` (938–1210) with WRAPPER.from_payload below.
       Insert the WRAPPER `__init__` / `__slots__` / `__getattr__` / `__setattr__` /
       `training_data` / `low_level` properties in place of the deleted fields.

  2. KEEP UNCHANGED (do NOT touch) the orchestration methods, lines ~1212–2154:
       structure_certificate, atom_inference, contested_claims, _atom_index,
       atom_trust, atom_diagnostics, curvature, atom_curvature,
       coordinate_fidelity_report, atom_coordinate_fidelity,
       topology_persistence_report, atom_topology_persistence,
       atom_angle_coordinate, shape_uncertainty, _oos_payload,
       _hybrid_linear_images_for_oos, _is_training_data, build_encode_atlas,
       reconstruct, reconstruct_training, predict, distill_encoder, encode,
       converged_latents, _amortized_encoder, encode_amortized, project,
       atom_reconstruct, attach_fisher, steer, per_atom_active_set,
       per_atom_latent_for, featurize, get_decoder, get_anchors,
       description_length, summary.
     These read `self.<field>` (→ core) and `self.training_data` (→ overlay) and
     call `self._core` setters where they mutate (attach_fisher). No edit needed.

  3. REPLACE `to_dict` (2156–2315) with WRAPPER.to_dict → `self._core.to_dict()`.
     REPLACE `save` (2317–2323) with WRAPPER.save → `self._core.to_json()`.
     REPLACE `from_dict` (2325–2537) with WRAPPER.from_dict → `ManifoldSaeCore(dict)`.
     KEEP `load` (2539–2547) VERBATIM (delegates to from_dict).

  4. REWIRE the MAINLINE fit return in `sae_manifold_fit`, lines 3283–3303:
       replace the `from_payload` + fisher-attach + `_preserve_linear_block_labels`
       block with the single builder call in `manifold_sae_from_fit_payload` below.

  5. REWIRE `StagewiseSAE.to_manifold_sae`, lines ~3520–3555: replace the direct
       `return ManifoldSAE(atoms=…, …)` with the `_stagewise_to_manifold_sae`
       body below (assemble a v1 to_dict-schema dict → `ManifoldSAE.from_dict`).

  6. DELETE `_preserve_linear_block_labels` (lines ~4421–4447) — it is subsumed by
       the builder's `declared_bases=` arm (commit 404f3871e). Confirm no other
       caller first (`grep _preserve_linear_block_labels`).

  COMPANION EDITS (outside _sae_manifold.py — surface under the pytest oracle):
   - tests/fixtures/manifold_sae/generate_golden.py:169/186 constructs
     `ManifoldSAE(_oos_projection_top1=…, structured_residual_diagnostics=…, …)`
     with dataclass kwargs. After the flip `__init__` takes `(core, *, training_data)`,
     so the generator must build a v1 dict and call `ManifoldSAE.from_dict(dict)`
     (or `ManifoldSaeCore(dict)`), then set the write-dropped
     `structured_residual_diagnostics` on the DICT (it is read-tolerated / never
     re-emitted — golden_roundtrip asserts exactly that).
   - Any test that constructs `ManifoldSAE(field=…)` directly (grep `ManifoldSAE(`)
     must move to `from_dict`. Tests using `from_payload`/`from_dict`/`load` are
     unaffected. Tests that set `fit.training_data = …` / `fit._oos_projection_top1
     = …` keep working (both are settable — see __setattr__).

──────────────────────────────────────────────────────────────────────────────
EQUIVALENCE RISKS to confirm under the pytest oracle (green = closed)
──────────────────────────────────────────────────────────────────────────────
 R1. [CLOSED] n_harmonics canonicalization on LOAD: Python `from_dict`
     canonicalized stale periodic n_harmonics via `_canonical_n_harmonics` (old
     line 2423). The core `ManifoldSaeCore(dict)` (`ManifoldSaePayload::from_json`)
     was a faithful serde parse that skipped it. FIXED in
     `manifold_sae_payload.rs::normalize_after_load` (calls the coercion module's
     `canonical_n_harmonics` on the decoder widths) — now both the fit-path builder
     and the from_json/load path canonicalize, matching legacy Python. Idempotent
     and a no-op on the golden fixture (`n_harmonics=[2,0,0]`, periodic H=2/width 5).
     Oracle: test_manifold_sae_golden_roundtrip.py + any periodic round-trip test.
 R2. `low_level` surface: consumers only read `.chosen_k`; the adapter also
     exposes atoms/fitted/assignments/coords/reml_score off the core. It does NOT
     carry `evidence_by_candidate` / `comparison` (not in the payload). If any
     consumer reads those off `low_level`, they must move to a diagnostics field.
     grep confirmed none do today.
 R3. mainline `penalties` arg: the builder stores `penalties` as the old
     from_payload did. In `sae_manifold_fit` the local `penalties` list is passed
     through unchanged. (The equiv test derives it as `fit.primitive_names[1:]`.)
 R4. to_manifold_sae parity: the SAC lift now round-trips through the v1 schema;
     confirm the SAC in-sample reconstruction + OOS surface match (examples/
     sac_experiments.py, test suites touching StagewiseSAE.to_manifold_sae).
──────────────────────────────────────────────────────────────────────────────

On swap, DELETE the `from gamfit._sae_manifold import (...)` shim below — those
names become local to `_sae_manifold.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# ── SWAP: delete this import block; these are module-local in _sae_manifold.py ──
# (The Rust pyclass is reached via `rust_module().ManifoldSaeCore` /
#  `rust_module().sae_manifold_core_from_fit_payload`, so no direct import of it.)
from gamfit._sae_manifold import (  # noqa: E402  (staging-only shim)
    _SaeTrainingDataHandle,
    _canonical_assignment,
    _jsonable_value,
    _training_data_handle,
    rust_module,
)
# ─────────────────────────────────────────────────────────────────────────────


class _LowLevelView:
    """Read-only adapter reproducing the `SaeManifoldFitResult` surface consumers
    touch off `model.low_level` — in practice only `.chosen_k` (verified by grep:
    test_sae_facade_audit / test_sae_manifold_softmax_dispatch /
    test_sae_dictionary_k_preserved). The remaining fields are forwarded off the
    core for faithfulness; `evidence_by_candidate` / `comparison` are not in the
    payload and are intentionally absent (see R2)."""

    __slots__ = ("_core",)

    def __init__(self, core: "ManifoldSaeCore") -> None:
        self._core = core

    @property
    def chosen_k(self) -> int:
        return int(self._core.chosen_k)

    @property
    def atoms(self) -> list[Any]:
        return list(self._core.atoms)

    @property
    def fitted(self) -> np.ndarray:
        return self._core.fitted

    @property
    def assignments(self) -> np.ndarray:
        return self._core.assignments

    @property
    def coords(self) -> list[np.ndarray]:
        return list(self._core.coords)

    @property
    def reml_score(self) -> float | None:
        return self._core.reml_score


class ManifoldSAE:
    """Fitted SAE-manifold model — a thin wrapper over a Rust-owned
    `ManifoldSaeCore` (#2091). All fit STATE (atoms, decoder blocks, coords,
    assignments, scalar config, report/certificate blocks, Fisher shard, selected
    ρ*) lives in the core and is read through `__getattr__`; the only Python-side
    field is the `training_data` metadata overlay (the core never retains X).

    The public method surface (predict/reconstruct/encode/steer/project/…) is
    UNCHANGED from the legacy dataclass and lives in `_sae_manifold.py`; this class
    body in staging shows only the scaffolding that replaces the dataclass storage.
    """

    __slots__ = ("_core", "_training_data")

    # Private design-field names whose core getter drops the leading underscore.
    _CORE_ALIASES: dict[str, str] = {
        "_basis_kinds": "basis_kinds",
        "_atom_dims": "atom_dims",
        "_basis_sizes": "basis_sizes",
        "_n_harmonics": "n_harmonics",
        "_duchon_centers": "duchon_centers",
        "_oos_projection_top1": "oos_projection_top1",
    }

    # Fields with a Rust setter (attach_fisher / test mutation); everything else is
    # read-only Rust-owned state.
    _SETTABLE_CORE: frozenset[str] = frozenset(
        {
            "fisher_factors",
            "fisher_mass_residual",
            "fisher_provenance",
            "metric_provenance",
            "oos_projection_top1",
        }
    )

    def __init__(
        self,
        core: "ManifoldSaeCore",
        *,
        training_data: Any = None,
    ) -> None:
        object.__setattr__(self, "_core", core)
        if training_data is None:
            # New fits do not retain X: expose a zero-byte handle whose shape/dtype
            # mirror the fitted array (matches the legacy lazy-artifact behavior
            # test_sae_manifold_lazy_result_artifact.py asserts: nbytes==0, asarray
            # raises, `is not x`).
            fitted = core.fitted
            training_data = _SaeTrainingDataHandle(tuple(fitted.shape), fitted.dtype)
        object.__setattr__(self, "_training_data", training_data)

    # ── state delegation ─────────────────────────────────────────────────────
    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup (slots / methods / properties) misses,
        # i.e. for a STATE field → forward to the core getter.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        core = object.__getattribute__(self, "_core")
        alias = type(self)._CORE_ALIASES.get(name, name)
        try:
            return getattr(core, alias)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r} "
                f"(not a ManifoldSaeCore field)"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_core", "_training_data"):
            object.__setattr__(self, name, value)
            return
        if name == "training_data":
            object.__setattr__(self, "_training_data", value)
            return
        alias = type(self)._CORE_ALIASES.get(name, name)
        if alias in type(self)._SETTABLE_CORE:
            setattr(self._core, alias, value)
            return
        raise AttributeError(
            f"ManifoldSAE.{name} is read-only Rust-owned state (no core setter); "
            f"settable: {sorted(type(self)._SETTABLE_CORE)} + training_data"
        )

    # ── Python-side overlay / forwarded properties ───────────────────────────
    @property
    def training_data(self) -> Any:
        return self._training_data

    @property
    def low_level(self) -> _LowLevelView:
        return _LowLevelView(self._core)

    @property
    def chosen_k(self) -> int:
        """Discovered atom count (= len(atoms)); forwards to the core."""
        return int(self._core.chosen_k)

    @property
    def reml_score(self) -> float | None:
        """DEPRECATED read alias for `penalized_loss_score` (#1231)."""
        return self._core.reml_score

    def __repr__(self) -> str:
        # VERBATIM from the legacy dataclass — reads self.fitted/self.coords/
        # description_length, all resolved via __getattr__ / the kept method.
        d_atom = int(self.coords[0].shape[1]) if self.coords else 0
        n, p = (self.fitted.shape if self.fitted.ndim == 2 else (self.fitted.shape[0], 1))
        try:
            dl = self.description_length()
        except Exception:
            dl = None
        bpt = "n/a" if dl is None else f"{float(dl['bits_per_token']):.3f}"
        return (
            f"ManifoldSAE(bits/token={bpt}, K={len(self.atoms)}, d_atom={d_atom}, "
            f"atom_topology={self.atom_topology!r}, assignment={self.assignment!r}, "
            f"alpha={self.alpha!r}, learnable_alpha={self.learnable_alpha}, "
            f"n={n}, p={p}, r2={self.reconstruction_r2:.3f})"
        )

    # ── serialization: routed through the core's serde (single source of truth) ─
    def to_dict(self) -> dict[str, Any]:
        """Round-trippable v1 JSON-compatible serialization — the core owns the
        schema (`reml_score` write-alias, `structured_residual_diagnostics`
        write-drop, channel-cov-factor compaction all live in Rust serde now)."""
        return dict(self._core.to_dict())

    def to_json(self) -> str:
        return self._core.to_json()

    def save(self, path: str | Path) -> None:
        """Write this fit to `path` as the canonical JSON payload the core emits."""
        Path(path).write_text(self._core.to_json())

    # ── constructors ─────────────────────────────────────────────────────────
    @classmethod
    def from_payload(
        cls,
        x: np.ndarray,
        payload: Mapping[str, Any],
        topology: str,
        assignment: str,
        penalties: list[str],
        alpha: float = 1.0,
        learnable_alpha: bool = False,
        *,
        assignment_label: str | None = None,
        tau: float = 0.5,
        sparsity_strength: float = 1.0,
        smoothness: float = 1.0,
        learning_rate: float = 0.04,
        max_iter: int = 50,
        random_state: int = 0,
        top_k: int | None = None,
        jumprelu_threshold: float = 0.0,
        fisher_factors: np.ndarray | None = None,
        fisher_provenance: str | None = None,
        declared_bases: list[str] | None = None,
    ) -> "ManifoldSAE":
        """Build the Rust-owned core from the RAW `sae_manifold_fit_minimal`
        payload and wrap it. The `sae_manifold_core_from_fit_payload` builder
        reproduces the legacy `from_payload ∘ to_dict` bit-for-bit AND folds in the
        post-fit Fisher attach (`fisher_factors`/`fisher_provenance`) and the
        `linear_block` relabel (`declared_bases`), so those become builder args
        rather than post-construction mutations. Signature/arg-order pinned by
        tests/test_sae_manifold_core_from_fit_payload_equiv.py."""
        canonical_assignment = _canonical_assignment(assignment, "assignment")
        raw_json = json.dumps(_jsonable_value(dict(payload)))
        core = rust_module().sae_manifold_core_from_fit_payload(
            raw_json,
            np.ascontiguousarray(np.asarray(x, dtype=np.float64)),
            str(topology),
            canonical_assignment,
            str(assignment if assignment_label is None else assignment_label),
            list(penalties),
            float(alpha),
            bool(learnable_alpha),
            float(tau),
            float(sparsity_strength),
            float(smoothness),
            float(learning_rate),
            int(max_iter),
            int(random_state),
            top_k,
            float(jumprelu_threshold),
            fisher_factors=(
                None
                if fisher_factors is None
                else np.ascontiguousarray(np.asarray(fisher_factors, dtype=np.float64))
            ),
            fisher_provenance=fisher_provenance,
            declared_bases=None if declared_bases is None else list(declared_bases),
        )
        # New fits do not retain X: overlay a zero-byte handle mirroring the input
        # shape/dtype (the legacy from_payload used `_training_data_handle(x)`).
        return cls(core, training_data=_training_data_handle(np.asarray(x)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ManifoldSAE":
        """Reconstruct from a `to_dict` payload. The core's `#[new]` validates the
        `gamfit.ManifoldSAE/v1` schema tag and parses through
        `ManifoldSaePayload::from_json` (the same NaN→reject, `reml_score`
        fallback, and channel-cov reconstruction the legacy reader had).
        n_harmonics canonicalization on load is now owned by the core's
        `ManifoldSaePayload::normalize_after_load` (R1 CLOSED — matches the legacy
        Python `_canonical_n_harmonics` + the fit-path builder), so no pre-core
        Python canonicalization is needed here."""
        core = rust_module().ManifoldSaeCore(dict(payload))
        # training_data: the payload always writes null (new fits do not retain X);
        # mirror the legacy from_dict, which built a handle from the fitted shape.
        fitted = np.asarray(payload["fitted"], dtype=float)
        raw_td = payload.get("training_data")
        training_data = (
            _SaeTrainingDataHandle(tuple(fitted.shape), fitted.dtype)
            if raw_td is None
            else np.asarray(raw_td, dtype=float)
        )
        return cls(core, training_data=training_data)

    @classmethod
    def load(cls, path: str | Path) -> "ManifoldSAE":
        """Load a fit written by `save` (VERBATIM from the legacy classmethod)."""
        return cls.from_dict(json.loads(Path(path).read_text()))


# ═══════════════════════════════════════════════════════════════════════════════
# Rewired fit-route bodies (drop into _sae_manifold.py per DIFF-PLAN steps 4–5)
# ═══════════════════════════════════════════════════════════════════════════════

def manifold_sae_from_fit_payload(
    *,
    x: np.ndarray,
    payload: Mapping[str, Any],
    resolved_topology: str,
    kind: str,
    assignment: Any,
    penalties: list[str],
    alpha_value: float,
    alpha_is_auto: bool,
    tau: float,
    sparsity: float,
    smoothness: float,
    effective_lr: float,
    max_iter_total: int,
    random_state: int,
    top_k_arg: int | None,
    jumprelu_threshold: float,
    fisher_shard: tuple[Any, Any, str] | None,
    bases: list[str],
) -> "ManifoldSAE":
    """MAINLINE `sae_manifold_fit` return — replaces `_sae_manifold.py` 3283–3303.

    ONE builder call replaces `from_payload` + the post-fit `model.fisher_factors`
    attach + `_preserve_linear_block_labels(model, bases)`: the fisher shard is
    threaded via `fisher_factors=`/`fisher_provenance=` and the linear_block
    relabel via `declared_bases=bases` (both folded into the Rust builder)."""
    return ManifoldSAE.from_payload(
        x,
        dict(payload),
        resolved_topology,
        kind,
        penalties,
        assignment_label=str(assignment),
        alpha=float(alpha_value),
        learnable_alpha=bool(alpha_is_auto),
        tau=float(tau),
        sparsity_strength=float(sparsity),
        smoothness=float(smoothness),
        learning_rate=float(effective_lr),
        max_iter=int(max_iter_total),
        random_state=int(random_state),
        top_k=top_k_arg,
        jumprelu_threshold=float(jumprelu_threshold),
        fisher_factors=None if fisher_shard is None else np.ascontiguousarray(fisher_shard[0]),
        fisher_provenance=None if fisher_shard is None else fisher_shard[2],
        declared_bases=list(bases),
    )


def stagewise_to_manifold_sae_dict(
    *,
    basis_kinds: list[str],
    decoder_blocks: list[np.ndarray],
    atom_dims: list[int],
    basis_sizes: list[int],
    n_harmonics: list[int],
    coords: list[np.ndarray],
    assignments: np.ndarray,
    fitted: np.ndarray,
    logits: np.ndarray,
    training: np.ndarray,
    assignment: str,
    reconstruction_r2: float,
    seed: Any,
    atom_topology: str,
    atom_topologies: list[str],
) -> dict[str, Any]:
    """Assemble a v1 `to_dict`-schema payload for `StagewiseSAE.to_manifold_sae`
    (DIFF-PLAN step 5). The SAC lift no longer constructs `ManifoldSAE(...)` with
    dataclass kwargs; it builds this dict and calls `ManifoldSAE.from_dict(...)`,
    so the core owns the state exactly like every other route. Per-atom entries
    carry the minimal SAC fields (no shape bands / covariance — SAC lifts a frozen
    dictionary), matching what the old direct construction supplied."""
    atoms_payload: list[dict[str, Any]] = []
    for k, block in enumerate(decoder_blocks):
        atoms_payload.append(
            {
                "basis": basis_kinds[k],
                "decoder_coefficients": np.asarray(block, dtype=float).tolist(),
                "assignments": np.asarray(assignments[:, k], dtype=float).tolist(),
                "coords": np.asarray(coords[k], dtype=float).tolist(),
                "coords_u_arc": None,
                "evidence": None,
                "active_dim": int(atom_dims[k]),
                "decoder_covariance_channel_factors": None,
                "shape_band_coords": None,
                "shape_band_mean": None,
                "shape_band_sd": None,
                "functional_evidence": None,
            }
        )
    return {
        "schema": "gamfit.ManifoldSAE/v1",
        "atom_topology": atom_topology,
        "atom_topologies": list(atom_topologies),
        "assignment": assignment,
        "assignment_label": str(seed.assignment) if hasattr(seed, "assignment") else assignment,
        "alpha": float(seed.alpha),
        "learnable_alpha": bool(seed.learnable_alpha),
        "tau": float(seed.tau),
        "sparsity_strength": float(seed.sparsity_strength),
        "smoothness": float(seed.smoothness),
        "learning_rate": float(seed.learning_rate),
        "max_iter": int(seed.max_iter),
        "random_state": int(seed.random_state),
        "top_k": None,
        "top_k_projection": None,
        "pre_topk": None,
        "jumprelu_threshold": float(seed.jumprelu_threshold),
        "oos_projection_top1": False,
        "dispersion": 1.0,
        "solver_plan": None,
        "primitive_names": ["sae_manifold_fit_stagewise"],
        "basis_specs": list(basis_kinds),
        "penalized_loss_score": None,
        "reml_score": None,
        "reconstruction_r2": float(reconstruction_r2),
        # SPEC: the centering-mean REDUCTION is computed in Rust (the shared
        # `column_mean` core, identical to the fit builder), never in production
        # Python. This call is pure marshaling of the result.
        "training_mean": rust_module()
        .sae_manifold_training_mean(np.ascontiguousarray(np.asarray(training, dtype=float)))
        .tolist(),
        "training_data": None,
        "training_data_retained": False,
        "fitted": np.asarray(fitted, dtype=float).tolist(),
        "assignments": np.asarray(assignments, dtype=float).tolist(),
        "logits": np.asarray(logits, dtype=float).tolist(),
        "diagnostics": {"atom_trust": [], "atoms": []},
        "coords": [np.asarray(c, dtype=float).tolist() for c in coords],
        "decoder_blocks": [np.asarray(b, dtype=float).tolist() for b in decoder_blocks],
        "atoms": atoms_payload,
        "basis_kinds": list(basis_kinds),
        "atom_dims": [int(d) for d in atom_dims],
        "basis_sizes": [int(s) for s in basis_sizes],
        "n_harmonics": [int(h) for h in n_harmonics],
        "duchon_centers": [None] * len(decoder_blocks),
        "atom_two_lens": None,
        "residual_gauge": None,
        "incoherence_report": None,
        "curvature_report": None,
        "coordinate_fidelity": None,
        "topology_persistence": None,
        "atom_inference": None,
        "certificates": None,
        "structure_certificate": None,
        "cotrain": None,
        "hybrid_split": None,
        "fisher_factors": None,
        "fisher_provenance": None,
        "metric_provenance": "Euclidean",
        "fisher_mass_residual": None,
        "selected_log_lambda_sparse": None,
        "selected_log_lambda_smooth": None,
        "selected_log_ard": None,
    }
