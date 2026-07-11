"""PyTorch bridge for gamfit's analytic primitives.

This subpackage exposes the gamfit Rust engine to PyTorch users so analytic
closed-form derivatives can participate in ``loss.backward()`` flows. Every
function here is a thin wrapper over the corresponding NumPy entry point in
:mod:`gamfit._api`; no derivative math is rewritten in Python or torch.

Layout:

- Closed-form Gaussian REML fits — :func:`gaussian_reml_fit`, its batched
  variant :func:`gaussian_reml_fit_batched`, the multi-smooth additive
  wrapper :func:`gaussian_reml_fit_additive`, and the multi-block
  per-smooth-λ entrypoint :func:`gaussian_reml_fit_blocks`. The
  closed-form paths have analytic forward and backward through Rust; the
  multi-block per-smooth-λ path also exposes backward VJPs through the Rust
  bridge for designs, penalties, response, and optional row weights.
- Basis evaluations — :func:`bspline_basis` and :func:`duchon_basis`. The
  B-spline backward routes through the Rust derivative primitive (chain
  rule applied at the boundary); the Duchon basis (any dimensionality)
  is currently forward-only with respect to its points input.
- Penalty matrix construction and the closed-form ridge solver — forward-only.
- Response-geometry transforms — forward-only numpy passthrough; for
  differentiable variants compose the underlying torch ops directly.
- :func:`from_fitted` — wrap a fitted :class:`gamfit.Model` as a frozen
  ``nn.Module``.

The subpackage requires torch: ``pip install torch``. Importing
:mod:`gamfit` itself does not pull in torch.
"""

from ._basis import (
    bspline_basis,
    bspline_basis_derivative,
    duchon_basis,
    gaussian_weighted_ridge,
    gaussian_weighted_ridge_batch,
    periodic_spline_curve_basis,
    smoothness_penalty,
    sphere_basis,
)
from ._reml import (
    AdditiveRemlOutput,
    GaussianRemlOutput,
    gaussian_reml_fit,
    gaussian_reml_fit_additive,
    gaussian_reml_fit_batched,
    gaussian_reml_fit_blocks,
)
from .geometry import (
    alr,
    closure,
    clr,
    inverse_alr,
    simplex_exp_map,
    simplex_frechet_mean,
    simplex_log_map,
    sphere_exp_map,
    sphere_frechet_mean,
    sphere_log_map,
)
from .manifold_sae import (
    CircularConcordanceReport,
    CircularPairConcordance,
    CircularReplicateCoverage,
    ManifoldSAE,
    ManifoldSAEOutput,
    circular_concordance,
)
from .modules import (
    AdaptiveTopK,
    GatedSAEDecoder,
    SoftmaxAssignmentSparsityPenalty,
    SparsityPenalty,
    TopKActivationPenalty,
    from_fitted,
)
from .penalties import (
    ARDPenalty,
    BlockOrthogonalityPenalty,
    GumbelTemperatureSchedule,
    OrderedBetaBernoulliPenalty,
    IsometryPenalty,
    IvaeRidgeMeanGauge,
    LazyPcaBasis,
    MechanismSparsityPenalty,
    MonotonicityPenalty,
    RiemannianGradientDescent,
    SmoothThresholdPenalty,
    TopologyAutoSelector,
)
from .interchange import InterchangeSwapDecoder
from .hyperbolic import PoincareAtoms
from .skip_transcoder import (
    SkipAffineSmooth,
    SkipTranscoderCandidateFailed,
    SkipTranscoderContinuousNonConvergence,
    SkipTranscoderFailedCandidate,
    SkipTranscoderProfile,
    SkipTranscoderSelectionCertificate,
    SkipTranscoderSelectionError,
    SkipTranscoderSelectionResult,
    SkipTranscoderTrial,
    select_skip_transcoder,
    skip_transcoder,
)
from ..smooth import (
    BSpline,
    Categorical,
    Duchon,
    Matern,
    Pca,
    PeriodicSplineCurve,
    ShapeConstraintLiteral,
    Smooth,
    Sphere,
    TensorBSpline,
)
from .fit import FitResult, fit
from .module import GAM
from .harvest import (
    HarvestShard,
    harvest_behavioral_fisher_probes,
    harvest_downstream_output_fisher_factors,
    harvest_output_fisher_factors,
    load_harvest_shard,
    save_harvest_shard,
)

__all__ = [
    # Recommended user-facing API
    "fit",
    "FitResult",
    "GAM",
    "HarvestShard",
    "harvest_behavioral_fisher_probes",
    "harvest_downstream_output_fisher_factors",
    "harvest_output_fisher_factors",
    "load_harvest_shard",
    "save_harvest_shard",
    "AdaptiveTopK",
    "ARDPenalty",
    "BlockOrthogonalityPenalty",
    "GumbelTemperatureSchedule",
    "OrderedBetaBernoulliPenalty",
    "InterchangeSwapDecoder",
    "IsometryPenalty",
    "IvaeRidgeMeanGauge",
    "LazyPcaBasis",
    "CircularConcordanceReport",
    "CircularPairConcordance",
    "CircularReplicateCoverage",
    "ManifoldSAE",
    "ManifoldSAEOutput",
    "circular_concordance",
    "PoincareAtoms",
    "SkipAffineSmooth",
    "SkipTranscoderCandidateFailed",
    "SkipTranscoderContinuousNonConvergence",
    "SkipTranscoderFailedCandidate",
    "SkipTranscoderProfile",
    "SkipTranscoderSelectionCertificate",
    "SkipTranscoderSelectionError",
    "SkipTranscoderSelectionResult",
    "SkipTranscoderTrial",
    "select_skip_transcoder",
    "skip_transcoder",
    "GatedSAEDecoder",
    "MechanismSparsityPenalty",
    "MonotonicityPenalty",
    "RiemannianGradientDescent",
    "SmoothThresholdPenalty",
    "SoftmaxAssignmentSparsityPenalty",
    "SparsityPenalty",
    "TopKActivationPenalty",
    "TopologyAutoSelector",
    "Smooth",
    "Duchon",
    "BSpline",
    "TensorBSpline",
    "Matern",
    "Pca",
    "PeriodicSplineCurve",
    "ShapeConstraintLiteral",
    "Sphere",
    "Categorical",
    # Lower-level primitives (advanced use cases)
    "AdditiveRemlOutput",
    "GaussianRemlOutput",
    "alr",
    "bspline_basis",
    "bspline_basis_derivative",
    "closure",
    "clr",
    "duchon_basis",
    "from_fitted",
    "gaussian_reml_fit",
    "gaussian_reml_fit_additive",
    "gaussian_reml_fit_batched",
    "gaussian_reml_fit_blocks",
    "gaussian_weighted_ridge",
    "gaussian_weighted_ridge_batch",
    "inverse_alr",
    "periodic_spline_curve_basis",
    "simplex_exp_map",
    "simplex_frechet_mean",
    "simplex_log_map",
    "smoothness_penalty",
    "sphere_basis",
    "sphere_exp_map",
    "sphere_frechet_mean",
    "sphere_log_map",
]
