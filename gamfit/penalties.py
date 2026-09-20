"""Analytic penalty descriptors for latent-coordinate and SAE-manifold fits."""

from __future__ import annotations

from ._penalties import (
    ARDPenalty,
    AnalyticPenaltyKind,
    AuxConditionalPriorPenalty,
    BlockOrthogonalityPenalty,
    BlockSparsityPenalty,
    GatedSAEDecoder,
    OrderedBetaBernoulliPenalty,
    IsometryPenalty,
    IvaeRidgeMeanGauge,
    SmoothThresholdPenalty,
    MechanismSparsityPenalty,
    NuclearNormPenalty,
    OrthogonalityPenalty,
    ParametricAuxConditionalPriorPenalty,
    Penalty,
    PENALTY_MANIFEST,
    ScadMcpPenalty,
    SoftmaxAssignmentSparsityPenalty,
    SparsityPenalty,
    TopKActivationPenalty,
    TotalVariationPenalty,
)
from ._sheaf import (
    SheafConsistencyPenalty,
)
from ._protocol import (
    PenaltyDescriptor,
)
from ._composite_penalty import (
    CompositePenalty,
)

__all__ = [
    "AnalyticPenaltyKind",
    "ARDPenalty",
    "AuxConditionalPriorPenalty",
    "BlockOrthogonalityPenalty",
    "BlockSparsityPenalty",
    "CompositePenalty",
    "GatedSAEDecoder",
    "IsometryPenalty",
    "IvaeRidgeMeanGauge",
    "MechanismSparsityPenalty",
    "NuclearNormPenalty",
    "OrderedBetaBernoulliPenalty",
    "OrthogonalityPenalty",
    "ParametricAuxConditionalPriorPenalty",
    "Penalty",
    "PENALTY_MANIFEST",
    "PenaltyDescriptor",
    "ScadMcpPenalty",
    "SheafConsistencyPenalty",
    "SmoothThresholdPenalty",
    "SoftmaxAssignmentSparsityPenalty",
    "SparsityPenalty",
    "TopKActivationPenalty",
    "TotalVariationPenalty",
]
