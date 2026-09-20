"""Result, prediction, and posterior-sample objects returned by gamfit.

These are produced by :func:`gamfit.fit` and the methods of
:class:`gamfit.Model` (``predict``, ``summary``, ``diagnose``, ``sample``,
...); they are exported here for type annotations and ``isinstance`` checks.
"""

from __future__ import annotations

from ._diagnostics import (
    Diagnostics,
)
from ._model import (
    AffineDesign,
    CompetingRisksCIF,
    CompetingRisksPrediction,
    MultinomialPrediction,
    SurvivalPrediction,
    TermBlock,
)
from ._sampling import (
    CumulativeIncidenceDraws,
    PairedPosteriorSamples,
    PosteriorPredictive,
    PosteriorSamples,
    SamplingConfig,
)
from ._partial_effect import (
    AxisLevels,
    PartialEffect,
)
from ._tables import (
    PredictionResult,
)
from ._schema import (
    SchemaCheck,
    SchemaIssue,
)
from ._summary import (
    Summary,
)
from ._validation import (
    FormulaValidation,
)

__all__ = [
    "AffineDesign",
    "AxisLevels",
    "CompetingRisksCIF",
    "CompetingRisksPrediction",
    "CumulativeIncidenceDraws",
    "Diagnostics",
    "FormulaValidation",
    "MultinomialPrediction",
    "PairedPosteriorSamples",
    "PartialEffect",
    "PosteriorPredictive",
    "PosteriorSamples",
    "PredictionResult",
    "SamplingConfig",
    "SchemaCheck",
    "SchemaIssue",
    "Summary",
    "SurvivalPrediction",
    "TermBlock",
]
