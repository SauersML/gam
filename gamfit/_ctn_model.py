"""A saved CTN plus an ordinary marginal-slope predictor.

Only O(n) generated scores are retained during cross-fitting. No influence
Jacobian or fitted-and-discarded mean term is constructed by this path.
"""
from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from ._calibrated_slope import CtnStage1
from ._tables import _table_column_views

_SCHEMA = "gamfit.CtnMarginalSlopeModel/v1"
_SCORE = "__gamfit_ctn_score"


def _slice(data, rows):
    columns, kind = _table_column_views(data)
    if kind == "pandas":
        return data.iloc[rows].reset_index(drop=True)
    if kind == "polars":
        return data[rows.tolist()]
    if kind == "pyarrow":
        return data.take(rows)
    return {key: np.asarray(value)[rows] for key, value in columns.items()}


def _with_columns(data, added):
    columns, kind = _table_column_views(data)
    if kind == "pandas":
        return data.assign(**added)
    if kind == "polars":
        import polars as pl
        return data.with_columns([pl.Series(name, values) for name, values in added.items()])
    if kind == "pyarrow":
        import pyarrow as pa
        for name, values in added.items():
            index = data.schema.get_field_index(name)
            data = (data.set_column(index, name, pa.array(values)) if index >= 0
                    else data.append_column(name, pa.array(values)))
        return data
    return {**columns, **added}


def _with_score(data, scores, score_column=_SCORE):
    columns, _ = _table_column_views(data)
    if score_column in columns:
        raise ValueError(f"reserved generated-score column: {score_column}")
    return _with_columns(data, {score_column: scores})


def _labels(columns, name, n):
    if name not in columns:
        raise ValueError(f"missing cross-fit column {name!r}")
    labels = []
    for value in columns[name]:
        if hasattr(value, "as_py"):
            value = value.as_py()
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must contain nonmissing string or numeric labels")
        if value == "" or isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{name} contains an empty or nonfinite label")
        labels.append(json.dumps(value, allow_nan=False))
    if len(labels) != n:
        raise ValueError("cross-fit column length differs from response")
    return labels


def crossfit_assignment(data, recipe):
    """Return validated row-aligned folds; group assignment ignores row order."""
    columns, _ = _table_column_views(data)
    if recipe.response not in columns:
        raise ValueError(f"missing score response {recipe.response!r}")
    n = len(columns[recipe.response])
    groups = _labels(columns, recipe.group_column, n) if recipe.group_column else None
    if recipe.fold_column:
        labels = _labels(columns, recipe.fold_column, n)
        levels = sorted(set(labels))
        if len(levels) < 2:
            raise ValueError("cross-fitting requires at least two nonempty folds")
        mapping = {label: i for i, label in enumerate(levels)}
        folds = np.array([mapping[label] for label in labels], dtype=np.int64)
    else:
        levels = sorted(set(groups), key=lambda g: hashlib.sha256(
            f"{recipe.seed}:{g}".encode()).digest())
        if len(levels) < recipe.folds:
            raise ValueError("fewer independent groups than requested folds")
        mapping = {group: i % recipe.folds for i, group in enumerate(levels)}
        folds = np.array([mapping[group] for group in groups], dtype=np.int64)
    if groups is not None:
        assigned = {}
        for group, fold in zip(groups, folds):
            if group in assigned and assigned[group] != fold:
                raise ValueError("a group crosses CTN fold boundaries")
            assigned[group] = fold
    if any(np.count_nonzero(folds != fold) < 2 for fold in np.unique(folds)):
        raise ValueError("every CTN fold needs at least two training observations")
    return folds


def _scores(model, data):
    columns, _ = _table_column_views(data)
    z = np.asarray(model.transformation_score(data), dtype=float)
    if z.shape != (len(next(iter(columns.values()))),) or not np.isfinite(z).all():
        raise ValueError("CTN produced invalid transformed scores")
    return z


class CtnMarginalSlopeModel:
    """One deployable predictor with a frozen full-training score transform.

    ``transform`` and ``outcome`` expose the two fitted components for audits.
    Prediction uncertainty, when requested, is conditional on the fitted CTN;
    it does not include nuisance-model estimation or model-selection uncertainty.
    """

    def __init__(self, transform, outcome, recipe, fold_digest, *, score_column=_SCORE):
        self.transform, self.outcome = transform, outcome
        self.recipe, self.fold_digest = recipe, fold_digest
        if not isinstance(score_column, str) or not score_column:
            raise ValueError("score_column must name the outcome's frozen latent-score column")
        self.score_column = score_column

    def transformation_score(self, data):
        return _scores(self.transform, data)

    def predict(self, data, **kwargs):
        return self.outcome.predict(_with_score(data, self.transformation_score(data), self.score_column), **kwargs)

    def survival_at(self, data, times, *, entry_time=0.0):
        """Prospective net survival conditional on event-free scalar entry time.

        Consumes raw PGS and baseline predictors only. ``times`` are absolute
        analysis times, not observed exit times. For time since recruitment,
        entry is zero. Disease-before-death incidence still requires a CIF.
        """
        times = np.asarray(times, dtype=float)
        if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
                or not np.isfinite(entry_time) or entry_time < 0
                or (times < entry_time).any() or (np.diff(times) <= 0).any()
                or times[-1] <= entry_time):
            raise ValueError("times must increase from entry and include a future horizon")
        response = self.outcome.formula.split("~", 1)[0].strip()
        match = re.fullmatch(r"Surv\(\s*([^()]+)\s*\)", response)
        if match is None:
            raise ValueError("survival_at requires a Surv outcome")
        names = [name.strip() for name in match.group(1).split(",")]
        if len(names) not in (2, 3) or any(not re.fullmatch(r"[A-Za-z_]\w*|0", name) for name in names):
            raise ValueError("survival_at requires named Surv columns with optional literal zero entry")
        columns, _ = _table_column_views(data)
        n = len(next(iter(columns.values())))
        added = {names[-2]: np.full(n, times[-1]), names[-1]: np.zeros(n)}
        if len(names) == 3 and names[0] != "0":
            added[names[0]] = np.full(n, entry_time)
        prediction = self.predict(_with_columns(data, added))
        cumulative = np.asarray(prediction.cumulative_hazard_at(times))
        at_entry = np.asarray(prediction.cumulative_hazard_at([entry_time]))
        return np.exp(-(cumulative - at_entry))

    def summary(self):
        return {"score_transform": self.transform.summary(), "outcome": self.outcome.summary(),
                "uncertainty": "conditional on fitted CTN", "fold_digest": self.fold_digest}

    def dumps(self):
        return json.dumps({"schema": _SCHEMA, "recipe": asdict(self.recipe),
                           "fold_digest": self.fold_digest,
                           "score_column": self.score_column,
                           "transform": base64.b64encode(self.transform.dumps()).decode("ascii"),
                           "outcome": base64.b64encode(self.outcome.dumps()).decode("ascii")},
                          allow_nan=False).encode("utf-8")

    def save(self, path):
        Path(path).write_bytes(self.dumps())

    @classmethod
    def from_payload(cls, payload):
        from ._api import loads
        if set(payload) != {"schema", "recipe", "fold_digest", "score_column", "transform", "outcome"} or payload["schema"] != _SCHEMA:
            raise ValueError("invalid CTN predictor archive")
        return cls(loads(base64.b64decode(payload["transform"], validate=True)),
                   loads(base64.b64decode(payload["outcome"], validate=True)),
                   CtnStage1(**payload["recipe"]), payload["fold_digest"], score_column=payload["score_column"])


def fit_ctn_chain(fit, data, formula, recipe, options):
    """Fit fold-local transforms, ordinary Stage 2, and a deployment CTN."""
    if options.get("z_column") is not None:
        raise ValueError("CTN and external z_column are mutually exclusive score inputs")
    if not (options.get("family") == "bernoulli-marginal-slope" or
            options.get("survival_likelihood") == "marginal-slope"):
        raise ValueError("CtnStage1 requires a marginal-slope outcome")
    config = dict(options.get("config") or {})
    if "ctn_stage1" in config or "frozen_score" in config:
        raise ValueError("the CTN chain owns the score transformation configuration")
    if options.get("fisher_rao_w") is not None:
        raise ValueError("CTN chain does not support fisher_rao_w")
    columns, _ = _table_column_views(data)
    if _SCORE in columns:
        raise ValueError(f"reserved generated-score column: {_SCORE}")
    folds = crossfit_assignment(data, recipe)
    z = np.empty(len(folds))
    stage = {"transformation_normal": True, "weights": recipe.weights,
             "offset": recipe.offset,
             "config": {"transformation_normal_config": recipe.response_config()}}
    warm = options.get("persistent_warm_start_root")
    for fold in np.unique(folds):
        train, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        stage["persistent_warm_start_root"] = Path(warm) / f"ctn-fold-{fold}" if warm else None
        try:
            transform = fit(_slice(data, train), f"{recipe.response} ~ {recipe.covariates}", **stage)
            z[held] = _scores(transform, _slice(data, held))
        except Exception as exc:
            raise RuntimeError(f"CTN fold {fold} failed; no in-sample replacement was used") from exc
    config["frozen_score"] = True
    outcome_options = {**options, "config": config, "z_column": _SCORE}
    outcome_options["persistent_warm_start_root"] = Path(warm) / "outcome" if warm else None
    outcome = fit(_with_score(data, z), formula, **outcome_options)
    stage["persistent_warm_start_root"] = Path(warm) / "ctn-deployment" if warm else None
    transform = fit(data, f"{recipe.response} ~ {recipe.covariates}", **stage)
    return CtnMarginalSlopeModel(transform, outcome, recipe,
                                 hashlib.sha256(folds.astype("<i8").tobytes()).hexdigest())
