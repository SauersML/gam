"""Event histories: marked counting processes with smooth covariate and time
effects and an evidence-selected per-subject latent state.

The latent chain is marginalised by adaptive Gauss-Hermite filtering, and the
fit carries a certificate that its coefficients are stationary under a
refinement of the quadrature and of the time mesh. Forecasts are expectations
under the filtered state, every probability the chronological integral of a
killed process. See ``docs/event-history.md``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._binding import rust_module

MARK_KINDS = ("recurrent", "once", "terminal")


def _column(frame: Any, name: str) -> np.ndarray:
    if isinstance(frame, dict):
        if name not in frame:
            raise KeyError(f"missing column {name!r}")
        return np.asarray(frame[name])
    if hasattr(frame, "columns") and name not in list(frame.columns):
        raise KeyError(f"missing column {name!r}")
    column = frame[name]
    if hasattr(column, "to_numpy"):
        return column.to_numpy()
    return np.asarray(column)


def _column_names(frame: Any) -> list[str]:
    if isinstance(frame, dict):
        return list(frame.keys())
    return [str(c) for c in frame.columns]


def _labels(values: np.ndarray, what: str) -> list[str]:
    """Identifiers as strings, refusing two source values that are unequal
    yet spell the same string (an integer ``1`` and a string ``"1"`` name two
    subjects, and silently merging them would attach one's events to the
    other)."""
    out = [str(v) for v in values]
    first: dict[str, Any] = {}
    for raw, label in zip(values, out):
        seen = first.setdefault(label, raw)
        if seen is not raw and not _same_value(seen, raw):
            raise ValueError(
                f"{what} {seen!r} and {raw!r} are different values that spell the same identifier {label!r}"
            )
    return out


def _same_value(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:
        return False


def _is_categorical(values: np.ndarray) -> bool:
    """A covariate column is categorical when it is not numeric: strings,
    booleans, pandas categoricals. Numeric codes stay continuous; declare
    such a column categorical by giving it string labels."""
    return values.dtype.kind in ("O", "U", "S", "b")


class EventHistoryModel:
    """A fitted event-history model."""

    def __init__(self, native: Any) -> None:
        self._native = native
        self._subject_index = {sid: i for i, sid in enumerate(native.subject_ids())}

    @property
    def mark_names(self) -> list[str]:
        return list(self._native.mark_names())

    @property
    def mark_kinds(self) -> dict[str, str]:
        """Kind of every mark: ``recurrent``, ``once`` or ``terminal``."""
        return dict(zip(self._native.mark_names(), self._native.mark_kinds()))

    @property
    def covariate_names(self) -> list[str]:
        return list(self._native.covariate_names())

    @property
    def covariate_levels(self) -> dict[str, list[str]]:
        """Level labels of every categorical covariate (empty for continuous)."""
        return dict(zip(self._native.covariate_names(), self._native.covariate_levels()))

    @property
    def subject_ids(self) -> list[str]:
        return list(self._native.subject_ids())

    @property
    def rank(self) -> int:
        """Rank of the latent covariance the evidence supports. The fit grows
        it from zero: each atom is proposed by the covariance score of the
        residuals, its loadings get the Gaussian prior whose precision
        maximises the marginal likelihood, and it is kept exactly when that
        prior places the loading's posterior mode away from zero."""
        return int(self._native.rank())

    @property
    def normaliser_rounds(self) -> np.ndarray:
        """How far the held risk-set normaliser moved at each re-centring
        round, in nats. Empty when the baselines are centred on the
        stationary prior; the last entry is how far the alternation settled."""
        return np.asarray(self._native.normaliser_rounds())

    @property
    def reference_masks(self) -> int:
        """How many distinct risk sets the marks define, and so how many
        reference populations the centring ran forward."""
        return int(self._native.reference_masks())

    @property
    def reference_certificate(self) -> float | None:
        """The largest disagreement, in nats, between the reference grid the
        risk-set normaliser was taken on and the same grid with every cell
        halved: what the grid's own resolution cost. ``None`` when the
        baselines are centred on the stationary prior."""
        return self._native.reference_certificate()

    @property
    def atom_evidence(self) -> np.ndarray:
        """The evidence each accepted atom's prior bought over the rank
        before it, in nats."""
        return np.asarray(self._native.atom_evidence())

    @property
    def rank_path(self) -> list[dict[str, Any]]:
        """Every rank step the evidence judged: the covariance score's top
        eigenvalue, the standardised gain of that direction, the proposed
        log-rate and whether it sat at the mesh's resolution limit, the
        log-precision of the loading prior the evidence chose, the evidence
        in nats under the score's model and the realised log-likelihood gain
        of the fitted candidate, whether the rate was held on a plateau,
        whether the atom was accepted, and whether its model reached a
        certified optimum."""
        return [dict(step) for step in self._native.rank_path()]

    @property
    def covariance(self) -> np.ndarray:
        """``C(0)``: the posterior mean of the covariance across marks of the
        latent log-intensity deviations at one time, shape ``(marks, marks)``.
        This is the latent object the model identifies; each atom contributes
        its mode ``a aᵀ`` plus the posterior spread of its loadings."""
        return np.asarray(self._native.covariance())

    @property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of :attr:`covariance`, descending."""
        return np.asarray(self._native.eigenvalues())

    @property
    def eigenvalue_sd(self) -> np.ndarray:
        """Posterior standard deviation of each eigenvalue, from the fit's own
        posterior covariance of the loadings: what says whether a direction
        is resolved."""
        return np.asarray(self._native.eigenvalue_sd())

    @property
    def eigenvectors(self) -> np.ndarray:
        """Unit eigenvectors of :attr:`covariance` as columns."""
        return np.asarray(self._native.eigenvectors())

    @property
    def effective_rank(self) -> float:
        """The participation ratio ``(tr C)² / tr(C²)``: the continuous count
        of directions the covariance uses."""
        return float(self._native.effective_rank())

    @property
    def loadings(self) -> np.ndarray:
        """Factor coordinates of the covariance at the posterior mode, shape
        ``(marks, rank)``, in the canonical gauge: atoms ordered by rate
        (slowest first) and each column signed so its largest entry is
        positive. With distinct rates the temporal covariance identifies
        each atom up to that gauge; at equal rates only :attr:`covariance`
        is identified."""
        return np.asarray(self._native.loadings())

    @property
    def rates(self) -> np.ndarray:
        """Atom rates in the data's time unit."""
        return np.asarray(self._native.rates())

    @property
    def rate_held(self) -> np.ndarray:
        """Whether each atom's rate sits at a limit of the mesh's resolution
        (a static frailty at the slow end, the mesh's own spacing at the fast
        end), where the likelihood is flat in it, so it was held there rather
        than fitted."""
        return np.asarray(self._native.rate_held(), dtype=bool)

    @property
    def atom_log_lambdas(self) -> np.ndarray:
        """Log-precision of each atom's loading prior, chosen by the evidence
        when the atom entered."""
        return np.asarray(self._native.atom_log_lambdas())

    @property
    def log_likelihood(self) -> float:
        return float(self._native.log_likelihood())

    @property
    def reml_score(self) -> float | None:
        return self._native.reml_score()

    @property
    def quadrature(self) -> dict[str, Any]:
        """The certificate: the Gauss-Hermite order and mesh refinement of the
        fit, and the largest coefficient shift (in posterior standard
        deviations) under the next order and the next mesh."""
        return dict(self._native.quadrature())

    def coefficients(self, mark: int | str) -> np.ndarray:
        """Coefficients of one mark's population log-intensity surface: the
        latent term enters as the deviation from a population rate, so
        ``exp(η⁰)`` is the intensity averaged over the latent state."""
        if isinstance(mark, str):
            mark = self.mark_names.index(mark)
        return np.asarray(self._native.coefficients(int(mark)))

    def temporal_covariance(self, lag: float) -> np.ndarray:
        """``C(Δ) = Σ_k E[a_k a_kᵀ] exp(-r_k Δ)`` across a lag of ``lag`` time
        units."""
        return np.asarray(self._native.temporal_covariance(float(lag)))

    def latent_state(self, subject: int | str) -> dict[str, np.ndarray]:
        """The smoothed latent state of one subject: at every node of its
        history, the posterior mean of the atoms (``mean``, nodes × atoms)
        and their posterior covariance (``covariance``, nodes × atoms ×
        atoms) given the whole history, with the node ``time``. A fitted path
        is an uncertain object; its covariance is what propagates that."""
        out = self._native.latent_state(self._subject(subject))
        return {
            "time": np.asarray(out["time"]),
            "mean": np.asarray(out["mean"]),
            "covariance": np.asarray(out["covariance"]),
        }

    def _subject(self, subject: int | str) -> int:
        if isinstance(subject, str):
            return self._subject_index[subject]
        return int(subject)

    def _covariate_values(self, covariates: Mapping[str, Any] | Sequence[Any]) -> list[float]:
        names = self.covariate_names
        levels = self.covariate_levels
        if isinstance(covariates, Mapping):
            missing = [c for c in names if c not in covariates]
            if missing:
                raise KeyError(f"missing covariates {missing}")
            raw = [covariates[c] for c in names]
        else:
            raw = list(covariates)
            if len(raw) != len(names):
                raise ValueError(f"expected {len(names)} covariate values, got {len(raw)}")
        values = []
        for name, value in zip(names, raw):
            if levels[name]:
                label = str(value)
                if label not in levels[name]:
                    raise ValueError(
                        f"unknown level {label!r} for categorical covariate {name!r}; levels: {levels[name]}"
                    )
                values.append(float(levels[name].index(label)))
            else:
                values.append(float(value))
        return values

    def _future(
        self,
        future: Mapping[str, Any] | Sequence[Any] | Sequence[tuple[float, Any]] | None,
        start: float,
    ) -> list[tuple[float, list[float]]]:
        """A covariate path over a window opening at ``start``: ``None`` (the
        current row holds), one covariate record (constant over the window),
        or a sequence of ``(start, record)`` pairs."""
        if future is None:
            return []
        if isinstance(future, Mapping):
            return [(float(start), self._covariate_values(future))]
        pairs = list(future)
        if pairs and all(isinstance(p, tuple) and len(p) == 2 for p in pairs):
            return [(float(t), self._covariate_values(record)) for t, record in pairs]
        return [(float(start), self._covariate_values(pairs))]

    @staticmethod
    def _forecast_dict(out: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "horizons": np.asarray(out["horizons"]),
            "survival": np.asarray(out["survival"]),
            "expected_counts": np.asarray(out["expected_counts"]),
        }

    def forecast(
        self,
        subject: int | str,
        horizons: Sequence[float],
        future: Mapping[str, Any] | Sequence[Any] | Sequence[tuple[float, Any]] | None = None,
    ) -> dict[str, Any]:
        """Forecast one subject beyond its exit: ``survival`` is the
        probability that no terminal mark has fired by each horizon and
        ``expected_counts`` (horizons × marks) is the expected count of each
        mark — its cumulative incidence when terminal, its first-occurrence
        probability when once-only. ``future`` is the covariate path over the
        window: absent, the row in force at exit holds; a record holds
        constant; ``[(start, record), ...]`` changes at the given times."""
        index = self._subject(subject)
        exit_ = float(self._native.subject_exits()[index])
        path = self._future(future, exit_)
        out = self._native.forecast(index, [float(h) for h in horizons], path)
        return self._forecast_dict(out)

    def population_forecast(
        self,
        covariates: Mapping[str, Any] | Sequence[Any] | Sequence[tuple[float, Any]],
        start: float,
        horizons: Sequence[float],
    ) -> dict[str, Any]:
        """Forecast a subject with no observed history from covariate values
        alone: the latent state starts at its stationary prior at ``start``.
        Population covariate values give the population tier; a subject's own
        score gives what the model says before its history is seen.
        ``covariates`` is one record (constant over the window) or a sequence
        of ``(start, record)`` pairs whose first start is at or before
        ``start``."""
        path = self._future(covariates, float(start))
        if not path:
            raise ValueError("population_forecast needs covariate values")
        out = self._native.population_forecast(float(start), [float(h) for h in horizons], path)
        return self._forecast_dict(out)

    def pit(self, subject: int | str) -> dict[str, Any]:
        """Predictive PIT of every spell of one subject's follow-up, in time
        order: one per event and, unless an event ended the follow-up, one
        for the censored tail after the last event. ``pit`` is
        ``1 − P(no event in the spell | history)``; ``observed`` says
        whether the spell ended with an event (an unobserved spell's PIT is
        the value its uniform is known to exceed, not a draw of it);
        ``marks`` lists the marks that fired at each spell's end (empty for
        the tail) and ``mark_probabilities`` (spells × marks) the predictive
        probability of each mark given an event then."""
        out = self._native.pit(self._subject(subject))
        names = self.mark_names
        return {
            "time": np.asarray(out["time"], dtype=float),
            "observed": np.asarray(out["observed"], dtype=bool),
            "pit": np.asarray(out["pit"], dtype=float),
            "marks": [[names[m] for m in fired] for fired in out["marks"]],
            "mark_probabilities": np.asarray(out["mark_probabilities"]),
        }

    def pit_distance(self) -> dict[str, Any]:
        """Distance of the predictive PIT distribution from the uniform law
        over the whole cohort: the largest gap between the Kaplan–Meier
        estimate of the PIT distribution — event spells as observations,
        censored tails as uniforms known to exceed their value — and the
        uniform law. Comparing the event PITs alone to the uniform law is
        wrong under censoring (they are uniform on ``[0, 1 − S(exit)]``);
        this estimate has no such floor and equals the ordinary
        Kolmogorov–Smirnov distance when nothing is censored. Returns
        ``distance`` (``None`` for no spells), ``spells`` and ``events``.
        With parameters fitted on the same data it is a summary, not a
        calibrated test."""
        return dict(self._native.pit_distance())

    def forecast_history(
        self,
        entry: float,
        exit: float,
        events: Sequence[tuple[float, Any]],
        covariates: Mapping[str, Any] | Sequence[Any] | Sequence[tuple[float, Any]],
        horizons: Sequence[float],
        *,
        cutoff: float | None = None,
        future: Mapping[str, Any] | Sequence[Any] | Sequence[tuple[float, Any]] | None = None,
    ) -> dict[str, Any]:
        """Forecast a history that is not a training subject's, from its own
        records: ``entry`` and ``exit``, ``events`` as ``(time, mark)`` pairs
        (mark names; an event at or before the entry is prior history), and
        ``covariates`` as one record holding from the entry or ``(start,
        record)`` pairs whose first start is at or before the entry. With
        ``cutoff``, the history is cut to what was known then — events at or
        before it, covariate segments begun before it, follow-up ending
        there — so the forecast is the one that could have been made at the
        cutoff, and records after it cannot change it; ``horizons`` and the
        forecast window then open at the cutoff. ``future`` is the covariate
        path over the window as in :meth:`forecast`."""
        mark_index = {name: i for i, name in enumerate(self.mark_names)}
        event_time = []
        event_mark = []
        for time, mark in events:
            label = str(mark)
            if label not in mark_index:
                raise ValueError(f"unknown mark {label!r}; marks: {self.mark_names}")
            event_time.append(float(time))
            event_mark.append(mark_index[label])
        segments = self._future(covariates, float(entry))
        if not segments:
            raise ValueError("forecast_history needs the history's covariate values")
        table = np.ascontiguousarray(
            np.asarray([values for _, values in segments], dtype=float).reshape(len(segments), -1)
        )
        starts = [start for start, _ in segments]
        window_start = float(exit) if cutoff is None else float(cutoff)
        path = self._future(future, window_start)
        out = self._native.forecast_history(
            float(entry),
            float(exit),
            event_time,
            event_mark,
            starts,
            table,
            None if cutoff is None else float(cutoff),
            [float(h) for h in horizons],
            path,
        )
        return self._forecast_dict(out)


def fit_event_history(
    subjects: Any,
    events: Any,
    covariates: Any,
    formula: str | Sequence[str],
    *,
    marks: Mapping[str, str] | Sequence[str] | None = None,
    id_column: str = "id",
    reference_profiles: Sequence[int] | None = None,
    reference_stratum: Sequence[Any] | None = None,
) -> EventHistoryModel:
    """Fit an event-history model.

    ``subjects`` has columns ``id, entry, exit``; ``events`` has ``id, time,
    mark``; ``covariates`` has ``id, start`` and the covariate columns, one row
    per covariate segment (a subject's covariates are constant from ``start``
    until its next segment). String, boolean or categorical covariate columns
    are categorical covariates; numeric columns are continuous. ``formula``
    is the right-hand side of a gam formula over the covariate columns and
    ``time``, e.g. ``"x + s(time)"``, or ``"1"`` for an intercept alone.

    ``marks`` declares the mark vocabulary and each mark's kind — a mapping
    ``{"relapse": "recurrent", "death": "terminal"}``, or a sequence of names
    that are all recurrent. Without it the names are the distinct values of
    the events' ``mark`` column, all recurrent, which needs at least one event.
    A terminal mark ends follow-up (the subject's ``exit`` is its time), a
    once-only mark removes the subject from that mark's risk set, a recurrent
    mark can fire any number of times.

    The rank of the latent covariance is grown from zero by the evidence:
    each atom is proposed by the covariance score of the residuals, its
    loadings get the Gaussian prior whose precision maximises the marginal
    likelihood, and it enters exactly when that prior places the loading's
    posterior mode away from zero.

    ``reference_profiles`` centres the baselines on the risk sets rather than
    on the stationary prior: ``exp(baseline)`` becomes the incidence among
    those still at risk at every time, instead of the rate over the cohort as
    it started. It names one row of ``covariates`` per reference stratum — the
    covariate profile that stratum's reference population carries — and
    ``reference_stratum`` gives each subject's stratum, in the order of the
    ``subjects`` table, as labels matching those rows' positions. With one
    profile and no strata every subject shares it. The two centrings are the
    same model at rank zero, where there are no loadings to centre.
    """
    rust = rust_module()
    subject_values = _column(subjects, id_column)
    subject_ids = _labels(subject_values, "subject identifiers")
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("subject identifiers must be distinct")
    index = {sid: i for i, sid in enumerate(subject_ids)}
    entry = _column(subjects, "entry").astype(float).tolist()
    exit_ = _column(subjects, "exit").astype(float).tolist()
    mark_values = _labels(_column(events, "mark"), "mark names")
    if marks is None:
        if not mark_values:
            raise ValueError(
                "the events table has no rows, so the mark vocabulary must be given: marks={name: kind}"
            )
        mark_names = sorted(set(mark_values))
        mark_kinds = ["recurrent"] * len(mark_names)
    elif isinstance(marks, Mapping):
        mark_names = [str(k) for k in marks.keys()]
        mark_kinds = [str(v).lower() for v in marks.values()]
    else:
        mark_names = [str(m) for m in marks]
        mark_kinds = ["recurrent"] * len(mark_names)
    for kind in mark_kinds:
        if kind not in MARK_KINDS:
            raise ValueError(f"unknown mark kind {kind!r}; expected one of {MARK_KINDS}")
    mark_index = {name: i for i, name in enumerate(mark_names)}
    unknown = sorted(set(mark_values) - set(mark_names))
    if unknown:
        raise ValueError(f"events carry marks {unknown} that are not in the mark vocabulary {mark_names}")
    event_subject = []
    for v in _column(events, id_column):
        label = str(v)
        if label not in index:
            raise ValueError(f"event subject {label!r} is not in the subjects table")
        event_subject.append(index[label])
    event_time = _column(events, "time").astype(float).tolist()
    event_mark = [mark_index[m] for m in mark_values]
    covariate_names = [
        c for c in _column_names(covariates) if c not in (id_column, "start")
    ]
    columns = []
    covariate_levels: list[list[str]] = []
    for name in covariate_names:
        values = _column(covariates, name)
        if _is_categorical(values):
            labels = [str(v) for v in values]
            levels = sorted(set(labels))
            covariate_levels.append(levels)
            code = {level: float(i) for i, level in enumerate(levels)}
            columns.append(np.asarray([code[v] for v in labels], dtype=float))
        else:
            covariate_levels.append([])
            columns.append(values.astype(float))
    n_segments = len(_column(covariates, "start"))
    table = (
        np.column_stack(columns)
        if columns
        else np.zeros((n_segments, 0), dtype=float)
    )
    segment_subject = []
    for v in _column(covariates, id_column):
        label = str(v)
        if label not in index:
            raise ValueError(f"covariate subject {label!r} is not in the subjects table")
        segment_subject.append(index[label])
    segment_start = _column(covariates, "start").astype(float).tolist()
    segment_row = list(range(len(segment_subject)))
    # The reference population: one covariate row per stratum, and each
    # subject's stratum. Without profiles the baselines keep the stationary
    # prior's centring.
    reference_rows = [int(r) for r in (reference_profiles or [])]
    if reference_stratum is not None and not reference_rows:
        raise ValueError("reference_stratum needs reference_profiles to assign subjects to")
    if not reference_rows:
        subject_stratum: list[int] = []
    elif reference_stratum is None:
        if len(reference_rows) != 1:
            raise ValueError(
                f"{len(reference_rows)} reference profiles need reference_stratum to say which subject is in which"
            )
        subject_stratum = [0] * len(subject_ids)
    else:
        labels = [str(v) for v in reference_stratum]
        if len(labels) != len(subject_ids):
            raise ValueError(
                f"{len(labels)} subject strata for {len(subject_ids)} subjects"
            )
        levels = sorted(set(labels))
        if len(levels) != len(reference_rows):
            raise ValueError(
                f"reference_stratum has {len(levels)} distinct values but {len(reference_rows)} profiles were given"
            )
        subject_stratum = [levels.index(v) for v in labels]
    for row in reference_rows:
        if not 0 <= row < table.shape[0]:
            raise ValueError(
                f"reference profile row {row} is outside the {table.shape[0]} covariate rows"
            )
    native = rust.fit_event_history(
        mark_names,
        mark_kinds,
        covariate_names,
        covariate_levels,
        np.ascontiguousarray(table, dtype=np.float64),
        subject_ids,
        entry,
        exit_,
        event_subject,
        event_time,
        event_mark,
        segment_subject,
        segment_start,
        segment_row,
        [str(formula)] if isinstance(formula, str) else [str(f) for f in formula],
        reference_rows,
        subject_stratum,
    )
    return EventHistoryModel(native)
