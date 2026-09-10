//! Event-history data: subjects, their marked events, their covariate
//! segments, and the node expansion that turns continuous follow-up into the
//! quadrature/event nodes the marginal likelihood is evaluated on.
//!
//! The compensator `∫ λ_d(t) R_d(t) dt` of every mark is integrated by
//! Gauss-Legendre quadrature on every cell of the mesh: the follow-up is cut
//! at entry, exit, covariate changes and events, and every piece is split
//! into `2^r` equal cells at mesh refinement `r`. Every quadrature node
//! carries an exposure weight per mark (the cell weight times the mark's
//! risk indicator `R_d`); every event carries a unit count and no exposure.
//! The latent state of the subject is represented at every node, so the
//! latent path is sampled at the resolution of the mesh: the temporal error
//! of the marginal is second order in the node spacing (the
//! Ornstein–Uhlenbeck covariance has a kink on the diagonal that no
//! quadrature order removes), which is why the mesh, not the order, is what
//! the fit refines.
//!
//! Marks have kinds. A recurrent mark can fire any number of times; a mark
//! that fires once leaves the subject's risk set for that mark and nothing
//! else; a terminal mark ends follow-up. The kinds are part of the model:
//! the training likelihood, the forecasts and the calibration diagnostics
//! all read them from the cohort.
//!
//! Covariates are predictable: at an event node the row in force is the one
//! from the left limit `X(t⁻)`, so a covariate that changes at the instant of
//! an event never explains the event that changed it.

use ndarray::Array2;
use thiserror::Error;

/// Errors raised by the event-history family.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum EventHistoryError {
    #[error("{reason}")]
    InvalidInput { reason: String },
    #[error("{reason}")]
    NumericalFailure { reason: String },
    #[error("{reason}")]
    Fit { reason: String },
}

impl From<EventHistoryError> for String {
    fn from(error: EventHistoryError) -> Self {
        error.to_string()
    }
}

fn invalid(reason: impl Into<String>) -> EventHistoryError {
    EventHistoryError::InvalidInput {
        reason: reason.into(),
    }
}

/// How a mark behaves when it fires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkKind {
    /// Can fire any number of times during follow-up.
    Recurrent,
    /// Fires at most once; afterwards the subject is no longer at risk for
    /// this mark, and follow-up continues.
    Once,
    /// Fires at most once and ends follow-up (an absorbing state).
    Terminal,
}

impl MarkKind {
    pub fn parse(name: &str) -> Result<Self, EventHistoryError> {
        match name.trim().to_ascii_lowercase().as_str() {
            "recurrent" => Ok(Self::Recurrent),
            "once" => Ok(Self::Once),
            "terminal" => Ok(Self::Terminal),
            other => Err(invalid(format!(
                "unknown mark kind {other:?}; expected recurrent, once or terminal"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Recurrent => "recurrent",
            Self::Once => "once",
            Self::Terminal => "terminal",
        }
    }
}

/// One observed event: its time and its mark index.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub time: f64,
    pub mark: usize,
}

/// A piece of the follow-up during which the subject's covariate row is
/// constant: from `start` until the next segment's `start` (or exit).
#[derive(Clone, Debug, PartialEq)]
pub struct CovariateSegment {
    pub start: f64,
    /// Row of [`EventHistoryCohort::covariates`] in force on this segment.
    pub row: usize,
}

/// One subject's observed history.
#[derive(Clone, Debug, PartialEq)]
pub struct SubjectHistory {
    pub id: String,
    /// Start of observation (delayed entry allowed).
    pub entry: f64,
    /// End of observation. A subject with a terminal event exits at that
    /// event's time.
    pub exit: f64,
    /// Events, in any order (sorted during validation). An event at or
    /// before `entry` is prior history: it is not compensated, but it opens
    /// the window with the risk sets it leaves behind, so a first-occurrence
    /// mark that has already fired stays out of the likelihood and out of
    /// every forecast. An event after `exit` is rejected.
    pub events: Vec<Event>,
    /// Covariate segments; the first must start at or before `entry`, every
    /// later one strictly inside `(entry, exit)`, with distinct starts.
    pub segments: Vec<CovariateSegment>,
}

impl SubjectHistory {
    /// The terminal event that ended this subject's follow-up, if any.
    pub fn terminal_event(&self, kinds: &[MarkKind]) -> Option<&Event> {
        self.events
            .iter()
            .find(|event| kinds.get(event.mark) == Some(&MarkKind::Terminal))
    }

    /// Covariate row in force at `t`: the segment with the latest start `≤ t`
    /// (`< t` for the left limit, which an event node uses).
    pub fn covariate_row_at(&self, t: f64, left_limit: bool) -> usize {
        self.segments
            .iter()
            .rev()
            .find(|s| {
                if left_limit {
                    s.start < t
                } else {
                    s.start <= t
                }
            })
            .map(|s| s.row)
            .unwrap_or(self.segments[0].row)
    }

    /// Whether the subject is still at risk for mark `d` at time `t` (after
    /// its events before `t`): a once-only mark that has fired leaves the risk
    /// set.
    pub fn at_risk(&self, d: usize, t: f64, kinds: &[MarkKind]) -> bool {
        match kinds[d] {
            MarkKind::Recurrent => true,
            MarkKind::Once | MarkKind::Terminal => {
                !self.events.iter().any(|e| e.mark == d && e.time < t)
            }
        }
    }

    /// The history as it was known at `cutoff`: the events at or before the
    /// cutoff, the covariate segments that had started before it, and
    /// follow-up ending there. Nothing recorded later enters — not a later
    /// event, not a later covariate change — so a forecast made from the
    /// prefix is a forecast made at the cutoff, and appending records after
    /// the cutoff cannot change it.
    ///
    /// A cutoff after the subject's exit is refused: the window between the
    /// exit and the cutoff was not observed, and extending the follow-up to
    /// it would fabricate event-free exposure. A subject whose follow-up a
    /// terminal event ended at or before the cutoff is returned whole: its
    /// history was complete at the cutoff.
    pub fn prefix(
        &self,
        cutoff: f64,
        kinds: &[MarkKind],
    ) -> Result<SubjectHistory, EventHistoryError> {
        if !cutoff.is_finite() || cutoff <= self.entry {
            return Err(invalid(format!(
                "subject {:?}: the cutoff {cutoff} must be finite and after the entry {}",
                self.id, self.entry
            )));
        }
        if self
            .terminal_event(kinds)
            .is_some_and(|event| event.time <= cutoff)
        {
            return Ok(self.clone());
        }
        if cutoff > self.exit {
            return Err(invalid(format!(
                "subject {:?}: the cutoff {cutoff} is after the exit {}; the follow-up between them was not observed",
                self.id, self.exit
            )));
        }
        Ok(SubjectHistory {
            id: self.id.clone(),
            entry: self.entry,
            exit: cutoff,
            events: self
                .events
                .iter()
                .filter(|event| event.time <= cutoff)
                .cloned()
                .collect(),
            segments: self
                .segments
                .iter()
                .filter(|segment| segment.start < cutoff)
                .cloned()
                .collect(),
        })
    }
}

/// A cohort of subjects with a shared covariate table and mark vocabulary.
#[derive(Clone, Debug)]
pub struct EventHistoryCohort {
    pub mark_names: Vec<String>,
    /// The kind of every mark (one per mark).
    pub mark_kinds: Vec<MarkKind>,
    /// Names of the covariate table's columns, visible to the formula.
    pub covariate_names: Vec<String>,
    /// Per column, the level labels of a categorical covariate whose codes
    /// in the table are `0, 1, …` (empty for a continuous column).
    pub covariate_levels: Vec<Vec<String>>,
    /// Covariate table; rows are referenced by [`CovariateSegment::row`].
    pub covariates: Array2<f64>,
    pub subjects: Vec<SubjectHistory>,
}

impl EventHistoryCohort {
    pub fn marks(&self) -> usize {
        self.mark_names.len()
    }

    /// One flag per mark: whether it is terminal.
    pub fn terminal_marks(&self) -> Vec<bool> {
        self.mark_kinds
            .iter()
            .map(|k| *k == MarkKind::Terminal)
            .collect()
    }

    /// Validate every subject and sort its events and segments in time.
    pub fn validate(&mut self) -> Result<(), EventHistoryError> {
        if self.mark_names.is_empty() {
            return Err(invalid("an event-history cohort needs at least one mark"));
        }
        {
            let mut names = self.mark_names.clone();
            names.sort();
            if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
                return Err(invalid(format!("duplicate mark name {:?}", w[0])));
            }
        }
        if self.mark_kinds.len() != self.marks() {
            return Err(invalid(format!(
                "{} mark kinds for {} marks",
                self.mark_kinds.len(),
                self.marks()
            )));
        }
        if self.subjects.is_empty() {
            return Err(invalid(
                "an event-history cohort needs at least one subject",
            ));
        }
        if self.covariates.iter().any(|v| !v.is_finite()) {
            return Err(invalid("covariate table contains a non-finite value"));
        }
        if self.covariate_names.len() != self.covariates.ncols() {
            return Err(invalid(format!(
                "{} covariate names for a table with {} columns",
                self.covariate_names.len(),
                self.covariates.ncols()
            )));
        }
        {
            let mut names = self.covariate_names.clone();
            names.sort();
            if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
                return Err(invalid(format!("duplicate covariate name {:?}", w[0])));
            }
        }
        if self.covariate_levels.len() != self.covariates.ncols() {
            return Err(invalid(format!(
                "{} covariate level lists for a table with {} columns",
                self.covariate_levels.len(),
                self.covariates.ncols()
            )));
        }
        for (j, levels) in self.covariate_levels.iter().enumerate() {
            if levels.is_empty() {
                continue;
            }
            for (i, value) in self.covariates.column(j).iter().enumerate() {
                if value.fract() != 0.0 || *value < 0.0 || *value >= levels.len() as f64 {
                    return Err(invalid(format!(
                        "categorical covariate {:?} has code {value} at row {i}, but its {} levels are coded 0..{}",
                        self.covariate_names[j],
                        levels.len(),
                        levels.len()
                    )));
                }
            }
        }
        {
            let mut ids: Vec<&str> = self.subjects.iter().map(|s| s.id.as_str()).collect();
            ids.sort_unstable();
            if let Some(w) = ids.windows(2).find(|w| w[0] == w[1]) {
                return Err(invalid(format!("duplicate subject identifier {:?}", w[0])));
            }
        }
        let marks = self.marks();
        let rows = self.covariates.nrows();
        let kinds = &self.mark_kinds;
        for subject in &mut self.subjects {
            if subject.id.is_empty() {
                return Err(invalid("subject identifier must be non-empty"));
            }
            if !subject.entry.is_finite()
                || !subject.exit.is_finite()
                || subject.exit <= subject.entry
            {
                return Err(invalid(format!(
                    "subject {:?} needs finite entry < exit, got {} and {}",
                    subject.id, subject.entry, subject.exit
                )));
            }
            if subject.segments.is_empty() {
                return Err(invalid(format!(
                    "subject {:?} needs at least one covariate segment",
                    subject.id
                )));
            }
            for segment in &subject.segments {
                if !segment.start.is_finite() {
                    return Err(invalid(format!(
                        "subject {:?} has a non-finite segment start",
                        subject.id
                    )));
                }
                if segment.row >= rows {
                    return Err(invalid(format!(
                        "subject {:?} references covariate row {} of a {}-row table",
                        subject.id, segment.row, rows
                    )));
                }
            }
            subject.segments.sort_by(|a, b| a.start.total_cmp(&b.start));
            if subject.segments[0].start > subject.entry {
                return Err(invalid(format!(
                    "subject {:?}: the first covariate segment must start at or before entry",
                    subject.id
                )));
            }
            if let Some(w) = subject
                .segments
                .windows(2)
                .find(|w| w[0].start == w[1].start)
            {
                return Err(invalid(format!(
                    "subject {:?} has two covariate segments starting at {}",
                    subject.id, w[0].start
                )));
            }
            if let Some(late) = subject.segments[1..]
                .iter()
                .find(|s| s.start <= subject.entry || s.start >= subject.exit)
            {
                return Err(invalid(format!(
                    "subject {:?}: covariate segment starting at {} is outside (entry, exit) = ({}, {})",
                    subject.id, late.start, subject.entry, subject.exit
                )));
            }
            subject.events.sort_by(|a, b| a.time.total_cmp(&b.time));
            for event in &subject.events {
                // An event at or before entry is prior history: it is not
                // compensated (its own period is not being modelled) but it
                // shapes the risk sets the window opens with, which is what
                // lets a first-occurrence mark know it has already fired.
                // Only an event after exit is outside the record.
                if !event.time.is_finite() || event.time > subject.exit {
                    return Err(invalid(format!(
                        "subject {:?} has an event at {}, after its exit {}",
                        subject.id, event.time, subject.exit
                    )));
                }
                if event.mark >= marks {
                    return Err(invalid(format!(
                        "subject {:?} has an event with mark {} but only {} marks are declared",
                        subject.id, event.mark, marks
                    )));
                }
            }
            for d in 0..marks {
                if kinds[d] == MarkKind::Recurrent {
                    continue;
                }
                let count = subject.events.iter().filter(|e| e.mark == d).count();
                if count > 1 {
                    return Err(invalid(format!(
                        "subject {:?} has {count} events of the {} mark {:?}, which can fire at most once",
                        subject.id,
                        kinds[d].name(),
                        d
                    )));
                }
            }
            if let Some(event) = subject
                .events
                .iter()
                .find(|e| kinds[e.mark] == MarkKind::Terminal && e.time <= subject.entry)
            {
                return Err(invalid(format!(
                    "subject {:?} has a terminal event at {}, at or before its entry {}; it has no follow-up to model",
                    subject.id, event.time, subject.entry
                )));
            }
            let terminal_events = subject
                .events
                .iter()
                .filter(|e| kinds[e.mark] == MarkKind::Terminal)
                .count();
            if terminal_events > 1 {
                return Err(invalid(format!(
                    "subject {:?} has {terminal_events} terminal events; a terminal event ends follow-up",
                    subject.id
                )));
            }
            if let Some(event) = subject.terminal_event(kinds)
                && event.time != subject.exit
            {
                return Err(invalid(format!(
                    "subject {:?}: terminal event at {} must end follow-up, but exit is {}",
                    subject.id, event.time, subject.exit
                )));
            }
        }
        Ok(())
    }

    /// Mean follow-up length across subjects: the cohort's own time scale.
    ///
    /// Latent rates are parameterised as `log(rate · time_scale)`, so a rate
    /// of one is "memory comparable to a follow-up", whatever the time unit.
    pub fn time_scale(&self) -> f64 {
        let mut mean = 0.0;
        for (i, subject) in self.subjects.iter().enumerate() {
            mean += (subject.exit - subject.entry - mean) / (i + 1) as f64;
        }
        mean
    }

    /// The mesh refinement past which the model resolves time more finely
    /// than the cohort itself does: the level at which every cell of the
    /// widest follow-up interval is narrower than the narrowest interval any
    /// subject's breakpoints already distinguish. A fit whose coefficients
    /// still move under refinement there is a numerical failure, not a
    /// coarse mesh, so the refinement ladder stops and says so.
    pub fn mesh_refinement_ceiling(&self) -> usize {
        let mut widest = 0.0_f64;
        let mut narrowest = f64::INFINITY;
        for subject in &self.subjects {
            for (left, right) in mesh_cells(subject, true, 0) {
                let width = right - left;
                if width > 0.0 {
                    widest = widest.max(width);
                    narrowest = narrowest.min(width);
                }
            }
        }
        if !(widest.is_finite() && narrowest.is_finite() && narrowest > 0.0 && widest > narrowest) {
            return 1;
        }
        (widest / narrowest).log2().ceil().max(1.0).min(60.0) as usize
    }

    /// Encode the label of a categorical covariate into its code.
    pub fn encode_level(&self, column: usize, label: &str) -> Result<f64, EventHistoryError> {
        let levels = &self.covariate_levels[column];
        levels
            .iter()
            .position(|l| l == label)
            .map(|i| i as f64)
            .ok_or_else(|| {
                invalid(format!(
                    "unknown level {label:?} for categorical covariate {:?}; levels: {:?}",
                    self.covariate_names[column], levels
                ))
            })
    }
}

/// The node expansion of one subject.
#[derive(Clone, Debug)]
pub struct SubjectNodes {
    /// Global row offset of this subject's first node.
    pub first_row: usize,
    /// Strictly increasing node times.
    pub times: Vec<f64>,
    /// `times[n+1] - times[n]`, length `times.len() - 1`.
    pub gaps: Vec<f64>,
    /// Quadrature weight of each node (the cell's Gauss-Legendre weight);
    /// zero on pure event nodes.
    pub weights: Vec<f64>,
    /// Exposure of each node to each mark, shape `(nodes, marks)`: the node
    /// weight times the mark's risk indicator at the node.
    pub exposures: Array2<f64>,
    /// Event counts per node and mark, shape `(nodes, marks)`.
    pub counts: Array2<f64>,
    /// Covariate row in force at each node (the left limit at an event).
    pub covariate_rows: Vec<usize>,
}

impl SubjectNodes {
    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Whether node `n` carries an event of any mark.
    pub fn is_event(&self, n: usize) -> bool {
        self.counts.row(n).iter().any(|&c| c > 0.0)
    }

    /// The exposures of node `n` to every mark.
    pub fn exposure_row(&self, n: usize) -> Vec<f64> {
        self.exposures.row(n).to_vec()
    }
}

/// Node expansion of a whole cohort plus the per-node data matrix.
#[derive(Clone, Debug)]
pub struct CohortNodes {
    pub marks: usize,
    pub subjects: Vec<SubjectNodes>,
    /// Per-node data: the covariate row in force followed by the node time
    /// in the last column, so a time smooth is a smooth of `time_column`.
    pub node_data: Array2<f64>,
    pub time_column: usize,
    pub total_nodes: usize,
    /// Mesh refinement level the nodes were expanded at.
    pub refinement: usize,
    /// The band `(r_min, r_max)` of latent rates per unit time the cohort's
    /// own breakpoints resolve, `√ε / longest follow-up` and `1 / median
    /// level-0 cell width`: a property of the data, the same at every
    /// quadrature order and mesh refinement. `None` when no subject has two
    /// distinct breakpoints.
    pub rate_band: Option<(f64, f64)>,
}

impl CohortNodes {
    /// The largest node count of any subject.
    pub fn max_subject_nodes(&self) -> usize {
        self.subjects.iter().map(|s| s.len()).max().unwrap_or(0)
    }
}

/// Gauss-Legendre order per mesh cell for a time smooth of this B-spline
/// degree: the order that integrates products of the basis exactly on a
/// cell, so the baseline part of the compensator is resolved to roundoff and
/// the mesh refinement only has to serve the latent path.
pub fn quadrature_order_for_degree(degree: usize) -> usize {
    2 * degree + 3
}

/// Mesh cells of one subject: the breakpoints (entry, exit, covariate
/// changes and, when `with_events`, event times), each interval between
/// consecutive breakpoints split into `2^refinement` equal cells.
pub(crate) fn mesh_cells(
    subject: &SubjectHistory,
    with_events: bool,
    refinement: usize,
) -> Vec<(f64, f64)> {
    let mut breakpoints = vec![subject.entry, subject.exit];
    breakpoints.extend(
        subject
            .segments
            .iter()
            .map(|s| s.start)
            .filter(|&t| t > subject.entry && t < subject.exit),
    );
    if with_events {
        breakpoints.extend(
            subject
                .events
                .iter()
                .map(|e| e.time)
                .filter(|&t| t > subject.entry && t < subject.exit),
        );
    }
    breakpoints.sort_by(|a, b| a.total_cmp(b));
    breakpoints.dedup();
    let parts = 1usize << refinement;
    let mut cells = Vec::with_capacity((breakpoints.len() - 1) * parts);
    for pair in breakpoints.windows(2) {
        let (left, right) = (pair[0], pair[1]);
        let width = (right - left) / parts as f64;
        for p in 0..parts {
            let a = left + width * p as f64;
            let b = if p + 1 == parts {
                right
            } else {
                left + width * (p + 1) as f64
            };
            cells.push((a, b));
        }
    }
    cells
}

/// Gauss-Legendre nodes and weights on one cell; `mid = left + half` (never
/// `(left + right) / 2`, which can overflow where the difference cannot).
pub(crate) fn cell_rule<'a>(
    left: f64,
    right: f64,
    nodes: &'a [f64],
    weights: &'a [f64],
) -> impl Iterator<Item = (f64, f64)> + 'a {
    let half = 0.5 * (right - left);
    let mid = left + half;
    nodes
        .iter()
        .zip(weights.iter())
        .map(move |(&x, &w)| (mid + half * x, half * w))
}

/// Expand every subject into quadrature and event nodes at mesh refinement
/// `refinement`.
pub fn expand_nodes(
    cohort: &EventHistoryCohort,
    quadrature_order: usize,
    refinement: usize,
) -> Result<CohortNodes, EventHistoryError> {
    if quadrature_order == 0 {
        return Err(invalid("quadrature order must be positive"));
    }
    if refinement >= usize::BITS as usize - 1 {
        return Err(invalid(format!(
            "mesh refinement level {refinement} is not representable"
        )));
    }
    let marks = cohort.marks();
    let kinds = &cohort.mark_kinds;
    let (gl_nodes, gl_weights) = gam_math::special::gauss_legendre(quadrature_order);
    let n_cov = cohort.covariates.ncols();
    let mut subjects = Vec::with_capacity(cohort.subjects.len());
    let mut data_rows: Vec<Vec<f64>> = Vec::new();
    let mut first_row = 0usize;
    for subject in &cohort.subjects {
        // (time, weight, mark)
        let mut raw: Vec<(f64, f64, Option<usize>)> = Vec::new();
        for (left, right) in mesh_cells(subject, true, refinement) {
            for (t, w) in cell_rule(left, right, &gl_nodes, &gl_weights) {
                raw.push((t, w, None));
            }
        }
        // Prior events carry no count node: they are history, not
        // observations of this window, and `at_risk` has already read them
        // into the risk sets.
        for event in subject.events.iter().filter(|e| e.time > subject.entry) {
            raw.push((event.time, 0.0, Some(event.mark)));
        }
        raw.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut times: Vec<f64> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        let mut counts: Vec<Vec<f64>> = Vec::new();
        for (time, weight, mark) in raw {
            if let Some(&last) = times.last()
                && time == last
            {
                let index = times.len() - 1;
                weights[index] += weight;
                if let Some(mark) = mark {
                    counts[index][mark] += 1.0;
                }
            } else {
                times.push(time);
                weights.push(weight);
                let mut row = vec![0.0; marks];
                if let Some(mark) = mark {
                    row[mark] += 1.0;
                }
                counts.push(row);
            }
        }
        let gaps: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
        if gaps.iter().any(|&g| !(g > 0.0 && g.is_finite())) {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!(
                    "subject {:?}: node times are not strictly increasing after merging",
                    subject.id
                ),
            });
        }
        let n = times.len();
        let mut covariate_rows = Vec::with_capacity(n);
        let mut exposures = Array2::<f64>::zeros((n, marks));
        let mut count_matrix = Array2::<f64>::zeros((n, marks));
        for (i, &t) in times.iter().enumerate() {
            let is_event = counts[i].iter().any(|&c| c > 0.0);
            let row = subject.covariate_row_at(t, is_event);
            covariate_rows.push(row);
            let mut data = Vec::with_capacity(n_cov + 1);
            data.extend(cohort.covariates.row(row).iter().copied());
            data.push(t);
            data_rows.push(data);
            for d in 0..marks {
                count_matrix[[i, d]] = counts[i][d];
                if weights[i] != 0.0 && subject.at_risk(d, t, kinds) {
                    exposures[[i, d]] = weights[i];
                }
            }
        }
        subjects.push(SubjectNodes {
            first_row,
            times,
            gaps,
            weights,
            exposures,
            counts: count_matrix,
            covariate_rows,
        });
        first_row += n;
    }
    let total_nodes = first_row;
    let mut node_data = Array2::<f64>::zeros((total_nodes, n_cov + 1));
    for (i, row) in data_rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            node_data[[i, j]] = v;
        }
    }
    // The rate band is read from the level-0 cells — the breakpoints the
    // data themselves distinguish — never from the quadrature nodes, whose
    // spacing is the mesh's, not the data's, and changes under refinement.
    let breakpoints: Vec<Vec<f64>> = cohort
        .subjects
        .iter()
        .map(|subject| {
            let cells = mesh_cells(subject, true, 0);
            let mut times: Vec<f64> = cells.first().map(|cell| vec![cell.0]).unwrap_or_default();
            times.extend(cells.iter().map(|cell| cell.1));
            times
        })
        .collect();
    let rate_band =
        super::covariance::resolvable_rate_band(breakpoints.iter().map(Vec::as_slice), 1.0);
    Ok(CohortNodes {
        marks,
        subjects,
        node_data,
        time_column: n_cov,
        total_nodes,
        refinement,
        rate_band,
    })
}

/// The rows the covariate/time bases are built on: for every subject its
/// entry, exit and covariate-change times plus the Gauss-Legendre nodes of
/// the event-free mesh. No event time enters, so a data-adaptive basis
/// (quantile knots, a data-driven range) is a function of the design alone
/// and the time basis spans every follow-up window to its ends.
pub fn design_rows(
    cohort: &EventHistoryCohort,
    quadrature_order: usize,
) -> Result<Array2<f64>, EventHistoryError> {
    if quadrature_order == 0 {
        return Err(invalid("quadrature order must be positive"));
    }
    let (gl_nodes, gl_weights) = gam_math::special::gauss_legendre(quadrature_order);
    let n_cov = cohort.covariates.ncols();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut push = |row: usize, t: f64| {
        let mut data = Vec::with_capacity(n_cov + 1);
        data.extend(cohort.covariates.row(row).iter().copied());
        data.push(t);
        rows.push(data);
    };
    for subject in &cohort.subjects {
        push(
            subject.covariate_row_at(subject.entry, false),
            subject.entry,
        );
        push(subject.covariate_row_at(subject.exit, false), subject.exit);
        for segment in &subject.segments[1..] {
            push(segment.row, segment.start);
        }
        for (left, right) in mesh_cells(subject, false, 0) {
            for (t, _) in cell_rule(left, right, &gl_nodes, &gl_weights) {
                push(subject.covariate_row_at(t, false), t);
            }
        }
    }
    let mut out = Array2::<f64>::zeros((rows.len(), n_cov + 1));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[[i, j]] = v;
        }
    }
    Ok(out)
}
