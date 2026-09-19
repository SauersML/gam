//! Probability-law specification and observed histories of the joint event
//! model (#2961).
//!
//! A history is a sequence of latent nodes with compensator quadrature points.
//! Each node's first point is its zero-weight anchor, carrying the node's
//! events and measurements; its other points integrate the cell ending at the
//! node with the node's pre-jump state. Events at one time form one node, as
//! an unordered multiset of marks. Entry is observed context: a once-only mark
//! that fired at or before entry opens the window outside its risk set, and no
//! event-free pre-entry exposure is invented.
//!
//! At rank zero every mark's intensity is a constant rate `r_d` and the
//! reference normaliser is identically one, so the complete-path log density of
//! a history is `sum_d y_d log r_d - E_d r_d`. The event counts `y_d` and the
//! at-risk exposures `E_d` are sufficient statistics.

use super::emission;
use crate::{EventHistoryError, MarkKind};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::ops::Range;

pub(super) fn invalid(reason: impl Into<String>) -> EventHistoryError {
    EventHistoryError::InvalidInput {
        reason: reason.into(),
    }
}

pub(super) fn numerical(reason: impl Into<String>) -> EventHistoryError {
    EventHistoryError::NumericalFailure {
        reason: reason.into(),
    }
}

/// Observation law of one measurement channel. A value outside the declared
/// support is refused; no channel is silently treated as Gaussian.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) enum MeasurementFamily {
    /// Continuous values, with a Student-t law whose degrees of freedom exceed
    /// two.
    StudentT,
    /// Cumulative probit over the ordered values `0..categories`, with
    /// `categories >= 2`. Two categories are the binary probit.
    Probit { categories: usize },
    /// Counts with mean proportional to a positive per-record exposure.
    NegativeBinomial,
}

/// One frozen function penalty over a contiguous column range of its basis:
/// `local` is the symmetric block over `columns`, and `rank` is its structural
/// rank from the builder.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct BasisPenalty {
    pub(super) columns: Range<usize>,
    pub(super) local: Array2<f64>,
    pub(super) rank: usize,
}

/// Model dimensions, frozen basis penalties, and the joint Gaussian law of
/// baseline genetic scores. Basis rows are supplied separately from responses,
/// so the same frozen GAM bases serve fitting, entry conditioning, and serving.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct JointSpecification {
    pub(super) signatures: usize,
    pub(super) marks: Vec<MarkKind>,
    pub(super) baseline_columns: usize,
    /// Columns of the population basis carrying decoder weights and
    /// measurement intercepts and slopes; column zero is the unit constant.
    pub(super) population_columns: usize,
    pub(super) drive_columns: usize,
    pub(super) entry_columns: usize,
    /// Frozen function penalties of the baseline, drive and population bases.
    /// Only the function priors read their values; the law checks their shape.
    pub(super) baseline_penalties: Vec<BasisPenalty>,
    pub(super) drive_penalties: Vec<BasisPenalty>,
    pub(super) population_penalties: Vec<BasisPenalty>,
    pub(super) measurements: Vec<MeasurementFamily>,
    pub(super) genetic_mean: Vec<f64>,
    pub(super) genetic_precision: Array2<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct MeasurementRecord {
    pub(super) node: usize,
    pub(super) channel: usize,
    /// None contributes its integrated likelihood, one.
    pub(super) value: Option<f64>,
    /// The positive exposure multiplying a count channel's mean. Only count
    /// channels carry one.
    pub(super) exposure: Option<f64>,
    /// A simultaneous measurement can explicitly observe the post-event state.
    pub(super) after_event: bool,
}

/// One compensator quadrature point: the node whose state it reads, the time
/// at which its design rows and reference moments are evaluated, and its
/// weight. The weight is zero exactly for the node's anchor, which sits at the
/// node time; every other point lies in its node's cell.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct CompensatorPoint {
    pub(super) node: usize,
    pub(super) time: f64,
    pub(super) weight: f64,
}

/// One observation window as latent nodes and compensator points. Visits are
/// conditioned-on observation times; informative visits must also be
/// represented as event marks. Once-only prevalence enters the entry design.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct JointHistory {
    pub(super) times: Vec<f64>,
    /// Compensator quadrature, sorted by node. Each node's first point is its
    /// zero-weight anchor at the node time; its other points integrate the cell
    /// `(t_{n-1}, t_n]` in time order with the node's pre-jump state held over
    /// it, so an event's latent coordinate also carries its cell's exposure.
    /// The entry node has only its anchor.
    pub(super) points: Vec<CompensatorPoint>,
    /// An upper bound on every cell weight's relative error against its exact
    /// share of the cell width, declared by the encoder and including the
    /// weight's own roundings: zero when each cell weight is its node-time
    /// difference, and a Gauss-Legendre rule's certified weight error plus
    /// those roundings otherwise.
    pub(super) weight_error: f64,
    /// The marks fired at each node, as an unordered multiset. Tied events use
    /// the pre-tie state for their intensities, and their jumps apply together
    /// after the node.
    pub(super) events: Vec<Vec<usize>>,
    pub(super) initially_at_risk: Vec<bool>,
    /// One row per point: the baseline basis at the point's time and context.
    pub(super) baseline_design: Array2<f64>,
    /// One row per point: the population basis at the point's time and context.
    pub(super) population_design: Array2<f64>,
    /// One row per gap, the predictable drive held over that gap.
    pub(super) drive_design: Array2<f64>,
    pub(super) entry_design: Vec<f64>,
    pub(super) genetics: Vec<Option<f64>>,
    pub(super) measurements: Vec<MeasurementRecord>,
}

impl JointSpecification {
    /// The rank-zero specification of these marks: constant rates, with no
    /// signatures, channels or genetic scores.
    pub(super) fn new(marks: Vec<MarkKind>) -> Result<Self, EventHistoryError> {
        if marks.is_empty() {
            return Err(invalid("the joint event model needs at least one mark"));
        }
        Ok(Self {
            signatures: 0,
            marks,
            baseline_columns: 1,
            population_columns: 1,
            drive_columns: 0,
            entry_columns: 0,
            baseline_penalties: vec![],
            drive_penalties: vec![],
            population_penalties: vec![],
            measurements: vec![],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
    }
}

impl JointHistory {
    /// Event counts and at-risk exposures by mark: the sufficient statistics of
    /// the rank-zero complete-path density.
    pub(super) fn rate_statistics(&self, marks: &[MarkKind]) -> (Vec<u64>, Vec<f64>) {
        let mut risk = self.initially_at_risk.clone();
        let mut counts = vec![0u64; marks.len()];
        let mut exposure = vec![0.0; marks.len()];
        let mut points = self.points.iter().peekable();
        for (node, fired) in self.events.iter().enumerate() {
            let mut weight = 0.0;
            while let Some(point) = points.next_if(|point| point.node == node) {
                weight += point.weight;
            }
            for d in 0..marks.len() {
                if risk[d] {
                    exposure[d] += weight;
                }
            }
            for &d in fired {
                counts[d] += 1;
                if marks[d] == MarkKind::Once {
                    risk[d] = false;
                }
            }
        }
        (counts, exposure)
    }

    /// The risk set the window leaves open at its exit, or `None` when a
    /// terminal event ended it.
    pub(super) fn open_risk_set(&self, marks: &[MarkKind]) -> Option<Vec<bool>> {
        let mut risk = self.initially_at_risk.clone();
        for &d in self.events.iter().flatten() {
            match marks[d] {
                MarkKind::Recurrent => (),
                MarkKind::Once => risk[d] = false,
                MarkKind::Terminal => return None,
            }
        }
        Some(risk)
    }
}

/// A validated specification: every history is checked against it before any
/// estimation reads it.
pub(super) struct JointLikelihood {
    pub(super) spec: JointSpecification,
}

impl JointLikelihood {
    pub(super) fn new(spec: JointSpecification) -> Result<Self, EventHistoryError> {
        let g = spec.genetic_mean.len();
        if spec.marks.is_empty()
            || spec.baseline_columns == 0
            || spec.population_columns == 0
            || (spec.signatures > 0 && spec.drive_columns == 0)
        {
            return Err(invalid(
                "joint model needs marks, a baseline basis, a population basis, and a state drive basis",
            ));
        }
        if spec.measurements.iter().any(|m| {
            matches!(m,
            MeasurementFamily::Probit { categories } if *categories < 2)
        }) {
            return Err(invalid("probit channels need at least two categories"));
        }
        if spec.genetic_precision.dim() != (g, g)
            || spec
                .genetic_precision
                .iter()
                .chain(&spec.genetic_mean)
                .any(|v| !v.is_finite())
        {
            return Err(invalid(
                "genetic mean and precision must have matching finite dimensions",
            ));
        }
        let mut lower = Array2::<f64>::zeros((g, g));
        for i in 0..g {
            for j in 0..=i {
                if spec.genetic_precision[[i, j]] != spec.genetic_precision[[j, i]] {
                    return Err(invalid("genetic precision must be symmetric"));
                }
                let value = spec.genetic_precision[[i, j]]
                    - (0..j).map(|q| lower[[i, q]] * lower[[j, q]]).sum::<f64>();
                if i == j {
                    if !value.is_finite() || value <= 0.0 {
                        return Err(invalid("genetic precision must be positive definite"));
                    }
                    lower[[i, j]] = value.sqrt();
                } else {
                    lower[[i, j]] = value / lower[[j, j]];
                }
            }
        }
        for (penalties, width) in [
            (&spec.baseline_penalties, spec.baseline_columns),
            (&spec.drive_penalties, spec.drive_columns),
            (&spec.population_penalties, spec.population_columns),
        ] {
            if penalties.iter().any(|penalty| {
                let size = penalty.columns.len();
                penalty.columns.end > width
                    || penalty.local.dim() != (size, size)
                    || penalty.rank > size
                    || penalty.local.iter().any(|v| !v.is_finite())
                    || (0..size).any(|i| (0..i).any(|j| penalty.local[[i, j]] != penalty.local[[j, i]]))
            }) {
                return Err(invalid(
                    "function penalties must be finite symmetric blocks within their basis columns",
                ));
            }
        }
        Ok(Self { spec })
    }

    pub(super) fn validate_history(&self, h: &JointHistory) -> Result<(), EventHistoryError> {
        let n = h.times.len();
        if n < 2
            || h.times.iter().any(|x| !x.is_finite())
            || h.times.windows(2).any(|t| t[0] >= t[1])
        {
            return Err(invalid(
                "joint history needs strictly increasing finite times including entry and exit",
            ));
        }
        let rows = h.points.len();
        if rows < n
            || h.events.len() != n
            || h.initially_at_risk.len() != self.spec.marks.len()
            || h.baseline_design.dim() != (rows, self.spec.baseline_columns)
            || h.population_design.dim() != (rows, self.spec.population_columns)
            || h.drive_design.dim() != (n - 1, self.spec.drive_columns)
            || h.entry_design.len() != self.spec.entry_columns
            || h.genetics.len() != self.spec.genetic_mean.len()
        {
            return Err(invalid("joint history dimensions do not match the model"));
        }
        if h.baseline_design
            .iter()
            .chain(h.population_design.iter())
            .chain(h.drive_design.iter())
            .chain(&h.entry_design)
            .chain(h.genetics.iter().flatten())
            .any(|x| !x.is_finite())
            || h.points.iter().any(|point| !point.weight.is_finite() || point.weight < 0.0)
        {
            return Err(invalid(
                "joint designs, observed genetics, and exposures must be finite; exposures must be nonnegative",
            ));
        }
        if h.population_design.column(0).iter().any(|&v| v != 1.0) {
            return Err(invalid(
                "the population basis must carry the unit constant in column zero",
            ));
        }
        if !(0.0..1.0).contains(&h.weight_error) {
            return Err(invalid("a declared relative weight error lies in [0, 1)"));
        }
        let mut next = 0;
        let mut previous = f64::NEG_INFINITY;
        // Each cell's weight sum, the magnitudes of its partial sums, and its
        // total weight.
        let mut cells = vec![(0.0_f64, 0.0_f64, 0.0_f64); n];
        for (index, point) in h.points.iter().enumerate() {
            let anchor = point.node == next;
            if (anchor && point.weight != 0.0)
                || (!anchor && (index == 0 || point.node + 1 != next || point.weight <= 0.0))
            {
                return Err(invalid(
                    "compensator points must follow their nodes, each opening with its zero-weight anchor",
                ));
            }
            if anchor {
                if point.time != h.times[point.node] {
                    return Err(invalid("an anchor point sits at its node time"));
                }
                next += 1;
            } else {
                // Cells do not overlap, so time order within each cell is time
                // order over all cell points. The entry node has no cell.
                if point.node == 0
                    || point.time <= h.times[point.node - 1]
                    || point.time > h.times[point.node]
                    || point.time < previous
                {
                    return Err(invalid(
                        "cell points lie in their node's cell (t_{n-1}, t_n], in time order; entry has no cell",
                    ));
                }
                previous = point.time;
                let (sum, partial, weights) = &mut cells[point.node];
                *sum += point.weight;
                *partial += sum.abs();
                *weights += point.weight;
            }
        }
        if next != n {
            return Err(invalid("every node needs its anchor point"));
        }
        if !h.events[0].is_empty() {
            return Err(invalid("entry carries no events"));
        }
        // A cell's exact weights integrate its width exactly. The computed sum
        // differs from the computed width by the declared weight error (relative
        // to the exact weights, which the computed ones bound by 1 - error) and,
        // to first order, by the running error bound of the sum and the width's
        // one rounded subtraction (Higham, ch. 3).
        let error = h.weight_error / (1.0 - h.weight_error);
        for node in 1..n {
            let (sum, partial, weights) = cells[node];
            let width = h.times[node] - h.times[node - 1];
            let rounding = f64::EPSILON * (partial + width) + error * weights;
            if !((sum - width).abs() <= rounding) {
                return Err(invalid(
                    "each cell's quadrature weights must integrate its width",
                ));
            }
        }
        let mut risk = h.initially_at_risk.clone();
        for (d, kind) in self.spec.marks.iter().enumerate() {
            if *kind != MarkKind::Once && !risk[d] {
                return Err(invalid(
                    "entry must be alive and recurrent marks remain at risk",
                ));
            }
        }
        for (node, fired) in h.events.iter().enumerate() {
            if fired.is_empty() {
                continue;
            }
            if fired.iter().any(|&d| d >= risk.len() || !risk[d]) {
                return Err(invalid("events must name at-risk marks"));
            }
            if fired
                .iter()
                .filter(|&&d| self.spec.marks[d] == MarkKind::Terminal)
                .count()
                > 1
            {
                return Err(invalid("at most one terminal event can end follow-up"));
            }
            for (d, kind) in self.spec.marks.iter().enumerate() {
                let count = fired.iter().filter(|&&e| e == d).count();
                match kind {
                    MarkKind::Recurrent => (),
                    MarkKind::Once | MarkKind::Terminal if count > 1 => {
                        return Err(invalid(
                            "a once-only or terminal mark fires at most once at a node",
                        ));
                    }
                    MarkKind::Once => {
                        if count == 1 {
                            risk[d] = false;
                        }
                    }
                    MarkKind::Terminal => {
                        if count == 1 && node != n - 1 {
                            return Err(invalid("a terminal event must end observation"));
                        }
                    }
                }
            }
        }
        for record in &h.measurements {
            if record.node >= n || record.channel >= self.spec.measurements.len() {
                return Err(invalid("measurement names an unknown node or channel"));
            }
            if record.after_event
                && h.events[record.node]
                    .iter()
                    .any(|&d| self.spec.marks[d] == MarkKind::Terminal)
            {
                return Err(invalid(
                    "a measurement cannot observe a state after termination",
                ));
            }
            let family = &self.spec.measurements[record.channel];
            match (family, record.exposure) {
                (MeasurementFamily::NegativeBinomial, Some(e)) if e.is_finite() && e > 0.0 => (),
                (MeasurementFamily::NegativeBinomial, _) => {
                    return Err(invalid(
                        "a count measurement needs a finite positive exposure",
                    ));
                }
                (_, Some(_)) => {
                    return Err(invalid("only count measurements carry an exposure"));
                }
                (_, None) => (),
            }
            if let Some(value) = record.value {
                emission::validate_value(family, value)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::data::{FrozenJointSchema, JointTables};
    use super::super::model::rank_zero_declarations;
    use super::*;
    use crate::test_support::{Bound, agrees};
    use gam_math::nested_dual::JetField;

    /// The rank-zero histories of follow-up records `(entry, exit, events)`,
    /// events as `(time, mark index)`, encoded as `fit_joint_event_model`
    /// encodes them: by the production table encoder under the rank-zero
    /// declarations of `marks`.
    fn rank_zero_histories(marks: &[MarkKind], records: &[(f64, f64, &[(f64, usize)])]) -> Vec<JointHistory> {
        let names: Vec<String> = (0..marks.len()).map(|d| format!("m{d}")).collect();
        let mut tables = JointTables::default();
        for (i, &(entry, exit, events)) in records.iter().enumerate() {
            let id = format!("s{i}");
            tables.subjects.id.push(id.clone());
            tables.subjects.entry.push(entry);
            tables.subjects.exit.push(exit);
            for &(time, mark) in events {
                tables.events.id.push(id.clone());
                tables.events.time.push(time);
                tables.events.mark.push(names[mark].clone());
            }
        }
        let declared = names.iter().cloned().zip(marks.iter().copied()).collect();
        let (schema, subjects) =
            FrozenJointSchema::fit(&rank_zero_declarations(Some(declared)), &tables).unwrap();
        assert_eq!(schema.mark_names, names);
        assert_eq!(schema.mark_kinds, marks);
        subjects.into_iter().map(|subject| subject.history).collect()
    }

    fn specification(k: usize, families: Vec<MeasurementFamily>, genes: usize) -> JointSpecification {
        JointSpecification {
            signatures: k,
            marks: vec![MarkKind::Recurrent],
            baseline_columns: 1,
            population_columns: 1,
            drive_columns: 1,
            entry_columns: 0,
            baseline_penalties: vec![],
            drive_penalties: vec![],
            population_penalties: vec![],
            measurements: families,
            genetic_mean: vec![0.0; genes],
            genetic_precision: Array2::eye(genes),
        }
    }

    /// Five nodes at quarters of a unit window, one event of mark 0 at node 2,
    /// and every cell integrated by one point of its width at its node time.
    fn history(genes: usize) -> JointHistory {
        let times = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let points: Vec<CompensatorPoint> = (0..times.len())
            .flat_map(|node| {
                let time = times[node];
                std::iter::once(CompensatorPoint { node, time, weight: 0.0 })
                    .chain((node > 0).then_some(CompensatorPoint { node, time, weight: 0.25 }))
            })
            .collect();
        JointHistory {
            times,
            events: vec![vec![], vec![], vec![0], vec![], vec![]],
            initially_at_risk: vec![true],
            baseline_design: Array2::ones((points.len(), 1)),
            population_design: Array2::ones((points.len(), 1)),
            points,
            weight_error: 0.0,
            drive_design: Array2::ones((4, 1)),
            entry_design: vec![],
            genetics: vec![None; genes],
            measurements: vec![],
        }
    }

    fn record(node: usize, channel: usize, value: Option<f64>, exposure: Option<f64>) -> MeasurementRecord {
        MeasurementRecord {
            node,
            channel,
            value,
            exposure,
            after_event: false,
        }
    }

    #[test]
    fn node_path_density_equals_the_rank_zero_sufficient_statistics() {
        let spec = JointSpecification::new(vec![
            MarkKind::Once,
            MarkKind::Recurrent,
            MarkKind::Once,
            MarkKind::Terminal,
        ])
        .unwrap();
        // Mark 2 is prevalent at entry, recurrent mark 1 fires twice, and the
        // incident once-only mark 0 ties with the terminal event.
        let history = rank_zero_histories(
            &spec.marks,
            &[(1.0, 7.5, &[(7.5, 3), (0.5, 2), (2.0, 1), (3.0, 1), (7.5, 0)])],
        )
        .remove(0);
        let model = JointLikelihood::new(spec.clone()).unwrap();
        model.validate_history(&history).unwrap();
        assert_eq!(history.events.last().unwrap().len(), 2);
        let log_rates = [-1.3, 0.4, -0.2, -2.1];
        // Both routes share one evaluation of each rate, deliberately: std
        // leaves exp's precision unspecified, so there is no accuracy to charge,
        // and a shared value's error is common to both sides and cancels from
        // their difference.
        let rates = log_rates.map(f64::exp);
        // Node by node, with each route carrying its own rounding bound: every
        // cell compensates the marks at risk there, and every event adds its
        // log rate. Every gap (1, 1, 4.5) is an exact difference of exactly
        // represented times and every partial exposure (1, 2, 6.5) is exact, so
        // the weights and exposures both routes scale by are exact inputs.
        let mut risk = history.initially_at_risk.clone();
        let mut path = Bound::exact(0.0);
        let mut recomputed = [0.0; 4];
        for (node, fired) in history.events.iter().enumerate() {
            let weight = history
                .points
                .iter()
                .filter(|point| point.node == node)
                .fold(0.0, |sum, point| sum + point.weight);
            for d in 0..4 {
                if risk[d] {
                    recomputed[d] += weight;
                    path = path.sub(&Bound::exact(rates[d]).scale(weight));
                }
            }
            for &d in fired {
                assert!(risk[d], "event of mark {d} outside its risk set");
                path = path.add(&Bound::exact(log_rates[d]));
                if spec.marks[d] != MarkKind::Recurrent {
                    risk[d] = false;
                }
            }
        }
        let (counts, exposure) = history.rate_statistics(&spec.marks);
        assert_eq!(counts, [1, 2, 0, 1]);
        assert_eq!(exposure, [6.5, 6.5, 0.0, 6.5]);
        assert_eq!(exposure, recomputed);
        let sufficient = (0..4).fold(Bound::exact(0.0), |sum, d| {
            sum.add(&Bound::exact(log_rates[d]).scale(counts[d] as f64))
                .sub(&Bound::exact(rates[d]).scale(exposure[d]))
        });
        agrees(&path, &sufficient, "node sums against the sufficient statistics");
        // A hand restatement of slice 0's node sequence for this record, whose
        // code this landing replaces: an exposure node per gap, then one
        // zero-exposure node per event, terminal last. As a regression
        // specification it gives bitwise the same statistics.
        let slice_zero_exposure = [0.0, 2.0 - 1.0, 0.0, 3.0 - 2.0, 0.0, 7.5 - 3.0, 0.0, 0.0];
        let slice_zero_events = [None, None, Some(1), None, Some(1), None, Some(0), Some(3)];
        let mut slice_risk = history.initially_at_risk.clone();
        let mut slice_counts = vec![0u64; 4];
        let mut slice_exposure = vec![0.0_f64; 4];
        for (&weight, event) in slice_zero_exposure.iter().zip(&slice_zero_events) {
            for d in 0..4 {
                if slice_risk[d] {
                    slice_exposure[d] += weight;
                }
            }
            if let Some(d) = *event {
                slice_counts[d] += 1;
                if spec.marks[d] == MarkKind::Once {
                    slice_risk[d] = false;
                }
            }
        }
        assert_eq!(counts, slice_counts);
        let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&exposure), bits(&slice_exposure));
        assert_eq!(history.open_risk_set(&spec.marks), None);
        let open = rank_zero_histories(&spec.marks, &[(0.0, 4.0, &[(1.0, 0), (0.0, 2)])]).remove(0);
        model.validate_history(&open).unwrap();
        assert_eq!(
            open.open_risk_set(&spec.marks),
            Some(vec![false, true, false, true])
        );
        assert_eq!(open.rate_statistics(&spec.marks).1, [1.0, 4.0, 0.0, 4.0]);
    }

    #[test]
    fn a_specification_needs_a_mark() {
        // The follow-up records a rank-zero history cannot hold are refused by the
        // table encoder, through the cohort rules (data_tests.rs:
        // records_a_rank_zero_history_could_not_hold_are_refused_by_the_cohort_rules).
        assert!(JointSpecification::new(Vec::new()).is_err());
        assert!(JointSpecification::new(vec![MarkKind::Recurrent]).is_ok());
    }

    #[test]
    fn measurement_contracts_refuse_unsupported_values_and_exposures() {
        let model = JointLikelihood::new(specification(
            0,
            vec![
                MeasurementFamily::StudentT,
                MeasurementFamily::Probit { categories: 2 },
                MeasurementFamily::Probit { categories: 4 },
                MeasurementFamily::NegativeBinomial,
            ],
            0,
        ))
        .unwrap();
        for accepted in [
            record(1, 0, Some(-3.7), None),
            record(1, 1, Some(1.0), None),
            record(1, 2, Some(3.0), None),
            record(1, 3, Some(4.0), Some(2.5)),
            record(1, 3, None, Some(1.0)),
            record(1, 0, None, None),
        ] {
            let mut h = history(0);
            h.measurements.push(accepted.clone());
            assert!(model.validate_history(&h).is_ok(), "{accepted:?}");
        }
        for refused in [
            record(1, 0, Some(f64::INFINITY), None),
            record(1, 0, Some(1.0), Some(1.0)),
            record(1, 1, Some(0.5), None),
            record(1, 1, Some(2.0), None),
            record(1, 2, Some(4.0), None),
            record(1, 2, Some(-1.0), None),
            record(1, 3, Some(2.5), Some(1.0)),
            record(1, 3, Some(-1.0), Some(1.0)),
            record(1, 3, Some(2.0), None),
            record(1, 3, Some(2.0), Some(0.0)),
            record(5, 0, Some(1.0), None),
            record(1, 4, Some(1.0), None),
        ] {
            let mut h = history(0);
            h.measurements.push(refused.clone());
            assert!(
                matches!(
                    model.validate_history(&h),
                    Err(EventHistoryError::InvalidInput { .. })
                ),
                "{refused:?}"
            );
        }
        let mut h = history(0);
        h.population_design[[2, 0]] = 0.5;
        assert!(model.validate_history(&h).is_err());
        let mut h = history(0);
        h.points[2].weight += 1e-9;
        assert!(model.validate_history(&h).is_err());
        assert!(
            JointLikelihood::new(specification(
                0,
                vec![MeasurementFamily::Probit { categories: 1 }],
                0
            ))
            .is_err()
        );
        let block = |values: Vec<f64>, rank: usize| BasisPenalty {
            columns: 1..3,
            local: Array2::from_shape_vec((2, 2), values).unwrap(),
            rank,
        };
        let mut spec = specification(0, vec![], 0);
        spec.population_columns = 3;
        spec.population_penalties = vec![block(vec![1.0, 0.5, 0.5, 2.0], 2)];
        assert!(JointLikelihood::new(spec.clone()).is_ok());
        for refused in [
            block(vec![1.0, 0.5, 0.4, 2.0], 2),
            block(vec![1.0, 0.5, 0.5, 2.0], 3),
            block(vec![f64::NAN, 0.0, 0.0, 1.0], 1),
        ] {
            spec.population_penalties = vec![refused];
            assert!(JointLikelihood::new(spec.clone()).is_err());
        }
        spec.population_penalties = vec![];
        spec.baseline_penalties = vec![block(vec![1.0, 0.5, 0.5, 2.0], 2)];
        assert!(JointLikelihood::new(spec).is_err());
    }

    #[test]
    fn tied_events_and_compensator_points_follow_the_node_contract() {
        let mut spec = specification(1, vec![], 0);
        spec.marks = vec![MarkKind::Recurrent, MarkKind::Once, MarkKind::Recurrent];
        let model = JointLikelihood::new(spec).unwrap();
        let tie = |node: usize, fired: Vec<usize>| {
            let mut h = history(0);
            h.initially_at_risk = vec![true; 3];
            h.events[2].clear();
            h.events[node] = fired;
            h
        };
        // A recurrent mark may repeat within a tie, and an event may share its
        // node with that node's cell exposure.
        assert!(model.validate_history(&tie(2, vec![2, 0, 1, 0])).is_ok());
        assert!(model.validate_history(&tie(1, vec![0])).is_ok());
        for refused in [tie(2, vec![1, 1]), tie(2, vec![3])] {
            assert!(model.validate_history(&refused).is_err(), "{:?}", refused.events);
        }
        let mut again = tie(2, vec![1]);
        again.events[4] = vec![1];
        assert!(model.validate_history(&again).is_err());
        let mut spec = specification(0, vec![], 0);
        spec.marks = vec![MarkKind::Recurrent, MarkKind::Terminal];
        let terminal = JointLikelihood::new(spec).unwrap();
        let mut h = history(0);
        h.initially_at_risk = vec![true; 2];
        h.events[2] = vec![1, 0];
        assert!(terminal.validate_history(&h).is_err());
        h.events[2].clear();
        h.events[4] = vec![1, 0];
        assert!(terminal.validate_history(&h).is_ok());
        // Point structure: each node opens with a zero-weight anchor, and cell
        // points follow their own anchor.
        let base = history(0);
        assert!(model.validate_history(&base).is_err()); // three marks declared, one at risk
        let single = JointLikelihood::new(specification(1, vec![], 0)).unwrap();
        assert!(single.validate_history(&base).is_ok());
        let mut anchored = base.clone();
        anchored.points[1].weight = 0.1;
        assert!(single.validate_history(&anchored).is_err());
        let mut orphan = base.clone();
        orphan.points[3].node = 3;
        assert!(single.validate_history(&orphan).is_err());
        let mut entry = base.clone();
        entry.points.insert(1, CompensatorPoint { node: 0, time: 0.0, weight: 0.25 });
        entry.baseline_design = Array2::ones((entry.points.len(), 1));
        entry.population_design = Array2::ones((entry.points.len(), 1));
        assert!(single.validate_history(&entry).is_err());
        // Point times: an anchor sits at its node time, and node 1's cell
        // points lie in (0, 0.25] in time order.
        let mut anchor_off = base.clone();
        anchor_off.points[1].time = 0.3;
        assert!(single.validate_history(&anchor_off).is_err());
        let split = |first: f64, second: f64| {
            let mut h = base.clone();
            h.points[2] = CompensatorPoint { node: 1, time: first, weight: 0.125 };
            h.points.insert(3, CompensatorPoint { node: 1, time: second, weight: 0.125 });
            h.baseline_design = Array2::ones((h.points.len(), 1));
            h.population_design = Array2::ones((h.points.len(), 1));
            h
        };
        assert!(single.validate_history(&split(0.1, 0.25)).is_ok());
        for refused in [split(0.2, 0.1), split(0.0, 0.25), split(0.1, 0.3)] {
            assert!(single.validate_history(&refused).is_err(), "{:?}", refused.points);
        }
    }

    #[test]
    fn a_certified_gauss_legendre_cell_integrates_its_width() {
        let single = JointLikelihood::new(specification(1, vec![], 0)).unwrap();
        let rule = gam_math::special::gauss_legendre_certified(5);
        assert!(rule.weight_relative_error.is_finite());
        // Node 1's cell (0, 0.25] by the five-point rule. Each weight is the
        // halved width times a certified weight; the width's subtraction and
        // the product each round by at most half an ulp, so a correct encoding
        // declares the certified error plus one epsilon.
        let encoded = |points: usize, declared: f64| {
            let mut h = history(0);
            let half = (h.times[1] - h.times[0]) * 0.5;
            let mid = h.times[0] + half;
            let tail = h.points.split_off(2);
            h.points.extend(rule.nodes.iter().zip(&rule.weights).take(points).map(|(&x, &w)| {
                CompensatorPoint {
                    node: 1,
                    time: mid + half * x,
                    weight: half * w,
                }
            }));
            h.points.extend(tail.into_iter().skip(1));
            h.weight_error = declared;
            h.baseline_design = Array2::ones((h.points.len(), 1));
            h.population_design = Array2::ones((h.points.len(), 1));
            h
        };
        let declared = rule.weight_relative_error + f64::EPSILON;
        assert!(single.validate_history(&encoded(5, declared)).is_ok());
        // A cell missing one of its points, or a declared error outside [0, 1),
        // is refused.
        for refused in [encoded(4, declared), encoded(5, -declared), encoded(5, 1.0)] {
            assert!(single.validate_history(&refused).is_err(), "{:?}", refused.points);
        }
    }
}
