//! Exhaustive verification of an explanation against the native model over a
//! declared finite family (#2951).
//!
//! An explanation is checked where checking is possible: on every member of a finite input
//! family the experiment declares, and under every member of a declared family of internal
//! interventions (the counterfactual contract). Two executables produce logit rows with a
//! per-entry forward-error radius against their exact values ([`Executable`]), each deriving
//! its own radii. Nothing is sampled, so every family statistic is
//! [`EvidenceStatus::Exact`] with [`ExactBasis::Exhaustive`] over the rows compared, a
//! [`EvidenceStatus::Counterexample`] naming the input that exceeds the declared tolerance, or
//! [`EvidenceStatus::Unresolved`] when some row's radii are too wide for a bound.
//!
//! # Per row
//!
//! For reference logits `r` and candidate logits `c`, each within its radius of exact:
//!
//! * `KL(softmax r ‖ softmax c)` and the reverse, each from [`kl_over_logit_boxes`]: the exact
//!   divergence of the centres with its evaluation error plus the shift any exact logits in the
//!   two boxes can make.
//! * The computed argmax of each side, and whether both are certified: the top entry's lower
//!   end clears every other entry's upper end, so the computed argmax is the exact one.
//! * The largest centred-logit gap `max_i |d_i|`, `d = (c − r) − mean(c − r)`, which is
//!   invariant to a constant shift of either side, as the distributions are. With
//!   `δ̂_i = fl(c_i − r_i)`, `μ̂ = fl(Σ δ̂/n)` and `d̂_i = fl(δ̂_i − μ̂)`: the subtraction errs by
//!   `u|δ_i|`, the mean by `γ_{n+1} Σ|δ̂|/n` (the rounded terms, `n − 1` additions and a
//!   division) and the last subtraction by `u|d̂_i|`, so the computed gap is within
//!   `γ_{n+2}(|δ̂_i| + Σ|δ̂|/n + |d̂_i|)` of the centres' gap. An exact logit inside the boxes
//!   moves `d_i` by at most `2 max_j (ρ^c_j + ρ^r_j)`, since `|e_i − mean e| ≤ max e − min e`.
//!   The row's band is the largest entry's, as `|max_i a_i − max_i b_i| ≤ max_i |a_i − b_i|`.
//!
//! # Over the family
//!
//! A family maximum is exact with the largest row value and the largest row error, by the same
//! inequality. When some row's lower end `value − error` exceeds the declared tolerance, the
//! row with the largest lower end is returned as a counterexample instead. The tolerance is an
//! experiment declaration with no default. The argmax disagreement fraction is an exact count
//! over the computed logits, whose one rounding is the division; the rows whose two argmaxes are
//! both certified are counted too, and on them the count is also the exact models'.

use super::bounds::{BoundError, KlBoundRegion, kl_over_logit_boxes};
use super::secant::BandedMatrix;
use super::supports::{EvidenceStatus, EvidenceStatusError, ExactBasis, Extremum};
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use gam_runtime::resource::MemoryGovernor;
use ndarray::ArrayView1;
use std::fmt;

/// A model executed on one input under one intervention: logit rows (one categorical
/// distribution per row) with a per-entry radius against their exact values.
pub trait Executable<V, I> {
    type Error: std::error::Error;

    fn logits(&self, governor: &MemoryGovernor, intervention: &V, input: &I) -> Result<BandedMatrix, Self::Error>;
}

impl<V, I, E, F> Executable<V, I> for F
where
    E: std::error::Error,
    F: Fn(&MemoryGovernor, &V, &I) -> Result<BandedMatrix, E>,
{
    type Error = E;

    fn logits(&self, governor: &MemoryGovernor, intervention: &V, input: &I) -> Result<BandedMatrix, E> {
        self(governor, intervention, input)
    }
}

/// The declared fidelity tolerances. They are experiment declarations with no default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    /// On each direction of the per-row KL divergence.
    pub kl: f64,
    /// On the per-row largest centred-logit gap.
    pub centred_logit_gap: f64,
}

/// A row of the family: the input's index in the declared family and the logit row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FamilyWitness {
    pub input: usize,
    pub row: usize,
}

/// The finite family a status is stated over: every logit row of every declared input,
/// under one declared intervention of the contract when `intervention` names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FamilyDomain {
    pub intervention: Option<usize>,
    pub inputs: usize,
    pub rows: usize,
}

pub type FamilyStatus = EvidenceStatus<FamilyWitness, FamilyDomain>;

/// One row's comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct RowComparison {
    /// `KL(softmax reference ‖ softmax candidate)` over both logit boxes.
    pub forward_kl: EvidenceStatus<(), KlBoundRegion>,
    /// `KL(softmax candidate ‖ softmax reference)` over both logit boxes.
    pub reverse_kl: EvidenceStatus<(), KlBoundRegion>,
    pub reference_argmax: usize,
    pub candidate_argmax: usize,
    /// Both computed argmaxes are the exact ones.
    pub argmax_certified: bool,
    /// The largest centred-logit gap of the centres and its band against the exact logits.
    pub centred_gap: f64,
    pub centred_gap_band: f64,
}

/// A refused verification.
#[derive(Debug)]
pub enum VerifyError<R, C> {
    Reference { intervention: usize, input: usize, error: R },
    Candidate { intervention: usize, input: usize, error: C },
    /// The two executables returned logit rows of different shapes, or rows without radii
    /// of their shape.
    Shape { input: usize, reference: (usize, usize), candidate: (usize, usize) },
    Bound(BoundError),
    Evidence(EvidenceStatusError),
    /// A declared tolerance that is negative or not finite.
    Tolerance { field: &'static str, value: f64 },
    EmptyFamily,
}

impl<R: fmt::Display, C: fmt::Display> fmt::Display for VerifyError<R, C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reference { intervention, input, error } => {
                write!(formatter, "reference at intervention {intervention}, input {input}: {error}")
            }
            Self::Candidate { intervention, input, error } => {
                write!(formatter, "candidate at intervention {intervention}, input {input}: {error}")
            }
            Self::Shape { input, reference, candidate } => write!(
                formatter,
                "input {input}: reference logits {reference:?} and candidate logits {candidate:?} do not pair"
            ),
            Self::Bound(error) => write!(formatter, "verification: {error}"),
            Self::Evidence(error) => write!(formatter, "verification: {error}"),
            Self::Tolerance { field, value } => {
                write!(formatter, "the declared {field} tolerance {value} is not finite and nonnegative")
            }
            Self::EmptyFamily => write!(formatter, "a verification needs a nonempty declared family"),
        }
    }
}

impl<R: fmt::Debug + fmt::Display, C: fmt::Debug + fmt::Display> std::error::Error for VerifyError<R, C> {}

impl<R, C> From<BoundError> for VerifyError<R, C> {
    fn from(error: BoundError) -> Self {
        Self::Bound(error)
    }
}

impl<R, C> From<EvidenceStatusError> for VerifyError<R, C> {
    fn from(error: EvidenceStatusError) -> Self {
        Self::Evidence(error)
    }
}

/// The first index of the largest value.
pub fn computed_argmax(values: ArrayView1<'_, f64>) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |best, (index, &value)| if value > best.1 { (index, value) } else { best })
        .0
}

/// The computed argmax is the exact one: its lower end clears every other entry's upper end,
/// each rounded outward.
pub fn certified_argmax(values: ArrayView1<'_, f64>, radius: ArrayView1<'_, f64>) -> bool {
    let top = computed_argmax(values);
    let floor = values[top] - radius[top];
    values
        .iter()
        .zip(radius.iter())
        .enumerate()
        .all(|(index, (value, radius))| index == top || floor.next_down() > (value + radius).next_up())
}

/// Compares one reference logit row with one candidate row, each with its radius.
pub fn compare_logit_row(
    reference: ArrayView1<'_, f64>,
    reference_radius: ArrayView1<'_, f64>,
    candidate: ArrayView1<'_, f64>,
    candidate_radius: ArrayView1<'_, f64>,
) -> Result<RowComparison, BoundError> {
    // The bound validates lengths, finiteness and radii before anything else reads the rows.
    let forward_kl = kl_over_logit_boxes(reference, reference_radius, candidate, candidate_radius)?;
    let reverse_kl = kl_over_logit_boxes(candidate, candidate_radius, reference, reference_radius)?;
    let n = reference.len();
    let gaps: Vec<f64> = candidate.iter().zip(reference.iter()).map(|(c, r)| c - r).collect();
    let absolute_mean = (gaps.iter().fold(0.0_f64, |sum, gap| (sum + gap.abs()).next_up()) / n as f64).next_up();
    let mean = gaps.iter().sum::<f64>() / n as f64;
    let widest = reference_radius
        .iter()
        .zip(candidate_radius.iter())
        .fold(0.0_f64, |widest, (a, b)| widest.max((a + b).next_up()));
    let growth = accumulation_growth(n + 2).next_up();
    let (mut centred_gap, mut centred_gap_band) = (0.0_f64, 0.0_f64);
    for gap in &gaps {
        let centred = gap - mean;
        let rounding = (growth * ((gap.abs() + absolute_mean).next_up() + centred.abs()).next_up()).next_up();
        centred_gap = centred_gap.max(centred.abs());
        centred_gap_band = centred_gap_band.max((rounding + (2.0 * widest).next_up()).next_up());
    }
    let (reference_argmax, candidate_argmax) = (computed_argmax(reference), computed_argmax(candidate));
    Ok(RowComparison {
        forward_kl,
        reverse_kl,
        reference_argmax,
        candidate_argmax,
        argmax_certified: certified_argmax(reference, reference_radius) && certified_argmax(candidate, candidate_radius),
        centred_gap,
        centred_gap_band,
    })
}

/// What one row proves about a nonnegative quantity whose family maximum is reported.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RowValue {
    /// The row's exact value up to its numerical error.
    Resolved { value: f64, numerical_error: f64 },
    /// Only a proven lower end.
    Unresolved { lower: f64 },
}

impl RowValue {
    /// A row's status as a row value: resolved when exact, otherwise its proven lower end.
    pub fn of<W, D>(status: &EvidenceStatus<W, D>) -> Self {
        match status {
            EvidenceStatus::Exact { value, numerical_error, .. } => {
                Self::Resolved { value: *value, numerical_error: *numerical_error }
            }
            other => Self::Unresolved { lower: other.lower_bound().unwrap_or(0.0).max(0.0) },
        }
    }
}

/// A row's witness, its proven lower end, and its value and error when it resolved.
type LowestRow<W> = (W, f64, Option<(f64, f64)>);

/// The maximum over a whole finite family of row values (module documentation, *Over the
/// family*): a counterexample at the row with the largest lower end when that exceeds
/// `tolerance`, otherwise `Exact` with the largest value and the largest error when every row
/// resolved, otherwise `Unresolved` between the largest lower end and `+∞`.
pub fn exhaustive_supremum<W: Clone, D>(
    rows: impl IntoIterator<Item = (W, RowValue)>,
    tolerance: Option<f64>,
    domain: D,
) -> Result<EvidenceStatus<W, D>, EvidenceStatusError> {
    let mut cardinality = 0u64;
    let mut largest: Option<(W, f64)> = None;
    let mut error = 0.0_f64;
    // The row with the largest proven lower end, and its value and error when it resolved.
    let mut lower: Option<LowestRow<W>> = None;
    let mut unresolved = false;
    for (witness, row) in rows {
        cardinality += 1;
        let (attained, resolved) = match row {
            RowValue::Resolved { value, numerical_error } => {
                if largest.as_ref().is_none_or(|(_, best)| value > *best) {
                    largest = Some((witness.clone(), value));
                }
                error = error.max(numerical_error);
                let attained = if numerical_error == 0.0 { value } else { (value - numerical_error).next_down() };
                (attained.max(0.0), Some((value, numerical_error)))
            }
            RowValue::Unresolved { lower } => {
                unresolved = true;
                (lower, None)
            }
        };
        if lower.as_ref().is_none_or(|(_, best, _)| attained > *best) {
            lower = Some((witness, attained, resolved));
        }
    }
    if let (Some(tolerance), Some((witness, attained, Some((value, numerical_error))))) = (tolerance, &lower)
        && *attained > tolerance
    {
        return EvidenceStatus::counterexample(*value, *numerical_error, tolerance, witness.clone());
    }
    match (unresolved, largest) {
        (false, Some((witness, value))) => {
            EvidenceStatus::exact(value, error, ExactBasis::Exhaustive { cardinality }, Some(witness), domain)
        }
        (false, None) => Err(EvidenceStatusError::EmptyExhaustiveFamily),
        (true, _) => {
            let (witness, attained) = lower.map_or((None, 0.0), |(witness, attained, _)| (Some(witness), attained));
            EvidenceStatus::unresolved(attained, f64::INFINITY, Extremum::Supremum, witness, domain)
        }
    }
}

/// Argmax agreement over the family: an exact count on the computed logits.
#[derive(Clone, Debug, PartialEq)]
pub struct ArgmaxAgreement {
    pub rows: u64,
    pub disagreeing: u64,
    /// Rows whose two argmaxes are both certified; on them the count is the exact models'.
    pub certified_rows: u64,
    /// The disagreeing fraction, exact over the family, witnessed by the first disagreeing row.
    pub fraction: FamilyStatus,
}

/// Everything one intervention's family shows.
#[derive(Clone, Debug, PartialEq)]
pub struct FamilyVerification {
    pub forward_kl: FamilyStatus,
    pub reverse_kl: FamilyStatus,
    pub centred_logit_gap: FamilyStatus,
    pub argmax: ArgmaxAgreement,
    /// Every row's comparison, in family order.
    pub rows: Vec<(FamilyWitness, RowComparison)>,
}

impl FamilyVerification {
    /// Both KL directions and the centred-logit gap are proven within `tolerance`.
    pub fn certified_within(&self, tolerance: &Tolerance) -> bool {
        self.forward_kl.certifies_at_most(tolerance.kl)
            && self.reverse_kl.certifies_at_most(tolerance.kl)
            && self.centred_logit_gap.certifies_at_most(tolerance.centred_logit_gap)
    }
}

/// The counterfactual contract: one family verification per declared intervention, in order.
#[derive(Clone, Debug, PartialEq)]
pub struct CounterfactualContract {
    pub interventions: Vec<FamilyVerification>,
}

impl CounterfactualContract {
    /// Every intervention's family is proven within `tolerance`.
    pub fn certified_within(&self, tolerance: &Tolerance) -> bool {
        self.interventions.iter().all(|family| family.certified_within(tolerance))
    }
}

fn check_tolerance<R, C>(tolerance: &Tolerance) -> Result<(), VerifyError<R, C>> {
    for (field, value) in [("KL", tolerance.kl), ("centred-logit gap", tolerance.centred_logit_gap)] {
        if !(value.is_finite() && value >= 0.0) {
            return Err(VerifyError::Tolerance { field, value });
        }
    }
    Ok(())
}

/// Verifies `candidate` against `reference` on every input of `family` under one intervention.
pub fn verify_family<V, I, R, C>(
    governor: &MemoryGovernor,
    reference: &R,
    candidate: &C,
    intervention: &V,
    family: &[I],
    tolerance: &Tolerance,
) -> Result<FamilyVerification, VerifyError<R::Error, C::Error>>
where
    R: Executable<V, I>,
    C: Executable<V, I>,
{
    check_tolerance(tolerance)?;
    family_at(governor, reference, candidate, (None, intervention), family, tolerance)
}

/// The counterfactual contract: `candidate` against `reference` on every input of `family`
/// under every declared intervention, reported per intervention.
pub fn verify_counterfactual_contract<V, I, R, C>(
    governor: &MemoryGovernor,
    reference: &R,
    candidate: &C,
    interventions: &[V],
    family: &[I],
    tolerance: &Tolerance,
) -> Result<CounterfactualContract, VerifyError<R::Error, C::Error>>
where
    R: Executable<V, I>,
    C: Executable<V, I>,
{
    check_tolerance(tolerance)?;
    if interventions.is_empty() {
        return Err(VerifyError::EmptyFamily);
    }
    let interventions = interventions
        .iter()
        .enumerate()
        .map(|(k, intervention)| family_at(governor, reference, candidate, (Some(k), intervention), family, tolerance))
        .collect::<Result<_, _>>()?;
    Ok(CounterfactualContract { interventions })
}

fn family_at<V, I, R, C>(
    governor: &MemoryGovernor,
    reference: &R,
    candidate: &C,
    (index, intervention): (Option<usize>, &V),
    family: &[I],
    tolerance: &Tolerance,
) -> Result<FamilyVerification, VerifyError<R::Error, C::Error>>
where
    R: Executable<V, I>,
    C: Executable<V, I>,
{
    if family.is_empty() {
        return Err(VerifyError::EmptyFamily);
    }
    let k = index.unwrap_or(0);
    let mut rows = Vec::new();
    for (input_index, input) in family.iter().enumerate() {
        let reference_rows = reference
            .logits(governor, intervention, input)
            .map_err(|error| VerifyError::Reference { intervention: k, input: input_index, error })?;
        let candidate_rows = candidate
            .logits(governor, intervention, input)
            .map_err(|error| VerifyError::Candidate { intervention: k, input: input_index, error })?;
        let shape = reference_rows.values.dim();
        if candidate_rows.values.dim() != shape
            || reference_rows.bands.dim() != shape
            || candidate_rows.bands.dim() != shape
        {
            return Err(VerifyError::Shape {
                input: input_index,
                reference: shape,
                candidate: candidate_rows.values.dim(),
            });
        }
        for row in 0..shape.0 {
            let comparison = compare_logit_row(
                reference_rows.values.row(row),
                reference_rows.bands.row(row),
                candidate_rows.values.row(row),
                candidate_rows.bands.row(row),
            )?;
            rows.push((FamilyWitness { input: input_index, row }, comparison));
        }
    }
    let domain = FamilyDomain { intervention: index, inputs: family.len(), rows: rows.len() };
    let forward_kl = exhaustive_supremum(
        rows.iter().map(|(witness, row)| (*witness, RowValue::of(&row.forward_kl))),
        Some(tolerance.kl),
        domain,
    )?;
    let reverse_kl = exhaustive_supremum(
        rows.iter().map(|(witness, row)| (*witness, RowValue::of(&row.reverse_kl))),
        Some(tolerance.kl),
        domain,
    )?;
    let centred_logit_gap = exhaustive_supremum(
        rows.iter().map(|(witness, row)| {
            (*witness, RowValue::Resolved { value: row.centred_gap, numerical_error: row.centred_gap_band })
        }),
        Some(tolerance.centred_logit_gap),
        domain,
    )?;
    let disagreeing = rows.iter().filter(|(_, row)| row.reference_argmax != row.candidate_argmax).count();
    let first = rows.iter().find(|(_, row)| row.reference_argmax != row.candidate_argmax).map(|(witness, _)| *witness);
    let fraction = disagreeing as f64 / rows.len() as f64;
    let argmax = ArgmaxAgreement {
        rows: rows.len() as u64,
        disagreeing: disagreeing as u64,
        certified_rows: rows.iter().filter(|(_, row)| row.argmax_certified).count() as u64,
        fraction: EvidenceStatus::exact(
            fraction,
            UNIT_ROUNDOFF * fraction,
            ExactBasis::Exhaustive { cardinality: rows.len() as u64 },
            first,
            domain,
        )?,
    };
    Ok(FamilyVerification { forward_kl, reverse_kl, centred_logit_gap, argmax, rows })
}

#[cfg(test)]
mod tests {
    //! A planted modular-addition network over `Z_5` whose entries are small dyadic numbers, so
    //! every value is exact: hidden unit `(a, b)` is `relu(x_a + x_{P+b} − 1)`, which is `1`
    //! exactly on the input `(a, b)`, and the readout puts `2` on class `a + b mod P`.

    use super::*;
    use crate::test_support::test_governor;
    use ndarray::{Array1, Array2, Axis, array};

    const P: usize = 5;
    const H: usize = P * P;

    /// The one-hot row `[e_a, e_b]` of the pair `(a, b)`.
    fn pair_input(a: usize, b: usize) -> Array1<f64> {
        let mut row = Array1::zeros(2 * P);
        row[a] = 1.0;
        row[P + b] = 1.0;
        row
    }

    fn family() -> Vec<Array1<f64>> {
        (0..P).flat_map(|a| (0..P).map(move |b| pair_input(a, b))).collect()
    }

    fn unit(a: usize, b: usize) -> usize {
        a * P + b
    }

    fn planted_tensors() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let mut w1 = Array2::zeros((H, 2 * P));
        let mut w2 = Array2::zeros((P, H));
        for a in 0..P {
            for b in 0..P {
                w1[[unit(a, b), a]] = 1.0;
                w1[[unit(a, b), P + b]] = 1.0;
                w2[[(a + b) % P, unit(a, b)]] = 2.0;
            }
        }
        (w1, Array1::from_elem(H, -1.0), w2)
    }

    /// `x → Σ_k R_k (h · relu(W1 x + b1))` with the hidden units scaled by `hidden`. Every
    /// operand is a small dyadic number, so the row is exact and its band is zero.
    fn network_logits(readouts: &[Array2<f64>], hidden: f64, input: &Array1<f64>) -> BandedMatrix {
        let (w1, b1, _) = planted_tensors();
        let activation = (w1.dot(input) + &b1).mapv(|value| hidden * value.max(0.0));
        let mut logits = Array1::zeros(P);
        for readout in readouts {
            logits += &readout.dot(&activation);
        }
        BandedMatrix { values: logits.insert_axis(Axis(0)), bands: Array2::zeros((1, P)) }
    }

    fn tolerance(value: f64) -> Tolerance {
        Tolerance { kl: value, centred_logit_gap: value }
    }

    /// The counterfactual contract of an explanation that splits the readout into two halves
    /// `W2a + W2b`, with `defect` added to the second half, over hidden-unit scales.
    fn contract(defect: &Array2<f64>, hidden_values: &[f64], declared: &Tolerance) -> CounterfactualContract {
        let (_, _, w2) = planted_tensors();
        let w2a = &w2 * 0.5;
        let w2b = &w2 - &w2a + defect;
        let native = [w2];
        let explanation = [w2a, w2b];
        let reference_run = |_: &MemoryGovernor, hidden: &f64, input: &Array1<f64>| -> Result<BandedMatrix, fmt::Error> {
            Ok(network_logits(&native, *hidden, input))
        };
        let candidate_run = |_: &MemoryGovernor, hidden: &f64, input: &Array1<f64>| -> Result<BandedMatrix, fmt::Error> {
            Ok(network_logits(&explanation, *hidden, input))
        };
        verify_counterfactual_contract(test_governor(), &reference_run, &candidate_run, hidden_values, &family(), declared)
            .expect("a verification")
    }

    fn assert_exact_zero(status: &FamilyStatus, cardinality: u64) {
        match status {
            EvidenceStatus::Exact { value, numerical_error, basis, .. } => {
                assert_eq!(*basis, ExactBasis::Exhaustive { cardinality });
                assert!(*value <= *numerical_error, "{value} exceeds its band {numerical_error}");
            }
            other => panic!("expected an exhaustive exact status, got {other:?}"),
        }
    }

    #[test]
    fn a_planted_explanation_equal_to_the_native_program_is_exact_zero_on_every_input() {
        let declared = tolerance(1e-12);
        let report = contract(&Array2::zeros((P, H)), &[1.0, 0.5, 0.0], &declared);
        assert_eq!(report.interventions.len(), 3);
        for family in &report.interventions {
            for status in [&family.forward_kl, &family.reverse_kl, &family.centred_logit_gap] {
                assert_exact_zero(status, (P * P) as u64);
            }
            assert_eq!(family.argmax.disagreeing, 0);
        }
        assert!(report.certified_within(&declared));
        // Every row is certified at full strength, where the answer leads by 2.
        assert_eq!(report.interventions[0].argmax.certified_rows, (P * P) as u64);
        // Positive control: the native rows compute the modular sum.
        let row = &report.interventions[0].rows[unit(3, 4)].1;
        assert_eq!(row.reference_argmax, (3 + 4) % P);
    }

    #[test]
    fn a_planted_explanation_differing_on_one_input_is_refuted_at_that_input_under_each_intervention() {
        let (a, b) = (3, 4);
        let wrong = (a + b + 1) % P;
        let mut defect = Array2::zeros((P, H));
        defect[[wrong, unit(a, b)]] = 4.0;
        let declared = tolerance(1e-3);
        let report = contract(&defect, &[1.0, 0.5, 0.0], &declared);
        let target = FamilyWitness { input: unit(a, b), row: 0 };
        for (k, family) in report.interventions.iter().enumerate().take(2) {
            for status in [&family.forward_kl, &family.reverse_kl, &family.centred_logit_gap] {
                match status {
                    EvidenceStatus::Counterexample { witness, threshold, .. } => {
                        assert_eq!(*witness, target, "intervention {k}");
                        assert_eq!(*threshold, 1e-3);
                    }
                    other => panic!("intervention {k}: expected a counterexample, got {other:?}"),
                }
            }
            assert_eq!(family.argmax.disagreeing, 1);
            assert_eq!(family.argmax.fraction.witness(), Some(&target));
        }
        // Full strength: the defect's logit 4 against the answer's 2, centred over P classes.
        let row = &report.interventions[0].rows[unit(a, b)].1;
        assert_eq!(row.candidate_argmax, wrong);
        let expected_gap = 4.0 - 4.0 / P as f64;
        assert!((row.centred_gap - expected_gap).abs() <= row.centred_gap_band);
        // With every hidden unit deleted neither side sees the defect: the contract holds under
        // that intervention and is refuted under the others.
        let deleted = &report.interventions[2];
        for status in [&deleted.forward_kl, &deleted.reverse_kl, &deleted.centred_logit_gap] {
            assert_exact_zero(status, (P * P) as u64);
        }
        assert!(!report.certified_within(&declared));
        for (witness, row) in &report.interventions[0].rows {
            if *witness != target {
                assert_eq!(row.centred_gap, 0.0, "{witness:?}");
            }
        }
    }

    #[test]
    fn a_two_class_row_matches_the_closed_form_divergences() {
        // With δ = (t, −t) over the uniform reference, `KL = log E_p e^δ − E_p δ = log cosh t`.
        let t: f64 = 0.75;
        let zero = array![0.0, 0.0];
        let moved = array![t, -t];
        let row = compare_logit_row(zero.view(), zero.view(), moved.view(), zero.view()).expect("a row");
        let (lower, upper) = (
            row.forward_kl.lower_bound().expect("exact"),
            row.forward_kl.upper_bound().expect("exact"),
        );
        let expected = t.cosh().ln();
        assert!(lower <= expected && expected <= upper, "{lower} {expected} {upper}");
        // Reverse: `Σ q_i log(2 q_i)` with `q = softmax(t, −t)`.
        let q = [1.0 / (1.0 + (-2.0 * t).exp()), 1.0 / (1.0 + (2.0 * t).exp())];
        let reverse_expected: f64 = q.iter().map(|&qi| qi * (2.0 * qi).ln()).sum();
        let (reverse_lower, reverse_upper) = (
            row.reverse_kl.lower_bound().expect("exact"),
            row.reverse_kl.upper_bound().expect("exact"),
        );
        // The test's own evaluation of the closed form rounds, by far less than 1e-15.
        assert!(reverse_lower - 1e-15 <= reverse_expected && reverse_expected <= reverse_upper + 1e-15);
        assert!((expected - reverse_expected).abs() > 1e-3, "the two directions differ");
        assert_eq!(row.centred_gap, t);
    }

    #[test]
    fn a_constant_logit_shift_is_invisible_and_a_radius_widens_every_band() {
        let governor = test_governor();
        let reference = |_: &MemoryGovernor, _: &(), x: &f64| -> Result<BandedMatrix, std::fmt::Error> {
            Ok(BandedMatrix { values: array![[*x, 0.5, -1.0]], bands: Array2::zeros((1, 3)) })
        };
        let shifted = |_: &MemoryGovernor, _: &(), x: &f64| -> Result<BandedMatrix, std::fmt::Error> {
            Ok(BandedMatrix { values: array![[*x + 4.0, 4.5, 3.0]], bands: Array2::zeros((1, 3)) })
        };
        let inputs = [0.25, -2.0, 1.0];
        let exact = verify_family(governor, &reference, &shifted, &(), &inputs, &tolerance(0.0)).expect("a verification");
        assert_exact_zero(&exact.centred_logit_gap, 3);
        assert_eq!(exact.argmax.disagreeing, 0);
        // A declared radius on the candidate: the bands must grow to cover it.
        let widened = |_: &MemoryGovernor, _: &(), x: &f64| -> Result<BandedMatrix, std::fmt::Error> {
            Ok(BandedMatrix { values: array![[*x + 4.0, 4.5, 3.0]], bands: Array2::from_elem((1, 3), 1e-3) })
        };
        let wide = verify_family(governor, &reference, &widened, &(), &inputs, &tolerance(1e-9)).expect("a verification");
        let (EvidenceStatus::Exact { numerical_error: narrow, .. }, EvidenceStatus::Exact { numerical_error: broad, .. }) =
            (&exact.forward_kl, &wide.forward_kl)
        else {
            panic!("expected exact statuses");
        };
        // At equal centres the box shift is `β²/8` with `β = 2 · 1e-3`: `5e-7`.
        assert!(*broad >= 5e-7 && broad > narrow, "{broad} {narrow}");
        assert!(!wide.certified_within(&tolerance(1e-9)), "a radius of 1e-3 cannot certify 1e-9");
        assert!(wide.centred_logit_gap.upper_bound().expect("exact") >= 2e-3, "twice the widest radius sum");
    }

    #[test]
    fn undeclared_tolerances_and_an_empty_family_are_refused() {
        let governor = test_governor();
        let row = |_: &MemoryGovernor, _: &(), _: &f64| -> Result<BandedMatrix, std::fmt::Error> {
            Ok(BandedMatrix { values: array![[0.0, 1.0]], bands: Array2::zeros((1, 2)) })
        };
        for (kl, gap) in [(f64::NAN, 0.0), (0.0, -1.0), (f64::INFINITY, 0.0)] {
            let declared = Tolerance { kl, centred_logit_gap: gap };
            assert!(matches!(
                verify_family(governor, &row, &row, &(), &[0.0], &declared),
                Err(VerifyError::Tolerance { .. })
            ));
        }
        assert!(matches!(verify_family(governor, &row, &row, &(), &[], &tolerance(0.0)), Err(VerifyError::EmptyFamily)));
    }
}
