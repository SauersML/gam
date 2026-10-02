//! Logit-row comparison with forward-error radii, and exhaustive suprema over a finite family
//! (#2951).
//!
//! The comparison a contract runs on every row it checks: two logit rows, each within a
//! per-entry radius of its exact value, and what can be certified about their distance.
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
//! A family maximum ([`exhaustive_supremum`]) is exact with the largest row value and the
//! largest row error, by the same inequality. When some row's lower end `value − error`
//! exceeds the declared tolerance, the row with the largest lower end is returned as a
//! counterexample instead.

use super::bounds::{BoundError, KlBoundRegion, kl_over_logit_boxes};
use super::supports::{EvidenceStatus, EvidenceStatusError, ExactBasis, Extremum};
use gam_linalg::roundoff::accumulation_growth;
use ndarray::ArrayView1;

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
}
