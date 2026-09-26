//! Nonlinear separation over the mask-moment zonotope (#2951).
//!
//! A candidate support `S` is sufficient at the declared tolerance `epsilon` iff a
//! native objective `F` (a distance of the edited network's output from the all-on
//! output) stays at or below `epsilon` over every admissible mask that keeps `S` at
//! all-on. The declared domain is a box `m_c in [lower_c, upper_c]` containing 1, and
//! the tensors depend on the mask only through the moment `q = sum_c (1 - m_c) v_c`
//! (P8), so that supremum equals the supremum over the zonotope
//!
//! ```text
//! Z_S = sum_{c not in S} [(1 - upper_c) v_c, (1 - lower_c) v_c]
//! ```
//!
//! The network is never linearized: every value is a native evaluation at a mask.
//!
//! # Ascent
//!
//! Frank-Wolfe ascent over the box. At the iterate `m_k` with moment gradient
//! `g_k = dF/dq`, the support function of [`super::moments`] returns
//! `h(g_k) = max_{q' in Z_S} g_k . q'` with an endpoint witness mask `s_k`, and the gap
//!
//! ```text
//! G_k = h(g_k) - g_k . q(m_k) = max_{q' in Z_S} g_k . (q' - q(m_k)) >= 0
//! ```
//!
//! vanishes exactly at a first-order stationary point of `F` over `Z_S`. The iterate
//! moves to `m_k + gamma (s_k - m_k)`, so the mask stays in the declared box as a
//! convex combination of witness masks and is carried by the iteration itself.
//!
//! The step halves from `gamma = 1` until the certified increase is at least
//! `gamma G_k / 2`. The factor `1/2` is derived: for a concave quadratic model
//! `phi(gamma) = G gamma - M gamma^2 / 2` the condition holds iff `gamma <= G / M`, the
//! model's maximiser, so the test accepts exactly the steps that do not overshoot the
//! model and halving returns one within a factor two of it. If `dF` is `L`-Lipschitz on
//! `Z_S` (the constant exists; the ascent never uses it) each accepted step raises `F` by
//! at least `min(G / 2, G^2 / (4 L D^2))` with `D` the diameter of `Z_S`, so the smallest
//! gap falls as `O(1 / sqrt k)`.
//!
//! # Termination
//!
//! Only derived criteria; there is no budget, iteration cap or grid:
//! * a certified counterexample, `F - roundoff > epsilon`;
//! * stationarity, the computed gap at or below its own roundoff bound;
//! * resolution: the increase a step must certify is at or below the value resolution
//!   of the iterate, or the trial mask rounds to the iterate.
//!
//! Every accepted step raises the computed value strictly, so the loop is finite.
//!
//! # What is reported
//!
//! The lower witness is the last iterate with its mask: the ascent is monotone, so it
//! is the best value seen. It bounds the supremum from below only; the ascent is local, so
//! a query that does not refute epsilon is unresolved with no upper side.

use super::moments::{
    MaskDomain, MaskMomentSystem, MomentGeometryError, MomentVector, WitnessEndpoint,
};
use super::supports::{EvidenceStatus, EvidenceStatusError, Extremum};
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_band, accumulation_growth};
use ndarray::Array1;
use std::fmt;

/// The region a separation status speaks about: `Z_S` of the declared domain, with the kept
/// set pinned at all-on.
#[derive(Clone, Debug, PartialEq)]
pub struct SeparationRegion {
    pub domain: MaskDomain,
    pub kept: Vec<bool>,
}

/// The evidence a separation query reports: the witness is a mask.
pub type SeparationStatus = EvidenceStatus<Vec<f64>, SeparationRegion>;

/// One native evaluation of the separation objective at a mask.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectiveJet {
    /// `F` at the mask.
    pub value: f64,
    /// A bound on `|computed F - exact F|` at the mask.
    pub value_roundoff: f64,
    /// `dF/dq` in the moment system's block layout: the native pullback contracted against
    /// the field basis, `<dF/dTheta, B_j>`.
    pub moment_gradient: MomentVector,
    /// A bound on the l2 norm of the computed minus the exact moment gradient.
    pub gradient_roundoff: f64,
}

/// A native objective the adversary maximises over the zonotope.
pub trait SeparationObjective {
    /// Evaluates `F` and `dF/dq` at an admissible mask.
    fn evaluate(&self, mask: &[f64]) -> Result<ObjectiveJet, String>;
}

/// The best mask found and its native value.
#[derive(Clone, Debug, PartialEq)]
pub struct LowerWitness {
    /// The mask, at all-on on the kept set.
    pub mask: Vec<f64>,
    /// `F` at the mask.
    pub value: f64,
    /// The objective's roundoff bound on `value`.
    pub value_roundoff: f64,
}

/// Why the ascent stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AscentTermination {
    /// The witness exceeds `epsilon` past its roundoff.
    CounterexampleCertified,
    /// The computed gap is at or below its roundoff bound.
    Stationary,
    /// The increase a step must certify is at or below the iterate's value resolution.
    ValueResolution,
    /// The trial mask rounds to the iterate.
    MaskResolution,
}

/// The outcome of one separation query.
#[derive(Clone, Debug, PartialEq)]
pub struct SeparationReport {
    /// The last iterate.
    pub witness: LowerWitness,
    /// `dF/dq` at the witness, from the same evaluation that produced its value.
    pub witness_gradient: MomentVector,
    /// The Frank-Wolfe gap at the witness.
    pub gap: f64,
    /// Its roundoff bound.
    pub gap_roundoff: f64,
    /// Why the ascent stopped.
    pub termination: AscentTermination,
    /// `F` at every accepted iterate, starting with the start mask.
    pub accepted_values: Vec<f64>,
    /// Native evaluations spent, including rejected trials.
    pub evaluations: usize,
    /// The strongest status the query proved about `sup_{Z_S} F` against `epsilon`.
    pub status: SeparationStatus,
}

/// A refused separation query.
#[derive(Clone, Debug, PartialEq)]
pub enum AdversaryError {
    /// The declared tolerance is not finite.
    NonFiniteEpsilon(f64),
    /// The start moves a control of the kept set off all-on.
    StartDeletesKeptControl { control: usize, value: f64 },
    /// The objective refused an evaluation.
    Objective { evaluation: usize, message: String },
    /// The objective returned a non-finite value or roundoff bound, or a negative bound.
    NonFiniteJet { evaluation: usize },
    /// The moment system refused the domain, a mask or a gradient.
    Geometry(MomentGeometryError),
    /// The evidence status refused the query's numbers.
    Evidence(EvidenceStatusError),
}

impl fmt::Display for AdversaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteEpsilon(epsilon) => {
                write!(f, "the declared tolerance {epsilon} is not finite")
            }
            Self::StartDeletesKeptControl { control, value } => write!(
                f,
                "start sets kept control {control} to {value}; kept controls stay at all-on"
            ),
            Self::Objective {
                evaluation,
                message,
            } => write!(f, "objective evaluation {evaluation} refused: {message}"),
            Self::NonFiniteJet { evaluation } => write!(
                f,
                "objective evaluation {evaluation} returned a non-finite or negative roundoff bound or value"
            ),
            Self::Geometry(error) => write!(f, "moment geometry refused: {error:?}"),
            Self::Evidence(error) => write!(f, "evidence status refused: {error}"),
        }
    }
}

impl std::error::Error for AdversaryError {}

/// A lower bound on the exact value: the subtraction, the band's product and sum, and the
/// final subtraction each round once.
fn certified_lower(value: f64, roundoff: f64) -> f64 {
    value - roundoff - accumulation_band(4, value.abs() + roundoff)
}

/// An upper bound on the exact value, by the same count as [`certified_lower`].
fn certified_upper(value: f64, roundoff: f64) -> f64 {
    value + roundoff + accumulation_band(4, value.abs() + roundoff)
}

/// Per-query scales of the free controls, computed once.
struct ControlScales {
    /// The declared interval of every control.
    intervals: Vec<(f64, f64)>,
    /// `|v_c|_2` over every part of a free control, zero on the kept set.
    generator_norm: Vec<f64>,
    /// `sum_{c free} max(|1 - lower_c|, |1 - upper_c|) |v_cj|` per moment coordinate.
    moment_magnitude: MomentVector,
    /// The most rounded operations on one term of `g . q(m)`: the deletion amount, the
    /// product, the moment sum over `C` controls, the pairing product and its sum over `P`
    /// coordinates.
    pairing_depth: usize,
    /// `gamma` for `R(m)`: `P` squares and sums and a square root per norm, a product per
    /// control and the sum over `C` controls.
    radius_growth: f64,
}

impl ControlScales {
    fn new(
        system: &MaskMomentSystem,
        domain: &MaskDomain,
        kept: &[bool],
    ) -> Result<Self, AdversaryError> {
        let controls = system.control_count();
        let coordinates: usize = system.blocks().iter().map(|block| block.dimension).sum();
        let mut intervals = Vec::with_capacity(controls);
        let mut generator_norm = vec![0.0; controls];
        let mut moment_magnitude = MomentVector {
            blocks: system
                .blocks()
                .iter()
                .map(|block| Array1::zeros(block.dimension))
                .collect(),
        };
        for control in 0..controls {
            let (lower, upper) =
                domain
                    .interval(control)
                    .ok_or(AdversaryError::Geometry(MomentGeometryError::ControlCount {
                        expected: controls,
                        found: domain.control_count(),
                    }))?;
            intervals.push((lower, upper));
            if kept[control] {
                continue;
            }
            let reach = (1.0 - lower).abs().max((1.0 - upper).abs());
            let mut squares = 0.0;
            for part in system.generator(control).unwrap_or(&[]) {
                squares += part.vector.dot(&part.vector);
                moment_magnitude.blocks[part.block].scaled_add(reach, &part.vector.mapv(f64::abs));
            }
            generator_norm[control] = squares.sqrt();
        }
        Ok(Self {
            intervals,
            generator_norm,
            moment_magnitude,
            pairing_depth: controls + coordinates + 2,
            radius_growth: accumulation_growth(controls + 2 * coordinates + 2),
        })
    }

    /// The Frank-Wolfe quantities at an iterate.
    fn geometry(
        &self,
        system: &MaskMomentSystem,
        domain: &MaskDomain,
        kept: &[bool],
        mask: &[f64],
        jet: &ObjectiveJet,
    ) -> Result<IterateGeometry, AdversaryError> {
        let gradient = &jet.moment_gradient;
        let support = system
            .support(domain, kept, gradient)
            .map_err(AdversaryError::Geometry)?;
        let moment = system.moment(domain, mask).map_err(AdversaryError::Geometry)?;
        let pairing: f64 = moment
            .blocks
            .iter()
            .zip(&gradient.blocks)
            .map(|(block, covector)| block.dot(covector))
            .sum();
        let pairing_magnitude: f64 = self
            .moment_magnitude
            .blocks
            .iter()
            .zip(&gradient.blocks)
            .map(|(column, covector)| {
                column
                    .iter()
                    .zip(covector.iter())
                    .map(|(magnitude, entry)| magnitude * entry.abs())
                    .sum::<f64>()
            })
            .sum();
        let gap = support.value - pairing;
        let radius = mask
            .iter()
            .zip(&self.intervals)
            .zip(&self.generator_norm)
            .map(|((&value, &(lower, upper)), &norm)| (value - lower).max(upper - value) * norm)
            .sum::<f64>()
            * (1.0 + self.radius_growth);
        // The support band contains the exact h at the computed gradient. The pairing band
        // covers g . q(m) at the rounded moment; one more band covers the subtraction. G is a
        // maximum of linear functions of g with slopes q' - q, |q' - q|_2 <= R, so a gradient
        // error of l2 norm delta_g moves it by at most delta_g R. The four nonnegative bounds
        // are summed with growth for their own four operations.
        let gap_roundoff = (support.band
            + accumulation_band(self.pairing_depth, pairing_magnitude)
            + accumulation_band(2, support.value.abs() + pairing.abs())
            + jet.gradient_roundoff * radius)
            * (1.0 + accumulation_growth(4));
        Ok(IterateGeometry {
            witness: support.witness,
            gap,
            gap_roundoff,
        })
    }

    /// `m + gamma (s - m)` toward the witness endpoints. The exact convex combination lies in
    /// the declared interval, so clamping removes only rounding.
    fn step_toward(&self, mask: &[f64], witness: &[WitnessEndpoint], step: f64) -> Vec<f64> {
        mask.iter()
            .zip(witness)
            .zip(&self.intervals)
            .map(|((&value, endpoint), &(lower, upper))| {
                let moved = match endpoint {
                    WitnessEndpoint::Kept => value,
                    WitnessEndpoint::Lower => lower + (1.0 - step) * (value - lower),
                    WitnessEndpoint::Upper => upper - (1.0 - step) * (upper - value),
                };
                moved.clamp(lower, upper)
            })
            .collect()
    }
}

/// The Frank-Wolfe quantities at one iterate, each with its derived roundoff.
struct IterateGeometry {
    witness: Vec<WitnessEndpoint>,
    gap: f64,
    gap_roundoff: f64,
}

fn evaluate_checked<O>(
    objective: &O,
    mask: &[f64],
    evaluations: &mut usize,
) -> Result<ObjectiveJet, AdversaryError>
where
    O: SeparationObjective + ?Sized,
{
    let evaluation = *evaluations;
    *evaluations += 1;
    let jet = objective
        .evaluate(mask)
        .map_err(|message| AdversaryError::Objective {
            evaluation,
            message,
        })?;
    let finite = jet.value.is_finite()
        && jet.value_roundoff.is_finite()
        && jet.value_roundoff >= 0.0
        && jet.gradient_roundoff.is_finite()
        && jet.gradient_roundoff >= 0.0;
    if !finite {
        return Err(AdversaryError::NonFiniteJet { evaluation });
    }
    Ok(jet)
}

enum StepOutcome {
    Accepted(Vec<f64>, ObjectiveJet),
    Unresolvable(AscentTermination),
}

/// What resolved the query, with the numbers its status needs.
enum Verdict {
    Counterexample,
    Open(AscentTermination),
}

/// Maximises `objective` over `Z_S` from `start` and classifies the supremum against the
/// declared `epsilon`.
///
/// `kept[c]` pins control `c` at all-on. `start` is an admissible mask with the kept set at
/// all-on; a replayed conflict mask warm-starts the query.
pub fn separate<O>(
    system: &MaskMomentSystem,
    domain: &MaskDomain,
    kept: &[bool],
    epsilon: f64,
    start: &[f64],
    objective: &O,
) -> Result<SeparationReport, AdversaryError>
where
    O: SeparationObjective + ?Sized,
{
    let controls = system.control_count();
    for found in [domain.control_count(), kept.len(), start.len()] {
        if found != controls {
            return Err(AdversaryError::Geometry(MomentGeometryError::ControlCount {
                expected: controls,
                found,
            }));
        }
    }
    if !epsilon.is_finite() {
        return Err(AdversaryError::NonFiniteEpsilon(epsilon));
    }
    let scales = ControlScales::new(system, domain, kept)?;
    for (control, ((&value, &is_kept), &(lower, upper))) in
        start.iter().zip(kept).zip(&scales.intervals).enumerate()
    {
        if !(lower <= value && value <= upper) {
            return Err(AdversaryError::Geometry(MomentGeometryError::MaskOutsideDomain {
                control,
                value,
                lower,
                upper,
            }));
        }
        if is_kept && value != 1.0 {
            return Err(AdversaryError::StartDeletesKeptControl { control, value });
        }
    }
    let mut evaluations = 0;
    let mut mask = start.to_vec();
    let mut jet = evaluate_checked(objective, &mask, &mut evaluations)?;
    let mut accepted_values = vec![jet.value];
    let (verdict, gap, gap_roundoff) = loop {
        let geometry = scales.geometry(system, domain, kept, &mask, &jet)?;
        let witness_lower = certified_lower(jet.value, jet.value_roundoff);
        if witness_lower > epsilon {
            break (Verdict::Counterexample, geometry.gap, geometry.gap_roundoff);
        }
        if geometry.gap <= geometry.gap_roundoff {
            let verdict = Verdict::Open(AscentTermination::Stationary);
            break (verdict, geometry.gap, geometry.gap_roundoff);
        }
        let lower_gap = geometry.gap - geometry.gap_roundoff;
        let resolution = jet.value_roundoff + 2.0 * UNIT_ROUNDOFF * jet.value.abs();
        let current_upper = certified_upper(jet.value, jet.value_roundoff);
        let mut step = 1.0;
        let outcome = loop {
            let required = 0.5 * step * lower_gap;
            if required <= resolution {
                break StepOutcome::Unresolvable(AscentTermination::ValueResolution);
            }
            let trial = scales.step_toward(&mask, &geometry.witness, step);
            if trial == mask {
                break StepOutcome::Unresolvable(AscentTermination::MaskResolution);
            }
            let trial_jet = evaluate_checked(objective, &trial, &mut evaluations)?;
            let increase =
                certified_lower(trial_jet.value, trial_jet.value_roundoff) - current_upper;
            if increase >= required {
                break StepOutcome::Accepted(trial, trial_jet);
            }
            step *= 0.5;
        };
        match outcome {
            StepOutcome::Accepted(trial, trial_jet) => {
                mask = trial;
                jet = trial_jet;
                accepted_values.push(jet.value);
            }
            StepOutcome::Unresolvable(reason) => {
                break (Verdict::Open(reason), geometry.gap, geometry.gap_roundoff);
            }
        }
    };
    let witness = LowerWitness {
        mask,
        value: jet.value,
        value_roundoff: jet.value_roundoff,
    };
    let region = SeparationRegion {
        domain: domain.clone(),
        kept: kept.to_vec(),
    };
    let (termination, status) = match verdict {
        Verdict::Counterexample => (
            AscentTermination::CounterexampleCertified,
            SeparationStatus::counterexample(
                witness.value,
                witness.value_roundoff,
                epsilon,
                witness.mask.clone(),
            ),
        ),
        Verdict::Open(reason) => (
            reason,
            SeparationStatus::unresolved(
                certified_lower(witness.value, witness.value_roundoff),
                f64::INFINITY,
                Extremum::Supremum,
                Some(witness.mask.clone()),
                region,
            ),
        ),
    };
    Ok(SeparationReport {
        witness,
        witness_gradient: jet.moment_gradient,
        gap,
        gap_roundoff,
        termination,
        accepted_values,
        evaluations,
        status: status.map_err(AdversaryError::Evidence)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter_decomposition::moments::{GeneratorPart, MomentBlock};
    use ndarray::{Array2, array};

    fn system_from_rows(rows: &Array2<f64>) -> MaskMomentSystem {
        MaskMomentSystem::new(
            vec![MomentBlock {
                dimension: rows.ncols(),
            }],
            rows.outer_iter()
                .map(|row| {
                    vec![GeneratorPart {
                        block: 0,
                        vector: row.to_owned(),
                    }]
                })
                .collect(),
        )
        .expect("well-formed generators")
    }

    fn unit_domain(controls: usize) -> MaskDomain {
        MaskDomain::new(vec![(0.0, 1.0); controls]).expect("unit intervals contain all-on")
    }

    /// `F(q) = (27/4) q_3 (1 - q_3)^2` with `v_c = e_c`, so `q_3 = 1 - m_3`: zero at both
    /// binary endpoints of `m_3`, maximum 1 at the interior `m_3 = 2/3`.
    struct InteriorPeak;

    impl InteriorPeak {
        fn value(deletion: f64) -> f64 {
            6.75 * deletion * (1.0 - deletion) * (1.0 - deletion)
        }
    }

    impl SeparationObjective for InteriorPeak {
        fn evaluate(&self, mask: &[f64]) -> Result<ObjectiveJet, String> {
            let deletion = 1.0 - mask[2];
            let value = Self::value(deletion);
            // The value expands into 6.75 (q - 2 q^2 + q^3), terms bounded by 6.75 q (1 + q)^2,
            // each carrying at most 8 rounded operations counting the rounding of q; the
            // derivative 6.75 (1 - 4 q + 3 q^2) has terms bounded by 6.75 (1 + q)(1 + 3 q) with at
            // most 8.
            let value_roundoff =
                accumulation_band(8, 6.75 * deletion * (1.0 + deletion) * (1.0 + deletion));
            let gradient_roundoff =
                accumulation_band(8, 6.75 * (1.0 + deletion) * (1.0 + 3.0 * deletion));
            Ok(ObjectiveJet {
                value,
                value_roundoff,
                moment_gradient: MomentVector {
                    blocks: vec![array![
                        0.0,
                        0.0,
                        6.75 * (1.0 - deletion) * (1.0 - 3.0 * deletion)
                    ]],
                },
                gradient_roundoff,
            })
        }
    }

    /// `F(q) = |M q|^2 / 2`: convex, zero with zero gradient at the all-on mask, so its maximum
    /// over the zonotope sits at a binary vertex.
    struct Quadratic {
        system: MaskMomentSystem,
        domain: MaskDomain,
        rows: Array2<f64>,
        map: Array2<f64>,
    }

    impl Quadratic {
        fn fixture() -> Self {
            let rows = array![
                [0.8, 0.1],
                [-0.3, 0.6],
                [0.5, -0.7],
                [0.2, 0.4],
                [-0.6, -0.2]
            ];
            let map = array![[1.0, -0.4], [-0.6, 0.9], [0.2, 0.7]];
            Self {
                system: system_from_rows(&rows),
                domain: unit_domain(rows.nrows()),
                rows,
                map,
            }
        }

        fn value_at_moment(&self, moment: &Array1<f64>) -> f64 {
            let residual = self.map.dot(moment);
            0.5 * residual.dot(&residual)
        }

        fn moment(&self, mask: &[f64]) -> Result<Array1<f64>, String> {
            self.system
                .moment(&self.domain, mask)
                .map_err(|error| format!("{error:?}"))?
                .blocks
                .into_iter()
                .next()
                .ok_or_else(|| "one block".to_string())
        }

        /// The exhaustive binary max over the free controls (test only): the true supremum
        /// over the zonotope, since a convex function peaks at a vertex.
        fn exhaustive_max(&self, kept: &[bool]) -> f64 {
            let controls = self.rows.nrows();
            (0..1usize << controls)
                .filter(|bits| (0..controls).all(|c| !kept[c] || (bits >> c) & 1 == 0))
                .map(|bits| {
                    let mask: Vec<f64> = (0..controls)
                        .map(|c| if (bits >> c) & 1 == 1 { 0.0 } else { 1.0 })
                        .collect();
                    self.value_at_moment(&self.moment(&mask).expect("endpoint mask admissible"))
                })
                .fold(f64::NEG_INFINITY, f64::max)
        }
    }

    impl SeparationObjective for Quadratic {
        fn evaluate(&self, mask: &[f64]) -> Result<ObjectiveJet, String> {
            let controls = self.rows.nrows();
            let dimension = self.rows.ncols();
            let classes = self.map.nrows();
            let moment = self.moment(mask)?;
            let residual = self.map.dot(&moment);
            let gradient = self.map.t().dot(&residual);
            // Magnitudes of the expanded terms: |q|_abs = |V|^T |1 - m|, |r|_abs = |M| |q|_abs,
            // |g|_abs = |M|^T |r|_abs.
            let deletion_abs: Array1<f64> = mask.iter().map(|value| (1.0 - value).abs()).collect();
            let abs_map = self.map.mapv(f64::abs);
            let moment_abs = self.rows.mapv(f64::abs).t().dot(&deletion_abs);
            let residual_abs = abs_map.dot(&moment_abs);
            let gradient_abs = abs_map.t().dot(&residual_abs);
            let value_roundoff = accumulation_band(
                controls + dimension + classes + 2,
                0.5 * residual_abs.dot(&residual_abs),
            );
            let gradient_roundoff = accumulation_band(
                controls + 2 * dimension + classes + 3,
                gradient_abs.dot(&gradient_abs).sqrt(),
            ) * (1.0 + accumulation_growth(dimension + 2));
            Ok(ObjectiveJet {
                value: 0.5 * residual.dot(&residual),
                value_roundoff,
                moment_gradient: MomentVector {
                    blocks: vec![gradient],
                },
                gradient_roundoff,
            })
        }
    }

    #[test]
    fn interior_witness_is_certified_where_binary_endpoints_see_nothing_2951() {
        let system = system_from_rows(&Array2::<f64>::eye(3));
        let domain = unit_domain(3);
        let kept = [false; 3];
        let epsilon = 0.5;
        // Negative control: the binary endpoint masks of the only live control give 0.
        for endpoint in [0.0, 1.0] {
            let value = InteriorPeak::value(1.0 - endpoint);
            assert!(value <= epsilon, "endpoint {endpoint} gives {value}");
        }
        let report = separate(&system, &domain, &kept, epsilon, &[1.0; 3], &InteriorPeak)
            .expect("interior query");
        assert_eq!(report.termination, AscentTermination::CounterexampleCertified);
        assert!(matches!(report.status, EvidenceStatus::Counterexample { .. }));
        assert!(report.status.refutes_at_most(epsilon));
        let live = report.witness.mask[2];
        assert!(live > 0.0 && live < 1.0, "interior mask, got {live}");
        assert_eq!(report.witness.mask[0], 1.0);
        assert_eq!(report.witness.mask[1], 1.0);
        assert_eq!(report.status.upper_bound(), None);
    }

    #[test]
    fn ascent_is_monotone_and_never_overclaims_the_interior_peak_2951() {
        let system = system_from_rows(&Array2::<f64>::eye(3));
        let domain = unit_domain(3);
        let report = separate(&system, &domain, &[false; 3], 1.5, &[1.0; 3], &InteriorPeak)
            .expect("interior query");
        assert!(matches!(
            report.termination,
            AscentTermination::Stationary
                | AscentTermination::ValueResolution
                | AscentTermination::MaskResolution
        ));
        assert!(
            report
                .accepted_values
                .windows(2)
                .all(|pair| pair[1] > pair[0])
        );
        assert!(matches!(report.status, EvidenceStatus::Unresolved { .. }));
        assert_eq!(report.status.upper_bound(), None);
        // The exact supremum is 1. The first accepted trial is the exact mask m_3 = 3/4, and
        // the accepted values increase strictly, so the reported lower side is at least that
        // iterate's certified value.
        let lower = report.status.lower_bound().expect("a finite lower witness");
        assert!(lower <= 1.0, "lower witness {lower} above the exact supremum");
        let first_accepted = InteriorPeak
            .evaluate(&[1.0, 1.0, 0.75])
            .expect("jet at m_3 = 3/4");
        assert_eq!(report.accepted_values[1], first_accepted.value);
        assert!(lower >= certified_lower(first_accepted.value, first_accepted.value_roundoff));
    }

    #[test]
    fn lower_witness_never_exceeds_the_exhaustive_vertex_maximum_2951() {
        let fixture = Quadratic::fixture();
        let kept = [false, true, false, false, false];
        let f_max = fixture.exhaustive_max(&kept);
        let start = [0.5, 1.0, 0.5, 0.5, 0.5];
        let report = separate(
            &fixture.system,
            &fixture.domain,
            &kept,
            2.0 * f_max,
            &start,
            &fixture,
        )
        .expect("quadratic query");
        assert_eq!(report.witness.mask[1], 1.0);
        let lower = report.status.lower_bound().expect("a finite lower witness");
        assert!(lower <= f_max, "lower witness {lower} above the vertex maximum {f_max}");
        assert_eq!(report.status.upper_bound(), None);
        assert!(
            report
                .accepted_values
                .windows(2)
                .all(|pair| pair[1] > pair[0])
        );
    }

    #[test]
    fn start_moving_a_kept_control_is_refused_2951() {
        let fixture = Quadratic::fixture();
        let kept = [false, true, false, false, false];
        let start = [1.0, 0.75, 1.0, 1.0, 1.0];
        let refused = separate(&fixture.system, &fixture.domain, &kept, 1.0, &start, &fixture);
        assert_eq!(
            refused,
            Err(AdversaryError::StartDeletesKeptControl {
                control: 1,
                value: 0.75
            })
        );
        // Positive control: the same start with the control free is accepted.
        let accepted = separate(&fixture.system, &fixture.domain, &[false; 5], 1.0, &start, &fixture);
        assert!(accepted.is_ok());
    }

    #[test]
    fn quadratic_gradient_matches_the_exact_central_difference_2951() {
        let fixture = Quadratic::fixture();
        let mask = [0.3, 0.7, 0.1, 0.9, 0.4];
        let jet = fixture.evaluate(&mask).expect("jet");
        let moment = fixture.moment(&mask).expect("admissible");
        let abs_map = fixture.map.mapv(f64::abs);
        // F is quadratic in q, so F(q + e_j) - F(q - e_j) = 2 g_j exactly at unit width: the
        // central difference has no truncation error. What remains is rounding. Each probe value
        // carries at most K + p + 3 operations on terms bounded by |M| (|q| + e_j); each probe
        // coordinate q_j +- 1 rounds by u (|q_j| + 1), which moves F by at most
        // |g|_abs,j u (|q_j| + 1); the analytic gradient is within delta_g of g at the exact
        // moment, and the probes sit at the rounded moment, which moves g by one more delta_g.
        let classes = fixture.map.nrows();
        let dimension = moment.len();
        let gradient = &jet.moment_gradient.blocks[0];
        for coordinate in 0..dimension {
            let mut plus = moment.clone();
            let mut minus = moment.clone();
            plus[coordinate] += 1.0;
            minus[coordinate] -= 1.0;
            let difference = 0.5 * (fixture.value_at_moment(&plus) - fixture.value_at_moment(&minus));
            let mut probe_abs = moment.mapv(f64::abs);
            probe_abs[coordinate] += 1.0;
            let residual_abs = abs_map.dot(&probe_abs);
            let gradient_abs = abs_map.t().dot(&residual_abs);
            let tolerance = accumulation_band(
                classes + dimension + 3,
                0.5 * residual_abs.dot(&residual_abs),
            ) + accumulation_band(2, gradient_abs[coordinate] * probe_abs[coordinate])
                + 2.0 * jet.gradient_roundoff;
            assert!(
                (difference - gradient[coordinate]).abs() <= tolerance,
                "coordinate {coordinate}: central difference {difference} vs analytic {} (tolerance {tolerance})",
                gradient[coordinate]
            );
        }
    }
}
