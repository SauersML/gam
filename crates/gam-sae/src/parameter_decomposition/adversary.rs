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
//! * a certified uniform bound, `U <= epsilon` (below);
//! * stationarity, the computed gap at or below its own roundoff bound;
//! * resolution: the increase a step must certify is at or below the value resolution
//!   of the iterate, or the trial mask rounds to the iterate.
//!
//! Every accepted step raises the computed value strictly, so the loop is finite.
//!
//! # What is reported
//!
//! The lower witness is the last iterate with its mask: the ascent is monotone, so it
//! is the best value seen. It bounds the supremum from below only; the ascent is local.
//! An upper bound is reported only when the objective states a Lipschitz constant `L`
//! of `dF` over the full zonotope of the declared domain: for every `q'` in `Z_S`,
//!
//! ```text
//! F(q') <= F(q) + g . (q' - q) + (L / 2) |q' - q|_2^2 <= F(q) + G(q) + (L / 2) R(m)^2
//! R(m)  = sum_{c not in S} max(m_c - lower_c, upper_c - m_c) |v_c|_2
//! ```
//!
//! since `q' - q = sum_c (m_c - m'_c) v_c`. Without `L` the upper side is underived. A
//! witness above a held bound refutes the stated constant, and the query refuses instead
//! of reporting an inverted interval.
//!
//! # Support search
//!
//! [`ZonotopeSeparationOracle`] is mpd-supports' [`SeparationOracle`] over this ascent, so
//! the minimum-code support search runs P12's CEGAR with the nonlinear separation. A support
//! query starts from the last refuting witness with the kept set reset to all-on, or else from
//! the center of the free box. It never starts at all-on: a distance objective is minimal
//! there with every derivative zero (P16), so the ascent would be stationary at once. When the
//! support keeps every control, the region is the single point all-on; it is evaluated natively
//! and reported exact over that one point. A point evaluation is a counterexample when it
//! refutes epsilon, and otherwise exact over its one mask.
//!
//! # Covering
//!
//! [`certify_by_covering`] bounds `sup_{Z_S} F` over sub-boxes `K = prod_c [a_c, b_c]` of the
//! declared box (#2951 comment 5717715578). With center `m0` and half-width generators
//! `w_c = ((b_c - a_c) / 2) v_c`,
//!
//! ```text
//! { q(m) - q(m0) : m in K } = { sum_c s_c w_c : s in [-1, 1]^C } = { q'(s') - q'(0) : s' in [-1, 1]^C }
//! ```
//!
//! for the half-width system over the signed box `[-1, 1]^C`. So the cell's gap and radius are
//! that system's Frank-Wolfe gap and radius at mask 0, and
//! `sup_K F <= F(m0) + G_K + (L / 2) R_K^2`. The computed center `m0` is the midpoint rounded,
//! so each half-width is the larger distance from `m0` to an endpoint, rounded up: the signed
//! box then covers the cell exactly, whether or not the midpoint was representable. The search
//! is depth-first:
//! * a cell is pruned when its bound is at most epsilon;
//! * otherwise it splits on the control with the largest first-order decrease of that bound,
//!   `h_c (|g . v_c| + L R_K |v_c|_2)`, since `dU/dh_c = |g . v_c| + L R_K |v_c|_2`;
//! * a cell whose split midpoint rounds to an endpoint while its bound still exceeds epsilon is
//!   unresolved.
//!
//! A certified counterexample at a center returns at once. Every center also lies in each cell
//! the search split to reach it, so a certified center value above an enclosing cell's bound
//! refutes the stated constant. Memory is one interval vector and the split stack, and the
//! covering is finite through the resolution floor.

use super::moments::{
    GeneratorPart, MaskDomain, MaskMomentSystem, MomentGeometryError, MomentVector, WitnessEndpoint,
};
use super::supports::{
    ComponentSet, EvidenceStatus, EvidenceStatusError, ExactBasis, Extremum, SeparationOracle,
};
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

/// A stated Lipschitz constant of the moment gradient.
#[derive(Clone, Debug, PartialEq)]
pub struct SmoothnessCertificate {
    /// `L` with `|dF(q) - dF(q')|_2 <= L |q - q'|_2` for all `q, q'` in the zonotope of the
    /// declared domain with nothing kept, which contains every `Z_S`.
    pub gradient_lipschitz: f64,
    /// Where `L` was derived.
    pub derivation: String,
}

/// A native objective the adversary maximises over the zonotope.
pub trait SeparationObjective {
    /// Evaluates `F` and `dF/dq` at an admissible mask.
    fn evaluate(&self, mask: &[f64]) -> Result<ObjectiveJet, String>;

    /// The Lipschitz constant of `dF`, when one is derived.
    fn smoothness(&self) -> Option<SmoothnessCertificate>;
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

/// The smallest smoothness upper bound over the iterates, each of which is valid.
#[derive(Clone, Debug, PartialEq)]
pub struct SmoothnessUpperBound {
    /// `F(q) + G(q) + (L / 2) R(m)^2`, rounded up past every roundoff bound.
    pub value: f64,
    /// The roundoff already folded into `value`, kept for audit.
    pub numerical_error: f64,
    /// The iterate the bound was taken at.
    pub mask: Vec<f64>,
    /// The stated constant the bound rests on.
    pub certificate: SmoothnessCertificate,
}

/// Why the ascent stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AscentTermination {
    /// The witness exceeds `epsilon` past its roundoff.
    CounterexampleCertified,
    /// The smoothness upper bound is at or below `epsilon`.
    UniformBoundCertified,
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
    /// The Frank-Wolfe gap at the witness.
    pub gap: f64,
    /// Its roundoff bound.
    pub gap_roundoff: f64,
    /// Present only when the objective states a smoothness constant.
    pub upper_bound: Option<SmoothnessUpperBound>,
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
    /// The stated Lipschitz constant is negative or not finite.
    InvalidCertificate { gradient_lipschitz: f64 },
    /// A certified witness value exceeds a smoothness bound, so the stated constant is
    /// false.
    CertificateRefuted { witness_lower: f64, bound: f64 },
    /// The moment system refused the domain, a mask or a gradient.
    Geometry(MomentGeometryError),
    /// The evidence status refused the query's numbers.
    Evidence(EvidenceStatusError),
    /// A covering needs a stated smoothness constant to bound any cell.
    MissingCertificate,
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
            Self::InvalidCertificate { gradient_lipschitz } => write!(
                f,
                "stated gradient Lipschitz constant {gradient_lipschitz} is negative or not finite"
            ),
            Self::CertificateRefuted {
                witness_lower,
                bound,
            } => write!(
                f,
                "a witness certified at {witness_lower} exceeds the smoothness bound {bound}; the stated constant is false"
            ),
            Self::Geometry(error) => write!(f, "moment geometry refused: {error:?}"),
            Self::Evidence(error) => write!(f, "evidence status refused: {error}"),
            Self::MissingCertificate => write!(
                f,
                "a covering needs a stated gradient Lipschitz constant; none is derived for this objective"
            ),
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
            radius,
            pairing_magnitude,
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
    /// `R(m)`, rounded up.
    radius: f64,
    /// `sum_j |g_j| sum_{c free} max(|1 - lower_c|, |1 - upper_c|) |v_cj|`, the magnitude the pairing
    /// band is taken over.
    pairing_magnitude: f64,
}

/// `F(q) + G(q) + (L / 2) R(m)^2`, rounded up, with the roundoff folded into it: the pad
/// covers the four additions and the product of the final sum.
fn smoothness_upper_bound(
    jet: &ObjectiveJet,
    geometry: &IterateGeometry,
    gradient_lipschitz: f64,
) -> (f64, f64) {
    let quadratic = 0.5 * gradient_lipschitz * geometry.radius * geometry.radius;
    let gap_upper = (geometry.gap + geometry.gap_roundoff).max(0.0);
    let pad = accumulation_band(5, jet.value.abs() + jet.value_roundoff + gap_upper + quadratic);
    let value = jet.value + jet.value_roundoff + gap_upper + quadratic + pad;
    (value, jet.value_roundoff + geometry.gap_roundoff + pad)
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
    UniformBound { upper: f64, numerical_error: f64 },
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
    let certificate = objective.smoothness();
    if let Some(stated) = &certificate {
        if !(stated.gradient_lipschitz.is_finite() && stated.gradient_lipschitz >= 0.0) {
            return Err(AdversaryError::InvalidCertificate {
                gradient_lipschitz: stated.gradient_lipschitz,
            });
        }
    }
    let mut evaluations = 0;
    let mut mask = start.to_vec();
    let mut jet = evaluate_checked(objective, &mask, &mut evaluations)?;
    let mut accepted_values = vec![jet.value];
    let mut upper_bound: Option<SmoothnessUpperBound> = None;
    let (verdict, gap, gap_roundoff) = loop {
        let geometry = scales.geometry(system, domain, kept, &mask, &jet)?;
        let witness_lower = certified_lower(jet.value, jet.value_roundoff);
        if let Some(stated) = &certificate {
            let (bound, numerical_error) =
                smoothness_upper_bound(&jet, &geometry, stated.gradient_lipschitz);
            let tighter = match &upper_bound {
                Some(held) => bound < held.value,
                None => true,
            };
            if tighter {
                upper_bound = Some(SmoothnessUpperBound {
                    value: bound,
                    numerical_error,
                    mask: mask.clone(),
                    certificate: stated.clone(),
                });
            }
        }
        if let Some(held) = &upper_bound {
            if witness_lower > held.value {
                return Err(AdversaryError::CertificateRefuted {
                    witness_lower,
                    bound: held.value,
                });
            }
        }
        if witness_lower > epsilon {
            break (Verdict::Counterexample, geometry.gap, geometry.gap_roundoff);
        }
        if let Some(held) = &upper_bound {
            if held.value <= epsilon {
                let verdict = Verdict::UniformBound {
                    upper: held.value,
                    numerical_error: held.numerical_error,
                };
                break (verdict, geometry.gap, geometry.gap_roundoff);
            }
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
        Verdict::UniformBound {
            upper,
            numerical_error,
        } => (
            AscentTermination::UniformBoundCertified,
            SeparationStatus::uniform_bound(upper, numerical_error, region),
        ),
        Verdict::Open(reason) => (
            reason,
            SeparationStatus::unresolved(
                certified_lower(witness.value, witness.value_roundoff),
                upper_bound
                    .as_ref()
                    .map_or(f64::INFINITY, |held| held.value),
                Extremum::Supremum,
                Some(witness.mask.clone()),
                region,
            ),
        ),
    };
    Ok(SeparationReport {
        witness,
        gap,
        gap_roundoff,
        upper_bound,
        termination,
        accepted_values,
        evaluations,
        status: status.map_err(AdversaryError::Evidence)?,
    })
}

/// The region an oracle status speaks about: `Z_S` for a support query, or the one-mask family
/// of a point evaluation.
#[derive(Clone, Debug, PartialEq)]
pub enum OracleRegion {
    Zonotope(SeparationRegion),
    Mask(Vec<f64>),
}

/// The evidence the oracle returns to the support search.
pub type OracleStatus = EvidenceStatus<Vec<f64>, OracleRegion>;

/// mpd-supports' separation oracle over the zonotope ascent (#2951 P7, P12).
pub struct ZonotopeSeparationOracle<'a, O: SeparationObjective + ?Sized> {
    system: &'a MaskMomentSystem,
    domain: &'a MaskDomain,
    epsilon: f64,
    objective: &'a O,
    last_witness: Option<Vec<f64>>,
}

impl<'a, O: SeparationObjective + ?Sized> ZonotopeSeparationOracle<'a, O> {
    /// `epsilon` is the declared tolerance. Every status is classified against it, so the
    /// support search runs at the same number.
    pub fn new(
        system: &'a MaskMomentSystem,
        domain: &'a MaskDomain,
        epsilon: f64,
        objective: &'a O,
    ) -> Result<Self, AdversaryError> {
        if domain.control_count() != system.control_count() {
            return Err(AdversaryError::Geometry(MomentGeometryError::ControlCount {
                expected: system.control_count(),
                found: domain.control_count(),
            }));
        }
        if !epsilon.is_finite() {
            return Err(AdversaryError::NonFiniteEpsilon(epsilon));
        }
        Ok(Self {
            system,
            domain,
            epsilon,
            objective,
            last_witness: None,
        })
    }

    /// The declared tolerance.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    fn interval(&self, control: usize) -> Result<(f64, f64), AdversaryError> {
        self.domain
            .interval(control)
            .ok_or(AdversaryError::Geometry(MomentGeometryError::ControlCount {
                expected: self.system.control_count(),
                found: self.domain.control_count(),
            }))
    }

    /// The last refuting witness with the kept set reset to all-on, else the center of the free
    /// box.
    fn start_mask(&self, kept: &[bool]) -> Result<Vec<f64>, AdversaryError> {
        (0..kept.len())
            .map(|control| {
                if kept[control] {
                    return Ok(1.0);
                }
                if let Some(witness) = &self.last_witness {
                    return Ok(witness[control]);
                }
                let (lower, upper) = self.interval(control)?;
                Ok(lower + 0.5 * (upper - lower))
            })
            .collect()
    }

    /// A native evaluation at one admissible mask, classified against epsilon.
    fn point_status(&self, mask: &[f64], region: OracleRegion) -> Result<OracleStatus, AdversaryError> {
        let controls = self.system.control_count();
        if mask.len() != controls {
            return Err(AdversaryError::Geometry(MomentGeometryError::ControlCount {
                expected: controls,
                found: mask.len(),
            }));
        }
        for (control, &value) in mask.iter().enumerate() {
            let (lower, upper) = self.interval(control)?;
            if !(lower <= value && value <= upper) {
                return Err(AdversaryError::Geometry(MomentGeometryError::MaskOutsideDomain {
                    control,
                    value,
                    lower,
                    upper,
                }));
            }
        }
        let mut evaluations = 0;
        let jet = evaluate_checked(self.objective, mask, &mut evaluations)?;
        let status = if certified_lower(jet.value, jet.value_roundoff) > self.epsilon {
            OracleStatus::counterexample(jet.value, jet.value_roundoff, self.epsilon, mask.to_vec())
        } else {
            OracleStatus::exact(
                jet.value,
                jet.value_roundoff,
                ExactBasis::Exhaustive { cardinality: 1 },
                Some(mask.to_vec()),
                region,
            )
        };
        status.map_err(AdversaryError::Evidence)
    }
}

/// Rebuilds a query's status over the oracle's region type, through the evidence constructors.
fn over_oracle_region(
    status: SeparationStatus,
    region: OracleRegion,
) -> Result<OracleStatus, AdversaryError> {
    let rebuilt = match status {
        EvidenceStatus::Exact {
            value,
            numerical_error,
            basis,
            witness,
            ..
        } => OracleStatus::exact(value, numerical_error, basis, witness, region),
        EvidenceStatus::UniformBound {
            upper,
            numerical_error,
            ..
        } => OracleStatus::uniform_bound(upper, numerical_error, region),
        EvidenceStatus::StatisticalEstimate {
            estimate,
            standard_error,
            samples,
            ..
        } => OracleStatus::statistical_estimate(estimate, standard_error, samples, region),
        EvidenceStatus::Counterexample {
            value,
            numerical_error,
            threshold,
            witness,
            ..
        } => OracleStatus::counterexample(value, numerical_error, threshold, witness),
        EvidenceStatus::Unresolved {
            lower,
            upper,
            extremum,
            witness,
            ..
        } => OracleStatus::unresolved(lower, upper, extremum, witness, region),
    };
    rebuilt.map_err(AdversaryError::Evidence)
}

impl<O: SeparationObjective + ?Sized> SeparationOracle for ZonotopeSeparationOracle<'_, O> {
    type Mask = Vec<f64>;
    type Domain = OracleRegion;
    type Error = AdversaryError;

    fn components(&self) -> usize {
        self.system.control_count()
    }

    fn perturbed_components(&self, mask: &Vec<f64>) -> Vec<usize> {
        mask.iter()
            .enumerate()
            .filter_map(|(control, value)| (*value != 1.0).then_some(control))
            .collect()
    }

    fn separate(&mut self, support: &ComponentSet) -> Result<OracleStatus, AdversaryError> {
        let controls = self.system.control_count();
        if support.components() != controls {
            return Err(AdversaryError::Geometry(MomentGeometryError::ControlCount {
                expected: controls,
                found: support.components(),
            }));
        }
        let mut kept = vec![false; controls];
        for &member in support.members() {
            kept[member] = true;
        }
        let region = OracleRegion::Zonotope(SeparationRegion {
            domain: self.domain.clone(),
            kept: kept.clone(),
        });
        if support.len() == controls {
            // The region is the single point all-on.
            return self.point_status(&vec![1.0; controls], region);
        }
        let start = self.start_mask(&kept)?;
        let report = separate(self.system, self.domain, &kept, self.epsilon, &start, self.objective)?;
        if report.status.refutes_at_most(self.epsilon) {
            self.last_witness = Some(report.witness.mask.clone());
        }
        over_oracle_region(report.status, region)
    }

    fn evaluate(&mut self, mask: &Vec<f64>) -> Result<OracleStatus, AdversaryError> {
        self.point_status(mask, OracleRegion::Mask(mask.clone()))
    }
}

/// A Lipschitz covering of `Z_S` by sub-boxes of the declared domain.
#[derive(Clone, Debug, PartialEq)]
pub struct CoveringReport {
    /// The best certified cell center.
    pub witness: LowerWitness,
    /// Cells whose bound is at most epsilon.
    pub pruned_cells: usize,
    /// Cells at mask resolution whose bound exceeds epsilon.
    pub unresolved_cells: usize,
    /// Native evaluations, one per visited cell.
    pub evaluations: usize,
    /// UniformBound when every cell is pruned, Counterexample when a center exceeds epsilon,
    /// otherwise Unresolved with the largest unresolved bound as its upper side.
    pub status: SeparationStatus,
}

/// One pending split on the depth-first stack. The frames on the stack are the cells enclosing
/// the current one.
struct SplitFrame {
    control: usize,
    parent: (f64, f64),
    midpoint: f64,
    right_visited: bool,
    /// The smallest bound over this split's cell and every cell enclosing it.
    enclosing_bound: f64,
}

/// The half-width system of a cell around its computed center, over the signed unit box.
///
/// A half-width is `max(m0_c - a_c, b_c - m0_c)`, rounded up past the subtraction, so
/// `[m0_c - h_c, m0_c + h_c]` contains `[a_c, b_c]` for the center as computed. The midpoint
/// `a + (b - a) / 2` can round away from the exact midpoint by an absolute `u |m0|`, which a
/// relative band on `(b - a) / 2` does not cover once the cell is narrow. A degenerate cell
/// (a kept control) keeps half-width zero.
fn half_width_system(
    system: &MaskMomentSystem,
    cell: &[(f64, f64)],
    center: &[f64],
) -> Result<(MaskMomentSystem, MaskDomain), AdversaryError> {
    let generators: Vec<Vec<GeneratorPart>> = cell
        .iter()
        .zip(center)
        .enumerate()
        .map(|(control, (&(lower, upper), &middle))| {
            let reach = (middle - lower).max(upper - middle);
            let half_width = if reach > 0.0 { reach.next_up() } else { 0.0 };
            system
                .generator(control)
                .unwrap_or(&[])
                .iter()
                .map(|part| GeneratorPart {
                    block: part.block,
                    vector: &part.vector * half_width,
                })
                .collect()
        })
        .collect();
    let scaled =
        MaskMomentSystem::new(system.blocks().to_vec(), generators).map_err(AdversaryError::Geometry)?;
    let signed =
        MaskDomain::new(vec![(-1.0, 1.0); cell.len()]).map_err(AdversaryError::Geometry)?;
    Ok((scaled, signed))
}

/// Bounds `sup_{Z_S} F` by a depth-first Lipschitz covering and classifies it against epsilon.
///
/// `kept[c]` pins control `c` at all-on. The objective must state a smoothness constant, since
/// no cell has a bound without one.
pub fn certify_by_covering<O>(
    system: &MaskMomentSystem,
    domain: &MaskDomain,
    kept: &[bool],
    epsilon: f64,
    objective: &O,
) -> Result<CoveringReport, AdversaryError>
where
    O: SeparationObjective + ?Sized,
{
    let controls = system.control_count();
    for found in [domain.control_count(), kept.len()] {
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
    let stated = objective
        .smoothness()
        .ok_or(AdversaryError::MissingCertificate)?;
    if !(stated.gradient_lipschitz.is_finite() && stated.gradient_lipschitz >= 0.0) {
        return Err(AdversaryError::InvalidCertificate {
            gradient_lipschitz: stated.gradient_lipschitz,
        });
    }
    let root = ControlScales::new(system, domain, kept)?;
    let mut cell: Vec<(f64, f64)> = root
        .intervals
        .iter()
        .zip(kept)
        .map(|(&interval, &is_kept)| if is_kept { (1.0, 1.0) } else { interval })
        .collect();
    let region = SeparationRegion {
        domain: domain.clone(),
        kept: kept.to_vec(),
    };
    let cell_center = |intervals: &[(f64, f64)]| -> Vec<f64> {
        intervals
            .iter()
            .map(|&(lower, upper)| (lower + 0.5 * (upper - lower)).clamp(lower, upper))
            .collect()
    };
    let mut evaluations = 0;
    // The root cell is evaluated before the loop, so the best witness always holds a native
    // evaluation.
    let root_center = cell_center(&cell);
    let root_jet = evaluate_checked(objective, &root_center, &mut evaluations)?;
    let mut best = LowerWitness {
        mask: root_center.clone(),
        value: root_jet.value,
        value_roundoff: root_jet.value_roundoff,
    };
    let mut pending = Some((root_center, root_jet));
    let mut stack: Vec<SplitFrame> = Vec::new();
    let mut pruned_cells = 0;
    let mut unresolved_cells = 0;
    let mut pruned_upper = f64::NEG_INFINITY;
    let mut pruned_error = 0.0_f64;
    let mut unresolved_upper = f64::NEG_INFINITY;
    loop {
        let (center, jet) = match pending.take() {
            Some(first) => first,
            None => {
                let center = cell_center(&cell);
                let jet = evaluate_checked(objective, &center, &mut evaluations)?;
                (center, jet)
            }
        };
        let witness_lower = certified_lower(jet.value, jet.value_roundoff);
        if witness_lower > certified_lower(best.value, best.value_roundoff) {
            best = LowerWitness {
                mask: center.clone(),
                value: jet.value,
                value_roundoff: jet.value_roundoff,
            };
        }
        // The center lies in every cell on the stack, and each of their bounds holds over its
        // whole cell if the stated constant is true. Its own cell's bound adds nonnegative terms
        // to F(m0), so only an enclosing cell can refute.
        if let Some(frame) = stack.last() {
            if witness_lower > frame.enclosing_bound {
                return Err(AdversaryError::CertificateRefuted {
                    witness_lower,
                    bound: frame.enclosing_bound,
                });
            }
        }
        if witness_lower > epsilon {
            let status = SeparationStatus::counterexample(
                jet.value,
                jet.value_roundoff,
                epsilon,
                center.clone(),
            )
            .map_err(AdversaryError::Evidence)?;
            return Ok(CoveringReport {
                witness: LowerWitness {
                    mask: center,
                    value: jet.value,
                    value_roundoff: jet.value_roundoff,
                },
                pruned_cells,
                unresolved_cells,
                evaluations,
                status,
            });
        }
        let (scaled, signed) = half_width_system(system, &cell, &center)?;
        let scales = ControlScales::new(&scaled, &signed, kept)?;
        let origin = vec![0.0; controls];
        let mut geometry = scales.geometry(&scaled, &signed, kept, &origin, &jet)?;
        // The support band covers the stored half-width generators. The half-width is rounded up,
        // so it covers the cell exactly, and only the product w_c = h_c v_c rounds:
        // |fl(w) - w| <= gamma_1 |w| per entry, so the exact cell gap can differ by
        // gamma_1 sum_j |g_j| sum_c |w_cj|: half the pairing magnitude, which was taken at reach 2.
        // The radius takes the same growth.
        geometry.gap_roundoff += accumulation_band(1, 0.5 * geometry.pairing_magnitude);
        geometry.radius *= 1.0 + accumulation_growth(1);
        let (bound, numerical_error) =
            smoothness_upper_bound(&jet, &geometry, stated.gradient_lipschitz);
        if bound <= epsilon {
            pruned_cells += 1;
            pruned_upper = pruned_upper.max(bound);
            pruned_error = pruned_error.max(numerical_error);
        } else {
            let split = (0..controls)
                .filter(|&control| !kept[control])
                .map(|control| {
                    let (lower, upper) = cell[control];
                    let half_width = 0.5 * (upper - lower);
                    let projection: f64 = system
                        .generator(control)
                        .unwrap_or(&[])
                        .iter()
                        .map(|part| part.vector.dot(&jet.moment_gradient.blocks[part.block]))
                        .sum();
                    let decrease = half_width
                        * (projection.abs()
                            + stated.gradient_lipschitz * geometry.radius * root.generator_norm[control]);
                    (control, decrease)
                })
                .filter(|&(control, decrease)| {
                    let (lower, upper) = cell[control];
                    let midpoint = lower + 0.5 * (upper - lower);
                    decrease > 0.0 && lower < midpoint && midpoint < upper
                })
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .map(|pair| pair.0);
            match split {
                Some(control) => {
                    let (lower, upper) = cell[control];
                    let midpoint = lower + 0.5 * (upper - lower);
                    let enclosing_bound = stack
                        .last()
                        .map_or(bound, |frame| frame.enclosing_bound.min(bound));
                    stack.push(SplitFrame {
                        control,
                        parent: (lower, upper),
                        midpoint,
                        right_visited: false,
                        enclosing_bound,
                    });
                    cell[control] = (lower, midpoint);
                    continue;
                }
                None => {
                    unresolved_cells += 1;
                    unresolved_upper = unresolved_upper.max(bound);
                }
            }
        }
        // Backtrack to the next unvisited right child.
        loop {
            match stack.last_mut() {
                None => break,
                Some(frame) if !frame.right_visited => {
                    frame.right_visited = true;
                    cell[frame.control] = (frame.midpoint, frame.parent.1);
                    break;
                }
                Some(frame) => {
                    cell[frame.control] = frame.parent;
                    stack.pop();
                }
            }
        }
        if stack.is_empty() {
            break;
        }
    }
    let status = if unresolved_cells == 0 {
        SeparationStatus::uniform_bound(pruned_upper, pruned_error, region)
    } else {
        SeparationStatus::unresolved(
            certified_lower(best.value, best.value_roundoff),
            unresolved_upper,
            Extremum::Supremum,
            Some(best.mask.clone()),
            region,
        )
    };
    Ok(CoveringReport {
        witness: best,
        pruned_cells,
        unresolved_cells,
        evaluations,
        status: status.map_err(AdversaryError::Evidence)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter_decomposition::moments::MomentBlock;
    use crate::parameter_decomposition::supports::{
        CardinalityCode, FailureHypergraph, minimum_code_support,
    };
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

        fn smoothness(&self) -> Option<SmoothnessCertificate> {
            None
        }
    }

    /// `F(q) = |M q|^2 / 2`: convex, zero with zero gradient at the all-on mask, so its maximum
    /// over the zonotope sits at a binary vertex.
    struct Quadratic {
        system: MaskMomentSystem,
        domain: MaskDomain,
        rows: Array2<f64>,
        map: Array2<f64>,
        certificate: Option<SmoothnessCertificate>,
    }

    enum CertificateKind {
        Derived,
        FalseZero,
        Absent,
    }

    impl Quadratic {
        fn fixture(kind: CertificateKind) -> Self {
            let rows = array![
                [0.8, 0.1],
                [-0.3, 0.6],
                [0.5, -0.7],
                [0.2, 0.4],
                [-0.6, -0.2]
            ];
            let map = array![[1.0, -0.4], [-0.6, 0.9], [0.2, 0.7]];
            // sigma_max(M)^2 <= |M|_F^2, rounded up across its K p squares and sums.
            let frobenius = map.iter().map(|entry| entry * entry).sum::<f64>();
            let certificate = match kind {
                CertificateKind::Derived => Some(SmoothnessCertificate {
                    gradient_lipschitz: frobenius * (1.0 + accumulation_growth(2 * map.len())),
                    derivation: "sigma_max(M)^2 <= |M|_F^2".to_string(),
                }),
                CertificateKind::FalseZero => Some(SmoothnessCertificate {
                    gradient_lipschitz: 0.0,
                    derivation: "positive control: a false constant".to_string(),
                }),
                CertificateKind::Absent => None,
            };
            Self {
                system: system_from_rows(&rows),
                domain: unit_domain(rows.nrows()),
                rows,
                map,
                certificate,
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

        fn smoothness(&self) -> Option<SmoothnessCertificate> {
            self.certificate.clone()
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
        assert!(report.upper_bound.is_none());
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
    fn smoothness_bound_covers_the_vertex_maximum_and_a_false_constant_claims_a_false_bound_2951() {
        let kept = [false; 5];
        let all_on = [1.0; 5];
        let derived = Quadratic::fixture(CertificateKind::Derived);
        let f_max = derived.exhaustive_max(&kept);
        let epsilon = 0.5 * f_max;
        // From the all-on mask the gradient is exactly zero, so the query is stationary and the
        // bound is (L / 2) R^2, which must cover the true maximum.
        let report = separate(&derived.system, &derived.domain, &kept, epsilon, &all_on, &derived)
            .expect("derived certificate");
        assert_eq!(report.termination, AscentTermination::Stationary);
        let upper = report.status.upper_bound().expect("a stated constant");
        assert!(upper >= f_max, "bound {upper} below the true maximum {f_max}");
        assert!(!report.status.certifies_at_most(epsilon));
        // Positive control: a false constant L = 0 at the same stationary start certifies
        // epsilon although the exhaustive vertex maximum exceeds it.
        let false_zero = Quadratic::fixture(CertificateKind::FalseZero);
        let false_claim = separate(
            &false_zero.system,
            &false_zero.domain,
            &kept,
            epsilon,
            &all_on,
            &false_zero,
        )
        .expect("false certificate at a stationary start");
        assert!(false_claim.status.certifies_at_most(epsilon));
        assert!(epsilon < f_max);
    }

    #[test]
    fn a_false_constant_is_refuted_by_the_ascents_own_witness_2951() {
        let kept = [false; 5];
        let center = [0.5; 5];
        let false_zero = Quadratic::fixture(CertificateKind::FalseZero);
        // At the center the gradient is (1/2) M^T M sum_c v_c != 0, so the gap is positive.
        // epsilon = F(center) blocks both a counterexample and the false bound F + G at the
        // center. Along the full vertex step F rises by G + |M d|^2 / 2 > G, so the next witness
        // exceeds the held false bound.
        let epsilon = false_zero.value_at_moment(&false_zero.moment(&center).expect("admissible"));
        let refuted = separate(
            &false_zero.system,
            &false_zero.domain,
            &kept,
            epsilon,
            &center,
            &false_zero,
        );
        assert!(matches!(
            refuted,
            Err(AdversaryError::CertificateRefuted { .. })
        ));
        // Positive control: the derived constant from the same start is never refuted.
        let derived = Quadratic::fixture(CertificateKind::Derived);
        let held = separate(&derived.system, &derived.domain, &kept, epsilon, &center, &derived);
        assert!(held.is_ok(), "derived constant refused: {held:?}");
    }

    #[test]
    fn lower_witness_never_exceeds_the_exhaustive_vertex_maximum_2951() {
        let fixture = Quadratic::fixture(CertificateKind::Absent);
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
        let fixture = Quadratic::fixture(CertificateKind::Absent);
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
        let fixture = Quadratic::fixture(CertificateKind::Absent);
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

    /// Support code lengths increasing with size, so the search proposes the smallest unrefuted
    /// supports first.
    struct SizeCode;

    impl CardinalityCode for SizeCode {
        type Error = std::convert::Infallible;

        fn support_bits(&self, components: usize, size: usize) -> Result<u64, Self::Error> {
            Ok(size.min(components) as u64)
        }
    }

    #[test]
    fn the_oracle_drives_the_support_search_to_the_only_live_control_2951() {
        let system = system_from_rows(&Array2::<f64>::eye(3));
        let domain = unit_domain(3);
        let epsilon = 0.5;
        let mut oracle =
            ZonotopeSeparationOracle::new(&system, &domain, epsilon, &InteriorPeak).expect("oracle");
        let search = minimum_code_support(&mut oracle, &SizeCode, epsilon, FailureHypergraph::new(3))
            .expect("support search");
        // F moves only through control 2, so every refutation perturbs it and every size-one support
        // without it is refuted from an interior start. Without a stated constant the support {2} is
        // never certified, so the search ends undecided there, with the full support certified by its
        // exact point evaluation.
        let undecided = search.undecided.as_ref().expect("an undecided candidate");
        assert_eq!(undecided.support.members(), &[2]);
        assert!(matches!(undecided.evidence, EvidenceStatus::Unresolved { .. }));
        let certified = search.certified.as_ref().expect("the full support certified");
        assert_eq!(certified.support.members(), &[0, 1, 2]);
        assert!(matches!(certified.evidence, EvidenceStatus::Exact { .. }));
        assert!(matches!(search.code, EvidenceStatus::Unresolved { .. }));
        assert!(!search.hypergraph.edges().is_empty());
        assert!(
            search
                .hypergraph
                .edges()
                .iter()
                .all(|edge| edge.perturbed.members().contains(&2))
        );
    }

    #[test]
    fn a_point_evaluation_is_a_counterexample_or_exact_over_its_one_mask_2951() {
        let system = system_from_rows(&Array2::<f64>::eye(3));
        let domain = unit_domain(3);
        let mut oracle =
            ZonotopeSeparationOracle::new(&system, &domain, 0.5, &InteriorPeak).expect("oracle");
        let refuting = oracle.evaluate(&vec![1.0, 1.0, 0.75]).expect("admissible mask");
        assert!(matches!(refuting, EvidenceStatus::Counterexample { .. }));
        assert!(refuting.refutes_at_most(0.5));
        // Positive control: all-on never refutes and is exact over its one mask.
        let all_on = oracle.evaluate(&vec![1.0; 3]).expect("all-on");
        assert!(matches!(
            all_on,
            EvidenceStatus::Exact {
                basis: ExactBasis::Exhaustive { cardinality: 1 },
                ..
            }
        ));
        assert!(!all_on.refutes_at_most(0.5));
        assert_eq!(all_on.domain(), Some(&OracleRegion::Mask(vec![1.0; 3])));
        assert!(matches!(
            oracle.evaluate(&vec![1.0, 1.0, 1.5]),
            Err(AdversaryError::Geometry(MomentGeometryError::MaskOutsideDomain { .. }))
        ));
    }

    #[test]
    fn a_support_query_starts_off_all_on_where_the_ascent_would_be_stationary_2951() {
        let fixture = Quadratic::fixture(CertificateKind::Absent);
        let kept = [false; 5];
        let epsilon = 0.25 * fixture.exhaustive_max(&kept);
        let mut oracle =
            ZonotopeSeparationOracle::new(&fixture.system, &fixture.domain, epsilon, &fixture)
                .expect("oracle");
        let empty = ComponentSet::new(5, Vec::new()).expect("empty support");
        let status = oracle.separate(&empty).expect("empty support query");
        assert!(status.refutes_at_most(epsilon));
        // Positive control: the same query started at all-on has a zero gradient, is stationary at
        // once, and refutes nothing.
        let from_all_on =
            separate(&fixture.system, &fixture.domain, &kept, epsilon, &[1.0; 5], &fixture)
                .expect("all-on start");
        assert_eq!(from_all_on.termination, AscentTermination::Stationary);
        assert!(!from_all_on.status.refutes_at_most(epsilon));
        // The full support is the one point all-on: exact and certified, where the ascent proves
        // only a lower witness.
        let full = oracle.separate(&ComponentSet::all(5)).expect("full support");
        assert!(full.certifies_at_most(epsilon));
        let ascent =
            separate(&fixture.system, &fixture.domain, &[true; 5], epsilon, &[1.0; 5], &fixture)
                .expect("all kept");
        assert!(!ascent.status.certifies_at_most(epsilon));
    }

    /// `F(q) = (27/4) q (1 - q)^2` on one control, `q = 1 - m`, with its derived smoothness constant:
    /// `F''(q) = (27/4)(6 q - 4)`, so `|F''| <= 27` on `[0, 1]`, attained at `q = 0`.
    struct CertifiedPeak;

    impl SeparationObjective for CertifiedPeak {
        fn evaluate(&self, mask: &[f64]) -> Result<ObjectiveJet, String> {
            let deletion = 1.0 - mask[0];
            // The same expanded-term bounds as InteriorPeak.
            Ok(ObjectiveJet {
                value: InteriorPeak::value(deletion),
                value_roundoff: accumulation_band(
                    8,
                    6.75 * deletion * (1.0 + deletion) * (1.0 + deletion),
                ),
                moment_gradient: MomentVector {
                    blocks: vec![array![6.75 * (1.0 - deletion) * (1.0 - 3.0 * deletion)]],
                },
                gradient_roundoff: accumulation_band(
                    8,
                    6.75 * (1.0 + deletion) * (1.0 + 3.0 * deletion),
                ),
            })
        }

        fn smoothness(&self) -> Option<SmoothnessCertificate> {
            Some(SmoothnessCertificate {
                gradient_lipschitz: 27.0,
                derivation: "|F''(q)| = (27/4)|6q - 4| <= 27 on [0, 1]".to_string(),
            })
        }
    }

    #[test]
    fn covering_certifies_just_above_the_interior_supremum_2951() {
        let system = system_from_rows(&array![[1.0]]);
        let domain = unit_domain(1);
        let report =
            certify_by_covering(&system, &domain, &[false], 1.02, &CertifiedPeak).expect("covering");
        assert!(matches!(report.status, EvidenceStatus::UniformBound { .. }));
        assert_eq!(report.unresolved_cells, 0);
        let upper = report.status.upper_bound().expect("a uniform bound");
        // The exact supremum is 1: a valid bound covers it, and the certificate sits at or below
        // epsilon.
        assert!(upper >= 1.0 && upper <= 1.02, "bound {upper}");
        assert!(report.pruned_cells >= 2);
    }

    #[test]
    fn covering_refutes_just_below_the_interior_supremum_and_never_certifies_it_2951() {
        let system = system_from_rows(&array![[1.0]]);
        let domain = unit_domain(1);
        let report =
            certify_by_covering(&system, &domain, &[false], 0.98, &CertifiedPeak).expect("covering");
        assert!(report.status.refutes_at_most(0.98));
        assert!(!report.status.certifies_at_most(0.98));
        let live = report.witness.mask[0];
        assert!(live > 0.0 && live < 1.0, "interior refuting center, got {live}");
    }

    #[test]
    fn covering_refuses_an_objective_without_a_stated_constant_2951() {
        let system = system_from_rows(&Array2::<f64>::eye(3));
        let domain = unit_domain(3);
        assert_eq!(
            certify_by_covering(&system, &domain, &[false; 3], 1.02, &InteriorPeak),
            Err(AdversaryError::MissingCertificate)
        );
        // Positive control: the certified one-control peak is accepted.
        let single = system_from_rows(&array![[1.0]]);
        assert!(certify_by_covering(&single, &unit_domain(1), &[false], 1.02, &CertifiedPeak).is_ok());
    }

    /// `F(q) = q / ε + 3` on one control, `q = 1 - m`: affine, so `L = 0` is exact. Over masks
    /// `m = 1 + k ε` every operation is exact, and in general the final sum rounds once.
    struct UlpRamp;

    impl SeparationObjective for UlpRamp {
        fn evaluate(&self, mask: &[f64]) -> Result<ObjectiveJet, String> {
            let scaled = (1.0 - mask[0]) / f64::EPSILON;
            Ok(ObjectiveJet {
                value: scaled + 3.0,
                value_roundoff: accumulation_band(1, scaled.abs() + 3.0),
                moment_gradient: MomentVector {
                    blocks: vec![array![1.0 / f64::EPSILON]],
                },
                gradient_roundoff: 0.0,
            })
        }

        fn smoothness(&self) -> Option<SmoothnessCertificate> {
            Some(SmoothnessCertificate {
                gradient_lipschitz: 0.0,
                derivation: "F is affine in q".to_string(),
            })
        }
    }

    /// mpd-verify batch 31 NOTE 1. On the declared box `[1, 1 + 3ε]` the midpoint `1 + 1.5ε`
    /// is a tie that rounds to `1 + 2ε`, so the lower endpoint sits `2ε` from the center while
    /// `(b - a) / 2 = 1.5ε`. A half-width of `(b - a) / 2` leaves a third of it uncovered, far
    /// past any relative band, and the covering certified `sup F <= 2.5 + O(u)` below the exact
    /// supremum `F(1) = 3`.
    #[test]
    fn covering_half_widths_cover_the_cell_around_a_rounded_center_2951() {
        let (lower, upper) = (1.0, 1.0 + 3.0 * f64::EPSILON);
        // The fixture exercises a rounded center: the computed midpoint is not the exact one.
        let center = lower + 0.5 * (upper - lower);
        assert_eq!(center, 1.0 + 2.0 * f64::EPSILON);
        assert!(center - lower > 0.5 * (upper - lower));
        let system = system_from_rows(&array![[1.0]]);
        let domain = MaskDomain::new(vec![(lower, upper)]).expect("the box contains all-on");
        let supremum = UlpRamp.evaluate(&[lower]).expect("the ramp evaluates").value;
        assert_eq!(supremum, 3.0);
        let epsilon = 2.75;
        let report = certify_by_covering(&system, &domain, &[false], epsilon, &UlpRamp).expect("covering");
        assert!(
            !report.status.certifies_at_most(epsilon),
            "the covering certified {:?} below the exact supremum {supremum}",
            report.status.upper_bound()
        );
        if let Some(upper_bound) = report.status.upper_bound() {
            assert!(upper_bound >= supremum, "bound {upper_bound} below the exact supremum");
        }
    }

    /// `F(q) = q^2` on one control, `q = 1 - m`, with a stated constant: the true `L = 2`, or a
    /// false `L = 0`.
    struct Square {
        gradient_lipschitz: f64,
    }

    impl SeparationObjective for Square {
        fn evaluate(&self, mask: &[f64]) -> Result<ObjectiveJet, String> {
            let deletion = 1.0 - mask[0];
            Ok(ObjectiveJet {
                value: deletion * deletion,
                value_roundoff: accumulation_band(2, deletion * deletion),
                moment_gradient: MomentVector {
                    blocks: vec![array![2.0 * deletion]],
                },
                gradient_roundoff: accumulation_band(1, 2.0 * deletion.abs()),
            })
        }

        fn smoothness(&self) -> Option<SmoothnessCertificate> {
            Some(SmoothnessCertificate {
                gradient_lipschitz: self.gradient_lipschitz,
                derivation: format!("stated {}; the exact constant is |F''| = 2", self.gradient_lipschitz),
            })
        }
    }

    /// mpd-verify batch 31 NOTE 2. A center is compared with the cells enclosing it. With the
    /// false `L = 0` the root cell `[0, 1]` (center `1/2`, `F = 1/4`, `|g| h = 1/2`) is bounded
    /// by `3/4`, and the center `1/8` of its descendant `[0, 1/4]` has `F = 49/64 > 3/4`, so the
    /// constant is refuted. Comparing a center only with its own cell's bound can never refute,
    /// since that bound is `F(m0)` plus nonnegative terms.
    #[test]
    fn covering_refutes_a_false_constant_at_a_center_inside_an_enclosing_cell_2951() {
        let system = system_from_rows(&array![[1.0]]);
        let domain = unit_domain(1);
        let epsilon = 0.7;
        let refused = certify_by_covering(
            &system,
            &domain,
            &[false],
            epsilon,
            &Square {
                gradient_lipschitz: 0.0,
            },
        );
        assert!(
            matches!(refused, Err(AdversaryError::CertificateRefuted { .. })),
            "the false constant was not refuted: {refused:?}"
        );
        if let Err(AdversaryError::CertificateRefuted {
            witness_lower,
            bound,
        }) = refused
        {
            assert!(witness_lower > bound);
            assert!(bound < 49.0 / 64.0);
        }
        // Positive control: the true constant is not refuted, and the same descendant center
        // refutes epsilon instead.
        let report = certify_by_covering(
            &system,
            &domain,
            &[false],
            epsilon,
            &Square {
                gradient_lipschitz: 2.0,
            },
        )
        .expect("the true constant is not refuted");
        assert!(report.status.refutes_at_most(epsilon));
    }
}
