//! The contract of a program decomposition and its two-part code (#2951).
//!
//! A contract declares only what behaviour is explained: a finite input family and the program
//! output that is read (logit rows, one categorical distribution per input). It declares no
//! tolerance. A program `P` is scored by the two-part code of the model's behaviour,
//!
//! ```text
//! L_total(P) = L(P) + L(behaviour | P),     L(behaviour | P) = n Σ_rows KL(model_row ‖ P_row) / ln 2,
//! ```
//!
//! the decoded message length of `P` plus the expected excess code length, in bits, of encoding `n`
//! observations of the model's output on each of the family's rows with `P`'s distributions instead
//! of the model's. The fidelity level is then chosen by the code itself; what moves the trade-off is
//! the amount of behaviour explained, the rows and the observations `n` of each, which are a
//! declaration of the data and not a tolerance.
//!
//! # Three parts: structure, explanations, precision
//!
//! The program's message splits into its structure (bases, interfaces, present blocks, rules,
//! wiring, laws) and its precision (each lattice's fraction bits and indices,
//! `operator_program::CodeAccount`); rounding a program changes only the second. The behaviour of
//! each input is then explained by the list of the program's rule instances that fire on it: every
//! group of a pointwise node whose law is not the zero law is an instance, active on an input when
//! its value there is not identically zero. The explanations are sent once per input (not per
//! observation). The library's firing counts `c_r` (how many of the `N` inputs instance `r`
//! fires on, `log₂(N + 1)` bits each) are sent once; then each input sends how many instances
//! fire, `k_x` (`log₂(R + 1)` bits for `R` instances), and the set itself in the code that gives
//! instance `r` probability `c_r / T`, `T = Σ c_r`, a set of `k_x` being any of its `k_x!`
//! orders:
//!
//! ```text
//! Σ_x L(explanation of x) = R log₂(N + 1) + N log₂(R + 1) + Σ_r c_r log₂(T / c_r) − Σ_x log₂ k_x!
//! ```
//!
//! An input is explained by a few frequent rules cheaply and by many rules dearly: an instance
//! that fires on every input still costs about `log₂ e` bits on every input, so the code prefers
//! computations that use few rules per input (the axis of per-input active components), and the
//! library of rules is paid once and amortised. The score is the total
//!
//! ```text
//! L(structure) + L(precision) + Σ_x L(explanation of x | program) + n Σ_rows KL/ln 2.
//! ```
//!
//! An activity the forward-error bands leave open (an interval of pre-activations that straddles
//! the law's zero set) may flip; each flip moves the code by at most
//! `log₂ T' + log₂ R + log₂(N + 1) + 4` bits (`T'` the largest possible `T`: its own term, the
//! change of `c_r`, of `T` and of `k_x`), and the explanation term carries that interval.
//!
//! # Evidence
//!
//! Each row's `KL` is [`crate::verify::compare_logit_row`]'s over both logit boxes: an exact value with its
//! evaluation error, or unresolved. The data term is their sum, exact up to the summed errors and
//! the summation's own `γ_n` band, or unresolved with an infinite upper end. The reported maxima
//! (the largest row KL, the argmax disagreement count) are outputs about the selected program,
//! stated by [`crate::verify::exhaustive_supremum`] over the rows; they select nothing.
//!
//! # Certified and measured bands
//!
//! The logit bands come from the banded execution (`operator_program::Trace`): a proven enclosure
//! of the exact-arithmetic output. In a deep network the enclosure compounds layer by layer (each
//! layer multiplies the carried error by its operator norms and its attention and norm slopes)
//! until it proves nothing about a row's KL. A distribution row whose proven band is not below one
//! logit unit takes instead a measured band ([`measured_output_band`]): the nodes' local roundings
//! with random signs, carried to the output to first order. Such a row's verdicts are
//! measurements; [`ProgramScore::measured_rows`] counts them, and every report states it.
//!
//! # Total variation
//!
//! KL is the data term because it is a code length, but it is not a metric. The evaluation also
//! reports a proven upper bound on each row's total variation distance, the smaller of Pinsker's
//! `√(KL/2)` at the row's proven KL upper end and the sharp softmax bound `tanh(w/4)` over the two
//! logit boxes (`bounds::total_variation_over_logit_boxes`); TV obeys the triangle inequality, so it is
//! the distance in which errors of composed replacements add. The program is always compared with the
//! model end to end, each part reading what the program's own earlier parts computed.
//!
//! # Causality
//!
//! A contract may declare, per readout, the slots that readout may read (for a causal sequence
//! model, the positions up to its own). A program whose readout reads another slot is refused: the
//! program is causal by construction, whatever the search that found it looked at.
//!
//! # Complete and sampled families
//!
//! A [`FamilyKind::Complete`] family is the whole declared input domain, so every status is
//! `Exact { Exhaustive }` over the domain. A [`FamilyKind::Sample`] family is a harvest from a
//! population the contract names; its statuses are exhaustive over the sample's rows, which is the
//! only domain they speak about. The rows are grouped into independent units (documents,
//! episodes); a unit disagrees when any of its rows' argmax differs from the model's or is not
//! certified. Two population bounds on the unit disagreement rate are reported, never promoted to
//! one another or to an exhaustive status:
//!
//! * the one-sided Clopper–Pearson bound, valid for a program fixed before the sample was drawn
//!   (a held-out test split);
//! * the Occam bound, valid for a program selected with the sample: a prefix code gives the
//!   selected program prior mass `2^{−L}` by Kraft's inequality, so with probability at least
//!   `1 − δ`, simultaneously for every program, `rate ≤ empirical + √((L ln 2 + ln(1/δ))/(2n))`
//!   (Hoeffding and a union bound weighted by the code), with `L` the decoded message length and
//!   `n` the number of units.

use super::bounds::BoundError;
use super::bounds::total_variation_over_logit_boxes;
use super::operator_program::{Declarations, EncodedProgram, FamilyInputs, Law, Node, OperatorProgram, ProgramError, Trace};
use super::precision::DecodableArtifact;
use super::secant::BandedMatrix;
use super::supports::{EvidenceStatus, EvidenceStatusError, ExactBasis, Extremum};
use super::verify::{RowValue, compare_logit_row, computed_argmax, exhaustive_supremum};
use gam_linalg::roundoff::accumulation_growth;
use ndarray::Array2;
use statrs::distribution::{Beta, ContinuousCDF};
use statrs::function::gamma::ln_gamma;
use std::collections::BTreeMap;
use std::f64::consts::LN_2;
use std::fmt;

/// What the family is.
#[derive(Clone, Debug, PartialEq)]
pub enum FamilyKind {
    /// Every input of the declared domain.
    Complete { description: String },
    /// Independent units drawn from a named population, with the confidence of the population
    /// bounds; `units[row]` is the unit each row belongs to.
    Sample { population: String, confidence: f64, units: Vec<usize> },
}

/// A declared contract: the behaviour to explain.
#[derive(Clone, Debug, PartialEq)]
pub struct Contract {
    pub declarations: Declarations,
    pub family: FamilyInputs,
    pub kind: FamilyKind,
    /// `n`: how many observations of each row's output distribution the code explains.
    pub observations: u64,
    /// How many categorical distributions each input's output holds: the output node's width is
    /// `readouts` blocks of equal width (one per readout position, say), each a distribution row.
    pub readouts: usize,
    /// Per readout, the slots it may read; `None` declares no causal restriction.
    pub readout_slots: Option<Vec<Vec<usize>>>,
}

/// The domain a contract status is stated over: the family's rows.
#[derive(Clone, Debug, PartialEq)]
pub struct ContractDomain {
    pub rows: usize,
    pub complete: bool,
}

/// A status over the family's rows, witnessed by a row index where it has one.
pub type ContractStatus = EvidenceStatus<usize, ContractDomain>;

/// The population claim of a sampled family, over its independent units.
#[derive(Clone, Debug, PartialEq)]
pub struct PopulationBound {
    /// Units with a row whose argmax differs from the model's or is not certified.
    pub disagreeing: u64,
    pub units: u64,
    pub confidence: f64,
    /// The one-sided Clopper–Pearson upper bound on the unit disagreement rate at `confidence`,
    /// for a program fixed before the sample.
    pub fixed_program_upper: f64,
    /// The Occam bound for the program selected on the sample, from its code length.
    pub selected_program_upper: f64,
    /// `disagreeing/units` with its binomial standard error.
    pub estimate: EvidenceStatus<(), String>,
}

/// Everything one evaluation of a program against the model shows.
#[derive(Clone, Debug, PartialEq)]
pub struct ContractEvaluation {
    /// `Σ_rows KL(model ‖ program)`, in nats.
    pub total_kl: ContractStatus,
    /// `max_rows KL(model ‖ program)`.
    pub max_kl: ContractStatus,
    /// Each row's proven upper bound on its KL (`+∞` where unresolved).
    pub kl_upper: Vec<f64>,
    pub argmax_disagreements: u64,
    /// Rows whose two argmaxes are not both certified.
    pub argmax_uncertified: u64,
    /// Per row, whether its argmax is certified equal to the model's.
    pub argmax_agrees: Vec<bool>,
    /// A proven upper bound on `max_rows TV(model, program)`.
    pub max_tv_upper: f64,
}

/// Which gated rule instances fire on each input of the family (module note, "Three parts").
#[derive(Clone, Debug, PartialEq)]
pub struct Explanation {
    /// The gated instances: groups of pointwise nodes whose law is not the zero law.
    pub instances: usize,
    /// The explanations' code length at the computed activities, and a proven interval around it.
    pub bits: f64,
    pub bits_lower: f64,
    pub bits_upper: f64,
    /// Per input, the instances proven active and those whose activity the bands leave open.
    pub active: Vec<u32>,
    pub open: Vec<u32>,
}

impl Explanation {
    /// The mean number of instances proven active per input: the size of an input's computation.
    pub fn mean_active(&self) -> f64 {
        if self.active.is_empty() {
            return 0.0;
        }
        self.active.iter().map(|a| f64::from(*a)).sum::<f64>() / self.active.len() as f64
    }

    /// The explanation code length per input.
    pub fn bits_per_input(&self) -> f64 {
        if self.active.is_empty() { 0.0 } else { self.bits / self.active.len() as f64 }
    }
}

/// `log₂ k!`.
fn log2_factorial(k: usize) -> f64 {
    ln_gamma(k as f64 + 1.0) / LN_2
}

/// The explanations of the family given `program`, from its execution `trace` on the family
/// (module note, "Three parts").
pub fn explanation(program: &OperatorProgram, trace: &Trace) -> Result<Explanation, ContractError> {
    let interfaces = program.interfaces()?;
    let rows = trace.values.first().map_or(0, |v| v.nrows());
    let (mut active, mut open) = (vec![0u32; rows], vec![0u32; rows]);
    let mut fired = vec![0usize; rows];
    let mut counts: Vec<usize> = Vec::new();
    for node in &program.nodes {
        let Node::Pointwise { input, laws } = node else { continue };
        let values = &trace.values[*input];
        let enclosure = trace.band(*input);
        let bands = enclosure.as_ref();
        for (group, law) in laws.iter().enumerate() {
            if *law == Law::Zero {
                continue;
            }
            let range = interfaces[*input].range(group);
            let mut count = 0usize;
            for row in 0..rows {
                let (mut fires, mut proven, mut zero) = (false, false, true);
                for column in range.clone() {
                    let v = values[[row, column]];
                    let r = bands.map_or(0.0, |b| b[[row, column]]);
                    fires |= law.vanishes_on(v, v) == Some(false);
                    match law.vanishes_on(v - r, v + r) {
                        Some(false) => proven = true,
                        Some(true) => {}
                        None => zero = false,
                    }
                }
                if fires {
                    count += 1;
                    fired[row] += 1;
                }
                if proven {
                    active[row] += 1;
                } else if !zero {
                    open[row] += 1;
                }
            }
            counts.push(count);
        }
    }
    let instances = counts.len();
    let (n, r) = (rows as f64, instances as f64);
    let total: usize = counts.iter().sum();
    let t = total as f64;
    let mut bits = r * (n + 1.0).log2() + n * (r + 1.0).log2();
    for &c in &counts {
        if c > 0 {
            bits += c as f64 * (t / c as f64).log2();
        }
    }
    for &k in &fired {
        bits -= log2_factorial(k);
    }
    let flips: f64 = open.iter().map(|o| f64::from(*o)).sum();
    let largest_total = t + flips;
    let per_flip = largest_total.max(1.0).log2() + (r + 1.0).log2() + (n + 1.0).log2() + 4.0;
    // Rounding: each of the `R + N + 2` terms is within a few ulps relative of itself.
    let rounding = 64.0 * f64::EPSILON * (bits.abs() + n * (r + 1.0).log2() + t * largest_total.max(2.0).log2());
    Ok(Explanation {
        instances,
        bits,
        bits_lower: (bits - flips * per_flip - rounding).max(0.0).next_down(),
        bits_upper: (bits + flips * per_flip + rounding).next_up(),
        active,
        open,
    })
}

/// A program's code under the contract: structure, precision, explanations and data (module note).
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramScore {
    /// The decoded message length of the program: `structure_bits + precision_bits`.
    pub program_bits: u64,
    pub structure_bits: u64,
    pub precision_bits: u64,
    /// Distribution rows of the program whose band is measured rather than proven ("Certified and
    /// measured bands"); the score's verdicts on those rows are measurements, not certificates.
    pub measured_rows: usize,
    /// The family's explanations given the program.
    pub explanation: Explanation,
    /// `L(behaviour | P) = Σ KL / ln 2` and its error; `+∞` error when some row is unresolved.
    pub data_bits: f64,
    pub data_bits_error: f64,
    pub evaluation: ContractEvaluation,
    /// For a sampled family, its population bounds.
    pub population: Option<PopulationBound>,
}

impl ProgramScore {
    /// A proven lower end of the total code length.
    pub fn total_lower(&self) -> f64 {
        (self.program_bits as f64 + self.explanation.bits_lower + (self.data_bits - self.data_bits_error).max(0.0)).next_down()
    }

    /// A proven upper end of the total code length.
    pub fn total_upper(&self) -> f64 {
        (self.program_bits as f64 + self.explanation.bits_upper + self.data_bits + self.data_bits_error).next_up()
    }

    /// The computed total.
    pub fn total(&self) -> f64 {
        self.program_bits as f64 + self.explanation.bits + self.data_bits
    }

    /// Whether this score is proven shorter than `other`.
    pub fn proven_shorter_than(&self, other: &ProgramScore) -> bool {
        self.total_upper() < other.total_lower()
    }
}

/// A refused evaluation.
#[derive(Debug)]
pub enum ContractError {
    Program(ProgramError),
    Bound(BoundError),
    Evidence(EvidenceStatusError),
    Shape { reference: (usize, usize), candidate: (usize, usize) },
    Declaration(String),
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Program(error) => write!(f, "contract: {error}"),
            Self::Bound(error) => write!(f, "contract: {error:?}"),
            Self::Evidence(error) => write!(f, "contract: {error}"),
            Self::Shape { reference, candidate } => {
                write!(f, "contract: reference logits {reference:?} and candidate logits {candidate:?} do not pair")
            }
            Self::Declaration(message) => write!(f, "contract: {message}"),
        }
    }
}

impl std::error::Error for ContractError {}

impl From<ProgramError> for ContractError {
    fn from(error: ProgramError) -> Self {
        Self::Program(error)
    }
}

impl From<EvidenceStatusError> for ContractError {
    fn from(error: EvidenceStatusError) -> Self {
        Self::Evidence(error)
    }
}

impl Contract {
    /// Refused unless a sampled family declares a confidence in `(0, 1)` and one unit per row, and
    /// the family has a row.
    pub fn validate(&self) -> Result<(), ContractError> {
        if let FamilyKind::Sample { confidence, units, .. } = &self.kind {
            if !(*confidence > 0.0 && *confidence < 1.0) {
                return Err(ContractError::Declaration(format!("confidence {confidence} is not in (0, 1)")));
            }
            if units.len() != self.family.rows {
                return Err(ContractError::Declaration(format!("{} unit labels for {} rows", units.len(), self.family.rows)));
            }
        }
        if self.family.rows == 0 || self.observations == 0 || self.readouts == 0 {
            return Err(ContractError::Declaration("an empty family, no observations or no readouts".to_string()));
        }
        Ok(())
    }

    fn domain(&self) -> ContractDomain {
        ContractDomain { rows: self.family.rows * self.readouts, complete: matches!(self.kind, FamilyKind::Complete { .. }) }
    }

    /// The program's output rows on the family, with forward-error bands.
    pub fn logits(&self, program: &OperatorProgram) -> Result<BandedMatrix, ContractError> {
        Ok(self.banded_trace(program)?.0)
    }

    /// [`Contract::logits`] with the number of distribution rows whose band is measured, not
    /// proven (module note, "Certified and measured bands").
    pub fn logits_labelled(&self, program: &OperatorProgram) -> Result<(BandedMatrix, usize), ContractError> {
        let (logits, _, measured) = self.banded_trace(program)?;
        Ok((logits, measured))
    }

    /// The program's output rows on the family with bands, its banded trace, and how many
    /// distribution rows carry a measured band.
    fn banded_trace(&self, program: &OperatorProgram) -> Result<(BandedMatrix, Trace, usize), ContractError> {
        if program.declarations != self.declarations {
            return Err(ContractError::Declaration("the program's declarations are not the contract's".to_string()));
        }
        let trace = program.execute(&self.family, true)?;
        let mut banded = trace.banded(program.output);
        // A row whose proven band is not below one logit unit (a factor e on a probability) proves
        // little about its KL: it takes the measured band instead.
        let unresolved: Vec<usize> = banded
            .bands
            .outer_iter()
            .enumerate()
            .filter(|(_, row)| row.iter().any(|r| !(r.is_finite() && *r < 1.0)))
            .map(|(row, _)| row)
            .collect();
        if !unresolved.is_empty() {
            let measured = measured_output_band(program, &self.family, &trace)?;
            for &row in &unresolved {
                banded.bands.row_mut(row).assign(&measured.row(row));
            }
        }
        let logits = BandedMatrix { values: self.distributions(&banded.values)?, bands: self.distributions(&banded.bands)? };
        Ok((logits, trace, unresolved.len() * self.readouts))
    }

    /// Refuse a program whose readout reads a slot the contract does not allow it.
    pub fn check_causal(&self, program: &OperatorProgram) -> Result<(), ContractError> {
        let Some(allowed) = &self.readout_slots else { return Ok(()) };
        if allowed.len() != self.readouts {
            return Err(ContractError::Declaration(format!("{} slot sets for {} readouts", allowed.len(), self.readouts)));
        }
        let parts: Vec<usize> = match &program.nodes[program.output] {
            Node::Concat { parts } if parts.len() == self.readouts => parts.clone(),
            _ => vec![program.output; self.readouts],
        };
        for (readout, (part, slots)) in parts.iter().zip(allowed).enumerate() {
            if let Some(slot) = program.slots_read(*part)?.into_iter().find(|s| !slots.contains(s)) {
                return Err(ContractError::Declaration(format!(
                    "readout {readout} reads slot {slot}, which the contract does not allow it"
                )));
            }
        }
        Ok(())
    }

    /// An output value (`inputs × readouts·classes`) as distribution rows (`inputs·readouts ×
    /// classes`), input-major.
    pub fn distributions(&self, values: &Array2<f64>) -> Result<Array2<f64>, ContractError> {
        let (rows, width) = values.dim();
        if width % self.readouts != 0 {
            return Err(ContractError::Declaration(format!("an output of width {width} for {} readouts", self.readouts)));
        }
        let classes = width / self.readouts;
        let flat: Vec<f64> = values.iter().copied().collect();
        Array2::from_shape_vec((rows * self.readouts, classes), flat).map_err(|e| ContractError::Declaration(e.to_string()))
    }

    /// Compare candidate logit rows against the reference's, row by row.
    pub fn evaluate(&self, reference: &BandedMatrix, candidate: &BandedMatrix) -> Result<ContractEvaluation, ContractError> {
        self.validate()?;
        let shape = reference.values.dim();
        if candidate.values.dim() != shape || reference.bands.dim() != shape || candidate.bands.dim() != shape {
            return Err(ContractError::Shape { reference: shape, candidate: candidate.values.dim() });
        }
        if shape.0 != self.family.rows * self.readouts {
            return Err(ContractError::Declaration(format!(
                "{} distribution rows for {} inputs × {} readouts",
                shape.0, self.family.rows, self.readouts
            )));
        }
        let mut kls = Vec::with_capacity(shape.0);
        let mut kl_upper = Vec::with_capacity(shape.0);
        let mut argmax_agrees = Vec::with_capacity(shape.0);
        let (mut disagreements, mut uncertified) = (0u64, 0u64);
        let mut max_tv_upper = 0.0_f64;
        let (mut sum, mut sum_error, mut sum_magnitude, mut sum_lower) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let mut resolved = true;
        for row in 0..shape.0 {
            // A band that is not a finite number (the forward-error analysis overflowed, or met
            // `∞ · 0`) proves nothing about the row: it is unresolved, its argmax uncertified.
            let unbounded = |bands: &Array2<f64>| bands.row(row).iter().any(|r| !r.is_finite());
            if unbounded(&reference.bands) || unbounded(&candidate.bands) {
                resolved = false;
                kl_upper.push(f64::INFINITY);
                max_tv_upper = 1.0;
                let differs = computed_argmax(reference.values.row(row)) != computed_argmax(candidate.values.row(row));
                disagreements += u64::from(differs);
                uncertified += 1;
                argmax_agrees.push(false);
                kls.push((row, RowValue::Unresolved { lower: 0.0 }));
                continue;
            }
            let tv = total_variation_over_logit_boxes(
                reference.values.row(row),
                reference.bands.row(row),
                candidate.values.row(row),
                candidate.bands.row(row),
            )
            .map_err(ContractError::Bound)?
            .upper_bound()
            .unwrap_or(1.0);
            let comparison = compare_logit_row(
                reference.values.row(row),
                reference.bands.row(row),
                candidate.values.row(row),
                candidate.bands.row(row),
            )
            .map_err(ContractError::Bound)?;
            // A resolved value whose error is not finite proves only its lower end.
            let kl = match RowValue::of(&comparison.forward_kl) {
                RowValue::Resolved { value, numerical_error } if !(value.is_finite() && numerical_error.is_finite()) => {
                    RowValue::Unresolved { lower: comparison.forward_kl.lower_bound().unwrap_or(0.0).max(0.0) }
                }
                other => other,
            };
            match kl {
                RowValue::Resolved { value, numerical_error } => {
                    sum += value;
                    sum_error += numerical_error;
                    sum_magnitude += value.abs();
                    sum_lower += (value - numerical_error).max(0.0);
                }
                RowValue::Unresolved { lower } => {
                    resolved = false;
                    sum_lower += lower;
                }
            }
            let kl_bound = comparison.forward_kl.upper_bound().unwrap_or(f64::INFINITY);
            kl_upper.push(kl_bound);
            max_tv_upper = max_tv_upper.max(tv.min((kl_bound / 2.0).sqrt().next_up()).min(1.0));
            let differs = comparison.reference_argmax != comparison.candidate_argmax;
            disagreements += u64::from(differs);
            uncertified += u64::from(!comparison.argmax_certified);
            argmax_agrees.push(!differs && comparison.argmax_certified);
            kls.push((row, kl));
        }
        let domain = self.domain();
        let max_kl = exhaustive_supremum(kls.iter().copied(), None, domain.clone())?;
        let total_kl = if resolved {
            let error = (sum_error + accumulation_growth(shape.0) * sum_magnitude).next_up();
            EvidenceStatus::exact(sum, error, ExactBasis::Exhaustive { cardinality: shape.0 as u64 }, None, domain)?
        } else {
            EvidenceStatus::unresolved(sum_lower.next_down(), f64::INFINITY, Extremum::Supremum, None, domain)?
        };
        Ok(ContractEvaluation {
            total_kl,
            max_kl,
            kl_upper,
            argmax_disagreements: disagreements,
            argmax_uncertified: uncertified,
            argmax_agrees,
            max_tv_upper,
        })
    }

    /// Encode `program`, decode the message, execute the decoded program with bands and score its
    /// two-part code against `reference` (the model's banded logits).
    pub fn score(&self, program: &OperatorProgram, reference: &BandedMatrix) -> Result<ProgramScore, ContractError> {
        self.check_causal(program)?;
        let message = program.encode()?;
        let program_bits = message.len_bits();
        let decoded = EncodedProgram { message, declarations: self.declarations.clone() }
            .decode()
            .map_err(ContractError::Declaration)?;
        let account = decoded.code_account()?;
        if account.total_bits != program_bits {
            return Err(ContractError::Declaration(format!("a message of {program_bits} bits accounted as {}", account.total_bits)));
        }
        let (structure_bits, precision_bits) = (account.structure_bits(), account.precision_bits());
        let (candidate, trace, measured_rows) = self.banded_trace(&decoded)?;
        let explanation = explanation(&decoded, &trace)?;
        drop(trace);
        let evaluation = self.evaluate(reference, &candidate)?;
        let n = self.observations as f64;
        let (data_bits, data_bits_error) = match &evaluation.total_kl {
            EvidenceStatus::Exact { value, numerical_error, .. } => {
                (n * value / LN_2, (n * numerical_error / LN_2).next_up() + (n * value / LN_2) * 2.0 * f64::EPSILON)
            }
            other => (n * other.lower_bound().unwrap_or(0.0) / LN_2, f64::INFINITY),
        };
        let population = match &self.kind {
            FamilyKind::Complete { .. } => None,
            FamilyKind::Sample { population, confidence, units } => {
                let mut disagrees: BTreeMap<usize, bool> = BTreeMap::new();
                for (row, agrees) in evaluation.argmax_agrees.iter().enumerate() {
                    *disagrees.entry(units[row / self.readouts]).or_insert(false) |= !agrees;
                }
                let k = disagrees.values().filter(|v| **v).count() as u64;
                let n = disagrees.len() as u64;
                let rate = k as f64 / n as f64;
                Some(PopulationBound {
                    disagreeing: k,
                    units: n,
                    confidence: *confidence,
                    fixed_program_upper: clopper_pearson_upper(k, n, *confidence)?,
                    selected_program_upper: occam_upper(k, n, program_bits, 1.0 - confidence)?,
                    estimate: EvidenceStatus::statistical_estimate(
                        rate,
                        (rate * (1.0 - rate) / n as f64).sqrt(),
                        n,
                        format!("P(unit argmax disagrees) under {population}"),
                    )?,
                })
            }
        };
        Ok(ProgramScore {
            program_bits,
            structure_bits,
            precision_bits,
            measured_rows,
            explanation,
            data_bits,
            data_bits_error,
            evaluation,
            population,
        })
    }
}

/// Probes of the measured band: random-sign draws of the local roundings.
const ROUNDING_PROBES: u64 = 4;

/// The measured band of the program's output on `family`: every node's local rounding (its band
/// with exact arguments), given independent random signs, carried to the output to first order by
/// one forward-mode pass (`derivatives::jvp_seeded`); the band is four times the largest output
/// change over the probes, entrywise. It estimates the rounding error the computed output carries
/// where the proven enclosure has compounded past use (deep networks), and proves nothing.
pub fn measured_output_band(program: &OperatorProgram, family: &FamilyInputs, trace: &Trace) -> Result<Array2<f64>, ContractError> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    let local = program.local_rounding(family, trace)?;
    let output = &trace.values[program.output];
    let mut band = Array2::<f64>::zeros(output.dim());
    for probe in 0..ROUNDING_PROBES {
        let mut rng = StdRng::seed_from_u64(0x5eed_0000 + probe);
        let seeds: BTreeMap<usize, Array2<f64>> = local
            .iter()
            .enumerate()
            .filter(|(_, r)| r.iter().any(|v| *v != 0.0))
            .map(|(node, r)| (node, r.mapv(|v| if rng.random::<bool>() { v } else { -v })))
            .collect();
        let change = super::derivatives::jvp_seeded(program, family, trace, &BTreeMap::new(), &seeds)?;
        band.zip_mut_with(&change, |b, c| *b = b.max(4.0 * c.abs()));
    }
    Ok(band)
}

/// The Occam bound `k/n + √((L ln 2 + ln(1/δ))/(2n))` (module note), capped at 1, every step
/// rounded upward so the computed figure is never below the bound.
pub fn occam_upper(k: u64, n: u64, code_bits: u64, delta: f64) -> Result<f64, ContractError> {
    if n == 0 || k > n || !(delta > 0.0 && delta < 1.0) {
        return Err(ContractError::Declaration(format!("{k} disagreements in {n} units at delta {delta}")));
    }
    // ln 2 and ln(1/δ) are within one ulp of exact; each product, sum, quotient and root below is
    // stepped one float up.
    let log_prior = (code_bits as f64 * LN_2.next_up()).next_up();
    let log_confidence = (1.0 / delta).next_up().ln().next_up().next_up();
    let variance = ((log_prior + log_confidence).next_up() / (2.0 * n as f64)).next_up();
    let slack = variance.sqrt().next_up();
    Ok(((k as f64 / n as f64).next_up() + slack).next_up().min(1.0))
}

/// The one-sided Clopper–Pearson upper bound on a binomial proportion with `k` successes in `n`: the
/// `confidence` quantile of `Beta(k + 1, n − k)`. The quantile solver's figure is raised until the
/// distribution function there is at least `confidence`, then stepped one float up, so the figure
/// is not below the quantile up to the distribution function's own evaluation error.
pub fn clopper_pearson_upper(k: u64, n: u64, confidence: f64) -> Result<f64, ContractError> {
    if n == 0 || k > n {
        return Err(ContractError::Declaration(format!("{k} successes in {n} draws")));
    }
    if k == n {
        return Ok(1.0);
    }
    let beta = Beta::new((k + 1) as f64, (n - k) as f64)
        .map_err(|error| ContractError::Declaration(format!("Clopper–Pearson beta: {error}")))?;
    let mut upper = beta.inverse_cdf(confidence);
    while upper < 1.0 && beta.cdf(upper) < confidence {
        upper = (upper + (1.0 - upper) * f64::EPSILON.sqrt()).next_up();
    }
    Ok(upper.next_up().min(1.0))
}
