//! Discrete latent-signature structures: how many signatures exist, which
//! disease-signature connections may be nonzero, and which genetic drives,
//! disease jumps and other Gaussian function families are zero-effect
//! functions. A structure is a model index with a normalized prior. Its
//! posterior weight uses the same coefficient-integrated joint evidence as the
//! fit of that structure. Candidates are proposed from derivatives of an
//! existing fit and decided only by their own resolved evidence; a failed or
//! unresolved evaluation is never evidence for a smaller structure.
use super::function_prior::FunctionPenalty;
use super::law::{invalid, numerical};
use super::structural_prior::ln_1p;
use crate::chain::log_sum_exp;
use crate::scalar::{div, ln, sqrt};
use crate::{EventHistoryError, MarkKind};
use gam_math::nested_dual::JetField;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Which final functions a model allows to be nonzero. Signature labels are a
/// gauge: the prior and the class identity are invariant to permuting
/// signatures together with their support columns and drive rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JointStructure {
    /// `decoder_support[mark][signature]`: connection `pi_dk` may be nonzero.
    /// An absent connection is the face `pi_dk = 0`, not a large negative logit.
    pub decoder_support: Vec<Vec<bool>>,
    /// `genetic_drive[signature][score]`: false is the zero-effect function
    /// `B_kg = 0`, in both the drive and the entry mean. The background drive
    /// `u_k` is present for every signature unless the DriveLevel zero effect
    /// pins the mean level `u_k + B_k' mu`; the GeneticDrive zero effect pins
    /// every `B_k` at once and keeps `u_k`. No prior reads these flags yet (review
    /// F1/F7): a structure owner applies them.
    pub genetic_drive: Vec<Vec<bool>>,
    /// `jumps[mark]`: `None` for terminal marks; `Some(false)` is `J_d = 0`.
    pub jumps: Vec<Option<bool>>,
    /// Gaussian function families fixed at the zero function: the limit of
    /// their strength, where the prior becomes a point mass. Sorted and unique.
    pub zero_effects: Vec<FunctionPenalty>,
}

/// One evaluated structure, from that structure's converged fitting route.
/// `log_basin_evidence` integrates one label basin. `log_domain_mass` is the
/// log posterior mass, under that route's approximation, of the fundamental
/// domain of the support's label stabilizer; it is exactly zero when no label
/// permutation preserves the support.
#[derive(Clone, Debug)]
pub struct StructureEvidence {
    pub structure: JointStructure,
    pub log_basin_evidence: f64,
    pub log_domain_mass: f64,
    pub log_evidence_error: f64,
}

/// Posterior weights over the evaluated structures. Predictions average their
/// final probabilities with these weights; no single structure is the answer.
#[derive(Clone, Debug)]
pub struct StructurePosterior<S = f64> {
    log_weights: Vec<S>,
    log_class_weights: Vec<S>,
    errors: Vec<f64>,
}

fn constant<S: JetField>(zero: &S, value: f64) -> S {
    zero.constant_like(value)
}

fn log_factorial<S: JetField>(zero: &S, n: usize) -> S {
    (1..=n).fold(zero.clone(), |sum, j| sum.add(&ln(&constant(zero, j as f64))))
}

fn log_choose<S: JetField>(zero: &S, n: usize, m: usize) -> S {
    let m = m.min(n - m);
    (1..=m).fold(zero.clone(), |sum, j| {
        sum.add(&ln(&constant(zero, (n - m + j) as f64)))
            .sub(&ln(&constant(zero, j as f64)))
    })
}

/// integral_0^1 q^m (1-q)^(n-m) dq: the inclusion rate integrated under a
/// uniform law. It depends only on the count, so it corrects for multiplicity.
fn log_beta_binomial<S: JetField>(zero: &S, n: usize, m: usize) -> S {
    ln(&constant(zero, (n + 1) as f64))
        .neg()
        .sub(&log_choose(zero, n, m))
}

/// p(K) = integral_0^1 q (1-q)^K dq = 1/((K+1)(K+2)).
fn log_rank_prior<S: JetField>(zero: &S, signatures: usize) -> S {
    ln(&constant(zero, (signatures + 1) as f64))
        .neg()
        .sub(&ln(&constant(zero, (signatures + 2) as f64)))
}

/// P(every signature column has a connection | K) under the connection law:
/// integral_0^1 (1-(1-q)^D)^K dq = prod_{j=1}^K jD/(jD+1).
fn log_admissible_mass<S: JetField>(zero: &S, marks: usize, signatures: usize) -> S {
    (1..=signatures).fold(zero.clone(), |sum, j| {
        sum.sub(&ln_1p(&div(
            &constant(zero, 1.0),
            &constant(zero, (j * marks) as f64),
        )))
    })
}

fn normal_logcdf<S: JetField>(x: &S) -> S {
    x.compose_unary(gam_math::probability::normal_logcdf_derivatives(x.value()))
}

/// log integral_0^infinity exp(g x - h x^2/2) dx for h > 0.
fn log_one_sided_gaussian_integral<S: JetField>(g: &S, h: &S) -> S {
    let root = sqrt(h);
    let z = div(g, &root);
    ln(&root)
        .neg()
        .add(&ln(&g.constant_like(2.0 * std::f64::consts::PI)).scale(0.5))
        .add(&z.mul(&z).scale(0.5))
        .add(&normal_logcdf(&z))
}

/// The Gaussian function families whose strength limit is a zero-effect
/// model of the same law, among a model's function penalties: every variation
/// block, whose limit leaves a constant function, and the genetic-drive, entry
/// and loading levels. Decoder and disease-jump limits are support and jump
/// flags; the baseline level and the structural laws change the observation or
/// dynamics family instead.
pub fn optional_zero_effects(penalties: &[FunctionPenalty]) -> Vec<FunctionPenalty> {
    let mut families: Vec<FunctionPenalty> = penalties
        .iter()
        .filter(|label| {
            matches!(
                label,
                FunctionPenalty::BaselineVariation { .. }
                    | FunctionPenalty::DecoderVariation { .. }
                    | FunctionPenalty::DriveVariation { .. }
                    | FunctionPenalty::MeasurementVariation { .. }
                    | FunctionPenalty::DriveLevel
                    | FunctionPenalty::GeneticDrive
                    | FunctionPenalty::EntryMean
                    | FunctionPenalty::EntryPrevalence { .. }
                    | FunctionPenalty::MeasurementEffect { .. }
            )
        })
        .cloned()
        .collect();
    families.sort();
    families.dedup();
    families
}

impl JointStructure {
    pub fn signatures(&self) -> usize {
        self.genetic_drive.len()
    }

    /// Checks dimensions and admissibility against the model's marks, genetic
    /// scores and optional zero-effect families. Every signature needs a
    /// decoder connection: the positive softplus decoder breaks every
    /// continuous rotation of the state, leaving label permutations as the
    /// only gauge. A signature seen only through linear measurement loadings
    /// would carry an orthogonal gauge with any other such signature of equal
    /// rate, and its coordinates would not be identified.
    pub fn validate(
        &self,
        marks: &[MarkKind],
        scores: usize,
        families: &[FunctionPenalty],
    ) -> Result<(), EventHistoryError> {
        let k = self.signatures();
        if self.decoder_support.len() != marks.len()
            || self.decoder_support.iter().any(|row| row.len() != k)
            || self.genetic_drive.iter().any(|row| row.len() != scores)
            || self.jumps.len() != marks.len()
            || self
                .jumps
                .iter()
                .zip(marks)
                .any(|(jump, kind)| jump.is_none() != (*kind == MarkKind::Terminal))
        {
            return Err(invalid(
                "structure dimensions do not match the marks, signatures, genetic scores and terminal marks",
            ));
        }
        if k == 0 && self.jumps.iter().any(|jump| *jump == Some(true)) {
            return Err(invalid(
                "a structure without signatures has no disease-jump functions",
            ));
        }
        if (0..k).any(|axis| self.decoder_support.iter().all(|row| !row[axis])) {
            return Err(invalid(
                "every signature needs a decoder connection to a mark",
            ));
        }
        if self.zero_effects.windows(2).any(|pair| pair[0] >= pair[1])
            || self.zero_effects.iter().any(|label| !families.contains(label))
            || optional_zero_effects(families).len() != families.len()
        {
            return Err(invalid(
                "zero effects must be sorted, unique optional Gaussian families of the model",
            ));
        }
        Ok(())
    }

    /// The normalized structure prior, with no tuning constants:
    /// - p(K) = 1/((K+1)(K+2)), a geometric law with its rate integrated out;
    /// - independent beta-binomial laws for connections, drives, jumps and
    ///   zero-effect families;
    /// - connections conditioned on every signature having one.
    pub fn log_prior(
        &self,
        marks: &[MarkKind],
        scores: usize,
        families: &[FunctionPenalty],
    ) -> Result<f64, EventHistoryError> {
        self.log_prior_over(marks, scores, families, &0.0)
    }

    fn log_prior_over<S: JetField>(
        &self,
        marks: &[MarkKind],
        scores: usize,
        families: &[FunctionPenalty],
        zero: &S,
    ) -> Result<S, EventHistoryError> {
        self.validate(marks, scores, families)?;
        let k = self.signatures();
        let connections = self.decoder_support.iter().flatten().filter(|&&v| v).count();
        let drives = self.genetic_drive.iter().flatten().filter(|&&v| v).count();
        let jump_slots = if k == 0 {
            0
        } else {
            self.jumps.iter().filter(|jump| jump.is_some()).count()
        };
        let jumps = self.jumps.iter().filter(|jump| **jump == Some(true)).count();
        Ok(log_rank_prior(zero, k)
            .add(&log_beta_binomial(zero, marks.len() * k, connections))
            .sub(&log_admissible_mass(zero, marks.len(), k))
            .add(&log_beta_binomial(zero, k * scores, drives))
            .add(&log_beta_binomial(zero, jump_slots, jumps))
            .add(&log_beta_binomial(
                zero,
                families.len(),
                self.zero_effects.len(),
            )))
    }

    /// ln K!. The class weight sums the prior times the evidence over the
    /// label orbit of the support, |orbit| p(S) Z_S. Because the prior is
    /// invariant under the stabilizer, Z_S is |Stab| times the integral over
    /// the stabilizer's fundamental domain, and |orbit| |Stab| = K!. No
    /// separation of label basins is assumed.
    pub fn log_label_multiplicity(&self) -> f64 {
        log_factorial(&0.0, self.signatures())
    }

    fn column_key(&self, axis: usize) -> Vec<bool> {
        let mut key: Vec<bool> = self.decoder_support.iter().map(|row| row[axis]).collect();
        key.extend_from_slice(&self.genetic_drive[axis]);
        key
    }

    /// The group sizes of signatures sharing connections and drive rows: the
    /// label permutations preserving the support permute within groups.
    fn stabilizer_groups(&self) -> Vec<usize> {
        let mut keys: Vec<Vec<bool>> = (0..self.signatures()).map(|axis| self.column_key(axis)).collect();
        keys.sort();
        let mut groups = Vec::new();
        let mut start = 0;
        for end in 1..=keys.len() {
            if end == keys.len() || keys[end] != keys[start] {
                groups.push(end - start);
                start = end;
            }
        }
        groups
    }

    /// ln |Stab(S)|, the sum of ln(group size!) over groups.
    pub fn log_stabilizer_order(&self) -> f64 {
        self.stabilizer_groups()
            .into_iter()
            .map(|size| log_factorial(&0.0, size))
            .sum()
    }

    /// Representative of the label class: signature columns in lexicographic
    /// order of (connections over marks, genetic drive row).
    pub fn canonical(&self) -> JointStructure {
        let mut order: Vec<usize> = (0..self.signatures()).collect();
        order.sort_by_key(|&axis| self.column_key(axis));
        JointStructure {
            decoder_support: self
                .decoder_support
                .iter()
                .map(|row| order.iter().map(|&axis| row[axis]).collect())
                .collect(),
            genetic_drive: order
                .iter()
                .map(|&axis| self.genetic_drive[axis].clone())
                .collect(),
            jumps: self.jumps.clone(),
            zero_effects: self.zero_effects.clone(),
        }
    }
}

impl StructurePosterior {
    /// Normalize prior times evidence over the evaluated label classes. Each
    /// class may be evaluated once; a repeated class would be double counted.
    pub fn new(
        evaluated: &[StructureEvidence],
        marks: &[MarkKind],
        scores: usize,
        families: &[FunctionPenalty],
    ) -> Result<Self, EventHistoryError> {
        Self::new_over(evaluated, marks, scores, families, &0.0)
    }
}

impl<S: JetField> StructurePosterior<S> {
    fn new_over(
        evaluated: &[StructureEvidence],
        marks: &[MarkKind],
        scores: usize,
        families: &[FunctionPenalty],
        zero: &S,
    ) -> Result<Self, EventHistoryError> {
        if evaluated.is_empty() {
            return Err(invalid("a structure posterior needs an evaluated structure"));
        }
        let mut classes: Vec<JointStructure> = Vec::with_capacity(evaluated.len());
        let mut log_class_weights = Vec::with_capacity(evaluated.len());
        for value in evaluated {
            if !value.log_basin_evidence.is_finite()
                || !value.log_evidence_error.is_finite()
                || value.log_evidence_error < 0.0
            {
                return Err(invalid(
                    "structure evidence and its error estimate must be finite and nonnegative",
                ));
            }
            // A trivial stabilizer's fundamental domain is the whole space.
            if !(value.log_domain_mass.is_finite() && value.log_domain_mass <= 0.0)
                || (value.structure.stabilizer_groups().iter().all(|&size| size == 1)
                    && value.log_domain_mass != 0.0)
            {
                return Err(invalid(
                    "a label-domain mass must be a probability, and exactly one without a label stabilizer",
                ));
            }
            let class = value.structure.canonical();
            if classes.contains(&class) {
                return Err(invalid(
                    "a structure label class was evaluated more than once",
                ));
            }
            log_class_weights.push(
                value
                    .structure
                    .log_prior_over(marks, scores, families, zero)?
                    .add(&log_factorial(zero, value.structure.signatures()))
                    .add(&constant(zero, value.log_domain_mass))
                    .add(&constant(zero, value.log_basin_evidence)),
            );
            classes.push(class);
        }
        let log_total = log_sum_exp(&log_class_weights);
        if !log_total.value().is_finite() {
            return Err(numerical("structure posterior normalizer is not representable"));
        }
        Ok(Self {
            log_weights: log_class_weights.iter().map(|w| w.sub(&log_total)).collect(),
            log_class_weights,
            errors: evaluated.iter().map(|v| v.log_evidence_error).collect(),
        })
    }

    pub fn log_weights(&self) -> &[S] {
        &self.log_weights
    }

    /// Each class's log-evidence error estimate `d_s`, in evaluation order.
    pub fn log_evidence_errors(&self) -> &[f64] {
        &self.errors
    }

    /// If every log evidence changes by at most its error estimate `d_s`,
    /// each normalized weight changes by a factor within exp(+-(d_s + max d)).
    /// These are estimates, not deterministic certificates.
    pub fn largest_log_weight_error(&self) -> f64 {
        let largest = self.errors.iter().copied().fold(0.0_f64, f64::max);
        self.errors
            .iter()
            .map(|d| d + largest)
            .fold(0.0_f64, f64::max)
    }

    /// The order of two evaluated classes when their gap exceeds both error
    /// estimates; `None` means both evidences need refinement before the
    /// comparison is decided.
    pub fn resolved_order(&self, left: usize, right: usize) -> Option<Ordering> {
        let gap = self.log_class_weights[left].value() - self.log_class_weights[right].value();
        (gap.abs() > self.errors[left] + self.errors[right]).then(|| gap.total_cmp(&0.0))
    }
}

/// Savage–Dickey Bayes factor of the larger structure against its face
/// `pi_dk = 0`. Signature categories carry Dirichlet weight one, so the face
/// conditional of the larger prior is exactly the smaller structure's prior,
/// and the prior marginal of `pi_dk` is Beta(1, C + lambda) with density
/// `C + lambda` at zero. The posterior density at the face comes from a
/// one-sided marginal expansion `log p(x | D) = c + g x - h x^2/2` of the
/// larger model at `x = pi_dk = 0`. This ORDERS candidates; a candidate's own
/// resolved evidence decides it.
pub fn connection_log_bayes_factor(
    face_score: f64,
    face_curvature: f64,
    signature_categories: usize,
    log_strength: f64,
) -> Result<f64, EventHistoryError> {
    connection_log_bayes_factor_over(&face_score, &face_curvature, signature_categories, &log_strength)
}

fn connection_log_bayes_factor_over<S: JetField>(
    face_score: &S,
    face_curvature: &S,
    signature_categories: usize,
    log_strength: &S,
) -> Result<S, EventHistoryError> {
    if signature_categories == 0
        || !face_score.value().is_finite()
        || !log_strength.value().is_finite()
        || !(face_curvature.value().is_finite() && face_curvature.value() > 0.0)
    {
        return Err(invalid(
            "a connection Bayes factor needs a signature category, a finite face score and strength, and positive face curvature",
        ));
    }
    let log_prior_density = log_sum_exp(&[
        ln(&constant(face_score, signature_categories as f64)),
        log_strength.clone(),
    ]);
    let value =
        log_prior_density.add(&log_one_sided_gaussian_integral(face_score, face_curvature));
    if !value.value().is_finite() {
        return Err(numerical("connection Bayes factor is not representable"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::super::function_prior_tests::{compensated_sum, rule};
    use super::*;
    use crate::scalar::exp;
    use crate::test_support::{Bound, agrees};

    fn families() -> Vec<FunctionPenalty> {
        vec![
            FunctionPenalty::BaselineVariation { penalty: 0 },
            FunctionPenalty::MeasurementEffect { channel: 0 },
        ]
    }

    fn enumerate<'a>(
        marks: &'a [MarkKind],
        signatures: usize,
        scores: usize,
        families: &'a [FunctionPenalty],
    ) -> impl Iterator<Item = JointStructure> + 'a {
        let connections = marks.len() * signatures;
        let drives = signatures * scores;
        let slots: Vec<usize> = (0..marks.len())
            .filter(|&d| marks[d] != MarkKind::Terminal)
            .collect();
        let jump_bits = if signatures == 0 { 0 } else { slots.len() };
        let total = connections + drives + jump_bits + families.len();
        (0..1usize << total).map(move |bits| {
            let bit = |j: usize| bits >> j & 1 == 1;
            let mut jumps: Vec<Option<bool>> = marks
                .iter()
                .map(|kind| (*kind != MarkKind::Terminal).then_some(false))
                .collect();
            for (index, &d) in slots.iter().enumerate().take(jump_bits) {
                jumps[d] = Some(bit(connections + drives + index));
            }
            JointStructure {
                decoder_support: (0..marks.len())
                    .map(|d| (0..signatures).map(|k| bit(d * signatures + k)).collect())
                    .collect(),
                genetic_drive: (0..signatures)
                    .map(|k| (0..scores).map(|g| bit(connections + k * scores + g)).collect())
                    .collect(),
                jumps,
                zero_effects: families
                    .iter()
                    .enumerate()
                    .filter(|(f, _)| bit(connections + drives + jump_bits + f))
                    .map(|(_, label)| label.clone())
                    .collect(),
            }
        })
    }

    #[test]
    fn structure_prior_normalizes_over_admissible_structures_at_every_rank() {
        // Measured bar: the admissible mass passes ln_1p through compose_unary, whose one-ulp
        // charge has no cited accuracy.
        let marks = [MarkKind::Recurrent, MarkKind::Once, MarkKind::Terminal];
        let families = families();
        let zero = Bound::exact(0.0);
        for scores in [1, 2] {
            for k in 0..=2 {
                let mut admissible = 0;
                let mut terms = Vec::new();
                let mut error = 0.0;
                for structure in enumerate(&marks, k, scores, &families) {
                    if structure.validate(&marks, scores, &families).is_ok() {
                        admissible += 1;
                        let term = exp(&structure
                            .log_prior_over(&marks, scores, &families, &zero)
                            .unwrap());
                        error += term.rounding();
                        terms.push(term.value);
                    } else {
                        assert!(structure.log_prior(&marks, scores, &families).is_err());
                    }
                }
                let (mass, summation) = compensated_sum(&terms);
                let expected = div(&Bound::exact(1.0), &Bound::exact(((k + 1) * (k + 2)) as f64));
                assert!(
                    (mass - expected.value).abs() <= error + summation + expected.rounding(),
                    "scores {scores}, K {k}: {mass} vs {}",
                    expected.value
                );
                if k == 2 && scores == 1 {
                    // 64 connection patterns, 49 with both columns nonempty, times four drive,
                    // four jump and four zero-effect patterns.
                    assert_eq!(admissible, 49 * 16 * 4);
                }
            }
        }
        // sum_{K=0}^{999} 1/((K+1)(K+2)) telescopes to 1 - 1/1001.
        let terms: Vec<Bound> = (0..1000).map(|k| exp(&log_rank_prior(&zero, k))).collect();
        let (head, summation) = compensated_sum(&terms.iter().map(|t| t.value).collect::<Vec<_>>());
        let error: f64 = terms.iter().map(Bound::rounding).sum();
        let expected = Bound::exact(1.0).sub(&div(&Bound::exact(1.0), &Bound::exact(1001.0)));
        assert!((head - expected.value).abs() <= error + summation + expected.rounding());
    }

    #[test]
    fn variation_blocks_and_levels_are_the_optional_zero_effect_families() {
        let penalties = [
            FunctionPenalty::Decoder { mark: 0 },
            FunctionPenalty::DecoderVariation { penalty: 1 },
            FunctionPenalty::BaselineVariation { penalty: 0 },
            FunctionPenalty::DriveLevel,
            FunctionPenalty::GeneticDrive,
            FunctionPenalty::DriveVariation { penalty: 0 },
            FunctionPenalty::EntryMean,
            FunctionPenalty::EntryPrevalence { mark: 1 },
            FunctionPenalty::DiseaseJump { mark: 0 },
            FunctionPenalty::MeasurementEffect { channel: 2 },
            FunctionPenalty::MeasurementVariation { penalty: 0 },
            FunctionPenalty::BaselineLevel,
            FunctionPenalty::TemporalVariation,
            FunctionPenalty::MeasurementPrecision { channel: 0 },
            FunctionPenalty::TailVarianceInflation { channel: 0 },
            FunctionPenalty::CountOverdispersion { channel: 3 },
            FunctionPenalty::CountMean { channel: 3 },
        ];
        let mut expected = vec![
            FunctionPenalty::DecoderVariation { penalty: 1 },
            FunctionPenalty::BaselineVariation { penalty: 0 },
            FunctionPenalty::DriveLevel,
            FunctionPenalty::GeneticDrive,
            FunctionPenalty::DriveVariation { penalty: 0 },
            FunctionPenalty::EntryMean,
            FunctionPenalty::EntryPrevalence { mark: 1 },
            FunctionPenalty::MeasurementEffect { channel: 2 },
            FunctionPenalty::MeasurementVariation { penalty: 0 },
        ];
        expected.sort();
        assert_eq!(optional_zero_effects(&penalties), expected);
        // A model's families must be exactly its optional ones.
        let marks = [MarkKind::Recurrent];
        let structure = JointStructure {
            decoder_support: vec![vec![]],
            genetic_drive: vec![],
            jumps: vec![Some(false)],
            zero_effects: vec![],
        };
        assert!(structure.validate(&marks, 0, &expected).is_ok());
        assert!(structure.validate(&marks, 0, &penalties).is_err());
    }

    #[test]
    fn label_permutations_share_a_class_prior_and_are_counted_once() {
        // Measured bar: the class prior's admissible mass passes ln_1p through compose_unary,
        // whose one-ulp charge has no cited accuracy.
        let marks = [MarkKind::Recurrent, MarkKind::Once];
        let families = families();
        let structure = JointStructure {
            decoder_support: vec![vec![true, false, true], vec![false, true, true]],
            genetic_drive: vec![vec![true, false], vec![false, false], vec![false, true]],
            jumps: vec![Some(true), Some(false)],
            zero_effects: vec![FunctionPenalty::MeasurementEffect { channel: 0 }],
        };
        // Signature columns reordered as (2, 0, 1), with their drive rows.
        let permuted = JointStructure {
            decoder_support: vec![vec![true, true, false], vec![true, false, true]],
            genetic_drive: vec![vec![false, true], vec![true, false], vec![false, false]],
            jumps: vec![Some(true), Some(false)],
            zero_effects: vec![FunctionPenalty::MeasurementEffect { channel: 0 }],
        };
        assert_ne!(structure, permuted);
        assert_eq!(structure.canonical(), permuted.canonical());
        assert_eq!(
            structure.log_prior(&marks, 2, &families).unwrap(),
            permuted.log_prior(&marks, 2, &families).unwrap()
        );
        agrees(
            &log_factorial(&Bound::exact(0.0), 3),
            &ln(&Bound::exact(6.0)),
            "log 3!",
        );
        assert_eq!(structure.log_stabilizer_order(), 0.0);
        let twins = JointStructure {
            decoder_support: vec![vec![true, true, false], vec![false, false, true]],
            genetic_drive: vec![vec![true], vec![true], vec![false]],
            jumps: vec![Some(false), Some(false)],
            zero_effects: vec![],
        };
        assert_eq!(twins.stabilizer_groups(), vec![1, 2]);
        let measurement_only = JointStructure {
            decoder_support: vec![vec![true, false], vec![false, false]],
            genetic_drive: vec![vec![], vec![]],
            jumps: vec![Some(false), Some(false)],
            zero_effects: vec![],
        };
        assert!(measurement_only.validate(&marks, 0, &families).is_err());
        let mut unsorted = structure.clone();
        unsorted.zero_effects = families.iter().rev().cloned().collect();
        assert!(unsorted.validate(&marks, 2, &families).is_err());
        let evidence = |structure: &JointStructure, value: f64| StructureEvidence {
            structure: structure.clone(),
            log_basin_evidence: value,
            log_domain_mass: 0.0,
            log_evidence_error: 0.1,
        };
        assert!(
            StructurePosterior::new(
                &[evidence(&structure, -3.0), evidence(&permuted, -2.0)],
                &marks,
                2,
                &families,
            )
            .is_err()
        );
        let mut shifted = evidence(&structure, -3.0);
        shifted.log_domain_mass = -0.5;
        assert!(StructurePosterior::new(&[shifted], &marks, 2, &families).is_err());
        let smaller = JointStructure {
            decoder_support: vec![vec![true], vec![false]],
            genetic_drive: vec![vec![false, false]],
            jumps: vec![Some(false), Some(false)],
            zero_effects: vec![],
        };
        let zero = Bound::exact(0.0);
        let larger_class = structure
            .log_prior_over(&marks, 2, &families, &zero)
            .unwrap()
            .add(&log_factorial(&zero, 3))
            .add(&Bound::exact(-3.0));
        let smaller_prior = smaller.log_prior_over(&marks, 2, &families, &zero).unwrap();
        // Each evidence carries error 0.1, so only a class gap above 0.2 is decided.
        for (gap, decided) in [(0.15_f64, None), (0.5, Some(Ordering::Greater))] {
            let gap = Bound::exact(gap);
            let smaller_evidence = larger_class.sub(&gap).sub(&smaller_prior);
            let evaluated = [
                evidence(&structure, -3.0),
                evidence(&smaller, smaller_evidence.value),
            ];
            let posterior = StructurePosterior::new_over(&evaluated, &marks, 2, &families, &zero)
                .unwrap();
            let weights = posterior.log_weights();
            let total = exp(&weights[0]).add(&exp(&weights[1]));
            agrees(&total, &Bound::exact(1.0), &format!("total weight, gap {}", gap.value));
            // The smaller class's evidence was rounded once while constructing it.
            let difference = weights[0].sub(&weights[1]);
            assert!(
                (difference.value - gap.value).abs()
                    <= difference.rounding() + smaller_evidence.rounding()
            );
            let production = StructurePosterior::new(&evaluated, &marks, 2, &families).unwrap();
            assert_eq!(production.resolved_order(0, 1), decided);
            assert_eq!(
                production.resolved_order(1, 0),
                decided.map(Ordering::reverse)
            );
            assert_eq!(production.log_evidence_errors(), &[0.1, 0.1]);
            assert_eq!(production.largest_log_weight_error(), 0.1 + 0.1);
        }
    }

    #[test]
    fn connection_bayes_factor_is_exact_for_a_truncated_gaussian_face_posterior() {
        // Let the likelihood cancel the Beta(1, C+lambda) prior marginal's shape, leaving
        // posterior (C+lambda) exp(a x - b x^2/2) on [0, 1]. The reduced model fixes x = 0 with
        // likelihood one, so the exact Bayes factor is integral_0^1 (C+lambda) exp(a x - b x^2/2).
        let integral = |order: usize, a: f64, b: f64| {
            let r = rule(order, 0.0, 1.0);
            let mut values = Vec::with_capacity(order);
            let mut error = 0.0;
            for (x, w) in r.points.iter().zip(&r.weights) {
                let term = exp(&x.scale(a).sub(&x.mul(x).scale(0.5 * b))).mul(w);
                // Own account, the weight's certified error, and the point's displacement through
                // |d log integrand/dx| = |a - b x|.
                error += term.rounding()
                    + term.value * (r.weight_relative_error + (a - b * x.value).abs() * r.point_error);
                values.push(term.value);
            }
            let (total, summation) = compensated_sum(&values);
            (total, error + summation)
        };
        for (a, b, categories, rho) in [
            (40.0_f64, 200.0_f64, 1, 0.0_f64),
            (-3.0, 400.0, 3, 1.5),
            (-80.0, 50.0, 2, -2.0),
            (0.0, 900.0, 5, 4.0),
        ] {
            let (coarse, _) = integral(257, a, b);
            let (fine, rounding) = integral(513, a, b);
            // Beyond x = 1, a x - b x^2/2 <= a - b/2 + (a - b)(x - 1) for b > a, so the tail is
            // below exp(a - b/2)/(b - a).
            let tail = (a - 0.5 * b).exp() / (b - a);
            let value = connection_log_bayes_factor_over(
                &Bound::exact(a),
                &Bound::exact(b),
                categories,
                &Bound::exact(rho),
            )
            .unwrap();
            let prior = Bound::exact(categories as f64).add(&exp(&Bound::exact(rho)));
            let exact = ln(&prior).add(&Bound::exact(0.0).compose_unary([fine.ln(), 0.0, 0.0, 0.0, 0.0]));
            // Relative quadrature error (order doubling, tail, rounding) becomes absolute in the
            // log. The production route's normal log-CDF charges one rounding of its value, a
            // measured accuracy: that special function has no derived accuracy statement.
            let bar = ((fine - coarse).abs() + tail + rounding) / fine
                + value.rounding()
                + exact.rounding();
            assert!(
                exact.value.abs() > bar,
                "{a},{b}: exact log Bayes factor {} is below its bar {bar}",
                exact.value
            );
            assert!(
                (value.value - exact.value).abs() <= bar,
                "{a},{b}: {} vs {} within {bar}",
                value.value,
                exact.value
            );
        }
        assert!(connection_log_bayes_factor(1.0, 0.0, 1, 0.0).is_err());
        assert!(connection_log_bayes_factor(1.0, 1.0, 0, 0.0).is_err());
    }

    #[test]
    fn admissibility_mass_matches_direct_enumeration_of_nonempty_columns() {
        // Measured bar: the admissible mass passes ln_1p through compose_unary, whose one-ulp
        // charge has no cited accuracy.
        let zero = Bound::exact(0.0);
        for marks in 1..=3 {
            for k in 1..=3 {
                let n = marks * k;
                let mut terms = Vec::new();
                let mut error = 0.0;
                for bits in 0..1usize << n {
                    if (0..k).all(|axis| (0..marks).any(|d| bits >> (d * k + axis) & 1 == 1)) {
                        let term = exp(&log_beta_binomial(&zero, n, bits.count_ones() as usize));
                        error += term.rounding();
                        terms.push(term.value);
                    }
                }
                let (mass, summation) = compensated_sum(&terms);
                let expected = exp(&log_admissible_mass(&zero, marks, k));
                assert!(
                    (mass - expected.value).abs() <= error + summation + expected.rounding(),
                    "marks {marks}, K {k}: {mass} vs {}",
                    expected.value
                );
            }
        }
    }
}
