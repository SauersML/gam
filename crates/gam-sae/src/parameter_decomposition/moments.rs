//! Mask-moment geometry of an executable parameter decomposition (#2951 P8, P10).
//!
//! # Moments (P8)
//!
//! A decomposition edits its tensors through deletion amounts `t_c = 1 − m_c`, and the edited
//! tensors depend on the mask only through the moment `q = Σ_c t_c·v_c`. Under the anchored lift
//! `Θ(m) = m_Δ·Θ* + B·Σ_c (m_c − m_Δ)·v_c`, this needs `m_Δ = 1`. When the residual mask is a
//! control, `Θ(m) = Θ* − t_Δ·(Θ* − B·Σ_c v_c) − B·q`, so it enters as one more generator, the scalar
//! `t_Δ` in a one-dimensional block.
//!
//! A control's generator lives in one or more [`MomentBlock`]s. Occurrence-level masks give
//! separate blocks, and a control that edits every tied use of a tensor carries a part in each
//! use's block.
//!
//! The mask domain is a declared experiment input with no default (#2951 SPEC tension 2): a
//! product of intervals `m_c ∈ [lower_c, upper_c]` that contain the all-on value 1. With the kept
//! set `S` pinned at all-on, the admissible moments form the zonotope
//! `Z_S = Σ_{c∉S} [(1 − upper_c)·v_c, (1 − lower_c)·v_c]`, which is the image of the box under
//! `m ↦ q`. So the supremum of any discrepancy over masks EQUALS its supremum over `Z_S`. The
//! network stays nonlinear; nothing here linearizes it.
//!
//! # Support function and witness mask (P8)
//!
//! `h(u) = Σ_{c∉S} max((1 − upper_c)·⟨u, v_c⟩, (1 − lower_c)·⟨u, v_c⟩)`. It is attained by the
//! endpoint mask that deletes most where the computed `⟨u, v_c⟩ > 0` and least elsewhere. A
//! control whose pairing lies inside its roundoff band ([`accumulation_band`]) is reported
//! unresolved: both of its endpoints attain the support to within the band, so a derivative taken
//! at the fixed witness (Danskin, P16) is one-sided there.
//!
//! # Fragmentation (P10)
//!
//! Splitting a cancelling pair `±a` into `n` pieces with independent uniform masks drives the mean
//! square of the error to `a²/(6n)`, while `sup|E_n| = |a|`. The expectation under a mask law and a
//! certified supremum are different numbers. A support value is the computed sum with a derived
//! band that contains the exact support of the stored generators; no expectation under a mask law
//! is ever reported as one.

use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_band, accumulation_growth};
use ndarray::Array1;

/// One moment coordinate space: the coefficient space of one edited tensor use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MomentBlock {
    pub dimension: usize,
}

/// A control's generator in one block.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratorPart {
    pub block: usize,
    pub vector: Array1<f64>,
}

/// A moment, or a direction paired with moments, laid out block by block.
#[derive(Clone, Debug, PartialEq)]
pub struct MomentVector {
    pub blocks: Vec<Array1<f64>>,
}

/// Where a witness puts one control's mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WitnessEndpoint {
    /// Pinned at all-on, `m_c = 1`, as a member of the kept set.
    Kept,
    /// `m_c = lower_c`, the most deletion the declared domain admits.
    Lower,
    /// `m_c = upper_c`, the least deletion the declared domain admits.
    Upper,
}

/// A malformed moment system, domain, mask or direction.
#[derive(Clone, Debug, PartialEq)]
pub enum MomentGeometryError {
    GeneratorBlockOutOfRange { control: usize, block: usize, blocks: usize },
    RepeatedGeneratorBlock { control: usize, block: usize },
    GeneratorDimension { control: usize, block: usize, expected: usize, found: usize },
    NonFiniteGenerator { control: usize, block: usize },
    MaskDomainExcludesAllOn { control: usize, lower: f64, upper: f64 },
    ControlCount { expected: usize, found: usize },
    MaskOutsideDomain { control: usize, value: f64, lower: f64, upper: f64 },
    DirectionBlockCount { expected: usize, found: usize },
    DirectionDimension { block: usize, expected: usize, found: usize },
    NonFiniteDirection { block: usize },
}

/// The declared product mask domain `m_c ∈ [lower_c, upper_c]`, one interval per control.
///
/// It is an experiment declaration, not a search box, and has no default (#2951 SPEC tension 2).
#[derive(Clone, Debug, PartialEq)]
pub struct MaskDomain {
    intervals: Vec<(f64, f64)>,
}

impl MaskDomain {
    /// Every interval must be finite and contain the all-on value 1, so that the kept set is
    /// admissible.
    pub fn new(intervals: Vec<(f64, f64)>) -> Result<Self, MomentGeometryError> {
        for (control, &(lower, upper)) in intervals.iter().enumerate() {
            if !(lower.is_finite() && upper.is_finite() && lower <= 1.0 && upper >= 1.0) {
                return Err(MomentGeometryError::MaskDomainExcludesAllOn {
                    control,
                    lower,
                    upper,
                });
            }
        }
        Ok(Self { intervals })
    }

    pub fn control_count(&self) -> usize {
        self.intervals.len()
    }

    pub fn interval(&self, control: usize) -> Option<(f64, f64)> {
        self.intervals.get(control).copied()
    }

    /// The mask values a witness names.
    pub fn mask_at(&self, witness: &[WitnessEndpoint]) -> Result<Vec<f64>, MomentGeometryError> {
        if witness.len() != self.intervals.len() {
            return Err(MomentGeometryError::ControlCount {
                expected: self.intervals.len(),
                found: witness.len(),
            });
        }
        Ok(witness
            .iter()
            .zip(&self.intervals)
            .map(|(endpoint, &(lower, upper))| match endpoint {
                WitnessEndpoint::Kept => 1.0,
                WitnessEndpoint::Lower => lower,
                WitnessEndpoint::Upper => upper,
            })
            .collect())
    }

    /// The least and the most deletion, `(1 − upper, 1 − lower)`.
    fn deletion_range(&self, control: usize) -> (f64, f64) {
        let (lower, upper) = self.intervals[control];
        (1.0 - upper, 1.0 - lower)
    }
}

/// The support `h(u)` of the admissible zonotope, with a mask attaining it.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportEvaluation {
    /// The computed support.
    pub value: f64,
    /// Radius around `value` containing the exact support of the stored generators. It covers the
    /// pairings' roundoff bands, the rounding of each term and of their sum, and, for every
    /// unresolved control, the most its endpoint choice can lose.
    pub band: f64,
    /// A mask attaining `value`: most deletion where the computed pairing is positive, least
    /// elsewhere.
    pub witness: Vec<WitnessEndpoint>,
    /// Free controls whose pairing lies inside its roundoff band, so either endpoint attains the
    /// support to within `band`.
    pub unresolved: Vec<usize>,
}

/// Controls, their generators and the moment blocks those generators live in (#2951 P8).
#[derive(Clone, Debug, PartialEq)]
pub struct MaskMomentSystem {
    blocks: Vec<MomentBlock>,
    generators: Vec<Vec<GeneratorPart>>,
}

impl MaskMomentSystem {
    /// `generators[c]` lists control `c`'s parts, at most one per block.
    pub fn new(
        blocks: Vec<MomentBlock>,
        generators: Vec<Vec<GeneratorPart>>,
    ) -> Result<Self, MomentGeometryError> {
        for (control, parts) in generators.iter().enumerate() {
            let mut seen = vec![false; blocks.len()];
            for part in parts {
                let block = blocks.get(part.block).ok_or(MomentGeometryError::GeneratorBlockOutOfRange {
                    control,
                    block: part.block,
                    blocks: blocks.len(),
                })?;
                if seen[part.block] {
                    return Err(MomentGeometryError::RepeatedGeneratorBlock {
                        control,
                        block: part.block,
                    });
                }
                seen[part.block] = true;
                if part.vector.len() != block.dimension {
                    return Err(MomentGeometryError::GeneratorDimension {
                        control,
                        block: part.block,
                        expected: block.dimension,
                        found: part.vector.len(),
                    });
                }
                if part.vector.iter().any(|value| !value.is_finite()) {
                    return Err(MomentGeometryError::NonFiniteGenerator {
                        control,
                        block: part.block,
                    });
                }
            }
        }
        Ok(Self { blocks, generators })
    }

    pub fn blocks(&self) -> &[MomentBlock] {
        &self.blocks
    }

    pub fn control_count(&self) -> usize {
        self.generators.len()
    }

    pub fn generator(&self, control: usize) -> Option<&[GeneratorPart]> {
        self.generators.get(control).map(Vec::as_slice)
    }

    /// The moment `q(m) = Σ_c (1 − m_c)·v_c` of a mask inside the declared domain.
    pub fn moment(&self, domain: &MaskDomain, mask: &[f64]) -> Result<MomentVector, MomentGeometryError> {
        self.check_controls(domain, mask.len())?;
        let mut blocks: Vec<Array1<f64>> = self
            .blocks
            .iter()
            .map(|block| Array1::zeros(block.dimension))
            .collect();
        for (control, (&value, parts)) in mask.iter().zip(&self.generators).enumerate() {
            let (lower, upper) = domain.intervals[control];
            if !(lower <= value && value <= upper) {
                return Err(MomentGeometryError::MaskOutsideDomain {
                    control,
                    value,
                    lower,
                    upper,
                });
            }
            let deletion = 1.0 - value;
            for part in parts {
                blocks[part.block].scaled_add(deletion, &part.vector);
            }
        }
        Ok(MomentVector { blocks })
    }

    /// The support function of `Z_S` in `direction`, with its witness mask (#2951 P8).
    ///
    /// `kept[c]` pins control `c` at all-on.
    pub fn support(
        &self,
        domain: &MaskDomain,
        kept: &[bool],
        direction: &MomentVector,
    ) -> Result<SupportEvaluation, MomentGeometryError> {
        self.check_controls(domain, kept.len())?;
        self.check_direction(direction)?;
        let mut value = 0.0;
        let mut absolute_terms = 0.0;
        let mut band = 0.0;
        let mut free = 0usize;
        let mut witness = Vec::with_capacity(self.generators.len());
        let mut unresolved = Vec::new();
        for control in 0..self.generators.len() {
            if kept[control] {
                witness.push(WitnessEndpoint::Kept);
                continue;
            }
            let (pairing, pairing_band) = self.pairing(control, direction);
            let (least, most) = domain.deletion_range(control);
            let (endpoint, deletion) = if pairing > 0.0 {
                (WitnessEndpoint::Lower, most)
            } else {
                (WitnessEndpoint::Upper, least)
            };
            let term = deletion * pairing;
            value += term;
            absolute_terms += term.abs();
            free += 1;
            // The deletion amount and the product each round once; the pairing's own error scales by
            // the deletion.
            band += 2.0 * UNIT_ROUNDOFF * term.abs() + deletion.abs() * pairing_band;
            if pairing.abs() <= pairing_band {
                unresolved.push(control);
                // The exact pairing is at most twice the band in magnitude, so the other endpoint
                // gains at most that much over the interval's length.
                band += 2.0 * (most - least) * pairing_band;
            }
            witness.push(endpoint);
        }
        // The terms were formed before summing, so the sum commits `free − 1` rounded additions.
        band += accumulation_growth(free.saturating_sub(1)) * absolute_terms;
        Ok(SupportEvaluation {
            value,
            band,
            witness,
            unresolved,
        })
    }

    fn check_controls(&self, domain: &MaskDomain, found: usize) -> Result<(), MomentGeometryError> {
        let expected = self.generators.len();
        for found in [domain.control_count(), found] {
            if found != expected {
                return Err(MomentGeometryError::ControlCount { expected, found });
            }
        }
        Ok(())
    }

    fn check_direction(&self, direction: &MomentVector) -> Result<(), MomentGeometryError> {
        if direction.blocks.len() != self.blocks.len() {
            return Err(MomentGeometryError::DirectionBlockCount {
                expected: self.blocks.len(),
                found: direction.blocks.len(),
            });
        }
        for (block, (spec, covector)) in self.blocks.iter().zip(&direction.blocks).enumerate() {
            if covector.len() != spec.dimension {
                return Err(MomentGeometryError::DirectionDimension {
                    block,
                    expected: spec.dimension,
                    found: covector.len(),
                });
            }
            if covector.iter().any(|value| !value.is_finite()) {
                return Err(MomentGeometryError::NonFiniteDirection { block });
            }
        }
        Ok(())
    }

    /// `⟨u, v_c⟩` and its roundoff band: an inner product over every part's entries.
    fn pairing(&self, control: usize, direction: &MomentVector) -> (f64, f64) {
        let mut product = 0.0;
        let mut absolute = 0.0;
        let mut terms = 0;
        for part in &self.generators[control] {
            for (&generator, &covector) in part.vector.iter().zip(direction.blocks[part.block].iter()) {
                let term = generator * covector;
                product += term;
                absolute += term.abs();
                terms += 1;
            }
        }
        (product, accumulation_band(terms, absolute))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn part(block: usize, vector: &[f64]) -> GeneratorPart {
        GeneratorPart {
            block,
            vector: Array1::from(vector.to_vec()),
        }
    }

    fn block(dimension: usize) -> MomentBlock {
        MomentBlock { dimension }
    }

    /// Integer generators in a 2-D and a 1-D block, so every moment, pairing and vertex below is
    /// computed without rounding and the comparisons are exact. Control 1 is tied across both
    /// blocks, 4 is a positive multiple of 0, 5 is antiparallel to 0, 6 is zero, and control 2
    /// takes a signed mask.
    fn integer_fixture() -> (MaskMomentSystem, MaskDomain) {
        let system = MaskMomentSystem::new(
            vec![block(2), block(1)],
            vec![
                vec![part(0, &[3.0, 1.0])],
                vec![part(0, &[-1.0, 2.0]), part(1, &[1.0])],
                vec![part(0, &[2.0, -2.0])],
                vec![part(1, &[-3.0])],
                vec![part(0, &[6.0, 2.0])],
                vec![part(0, &[-3.0, -1.0])],
                vec![part(0, &[0.0, 0.0]), part(1, &[0.0])],
            ],
        )
        .expect("well-formed generators");
        let mut intervals = vec![(0.0, 1.0); 7];
        intervals[2] = (-1.0, 1.0);
        (system, MaskDomain::new(intervals).expect("every interval contains all-on"))
    }

    /// A single 2-D block with integer generators and one signed control; (1, 2) and (2, 4) share a
    /// direction.
    fn planar_fixture() -> (MaskMomentSystem, MaskDomain) {
        let generators = [[1.0, 2.0], [-2.0, 1.0], [3.0, -1.0], [0.0, 2.0], [-1.0, -1.0], [2.0, 4.0], [1.0, 0.0], [-3.0, 2.0]];
        let system = MaskMomentSystem::new(
            vec![block(2)],
            generators.iter().map(|vector| vec![part(0, vector)]).collect(),
        )
        .expect("well-formed generators");
        let mut intervals = vec![(0.0, 1.0); 8];
        intervals[6] = (-1.0, 1.0);
        (system, MaskDomain::new(intervals).expect("every interval contains all-on"))
    }

    /// Every endpoint mask of the declared domain, with the kept controls at all-on.
    fn endpoint_masks(domain: &MaskDomain, kept: &[bool]) -> Vec<Vec<f64>> {
        let free: Vec<usize> = (0..kept.len()).filter(|&control| !kept[control]).collect();
        (0..1usize << free.len())
            .map(|bits| {
                let mut mask = vec![1.0; kept.len()];
                for (position, &control) in free.iter().enumerate() {
                    let (lower, upper) = domain.interval(control).expect("control in range");
                    mask[control] = if (bits >> position) & 1 == 1 { lower } else { upper };
                }
                mask
            })
            .collect()
    }

    fn pair(moment: &MomentVector, direction: &MomentVector) -> f64 {
        moment
            .blocks
            .iter()
            .zip(&direction.blocks)
            .map(|(q, u)| q.dot(u))
            .sum()
    }

    fn covector(values: [f64; 3]) -> MomentVector {
        MomentVector {
            blocks: vec![array![values[0], values[1]], array![values[2]]],
        }
    }

    fn planar_covector(values: [f64; 2]) -> MomentVector {
        MomentVector {
            blocks: vec![array![values[0], values[1]]],
        }
    }

    #[test]
    fn support_equals_the_exhaustive_endpoint_maximum_and_bounds_interior_masks_2951() {
        let (system, domain) = integer_fixture();
        let directions = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, -1.0, 3.0],
            [-1.0, -4.0, 1.0],
            [1.0, -3.0, -2.0],
            [0.0, 0.0, 0.0],
        ];
        for kept in [vec![false; 7], vec![false, false, false, true, false, true, false]] {
            let masks = endpoint_masks(&domain, &kept);
            for values in directions {
                let direction = covector(values);
                let support = system.support(&domain, &kept, &direction).expect("valid inputs");
                let exhaustive = masks
                    .iter()
                    .map(|mask| pair(&system.moment(&domain, mask).expect("endpoint mask admissible"), &direction))
                    .fold(f64::NEG_INFINITY, f64::max);
                assert_eq!(support.value, exhaustive, "direction {values:?}, kept {kept:?}");
                let witness_mask = domain.mask_at(&support.witness).expect("one endpoint per control");
                let attained = pair(&system.moment(&domain, &witness_mask).expect("witness admissible"), &direction);
                assert_eq!(attained, support.value, "the witness attains the support");
                for (control, &is_kept) in kept.iter().enumerate() {
                    assert_eq!(support.witness[control] == WitnessEndpoint::Kept, is_kept);
                }
                // Interior dyadic masks are exact too, and no admissible mask exceeds the support.
                for shade in [0.25, 0.5, 0.75] {
                    let mask: Vec<f64> = (0..7)
                        .map(|control| {
                            if kept[control] {
                                1.0
                            } else {
                                let (lower, upper) = domain.interval(control).expect("control in range");
                                lower + shade * (upper - lower)
                            }
                        })
                        .collect();
                    let interior = pair(&system.moment(&domain, &mask).expect("interior admissible"), &direction);
                    assert!(interior <= support.value, "interior {interior} exceeds support {}", support.value);
                }
            }
        }
        // Positive control: a support that treats the signed control's interval as [0, 1] undershoots
        // the exhaustive maximum where that control's pairing is positive.
        let direction = covector([1.0, -1.0, 0.0]);
        let unit_intervals: f64 = (0..7)
            .map(|control| {
                system
                    .generator(control)
                    .expect("control in range")
                    .iter()
                    .map(|part| part.vector.dot(&direction.blocks[part.block]))
                    .sum::<f64>()
                    .max(0.0)
            })
            .sum();
        let support = system.support(&domain, &[false; 7], &direction).expect("valid inputs");
        assert_eq!((unit_intervals, support.value), (10.0, 14.0));
        // A mask outside the declared domain is refused rather than silently reduced.
        let mut outside = vec![1.0; 7];
        outside[0] = -1.0;
        assert_eq!(
            system.moment(&domain, &outside),
            Err(MomentGeometryError::MaskOutsideDomain {
                control: 0,
                value: -1.0,
                lower: 0.0,
                upper: 1.0
            })
        );
    }

    #[test]
    fn support_reports_orthogonal_controls_as_unresolved_2951() {
        let (system, domain) = integer_fixture();
        // ⟨(1, −3), (3, 1)⟩ = 0 for controls 0, 4 and 5; control 3 lives only in the block the direction
        // leaves at 0; control 6 is zero.
        let direction = covector([1.0, -3.0, 0.0]);
        let support = system.support(&domain, &[false; 7], &direction).expect("valid inputs");
        assert_eq!(support.unresolved, vec![0, 3, 4, 5, 6]);
        let masks = endpoint_masks(&domain, &[false; 7]);
        let exhaustive = masks
            .iter()
            .map(|mask| pair(&system.moment(&domain, mask).expect("admissible"), &direction))
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(support.value, exhaustive);
        // Negative control: a direction with no orthogonal generator resolves every nonzero control.
        let generic = system
            .support(&domain, &[false; 7], &covector([2.0, -1.0, 3.0]))
            .expect("valid inputs");
        assert_eq!(generic.unresolved, vec![6]);
    }

    #[test]
    fn support_and_witness_are_invariant_under_a_moment_basis_change_2951() {
        let (system, domain) = planar_fixture();
        let kept = [false; 8];
        let apply = |matrix: [[f64; 2]; 2], vector: [f64; 2]| {
            [
                matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
                matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
            ]
        };
        let transpose = |matrix: [[f64; 2]; 2]| [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]];
        // q → G·q with a unimodular G, so G⁻¹ is integer and every transformed quantity stays exact.
        for (forward, inverse) in [
            ([[2.0, 1.0], [1.0, 1.0]], [[1.0, -1.0], [-1.0, 2.0]]),
            ([[0.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]),
        ] {
            let transformed = MaskMomentSystem::new(
                vec![block(2)],
                (0..8)
                    .map(|control| {
                        let vector = &system.generator(control).expect("control in range")[0].vector;
                        vec![part(0, &apply(forward, [vector[0], vector[1]]))]
                    })
                    .collect(),
            )
            .expect("well-formed generators");
            // u → G⁻ᵀ·u keeps every pairing ⟨u, v_c⟩.
            for values in [[1.0, 0.0], [2.0, -3.0], [-1.0, 4.0], [5.0, 1.0]] {
                let original = system.support(&domain, &kept, &planar_covector(values)).expect("valid");
                let changed = transformed
                    .support(&domain, &kept, &planar_covector(apply(transpose(inverse), values)))
                    .expect("valid");
                assert_eq!(changed.value, original.value);
                assert_eq!(changed.witness, original.witness);
                // Positive control: for a non-orthogonal G, transforming the direction like a moment
                // breaks the invariance. At u = (2, −3), GᵀG·u = (1, 0) gives support 8 against 14.
                if forward != transpose(inverse) && values == [2.0, -3.0] {
                    let wrong = transformed
                        .support(&domain, &kept, &planar_covector(apply(forward, values)))
                        .expect("valid");
                    assert_eq!((wrong.value, original.value), (8.0, 14.0));
                }
            }
        }
    }
}
