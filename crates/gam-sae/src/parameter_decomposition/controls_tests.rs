#![cfg(test)]
//! Cross-module adversarial and null controls for the manifold parameter
//! decomposition (#2951), against the landed modules only.
//!
//! Each test composes at least two modules and carries a positive control that the
//! guard under test must catch:
//!
//! * a positive collinear refinement keeps every whole-group support's risk, its
//!   witness and the minimum support, while a bent refinement with the same sum
//!   does not (`moments`, `supports`; P10, A5);
//! * splitting a cancelling pair shrinks the mask-law mean square but not the
//!   supremum, so a support certified by the average is refuted by the support
//!   function's witness (`moments`, `supports`; P10, A6);
//! * a `Q, −Q` junk pair reproduces the all-on block exactly yet enters every
//!   minimum support, and tying its two masks makes it inert (`rewrite`,
//!   `supports`; A13 negative control);
//! * the identity program is shorter than the projector family when the family's
//!   labels are shipped at declared precision and fidelity is measured on the
//!   decoded artifact (`codec`, `precision`; P11, P18, A8);
//! * a redundant pair is one OR edge, and an interior mask keeps a component that
//!   endpoint masks miss (`rewrite`, `supports`; P7, P12, A7);
//! * gauge-equivalent factorizations execute one intervention under covariant
//!   masks and move no generator, while a diagonal mask on a sheared basis is a
//!   different intervention (`rewrite`, `moments`; P1, A2);
//! * a softmax-gauge component, a constant logit shift, is dropped by the
//!   oscillation certificate. A sup-norm gap certificate keeps it and leaves the
//!   search unresolved (`moments`, `bounds`, gam-math `categorical`, `supports`;
//!   P8, P15, A6);
//! * a collinear family resolves one mode whether its members are positive
//!   multiples or a cancelling pair, but only the positive refinement merges into
//!   one ablation control (`families`, `moments`; P10).

use std::convert::Infallible;
use std::f64::consts::PI;

use gam_linalg::roundoff::{accumulation_band, accumulation_growth};
use gam_math::categorical::categorical_kl_from_logits_with_error;
use gam_math::gaussian_activation::GaussianActivation;
use ndarray::{Array1, Array2, ArrayView1, array};

use super::bounds::{LogitGapNorm, kl_bound_from_logit_gap, softmax_kl_oscillation_bound};
use super::codec::{
    BitString, DagNode, LibraryPacketArtifact, code_saving_at_proven_fidelity, decode_fixed_index, decode_ordered_dag,
    decode_prefix_integer, decode_support_packet, encode_fixed_index, encode_ordered_dag, encode_prefix_integer,
    encode_support_packets, prefix_integer_len_bits, subset_code_len_bits, union_support_library,
};
use super::families::{FamilyError, principal_field};
use super::moments::{GeneratorPart, MaskDomain, MaskMomentSystem, MomentBlock, MomentVector, WitnessEndpoint};
use super::precision::{DecodableArtifact, PeriodicQuotient, QuotientCode, decode_then_evaluate};
use super::rewrite::{ComponentMask, ComponentMlp, ComponentRead, MlpMask, NativeMlp};
use super::supports::{
    CardinalityCode, ComponentSet, EvidenceStatus, ExactBasis, Extremum, FailureHypergraph, SeparationOracle,
    minimum_code_support, replay_conflicts,
};

/// A support code whose length is the support size, so a minimum code is a minimum
/// cardinality.
struct SizeCode;

impl CardinalityCode for SizeCode {
    type Error = Infallible;

    fn support_bits(&self, components: usize, size: usize) -> Result<u64, Infallible> {
        Ok(size.min(components) as u64)
    }
}

fn set(components: usize, members: &[usize]) -> ComponentSet {
    ComponentSet::new(components, members.to_vec()).expect("members in range")
}

/// `A(m) = {c : m_c != 1}`.
fn perturbed(mask: &[f64]) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .filter_map(|(component, &level)| (level != 1.0).then_some(component))
        .collect()
}

fn part(block: usize, vector: &[f64]) -> GeneratorPart {
    GeneratorPart {
        block,
        vector: Array1::from(vector.to_vec()),
    }
}

fn planar_system(generators: &[[f64; 2]]) -> MaskMomentSystem {
    MaskMomentSystem::new(
        vec![MomentBlock { dimension: 2 }],
        generators.iter().map(|vector| vec![part(0, vector)]).collect(),
    )
    .expect("well-formed planar generators")
}

fn unit_domain(controls: usize) -> MaskDomain {
    MaskDomain::new(vec![(0.0, 1.0); controls]).expect("the unit interval contains all-on")
}

/// The risk of a support: a supremum, its roundoff band and a mask attaining it.
struct Risk {
    value: f64,
    band: f64,
    mask: Vec<f64>,
}

/// Separation through the support function (P8). A mask's discrepancy is
/// `max_k |⟨u_k, q(m)⟩|` over declared readouts, so a support's risk is the largest
/// absolute support over the readouts: an algebraic supremum over the whole declared
/// box, attained by the support function's witness mask.
#[derive(Clone)]
struct MomentOracle {
    system: MaskMomentSystem,
    domain: MaskDomain,
    readouts: Vec<MomentVector>,
}

impl MomentOracle {
    fn kept(&self, support: &ComponentSet) -> Vec<bool> {
        (0..self.system.control_count())
            .map(|control| support.members().binary_search(&control).is_ok())
            .collect()
    }

    /// Each readout's computed supremum lies within its band of the exact one, so the
    /// largest computed value lies within the largest band of the exact maximum.
    fn risk(&self, support: &ComponentSet) -> Result<Risk, String> {
        let kept = self.kept(support);
        let mut value = f64::NEG_INFINITY;
        let mut band = 0.0_f64;
        let mut witness = Vec::new();
        for readout in &self.readouts {
            let side = self
                .system
                .absolute_supremum(&self.domain, &kept, readout)
                .map_err(|error| format!("{error:?}"))?;
            band = band.max(side.band);
            if side.value > value {
                value = side.value;
                witness = side.witness;
            }
        }
        let mask = self.domain.mask_at(&witness).map_err(|error| format!("{error:?}"))?;
        Ok(Risk { value, band, mask })
    }

    /// A bound on the rounding of `⟨u, q(m)⟩`. Each moment coordinate sums `C` products
    /// of a rounded deletion `1 − m_c` and a generator entry, and the pairing sums `n`
    /// more products over the `n` moment coordinates, so every term carries at most
    /// `C + n + 1` roundings and the computed value is within `γ_{C+n+1}` of the exact
    /// one relative to `Σ_c |1 − m_c| Σ_i |v_{c,i}| |u_i|` (Higham, ASNA 2nd ed., §3.4).
    fn pairing_band(&self, mask: &[f64], readout: &MomentVector) -> f64 {
        let controls = self.system.control_count();
        let dimension: usize = self.system.blocks().iter().map(|block| block.dimension).sum();
        let absolute: f64 = mask
            .iter()
            .enumerate()
            .map(|(control, &level)| {
                let parts = self.system.generator(control).expect("control in range");
                (1.0 - level).abs()
                    * parts
                        .iter()
                        .map(|part| {
                            part.vector
                                .iter()
                                .zip(readout.blocks[part.block].iter())
                                .map(|(generator, covector)| (generator * covector).abs())
                                .sum::<f64>()
                        })
                        .sum::<f64>()
            })
            .sum();
        accumulation_growth(controls + dimension + 1) * absolute
    }
}

impl SeparationOracle for MomentOracle {
    type Mask = Vec<f64>;
    type Domain = &'static str;
    type Error = String;

    fn components(&self) -> usize {
        self.system.control_count()
    }

    fn perturbed_components(&self, mask: &Vec<f64>) -> Vec<usize> {
        perturbed(mask)
    }

    fn separate(&mut self, support: &ComponentSet) -> Result<EvidenceStatus<Vec<f64>, &'static str>, String> {
        let risk = self.risk(support)?;
        EvidenceStatus::exact(
            risk.value,
            risk.band,
            ExactBasis::Algebraic,
            Some(risk.mask),
            "support function over the declared mask box",
        )
        .map_err(|error| error.to_string())
    }

    fn evaluate(&mut self, mask: &Vec<f64>) -> Result<EvidenceStatus<Vec<f64>, &'static str>, String> {
        let moment = self
            .system
            .moment(&self.domain, mask)
            .map_err(|error| format!("{error:?}"))?;
        let mut value = 0.0_f64;
        let mut band = 0.0_f64;
        for readout in &self.readouts {
            let pairing: f64 = moment
                .blocks
                .iter()
                .zip(&readout.blocks)
                .map(|(coordinates, covector)| coordinates.dot(covector))
                .sum();
            value = value.max(pairing.abs());
            band = band.max(self.pairing_band(mask, readout));
        }
        EvidenceStatus::exact(
            value,
            band,
            ExactBasis::Exhaustive { cardinality: 1 },
            Some(mask.clone()),
            "one declared mask",
        )
        .map_err(|error| error.to_string())
    }
}

/// The dilution defect as an oracle: it reports the root mean square of the first
/// readout under independent uniform masks, with its roundoff, as if it bounded the
/// risk.
struct AverageAsBoundOracle {
    honest: MomentOracle,
}

impl SeparationOracle for AverageAsBoundOracle {
    type Mask = Vec<f64>;
    type Domain = &'static str;
    type Error = String;

    fn components(&self) -> usize {
        self.honest.components()
    }

    fn perturbed_components(&self, mask: &Vec<f64>) -> Vec<usize> {
        perturbed(mask)
    }

    fn separate(&mut self, support: &ComponentSet) -> Result<EvidenceStatus<Vec<f64>, &'static str>, String> {
        let kept = self.honest.kept(support);
        let law = self
            .honest
            .system
            .uniform_mask_law_moments(&self.honest.domain, &kept, &self.honest.readouts[0])
            .map_err(|error| format!("{error:?}"))?;
        EvidenceStatus::uniform_bound(
            (law.mean_square + law.mean_square_band).sqrt(),
            0.0,
            "root mean square under iid uniform masks",
        )
        .map_err(|error| error.to_string())
    }

    fn evaluate(&mut self, mask: &Vec<f64>) -> Result<EvidenceStatus<Vec<f64>, &'static str>, String> {
        self.honest.evaluate(mask)
    }
}

/// Exhaustive separation of a component MLP over a declared finite grid of read-in
/// mask levels at fixed inputs, with the write-out factor all on. A mask's discrepancy
/// is the largest absolute output difference from the native block.
///
/// Every fixture below uses tensors, biases, inputs and levels that are dyadic
/// rationals with a few significant bits, under ReLU, and every product, sum and
/// `max` of such values is exact in `f64`. So the discrepancy carries no numerical
/// error, and the grid is a test instrument, never a production search.
struct BlockGridOracle {
    block: ComponentMlp,
    inputs: Array2<f64>,
    native: Array2<f64>,
    levels: Vec<f64>,
}

impl BlockGridOracle {
    fn new(block: ComponentMlp, inputs: Array2<f64>, levels: Vec<f64>) -> Self {
        let native = block.native().execute(inputs.view()).expect("the native block executes");
        Self {
            block,
            inputs,
            native,
            levels,
        }
    }

    fn outputs(&self, mask: &[f64]) -> Result<Array2<f64>, String> {
        let mask = Array1::from(mask.to_vec());
        self.block
            .execute(
                self.inputs.view(),
                MlpMask {
                    read_in: ComponentMask::Components(mask.view()),
                    write_out: ComponentMask::AllOn,
                },
            )
            .map_err(|error| error.to_string())
    }

    fn distance(&self, mask: &[f64]) -> Result<f64, String> {
        Ok((&self.outputs(mask)? - &self.native)
            .iter()
            .fold(0.0_f64, |largest, difference| largest.max(difference.abs())))
    }
}

impl SeparationOracle for BlockGridOracle {
    type Mask = Vec<f64>;
    type Domain = &'static str;
    type Error = String;

    fn components(&self) -> usize {
        self.block.read_in().components()
    }

    fn perturbed_components(&self, mask: &Vec<f64>) -> Vec<usize> {
        perturbed(mask)
    }

    fn separate(&mut self, support: &ComponentSet) -> Result<EvidenceStatus<Vec<f64>, &'static str>, String> {
        let components = self.components();
        let free: Vec<usize> = (0..components)
            .filter(|component| support.members().binary_search(component).is_err())
            .collect();
        let mut mask = vec![1.0; components];
        let mut best = (f64::NEG_INFINITY, mask.clone());
        let mut digits = vec![0usize; free.len()];
        let mut cardinality = 0u64;
        'masks: loop {
            for (slot, &component) in free.iter().enumerate() {
                mask[component] = self.levels[digits[slot]];
            }
            let distance = self.distance(&mask)?;
            cardinality += 1;
            if distance > best.0 {
                best = (distance, mask.clone());
            }
            for digit in digits.iter_mut() {
                *digit += 1;
                if *digit < self.levels.len() {
                    continue 'masks;
                }
                *digit = 0;
            }
            break;
        }
        EvidenceStatus::exact(
            best.0,
            0.0,
            ExactBasis::Exhaustive { cardinality },
            Some(best.1),
            "declared read-in mask grid",
        )
        .map_err(|error| error.to_string())
    }

    fn evaluate(&mut self, mask: &Vec<f64>) -> Result<EvidenceStatus<Vec<f64>, &'static str>, String> {
        EvidenceStatus::exact(
            self.distance(mask)?,
            0.0,
            ExactBasis::Exhaustive { cardinality: 1 },
            Some(mask.clone()),
            "one declared mask",
        )
        .map_err(|error| error.to_string())
    }
}

#[test]
fn a_positive_collinear_refinement_keeps_every_whole_group_risk_and_the_minimum_support_2951() {
    // Dyadic generators, so every pairing, support value and sum below is exact.
    let coarse_generators = [[4.0, 2.0], [-1.0, 3.0], [2.0, -6.0]];
    // Control 0 split into its dyadic pieces 1/2, 1/4 and 1/4, the last two appended.
    let refined_generators = [[2.0, 1.0], [-1.0, 3.0], [2.0, -6.0], [1.0, 0.5], [1.0, 0.5]];
    // Two pieces that also sum to (4, 2) but are not collinear.
    let bent_generators = [[2.0, 2.0], [-1.0, 3.0], [2.0, -6.0], [2.0, 0.0]];
    let tolerance = 7.5;
    let oracle = |generators: &[[f64; 2]]| MomentOracle {
        system: planar_system(generators),
        domain: unit_domain(generators.len()),
        readouts: [[1.0, 0.0], [0.0, 1.0], [-1.0, 2.0]]
            .iter()
            .map(|readout| MomentVector {
                blocks: vec![Array1::from(readout.to_vec())],
            })
            .collect(),
    };
    let mut coarse = oracle(&coarse_generators);
    let mut refined = oracle(&refined_generators);
    let mut bent = oracle(&bent_generators);

    // The merge folds the refinement back into the coarse system bit for bit and folds
    // nothing of the bent one.
    let merge = refined
        .system
        .merge_positive_collinear(&refined.domain)
        .expect("valid domain");
    assert_eq!(merge.groups, vec![vec![0, 3, 4], vec![1], vec![2]]);
    assert_eq!(merge.system, coarse.system);
    assert_eq!(merge.domain, coarse.domain);
    let bent_merge = bent
        .system
        .merge_positive_collinear(&bent.domain)
        .expect("valid domain");
    assert_eq!(bent_merge.groups.len(), 4);

    // Every whole-group support has the coarse risk and the coarse witness, each piece
    // copying its parent's mask value, exhaustively over the 2^3 coarse supports.
    let represent = |mask: &Vec<f64>| Some(vec![mask[0], mask[1], mask[2], mask[0], mask[0]]);
    for bits in 0..8usize {
        let members: Vec<usize> = (0..3).filter(|control| bits >> control & 1 == 1).collect();
        let preimage: Vec<usize> = members
            .iter()
            .flat_map(|&control| merge.groups[control].iter().copied())
            .collect();
        let coarse_risk = coarse.risk(&set(3, &members)).expect("valid support");
        let refined_risk = refined.risk(&set(5, &preimage)).expect("valid support");
        assert_eq!(refined_risk.value, coarse_risk.value, "support {members:?}");
        assert_eq!(Some(refined_risk.mask), represent(&coarse_risk.mask), "support {members:?}");
    }

    // Positive control: the bent pieces move the risk of keeping control 2 alone from 7
    // to 9, across the tolerance.
    assert_eq!(coarse.risk(&set(3, &[2])).expect("valid support").value, 7.0);
    assert_eq!(bent.risk(&set(4, &[2])).expect("valid support").value, 9.0);
    assert!(coarse.separate(&set(3, &[2])).expect("exact separation").certifies_at_most(tolerance));
    assert!(bent.separate(&set(4, &[2])).expect("exact separation").refutes_at_most(tolerance));

    let coarse_search = minimum_code_support(&mut coarse, &SizeCode, tolerance, FailureHypergraph::new(3))
        .expect("the coarse search closes");
    assert_eq!(
        coarse_search.certified.as_ref().expect("a certified support").support.members(),
        &[2]
    );
    assert!(matches!(
        coarse_search.code,
        EvidenceStatus::Exact {
            basis: ExactBasis::ClosedSearch,
            ..
        }
    ));
    assert_eq!(coarse_search.code.upper_bound(), Some(1.0));

    let cold = minimum_code_support(&mut refined, &SizeCode, tolerance, FailureHypergraph::new(5))
        .expect("the refined search closes");
    assert_eq!(cold.certified.as_ref().expect("a certified support").support.members(), &[2]);
    assert_eq!(cold.code.upper_bound(), coarse_search.code.upper_bound());

    // Every coarse conflict replays into the refinement and is still bad there, and the
    // replayed search reaches the same code with fewer separations.
    let replay = replay_conflicts(&coarse_search.hypergraph, represent, &mut refined, tolerance)
        .expect("the replay evaluates");
    assert_eq!(
        (replay.unrepresentable, replay.no_longer_bad, replay.without_witness),
        (0, 0, 0)
    );
    assert_eq!(replay.hypergraph.edges().len(), coarse_search.hypergraph.edges().len());
    let warm = minimum_code_support(&mut refined, &SizeCode, tolerance, replay.hypergraph)
        .expect("the replayed search closes");
    assert_eq!(warm.code.upper_bound(), cold.code.upper_bound());
    assert!(warm.separations < cold.separations);

    // Positive control: the bent refinement needs two components where the coarse and
    // the collinear refinement need one.
    let bent_search = minimum_code_support(&mut bent, &SizeCode, tolerance, FailureHypergraph::new(4))
        .expect("the bent search closes");
    assert_eq!(bent_search.code.upper_bound(), Some(2.0));
    assert!(bent_search.certified.expect("a certified support").support.members().contains(&2));
}

#[test]
fn a_support_certified_by_a_mask_law_average_is_refuted_by_the_support_witness_2951() {
    let tolerance = 0.625;
    let readout = MomentVector {
        blocks: vec![array![1.0]],
    };
    let mut previous_rms = f64::INFINITY;
    for pieces in [1usize, 2, 4] {
        // A cancelling pair ±1, each side split into `pieces` exact dyadic pieces.
        let piece = 1.0 / pieces as f64;
        let controls = 2 * pieces;
        let generators = (0..controls)
            .map(|index| vec![part(0, &[if index < pieces { piece } else { -piece }])])
            .collect();
        let mut honest = MomentOracle {
            system: MaskMomentSystem::new(vec![MomentBlock { dimension: 1 }], generators)
                .expect("well-formed generators"),
            domain: unit_domain(controls),
            readouts: vec![readout.clone()],
        };
        let free = vec![false; controls];
        let law = honest
            .system
            .uniform_mask_law_moments(&honest.domain, &free, &readout)
            .expect("valid law");
        let rms = law.mean_square.sqrt();
        assert!(rms < previous_rms, "the average must shrink as the pair is split");
        previous_rms = rms;

        // Positive control: an oracle presenting that average as a bound certifies
        // deleting everything.
        let mut diluted = AverageAsBoundOracle { honest: honest.clone() };
        let average_support = minimum_code_support(&mut diluted, &SizeCode, tolerance, FailureHypergraph::new(controls))
            .expect("the averaged search closes")
            .certified
            .expect("the average certifies a support")
            .support;
        assert!(average_support.is_empty(), "the average certifies the empty support");

        // The guard: the supremum stays |a| = 1 for every split, its witness refutes the
        // averaged support, and the average claimed as a bound is refuted.
        let empty = honest.risk(&average_support).expect("valid support");
        assert_eq!(empty.value, 1.0);
        let counterexample =
            EvidenceStatus::<Vec<f64>, &'static str>::counterexample(empty.value, empty.band, tolerance, empty.mask)
                .expect("a violation beyond roundoff");
        assert!(counterexample.refutes_at_most(tolerance));
        let claim = honest
            .system
            .check_claimed_absolute_bound(&honest.domain, &free, &readout, rms)
            .expect("finite claim");
        assert!(
            matches!(claim, EvidenceStatus::Counterexample { value, .. } if value == 1.0),
            "{claim:?}"
        );
        let witness = claim.into_witness().expect("a counterexample carries its witness");
        let mask = honest.domain.mask_at(&witness).expect("one endpoint per control");
        assert!(honest.evaluate(&mask).expect("exact evaluation").refutes_at_most(tolerance));

        // Keeping k₊ positive and k₋ negative pieces leaves the risk
        // max(n − k₊, n − k₋)/n, so the minimum support keeps ⌈n(1 − ε)⌉ of each sign.
        let honest_search = minimum_code_support(&mut honest, &SizeCode, tolerance, FailureHypergraph::new(controls))
            .expect("the honest search closes");
        let per_sign = (pieces as f64 * (1.0 - tolerance)).ceil() as usize;
        assert_eq!(honest_search.code.upper_bound(), Some((2 * per_sign) as f64));
        let certified = honest_search.certified.expect("a certified support").support;
        let positives = certified.members().iter().filter(|&&control| control < pieces).count();
        assert_eq!(2 * positives, certified.len(), "pieces {pieces}: {:?}", certified.members());
    }
}

#[test]
fn a_q_minus_q_junk_pair_reproduces_the_all_on_block_and_enters_every_minimum_support_2951() {
    // d = H = 2 under ReLU with small integer tensors, so every summed input, activation
    // and output below is exact.
    let read_in = array![[1.0, 2.0], [-1.0, 1.0]];
    let write_out = array![[1.0, 0.0], [1.0, -1.0]];
    let native = NativeMlp::new(
        read_in.clone(),
        array![0.0, 1.0],
        write_out.clone(),
        array![0.0, 0.0],
        GaussianActivation::Relu,
    )
    .expect("shapes compose");
    let inputs = array![[1.0, 0.0], [0.0, 1.0], [2.0, -1.0], [-1.0, 3.0]];
    let identity = Array2::<f64>::eye(2);
    let decompose = |read: &Array2<f64>, candidate: &Array2<f64>| {
        ComponentMlp::new(
            native.clone(),
            ComponentRead {
                read: read.view(),
                candidate_write: candidate.view(),
            },
            ComponentRead {
                read: identity.view(),
                candidate_write: write_out.view(),
            },
        )
        .expect("the read covers the input and the candidate is an exact write")
    };
    // Clean: one component per input direction.
    let clean = decompose(&identity, &read_in);
    // Junk: the same two components, plus a pair that reads (1, 1) and writes a = (2, −1)
    // and −a. N R = W₁ exactly, so the solved write is the candidate itself.
    let junk_read = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
    let junk_write = array![[1.0, 2.0, 2.0, -2.0], [-1.0, 1.0, -1.0, 1.0]];
    let junk = decompose(&junk_read, &junk_write);
    assert_eq!(junk.read_in().write(), junk_write);
    assert_eq!(junk.read_in().write().dot(&junk.read_in().read()), read_in);

    // All on, routed natively or through the all-ones factored mask, both decompositions
    // reproduce the native block exactly.
    let native_outputs = native.execute(inputs.view()).expect("native block");
    for (name, decomposition) in [("clean", &clean), ("junk", &junk)] {
        let all_on = MlpMask {
            read_in: ComponentMask::AllOn,
            write_out: ComponentMask::AllOn,
        };
        assert_eq!(
            decomposition.execute(inputs.view(), all_on).expect("all-on block"),
            native_outputs,
            "{name}"
        );
        let ones = Array1::<f64>::ones(decomposition.read_in().components());
        let factored = MlpMask {
            read_in: ComponentMask::Components(ones.view()),
            write_out: ComponentMask::AllOn,
        };
        assert_eq!(
            decomposition.execute(inputs.view(), factored).expect("all-ones block"),
            native_outputs,
            "{name}: the all-ones factored mask"
        );
    }

    let tolerance = 0.5;
    let mut clean_oracle = BlockGridOracle::new(clean, inputs.clone(), vec![0.0, 1.0]);
    let mut junk_oracle = BlockGridOracle::new(junk, inputs, vec![0.0, 1.0]);

    // Positive control: invisible at all-on, the pair still refutes keeping only the
    // regular components, through a mask that splits the pair and perturbs nothing else.
    let regular_only = junk_oracle.separate(&set(4, &[0, 1])).expect("exhaustive separation");
    assert!(regular_only.refutes_at_most(tolerance));
    let witness = regular_only.witness().expect("a witness mask");
    let split = junk_oracle.perturbed_components(witness);
    assert!(!split.is_empty() && split.iter().all(|&component| component >= 2), "{split:?}");
    assert_ne!(witness[2], witness[3], "the witness splits the pair");

    // So every minimum support pays for the pair: each junk component alone is an edge.
    let clean_search = minimum_code_support(&mut clean_oracle, &SizeCode, tolerance, FailureHypergraph::new(2))
        .expect("the clean search closes");
    let junk_search = minimum_code_support(&mut junk_oracle, &SizeCode, tolerance, FailureHypergraph::new(4))
        .expect("the junk search closes");
    assert_eq!(
        clean_search.certified.as_ref().expect("a certified support").support.members(),
        &[0, 1]
    );
    assert_eq!(
        junk_search.certified.as_ref().expect("a certified support").support.members(),
        &[0, 1, 2, 3]
    );
    for component in [2usize, 3] {
        assert!(
            junk_search
                .hypergraph
                .edges()
                .iter()
                .any(|edge| edge.perturbed.members() == [component]),
            "junk component {component} alone is a failure edge"
        );
    }

    // Null control: tying the pair's masks, m₂ = m₃, makes it inert. Every tied mask
    // executes the clean decomposition's block exactly.
    for bits in 0..8usize {
        let regular = [(bits & 1) as f64, (bits >> 1 & 1) as f64];
        let tied = (bits >> 2 & 1) as f64;
        assert_eq!(
            junk_oracle
                .outputs(&[regular[0], regular[1], tied, tied])
                .expect("junk block"),
            clean_oracle.outputs(&regular).expect("clean block"),
            "regular {regular:?}, tied pair {tied}"
        );
    }
}

/// The A8 program alphabet: an input read, the native identity, one instance of the
/// projector family, and a sum masked by the input's packet.
const INPUT: usize = 0;
const IDENTITY: usize = 1;
const FAMILY_INSTANCE: usize = 2;
const MASKED_SUM: usize = 3;
const LABELS: usize = 4;

/// `P_S x = Σ_{c∈S} (2/C) ⟨v(t_c), x⟩ v(t_c)` with `v(t) = (cos t, sin t)`, for the
/// labels a program decoded.
fn projector_output(labels: &[f64], support: &[usize], x: [f64; 2]) -> [f64; 2] {
    let weight = 2.0 / labels.len() as f64;
    let mut output = [0.0_f64; 2];
    for &instance in support {
        let (sine, cosine) = labels[instance].sin_cos();
        let overlap = weight * (cosine * x[0] + sine * x[1]);
        output[0] += overlap * cosine;
        output[1] += overlap * sine;
    }
    output
}

fn residual_norm(x: [f64; 2], output: [f64; 2]) -> f64 {
    ((x[0] - output[0]).powi(2) + (x[1] - output[1]).powi(2)).sqrt()
}

/// A bound on the rounding of `‖x − P_S x‖` for a unit `x` and any support of a
/// `C`-instance family. Every partial sum and the output stay within norm 2
/// (`‖P_S x‖ ≤ 2|S|/C ≤ 2`), each output coordinate is at most `2C + 12` operations
/// from exact inputs (direction, dot product, scale, accumulate, subtract, norm), and
/// charging each operation `2u` also covers a library `sin`/`cos` within one ulp. The
/// distortion, a norm of magnitude at most 3, then moves by at most
/// `3·γ_{2(2C + 12)}`.
fn projector_distortion_roundoff(instances: usize) -> f64 {
    3.0 * accumulation_growth(2 * (2 * instances + 12))
}

/// Executes a decoded projector-family program: each input's packet names its
/// instances, and the output is `P_S x`.
fn execute_family(labels: &[f64], packets: &[BitString], inputs: &[[f64; 2]]) -> Result<Vec<[f64; 2]>, String> {
    packets
        .iter()
        .zip(inputs)
        .map(|(packet, &x)| {
            let support = decode_support_packet(labels.len(), packet).map_err(|error| error.to_string())?;
            Ok(projector_output(labels, &support, x))
        })
        .collect()
}

fn largest_distortion(outputs: &[[f64; 2]], native: &[[f64; 2]]) -> f64 {
    outputs
        .iter()
        .zip(native)
        .map(|(&output, &x)| residual_norm(x, output))
        .fold(0.0_f64, f64::max)
}

/// The projector family's library: its program graph, the label resolution `b + 1` in
/// the prefix integer code, then each instance's label as a `b`-bit quotient cell.
fn family_library(nodes: &[DagNode], labels: &QuotientCode) -> BitString {
    let mut library = BitString::new();
    encode_ordered_dag(&mut library, LABELS, nodes).expect("family graph");
    encode_prefix_integer(&mut library, u64::from(labels.resolution_bits()) + 1).expect("label resolution");
    for &index in labels.indices() {
        encode_fixed_index(&mut library, index as usize, 1usize << labels.resolution_bits()).expect("label cell");
    }
    library
}

/// Reads a family library back into its label code, from its bits and the family's
/// declared quotient alone.
fn decode_family_labels(library: &BitString, quotient: PeriodicQuotient) -> QuotientCode {
    let mut reader = library.reader();
    let nodes = decode_ordered_dag(&mut reader, LABELS).expect("family graph");
    let instances = nodes.iter().filter(|node| node.label == FAMILY_INSTANCE).count();
    let resolution_bits =
        u32::try_from(decode_prefix_integer(&mut reader).expect("label resolution") - 1).expect("a u32 resolution");
    let mut indices = Vec::with_capacity(instances);
    for _ in 0..instances {
        indices.push(decode_fixed_index(&mut reader, 1usize << resolution_bits).expect("label cell") as u64);
    }
    assert_eq!(reader.finish(), Ok(()));
    QuotientCode::from_indices(quotient, resolution_bits, indices).expect("decoded label cells")
}

/// A packetless program as transmitted: its graph is the whole library, and it decodes to the
/// graph's nodes.
struct PacketlessProgram<'a>(&'a LibraryPacketArtifact);

impl DecodableArtifact for PacketlessProgram<'_> {
    type Decoded = Vec<DagNode>;

    fn decode(&self) -> Result<Vec<DagNode>, String> {
        let mut reader = self.0.library.reader();
        let nodes = decode_ordered_dag(&mut reader, LABELS).map_err(|error| error.to_string())?;
        reader.finish().map_err(|error| error.to_string())?;
        if self.0.packets.iter().any(|packet| !packet.is_empty()) {
            return Err("a packetless program reads no packet".to_string());
        }
        Ok(nodes)
    }
}

#[test]
fn identity_is_shorter_than_a_projector_family_whose_labels_are_shipped_and_decoded_2951() {
    let instances = 16usize;
    let tolerance = 0.25;
    let distortion_roundoff = projector_distortion_roundoff(instances);
    let quotient = PeriodicQuotient::new(PI).expect("the RP1 period is admissible");
    let labels: Vec<f64> = (0..instances)
        .map(|instance| PI * instance as f64 / instances as f64)
        .collect();
    let inputs: Vec<[f64; 2]> = (0..24)
        .map(|index| {
            let angle = 2.0 * PI * (index as f64 + 1.0 / 3.0) / 24.0;
            [angle.cos(), angle.sin()]
        })
        .collect();

    // Identity: read the input and apply the native identity. No labels, no packets.
    let identity_nodes = vec![
        DagNode { label: INPUT, arguments: vec![] },
        DagNode { label: IDENTITY, arguments: vec![0] },
    ];
    let mut identity_library = BitString::new();
    encode_ordered_dag(&mut identity_library, LABELS, &identity_nodes).expect("identity graph");
    let identity = LibraryPacketArtifact {
        library: identity_library,
        packets: vec![BitString::new(); inputs.len()],
    };
    let mut reader = identity.library.reader();
    assert_eq!(decode_ordered_dag(&mut reader, LABELS), Ok(identity_nodes.clone()));
    assert_eq!(reader.finish(), Ok(()));

    // Projector family: the graph, and the labels at 4 bits on RP1, whose 16 cells are the
    // family's own grid, so the decoded labels are the labels.
    let mut family_nodes = vec![DagNode { label: INPUT, arguments: vec![] }];
    family_nodes.extend(std::iter::repeat_n(
        DagNode { label: FAMILY_INSTANCE, arguments: vec![0] },
        instances,
    ));
    family_nodes.push(DagNode {
        label: MASKED_SUM,
        arguments: (1..=instances).collect(),
    });
    let fine = QuotientCode::encode(&labels, quotient, 4).expect("labels encode");
    assert_eq!(fine.indices(), (0..instances as u64).collect::<Vec<u64>>().as_slice());
    let fine_library = family_library(&family_nodes, &fine);
    let decoded_labels = decode_family_labels(&fine_library, quotient)
        .decode()
        .expect("labels decode");
    assert_eq!(decoded_labels, labels);
    // The labels are paid for: the resolution codeword and b bits per instance.
    let mut graph_only = BitString::new();
    encode_ordered_dag(&mut graph_only, LABELS, &family_nodes).expect("family graph");
    assert_eq!(
        fine_library.len_bits(),
        graph_only.len_bits() + prefix_integer_len_bits(5).expect("length") + 4 * instances as u64
    );

    // Per input, the fewest largest-overlap decoded instances whose distortion, with its
    // roundoff, meets the tolerance.
    let supports: Vec<Vec<usize>> = inputs
        .iter()
        .map(|&x| {
            let overlap = |instance: usize| {
                let (sine, cosine) = decoded_labels[instance].sin_cos();
                (cosine * x[0] + sine * x[1]).abs()
            };
            let mut order: Vec<usize> = (0..instances).collect();
            order.sort_by(|&left, &right| overlap(right).total_cmp(&overlap(left)));
            (1..=instances)
                .map(|count| {
                    let mut support = order[..count].to_vec();
                    support.sort_unstable();
                    support
                })
                .find(|support| {
                    residual_norm(x, projector_output(&decoded_labels, support, x)) + distortion_roundoff <= tolerance
                })
                .expect("the all-on sum is the identity, so some support meets the tolerance")
        })
        .collect();
    let library = union_support_library(&supports).expect("union library");
    assert_eq!(library.components, (0..instances).collect::<Vec<usize>>());
    // P11: error ε on a unit input needs |S| ≥ C(1 − ε)/2.
    let minimum_cardinality = (instances as f64 * (1.0 - tolerance) / 2.0).ceil() as usize;
    assert!(library.supports.iter().all(|support| support.len() >= minimum_cardinality));
    let family = LibraryPacketArtifact {
        library: fine_library,
        packets: encode_support_packets(instances, &library.supports).expect("packets"),
    };

    // The decoded family's distortion as evidence: exhaustive over the declared inputs,
    // with the derived roundoff as its numerical error.
    let distortion_status = |outputs: &Vec<[f64; 2]>, native: &Vec<[f64; 2]>| {
        EvidenceStatus::<(), &'static str>::exact(
            largest_distortion(outputs, native),
            distortion_roundoff,
            ExactBasis::Exhaustive {
                cardinality: native.len() as u64,
            },
            None,
            "declared unit inputs",
        )
        .map_err(|error| error.to_string())
    };
    // Fidelity on the decoded identity: decode its graph, check it is the identity, execute.
    let identity_fidelity = decode_then_evaluate(
        &PacketlessProgram(&identity),
        |nodes: &Vec<DagNode>| {
            if *nodes == identity_nodes {
                Ok(inputs.clone())
            } else {
                Err("the decoded graph is not the identity program".to_string())
            }
        },
        &inputs,
        distortion_status,
        tolerance,
    )
    .expect("the decoded identity executes");
    assert_eq!(identity_fidelity.verdict(), super::precision::FidelityVerdict::Meets);

    // Fidelity on the decoded artifact: decode the labels, decode each packet, execute.
    let fidelity = decode_then_evaluate(
        &decode_family_labels(&family.library, quotient),
        |decoded: &Vec<f64>| execute_family(decoded, &family.packets, &inputs),
        &inputs,
        distortion_status,
        tolerance,
    )
    .expect("the decoded family executes");
    assert_eq!(fidelity.verdict(), super::precision::FidelityVerdict::Meets);
    let saving = code_saving_at_proven_fidelity(
        (family.total_bits(), &fidelity),
        (identity.total_bits(), &identity_fidelity),
    )
    .expect("both artifacts are proven within the declared tolerance");
    assert!(
        saving > 0,
        "identity ({} bits) must be shorter than the projector family ({} bits)",
        identity.total_bits(),
        family.total_bits()
    );
    let least_packet = (minimum_cardinality..=instances)
        .map(|cardinality| subset_code_len_bits(instances, cardinality).expect("length"))
        .min()
        .expect("admissible cardinalities");
    let family_floor = family.library.len_bits() + inputs.len() as u64 * least_packet;
    assert!(family.total_bits() >= family_floor);
    assert!(identity.total_bits() < family_floor);

    // A 0-bit label code carries no coordinate, and precision refuses it.
    assert!(QuotientCode::encode(&labels, quotient, 0).is_err());

    // Positive control: labels sent at 1 bit make a shorter artifact, and the same
    // packets on the labels before coding still meet the tolerance. The decoder puts
    // every instance at t = 0 or t = π/2, so the decoded artifact violates the tolerance
    // and the comparison refuses it.
    let collapsed = QuotientCode::encode(&labels, quotient, 1).expect("labels encode");
    let collapsed_family = LibraryPacketArtifact {
        library: family_library(&family_nodes, &collapsed),
        packets: family.packets.clone(),
    };
    assert!(collapsed_family.total_bits() < family.total_bits());
    let before_coding = largest_distortion(
        &execute_family(&labels, &collapsed_family.packets, &inputs).expect("the uncoded family executes"),
        &inputs,
    );
    assert!(before_coding + distortion_roundoff <= tolerance);
    let collapsed_labels = decode_family_labels(&collapsed_family.library, quotient);
    assert!(
        collapsed_labels
            .decode()
            .expect("labels decode")
            .iter()
            .all(|&label| label == 0.0 || label == PI / 2.0)
    );
    let collapsed_fidelity = decode_then_evaluate(
        &collapsed_labels,
        |decoded: &Vec<f64>| execute_family(decoded, &collapsed_family.packets, &inputs),
        &inputs,
        distortion_status,
        tolerance,
    )
    .expect("the decoded collapsed family executes");
    assert_eq!(
        collapsed_fidelity.verdict(),
        super::precision::FidelityVerdict::Violates,
        "the collapsed artifact's distortion {:?}",
        collapsed_fidelity.status()
    );
    assert!(
        code_saving_at_proven_fidelity(
            (collapsed_family.total_bits(), &collapsed_fidelity),
            (identity.total_bits(), &identity_fidelity),
        )
        .is_err()
    );
}

#[test]
fn a_redundant_pair_is_one_or_edge_and_an_interior_mask_keeps_a_component_endpoints_miss_2951() {
    // d = 1, H = 3 under ReLU at the one input x = 1. Components 0 and 1 each write −1
    // into unit A, whose bias is 1, so σ_A = max(0, 1 − m₀ − m₁). Component 2 writes 1
    // into units B and C, whose biases are 0 and −1/2. The write-out −σ_A − 2σ_B + 4σ_C
    // makes the discrepancy max(0, 1 − m₀ − m₁) + tent(m₂), with
    // tent(m) = 2m − 4·max(0, m − 1/2): zero at both endpoints and 1 at m = 1/2.
    let read = array![[1.0], [1.0], [1.0]];
    let write = array![[-1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]];
    let write_out = array![[-1.0, -2.0, 4.0]];
    let hidden_identity = Array2::<f64>::eye(3);
    let native = NativeMlp::new(
        write.dot(&read),
        array![1.0, 0.0, -0.5],
        write_out.clone(),
        array![0.0],
        GaussianActivation::Relu,
    )
    .expect("shapes compose");
    let block = ComponentMlp::new(
        native,
        ComponentRead {
            read: read.view(),
            candidate_write: write.view(),
        },
        ComponentRead {
            read: hidden_identity.view(),
            candidate_write: write_out.view(),
        },
    )
    .expect("the candidates are exact writes");
    let inputs = array![[1.0]];
    let tolerance = 0.75;
    let mut endpoints = BlockGridOracle::new(block.clone(), inputs.clone(), vec![0.0, 1.0]);
    let mut interior = BlockGridOracle::new(block, inputs, vec![0.0, 0.5, 1.0]);

    let endpoint_search = minimum_code_support(&mut endpoints, &SizeCode, tolerance, FailureHypergraph::new(3))
        .expect("the endpoint search closes");
    let endpoint_support = endpoint_search
        .certified
        .as_ref()
        .expect("a certified support")
        .support
        .clone();
    assert_eq!(endpoint_support.len(), 1);
    assert!(!endpoint_support.members().contains(&2), "endpoint masks never see component 2");
    assert!(
        endpoint_search
            .hypergraph
            .edges()
            .iter()
            .any(|edge| edge.perturbed.members() == [0, 1]),
        "the redundant pair is one OR edge"
    );

    let interior_search = minimum_code_support(&mut interior, &SizeCode, tolerance, FailureHypergraph::new(3))
        .expect("the interior search closes");
    let interior_support = interior_search
        .certified
        .as_ref()
        .expect("a certified support")
        .support
        .clone();
    assert_eq!(interior_support.len(), 2);
    assert!(interior_support.members().contains(&2));
    assert_eq!(
        interior_support.members().iter().filter(|&&component| component < 2).count(),
        1,
        "one member of the redundant pair suffices"
    );

    // Positive control: the endpoint-certified support is refuted by an interior mask on
    // component 2.
    let refuted = interior.separate(&endpoint_support).expect("exhaustive separation");
    assert!(refuted.refutes_at_most(tolerance));
    let witness = refuted.witness().expect("a witness mask").clone();
    assert_eq!(witness[2], 0.5);
    let attained = interior.distance(&witness).expect("the block executes");
    let counterexample = EvidenceStatus::<Vec<f64>, &'static str>::counterexample(attained, 0.0, tolerance, witness)
        .expect("a violation");
    assert!(counterexample.refutes_at_most(tolerance));

    // Positive control: scalar importances keep neither member of the pair, since each
    // single deletion is harmless, and the support they leave is refuted.
    for single in [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]] {
        assert_eq!(endpoints.distance(&single).expect("the block executes"), 0.0);
    }
    assert!(
        endpoints
            .separate(&set(3, &[2]))
            .expect("exhaustive separation")
            .refutes_at_most(tolerance)
    );
}

#[test]
fn gauge_equivalent_factorizations_execute_one_intervention_under_covariant_masks_2951() {
    // W₁ = U R with dyadic entries, inputs and masks, so every summed input and moment
    // below is exact.
    let write = array![[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]];
    let read = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let read_in = write.dot(&read);
    let write_out = array![[1.0, -1.0], [2.0, 1.0]];
    let native = NativeMlp::new(
        read_in.clone(),
        array![0.5, -0.25],
        write_out.clone(),
        array![0.0, 0.0],
        GaussianActivation::ExactGelu,
    )
    .expect("shapes compose");
    let identity = Array2::<f64>::eye(2);
    let factor = |write: &Array2<f64>, read: &Array2<f64>| {
        ComponentMlp::new(
            native.clone(),
            ComponentRead {
                read: read.view(),
                candidate_write: write.view(),
            },
            ComponentRead {
                read: identity.view(),
                candidate_write: write_out.view(),
            },
        )
        .expect("an exact factor of W₁")
    };
    let inputs = array![[1.0, 0.0], [0.0, 1.0], [2.0, -1.0], [-1.0, 3.0], [0.5, 0.25]];

    // A monomial gauge S = P D: new component c′ is old component π(c′), its write scaled
    // by s_c′ and its read by 1/s_c′.
    let permutation = [2usize, 0, 1];
    let scales = [2.0, -0.5, 4.0];
    let gauged_write = Array2::from_shape_fn((2, 3), |(row, component)| {
        scales[component] * write[[row, permutation[component]]]
    });
    let gauged_read = Array2::from_shape_fn((3, 2), |(component, column)| {
        read[[permutation[component], column]] / scales[component]
    });
    // A shear S = I + e₀e₁ᵀ.
    let shear = array![[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let shear_inverse = array![[1.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert_eq!(shear.dot(&shear_inverse), Array2::<f64>::eye(3));
    let sheared_write = write.dot(&shear);
    let sheared_read = shear_inverse.dot(&read);
    let original = factor(&write, &read);
    let gauged = factor(&gauged_write, &gauged_read);
    let sheared = factor(&sheared_write, &sheared_read);
    for decomposition in [&original, &gauged, &sheared] {
        assert_eq!(decomposition.read_in().write().dot(&decomposition.read_in().read()), read_in);
    }

    let execute = |decomposition: &ComponentMlp, mask: &[f64]| {
        let mask = Array1::from(mask.to_vec());
        decomposition
            .execute(
                inputs.view(),
                MlpMask {
                    read_in: ComponentMask::Components(mask.view()),
                    write_out: ComponentMask::AllOn,
                },
            )
            .expect("masked block")
    };
    // Continuous, binary, signed, all-off and all-ones masks.
    let masks = [
        [1.0, 0.0, 1.0],
        [0.5, -1.0, 0.25],
        [0.0, 0.0, 0.0],
        [0.75, 0.5, -1.5],
        [1.0, 1.0, 1.0],
    ];
    // The covariant image S⁻¹ M S of a diagonal mask under the monomial gauge is the
    // permuted diagonal mask, and the two factorizations execute one intervention.
    for mask in masks {
        let covariant: Vec<f64> = permutation.iter().map(|&old| mask[old]).collect();
        assert_eq!(execute(&gauged, &covariant), execute(&original, &mask), "mask {mask:?}");
    }
    // Null control: a scalar mask cI is intrinsic under every gauge, the shear included.
    for scalar in [0.5, -1.0, 0.0] {
        let mask = [scalar; 3];
        let reference = execute(&original, &mask);
        assert_eq!(execute(&gauged, &mask), reference, "scalar {scalar}");
        assert_eq!(execute(&sheared, &mask), reference, "scalar {scalar}");
    }
    // Positive control: the same diagonal mask on the sheared basis is a different
    // intervention (P1). Its covariant image is not diagonal, so no component mask on the
    // sheared basis names it; as an operator it does restore the intervention.
    let diagonal = [1.0, 0.0, 1.0];
    assert_ne!(execute(&sheared, &diagonal), execute(&original, &diagonal));
    let operator = Array2::from_diag(&Array1::from(diagonal.to_vec()));
    let conjugated = shear_inverse.dot(&operator).dot(&shear);
    assert!((0..3).any(|row| (0..3).any(|column| row != column && conjugated[[row, column]] != 0.0)));
    assert_eq!(
        sheared_write.dot(&conjugated).dot(&sheared_read),
        write.dot(&operator).dot(&read)
    );

    // The moments see the same gauge. Component c's generator is vec(u_c r_cᵀ) in one H·d
    // block, and q(m) = vec(U (I − M) R) is exactly what the rewrite removes from the
    // summed input.
    let moment_system = |write: &Array2<f64>, read: &Array2<f64>| {
        MaskMomentSystem::new(
            vec![MomentBlock { dimension: 4 }],
            (0..3)
                .map(|component| {
                    vec![GeneratorPart {
                        block: 0,
                        vector: Array1::from_shape_fn(4, |index| {
                            write[[index / 2, component]] * read[[component, index % 2]]
                        }),
                    }]
                })
                .collect(),
        )
        .expect("well-formed generators")
    };
    let domain = MaskDomain::new(vec![(-2.0, 2.0); 3]).expect("the interval contains all-on");
    let original_moments = moment_system(&write, &read);
    let gauged_moments = moment_system(&gauged_write, &gauged_read);
    let sheared_moments = moment_system(&sheared_write, &sheared_read);
    let native_summed_input = native.summed_input(inputs.view()).expect("native summed input");
    for mask in masks {
        let covariant: Vec<f64> = permutation.iter().map(|&old| mask[old]).collect();
        let moment = original_moments.moment(&domain, &mask).expect("admissible mask");
        assert_eq!(gauged_moments.moment(&domain, &covariant).expect("admissible mask"), moment);
        let removed = Array2::from_shape_vec((2, 2), moment.blocks[0].to_vec()).expect("an H × d moment");
        let component_mask = Array1::from(mask.to_vec());
        let summed_input = original
            .summed_input(inputs.view(), ComponentMask::Components(component_mask.view()))
            .expect("masked summed input");
        assert_eq!(summed_input, &native_summed_input - &inputs.dot(&removed.t()), "mask {mask:?}");
    }
    let free = [false; 3];
    for values in [[1.0, 0.0, 0.0, 0.0], [2.0, -1.0, 3.0, 1.0], [-1.0, 4.0, 1.0, -2.0]] {
        let direction = MomentVector {
            blocks: vec![Array1::from(values.to_vec())],
        };
        let before = original_moments.support(&domain, &free, &direction).expect("valid direction");
        let after = gauged_moments.support(&domain, &free, &direction).expect("valid direction");
        assert_eq!(after.value, before.value, "direction {values:?}");
        let permuted: Vec<WitnessEndpoint> = permutation.iter().map(|&old| before.witness[old]).collect();
        assert_eq!(after.witness, permuted, "direction {values:?}");
    }
    // Positive control: the shear moves the generators while W₁ stays put, so the
    // admissible zonotope's support in direction e₁ is 10 against 6.
    let probe = MomentVector {
        blocks: vec![array![0.0, 1.0, 0.0, 0.0]],
    };
    assert_eq!(original_moments.support(&domain, &free, &probe).expect("valid direction").value, 6.0);
    assert_eq!(sheared_moments.support(&domain, &free, &probe).expect("valid direction").value, 10.0);
}

/// Which P15 certificate a [`LogitMaskOracle`] derives over the declared mask box.
#[derive(Clone, Copy, Debug)]
enum LogitCertificate {
    /// `KL ≤ osc(δ)²/8`. For `δ = −q`, `δ_k − δ_l = ⟨e_l − e_k, q⟩`, so the supremum of the oscillation over the box is
    /// the largest support `h(e_l − e_k)` over ordered logit pairs. A constant shift of the logits changes no
    /// probability, so the bound is read through the shifted gap `‖δ − c·1‖_∞ = osc(δ)/2` of the best shift `c`.
    Oscillation,
    /// `KL ≤ ‖δ‖_∞²/2` with `‖δ‖_∞` bounded by the largest absolute support of a single logit `e_k`: blind to the
    /// softmax gauge.
    SupremumGap,
}

/// Separation for logits affine in the moment, `z(m) = z_* − q(m)`, with one moment coordinate per logit.
///
/// The upper side is `bounds`' P15 certificate over the whole declared box, fed a support-function bound on the logit
/// gap from `moments`. The lower side is gam-math's categorical KL with its rounding bound at the support function's
/// witness mask. The fixtures' reference logits and generators are dyadic with a few bits, so the perturbed logits are
/// exact and the categorical rounding bound is the whole numerical error.
struct LogitMaskOracle {
    system: MaskMomentSystem,
    domain: MaskDomain,
    reference: Vec<f64>,
    certificate: LogitCertificate,
}

impl LogitMaskOracle {
    fn divergence(&self, mask: &[f64]) -> Result<(f64, f64), String> {
        let moment = self
            .system
            .moment(&self.domain, mask)
            .map_err(|error| format!("{error:?}"))?;
        let perturbed: Vec<f64> = self
            .reference
            .iter()
            .zip(moment.blocks[0].iter())
            .map(|(logit, shift)| logit - shift)
            .collect();
        categorical_kl_from_logits_with_error(&self.reference, &perturbed).map_err(|error| error.to_string())
    }

    /// The certificate's upper bound on `sup KL` over the masks that keep `support`, and a witness mask.
    fn certify(&self, support: &ComponentSet) -> Result<(f64, Vec<f64>), String> {
        let kept: Vec<bool> = (0..self.system.control_count())
            .map(|control| support.members().binary_search(&control).is_ok())
            .collect();
        let logits = self.reference.len();
        let unit = |index: usize| Array1::from_shape_fn(logits, |coordinate| if coordinate == index { 1.0 } else { 0.0 });
        let mut value = f64::NEG_INFINITY;
        let mut band = 0.0_f64;
        let mut witness = Vec::new();
        for first in 0..logits {
            for second in 0..logits {
                let evaluation = match self.certificate {
                    LogitCertificate::Oscillation if first != second => self.system.support(
                        &self.domain,
                        &kept,
                        &MomentVector {
                            blocks: vec![&unit(first) - &unit(second)],
                        },
                    ),
                    LogitCertificate::SupremumGap if first == second => self.system.absolute_supremum(
                        &self.domain,
                        &kept,
                        &MomentVector {
                            blocks: vec![unit(first)],
                        },
                    ),
                    LogitCertificate::Oscillation | LogitCertificate::SupremumGap => continue,
                }
                .map_err(|error| format!("{error:?}"))?;
                band = band.max(evaluation.band);
                if evaluation.value > value {
                    value = evaluation.value;
                    witness = evaluation.witness;
                }
            }
        }
        // Each computed support lies within its band of the exact one, so the largest computed value plus the largest
        // band, rounded up, bounds the exact maximum. Halving it is exact.
        let rounded = value + band;
        let bound = if band == 0.0 { rounded } else { rounded.next_up() };
        let gap = match self.certificate {
            LogitCertificate::Oscillation => bound / 2.0,
            LogitCertificate::SupremumGap => bound,
        };
        let upper = kl_bound_from_logit_gap(gap, LogitGapNorm::Supremum)
            .map_err(|error| error.to_string())?
            .upper_bound()
            .ok_or_else(|| "a uniform bound has an upper side".to_string())?;
        let mask = self.domain.mask_at(&witness).map_err(|error| format!("{error:?}"))?;
        Ok((upper, mask))
    }
}

impl SeparationOracle for LogitMaskOracle {
    type Mask = Vec<f64>;
    type Domain = &'static str;
    type Error = String;

    fn components(&self) -> usize {
        self.system.control_count()
    }

    fn perturbed_components(&self, mask: &Vec<f64>) -> Vec<usize> {
        perturbed(mask)
    }

    fn separate(&mut self, support: &ComponentSet) -> Result<EvidenceStatus<Vec<f64>, &'static str>, String> {
        let (upper, mask) = self.certify(support)?;
        let (divergence, error) = self.divergence(&mask)?;
        let lower = (divergence - error).next_down();
        EvidenceStatus::unresolved(
            lower,
            upper,
            Extremum::Supremum,
            Some(mask),
            "P15 certificate over the declared mask box, categorical KL at the witness",
        )
        .map_err(|error| error.to_string())
    }

    fn evaluate(&mut self, mask: &Vec<f64>) -> Result<EvidenceStatus<Vec<f64>, &'static str>, String> {
        let (divergence, error) = self.divergence(mask)?;
        EvidenceStatus::exact(
            divergence,
            error,
            ExactBasis::Exhaustive { cardinality: 1 },
            Some(mask.clone()),
            "one declared mask",
        )
        .map_err(|error| error.to_string())
    }
}

#[test]
fn a_softmax_gauge_component_is_dropped_by_the_oscillation_certificate_and_kept_by_a_gap_norm_certificate_2951() {
    // Logits z(m) = z_* − q(m) over three controls: a real edit, the constant shift (1, 1, 1), and a small edit.
    let reference = vec![0.5, -0.25, 1.0];
    let generators = [[2.0, -1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.5, -0.5]];
    let tolerance = 0.25;
    let oracle = |certificate: LogitCertificate| LogitMaskOracle {
        system: MaskMomentSystem::new(
            vec![MomentBlock { dimension: 3 }],
            generators.iter().map(|vector| vec![part(0, vector)]).collect(),
        )
        .expect("well-formed generators"),
        domain: unit_domain(3),
        reference: reference.clone(),
        certificate,
    };
    let mut oscillation = oracle(LogitCertificate::Oscillation);
    let mut supremum_gap = oracle(LogitCertificate::SupremumGap);

    // Null control: deleting the shift alone moves every logit by one and no probability. The categorical KL is zero
    // within its rounding bound, and bounds' oscillation bound on that pair sits at its rounding floor, while the
    // sup-norm gap bound of the same pair is 1/2.
    let shifted: Vec<f64> = reference.iter().map(|logit| logit - 1.0).collect();
    let (shift_divergence, shift_error) = oscillation.divergence(&[1.0, 0.0, 1.0]).expect("valid logits");
    assert!(shift_divergence - shift_error <= 0.0, "KL {shift_divergence} error {shift_error}");
    let floor = accumulation_band(8, 2.0);
    let shift_oscillation = softmax_kl_oscillation_bound(ArrayView1::from(&reference[..]), ArrayView1::from(&shifted[..]))
        .expect("finite logits")
        .upper_bound()
        .expect("a uniform bound has an upper side");
    assert!(shift_oscillation <= floor * floor, "oscillation bound {shift_oscillation}");
    let shift_gap = kl_bound_from_logit_gap(1.0, LogitGapNorm::Supremum)
        .expect("finite gap")
        .upper_bound()
        .expect("a uniform bound has an upper side");
    assert!(shift_gap > tolerance, "gap bound {shift_gap}");

    // The oscillation certificate closes the search with the gauge component free: keeping control 0 leaves
    // osc ≤ 1 and KL ≤ 1/8.
    let oscillation_search = minimum_code_support(&mut oscillation, &SizeCode, tolerance, FailureHypergraph::new(3))
        .expect("the oscillation search closes");
    assert_eq!(
        oscillation_search.certified.as_ref().expect("a certified support").support.members(),
        &[0]
    );
    assert!(matches!(
        oscillation_search.code,
        EvidenceStatus::Exact {
            basis: ExactBasis::ClosedSearch,
            ..
        }
    ));
    assert_eq!(oscillation_search.code.upper_bound(), Some(1.0));

    // Positive control: the sup-norm gap certificate sees the shift in every logit. It never certifies {0}, whose
    // witness is harmless, so its search ends unresolved on {0}, and only keeping the shift too certifies.
    let gap_search = minimum_code_support(&mut supremum_gap, &SizeCode, tolerance, FailureHypergraph::new(3))
        .expect("the gap search ends");
    assert!(matches!(gap_search.code, EvidenceStatus::Unresolved { .. }), "{:?}", gap_search.code);
    assert_eq!(gap_search.undecided.as_ref().expect("an undecided candidate").support.members(), &[0]);
    assert!(!supremum_gap.separate(&set(3, &[0])).expect("valid support").certifies_at_most(tolerance));
    assert!(supremum_gap.separate(&set(3, &[0, 1])).expect("valid support").certifies_at_most(tolerance));

    // Both certificates dominate the categorical KL at every endpoint mask every support admits.
    let mut largest_violation_of_a_quartered_certificate = f64::NEG_INFINITY;
    for kept_bits in 0..8usize {
        let members: Vec<usize> = (0..3).filter(|control| kept_bits >> control & 1 == 1).collect();
        let support = set(3, &members);
        let (oscillation_upper, oscillation_witness) = oscillation.certify(&support).expect("valid support");
        let (gap_upper, gap_witness) = supremum_gap.certify(&support).expect("valid support");
        for witness in [&oscillation_witness, &gap_witness] {
            assert!(members.iter().all(|&control| witness[control] == 1.0), "the witness keeps the support");
        }
        for mask_bits in 0..8usize {
            if (0..3).any(|control| kept_bits >> control & 1 == 1 && mask_bits >> control & 1 == 0) {
                continue;
            }
            let mask: Vec<f64> = (0..3).map(|control| (mask_bits >> control & 1) as f64).collect();
            let (divergence, error) = oscillation.divergence(&mask).expect("valid logits");
            assert!(divergence - error <= oscillation_upper, "support {members:?} mask {mask:?}");
            assert!(divergence - error <= gap_upper, "support {members:?} mask {mask:?}");
            largest_violation_of_a_quartered_certificate =
                largest_violation_of_a_quartered_certificate.max(divergence - error - oscillation_upper / 4.0);
        }
    }
    // Positive control: the dominance check has teeth. A certificate a quarter as large is violated at some mask.
    assert!(largest_violation_of_a_quartered_certificate > 0.0);
}

#[test]
fn a_collinear_family_resolves_one_mode_but_only_positive_members_merge_into_one_control_2951() {
    // Three 1 × 2 components per case, read both as family members and as moment generators.
    let readout = MomentVector {
        blocks: vec![array![1.0, 0.0]],
    };
    let cases: [(&str, [[f64; 2]; 3], usize, Vec<Vec<usize>>); 3] = [
        // A positive collinear refinement of (4, 2).
        ("refined", [[2.0, 1.0], [1.0, 0.5], [1.0, 0.5]], 1, vec![vec![0, 1, 2]]),
        // The same direction with a sign change: a cancelling pair.
        ("antiparallel", [[2.0, 1.0], [-2.0, -1.0], [1.0, 0.5]], 1, vec![vec![0, 2], vec![1]]),
        // Pieces that sum to (4, 2) but span the plane.
        ("bent", [[3.0, 1.0], [0.0, 2.0], [1.0, -1.0]], 2, vec![vec![0], vec![1], vec![2]]),
    ];
    for (name, generators, resolved, groups) in cases {
        let tensors: Vec<Array2<f64>> = generators.iter().map(|vector| array![[vector[0], vector[1]]]).collect();
        let views: Vec<_> = tensors.iter().map(|tensor| tensor.view()).collect();
        let field = principal_field(&views, &[0, 1, 2], resolved).expect("the resolved dimension is admitted");
        assert_eq!(field.resolved_dimension, resolved, "{name}: eigenvalues {:?}", field.eigenvalues);
        if resolved == 1 {
            assert!(
                matches!(
                    principal_field(&views, &[0, 1, 2], 2),
                    Err(FamilyError::UnresolvedDimension { resolved: 1, .. })
                ),
                "{name}: a collinear family has one mode"
            );
        }
        let system = planar_system(&generators);
        let merge = system.merge_positive_collinear(&unit_domain(3)).expect("valid domain");
        assert_eq!(merge.groups, groups, "{name}");
        let supremum = system
            .absolute_supremum(&unit_domain(3), &[false; 3], &readout)
            .expect("valid direction");
        let net = system
            .moment(&unit_domain(3), &[0.0; 3])
            .expect("admissible mask")
            .blocks[0][0]
            .abs();
        match name {
            // One family mode and one control: deleting everything attains the supremum.
            "refined" => assert_eq!((supremum.value, net), (4.0, 4.0)),
            // Positive control: one family mode, yet the cancelling pair's supremum 3 is not its net deletion 1, so the
            // family label cannot stand for one ablation control.
            "antiparallel" => assert_eq!((supremum.value, net), (3.0, 1.0)),
            _ => assert_eq!(merge.groups.len(), 3, "{name}: two modes and no merge"),
        }
    }
}
