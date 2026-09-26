#![cfg(test)]
//! Cross-module adversarial and null controls for the manifold parameter
//! decomposition (#2951), against the landed modules only.
//!
//! Each test composes at least two modules and carries a positive control that the
//! guard under test must catch:
//!
//! * a `Q, −Q` junk pair reproduces the all-on block exactly yet enters every
//!   sufficient support, and tying its two masks makes it inert (`rewrite`,
//!   `supports`; A13 negative control);
//! * the identity program is shorter than the projector family when the family's
//!   labels are shipped at declared precision and fidelity is measured on the
//!   decoded artifact (`codec`, `precision`; P11, P18, A8);
//! * a redundant pair is one OR edge, and an interior mask keeps a component that
//!   endpoint masks miss (`rewrite`, `supports`; P7, P12, A7);
//! * gauge-equivalent factorizations execute one intervention under covariant
//!   masks and move no generator, while a diagonal mask on a sheared basis is a
//!   different intervention (`rewrite`, `moments`; P1, A2).

use std::f64::consts::PI;

use gam_linalg::roundoff::accumulation_growth;
use gam_math::gaussian_activation::GaussianActivation;
use ndarray::{Array1, Array2, array};

use super::codec::{
    BitString, DagNode, LibraryPacketArtifact, code_saving_at_proven_fidelity, decode_fixed_index, decode_ordered_dag,
    decode_prefix_integer, decode_support_packet, encode_fixed_index, encode_ordered_dag, encode_prefix_integer,
    encode_support_packets, prefix_integer_len_bits, subset_code_len_bits, union_support_library,
};
use super::moments::{GeneratorPart, MaskDomain, MaskMomentSystem, MomentBlock, MomentVector, WitnessEndpoint};
use super::precision::{DecodableArtifact, PeriodicQuotient, QuotientCode, decode_then_evaluate};
use super::rewrite::{ComponentMask, ComponentMlp, ComponentRead, MlpMask, NativeMlp};
use super::supports::{ComponentSet, EvidenceStatus, ExactBasis, SeparationOracle, ranked_support};

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
fn a_q_minus_q_junk_pair_reproduces_the_all_on_block_and_enters_every_sufficient_support_2951() {
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

    // So every sufficient support pays for the pair: dropping either junk component alone is
    // refuted, and by upward closure no support without it is sufficient.
    let clean_search = ranked_support(&mut clean_oracle, &[0, 1], tolerance).expect("the clean run is certified");
    let junk_search = ranked_support(&mut junk_oracle, &[0, 1, 2, 3], tolerance).expect("the junk run is certified");
    assert_eq!(clean_search.found.support.members(), &[0, 1]);
    assert!(clean_search.found.evidence.certifies_at_most(tolerance));
    assert_eq!(junk_search.found.support.members(), &[0, 1, 2, 3]);
    assert!(junk_search.found.evidence.certifies_at_most(tolerance));
    for dropped in [2usize, 3] {
        let kept: Vec<usize> = (0..4).filter(|&component| component != dropped).collect();
        assert!(
            junk_oracle.separate(&set(4, &kept)).expect("exhaustive separation").refutes_at_most(tolerance),
            "junk component {dropped} is needed"
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

    // At the endpoints either member of the pair suffices alone, while dropping both is refuted:
    // the pair is one OR edge, and endpoint masks never see component 2.
    for member in [0usize, 1] {
        assert!(endpoints.separate(&set(3, &[member])).expect("exhaustive separation").certifies_at_most(tolerance));
    }
    let neither = endpoints.separate(&set(3, &[])).expect("exhaustive separation");
    assert!(neither.refutes_at_most(tolerance));
    let endpoint_search = ranked_support(&mut endpoints, &[0, 1, 2], tolerance).expect("the whole run is certified");
    let endpoint_support = endpoint_search.found.support.clone();
    assert_eq!(endpoint_support.members(), &[0]);
    assert!(endpoint_search.found.evidence.certifies_at_most(tolerance));

    // Interior masks need component 2 and still one member of the pair.
    let interior_search = ranked_support(&mut interior, &[0, 2, 1], tolerance).expect("the whole run is certified");
    assert_eq!(interior_search.found.support.members(), &[0, 2]);
    assert!(interior_search.found.evidence.certifies_at_most(tolerance));
    assert_eq!(
        interior_search.refuted_shorter.as_ref().map(|shorter| shorter.support.members().to_vec()),
        Some(vec![0])
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
