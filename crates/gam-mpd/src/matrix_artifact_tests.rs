use super::*;
use crate::acceptance::{CostCache, structural_cost};
use crate::matrix_rule::{Node as MatrixNode, PinvConvention};
use crate::operator_program::{Slot, SlotValues};
use ndarray::array;

fn model() -> OperatorProgram {
    let d = Interface::native(2).expect("valid matrix artifact fixture or codeword");
    let dense = |name, value: Array2<f64>| {
        Arc::new(
            Operator::dense(
                name,
                d.clone(),
                d.clone(),
                value,
                exact_precision([4.0, 0.5]).expect("valid matrix artifact fixture or codeword"),
                Provenance::default(),
            )
            .expect("valid matrix artifact fixture or codeword"),
        )
    };
    let diagonal = |name, value: Array1<f64>| {
        Arc::new(
            Operator::diag(
                name,
                d.clone(),
                value,
                exact_precision([4.0, 0.5]).expect("valid matrix artifact fixture or codeword"),
                Provenance::default(),
            )
            .expect("valid matrix artifact fixture or codeword"),
        )
    };
    OperatorProgram {
        declarations: Declarations {
            parameters: 0,
            domains: vec![],
            slots: vec![Slot::Raw { width: 2 }],
        },
        bases: vec![],
        rules: vec![],
        operators: vec![
            dense("V", array![[2.0, 0.0], [0.0, 4.0]]),
            diagonal("g", array![3.0, 2.0]),
            diagonal("gf", array![1.0, 4.0]),
            dense("A", Array2::eye(2)),
            dense("B", Array2::eye(2)),
            Arc::new(Operator::identity("I", d)),
        ],
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Affine {
                terms: vec![(0, 1)],
                bias: None,
            },
            Node::Affine {
                terms: vec![(1, 0)],
                bias: None,
            },
            Node::Affine {
                terms: vec![(2, 3)],
                bias: None,
            },
            Node::Affine {
                terms: vec![(2, 4)],
                bias: None,
            },
            Node::Affine {
                terms: vec![(3, 5), (4, 5)],
                bias: None,
            },
            Node::Affine {
                terms: vec![(5, 2)],
                bias: None,
            },
        ],
        output: 6,
    }
}
fn body() -> Arc<MatrixRule> {
    Arc::new(MatrixRule {
        inputs: vec![
            MatrixType::Matrix { rows: 2, cols: 2 },
            MatrixType::Vector { len: 2 },
            MatrixType::Vector { len: 2 },
        ],
        nodes: vec![
            MatrixNode::Param { index: 0 },
            MatrixNode::Param { index: 1 },
            MatrixNode::Param { index: 2 },
            MatrixNode::Divide {
                numerator: 1,
                denominator: 2,
            },
            MatrixNode::Diag { input: 3 },
            MatrixNode::Pinv {
                input: 0,
                convention: PinvConvention::SvdResolutionBand,
            },
            MatrixNode::MatMul { left: 4, right: 5 },
            MatrixNode::Scale {
                input: 6,
                coefficient: 0.75,
            },
        ],
        output: 7,
    })
}
fn generic() -> Artifact {
    let native = Artifact::native(&model()).expect("valid matrix artifact fixture or codeword");
    native
        .derive(
            3,
            OperatorLaw::Expression {
                body: body(),
                sources: vec![0, 1, 2],
            },
            1.0,
            vec![],
        )
        .expect("valid matrix artifact fixture or codeword")
        .derive(
            4,
            OperatorLaw::Expression {
                body: body(),
                sources: vec![0, 1, 2],
            },
            0.5,
            vec![],
        )
        .expect("valid matrix artifact fixture or codeword")
        .bind("A", &[2], 3)
        .expect("valid matrix artifact fixture or codeword")
        .bind("B", &[2], 4)
        .expect("valid matrix artifact fixture or codeword")
}
fn family() -> FamilyInputs {
    FamilyInputs {
        rows: 2,
        layout: None,
        slots: vec![SlotValues::Raw(array![[1.0, -0.5], [0.0, 2.0]])],
    }
}
#[test]
fn versioned_body_is_shared_paid_once_and_replays_without_native_source() {
    let artifact = generic();
    artifact
        .validate_coverage(&model())
        .expect("valid matrix artifact fixture or codeword");
    assert_eq!(
        artifact
            .matrix_rules()
            .expect("valid matrix artifact fixture or codeword")
            .len(),
        1,
        "separate Arc bodies share only their exact encoded identity"
    );
    assert_eq!(
        artifact
            .derived_literals()
            .expect("valid matrix artifact fixture or codeword"),
        3,
        "two call scales plus one shared body coefficient"
    );
    let cost = structural_cost(&artifact, &mut CostCache::default())
        .expect("valid matrix artifact fixture or codeword");
    let expected = artifact
        .execute(&family())
        .expect("valid matrix artifact fixture or codeword")
        .values;
    let declarations = artifact.program.declarations.clone();
    let bytes = artifact
        .to_bytes()
        .expect("valid matrix artifact fixture or codeword");
    assert_eq!(&bytes[..8], &VERSIONED_ENVELOPE.to_le_bytes());
    assert_eq!(&bytes[8..16], &MATRIX_ARTIFACT_VERSION.to_le_bytes());
    assert_eq!(
        decode_prefix_integer(
            &mut artifact
                .encode()
                .expect("valid matrix artifact fixture or codeword")
                .reader()
        )
        .expect("valid matrix artifact fixture or codeword"),
        1
    );
    drop(artifact); // Replay retains no checkpoint/model/body oracle.
    let decoded = Artifact::from_bytes(&bytes, &declarations)
        .expect("valid matrix artifact fixture or codeword");
    assert_eq!(
        decoded
            .to_bytes()
            .expect("valid matrix artifact fixture or codeword"),
        bytes
    );
    assert_eq!(
        decoded
            .execute(&family())
            .expect("valid matrix artifact fixture or codeword")
            .values,
        expected
    );
    assert_eq!(
        structural_cost(&decoded, &mut CostCache::default())
            .expect("valid matrix artifact fixture or codeword"),
        cost
    );
    let [a, b] = decoded.derived.as_slice() else {
        panic!("two calls")
    };
    let (OperatorLaw::Expression { body: a, .. }, OperatorLaw::Expression { body: b, .. }) =
        (&a.law, &b.law)
    else {
        panic!("explicit bodies")
    };
    assert!(
        Arc::ptr_eq(a, b),
        "one decoded stored body supplies both calls"
    );
}
#[test]
fn pool_cost_matches_body_shapes_edges_parameters_and_call_wire() {
    let artifact = generic();
    let bodies = artifact
        .matrix_rules()
        .expect("valid matrix artifact fixture or codeword");
    let cost = bodies[0]
        .1
        .cost()
        .expect("valid matrix artifact fixture or codeword");
    let mut without_calls = artifact.clone();
    without_calls.derived.clear();
    let fixed =
        |n| u64::from(fixed_index_len_bits(n).expect("valid matrix artifact fixture or codeword"));
    let prefix = |n| prefix_integer_len_bits(n).expect("valid matrix artifact fixture or codeword");
    let call_bits = fixed(6) + fixed(3) + fixed(1) + 3 * fixed(6) + prefix(1);
    let expected = without_calls
        .binding_bits()
        .expect("valid matrix artifact fixture or codeword")
        - prefix(1)
        + prefix(1)
        + prefix(MATRIX_ARTIFACT_VERSION)
        + prefix(2)
        + prefix(bodies[0].0.len_bits() + 1)
        + cost.structure_bits
        + prefix(3)
        + 2 * call_bits;
    assert_eq!(
        artifact
            .binding_bits()
            .expect("valid matrix artifact fixture or codeword"),
        expected
    );
    assert_eq!(cost.c32(), bodies[0].0.len_bits());
    assert_eq!(
        artifact
            .derived_literals()
            .expect("valid matrix artifact fixture or codeword"),
        2 + cost.literals
    );
    let mut zero_a = artifact.clone();
    let mut zero_b = (*body()).clone();
    zero_b.nodes[7] = MatrixNode::Scale {
        input: 6,
        coefficient: -0.0,
    };
    zero_a.derived[1].law = OperatorLaw::Expression {
        body: Arc::new(zero_b.clone()),
        sources: vec![0, 1, 2],
    };
    zero_b.nodes[7] = MatrixNode::Scale {
        input: 6,
        coefficient: 0.0,
    };
    zero_a.derived[0].law = OperatorLaw::Expression {
        body: Arc::new(zero_b),
        sources: vec![0, 1, 2],
    };
    assert_eq!(
        zero_a
            .matrix_rules()
            .expect("valid matrix artifact fixture or codeword")
            .len(),
        2,
        "positive/negative zero are distinct body messages"
    );
    assert_eq!(
        zero_a
            .derived_literals()
            .expect("valid matrix artifact fixture or codeword"),
        4
    );
}
/// Independently spell the original empty-block/exception legacy grammar, so the
/// new encoder cannot redefine the expected message to make compatibility pass.
fn legacy_bytes(artifact: &Artifact) -> Vec<u8> {
    let program = artifact
        .message_program()
        .expect("valid matrix artifact fixture or codeword")
        .encode()
        .expect("valid matrix artifact fixture or codeword");
    let mut message = BitString::new();
    encode_prefix_integer(&mut message, artifact.native_nodes as u64 + 1)
        .expect("valid matrix artifact fixture or codeword");
    encode_prefix_integer(&mut message, program.len_bits() + 1)
        .expect("valid matrix artifact fixture or codeword");
    message.append(&program);
    encode_prefix_integer(&mut message, 1).expect("valid matrix artifact fixture or codeword");
    encode_prefix_integer(&mut message, artifact.places.len() as u64 + 1)
        .expect("valid matrix artifact fixture or codeword");
    for (native, node) in &artifact.places {
        encode_fixed_index(&mut message, *native, artifact.native_nodes)
            .expect("valid matrix artifact fixture or codeword");
        encode_fixed_index(&mut message, *node, artifact.program.nodes.len())
            .expect("valid matrix artifact fixture or codeword");
    }
    encode_prefix_integer(&mut message, 1).expect("valid matrix artifact fixture or codeword");
    encode_prefix_integer(&mut message, artifact.derived.len() as u64 + 1)
        .expect("valid matrix artifact fixture or codeword");
    for d in &artifact.derived {
        encode_fixed_index(&mut message, d.operator, artifact.program.operators.len())
            .expect("valid matrix artifact fixture or codeword");
        encode_fixed_index(&mut message, d.law.index(), LAWS)
            .expect("valid matrix artifact fixture or codeword");
        for source in d.law.sources() {
            encode_fixed_index(&mut message, source, artifact.program.operators.len())
                .expect("valid matrix artifact fixture or codeword");
        }
        for integer in d.law.integers() {
            encode_prefix_integer(&mut message, integer + 1)
                .expect("valid matrix artifact fixture or codeword");
        }
        message
            .push_bits(u64::from(d.scale.to_bits()), 32)
            .expect("valid matrix artifact fixture or codeword");
        encode_prefix_integer(&mut message, 1).expect("valid matrix artifact fixture or codeword");
    }
    let mut bytes = message.len_bits().to_le_bytes().to_vec();
    bytes.extend_from_slice(message.packed_bytes());
    bytes
}
#[test]
fn legacy_native_and_template_codec_bytes_are_unchanged() {
    let native = Artifact::native(&model()).expect("valid matrix artifact fixture or codeword");
    let legacy = native
        .derive(
            3,
            OperatorLaw::Copy {
                value: 0,
                gain: 1,
                final_gain: 2,
            },
            0.75,
            vec![],
        )
        .expect("valid matrix artifact fixture or codeword");
    for artifact in [native, legacy] {
        let expected = legacy_bytes(&artifact);
        assert_eq!(
            artifact
                .to_bytes()
                .expect("valid matrix artifact fixture or codeword"),
            expected
        );
        let decoded = Artifact::from_bytes(&expected, &artifact.program.declarations)
            .expect("valid matrix artifact fixture or codeword");
        assert_eq!(
            decoded
                .to_bytes()
                .expect("valid matrix artifact fixture or codeword"),
            expected
        );
        assert_eq!(
            artifact
                .derived_literals()
                .expect("valid matrix artifact fixture or codeword"),
            artifact.derived.len() as u64
        );
    }
}
/// Assemble new-format messages directly to test decoder guards that a valid
/// encoder refuses to emit. No source checkpoint is consulted by the decoder.
fn forged_message(
    artifact: &Artifact,
    bodies: &[BitString],
    source0: usize,
    scale: f32,
) -> BitString {
    let program = artifact
        .message_program()
        .expect("valid matrix artifact fixture or codeword")
        .encode()
        .expect("valid matrix artifact fixture or codeword");
    let mut out = BitString::new();
    for v in [
        1,
        MATRIX_ARTIFACT_VERSION,
        artifact.native_nodes as u64 + 1,
        program.len_bits() + 1,
    ] {
        encode_prefix_integer(&mut out, v).expect("valid matrix artifact fixture or codeword");
    }
    out.append(&program);
    encode_prefix_integer(&mut out, 1).expect("valid matrix artifact fixture or codeword");
    encode_prefix_integer(&mut out, artifact.places.len() as u64 + 1)
        .expect("valid matrix artifact fixture or codeword");
    for (native, node) in &artifact.places {
        encode_fixed_index(&mut out, *native, artifact.native_nodes)
            .expect("valid matrix artifact fixture or codeword");
        encode_fixed_index(&mut out, *node, artifact.program.nodes.len())
            .expect("valid matrix artifact fixture or codeword");
    }
    encode_prefix_integer(&mut out, 1).expect("valid matrix artifact fixture or codeword");
    encode_prefix_integer(&mut out, bodies.len() as u64 + 1)
        .expect("valid matrix artifact fixture or codeword");
    for body in bodies {
        encode_prefix_integer(&mut out, body.len_bits() + 1)
            .expect("valid matrix artifact fixture or codeword");
        out.append(body);
    }
    encode_prefix_integer(&mut out, 2).expect("valid matrix artifact fixture or codeword");
    encode_fixed_index(&mut out, 3, 6).expect("valid matrix artifact fixture or codeword");
    encode_fixed_index(&mut out, 2, MATRIX_LAWS)
        .expect("valid matrix artifact fixture or codeword");
    encode_fixed_index(&mut out, 0, bodies.len())
        .expect("valid matrix artifact fixture or codeword");
    for source in [source0, 1, 2] {
        encode_fixed_index(&mut out, source, 6).expect("valid matrix artifact fixture or codeword");
    }
    out.push_bits(u64::from(scale.to_bits()), 32)
        .expect("valid matrix artifact fixture or codeword");
    encode_prefix_integer(&mut out, 1).expect("valid matrix artifact fixture or codeword");
    out
}
#[test]
fn decoder_rejects_cycles_nonfinite_unused_bodies_and_bad_versions() {
    let artifact = generic();
    let declarations = &artifact.program.declarations;
    let message = body()
        .encode()
        .expect("valid matrix artifact fixture or codeword");
    let valid = forged_message(&artifact, std::slice::from_ref(&message), 0, 1.0);
    assert!(Artifact::decode(&valid, declarations).is_ok());
    assert!(
        Artifact::decode(
            &forged_message(&artifact, std::slice::from_ref(&message), 3, 1.0),
            declarations
        )
        .is_err()
    );
    assert!(
        Artifact::decode(
            &forged_message(&artifact, std::slice::from_ref(&message), 0, f32::INFINITY),
            declarations
        )
        .is_err()
    );
    assert!(
        Artifact::decode(
            &forged_message(&artifact, &[message.clone(), message.clone()], 0, 1.0),
            declarations
        )
        .is_err()
    );
    let mut unused = (*body()).clone();
    unused.nodes[7] = MatrixNode::Scale {
        input: 6,
        coefficient: 0.25,
    };
    assert!(
        Artifact::decode(
            &forged_message(
                &artifact,
                &[
                    message,
                    unused
                        .encode()
                        .expect("valid matrix artifact fixture or codeword")
                ],
                0,
                1.0
            ),
            declarations
        )
        .is_err()
    );
    let mut bytes = artifact
        .to_bytes()
        .expect("valid matrix artifact fixture or codeword");
    bytes[8..16].copy_from_slice(&3_u64.to_le_bytes());
    assert!(Artifact::from_bytes(&bytes, declarations).is_err());
    for length in 0..24 {
        assert!(Artifact::from_bytes(&bytes[..length], declarations).is_err());
    }
    let mut message = BitString::new();
    encode_prefix_integer(&mut message, 1).expect("valid matrix artifact fixture or codeword");
    encode_prefix_integer(&mut message, 3).expect("valid matrix artifact fixture or codeword");
    assert!(Artifact::decode(&message, declarations).is_err());
    let mut invalid = artifact.clone();
    invalid.derived[0].law = OperatorLaw::Expression {
        body: body(),
        sources: vec![3, 1, 2],
    };
    assert!(invalid.encode().is_err());
    assert!(structural_cost(&invalid, &mut CostCache::default()).is_err());
}
#[test]
fn native_codec_preserves_new_envelope_and_explicit_rule_cost() {
    let native = model();
    let cache = crate::operator_program::NativeOperatorCodec::new(&native, 1 << 20)
        .expect("valid matrix artifact fixture or codeword");
    let artifact = generic();
    // Different source Arc allocations must fall back safely; exact bytes still agree.
    let encoded = EncodedArtifact::of_with_native_codec(&artifact, &cache)
        .expect("valid matrix artifact fixture or codeword");
    assert_eq!(
        encoded.message,
        artifact
            .encode()
            .expect("valid matrix artifact fixture or codeword")
    );
    let decoded = encoded
        .using_native_codec(&cache)
        .decode()
        .expect("valid matrix artifact fixture or codeword");
    assert_eq!(
        decoded
            .to_bytes()
            .expect("valid matrix artifact fixture or codeword"),
        artifact
            .to_bytes()
            .expect("valid matrix artifact fixture or codeword")
    );
    assert_eq!(
        structural_cost(&decoded, &mut CostCache::default())
            .expect("valid matrix artifact fixture or codeword"),
        structural_cost(&artifact, &mut CostCache::default())
            .expect("valid matrix artifact fixture or codeword")
    );
}

#[test]
fn explicit_body_matches_template_measures_and_pays_its_extra_structure() {
    use crate::acceptance::{Constraint, Episode, FamilyRun, Local, assess_once};
    let native = model();
    let rows = family();
    let legacy = Artifact::native(&native)
        .expect("native fixture")
        .derive(
            3,
            OperatorLaw::Copy {
                value: 0,
                gain: 1,
                final_gain: 2,
            },
            0.75,
            vec![],
        )
        .expect("first template")
        .derive(
            4,
            OperatorLaw::Copy {
                value: 0,
                gain: 1,
                final_gain: 2,
            },
            0.375,
            vec![],
        )
        .expect("second template")
        .bind("A", &[2], 3)
        .expect("first boundary")
        .bind("B", &[2], 4)
        .expect("second boundary");
    let explicit = generic();
    assert_eq!(
        explicit.execute(&rows).expect("explicit execution").values,
        legacy.execute(&rows).expect("template execution").values
    );
    let local = Local::new(&native, rows.clone(), None, 16);
    let run = FamilyRun {
        model: &native,
        family: rows.clone(),
        readouts: 1,
        episodes: vec![Episode {
            id: "clean".into(),
            group: "clean".into(),
            edits: vec![],
        }],
    };
    let constraint = Constraint {
        local: 1.0,
        run: 1.0,
    };
    let a = assess_once(
        &local,
        &run,
        &explicit,
        constraint,
        &mut CostCache::default(),
    )
    .expect("decoded explicit assessment");
    let b = assess_once(&local, &run, &legacy, constraint, &mut CostCache::default())
        .expect("decoded template assessment");
    assert_eq!(a.local_measure, b.local_measure);
    assert_eq!(a.run_measure, b.run_measure);
    assert_eq!(a.local, b.local);
    assert_eq!(a.run, b.run);
    assert!(
        a.cost.total() > b.cost.total(),
        "identical execution does not erase the explicit definition price"
    );
}

#[test]
fn copy_template_expansion_preserves_calls_residuals_and_native_places() {
    let source = Artifact::native(&model())
        .expect("native fixture")
        .derive(
            3,
            OperatorLaw::Copy {
                value: 0,
                gain: 1,
                final_gain: 2,
            },
            0.75,
            vec![(0, vec![0.25, 0.125])],
        )
        .expect("template with residual")
        .bind("A", &[2], 3)
        .expect("native write");
    let explicit = source
        .expand_copy_templates()
        .expect("explicit arithmetic expansion");
    assert_eq!(explicit.program.nodes, source.program.nodes);
    assert_eq!(explicit.places, source.places);
    assert_eq!(explicit.blocks, source.blocks);
    assert_eq!(
        explicit.derived[0].scale.to_bits(),
        source.derived[0].scale.to_bits()
    );
    assert_eq!(explicit.derived[0].residual, source.derived[0].residual);
    assert!(matches!(
        explicit.derived[0].law,
        OperatorLaw::Expression { .. }
    ));
    assert_eq!(
        explicit
            .execute(&family())
            .expect("explicit execution")
            .values,
        source
            .execute(&family())
            .expect("template execution")
            .values
    );
    let decoded = Artifact::from_bytes(
        &explicit.to_bytes().expect("new bytes"),
        &source.program.declarations,
    )
    .expect("independent replay");
    assert_eq!(
        decoded
            .execute(&family())
            .expect("replayed execution")
            .values,
        source
            .execute(&family())
            .expect("template execution")
            .values
    );
    assert!(
        structural_cost(&explicit, &mut CostCache::default())
            .expect("explicit cost")
            .total()
            > structural_cost(&source, &mut CostCache::default())
                .expect("template cost")
                .total()
    );
}
