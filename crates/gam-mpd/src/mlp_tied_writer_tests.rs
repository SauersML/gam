use super::*;
use gam_mpd::operator_program::{Declarations, FamilyInputs, Interface, Law, Slot, SlotValues};
use ndarray::array;
fn native(gated: bool) -> (Artifact, FamilyInputs) {
    let model = Interface::native(2).expect("model interface");
    let hidden = Interface::native(3).expect("hidden interface");
    let dense = |name: &str, rows: Interface, cols: Interface, values: Array2<f64>| {
        Arc::new(
            Operator::dense(
                name,
                rows,
                cols,
                values.clone(),
                exact_precision(values.iter().copied()).expect("fixture precision"),
                Provenance::native(name),
            )
            .expect("fixture matrix"),
        )
    };
    let mut operators = vec![
        Arc::new(Operator::identity("I", model.clone())),
        dense("norm", model.clone(), model.clone(), Array2::eye(2)),
        dense(
            "blocks.7.c_fc",
            hidden.clone(),
            model.clone(),
            array![[1., 0.], [0., 1.], [1., 1.]],
        ),
        dense(
            "blocks.7.down_proj",
            model.clone(),
            hidden.clone(),
            array![[2., 0., 0.5], [0., -1., 0.5]],
        ),
        dense(
            "blocks.7.down_bias",
            model.clone(),
            Interface::constant(),
            array![[0.25], [-0.5]],
        ),
    ];
    let mut nodes = vec![
        Node::Raw { slot: 0 },
        Node::Affine {
            terms: vec![(0, 1)],
            bias: None,
        },
        Node::Affine {
            terms: vec![(1, 2)],
            bias: None,
        },
    ];
    let active = if gated {
        operators.push(dense(
            "blocks.7.gate_proj",
            hidden,
            model,
            array![[1., 0.], [0., 1.], [1., 1.]],
        ));
        nodes.push(Node::Affine {
            terms: vec![(1, 5)],
            bias: None,
        });
        nodes.push(Node::Pointwise {
            input: 3,
            laws: vec![Law::Silu],
        });
        nodes.push(Node::Hadamard { left: 4, right: 2 });
        5
    } else {
        nodes.push(Node::Pointwise {
            input: 2,
            laws: vec![Law::Silu],
        });
        3
    };
    nodes.push(Node::Affine {
        terms: vec![(0, 0), (active, 3)],
        bias: Some(4),
    });
    let p = OperatorProgram {
        declarations: Declarations {
            domains: vec![],
            slots: vec![Slot::Raw { width: 2 }],
            parameters: 0,
        },
        bases: vec![],
        rules: vec![],
        operators,
        output: nodes.len() - 1,
        nodes,
    };
    (
        Artifact::native(&p).expect("native fixture"),
        FamilyInputs {
            rows: 2,
            slots: vec![SlotValues::Raw(array![[1., 2.], [-1., 0.5]])],
            layout: None,
        },
    )
}
#[test]
fn gain_column_order_signs_and_exact_zero_reader_degeneracy() {
    let up = array![[1., 0.], [0., 2.], [0., 0.]];
    let down = array![[2., 0., 7.], [0., -1., 8.]];
    let fit = WriterFit::of(&up, &down).expect("transposed weights");
    assert_eq!(fit.gains, array![2., -0.5, 0.]);
    assert_eq!(fit.zero_reader_rows, 1);
    assert_eq!(fit.residual, array![[0., 0., 7.], [0., 0., 8.]]);
    assert!(WriterFit::of(&array![[1e-200]], &array![[1.]]).is_err()); // nonzero underflow is not exact degeneracy.
    assert!(WriterFit::of(&array![[1e200]], &array![[1.]]).is_err()); // explicit dot overflow, no silent zero fit.
}
#[test]
fn orthogonal_writer_counterexample_requires_full_residual_rank() {
    let fit = WriterFit::of(&Array2::eye(2), &array![[0., 1.], [1., 0.]]).expect("counterexample");
    assert_eq!(fit.gains, array![0., 0.]);
    assert_eq!(fit.residual.iter().map(|v| v * v).sum::<f64>(), 2.);
    let s = WriterSvd::of(&fit.residual).expect("SVD");
    let (base, _inputs) = native(false);
    let source = &base.program.operators[1];
    let rank1 = s
        .operator(source, 1, "counterexample rank1")
        .expect("rank1");
    assert!(
        (&fit.residual - &rank1.matrix())
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            > 0.99
    );
}
#[test]
fn full_hidden_and_gated_places_survive_rule_codec_and_edits() {
    for gated in [false, true] {
        let (base, inputs) = native(gated);
        let map = MlpWriterMap::all(&base.program)
            .expect("all MLPs")
            .remove(0);
        assert_eq!(map.layer, 7);
        assert_eq!(map.gate.is_some(), gated);
        let fit = WriterFit::of(
            &base.program.operators[map.reader].matrix(),
            &base.program.operators[map.writer].matrix(),
        )
        .expect("weight fit");
        let candidate = tied_candidate(&base, &map, &fit, None).expect("exact tied fixture");
        let decoded = Artifact::from_bytes(
            &candidate.to_bytes().expect("encode"),
            &base.program.declarations,
        )
        .expect("decode");
        decoded.validate_coverage(&base.program).expect("coverage");
        assert_eq!(decoded.places.len(), base.places.len());
        for place in [Some(map.pre), Some(map.active), Some(map.output), map.gate]
            .into_iter()
            .flatten()
        {
            let run = |a: &Artifact| {
                a.execute_edited(&inputs, |node, value, _| {
                    if Some(node) == a.place(place) {
                        *value *= 0.5;
                    }
                    Ok(())
                })
                .expect("held native edit")
            };
            let a = run(&base);
            let b = run(&decoded);
            let error = (&a.values[base.program.output] - &b.values[decoded.program.output])
                .iter()
                .fold(0_f64, |v, x| v.max(x.abs()));
            assert!(error < 1e-12, "error{error}");
        }
        let pre = decoded
            .place(map.pre)
            .expect("native full-hidden reader place");
        let native_reader = match &decoded.program.nodes[pre] {
            Node::Affine { terms, .. } => terms[0].1,
            _ => panic!("native reader is affine"),
        };
        let tied_reader = decoded
            .program
            .rules
            .iter()
            .flat_map(|r| &r.nodes)
            .filter_map(|n| match n {
                Node::Transposed { operator, .. } => Some(*operator),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(tied_reader, vec![native_reader]);
        let mut cache = gam_mpd::acceptance::CostCache::default();
        let before = gam_mpd::acceptance::structural_cost(&base, &mut cache).expect("native price");
        let after =
            gam_mpd::acceptance::structural_cost(&candidate, &mut cache).expect("tied price");
        assert_eq!(before.literals - after.literals, 3); // six writer reals removed; three gains paid.
    }
}
#[test]
fn full_rank_endpoint_and_zero_baseline_retain_hidden_interfaces() {
    let (mut base, inputs) = native(false);
    let old = base.program.operators[3].clone();
    base.program.operators[3] = Arc::new(
        Operator::dense(
            old.name.clone(),
            old.rows.clone(),
            old.cols.clone(),
            array![[2., 0.25, 0.5], [0.125, -1., 0.75]],
            exact_precision([2., 0.25, 0.5, 0.125, -1., 0.75]).expect("perturbed precision"),
            old.provenance.clone(),
        )
        .expect("perturbed writer"),
    );
    let map = MlpWriterMap::all(&base.program).expect("maps").remove(0);
    let down = &base.program.operators[map.writer];
    let s = WriterSvd::of(&down.matrix()).expect("native SVD");
    let zero = svd_candidate(
        &base,
        &map,
        s.operator(down, 0, "zero endpoint").expect("zero operator"),
    )
    .expect("zero candidate");
    assert_eq!(zero.places, base.places);
    assert_eq!(
        zero.execute_edited(&inputs, |_, _, _| Ok(()))
            .expect("zero forward")
            .values[zero.program.output],
        array![[1.25, 1.5], [-0.75, 0.]]
    );
    let fit =
        WriterFit::of(&base.program.operators[map.reader].matrix(), &down.matrix()).expect("fit");
    let residual = WriterSvd::of(&fit.residual)
        .expect("residual SVD")
        .operator(down, 2, "full residual")
        .expect("full rank");
    let candidate = tied_candidate(&base, &map, &fit, Some(residual)).expect("full candidate");
    assert_eq!(candidate.places.len(), base.places.len());
    assert!(fit.residual.iter().any(|v| *v != 0.));
    let expected = base
        .execute_edited(&inputs, |_, _, _| Ok(()))
        .expect("native full endpoint");
    let actual = candidate
        .execute_edited(&inputs, |_, _, _| Ok(()))
        .expect("numerical full endpoint");
    let error = (&expected.values[base.program.output] - &actual.values[candidate.program.output])
        .iter()
        .fold(0_f64, |a, b| a.max(b.abs()));
    assert!(
        error < 1e-6,
        "balanced f32 full-rank numerical endpoint error{error}"
    );
}
