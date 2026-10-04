//! Causal coefficient calibration; the known fixture is not a discovery claim.
use crate::{
    artifact::Artifact,
    down_edit_family::{self, Direction, Family},
    operator_program::{
        Declarations, FamilyInputs, Interface, Law, Node, Operator, OperatorBody, OperatorProgram,
        Slot, SlotValues, exact_precision,
    },
    parameter_response_program::compose,
};
use ndarray::{Array2, array};
use std::{collections::BTreeMap, sync::Arc};
fn dense(values: Array2<f64>) -> Arc<Operator> {
    Arc::new(
        Operator::dense(
            "test coefficient",
            Interface::native(values.nrows()).expect("rows"),
            Interface::native(values.ncols()).expect("cols"),
            values.clone(),
            exact_precision(values.iter().copied()).expect("precision"),
            Default::default(),
        )
        .expect("dense"),
    )
}
fn setup() -> (Family, Artifact, usize) {
    let clean = OperatorProgram {
        declarations: Declarations {
            domains: vec![],
            slots: vec![Slot::Raw { width: 2 }],
            parameters: 0,
        },
        bases: vec![],
        rules: vec![],
        operators: vec![
            dense(array![[1., 0.5], [-0.25, 1.]]),
            dense(array![[1., -0.5], [0.25, 0.75]]),
        ],
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Affine {
                terms: vec![(0, 0)],
                bias: None,
            },
            Node::Pointwise {
                input: 1,
                laws: vec![Law::Relu],
            },
            Node::Affine {
                terms: vec![(2, 1)],
                bias: None,
            },
        ],
        output: 3,
    };
    let mut native = clean.clone();
    native.operators.push(Arc::new(Operator::identity(
        "native downstream",
        Interface::native(2).expect("width"),
    )));
    native.nodes.push(Node::Affine {
        terms: vec![(3, 2), (0, 2)],
        bias: None,
    });
    native.output = 4;
    let family = down_edit_family::build(
        &native,
        0,
        3,
        &[Direction {
            output: array![1., -1.],
            hidden: array![0.5, -0.25],
        }],
    )
    .expect("family");
    let mut response = clean.clone();
    let wrong = dense(array![[0.125, 0.125]]);
    response.operators[1] = wrong.clone();
    let function = compose(
        &clean,
        &[response],
        &[family.program.operators[family.direction_operators[0].0].clone()],
    )
    .expect("compose");
    let reads = [family.native_read, family.control_nodes[0]];
    let mut candidate = Artifact::native(&family.program)
        .expect("artifact")
        .replace_function_inputs("causal fit fixture", &function, &reads, family.native_write)
        .expect("graft");
    family
        .retain_directions(&mut candidate)
        .expect("paid directions");
    candidate
        .validate_coverage(&family.program)
        .expect("coverage");
    assert!(candidate.place(family.node_mapping[2]).is_none());
    let parameter = candidate
        .program
        .operators
        .iter()
        .position(|op| Arc::ptr_eq(op, &wrong))
        .expect("trainable response");
    (family, candidate, parameter)
}
fn softmax(x: ndarray::ArrayView1<'_, f64>) -> Vec<f64> {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<_> = x.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.iter().map(|e| e / sum).collect()
}
fn target(family: &Family, x: &Array2<f64>, a: f64) -> Array2<f64> {
    let source = family.literal_native(&[a]).expect("native parameter edit");
    let base = FamilyInputs {
        rows: x.nrows(),
        slots: vec![SlotValues::Raw(x.clone())],
        layout: None,
    };
    source.execute(&base, false).expect("target").values[source.output].clone()
}
fn inputs(family: &Family, x: &Array2<f64>, a: f64) -> FamilyInputs {
    family
        .inputs(
            &FamilyInputs {
                rows: x.nrows(),
                slots: vec![SlotValues::Raw(x.clone())],
                layout: None,
            },
            &[a],
        )
        .expect("global scalar inputs")
}
fn loss(family: &Family, candidate: &Artifact, x: &Array2<f64>, amplitudes: &[f64]) -> f64 {
    let mut sum = 0.;
    for &a in amplitudes {
        let p = target(family, x, a);
        let trace = candidate
            .execute(&inputs(family, x, a))
            .expect("own prediction");
        let q = &trace.values[candidate.program.output];
        for row in 0..x.nrows() {
            sum += crate::acceptance::kl_logits(p.row(row), q.row(row)).0;
        }
    }
    sum / (x.nrows() * amplitudes.len()) as f64
}
fn values(candidate: &Artifact, parameter: usize) -> Array2<f64> {
    match &candidate.program.operators[parameter].body {
        OperatorBody::Dense { values, .. } => Some(values.clone()),
        _ => None,
    }
    .expect("dense test coefficient")
}
fn with_values(candidate: &Artifact, parameter: usize, values: Array2<f64>) -> Artifact {
    let mut out = candidate.clone();
    out.program.operators[parameter] = dense(values);
    out
}
fn gradient(
    family: &Family,
    candidate: &Artifact,
    parameter: usize,
    x: &Array2<f64>,
    amplitudes: &[f64],
) -> Array2<f64> {
    use crate::{artifact_device::mapped_inlined, device_program::DeviceProgram};
    use gam_gpu::tensor::{Arithmetic, Device};
    let device = Device::host();
    let (flat, _) = mapped_inlined(&candidate.program).expect("explicit calls");
    let graph = DeviceProgram::compile_values(&device, &flat).expect("values graph");
    let mut total = Array2::zeros(values(candidate, parameter).dim());
    for &a in amplitudes {
        let f = inputs(family, x, a);
        let trace = graph.forward(&f).expect("intervened forward");
        let q = device
            .download(trace.value(flat.output).expect("resident output"))
            .expect("test logits");
        let p = target(family, x, a);
        let mut seed = Array2::zeros(q.dim());
        for row in 0..x.nrows() {
            let pp = softmax(p.row(row));
            let qq = softmax(q.row(row));
            for col in 0..q.ncols() {
                seed[[row, col]] = (qq[col] - pp[col]) / (x.nrows() * amplitudes.len()) as f64;
            }
        }
        let (_, g) = graph
            .vjp_values_dense(
                &trace,
                BTreeMap::from([(flat.output, device.upload(seed.view()).expect("seed"))]),
                &[],
                &[parameter],
                Arithmetic::F64,
            )
            .expect("intervention VJP");
        total += &device.download(&g[&parameter]).expect("gradient");
    }
    total
}
#[test]
fn wrong_response_is_invisible_clean_but_has_correct_intervened_kl_gradient() {
    let (f, c, p) = setup();
    let x = array![[0.6, -0.2], [-0.4, 0.8]];
    assert!(loss(&f, &c, &x, &[0.]).abs() < 1e-13);
    assert!(
        gradient(&f, &c, p, &x, &[0.])
            .iter()
            .all(|v| v.abs() < 1e-13)
    );
    let amplitudes = [-0.5, 0.5];
    let g = gradient(&f, &c, p, &x, &amplitudes);
    assert!(g.iter().any(|v| v.abs() > 1e-4));
    for column in 0..2 {
        let mut plus = values(&c, p);
        let mut minus = plus.clone();
        plus[[0, column]] += 1e-6;
        minus[[0, column]] -= 1e-6;
        let difference = (loss(&f, &with_values(&c, p, plus), &x, &amplitudes)
            - loss(&f, &with_values(&c, p, minus), &x, &amplitudes))
            / 2e-6;
        assert!(
            (g[[0, column]] - difference).abs() < 1e-8,
            "column {column}: {} vs {difference}",
            g[[0, column]]
        );
    }
}
#[test]
fn causal_updates_improve_saved_predictions_on_unfitted_contexts_and_edit_strengths() {
    let (f, mut c, p) = setup();
    let train = array![[0.6, -0.2], [-0.4, 0.8]];
    let heldout = array![[1., -0.3], [-0.6, 1.1]];
    let amplitudes = [-0.5, 0.5];
    let heldout_amplitudes = [-0.75, 0.25];
    let initial = loss(&f, &c, &train, &amplitudes);
    let initial_heldout = loss(&f, &c, &heldout, &heldout_amplitudes);
    for _ in 0..32 {
        let g = gradient(&f, &c, p, &train, &amplitudes);
        let v = values(&c, p);
        let current = loss(&f, &c, &train, &amplitudes);
        let mut accepted = None;
        for rate in [1., 0.5, 0.25, 0.125] {
            let next = with_values(&c, p, &v - &(&g * rate));
            if loss(&f, &next, &train, &amplitudes) < current {
                accepted = Some(next);
                break;
            }
        }
        if let Some(next) = accepted {
            c = next;
        } else {
            break;
        }
    }
    let bytes = c
        .f32_literals()
        .expect("F32 snapshot")
        .to_bytes()
        .expect("ordinary saved bytes");
    let saved = Artifact::from_bytes(&bytes, &f.program.declarations).expect("independent replay");
    saved.validate_coverage(&f.program).expect("coverage");
    assert!(loss(&f, &saved, &train, &amplitudes) < initial * 0.25);
    assert!(loss(&f, &saved, &heldout, &heldout_amplitudes) < initial_heldout * 0.25);
    assert!(loss(&f, &saved, &heldout, &[0.]).abs() < 1e-13);
}

#[test]
fn grouped_native_boundaries_preserve_shared_reader_and_autonomous_response() {
    use crate::operator_program::LabelKind;
    let grouped = Interface::uniform(2, 1, LabelKind::Unit, 0).expect("grouped native");
    let hidden = Interface::uniform(3, 1, LabelKind::Unit, 0).expect("native hidden");
    let native_input = Interface::native(2).expect("Raw");
    let up_values = array![[1., 0.5], [-0.25, 1.], [0.5, -1.]];
    let down_values = array![[1., -0.5, 0.25], [0.25, 0.75, -0.5]];
    let typed = |rows: Interface, cols: Interface, values: Array2<f64>| {
        Arc::new(
            Operator::dense(
                "native typed",
                rows,
                cols,
                values.clone(),
                exact_precision(values.iter().copied()).expect("precision"),
                Default::default(),
            )
            .expect("typed"),
        )
    };
    let native = OperatorProgram {
        declarations: Declarations {
            domains: vec![],
            slots: vec![Slot::Raw { width: 2 }],
            parameters: 0,
        },
        bases: vec![],
        rules: vec![],
        operators: vec![
            typed(grouped.clone(), native_input, Array2::eye(2)),
            typed(hidden.clone(), grouped.clone(), up_values.clone()),
            typed(grouped.clone(), hidden, down_values.clone()),
            Arc::new(Operator::identity("tail", grouped)),
        ],
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Affine {
                terms: vec![(0, 0)],
                bias: None,
            },
            Node::Affine {
                terms: vec![(1, 1)],
                bias: None,
            },
            Node::Pointwise {
                input: 2,
                laws: vec![Law::Relu; 3],
            },
            Node::Affine {
                terms: vec![(3, 2)],
                bias: None,
            },
            Node::Affine {
                terms: vec![(4, 3), (1, 3)],
                bias: None,
            },
        ],
        output: 5,
    };
    let family = down_edit_family::build(
        &native,
        1,
        4,
        &[Direction {
            output: array![1., -1.],
            hidden: array![0.5, -0.25, 0.125],
        }],
    )
    .expect("grouped family");
    let clean = OperatorProgram {
        declarations: Declarations {
            domains: vec![],
            slots: vec![Slot::Raw { width: 2 }],
            parameters: 0,
        },
        bases: vec![],
        rules: vec![],
        operators: vec![dense(up_values.clone()), dense(down_values)],
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Affine {
                terms: vec![(0, 0)],
                bias: None,
            },
            Node::Pointwise {
                input: 1,
                laws: vec![Law::Relu],
            },
            Node::Affine {
                terms: vec![(2, 1)],
                bias: None,
            },
        ],
        output: 3,
    };
    let mut response = clean.clone();
    response.operators[1] = dense(array![[0.5, -0.25, 0.125]]);
    let function = compose(
        &clean,
        &[response],
        &[family.program.operators[family.direction_operators[0].0].clone()],
    )
    .expect("grouped directions");
    let mut candidate = Artifact::native(&family.program)
        .expect("native")
        .replace_function_inputs(
            "grouped responses",
            &function,
            &[family.native_read, family.control_nodes[0]],
            family.native_write,
        )
        .expect("grouped graft");
    family
        .retain_directions(&mut candidate)
        .expect("paid directions");
    candidate
        .validate_coverage(&family.program)
        .expect("coverage");
    let reader_copies = candidate
        .program
        .operators
        .iter()
        .filter(|op| match &op.body {
            OperatorBody::Dense { values, .. } => values == &up_values,
            _ => false,
        })
        .count();
    assert_eq!(
        reader_copies, 1,
        "same learned reader must stay one global parameter"
    );
    let bytes = candidate.to_bytes().expect("bytes");
    let saved = Artifact::from_bytes(&bytes, &family.program.declarations).expect("saved");
    let x = array![[0.6, -0.2], [-0.4, 0.8]];
    for a in [-0.75, 0., 0.5] {
        let expected = target(&family, &x, a);
        let input = inputs(&family, &x, a);
        let actual =
            saved.execute(&input).expect("own response").values[saved.program.output].clone();
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
    }
}
