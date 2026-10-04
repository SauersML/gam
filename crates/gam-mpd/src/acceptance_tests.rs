#![cfg(test)]
//! The one acceptance path on small programs built to catch its shortcuts: each test names the
//! shortcut (a)–(i) it guards against.

use super::acceptance::{
    Budget, Change, Constraint, Context, CostCache, Edit, Episode, FamilyRun, Local, Outcome, Proposal, Proposer, RunCheck, assess, code_plus_kl,
    execution_cost, kl_logits, search, structural_cost,
};
use super::artifact::{Argument, Artifact, Callee};
use super::operator_program::{
    Declarations, FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance, Rule, Slot, SlotValues, exact_precision,
};
use super::precision::FidelityVerdict;
use ndarray::{Array1, Array2, array};
use std::sync::Arc;

fn dense(name: &str, rows: &Interface, cols: &Interface, values: Array2<f64>) -> Operator {
    let precision = exact_precision(values.iter().copied()).expect("a lattice");
    Operator::dense(name, rows.clone(), cols.clone(), values, precision, Provenance::native(name)).expect("a dense operator")
}

fn column(name: &str, rows: &Interface, values: &[f64]) -> Operator {
    dense(name, rows, &Interface::constant(), Array2::from_shape_vec((values.len(), 1), values.to_vec()).expect("a column"))
}

fn native(width: usize) -> Interface {
    Interface::native(width).expect("an interface")
}

fn raw_program(width: usize, operators: Vec<Operator>, nodes: Vec<Node>) -> OperatorProgram {
    let output = nodes.len() - 1;
    OperatorProgram {
        declarations: Declarations { domains: Vec::new(), slots: vec![Slot::Raw { width }], parameters: 0 },
        bases: Vec::new(),
        operators: operators.into_iter().map(Arc::new).collect(),
        rules: Vec::new(),
        nodes,
        output,
    }
}

fn raw_family(rows: Vec<Vec<f64>>) -> FamilyInputs {
    let width = rows[0].len();
    let n = rows.len();
    let values = Array2::from_shape_vec((n, width), rows.into_iter().flatten().collect()).expect("rows");
    FamilyInputs { rows: n, slots: vec![SlotValues::Raw(values)], layout: None }
}

fn grid(points: &[f64], width: usize) -> FamilyInputs {
    let mut rows: Vec<Vec<f64>> = vec![Vec::new()];
    for _ in 0..width {
        rows = rows.into_iter().flat_map(|row| points.iter().map(move |p| [row.clone(), vec![*p]].concat())).collect();
    }
    raw_family(rows)
}

fn clean_run<'a>(model: &'a OperatorProgram, family: &FamilyInputs) -> FamilyRun<'a> {
    FamilyRun {
        model,
        family: family.clone(),
        readouts: 1,
        episodes: vec![Episode { id: "clean".to_string(), group: "clean".to_string(), edits: Vec::new() }],
    }
}

/// A rule of one input of interface `input` whose body is `nodes` after `Param { 0 }` (node 0).
fn rule(name: &str, inputs: Vec<Interface>, nodes: Vec<Node>) -> Rule {
    let output = nodes.len() - 1;
    Rule { name: name.to_string(), inputs, nodes, output }
}

/// Fixed candidates, each offered while it is not the current explanation.
struct Fixed(Vec<Proposal>);

impl Proposer for Fixed {
    fn name(&self) -> &str {
        "fixed"
    }

    fn propose(&self, context: &Context<'_>) -> Result<Vec<Proposal>, String> {
        Ok(self.0.iter().filter(|p| p.candidate != *context.current).cloned().collect())
    }
}

fn proposal(description: &str, candidate: Artifact) -> Proposal {
    Proposal { source: "test".to_string(), description: description.to_string(), candidate }
}

const BUDGET: Budget = Budget { certifications: 64, rounds: 16 };

/// The MLP model: `h = A x`, `w = W_out relu(W_in h + b_in)` (units 2 and 3 duplicate units 0
/// and 1), `y = h + w`, logits `s U y`. Nodes: 0 x, 1 h, 2 pre, 3 act, 4 w (the write), 5 y, 6 logits.
fn mlp(readout_scale: f64) -> OperatorProgram {
    let (m, units, classes) = (native(2), Interface::uniform(4, 1, LabelKind::Unit, 0).expect("units"), native(3));
    let operators = vec![
        dense("A", &m, &m, array![[1.0, 0.5], [-0.25, 1.0]]),
        dense("W_in", &units, &m, array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        column("b_in", &units, &[-0.5, -0.5, -0.5, -0.5]),
        dense("W_out", &m, &units, array![[0.5, -0.25, 0.25, 0.5], [0.25, 0.5, 0.25, -0.5]]),
        Operator::identity("I", m.clone()),
        dense("U", &classes, &m, array![[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]] * readout_scale),
    ];
    raw_program(
        2,
        operators,
        vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Affine { terms: vec![(1, 1)], bias: Some(2) },
            Node::Pointwise { input: 2, laws: vec![Law::Relu; 4] },
            Node::Affine { terms: vec![(3, 3)], bias: None },
            Node::Affine { terms: vec![(1, 4), (4, 4)], bias: None },
            Node::Affine { terms: vec![(5, 5)], bias: None },
        ],
    )
}

fn mlp_family() -> FamilyInputs {
    grid(&[-2.0, -1.0, -0.25, 0.5, 1.5], 2)
}

#[test]
fn local_cuda_rejects_host_instead_of_silently_falling_back() {
    let model = mlp(1.0);
    let local = Local::new(&model, mlp_family(), None, 64);
    assert!(local.backend_name().starts_with("CPU"));
    let result = local.with_cuda(gam_gpu::tensor::Device::host(), 1024);
    assert!(result.err().expect("host must not satisfy required CUDA").contains("float64 accelerator"));
}

/// The MLP with its duplicate units merged: exact, ten literals instead of twenty.
fn merged(model: &OperatorProgram) -> Artifact {
    let start = Artifact::native(model).expect("an artifact");
    let (m, units) = (native(2), Interface::uniform(2, 1, LabelKind::Unit, 0).expect("units"));
    let k = model.operators.len();
    let body = rule(
        "merged units",
        vec![m.clone()],
        vec![
            Node::Param { index: 0 },
            Node::Affine { terms: vec![(0, k)], bias: Some(k + 1) },
            Node::Pointwise { input: 1, laws: vec![Law::Relu; 2] },
            Node::Affine { terms: vec![(2, k + 2)], bias: None },
        ],
    );
    let operators = vec![
        dense("W_in merged", &units, &m, array![[1.0, 0.0], [0.0, 1.0]]),
        column("b_in merged", &units, &[-0.5, -0.5]),
        dense("W_out merged", &m, &units, array![[0.75, 0.25], [0.5, 0.0]]),
    ];
    start.replace_block("mlp", Callee::New(body), vec![Argument::Native(1)], 4, operators).expect("a replacement")
}

/// A one-term rule `Param → A_k Param` writing interface `rows` from `cols`.
fn linear(start: &Artifact, name: &str, read: usize, write: usize, op: Operator) -> Artifact {
    let k = start.program.operators.len();
    let cols = op.cols.clone();
    let body = rule(name, vec![cols], vec![Node::Param { index: 0 }, Node::Affine { terms: vec![(0, k)], bias: None }]);
    start.replace_block(name, Callee::New(body), vec![Argument::Native(read)], write, vec![op]).expect("a replacement")
}

/// (a) A pure two-input product is reachable although neither input alone predicts anything: the
/// native block writes `(a b) e` through an opaque outer product of all its inputs; the
/// single-input rules are cheaper and tried first, and refused; the product rule is accepted.
#[test]
fn a_pure_product_is_reachable_without_single_feature_gain() {
    let (x, m, classes, one) = (native(4), native(2), native(3), native(1));
    let pairs = Interface::uniform(1, 16, LabelKind::Pair, 0).expect("the outer product's interface");
    let mut w = Array2::<f64>::zeros((2, 16));
    // Outer product coordinate 0·4 + 1 is a·b, and 1·4 + 0 is b·a.
    w[[0, 1]] = 0.5;
    w[[0, 4]] = 0.5;
    w[[1, 1]] = 0.25;
    w[[1, 4]] = 0.25;
    let model = raw_program(
        4,
        vec![
            Operator::identity("I4", x.clone()),
            dense("W", &m, &pairs, w),
            dense("lift", &m, &x, array![[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            Operator::identity("I2", m.clone()),
            dense("U", &classes, &m, array![[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]),
        ],
        vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Outer { left: 1, right: 1 },
            Node::Affine { terms: vec![(2, 1)], bias: None },
            Node::Affine { terms: vec![(1, 2), (3, 3)], bias: None },
            Node::Affine { terms: vec![(4, 4)], bias: None },
        ],
    );
    let mut rows = Vec::new();
    for a in [-1.0, 1.0] {
        for b in [-1.0, 1.0] {
            for c in [-0.5, 0.5] {
                for d in [-0.5, 0.5] {
                    rows.push(vec![a, b, c, d]);
                }
            }
        }
    }
    let family = raw_family(rows);
    let start = Artifact::native(&model).expect("an artifact");
    // Least squares on one input: the product has no linear part in either, so the best single
    // rule writes nothing.
    let single = |name: &str| linear(&start, name, 1, 3, dense(name, &m, &x, Array2::zeros((2, 4))));
    let k = model.operators.len();
    let product = rule(
        "product",
        vec![x.clone()],
        vec![
            Node::Param { index: 0 },
            Node::Affine { terms: vec![(0, k)], bias: None },
            Node::Affine { terms: vec![(0, k + 1)], bias: None },
            Node::Hadamard { left: 1, right: 2 },
            Node::Affine { terms: vec![(3, k + 2)], bias: None },
        ],
    );
    let product = start
        .replace_block(
            "product",
            Callee::New(product),
            vec![Argument::Native(1)],
            3,
            vec![
                dense("read a", &one, &x, array![[1.0, 0.0, 0.0, 0.0]]),
                dense("read b", &one, &x, array![[0.0, 1.0, 0.0, 0.0]]),
                dense("write e", &m, &one, array![[1.0], [0.5]]),
            ],
        )
        .expect("a replacement");
    let proposer = Fixed(vec![proposal("a alone", single("a alone")), proposal("b alone", single("b alone")), proposal("a b", product)]);
    let local = Local::new(&model, family.clone(), None, 64);
    let run = clean_run(&model, &family);
    let constraint = Constraint { local: 1e-9, run: 1e-9 };
    let searched = search(&local, &run, &[&proposer], &start, constraint, BUDGET).expect("a search");
    assert_eq!(searched.artifact.blocks.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(), ["product"]);
    let tried: Vec<(&str, &Outcome)> = searched.steps.iter().map(|s| (s.description.as_str(), &s.outcome)).collect();
    assert!(matches!(tried[0].1, Outcome::LocalScreen(_)) && matches!(tried[1].1, Outcome::LocalScreen(_)), "{tried:?}");
    assert_eq!(*tried[2].1, Outcome::Accepted);
    assert!(searched.steps[0].saving > searched.steps[2].saving, "the single rules save more bits and are tried first");
}

/// (b) A nonzero prediction where the native amplitude is zero executes as predicted: every native
/// unit is off at `x = (−1, −1)`, so the native write is zero there, and the decoded linear rule
/// writes `L h` there, which `D_local` charges.
#[test]
fn a_nonzero_prediction_at_a_native_zero_executes() {
    let model = mlp(1.0);
    let family = raw_family(vec![vec![-1.0, -1.0], vec![1.0, 0.5], vec![1.5, 1.5]]);
    let start = Artifact::native(&model).expect("an artifact");
    let m = native(2);
    let candidate = linear(&start, "linear", 1, 4, dense("L", &m, &m, array![[0.5, 0.0], [0.0, 0.5]]));
    let decoded = Artifact::from_bytes(&candidate.to_bytes().expect("bytes"), &model.declarations).expect("decoded");
    let native_trace = model.execute(&family, false).expect("native");
    assert!(native_trace.values[4].row(0).iter().all(|v| *v == 0.0), "every native unit is off at row 0");
    let h = native_trace.values[1].row(0).to_owned();
    let trace = decoded.execute(&family).expect("execution");
    let block = &decoded.blocks[0];
    let written = trace.values[block.write].row(0).to_owned();
    assert_eq!(written, h.mapv(|v| 0.5 * v));
    assert!(written.iter().any(|v| *v != 0.0));
    let local = Local::new(&model, family.clone(), None, 64);
    let measured = local.measure(&decoded).expect("a measure");
    let scale = local.scale(4).expect("a scale");
    let at_zero = written.iter().map(|v| v * v).sum::<f64>().sqrt() / scale;
    assert!(measured.blocks[0].worst >= at_zero);
}

/// (c) Writer errors are priced by the joint native output error: two writers into one output
/// with equal and opposite errors write the native output exactly and pass; the same errors
/// aligned double and fail; each writer's own error is the same in both.
#[test]
fn writer_errors_are_priced_jointly() {
    let (m, classes) = (native(2), native(3));
    let (w1, w2, e) = (array![[1.0, 0.5], [0.0, 1.0]], array![[0.5, 0.0], [0.25, 0.5]], array![[0.25, 0.0], [0.0, 0.25]]);
    let model = raw_program(
        2,
        vec![
            Operator::identity("I", m.clone()),
            dense("W1", &m, &m, w1.clone()),
            dense("W2", &m, &m, w2.clone()),
            dense("U", &classes, &m, array![[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]),
        ],
        vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Affine { terms: vec![(1, 1), (1, 2)], bias: None },
            Node::Affine { terms: vec![(1, 0), (2, 0)], bias: None },
            Node::Affine { terms: vec![(3, 3)], bias: None },
        ],
    );
    let family = grid(&[-1.0, -0.5, 0.5, 1.0], 2);
    let start = Artifact::native(&model).expect("an artifact");
    let writers = |name: &str, first: Array2<f64>, second: Array2<f64>| {
        let k = model.operators.len();
        let body = rule(name, vec![m.clone()], vec![Node::Param { index: 0 }, Node::Affine { terms: vec![(0, k), (0, k + 1)], bias: None }]);
        start
            .replace_block(name, Callee::New(body), vec![Argument::Native(1)], 2, vec![dense("W1'", &m, &m, first), dense("W2'", &m, &m, second)])
            .expect("a replacement")
    };
    let opposing = writers("opposing", &w1 + &e, &w2 - &e);
    let aligned = writers("aligned", &w1 + &e, &w2 + &e);
    let local = Local::new(&model, family.clone(), None, 64);
    let run = clean_run(&model, &family);
    let constraint = Constraint { local: 0.01, run: 1.0 };
    let mut cache = CostCache::default();
    let opposing = assess(&local, &run, &opposing, constraint, &mut cache).expect("an assessment");
    let aligned = assess(&local, &run, &aligned, constraint, &mut cache).expect("an assessment");
    assert_eq!(opposing.local.verdict(), FidelityVerdict::Meets);
    assert_eq!(opposing.disagreements().0, 0.0);
    assert_eq!(aligned.local.verdict(), FidelityVerdict::Violates);
    // Each writer's own error, ‖E h‖, is the same in both candidates; only the joint output differs.
    let h = model.execute(&family, false).expect("native").values[1].clone();
    let own = h.dot(&e.t());
    let scale = local.scale(2).expect("a scale");
    let worst_own = own.outer_iter().map(|r| r.iter().map(|v| v * v).sum::<f64>().sqrt()).fold(0.0, f64::max) / scale;
    assert!((aligned.disagreements().0 - 2.0 * worst_own).abs() < 1e-12);
}

/// (d) Two states equal under the final-norm logit lens but different later fail local fidelity:
/// a block that writes twice the native write reads the same through the final RMS norm, while the
/// later linear read of the stream sees the difference.
#[test]
fn a_logit_lens_equal_state_fails_local_fidelity() {
    let (m, classes) = (native(2), native(3));
    let model = raw_program(
        2,
        vec![
            dense("A", &m, &m, array![[1.0, 0.5], [-0.25, 1.0]]),
            dense("B", &m, &m, array![[0.5, 0.25], [0.0, 1.0]]),
            dense("U", &classes, &m, array![[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]),
            dense("V", &classes, &m, array![[0.25, 0.0], [0.0, -0.25], [0.5, 0.5]]),
            Operator::identity("I", classes.clone()),
        ],
        vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Affine { terms: vec![(1, 1)], bias: None },
            Node::RmsNorm { input: 2, epsilon: 2f64.powi(-60) },
            Node::Affine { terms: vec![(3, 2)], bias: None },
            Node::Affine { terms: vec![(2, 3)], bias: None },
            Node::Affine { terms: vec![(4, 4), (5, 4)], bias: None },
        ],
    );
    let family = grid(&[-1.0, -0.5, 0.5, 1.0], 2);
    let start = Artifact::native(&model).expect("an artifact");
    let doubled = linear(&start, "doubled", 1, 2, dense("2B", &m, &m, array![[1.0, 0.5], [0.0, 2.0]]));
    // Under the lens (final norm, then U) the doubled write reads as the native one.
    let lens = |w: &Array2<f64>| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((w.nrows(), 3));
        let u = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]];
        for (r, row) in w.outer_iter().enumerate() {
            let norm = (row.iter().map(|v| v * v).sum::<f64>() / 2.0 + 2f64.powi(-60)).sqrt();
            out.row_mut(r).assign(&u.dot(&row.mapv(|v| v / norm)));
        }
        out
    };
    let native_write = model.execute(&family, false).expect("native").values[2].clone();
    let doubled_write = doubled.execute(&family).expect("execution").values[doubled.blocks[0].write].clone();
    let (a, b) = (lens(&native_write), lens(&doubled_write));
    let lens_kl = (0..a.nrows()).map(|r| kl_logits(a.row(r), b.row(r)).0).fold(0.0, f64::max);
    assert!(lens_kl < 1e-12, "the logit lens cannot tell the states apart: {lens_kl}");
    let local = Local::new(&model, family.clone(), None, 64);
    let run = clean_run(&model, &family);
    let assessment = assess(&local, &run, &doubled, Constraint { local: 0.5, run: 10.0 }, &mut CostCache::default()).expect("an assessment");
    assert_eq!(assessment.local.verdict(), FidelityVerdict::Violates);
    // The error is the native write itself: its worst row over the root mean square row.
    let norms: Vec<f64> = native_write.outer_iter().map(|r| r.iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
    let rms = (norms.iter().map(|n| n * n).sum::<f64>() / norms.len() as f64).sqrt();
    let worst = norms.iter().copied().fold(0.0, f64::max) / rms;
    assert!((assessment.disagreements().0 - worst).abs() < 1e-12, "{} against {worst}", assessment.disagreements().0);
}

/// (e) Shared parameterized rules whose instances differ in gain and spectrum stay candidates and
/// are accepted: two blocks `A diag(s₁)` and `A diag(s₂)` as one stored body `A (s ⊙ x)` called
/// with each instance's gains, `16 + 4 + 4` literals instead of `16 + 16`.
#[test]
fn shared_parameterized_rules_with_different_spectra_are_accepted() {
    let (m, classes) = (native(4), native(3));
    let a = array![[1.0, 0.5, 0.0, -0.25], [0.0, 1.0, 0.5, 0.0], [0.25, 0.0, 1.0, 0.5], [-0.5, 0.0, 0.0, 1.0]];
    let (s1, s2) = (array![1.0, 2.0, 0.5, 1.0], array![2.0, 1.0, 1.0, 0.5]);
    let scaled = |s: &Array1<f64>| &a * &s.view().insert_axis(ndarray::Axis(0));
    let (m1, m2) = (scaled(&s1), scaled(&s2));
    let frobenius = |x: &Array2<f64>| x.iter().map(|v| v * v).sum::<f64>();
    assert!(frobenius(&m1) != frobenius(&m2), "the instances' spectra differ");
    let model = raw_program(
        4,
        vec![
            Operator::identity("I", m.clone()),
            dense("M1", &m, &m, m1),
            dense("M2", &m, &m, m2),
            dense("U", &classes, &m, array![[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5], [-1.0, 1.0, 0.25, 0.25]]),
        ],
        vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Affine { terms: vec![(1, 1)], bias: None },
            Node::Affine { terms: vec![(1, 2)], bias: None },
            Node::Affine { terms: vec![(1, 0), (2, 0), (3, 0)], bias: None },
            Node::Affine { terms: vec![(4, 3)], bias: None },
        ],
    );
    let family = grid(&[-1.0, 0.5, 1.0], 4);
    let start = Artifact::native(&model).expect("an artifact");
    let k = model.operators.len();
    let body = rule(
        "gained A",
        vec![m.clone(), m.clone()],
        vec![
            Node::Param { index: 0 },
            Node::Param { index: 1 },
            Node::Hadamard { left: 0, right: 1 },
            Node::Affine { terms: vec![(2, k)], bias: None },
        ],
    );
    let gains = |name: &str, s: &Array1<f64>| Argument::Constant(column(name, &m, s.as_slice().expect("contiguous")));
    let shared = start
        .replace_block("first", Callee::New(body), vec![Argument::Native(1), gains("s1", &s1)], 2, vec![dense("A", &m, &m, a.clone())])
        .expect("a replacement");
    let rule_index = shared.program.rules.len() - 1;
    let shared = shared
        .replace_block("second", Callee::Existing(rule_index), vec![Argument::Native(1), gains("s2", &s2)], 3, Vec::new())
        .expect("a replacement");
    let local = Local::new(&model, family.clone(), None, 128);
    let run = clean_run(&model, &family);
    let proposer = Fixed(vec![proposal("shared", shared)]);
    let searched = search(&local, &run, &[&proposer], &start, Constraint { local: 1e-12, run: 1e-12 }, BUDGET).expect("a search");
    assert_eq!(searched.steps.len(), 1);
    assert_eq!(searched.steps[0].outcome, Outcome::Accepted);
    let calls: Vec<usize> = searched
        .artifact
        .program
        .nodes
        .iter()
        .filter_map(|n| match n {
            Node::Call { rule, .. } => Some(*rule),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0], calls[1], "both instances call the one stored body");
    let native_cost = structural_cost(&start, &mut CostCache::default()).expect("a cost");
    assert_eq!(native_cost.literals - searched.assessment.cost.literals, 8);
}

/// (f) Quantizing an opaque program is not a discovery: rounding the MLP's write onto a coarse
/// lattice leaves `C` unchanged, so the search never tries it, although the code-plus-KL baseline,
/// which charges lattice indices, prefers it.
#[test]
fn quantizing_is_not_a_discovery() {
    let model = mlp(2f64.powi(-12));
    let family = mlp_family();
    let start = Artifact::native(&model).expect("an artifact");
    let mut quantized = start.clone();
    let w_out = quantized.program.operators[3].clone();
    let coarse = w_out.matrix().mapv(|v| (v * 2.0).round() / 2.0);
    quantized.program.operators[3] = Arc::new(dense("W_out", &w_out.rows, &w_out.cols, coarse));
    let mut cache = CostCache::default();
    let (before, after) = (structural_cost(&start, &mut cache).expect("a cost"), structural_cost(&quantized, &mut cache).expect("a cost"));
    assert_eq!(before, after);
    let baseline = |artifact: &Artifact| code_plus_kl(&model, artifact, &family, 1, 1).expect("a baseline");
    assert!(baseline(&quantized) < baseline(&start), "the baseline counts the coarser lattice as a saving");
    let local = Local::new(&model, family.clone(), None, 64);
    let run = clean_run(&model, &family);
    let proposer = Fixed(vec![proposal("quantized", quantized)]);
    let searched = search(&local, &run, &[&proposer], &start, Constraint { local: 1.0, run: 1.0 }, BUDGET).expect("a search");
    assert!(searched.steps.is_empty(), "a candidate that saves nothing is never tried");
    assert!(searched.artifact.blocks.is_empty());
}

/// (g) A high-rank, always-used rotation is not penalized for always running: the native block's
/// dense rotation (every block present) is accepted as the rule of its two plane rotations, each an
/// instance active on every input; the activity listing is reported, not charged.
#[test]
fn an_always_running_rotation_is_not_penalized() {
    let (x, classes) = (native(4), native(3));
    let planes = Interface::uniform(2, 2, LabelKind::Plane, 0).expect("planes");
    let (c1, s1, c2, s2) = (0.6f32 as f64, 0.8f32 as f64, 0.28f32 as f64, -0.96f32 as f64);
    let rotation = array![[c1, -s1, 0.0, 0.0], [s1, c1, 0.0, 0.0], [0.0, 0.0, c2, -s2], [0.0, 0.0, s2, c2]];
    let model = raw_program(
        4,
        vec![
            dense("lift", &planes, &x, Array2::eye(4)),
            dense("R", &planes, &planes, rotation.clone()),
            Operator::identity("I", planes.clone()),
            dense("U", &classes, &planes, array![[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5], [-1.0, 1.0, 0.25, 0.25]]),
        ],
        vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Affine { terms: vec![(1, 1)], bias: None },
            Node::Affine { terms: vec![(1, 2), (2, 2)], bias: None },
            Node::Affine { terms: vec![(3, 3)], bias: None },
        ],
    );
    let family = grid(&[-1.0, -0.25, 0.5, 1.0], 4);
    let start = Artifact::native(&model).expect("an artifact");
    let k = model.operators.len();
    let precision = exact_precision(rotation.iter().copied()).expect("a lattice");
    let sparse = Operator::blocks(
        "plane rotations",
        planes.clone(),
        planes.clone(),
        rotation,
        array![[true, false], [false, true]],
        precision,
        Provenance::default(),
    )
    .expect("a block-diagonal operator");
    let body = rule(
        "rotate planes",
        vec![planes.clone()],
        vec![
            Node::Param { index: 0 },
            Node::Affine { terms: vec![(0, k)], bias: None },
            Node::Pointwise { input: 1, laws: vec![Law::Identity, Law::Identity] },
        ],
    );
    let rotated = start.replace_block("rotation", Callee::New(body), vec![Argument::Native(1)], 2, vec![sparse]).expect("a replacement");
    let local = Local::new(&model, family.clone(), None, 256);
    let run = clean_run(&model, &family);
    let proposer = Fixed(vec![proposal("plane rotations", rotated)]);
    let searched = search(&local, &run, &[&proposer], &start, Constraint { local: 1e-12, run: 1e-12 }, BUDGET).expect("a search");
    assert_eq!(searched.steps[0].outcome, Outcome::Accepted);
    let listing = execution_cost(&searched.artifact, &family, 256).expect("an execution cost");
    assert_eq!(listing.instances, 2);
    assert_eq!(listing.mean_active, 2.0, "both plane instances run on every input");
    assert!(listing.bits > 0.0);
    // `C` takes no input: the listing is nowhere in it.
    let cost = structural_cost(&searched.artifact, &mut CostCache::default()).expect("a cost");
    assert_eq!(cost, searched.assessment.cost);
    assert_eq!(cost.total(), cost.structure_bits + 32 * cost.literals + cost.binding_bits);
}

/// (h) A smaller program that violates the local tolerance is refused even though code-plus-KL
/// prefers it: deleting the MLP saves twenty literals, and with a readout nearly blind to it the
/// baseline's data bits barely rise.
#[test]
fn a_smaller_violating_program_is_refused() {
    let model = mlp(2f64.powi(-12));
    let family = mlp_family();
    let start = Artifact::native(&model).expect("an artifact");
    let m = native(2);
    let nothing = Operator::blocks(
        "nothing",
        m.clone(),
        m.clone(),
        Array2::zeros((2, 2)),
        array![[false]],
        exact_precision([0.0]).expect("a lattice"),
        Provenance::default(),
    )
    .expect("an empty operator");
    let deleted = linear(&start, "deleted", 1, 4, nothing);
    let baseline = |artifact: &Artifact| code_plus_kl(&model, artifact, &family, 1, 1).expect("a baseline");
    assert!(baseline(&deleted) < baseline(&start), "the baseline prefers the deletion");
    let local = Local::new(&model, family.clone(), None, 64);
    let run = clean_run(&model, &family);
    let proposer = Fixed(vec![proposal("delete the MLP", deleted)]);
    let searched = search(&local, &run, &[&proposer], &start, Constraint { local: 0.1, run: 1.0 }, BUDGET).expect("a search");
    assert!(matches!(searched.steps[0].outcome, Outcome::LocalScreen(_)), "{:?}", searched.steps);
    assert!(searched.artifact.blocks.is_empty());
}

/// (i) The accepted artifact runs and intervenes after serialization: the search accepts the
/// merged MLP; its bytes alone, with the declarations, decode to an artifact that executes and
/// answers a native intervention exactly as before, holding none of the replaced native weights.
#[test]
fn the_accepted_artifact_runs_and_intervenes_after_serialization() {
    let model = mlp(1.0);
    let family = mlp_family();
    let start = Artifact::native(&model).expect("an artifact");
    let local = Local::new(&model, family.clone(), None, 64);
    let run = FamilyRun {
        model: &model,
        family: family.clone(),
        readouts: 1,
        episodes: vec![
            Episode { id: "clean".to_string(), group: "clean".to_string(), edits: Vec::new() },
            Episode {
                id: "scale h0".to_string(),
                group: "scale".to_string(),
                edits: vec![Edit { node: 1, rows: None, columns: 0..1, change: Change::Scale(2.0) }],
            },
        ],
    };
    let proposer = Fixed(vec![proposal("merge duplicate units", merged(&model))]);
    let searched = search(&local, &run, &[&proposer], &start, Constraint { local: 1e-12, run: 1e-12 }, BUDGET).expect("a search");
    assert_eq!(searched.steps[0].outcome, Outcome::Accepted);
    let accepted = searched.artifact;
    let bytes = accepted.to_bytes().expect("bytes");
    let declarations = accepted.program.declarations.clone();
    let expected_trace = accepted.execute(&family).expect("execution");
    let expected_scores = run.episodes(&accepted).expect("scores");
    let (blocks, places) = (accepted.blocks.clone(), accepted.places.clone());
    drop(accepted);
    let decoded = Artifact::from_bytes(&bytes, &declarations).expect("decoded");
    assert_eq!(decoded.blocks, blocks);
    assert_eq!(decoded.places, places);
    // The native units (the only width-4 interface) left with the block they computed.
    assert!(decoded.program.operators.iter().all(|op| op.rows.width() != 4 && op.cols.width() != 4));
    let trace = decoded.execute(&family).expect("execution");
    assert_eq!(trace.values[decoded.program.output], expected_trace.values[decoded.program.output]);
    let scores = run.episodes(&decoded).expect("scores");
    assert_eq!(scores, expected_scores);
    let scaled = scores.iter().find(|e| e.id == "scale h0").expect("the intervention");
    assert!(scaled.native_effect > 0.0 && scaled.kl < 1e-20, "{scaled:?}");
    assert_eq!(scaled.unheld, 0);
}

/// The artifact's message decodes to the same blocks, places and exceptions, and to a program that
/// executes identically.
#[test]
fn an_artifact_round_trips() {
    let model = mlp(1.0);
    let family = mlp_family();
    let mut artifact = merged(&model);
    artifact.exceptions.push(super::artifact::Exception { context: Vec::new(), node: artifact.blocks[0].write, column: 1, value: 0.125 });
    let bytes = artifact.to_bytes().expect("bytes");
    let decoded = Artifact::from_bytes(&bytes, &model.declarations).expect("decoded");
    assert_eq!(decoded.blocks, artifact.blocks);
    assert_eq!(decoded.places, artifact.places);
    assert_eq!(decoded.exceptions, artifact.exceptions);
    let (a, b) = (artifact.execute(&family).expect("execution"), decoded.execute(&family).expect("execution"));
    assert_eq!(a.values[artifact.program.output], b.values[decoded.program.output]);
    assert_eq!(decoded.encode().expect("a message"), artifact.encode().expect("a message"));
}

/// A native intervention inside a replaced block has no place in the explanation: it runs clean
/// there and so predicts no effect, which `D_run` charges.
#[test]
fn an_unheld_place_predicts_no_effect() {
    let model = mlp(1.0);
    let family = mlp_family();
    let artifact = merged(&model);
    assert!(artifact.place(3).is_none(), "the native units are inside the replaced block");
    let run = FamilyRun {
        model: &model,
        family: family.clone(),
        readouts: 1,
        episodes: vec![Episode {
            id: "unit 0 off".to_string(),
            group: "unit".to_string(),
            edits: vec![Edit { node: 3, rows: None, columns: 0..1, change: Change::Scale(0.0) }],
        }],
    };
    let scores = run.episodes(&artifact).expect("scores");
    assert_eq!(scores[0].unheld, 1);
    assert!((scores[0].kl - scores[0].native_effect).abs() < 1e-12, "{:?}", scores[0]);
}

/// The counterexample ascent tests inputs beyond the declared family: a linear rule for a block
/// that adds `gelu(x₀ − 2) e` is close on the family (`x₀ ≤ 1`) and far at the box's edge, which
/// the ascent reaches from the family's rows.
#[test]
fn the_counterexample_ascent_finds_a_worse_input() {
    let (m, classes, one) = (native(2), native(3), native(1));
    let model = raw_program(
        2,
        vec![
            Operator::identity("I", m.clone()),
            dense("A", &m, &m, array![[1.0, 0.5], [-0.25, 1.0]]),
            dense("first", &one, &m, array![[1.0, 0.0]]),
            column("minus two", &one, &[-2.0]),
            dense("e", &m, &one, array![[1.0], [-0.5]]),
            dense("U", &classes, &m, array![[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]),
        ],
        vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Affine { terms: vec![(1, 2)], bias: Some(3) },
            Node::Pointwise { input: 2, laws: vec![Law::GeluTanh] },
            Node::Affine { terms: vec![(1, 1), (3, 4)], bias: None },
            Node::Affine { terms: vec![(1, 0), (4, 0)], bias: None },
            Node::Affine { terms: vec![(5, 5)], bias: None },
        ],
    );
    let family = grid(&[-1.0, -0.5, 0.0, 0.5, 1.0], 2);
    let start = Artifact::native(&model).expect("an artifact");
    let candidate = linear(&start, "linear part", 1, 4, dense("A'", &m, &m, array![[1.0, 0.5], [-0.25, 1.0]]));
    let declared = Local::new(&model, family.clone(), None, 64).measure(&candidate).expect("a measure");
    let domain = vec![super::acceptance::SlotDomain::Box { lower: Array1::from_elem(2, -3.0), upper: Array1::from_elem(2, 3.0) }];
    let ascent = super::acceptance::Ascent { domain, pool: Vec::new(), evaluations: 16 };
    let searched = Local::new(&model, family.clone(), Some(ascent), 64).measure(&candidate).expect("a measure");
    assert!(searched.counterexamples > 0, "{searched:?}");
    assert!(searched.blocks[0].worst > 2.0 * declared.blocks[0].worst, "{} against {}", searched.blocks[0].worst, declared.blocks[0].worst);
    assert_eq!(searched.blocks[0].scale, declared.blocks[0].scale, "the scale is the declared family's");
}

#[test]
fn exception_on_replaced_write_cannot_escape_local_tolerance() {
    let model = mlp(0.0); // Readout cannot see any residual error.
    let family = mlp_family();
    let local = Local::new(&model, family.clone(), None, 32);
    let run = clean_run(&model, &family);
    let mut candidate = merged(&model);
    let write = candidate.blocks[0].write;
    candidate.exceptions.push(super::artifact::Exception {
        context: vec![], // Raw-only family has empty token context at every row.
        node: write,
        column: 0,
        value: 1000.0,
    });
    let decoded = Artifact::from_bytes(&candidate.to_bytes().unwrap(), &model.declarations).unwrap();
    let native_trace = model.execute(&family, false).unwrap();
    let actual = decoded.execute(&family).unwrap();
    assert!((actual.values[write][[0, 0]] - native_trace.values[4][[0, 0]]).abs() > 999.0);
    let assessment = assess(&local, &run, &candidate, Constraint { local: 0.01, run: 0.01 }, &mut CostCache::default()).expect("score the exception");
    assert_eq!(assessment.local.verdict(), FidelityVerdict::Violates);
    assert_eq!(assessment.run.verdict(), FidelityVerdict::Meets);
}

/// A derived operator: an output map that is the copy law of its value map,
/// `O = λ diag(g / g_f) V⁺`, is computed by the decoder from `V` and the two gains instead of sent,
/// so the artifact saves its literals but for `λ`; its message holds no reals for it, and the
/// decoded artifact computes the same map and is accepted.
#[test]
fn a_derived_operator_is_computed_not_sent() {
    let (d, w, classes) = (native(4), native(2), native(3));
    let value = array![[1.0, 0.5, 0.0, -0.25], [0.0, 1.0, 0.5, 0.25]];
    let (gain, final_gain) = (array![1.0, 2.0, 0.5, 1.0], array![2.0, 1.0, 1.0, 0.5]);
    let lambda = 0.75_f32;
    let output = super::rules::copy_prediction(&value, &gain, &final_gain).expect("a copy").mapv(|v| f64::from((v * f64::from(lambda)) as f32));
    let diag = |name: &str, v: &Array1<f64>| {
        let precision = exact_precision(v.iter().copied()).expect("a lattice");
        Operator::diag(name, d.clone(), v.clone(), precision, Provenance::default()).expect("a diagonal")
    };
    let model = raw_program(
        4,
        vec![
            diag("g", &gain),
            dense("V", &w, &d, value),
            dense("O", &d, &w, output),
            Operator::identity("I", d.clone()),
            diag("g_f", &final_gain),
            dense("U", &classes, &d, array![[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5], [-1.0, 1.0, 0.25, 0.25]]),
        ],
        vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Affine { terms: vec![(1, 1)], bias: None },
            Node::Affine { terms: vec![(2, 2)], bias: None },
            Node::Affine { terms: vec![(0, 3), (3, 3)], bias: None },
            Node::Affine { terms: vec![(4, 4)], bias: None },
            Node::Affine { terms: vec![(5, 5)], bias: None },
        ],
    );
    let family = grid(&[-1.0, 0.5, 1.0], 4);
    let start = Artifact::native(&model).expect("an artifact");
    let derived = start
        .derive(2, super::artifact::OperatorLaw::Copy { value: 1, gain: 0, final_gain: 4 }, lambda, Vec::new())
        .and_then(|a| a.bind("copy head", &[1], 3))
        .expect("a derivation");
    let mut cache = CostCache::default();
    let (before, after) = (structural_cost(&start, &mut cache).expect("a cost"), structural_cost(&derived, &mut cache).expect("a cost"));
    assert_eq!(before.literals - after.literals, 8 - 1, "O's eight reals are computed; λ is sent");
    assert!(after.total() < before.total());
    let decoded = Artifact::from_bytes(&derived.to_bytes().expect("bytes"), &model.declarations).expect("decoded");
    assert_eq!(decoded.derived, derived.derived);
    assert_eq!(decoded.program.operators[2].matrix(), derived.program.operators[2].matrix());
    assert_eq!(derived.message_program().expect("a message program").operators[2].real_count(), 0);
    let local = Local::new(&model, family.clone(), None, 64);
    let run = clean_run(&model, &family);
    let proposer = Fixed(vec![proposal("copy head", derived)]);
    let searched = search(&local, &run, &[&proposer], &start, Constraint { local: 1e-6, run: 1e-9 }, BUDGET).expect("a search");
    assert_eq!(searched.steps[0].outcome, Outcome::Accepted, "{:?}", searched.steps);
    assert_eq!(searched.artifact.derived.len(), 1);
}

// Append to acceptance_tests.rs. Uses its existing mlp/merged helpers.
#[test]
fn clearing_bindings_cannot_erase_local_disagreement() {
    let model = mlp(0.0);
    let family = mlp_family();
    let local = Local::new(&model, family.clone(), None, 32);
    let run = clean_run(&model, &family);
    let mut candidate = merged(&model);
    candidate.blocks.clear();
    // Corrupt an operator used only by the replacement; zero readout hides it.
    let write = candidate.place(4).unwrap();
    let rule = match candidate.program.nodes[write] {
        Node::Call { rule, .. } => rule,
        _ => panic!("replacement call"),
    };
    let op = match candidate.program.rules[rule].nodes.last().unwrap() {
        Node::Affine { terms, .. } => terms[0].1,
        _ => panic!("replacement output"),
    };
    let old = &candidate.program.operators[op];
    candidate.program.operators[op] =
        Arc::new(dense("corrupt", &old.rows, &old.cols, Array2::from_elem((old.rows.width(), old.cols.width()), 1000.0)));
    let result = assess(&local, &run, &candidate, Constraint { local: 0.01, run: 0.01 }, &mut CostCache::default());
    assert!(result.as_ref().map_or(true, |assessment| !assessment.meets()), "missing blocks cannot certify D_local=0");
}

#[test]
fn existing_block_cannot_cover_an_unbound_upstream_change() {
    let model = mlp(0.0);
    let mut candidate = merged(&model);
    assert!(candidate.validate_coverage(&model).is_ok(), "whole MLP replacement remains valid");
    let upstream = candidate.place(1).unwrap();
    let op = match &candidate.program.nodes[upstream] {
        Node::Affine { terms, .. } => terms[0].1,
        _ => panic!("upstream affine"),
    };
    let old = &candidate.program.operators[op];
    candidate.program.operators[op] = Arc::new(dense("changed upstream", &old.rows, &old.cols, array![[2.0, 1.0], [-0.5, 2.0]]));
    assert!(candidate.validate_coverage(&model).is_err());
    let bound = candidate.bind("upstream", &[0], 1).unwrap();
    assert!(bound.validate_coverage(&model).is_ok(), "declaring both changed blocks is valid");
}

#[test]
fn output_exception_requires_its_own_measured_boundary() {
    let model = mlp(0.0);
    let mut candidate = Artifact::native(&model).unwrap();
    candidate.exceptions.push(super::artifact::Exception { context: vec![], node: model.output, column: 0, value: 1.0 });
    assert!(candidate.validate_coverage(&model).is_err());
    assert!(candidate.bind("readout exception", &[5], 6).unwrap().validate_coverage(&model).is_ok());
}

// Append to acceptance_tests.rs.
#[test]
fn gain_literals_round_recursively_in_nodes_and_shared_rules() {
    use super::operator_program::Coefficient;
    let model = raw_program(
        2,
        vec![],
        vec![
            Node::Raw { slot: 0 },
            Node::Gain {
                input: 0,
                coefficient: Coefficient::Number(1.0),
            },
        ],
    );
    let mut candidate = Artifact::native(&model)
        .unwrap()
        .replace_block(
            "gain",
            Callee::New(rule(
                "gain",
                vec![native(2)],
                vec![
                    Node::Param { index: 0 },
                    Node::Gain {
                        input: 0,
                        coefficient: Coefficient::Product(vec![
                            Coefficient::Number(0.1),
                            Coefficient::Sum(vec![
                                Coefficient::Number(0.3),
                                Coefficient::Number(0.5),
                            ]),
                        ]),
                    },
                ],
            )),
            vec![Argument::Native(0)],
            1,
            vec![],
        )
        .unwrap();
    // Also put a non-f32 gain in the ordinary node list.
    let previous = candidate.program.output;
    candidate.program.nodes.push(Node::Gain {
        input: previous,
        coefficient: Coefficient::Number(0.7),
    });
    candidate.program.output = candidate.program.nodes.len() - 1;
    assert!(!candidate.has_f32_literals());
    let rounded = candidate.f32_literals().unwrap();
    assert!(rounded.has_f32_literals());
    let decoded = Artifact::from_bytes(&rounded.to_bytes().unwrap(), &model.declarations).unwrap();
    let family = raw_family(vec![vec![1.0, 2.0]]);
    let output = decoded.execute(&family).unwrap().values[decoded.program.output].clone();
    let expected = (0.1f32 as f64) * ((0.3f32 as f64) + 0.5) * (0.7f32 as f64);
    assert_eq!(output[[0, 0]], expected);
    assert_eq!(output[[0, 1]], expected * 2.0);
}

#[test]
fn native_rms_norm_survives_artifact_roundtrip_coverage() {
    use crate::artifact::EncodedArtifact;
    use crate::precision::DecodableArtifact;
    let mut model = mlp(1.0);
    let output = model.output;
    model.nodes.push(Node::RmsNorm { input: output, epsilon: 1e-5 });
    model.output = model.nodes.len() - 1;
    let native = Artifact::native(&model).unwrap().f32_literals().unwrap();
    let decoded = EncodedArtifact::of(&native).unwrap().decode().unwrap();
    decoded.validate_coverage(&model).unwrap();
}

/// C32 removes exact node-lattice payloads before charging each real once at 32 bits.
#[test]
fn c32_root_real_literals_have_value_independent_price() {
    use super::operator_program::Coefficient;
    let make = |gain: f64, epsilon: f64| {
        let program = raw_program(2, vec![], vec![
            Node::Raw { slot: 0 },
            Node::Gain { input: 0, coefficient: Coefficient::Product(vec![
                Coefficient::Number(gain),
                Coefficient::Sum(vec![Coefficient::Number(0.25), Coefficient::Number(0.5)]),
            ]) },
            Node::RmsNorm { input: 1, epsilon },
        ]);
        Artifact::native(&program).unwrap().f32_literals().unwrap()
    };
    let (first, second) = (make(0.1, 1e-5), make(16.0, 2f64.powi(-60)));
    let mut cache = CostCache::default();
    let (a, b) = (structural_cost(&first, &mut cache).unwrap(), structural_cost(&second, &mut cache).unwrap());
    assert_eq!(a.literals, 4, "three Gain numbers and one exact architecture epsilon");
    assert_eq!(a, b, "literal magnitudes and exact codec precision do not change C32");
    assert_ne!(first.program.code_bits().unwrap(), second.program.code_bits().unwrap(), "the preserved exact wire account is separate");
    let (header, bases, rules, nodes) = first.program.frame_bits().unwrap();
    let (count, payload) = first.program.frame_literal_payload().unwrap();
    assert_eq!(count, 4);
    assert_eq!(a.structure_bits, header + bases.iter().sum::<u64>() + rules + nodes - payload);
}

/// Shared numerical bodies are charged once; a call pays structure, never another copy of the literals.
#[test]
fn c32_shared_rule_real_literals_are_paid_once() {
    use super::operator_program::Coefficient;
    let body = rule("shared numerical body", vec![native(2)], vec![
        Node::Param { index: 0 },
        Node::Gain { input: 0, coefficient: Coefficient::Product(vec![
            Coefficient::Number(0.25), Coefficient::Number(0.5),
        ]) },
        Node::RmsNorm { input: 1, epsilon: 1e-5 },
    ]);
    let mut one = raw_program(2, vec![], vec![Node::Raw { slot: 0 }, Node::Call { rule: 0, arguments: vec![0] }]);
    one.rules.push(body.clone());
    let mut two = one.clone();
    two.nodes.push(Node::Call { rule: 0, arguments: vec![1] });
    two.output = 2;
    let cost = |program: &OperatorProgram| structural_cost(&Artifact::native(program).unwrap(), &mut CostCache::default()).unwrap();
    let (a, b) = (cost(&one), cost(&two));
    assert_eq!((a.literals, b.literals), (3, 3));
    assert!(b.structure_bits > a.structure_bits, "the extra call and its wiring still cost structure");
    let mut duplicated = two;
    duplicated.rules.push(body);
    duplicated.nodes[2] = Node::Call { rule: 1, arguments: vec![1] };
    assert_eq!(cost(&duplicated).literals, 6, "separately stored bodies have independent payloads");
}

/// A pricing correction must neither round a native architecture epsilon nor change its decoded execution.
#[test]
fn c32_preserves_exact_native_epsilon_and_execution() {
    let epsilon = 1e-5;
    assert_ne!(epsilon, f64::from(epsilon as f32));
    let program = raw_program(2, vec![], vec![Node::Raw { slot: 0 }, Node::RmsNorm { input: 0, epsilon }]);
    let artifact = Artifact::native(&program).unwrap().f32_literals().unwrap();
    let decoded = Artifact::from_bytes(&artifact.to_bytes().unwrap(), &program.declarations).unwrap();
    assert_eq!(decoded.program.nodes, program.nodes, "wire decoding preserves the exact architecture value");
    let family = raw_family(vec![vec![0.25, 2.0], vec![-0.5, 0.75]]);
    assert_eq!(decoded.execute(&family).unwrap().values, program.execute(&family, false).unwrap().values);
    decoded.validate_coverage(&program).unwrap();
    assert_eq!(structural_cost(&decoded, &mut CostCache::default()).unwrap().literals, 1);
}

/// An exception value is a literal; its selectors and conditionals are the binding remainder.
#[test]
fn c32_exception_value_and_binding_structure_are_separate() {
    use super::codec::{fixed_index_len_bits, prefix_integer_len_bits};
    let program = raw_program(2, vec![], vec![Node::Raw { slot: 0 }]);
    let plain = Artifact::native(&program).unwrap();
    let mut exception = plain.clone();
    exception.exceptions.push(super::artifact::Exception { context: vec![], node: 0, column: 1, value: 0.125 });
    let (a, b) = (
        structural_cost(&plain, &mut CostCache::default()).unwrap(),
        structural_cost(&exception, &mut CostCache::default()).unwrap(),
    );
    let binding_delta = prefix_integer_len_bits(2).unwrap() - prefix_integer_len_bits(1).unwrap()
        + prefix_integer_len_bits(1).unwrap() + u64::from(fixed_index_len_bits(1).unwrap()) + u64::from(fixed_index_len_bits(2).unwrap());
    assert_eq!(b.literals, a.literals + 1);
    assert_eq!(b.structure_bits, a.structure_bits);
    assert_eq!(b.binding_bits - a.binding_bits, binding_delta, "the 32 value bits are not in binding structure");
    assert_eq!(b.total() - a.total(), binding_delta + 32);
    assert_eq!(exception.encode().unwrap().len_bits() - plain.encode().unwrap().len_bits(), binding_delta + 32);
    exception.exceptions[0].value = 123.5;
    assert_eq!(structural_cost(&exception, &mut CostCache::default()).unwrap(), b);
}

/// Computed operator entries are absent from literal counts; only the scale and residual cells are sent.
#[test]
fn c32_derived_scale_and_residual_are_each_paid_once() {
    let (d, w) = (native(2), native(1));
    let diagonal = |name: &str| Operator::diag(name, d.clone(), array![1.0, 1.0], exact_precision([1.0]).unwrap(), Provenance::default()).unwrap();
    let program = raw_program(2, vec![
        diagonal("gain"), dense("value", &w, &d, array![[1.0, 0.0]]),
        dense("output", &d, &w, array![[1.0], [0.0]]), diagonal("final gain"),
    ], vec![Node::Raw { slot: 0 }]);
    let start = Artifact::native(&program).unwrap();
    let law = super::artifact::OperatorLaw::Copy { value: 1, gain: 0, final_gain: 3 };
    let plain = start.derive(2, law.clone(), 1.0, vec![]).unwrap();
    let residual = start.derive(2, law.clone(), 0.5, vec![(0, vec![0.125]), (1, vec![0.25])]).unwrap();
    let cost = |a: &Artifact| structural_cost(a, &mut CostCache::default()).unwrap();
    assert_eq!(cost(&start).literals - cost(&plain).literals, 2 - 1);
    assert_eq!(cost(&residual).literals - cost(&plain).literals, 2);
    assert_eq!(residual.derived_literals().unwrap(), 3);
    assert_eq!(residual.message_program().unwrap().operators[2].real_count(), 0);
    let different_scale = start.derive(2, law, 0.75, vec![]).unwrap();
    assert_eq!(cost(&plain), cost(&different_scale));
    let decoded = Artifact::from_bytes(&residual.to_bytes().unwrap(), &program.declarations).unwrap();
    assert_eq!(cost(&residual), cost(&decoded));
}

/// Indicator values are fixed primitives; the domain's finite size is structural.
#[test]
fn c32_indicator_basis_has_no_independent_numerical_literals() {
    use super::operator_program::{Basis, Domain};
    let program = |size| OperatorProgram {
        declarations: Declarations { domains: vec![Domain { size }], slots: vec![Slot::Token { domain: 0 }], parameters: 0 },
        bases: vec![Basis::Indicator { domain: 0 }],
        operators: vec![], rules: vec![],
        nodes: vec![Node::Feature { slot: 0, basis: 0 }, Node::Readout { input: 0, basis: 0 }], output: 1,
    };
    for size in [3, 17] {
        let p = program(size);
        let cost = structural_cost(&Artifact::native(&p).unwrap(), &mut CostCache::default()).unwrap();
        assert_eq!(cost.literals, 0);
        assert!(cost.structure_bits > 0);
    }
}

/// The current grammar sends rotary bases and inverse-square-root arguments as independent arithmetic knobs.
#[test]
fn c32_attention_arithmetic_integer_literals_are_paid_once() {
    use super::operator_program::{Rotary, Scale};
    let make = |n, base| raw_program(2, vec![], vec![
        Node::Raw { slot: 0 },
        Node::Attend { query: 0, key: 0, value: 0, scale: Scale::InverseSqrt(n),
            rotary: Some(Rotary { base, dims: 2, half_split: true }), causal: true },
    ]);
    let (a, b) = (make(2, 10000), make(7, 65537));
    let cost = |p: &OperatorProgram| structural_cost(&Artifact::native(p).unwrap(), &mut CostCache::default()).unwrap();
    assert_eq!(cost(&a).literals, 2, "scale argument and rotary base are independent numeric literals");
    assert_eq!(cost(&a), cost(&b), "changing arithmetic literal values preserves their fixed32 price");
    assert_ne!(a.code_bits().unwrap(), b.code_bits().unwrap(), "exact integer wire lengths remain separate");
    let artifact = Artifact::native(&a).unwrap();
    let decoded = Artifact::from_bytes(&artifact.to_bytes().unwrap(), &a.declarations).unwrap();
    assert_eq!(decoded.program.nodes, a.nodes);
    let mut primitive = a.clone();
    primitive.nodes[1] = Node::Attend { query: 0, key: 0, value: 0, scale: Scale::One, rotary: None, causal: true };
    assert_eq!(cost(&primitive).literals, 0, "the fixed primitive one is not a fitted scale literal");
}

/// A bilinear scale equal to input width still lacks an explicit dimension derivation in the current enum.
#[test]
fn c32_bilinear_scale_argument_is_not_free_when_equal_to_width() {
    use super::operator_program::Scale;
    let make = |n| raw_program(2, vec![], vec![Node::Raw { slot: 0 }, Node::Bilinear { left: 0, right: 0, scale: Scale::InverseSqrt(n) }]);
    let cost = |p: &OperatorProgram| structural_cost(&Artifact::native(p).unwrap(), &mut CostCache::default()).unwrap();
    assert_eq!(cost(&make(2)).literals, 1);
    assert_eq!(cost(&make(2)), cost(&make(63)));
}

/// Dimensions specify structure even when the same node also has priced arithmetic literals.
#[test]
fn c32_rotary_dimension_is_structure_not_an_extra_numeric_literal() {
    use super::operator_program::{Rotary, Scale};
    let make = |dims| raw_program(4, vec![], vec![Node::Raw { slot: 0 }, Node::Attend {
        query: 0, key: 0, value: 0, scale: Scale::One,
        rotary: Some(Rotary { base: 10000, dims, half_split: true }), causal: true,
    }]);
    let cost = |p: &OperatorProgram| structural_cost(&Artifact::native(p).unwrap(), &mut CostCache::default()).unwrap();
    let (a, b) = (cost(&make(2)), cost(&make(4)));
    assert_eq!((a.literals, b.literals), (1, 1));
    assert_ne!(a.structure_bits, b.structure_bits, "the chosen dimension remains explicitly coded structure");
}

#[test]
fn copy_joint_masks_keep_cancellation_and_compose_four_layers() {
    use super::proposals::CopyMasks;
    use super::run_check::LayerNodes;
    let (d, w) = (native(2), native(1));
    let diag = |name: &str| Operator::diag(name, d.clone(), array![1.0, 1.0], exact_precision([1.0]).unwrap(), Provenance::default()).unwrap();
    let mut operators = vec![diag("final_norm.gain")];
    let mut nodes = vec![Node::Raw { slot: 0 }];
    let mut layers = Vec::new();
    for layer in 0..4 {
        operators.push(diag(&format!("blocks.{layer}.rms1.gain")));
        let mut reads = Vec::new();
        let mut terms = Vec::new();
        for head in 0..6 {
            let v = operators.len();
            operators.push(dense(&format!("blocks.{layer}.v{head}"), &w, &d, array![[1.0, 0.0]]));
            let read = nodes.len();
            nodes.push(Node::Affine { terms: vec![(0, v)], bias: None });
            reads.push(read);
            let o = operators.len();
            let error = match head { 0 => 1.0, 1 => -1.0, _ => 0.0 };
            operators.push(dense(&format!("blocks.{layer}.o{head}"), &d, &w, array![[1.0], [error]]));
            terms.push((read, o));
        }
        let attention = nodes.len();
        nodes.push(Node::Affine { terms, bias: None });
        layers.push(LayerNodes { stream: 0, normed_stream: 0, queries: reads.clone(), keys: reads.clone(), values: reads.clone(), reads,
            attention, attended: attention, normed: 0, pre: 0, active: 0, mlp: attention, residual: attention });
    }
    let model = raw_program(2, operators, nodes);
    let start = Artifact::native(&model).unwrap();
    assert!(CopyMasks::cardinalities([6; 4], 255, 1 << 24).is_err());
    assert!(CopyMasks::cardinalities([6; 4], 256, (1 << 24) - 1).is_err());
    let bank = CopyMasks::new(&start, &layers, 1, 256, 1 << 24).unwrap();
    assert_eq!((bank.layer_candidates, bank.joint_candidates), (256, 1 << 24));
    assert_eq!(bank.layer_masks().filter(|(layer, _)| *layer == 0).count(), 64);
    let local = Local::new(&model, raw_family(vec![vec![1.0, 0.0]]), None, 1);
    for mask in [1, 2] {
        let candidate = bank.checked(&[mask, 0, 0, 0], &model).unwrap().0;
        assert!(local.screen(&candidate).unwrap().worst().unwrap().lower > 0.1);
    }
    let composed = bank.compose(&[3, 3, 3, 3]).unwrap();
    let repeated = bank.compose(&[3, 3, 3, 3]).unwrap();
    let mut reference = start.clone();
    for layer in 0..4 {
        for head in 0..2 {
            let single = bank.layer(layer, 1 << head).unwrap();
            let derived = &single.derived[0];
            reference = reference.derive(derived.operator, derived.law.clone(), derived.scale, derived.residual.clone()).unwrap();
            assert!(Arc::ptr_eq(&composed.program.operators[derived.operator], &repeated.program.operators[derived.operator]), "computed Copy operators are shared across masks");
        }
        reference = reference.bind(&format!("attention {layer}"), &[0], layers[layer].attention).unwrap();
    }
    assert_eq!(composed, reference, "cached composition matches the canonical derive/bind path");
    let separate_bytes: usize = (0..4).map(|layer| bank.layer(layer, 3).unwrap().to_bytes().unwrap().len()).sum();
    assert!(composed.to_bytes().unwrap().len() < separate_bytes, "composition transmits remaining native computation once");
    let joint = bank.checked(&[3, 3, 3, 3], &model).unwrap().0;
    assert_eq!(joint.derived.len(), 8);
    assert_eq!(joint.blocks.len(), 4);
    assert!(local.screen(&joint).unwrap().blocks.iter().all(|b| b.upper < 1e-12));
    assert!(bank.layer_masks().any(|(l, m)| l == 0 && m == 3), "failed singletons never remove their cancelling joint mask");
    assert!(bank.compose(&[64, 0, 0, 0]).is_err());
    assert!(bank.compose(&[0]).is_err());
    let empty = bank.checked(&[0; 4], &model).unwrap().0;
    assert!(empty.derived.is_empty() && empty.blocks.is_empty());
}

#[test]
fn copy_mask_classification_rejects_invalid_nonmax_local_evidence() {
    use super::acceptance::{BlockError, LocalMeasure};
    let block = BlockError { name: "valid".into(), worst: 0.1, row: 0, lower_row: 0, numerical_error: 0.0, lower: 0.1, upper: 0.1, scale: 1.0 };
    for kind in 0..4 {
        let mut bad = block.clone();
        match kind { 0 => bad.worst = f64::NAN, 1 => bad.numerical_error = f64::INFINITY,
            2 => bad.upper = f64::NAN, 3 => bad.upper = 0.0, _ => unreachable!() }
        let measure = LocalMeasure { blocks: vec![block.clone(), bad], rows: 1, family_rows: 1, counterexamples: 0 };
        assert!(measure.status().is_err(), "invalid nonmax evidence must remain unresolved, never prune or pass");
    }
}

#[test]
fn assess_once_matches_complete_cached_assessment_without_measurement_key() {
    let width = native(1);
    let classes = native(2);
    let model = raw_program(
        1,
        vec![Operator::identity("I", width.clone()), dense("head", &classes, &width, array![[1.0], [-1.0]])],
        vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0, 0)], bias: None }, Node::Affine { terms: vec![(1, 1)], bias: None }],
    );
    let family = grid(&[-1.0, 0.5, 1.0], 1);
    let start = Artifact::native(&model).unwrap();
    let k = model.operators.len();
    let candidate = start
        .replace_block(
            "changed",
            Callee::New(rule("gain", vec![width.clone()], vec![Node::Param { index: 0 }, Node::Affine { terms: vec![(0, k)], bias: None }])),
            vec![Argument::Native(0)],
            1,
            vec![dense("gain", &width, &width, array![[1.25]])],
        )
        .unwrap();
    let local = Local::new(&model, family.clone(), None, 64);
    let run = clean_run(&model, &family);
    let codec = super::operator_program::NativeOperatorCodec::new(&start.program, 1 << 20).unwrap();
    let mut once_cache = CostCache::default();
    // Native exercises witnessed same-index hits; replacement compacts operators
    // and must preserve ordinary results through the permutation fallback.
    for candidate in [&start, &candidate] {
    for constraint in [Constraint { local: 0.01, run: 0.01 }, Constraint { local: 1.0, run: 1.0 }] {
        let old = assess(&local, &run, candidate, constraint, &mut CostCache::default()).unwrap();
        let once = super::acceptance::assess_once(&local, &run, candidate, constraint, &mut once_cache).unwrap();
        assert_eq!(old.cost, once.cost);
        assert_eq!(old.local, once.local);
        assert_eq!(old.local_measure, once.local_measure);
        assert_eq!(old.run, once.run);
        assert_eq!(old.run_measure, once.run_measure);
        assert_eq!(old.meets(), once.meets());
        let reused = super::acceptance::assess_once_with_native_codec(
            &local, &run, candidate, constraint, &mut once_cache, &codec,
        ).unwrap();
        assert_eq!(old.cost, reused.cost);
        assert_eq!(old.local, reused.local);
        assert_eq!(old.local_measure, reused.local_measure);
        assert_eq!(old.run, reused.run);
        assert_eq!(old.run_measure, reused.run_measure);
        assert_eq!(old.meets(), reused.meets());
    }
    }
    assert!(codec.usage().encoded_native_operator_hits > 0);
    assert!(codec.usage().decoded_native_operator_hits > 0);
}
