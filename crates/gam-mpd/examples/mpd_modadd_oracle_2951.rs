//! The known mechanism of the p = 31 modular-addition transformer, written by hand as an operator
//! program and scored by the engine's own code: the target the engine must reach without being
//! told the answer (#2951). An analyst's check: nothing here is an input to the engine.
//!
//! The oracle reads the residual at `=` through the Fourier features `cos/sin(ω_k a)`,
//! `cos/sin(ω_k b)` of the five frequencies the embedding uses (`k ∈ {5, 7, 8, 10, 14}`), each unit
//! reads the four features of one frequency (the one explaining most of its pre-activation), and
//! each unit writes only the class plane `cos/sin(ω_k c)` of its frequency; the embedding is
//! restricted to the constant and those planes. Every coefficient is a least-squares fit to the
//! model, and the operators' lattices are chosen by the engine's precision moves alone.
//!
//! `mpd_modadd_oracle_2951 EXPORT_DIR` (the p31 export `gam_mpd::import` reads, e.g.
//! `~/mpd-data/engine/p31_s0_generic`). At `n = 10³` and `10⁶` it scores the native program, the
//! oracle and the engine's blind result, and fails unless the engine's total is within the stated
//! gap of the oracle's and, at `10⁶`, shorter than the native program with no argmax error.

use gam_mpd::contract::{Contract, ProgramScore};
use gam_mpd::dense::svd;
use gam_mpd::derivatives::CurvaturePrecision;
use gam_mpd::engine::{Budget, Coarsen, Primitive, decompose, decompose_with_reference, library};
use gam_mpd::import::import;
use gam_mpd::operator_program::{Interface, LabelKind, Node, Operator, OperatorProgram, Provenance, SlotValues};
use gam_mpd::operator_rewrites::insert_node;
use gam_mpd::precision::DeclaredPrecision;
use ndarray::{Array1, Array2, Axis, s};
use std::f64::consts::TAU;
use std::path::PathBuf;

const P: usize = 31;
const FREQUENCIES: [usize; 5] = [5, 7, 8, 10, 14];

fn fine() -> DeclaredPrecision {
    DeclaredPrecision::new(40).expect("a precision")
}

/// `X ≈ A W` by minimum-norm least squares over the singular values above the band: `W` and the
/// residual sum of squares per column.
fn least_squares(a: &Array2<f64>, x: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let decomposed = svd(a.view(), false).expect("svd");
    let mut projected = decomposed.u.t().dot(x);
    for (k, sigma) in decomposed.singular_values.iter().enumerate() {
        let inverse = if *sigma > decomposed.band { 1.0 / sigma } else { 0.0 };
        projected.row_mut(k).mapv_inplace(|v| v * inverse);
    }
    let solution = decomposed.vt.t().dot(&projected);
    let residual = x - &a.dot(&solution);
    let squares = residual.map_axis(Axis(0), |c| c.iter().map(|v| v * v).sum());
    (solution, squares)
}

fn fit(a: &Array2<f64>, x: &Array2<f64>) -> Array2<f64> {
    least_squares(a, x).0
}

fn centred(m: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let mean = m.mean_axis(Axis(0)).expect("rows");
    (m - &mean, mean)
}

/// The hand-written mechanism (module note), from the imported native program.
fn oracle(model: &OperatorProgram, contract: &Contract) -> OperatorProgram {
    let trace = model.execute(&contract.family, false).expect("executes");
    let rows = contract.family.rows;
    let tokens = |slot: usize| match &contract.family.slots[slot] {
        SlotValues::Tokens(t) => t.clone(),
        SlotValues::Raw(_) => panic!("token slots"),
    };
    let (a, b) = (tokens(0), tokens(1));
    let k_count = FREQUENCIES.len();
    let angle = |k: usize, t: usize| TAU * ((k * t) % P) as f64 / P as f64;
    // The layer: a pointwise node, its pre-activation, the residual it reads, its writer and logits.
    let act = model.nodes.iter().position(|n| matches!(n, Node::Pointwise { .. })).expect("a ReLU layer");
    let Node::Pointwise { input: pre, .. } = model.nodes[act] else { unreachable!() };
    let Node::Affine { terms: pre_terms, .. } = &model.nodes[pre] else { panic!("an affine pre-activation") };
    let mid = pre_terms[0].0;
    let fin = model
        .nodes
        .iter()
        .position(|n| matches!(n, Node::Affine { terms, .. } if terms.iter().any(|(x, _)| *x == act)))
        .expect("the layer's writer");
    let Node::Affine { terms: fin_terms, .. } = &model.nodes[fin] else { unreachable!() };
    let w_out = fin_terms.iter().find(|(x, _)| *x == act).expect("the write").1;
    let logits = model
        .nodes
        .iter()
        .position(|n| matches!(n, Node::Affine { terms, .. } if terms.iter().any(|(x, _)| *x == fin)))
        .expect("the logits");
    let Node::Affine { terms: logit_terms, .. } = &model.nodes[logits] else { unreachable!() };
    let w_u = logit_terms[0].1;
    // Read: z = R x, the least-squares Fourier features of (a, b) at each frequency.
    let features = Array2::from_shape_fn((rows, 4 * k_count), |(row, column)| {
        let (k, part) = (FREQUENCIES[column / 4], column % 4);
        let t = if part < 2 { a[row] } else { b[row] } as usize;
        if part % 2 == 0 { angle(k, t).cos() } else { angle(k, t).sin() }
    });
    let (x, _) = centred(&trace.values[mid]);
    let (f, _) = centred(&features);
    let read = fit(&x, &f).t().to_owned();
    let z = trace.values[mid].dot(&read.t());
    // Each unit: the frequency whose four features explain most of its pre-activation.
    let units = trace.values[pre].ncols();
    let mut coefficients = Array2::<f64>::zeros((units, 4 * k_count));
    let mut present = Array2::from_elem((units, k_count), false);
    let mut bias = Array1::<f64>::zeros(units);
    let mut frequency_of = vec![0usize; units];
    for unit in 0..units {
        let target = trace.values[pre].column(unit).to_owned().insert_axis(Axis(1));
        let mut best = (f64::INFINITY, 0, Array2::zeros((5, 1)));
        for k in 0..k_count {
            let mut design = Array2::<f64>::ones((rows, 5));
            design.slice_mut(s![.., ..4]).assign(&z.slice(s![.., 4 * k..4 * k + 4]));
            let (solution, squares) = least_squares(&design, &target);
            if squares[0] < best.0 {
                best = (squares[0], k, solution);
            }
        }
        let (_, k, w) = best;
        frequency_of[unit] = k;
        present[[unit, k]] = true;
        coefficients.slice_mut(s![unit, 4 * k..4 * k + 4]).assign(&w.slice(s![..4, 0]));
        bias[unit] = w[[4, 0]];
    }
    // Write: each unit's logit contribution `W_U W_out[:, n]` on its frequency's class plane, its
    // mean over classes dropped (the softmax shift).
    let contribution = model.operators[w_u].matrix().dot(&model.operators[w_out].matrix());
    let classes = contribution.nrows();
    let table = Array2::from_shape_fn((classes, 2 * k_count), |(c, column)| {
        let k = FREQUENCIES[column / 2];
        if column % 2 == 0 { angle(k, c).cos() } else { angle(k, c).sin() }
    });
    let mut writes = Array2::<f64>::zeros((2 * k_count, units));
    let mut write_present = Array2::from_elem((k_count, units), false);
    for unit in 0..units {
        let k = frequency_of[unit];
        let u = contribution.column(unit).to_owned();
        let u = &u - u.mean().expect("classes");
        let w = fit(&table.slice(s![.., 2 * k..2 * k + 2]).to_owned(), &u.insert_axis(Axis(1)));
        writes.slice_mut(s![2 * k..2 * k + 2, unit]).assign(&w.column(0));
        write_present[[k, unit]] = true;
    }
    // The embedding restricted to the constant and the five planes over the 31 number tokens.
    let embedding = model.operators.iter().position(|op| op.name == "W_E").expect("W_E");
    let mut e = model.operators[embedding].matrix();
    let mut basis = Array2::<f64>::ones((P, 1 + 2 * k_count));
    for (i, &k) in FREQUENCIES.iter().enumerate() {
        for t in 0..P {
            basis[[t, 1 + 2 * i]] = angle(k, t).cos();
            basis[[t, 2 + 2 * i]] = angle(k, t).sin();
        }
    }
    let numbers = e.slice(s![.., ..P]).t().to_owned();
    let projected = basis.dot(&fit(&basis, &numbers));
    e.slice_mut(s![.., ..P]).assign(&projected.t());

    let mut program = model.clone();
    let read_factors = Interface::uniform(k_count, 4, LabelKind::Factor, 0).expect("interface");
    let write_factors = Interface::uniform(k_count, 2, LabelKind::Factor, k_count as u32).expect("interface");
    let mid_interface = program.interfaces().expect("interfaces")[mid].clone();
    let unit_interface = program.interfaces().expect("interfaces")[pre].clone();
    let class_interface = program.operators[w_u].rows.clone();
    let push = |program: &mut OperatorProgram, op: Operator| {
        program.operators.push(std::sync::Arc::new(op));
        program.operators.len() - 1
    };
    let e_op = &program.operators[embedding];
    let e_new = Operator::dense("W_E on five planes", e_op.rows.clone(), e_op.cols.clone(), e, fine(), Provenance::default()).expect("dense");
    program.operators[embedding] = std::sync::Arc::new(e_new);
    let read_op = push(&mut program, Operator::dense("Fourier reads", read_factors.clone(), mid_interface, read, fine(), Provenance::default()).expect("dense"));
    let coefficient_op = push(
        &mut program,
        Operator::blocks("unit reads", unit_interface.clone(), read_factors, coefficients, present, fine(), Provenance::default()).expect("blocks"),
    );
    let bias_op = push(&mut program, Operator::dense("unit biases", unit_interface.clone(), Interface::constant(), bias.insert_axis(Axis(1)), fine(), Provenance::default()).expect("dense"));
    let write_op = push(
        &mut program,
        Operator::blocks("unit writes", write_factors.clone(), unit_interface, writes, write_present, fine(), Provenance::default()).expect("blocks"),
    );
    let table_op = push(&mut program, Operator::dense("class planes", class_interface, write_factors, table, fine(), Provenance::default()).expect("dense"));
    // Nodes: z before the pre-activation, y after the layer; the residual drops the layer's write
    // and the logits read y.
    let z_node = insert_node(&mut program, pre, Node::Affine { terms: vec![(mid, read_op)], bias: None });
    let (pre, act) = (pre + 1, act + 1);
    program.nodes[pre] = Node::Affine { terms: vec![(z_node, coefficient_op)], bias: Some(bias_op) };
    let y_node = insert_node(&mut program, act + 1, Node::Affine { terms: vec![(act, write_op)], bias: None });
    let reads = |program: &OperatorProgram, argument: usize, op: usize| {
        program
            .nodes
            .iter()
            .position(|n| matches!(n, Node::Affine { terms, .. } if terms.contains(&(argument, op))))
            .expect("a reader")
    };
    let fin = reads(&program, act, w_out);
    if let Node::Affine { terms, .. } = &mut program.nodes[fin] {
        terms.retain(|term| *term != (act, w_out));
    }
    let logits = reads(&program, fin, w_u);
    if let Node::Affine { terms, .. } = &mut program.nodes[logits] {
        terms.push((y_node, table_op));
    }
    program.prune();
    program.interfaces().expect("the oracle is well formed");
    // Every operator on its lattice, chosen by the engine's own precision moves from the oracle
    // alone: the start set is the oracle, the library only re-precises and coarsens.
    let reference = contract.logits(model).expect("reference");
    let precisions: Vec<Box<dyn Primitive>> = vec![Box::new(CurvaturePrecision { probes: 4 }), Box::new(Coarsen)];
    decompose_with_reference(&reference, &[&program], contract, &precisions, &Budget::default()).expect("precisions").program
}

fn line(label: &str, score: &ProgramScore) {
    eprintln!(
        "{label}: {} bits = structure {} + precision {}; explanations {:.1}; data {:.1}; total {:.1}; {} argmax disagreements",
        score.program_bits,
        score.structure_bits,
        score.precision_bits,
        score.explanation.bits,
        score.data_bits,
        score.total(),
        score.evaluation.argmax_disagreements
    );
}

/// The engine, run blind on the model, reaches the hand-written mechanism's code length within
/// the stated gap, and at large `n` is shorter than the native program with no argmax error.
fn main() -> Result<(), String> {
    let dir = PathBuf::from(std::env::args().nth(1).ok_or("mpd_modadd_oracle_2951 EXPORT_DIR")?);
    let imported = import(&dir)?;
    // The stated gap: the engine's total may exceed the oracle's by at most this fraction.
    const GAP: f64 = 0.1;
    for n in [1_000u64, 1_000_000] {
        let mut contract = imported.contract.clone();
        contract.observations = n;
        let model = &imported.program;
        let reference = contract.logits(model).map_err(|e| e.to_string())?;
        let native = contract.score(model, &reference).map_err(|e| e.to_string())?;
        let mechanism = oracle(model, &contract);
        let target = contract.score(&mechanism, &reference).map_err(|e| e.to_string())?;
        let result = decompose(model, &contract, &library(), &Budget::default()).map_err(|e| e.to_string())?;
        eprintln!("n = {n}");
        line("  native", &native);
        line("  oracle", &target);
        line("  engine", &result.score);
        if result.score.total() > (1.0 + GAP) * target.total() {
            return Err(format!("n = {n}: engine {} against oracle {}", result.score.total(), target.total()));
        }
        if n >= 1_000_000 && (native.proven_shorter_than(&result.score) || result.score.evaluation.argmax_disagreements > 0) {
            return Err(format!("n = {n}: engine {} against native {} with {} argmax errors", result.score.total(), native.total(), result.score.evaluation.argmax_disagreements));
        }
    }
    Ok(())
}
