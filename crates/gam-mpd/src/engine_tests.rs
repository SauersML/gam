#![cfg(test)]
//! The engine on planted programs: exact structure is found and certified, a recovered labelling
//! is the planted cycle up to an automorphism, and restating a program (duplicating or rescaling a
//! component, re-partitioning an interface) never makes it look shorter.

use super::contract::{Contract, FamilyKind};
use super::engine::{Budget, Coarsen, DropBlocks, Primitive, decompose, decompose_from};
use super::operator_program::{
    Basis, Declarations, Domain, FamilyInputs, Interface, LabelKind, Node, Operator, OperatorBody, OperatorProgram,
    Provenance, Slot, SlotValues,
};
use super::engine::{EngineError, Edit, Exactness, Proposal, SearchContext};
use super::fit::ProposalKind;
use super::operator_rewrites::change_basis;
use super::precision::DeclaredPrecision;
use ndarray::Array2;
use std::f64::consts::TAU;
use std::sync::Arc;

const P: usize = 7;
const PLANTED: usize = 2;
const WIDTH: usize = 4;
const REPEATS: u64 = 1000;

fn precision(bits: i32) -> DeclaredPrecision {
    DeclaredPrecision::new(bits).expect("a precision in range")
}

/// A table over `Z_7` that is one character plane (`k = 2`) plus a constant: `E[:, x] = c +
/// u cos(2ω x) + v sin(2ω x)`.
fn planted_table() -> Array2<f64> {
    let c = [0.25, -0.5, 0.125, 0.75];
    let u = [1.0, 0.5, -0.75, 0.25];
    let v = [-0.25, 1.0, 0.5, -0.5];
    Array2::from_shape_fn((WIDTH, P), |(i, x)| {
        let angle = TAU * ((PLANTED * x) % P) as f64 / P as f64;
        c[i] + u[i] * angle.cos() + v[i] * angle.sin()
    })
}

/// Two token slots over `Z_7`, each embedded by the planted table, summed and read out over seven
/// classes by a dense map. With `declared`, the domain declares the cycle `x → x + 1`.
fn planted_program(declared: bool) -> (OperatorProgram, Contract) {
    let cycle = declared.then(|| (0..P as u32).map(Some).collect::<Vec<_>>());
    let declarations = Declarations { parameters: 0,
        domains: vec![Domain { size: P, cycle }, Domain { size: P, cycle: None }],
        slots: vec![Slot::Token { domain: 0 }, Slot::Token { domain: 0 }],
    };
    let tokens = Interface::uniform(P, 1, LabelKind::Token, 0).expect("interface");
    let model = Interface::native(WIDTH).expect("interface");
    let classes = Interface::uniform(P, 1, LabelKind::Token, 0).expect("interface");
    let readout = Array2::from_shape_fn((P, WIDTH), |(c, i)| ((c * 3 + i * 5) as f64 * 0.37).sin());
    let operators = vec![
        Operator::dense("E", model.clone(), tokens, planted_table(), precision(30), Provenance::native("E"))
            .expect("dense"),
        Operator::dense("U", classes, model, readout, precision(30), Provenance::native("U")).expect("dense"),
    ];
    let nodes = vec![
        Node::Feature { slot: 0, basis: 0 },
        Node::Feature { slot: 1, basis: 0 },
        Node::Affine { terms: vec![(0, 0), (1, 0)], bias: None },
        Node::Affine { terms: vec![(2, 1)], bias: None },
        Node::Readout { input: 3, basis: 1 },
    ];
    let program = OperatorProgram { rules: Vec::new(),
        declarations: declarations.clone(),
        bases: vec![Basis::Indicator { domain: 0 }, Basis::Indicator { domain: 1 }],
        operators: operators.into_iter().map(Arc::new).collect(),
        nodes,
        output: 4,
    };
    let pairs: Vec<(u32, u32)> = (0..P as u32).flat_map(|a| (0..P as u32).map(move |b| (a, b))).collect();
    let contract = Contract {
        declarations,
        family: FamilyInputs {
            layout: None,
            rows: pairs.len(),
            slots: vec![
                SlotValues::Tokens(pairs.iter().map(|q| q.0).collect()),
                SlotValues::Tokens(pairs.iter().map(|q| q.1).collect()),
            ],
        },
        kind: FamilyKind::Complete { description: "Z_7^2".to_string() },
        // Every input observed REPEATS times: the amount of behaviour the program explains.
        observations: REPEATS,
        readouts: 1,
        readout_slots: None,
    };
    (program, contract)
}

/// The contract's declared cycle as a prior: the character basis of each declared cycle.
struct DeclaredCharacters;

impl Primitive for DeclaredCharacters {
    fn name(&self) -> &'static str {
        "declared_characters"
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError> {
        let program = context.program;
        let mut out = Vec::new();
        for (index, basis) in program.bases.iter().enumerate() {
            let Basis::Indicator { domain } = basis else { continue };
            let Some(cycle) = program.declarations.domains[*domain].cycle.clone() else { continue };
            if let Some(candidate) = change_basis(program, index, cycle, true)? {
                out.push(Proposal {
                    primitive: "declared_characters",
                    kind: ProposalKind::Expose,
                    exactness: Exactness::Exact { derivation: "indicators are Φ⁻¹ times the declared characters".to_string() },
                    description: format!("declared characters on domain {domain}"),
                    edit: Edit::Program(Box::new(candidate)),
                });
            }
        }
        Ok(out)
    }
}

fn budget() -> Budget {
    Budget { screenings: 100_000, certifications: 1_000, refit: None }
}

/// The present column labels of the operator reading the character features.
fn kept_planes(program: &OperatorProgram) -> Vec<u32> {
    let op = program
        .operators
        .iter()
        .find(|op| op.cols.groups().iter().any(|g| g.label.kind == LabelKind::Plane))
        .expect("an operator reads the character basis");
    let OperatorBody::Dense { present, .. } = &op.body else { panic!("the table is dense") };
    (0..op.cols.group_count())
        .filter(|c| present.column(*c).iter().any(|k| *k) && op.cols.groups()[*c].label.kind == LabelKind::Plane)
        .map(|c| op.cols.groups()[c].label.index)
        .collect()
}

#[test]
fn a_declared_cycle_exposes_the_planted_plane_and_drops_the_rest() {
    let (program, contract) = planted_program(true);
    let library: Vec<Box<dyn Primitive>> = vec![Box::new(DeclaredCharacters), Box::new(DropBlocks), Box::new(Coarsen)];
    let result = decompose(&program, &contract, &library, &budget()).expect("decomposes");
    assert!(result.score.proven_shorter_than(&contract.score(&program, &contract.logits(&program).expect("reference")).expect("native score")));
    assert!(result.score.program_bits < program.code_bits().expect("native bits"));
    assert_eq!(kept_planes(&result.program), vec![PLANTED as u32]);
    assert!(result.program.bases.iter().any(|b| matches!(b, Basis::Characters { declared: true, .. })));
}

#[test]
fn duplicating_a_component_never_shortens_the_program_or_changes_the_verdict() {
    let (program, contract) = planted_program(true);
    let reference = contract.logits(&program).expect("executes");
    let base = contract.score(&program, &reference).expect("score");
    for copies in [2usize, 3] {
        let mut duplicated = program.clone();
        let original = duplicated.operators[1].clone();
        let OperatorBody::Dense { values, precision: q, .. } = &original.body else { panic!("dense") };
        // `copies` operators whose sum is the original: each is the original over `copies`, exact on
        // a lattice `⌈log₂ copies⌉` bits finer when `copies` is a power of two, and otherwise the
        // residual carried by the first copy.
        let finer = precision(q.fraction_bits() + 2);
        let share = values.mapv(|v| v / copies as f64);
        let mut parts = Vec::new();
        let mut total = Array2::<f64>::zeros(values.dim());
        for copy in 1..copies {
            let part = Operator::dense(format!("U{copy}"), original.rows.clone(), original.cols.clone(), share.clone(), finer, Provenance::default())
                .expect("dense");
            total += &part.matrix();
            parts.push(part);
        }
        let first = Operator::dense("U0", original.rows.clone(), original.cols.clone(), values - &total, finer, Provenance::default())
            .expect("dense");
        parts.insert(0, first);
        let start = duplicated.operators.len();
        duplicated.operators.extend(parts.into_iter().map(Arc::new));
        duplicated.nodes[3] = Node::Affine { terms: (0..copies).map(|i| (2, start + i)).collect(), bias: None };
        duplicated.prune();
        let fidelity = contract.score(&duplicated, &reference).expect("score");
        assert_eq!(fidelity.evaluation.argmax_disagreements, base.evaluation.argmax_disagreements);
        assert!(fidelity.total() > base.total(), "{copies} copies: {} bits against {}", fidelity.total(), base.total());
    }
}

#[test]
fn rescaling_a_component_by_a_power_of_two_changes_only_the_precision_field() {
    let (program, contract) = planted_program(true);
    let reference = contract.logits(&program).expect("executes");
    let base = contract.score(&program, &reference).expect("score");
    let mut rescaled = program.clone();
    let (e, u) = (rescaled.operators[0].clone(), rescaled.operators[1].clone());
    let OperatorBody::Dense { precision: pe, .. } = e.body.clone() else { panic!("dense") };
    let OperatorBody::Dense { precision: pu, .. } = u.body.clone() else { panic!("dense") };
    rescaled.operators[0] = Arc::new(
        Operator::dense("E", e.rows.clone(), e.cols.clone(), e.matrix() * 4.0, precision(pe.fraction_bits() - 2), Provenance::default())
            .expect("dense"),
    );
    rescaled.operators[1] = Arc::new(
        Operator::dense("U", u.rows.clone(), u.cols.clone(), u.matrix() * 0.25, precision(pu.fraction_bits() + 2), Provenance::default())
            .expect("dense"),
    );
    let fidelity = contract.score(&rescaled, &reference).expect("score");
    assert_eq!(fidelity.evaluation.argmax_disagreements, base.evaluation.argmax_disagreements);
    // The lattice indices are identical; only the two precision codewords can differ in length.
    let bits_of = |p: i32| super::codec::signed_prefix_integer_len_bits(i64::from(p)).expect("length");
    let field = |a: i32, b: i32| bits_of(a) + bits_of(b);
    let expected = base.program_bits + field(pe.fraction_bits() - 2, pu.fraction_bits() + 2) - field(pe.fraction_bits(), pu.fraction_bits());
    assert_eq!(fidelity.program_bits, expected);
}

#[test]
fn a_finer_partition_of_an_interface_changes_only_the_structure_code() {
    let (program, contract) = planted_program(true);
    let reference = contract.logits(&program).expect("executes");
    let base = contract.score(&program, &reference).expect("score");
    let mut split = program.clone();
    let e = split.operators[0].clone();
    let u = split.operators[1].clone();
    let OperatorBody::Dense { precision: pe, .. } = e.body.clone() else { panic!("dense") };
    let OperatorBody::Dense { precision: pu, .. } = u.body.clone() else { panic!("dense") };
    let units = Interface::uniform(WIDTH, 1, LabelKind::Unit, 0).expect("interface");
    split.operators[0] = Arc::new(Operator::dense("E", units.clone(), e.cols.clone(), e.matrix(), pe, Provenance::default()).expect("dense"));
    split.operators[1] = Arc::new(Operator::dense("U", u.rows.clone(), units, u.matrix(), pu, Provenance::default()).expect("dense"));
    let fidelity = contract.score(&split, &reference).expect("score");
    assert_eq!(fidelity.evaluation.argmax_disagreements, base.evaluation.argmax_disagreements);
    assert_eq!(fidelity.evaluation.max_kl, base.evaluation.max_kl);
    let reals = |p: &OperatorProgram| p.code_account().expect("account").operator_bits.iter().map(|(_, r)| *r).sum::<u64>();
    assert_eq!(reals(&split), reals(&program));
}

#[test]
fn a_warm_start_never_leaves_the_result_longer_than_the_native_start() {
    let (program, contract) = planted_program(true);
    let reference = contract.logits(&program).expect("executes");
    let native = contract.score(&program, &reference).expect("native score");
    // A start chosen for little behaviour: the readout removed, every input read as uniform. At
    // REPEATS observations per input it is far longer than the native program, and a search that
    // only restricts can never put the readout back.
    let mut empty = program.clone();
    let OperatorBody::Dense { values, present, .. } = &mut Arc::make_mut(&mut empty.operators[1]).body else { panic!("dense") };
    present.fill(false);
    values.fill(0.0);
    let warm = contract.score(&empty, &reference).expect("warm score");
    assert!(native.proven_shorter_than(&warm), "the warm start {} must be longer than native {}", warm.total(), native.total());
    let library: Vec<Box<dyn Primitive>> = vec![Box::new(DropBlocks), Box::new(Coarsen)];
    let from_native = decompose(&program, &contract, &library, &budget()).expect("decomposes");
    let from_warm = decompose_from(&program, &empty, &contract, &library, &budget()).expect("decomposes");
    // Never proven longer than the native program, nor than the native start's own search.
    assert!(!native.proven_shorter_than(&from_warm.score), "{} against native {}", from_warm.score.total(), native.total());
    assert!(
        !from_native.score.proven_shorter_than(&from_warm.score),
        "warm-started {} against the native start's {}",
        from_warm.score.total(),
        from_native.score.total()
    );
}

#[test]
fn rounding_a_program_leaves_its_structure_bits_unchanged() {
    let (program, contract) = planted_program(true);
    let reference = contract.logits(&program).expect("executes");
    let base = contract.score(&program, &reference).expect("score");
    for fraction_bits in [12, 4, 1, -2] {
        let mut rounded = program.clone();
        for operator in 0..rounded.operators.len() {
            super::engine::apply_edit(&mut rounded, &Edit::Precision { operator, precision: precision(fraction_bits) }).expect("rounds");
        }
        let score = contract.score(&rounded, &reference).expect("score");
        assert_eq!(score.structure_bits, base.structure_bits, "at 2^-{fraction_bits}");
        assert_eq!(score.structure_bits + score.precision_bits, score.program_bits);
        assert!(score.precision_bits < base.precision_bits);
    }
}

#[test]
fn the_explanations_list_each_inputs_active_units_in_the_librarys_code() {
    let (mut program, contract) = planted_program(true);
    let units = Interface::uniform(WIDTH, 1, LabelKind::Unit, 0).expect("interface");
    let (e, u) = (program.operators[0].clone(), program.operators[1].clone());
    let shifted = e.matrix() - 0.5;
    program.operators[0] = Arc::new(Operator::dense("E", units.clone(), e.cols.clone(), shifted, precision(30), Provenance::default()).expect("dense"));
    program.operators[1] = Arc::new(Operator::dense("U", u.rows.clone(), units, u.matrix(), precision(30), Provenance::default()).expect("dense"));
    program.nodes = vec![
        Node::Feature { slot: 0, basis: 0 },
        Node::Feature { slot: 1, basis: 0 },
        Node::Affine { terms: vec![(0, 0), (1, 0)], bias: None },
        Node::Pointwise { input: 2, laws: vec![super::operator_program::Law::Relu; WIDTH] },
        Node::Affine { terms: vec![(3, 1)], bias: None },
        Node::Readout { input: 4, basis: 1 },
    ];
    program.output = 5;
    let reference = contract.logits(&program).expect("executes");
    let score = contract.score(&program, &reference).expect("score");
    let pre = &program.execute(&contract.family, false).expect("executes").values[2];
    let n = pre.nrows() as f64;
    // The listing code, term by term: the counts once, then per input its size and its set.
    let counts: Vec<usize> = (0..WIDTH).map(|unit| pre.column(unit).iter().filter(|v| **v > 0.0).count()).collect();
    let active: usize = counts.iter().sum();
    let t = active as f64;
    let r = WIDTH as f64;
    let mut expected = r * (n + 1.0).log2() + n * (r + 1.0).log2();
    for row in pre.outer_iter() {
        let mut factorial = 1.0_f64;
        for (unit, v) in row.iter().enumerate() {
            if *v > 0.0 {
                expected += (t / counts[unit] as f64).log2();
                factorial *= (row.iter().take(unit + 1).filter(|x| **x > 0.0).count()) as f64;
            }
        }
        expected -= factorial.log2();
    }
    let explanation = &score.explanation;
    assert_eq!(explanation.instances, WIDTH);
    assert!((explanation.bits - expected).abs() <= 1e-9 * expected, "{} against {expected}", explanation.bits);
    assert!(explanation.bits_lower <= explanation.bits && explanation.bits <= explanation.bits_upper);
    let open: u32 = explanation.open.iter().sum();
    assert!((explanation.mean_active() * n - active as f64).abs() <= f64::from(open) + 1e-9);
    assert!((score.total() - (score.program_bits as f64 + explanation.bits + score.data_bits)).abs() < 1e-6);
}

/// A group drop priced from the per-block table changes the program's message by exactly what
/// re-encoding the dropped program measures, for every row and column group of every operator.
#[test]
fn a_group_drop_is_priced_exactly_from_the_block_table() {
    use super::engine::{BlockBits, GroupAxis, apply_edit};
    let (program, _) = planted_program(false);
    let before = program.code_bits().expect("bits") as i64;
    for (index, op) in program.operators.iter().enumerate() {
        let OperatorBody::Dense { present, .. } = &op.body else { continue };
        let table = BlockBits::of(op).expect("table").expect("dense");
        for (axis, count) in [(GroupAxis::Columns, present.ncols()), (GroupAxis::Rows, present.nrows())] {
            for group in 0..count {
                let blocks: Vec<(usize, usize)> = present
                    .indexed_iter()
                    .filter(|((r, c), keep)| **keep && if axis == GroupAxis::Columns { *c == group } else { *r == group })
                    .map(|(rc, _)| rc)
                    .collect();
                if blocks.is_empty() {
                    continue;
                }
                let mut dropped = program.clone();
                apply_edit(&mut dropped, &Edit::DropGroup { operator: index, axis, group }).expect("drop");
                let after = dropped.code_bits().expect("bits") as i64;
                assert_eq!(table.drop_delta(&blocks).expect("delta"), after - before, "{} {axis:?} {group}", op.name);
            }
        }
    }
}

/// A candidate's message length measured from its base (changed operators and frame only) is
/// its full length, for restrictions, precision moves and law replacements.
#[test]
fn a_candidates_length_from_its_base_is_its_length() {
    use super::engine::{GroupAxis, apply_edit};
    use super::operator_program::Law;
    let (program, _) = planted_program(false);
    let base = program.code_bits().expect("bits");
    let dense = program.operators.iter().position(|o| matches!(o.body, OperatorBody::Dense { .. })).expect("dense");
    let mut edits = vec![
        Edit::DropGroup { operator: dense, axis: GroupAxis::Columns, group: 0 },
        Edit::Precision { operator: dense, precision: precision(3) },
    ];
    if let Some((node, count)) = program.nodes.iter().enumerate().find_map(|(i, n)| match n {
        Node::Pointwise { laws, .. } => Some((i, laws.len())),
        Node::Feature { .. }
        | Node::Raw { .. }
        | Node::Constant { .. }
        | Node::Affine { .. }
        | Node::Bilinear { .. }
        | Node::Softmax { .. }
        | Node::Mix { .. }
        | Node::Hadamard { .. }
        | Node::Readout { .. }
        | Node::Outer { .. }
        | Node::Concat { .. }
        | Node::Param { .. }
        | Node::Call { .. }
        | Node::Gain { .. }
        | Node::Attend { .. }
        | Node::RmsNorm { .. }
        | Node::Transposed { .. } => None,
    }) {
        edits.push(Edit::Laws { node, laws: vec![Law::Identity; count], blocks: Vec::new() });
    }
    for edit in edits {
        let mut candidate = program.clone();
        apply_edit(&mut candidate, &edit).expect("edit");
        assert_eq!(
            candidate.code_bits_from(&program, base, &mut std::collections::BTreeMap::new()).expect("from base"),
            candidate.code_bits().expect("bits"),
            "{edit:?}"
        );
    }
}

/// A raw slot of width 2 read by one dense map `W` into two classes, with three inputs.
fn raw_readout(w: Array2<f64>) -> (OperatorProgram, Contract) {
    let declarations = Declarations { parameters: 0, domains: vec![], slots: vec![Slot::Raw { width: 2 }] };
    let native = Interface::native(2).expect("interface");
    let program = OperatorProgram {
        rules: Vec::new(),
        declarations: declarations.clone(),
        bases: vec![],
        operators: vec![Arc::new(
            Operator::dense("W", native.clone(), native, w, precision(30), Provenance::native("W")).expect("dense"),
        )],
        nodes: vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0, 0)], bias: None }],
        output: 1,
    };
    let x = Array2::from_shape_fn((3, 2), |(i, j)| 0.37 + 0.11 * i as f64 - 0.29 * j as f64);
    let contract = Contract {
        declarations,
        family: FamilyInputs { layout: None, rows: 3, slots: vec![SlotValues::Raw(x)] },
        kind: FamilyKind::Complete { description: "three inputs".to_string() },
        observations: 1,
        readouts: 1,
        readout_slots: None,
    };
    (program, contract)
}

/// A row whose proven band is wider than one logit unit keeps that band: the sampled rounding
/// estimate is a diagnostic and never replaces an enclosure, so nothing the
/// score certifies rests on it.
#[test]
fn a_wide_proven_band_is_never_replaced_by_the_rounding_estimate() {
    use super::contract::measured_output_band;
    let w = Array2::from_shape_fn((2, 2), |(i, j)| ((i * 2 + j) as f64 * 0.7 + 0.3).sin());
    // Logits near 1e18: the forward-error enclosure is hundreds of logit units wide.
    let (large, contract) = raw_readout(w.mapv(|v| v * 1e18));
    let trace = large.execute(&contract.family, true).expect("executes");
    let proven = trace.band(large.output).expect("bands");
    assert!(proven.iter().all(|r| *r >= 1.0), "{proven:?}");
    let estimate = measured_output_band(&large, &contract.family, &trace).expect("estimate");
    assert_ne!(estimate, proven);
    let logits = contract.logits(&large).expect("logits");
    assert_eq!(logits.bands, proven);
    let score = contract.score(&large, &logits).expect("score");
    // The certified data term rests on the proven boxes only: its error covers their width.
    assert!(score.data_bits_error.is_infinite() || score.data_bits_error > 0.0);
}

/// A call is counted as its body inlined: the same ReLU layer written inline and wrapped in a rule
/// has the same instances, activities and explanation bits.
#[test]
fn a_call_counts_the_activity_of_its_body_as_inlined() {
    use super::contract::explanation;
    use super::operator_program::{Law, Rule};
    let declarations = Declarations { parameters: 0, domains: vec![], slots: vec![Slot::Raw { width: 3 }] };
    let native = Interface::native(3).expect("interface");
    let units = Interface::uniform(2, 1, LabelKind::Unit, 0).expect("interface");
    let values = Array2::from_shape_fn((2, 3), |(i, j)| ((i * 3 + j) as f64 * 0.7 + 0.2).sin());
    let w = Arc::new(Operator::dense("W", units, native.clone(), values, precision(30), Provenance::native("W")).expect("dense"));
    let layer = |input: usize| {
        vec![
            Node::Affine { terms: vec![(input, 0)], bias: None },
            Node::Pointwise { input: input + 1, laws: vec![Law::Relu, Law::Relu] },
        ]
    };
    let inlined = OperatorProgram {
        rules: Vec::new(),
        declarations: declarations.clone(),
        bases: vec![],
        operators: vec![w.clone()],
        nodes: std::iter::once(Node::Raw { slot: 0 }).chain(layer(0)).collect(),
        output: 2,
    };
    let rule = Rule {
        name: "layer".to_string(),
        inputs: vec![native],
        nodes: std::iter::once(Node::Param { index: 0 }).chain(layer(0)).collect(),
        output: 2,
    };
    let called = OperatorProgram {
        rules: vec![rule],
        declarations: declarations.clone(),
        bases: vec![],
        operators: vec![w],
        nodes: vec![Node::Raw { slot: 0 }, Node::Call { rule: 0, arguments: vec![0] }],
        output: 1,
    };
    // Pre-activations well away from zero, of both signs.
    let x = Array2::from_shape_fn((6, 3), |(i, j)| (if (i + j) % 3 == 0 { 1.5 } else { -0.75 }) * (1.0 + 0.25 * i as f64));
    let family = FamilyInputs { layout: None, rows: 6, slots: vec![SlotValues::Raw(x)] };
    let explain = |program: &OperatorProgram| {
        let trace = program.execute(&family, true).expect("executes");
        explanation(program, &family, &trace).expect("explanation")
    };
    let (inline, call) = (explain(&inlined), explain(&called));
    assert_eq!(inline.instances, 2);
    assert!(inline.active.iter().any(|a| *a > 0) && inline.active.iter().any(|a| *a < 2), "{:?}", inline.active);
    assert_eq!(inline, call);
    let contract = Contract {
        declarations,
        family,
        kind: FamilyKind::Complete { description: "six inputs".to_string() },
        observations: 1,
        readouts: 1,
        readout_slots: None,
    };
    let reference = contract.logits(&inlined).expect("reference");
    let (a, b) = (contract.score(&inlined, &reference).expect("inline"), contract.score(&called, &reference).expect("call"));
    assert_eq!(a.explanation, b.explanation);
}
