#![cfg(test)]
//! The engine on planted programs: exact structure is found and certified, a recovered labelling
//! is the planted cycle up to an automorphism, and restating a program (duplicating or rescaling a
//! component, re-partitioning an interface) never makes it look shorter.

use super::contract::{Contract, FamilyKind};
use super::engine::{Budget, Coarsen, DropBlocks, Primitive, decompose};
use super::operator_program::{
    Basis, Declarations, Domain, FamilyInputs, Interface, LabelKind, Node, Operator, OperatorBody, OperatorProgram,
    Provenance, Slot, SlotValues,
};
use super::engine::{EngineError, Edit, Exactness, Proposal, SearchContext};
use super::fit::ProposalKind;
use super::operator_rewrites::{PlaneBasis, change_basis};
use super::precision::DeclaredPrecision;
use ndarray::Array2;
use std::f64::consts::TAU;

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
        operators,
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
fn a_recovered_cycle_is_the_planted_one_up_to_an_automorphism() {
    let (program, contract) = planted_program(false);
    let library: Vec<Box<dyn Primitive>> = vec![Box::new(PlaneBasis), Box::new(DropBlocks), Box::new(Coarsen)];
    let result = decompose(&program, &contract, &library, &budget()).expect("decomposes");
    assert!(result.score.proven_shorter_than(&contract.score(&program, &contract.logits(&program).expect("reference")).expect("native score")));
    let positions = result
        .program
        .bases
        .iter()
        .find_map(|b| match b {
            Basis::Characters { positions, declared: false, .. } => Some(positions.clone()),
            _ => None,
        })
        .expect("a recovered character basis was accepted");
    let a0 = positions[0].expect("token 0 is on the cycle") as usize;
    let unit = (1..P)
        .find(|&k| (0..P).all(|x| positions[x] == Some(((k * x + a0) % P) as u32)))
        .expect("the recovered labelling is an affine relabelling of x -> x + 1");
    // One plane survives, and it is the planted frequency read in the recovered labelling.
    let planes = kept_planes(&result.program);
    assert_eq!(planes.len(), 1);
    let read = planes[0] as usize;
    assert!((read * unit) % P == PLANTED || (read * unit) % P == P - PLANTED, "plane {read} under unit {unit}");
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
        duplicated.operators.extend(parts);
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
    let OperatorBody::Dense { precision: pe, .. } = e.body else { panic!("dense") };
    let OperatorBody::Dense { precision: pu, .. } = u.body else { panic!("dense") };
    rescaled.operators[0] =
        Operator::dense("E", e.rows.clone(), e.cols.clone(), e.matrix() * 4.0, precision(pe.fraction_bits() - 2), Provenance::default())
            .expect("dense");
    rescaled.operators[1] =
        Operator::dense("U", u.rows.clone(), u.cols.clone(), u.matrix() * 0.25, precision(pu.fraction_bits() + 2), Provenance::default())
            .expect("dense");
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
    let OperatorBody::Dense { precision: pe, .. } = e.body else { panic!("dense") };
    let OperatorBody::Dense { precision: pu, .. } = u.body else { panic!("dense") };
    let units = Interface::uniform(WIDTH, 1, LabelKind::Unit, 0).expect("interface");
    split.operators[0] = Operator::dense("E", units.clone(), e.cols.clone(), e.matrix(), pe, Provenance::default()).expect("dense");
    split.operators[1] = Operator::dense("U", u.rows.clone(), units, u.matrix(), pu, Provenance::default()).expect("dense");
    let fidelity = contract.score(&split, &reference).expect("score");
    assert_eq!(fidelity.evaluation.argmax_disagreements, base.evaluation.argmax_disagreements);
    assert_eq!(fidelity.evaluation.max_kl, base.evaluation.max_kl);
    let reals = |p: &OperatorProgram| p.code_account().expect("account").operator_bits.iter().map(|(_, r)| *r).sum::<u64>();
    assert_eq!(reals(&split), reals(&program));
}
