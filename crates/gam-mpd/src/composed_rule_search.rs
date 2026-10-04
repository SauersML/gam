//! Finite structural proposals for shared vector computations.
//!
//! The grammar chooses operations and their operands; every use learns its readers and
//! writer jointly with any affine operations inside the shared rule. Width is explicit:
//! full-dimensional payloads are allowed. This is a proposal language, not a claim that
//! native algorithms are in this grammar or that fitting finds its optimum. In particular,
//! it has no temporal state or attention primitive. Native bindings and autonomous fidelity
//! must be checked after importing a proposal into the complete native artifact.
use crate::operator_program::{
    Coefficient, Declarations, Interface, Law, Node, Operator, OperatorProgram, Rule, Slot,
    exact_precision,
};
use crate::resident_rule_fit::OutputGroup;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unary {
    Relu,
    Silu,
    Gelu,
    GeluTanh,
}
impl Unary {
    fn law(self) -> Law {
        match self {
            Self::Relu => Law::Relu,
            Self::Silu => Law::Silu,
            Self::Gelu => Law::Gelu,
            Self::GeluTanh => Law::GeluTanh,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Binary {
    Add,
    Subtract,
    Multiply,
}

/// A tree describes construction history; repeated identical subexpressions execute once.
/// Repeated `Affine` expressions share coefficients within a body. Different argument
/// indices always have independently learned readers, even at the same native input.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Expr {
    Argument(usize),
    Unary(Unary, Box<Expr>),
    Binary(Binary, Box<Expr>, Box<Expr>),
    Affine(Box<Expr>),
}
impl Expr {
    fn arguments(&self, out: &mut BTreeSet<usize>) {
        match self {
            Self::Argument(i) => {
                out.insert(*i);
            }
            Self::Unary(_, x) | Self::Affine(x) => x.arguments(out),
            Self::Binary(_, a, b) => {
                a.arguments(out);
                b.arguments(out);
            }
        }
    }
    pub fn arity(&self) -> Result<usize, String> {
        let mut used = BTreeSet::new();
        self.arguments(&mut used);
        if used.iter().copied().ne(0..used.len()) {
            return Err("expression arguments must be contiguous from zero".into());
        }
        Ok(used.len())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Grammar {
    pub arguments: usize,
    /// Maximum number of operations in the expression tree (before common subexpressions).
    pub max_operations: usize,
    /// Bounds ALL intermediate expressions, including unused argument renamings.
    pub max_expressions: usize,
    pub unary: Vec<Unary>,
    pub binary: Vec<Binary>,
    /// Include a learned full-width affine operation at any expression position.
    pub affine: bool,
}
#[derive(Clone, Debug, Serialize)]
pub struct Inventory {
    pub expressions: Vec<Expr>,
    pub truncated: bool,
    pub intermediate_expressions: usize,
}

/// Syntactic opportunities for learned sharing, not identification of a native law.
/// Redundancy flags mark syntactic boundary/adjacency patterns in unrestricted-map
/// parameterizations. Shared subexpressions can prevent absorption, so flags are
/// warnings rather than proofs of redundancy. They establish neither binary64
/// equivalence nor equivalence under internal controls.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SharingAnalysis {
    /// Unique Affine expressions, matching the compiler's exact-expression CSE.
    pub learned_affine_expressions: usize,
    pub output_boundary_affine: bool,
    pub input_boundary_affine: bool,
    pub boundary_affine_redundancy: bool,
    pub adjacent_affine_redundancy: bool,
    /// Unique learned affine expressions with nonlinear ancestors and descendants.
    pub interior_learned_affine_expressions: usize,
    pub has_interior_learned_sharing: bool,
}
fn nonlinear_subtree(expr: &Expr) -> bool {
    match expr {
        Expr::Argument(_) => false,
        Expr::Unary(_, _) | Expr::Binary(Binary::Multiply, _, _) => true,
        Expr::Affine(input) => nonlinear_subtree(input),
        Expr::Binary(_, left, right) => nonlinear_subtree(left) || nonlinear_subtree(right),
    }
}
/// Analyze the expression without rewriting, fitting, or changing its parameterization.
pub fn sharing_analysis(expr: &Expr) -> SharingAnalysis {
    fn visit(
        expr: &Expr,
        nonlinear_consumer: bool,
        affines: &mut BTreeMap<Expr, bool>,
        input_boundary: &mut bool,
        adjacent: &mut bool,
    ) {
        match expr {
            Expr::Argument(_) => return,
            Expr::Affine(input) => {
                let interior = nonlinear_consumer && nonlinear_subtree(input);
                affines
                    .entry(expr.clone())
                    .and_modify(|present| *present |= interior)
                    .or_insert(interior);
                *input_boundary |= matches!(input.as_ref(), Expr::Argument(_));
                *adjacent |= matches!(input.as_ref(), Expr::Affine(_));
                visit(input, nonlinear_consumer, affines, input_boundary, adjacent);
            }
            Expr::Unary(_, input) => visit(input, true, affines, input_boundary, adjacent),
            Expr::Binary(op, left, right) => {
                let nonlinear_consumer = nonlinear_consumer || *op == Binary::Multiply;
                visit(left, nonlinear_consumer, affines, input_boundary, adjacent);
                visit(right, nonlinear_consumer, affines, input_boundary, adjacent);
            }
        }
    }
    let mut affines = BTreeMap::new();
    let mut input_boundary = false;
    let mut adjacent = false;
    visit(
        expr,
        false,
        &mut affines,
        &mut input_boundary,
        &mut adjacent,
    );
    let interior = affines.values().filter(|value| **value).count();
    let output_boundary = matches!(expr, Expr::Affine(_));
    SharingAnalysis {
        learned_affine_expressions: affines.len(),
        output_boundary_affine: output_boundary,
        input_boundary_affine: input_boundary,
        boundary_affine_redundancy: output_boundary || input_boundary,
        adjacent_affine_redundancy: adjacent,
        interior_learned_affine_expressions: interior,
        has_interior_learned_sharing: interior != 0,
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct SharingSelection {
    /// Original inventory IDs; neither enumeration nor IDs are renumbered.
    pub interior_indices: Vec<usize>,
    /// Complement available as separate non-affine/boundary and other baselines.
    pub baseline_indices: Vec<usize>,
    /// Analysis in the original inventory order, including every excluded expression.
    pub analyses: Vec<SharingAnalysis>,
    pub source_inventory_truncated: bool,
    pub scope: &'static str,
}
/// Optional declared subinventory. The only selection predicate is the existence of
/// an affine with nonlinear computation before it and on a path from it to output.
/// Adjacent/boundary flags remain reported, not silently used as additional exclusions.
pub fn interior_learned_selection(inventory: &Inventory) -> SharingSelection {
    let analyses: Vec<_> = inventory.expressions.iter().map(sharing_analysis).collect();
    let (interior_indices, baseline_indices) =
        (0..analyses.len()).partition(|index| analyses[*index].has_interior_learned_sharing);
    SharingSelection {
        interior_indices,
        baseline_indices,
        analyses,
        source_inventory_truncated: inventory.truncated,
        scope: "Declared syntactic interior-learned-sharing subinventory only. Every excluded original ID remains a baseline; no program rewriting, algebraic identification, fidelity pruning, or guarantee of reusable law. Source truncation remains explicit.",
    }
}

/// Breadth by expression size; products can be proposed directly even if neither factor
/// alone improves a fit. Only operand exchange for commutative primitives is canonicalized;
/// there are no approximate algebraic equalities or fit-dependent pruning here.
pub fn enumerate(grammar: &Grammar) -> Result<Inventory, String> {
    if grammar.arguments == 0 || grammar.max_expressions < grammar.arguments {
        return Err("grammar needs arguments and an intermediate budget covering them".into());
    }
    let unary: BTreeSet<_> = grammar.unary.iter().copied().collect();
    let binary: BTreeSet<_> = grammar.binary.iter().copied().collect();
    let mut by_size = vec![
        (0..grammar.arguments)
            .map(Expr::Argument)
            .collect::<Vec<_>>(),
    ];
    let mut seen: BTreeSet<_> = by_size[0].iter().cloned().collect();
    let mut truncated = false;
    'sizes: for size in 1..=grammar.max_operations {
        let mut next = Vec::new();
        let mut add = |expr: Expr| {
            if seen.contains(&expr) {
                return true;
            }
            if seen.len() == grammar.max_expressions {
                return false;
            }
            seen.insert(expr.clone());
            next.push(expr);
            true
        };
        for x in &by_size[size - 1] {
            for op in &unary {
                if !add(Expr::Unary(*op, Box::new(x.clone()))) {
                    truncated = true;
                    by_size.push(next);
                    break 'sizes;
                }
            }
            if grammar.affine && !add(Expr::Affine(Box::new(x.clone()))) {
                truncated = true;
                by_size.push(next);
                break 'sizes;
            }
        }
        for left_size in 0..size {
            let right_size = size - 1 - left_size;
            for a in &by_size[left_size] {
                for b in &by_size[right_size] {
                    for op in &binary {
                        if *op != Binary::Subtract && a > b {
                            continue;
                        }
                        if !add(Expr::Binary(*op, Box::new(a.clone()), Box::new(b.clone()))) {
                            truncated = true;
                            by_size.push(next);
                            break 'sizes;
                        }
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        by_size.push(next);
    }
    Ok(Inventory {
        expressions: by_size
            .into_iter()
            .flatten()
            .filter(|x| x.arity().is_ok())
            .collect(),
        truncated,
        intermediate_expressions: seen.len(),
    })
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct UseSpec {
    pub input_width: usize,
    pub output_width: usize,
}
pub struct Proposal {
    pub program: OperatorProgram,
    pub trainable: Vec<usize>,
    pub groups: Vec<OutputGroup>,
}

struct Random(u64);
impl Random {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        2. * ((z >> 11) as f64 / 9007199254740992.) - 1.
    }
}
struct Builder {
    interface: Interface,
    operators: Vec<Arc<Operator>>,
    trainable: Vec<usize>,
    positive_identity: Option<usize>,
    random: Random,
}
impl Builder {
    fn dense(
        &mut self,
        rows: Interface,
        cols: Interface,
        bias: bool,
        name: String,
    ) -> Result<usize, String> {
        rows.width()
            .checked_mul(cols.width())
            .ok_or("operator shape overflow")?;
        let amplitude = 0.1 / (cols.width() as f64).sqrt();
        let values = Array2::from_shape_fn((rows.width(), cols.width()), |(i, j)| {
            if bias {
                0.
            } else {
                // Distinct nonzero readers keep a product's two branches trainable.
                let identity = if i == j { 1. } else { 0. };
                identity + amplitude * self.random.unit()
            }
        });
        let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
        let index = self.operators.len();
        self.operators.push(Arc::new(
            Operator::dense(name, rows, cols, values, precision, Default::default())
                .map_err(|e| e.to_string())?,
        ));
        self.trainable.push(index);
        Ok(index)
    }
    fn identity(&mut self) -> usize {
        if let Some(index) = self.positive_identity {
            return index;
        }
        let index = self.operators.len();
        self.operators.push(Arc::new(Operator::identity(
            "addition",
            self.interface.clone(),
        )));
        self.positive_identity = Some(index);
        index
    }
    fn expr(
        &mut self,
        expr: &Expr,
        nodes: &mut Vec<Node>,
        cached: &mut BTreeMap<Expr, usize>,
    ) -> Result<usize, String> {
        if let Some(index) = cached.get(expr) {
            return Ok(*index);
        }
        let node = match expr {
            Expr::Argument(index) => Node::Param { index: *index },
            Expr::Unary(op, input) => Node::Pointwise {
                input: self.expr(input, nodes, cached)?,
                laws: vec![op.law()],
            },
            Expr::Affine(input) => {
                let input = self.expr(input, nodes, cached)?;
                let matrix = self.dense(
                    self.interface.clone(),
                    self.interface.clone(),
                    false,
                    "shared internal affine matrix".into(),
                )?;
                let bias = self.dense(
                    self.interface.clone(),
                    Interface::constant(),
                    true,
                    "shared internal affine offset".into(),
                )?;
                Node::Affine {
                    terms: vec![(input, matrix)],
                    bias: Some(bias),
                }
            }
            Expr::Binary(op, a, b) => {
                let left = self.expr(a, nodes, cached)?;
                let right = self.expr(b, nodes, cached)?;
                match op {
                    Binary::Multiply => Node::Hadamard { left, right },
                    Binary::Add | Binary::Subtract => {
                        let identity = self.identity();
                        let right = if *op == Binary::Subtract {
                            let index = nodes.len();
                            nodes.push(Node::Gain {
                                input: right,
                                coefficient: Coefficient::Number(-1.),
                            });
                            index
                        } else {
                            right
                        };
                        Node::Affine {
                            terms: vec![(left, identity), (right, identity)],
                            bias: None,
                        }
                    }
                }
            }
        };
        let index = nodes.len();
        nodes.push(node);
        cached.insert(expr.clone(), index);
        Ok(index)
    }
    fn rule(&mut self, expr: &Expr, arity: usize) -> Result<Rule, String> {
        let mut nodes = Vec::new();
        let output = self.expr(expr, &mut nodes, &mut BTreeMap::new())?;
        Ok(Rule {
            name: "searched vector expression".into(),
            inputs: vec![self.interface.clone(); arity],
            nodes,
            output,
        })
    }
}

pub fn compile(expr: &Expr, width: usize, uses: &[UseSpec], seed: u64) -> Result<Proposal, String> {
    compile_mode(expr, width, uses, seed, true)
}

/// One use as a one-input function suitable for `Artifact::replace_function`.
/// Shared operator Arcs and callable bodies remain shared across exported uses.
pub fn function(proposal: &Proposal, slot: usize) -> Result<OperatorProgram, String> {
    let Node::Concat { parts } = &proposal.program.nodes[proposal.program.output] else {
        return Err("composed proposal requires per-use concatenated outputs".into());
    };
    let output = *parts.get(slot).ok_or("composed use is absent")?;
    let Some(Slot::Raw { width }) = proposal.program.declarations.slots.get(slot) else {
        return Err("composed use requires a raw input".into());
    };
    let mut live = vec![false; proposal.program.nodes.len()];
    let mut stack = vec![output];
    while let Some(node) = stack.pop() {
        if live[node] {
            continue;
        }
        live[node] = true;
        stack.extend(proposal.program.nodes[node].arguments());
    }
    let mut mapping = vec![usize::MAX; live.len()];
    let mut nodes = Vec::new();
    for (old, node) in proposal.program.nodes.iter().enumerate() {
        if live[old] {
            mapping[old] = nodes.len();
            nodes.push(node.clone());
        }
    }
    let ops: Vec<_> = (0..proposal.program.operators.len()).collect();
    let rules: Vec<_> = (0..proposal.program.rules.len()).collect();
    for node in &mut nodes {
        crate::operator_program::remap_node(node, &mapping, &ops, &[], &rules);
        if let Node::Raw { slot: source_slot } = node {
            if *source_slot != slot {
                return Err("function crosses another use's input".into());
            }
            *source_slot = 0;
        }
    }
    let mut program = proposal.program.clone();
    program.declarations.slots = vec![Slot::Raw { width: *width }];
    program.nodes = nodes;
    program.output = mapping[output];
    program.interfaces().map_err(|e| e.to_string())?;
    Ok(program)
}
/// Same initial numerical function and external maps as `compile`, but body coefficients
/// can move independently at every use. It is a fitting/control comparison, not a claim
/// that sharing is beneficial. Shared versus separate costs come from the actual artifact.
pub fn compile_untied(
    expr: &Expr,
    width: usize,
    uses: &[UseSpec],
    seed: u64,
) -> Result<Proposal, String> {
    compile_mode(expr, width, uses, seed, false)
}
fn compile_mode(
    expr: &Expr,
    width: usize,
    uses: &[UseSpec],
    seed: u64,
    shared: bool,
) -> Result<Proposal, String> {
    if width == 0
        || uses.is_empty()
        || uses
            .iter()
            .any(|u| u.input_width == 0 || u.output_width == 0)
    {
        return Err("positive vector widths and at least one use required".into());
    }
    let arity = expr.arity()?;
    let interface = Interface::native(width).map_err(|e| e.to_string())?;
    let mut b = Builder {
        interface,
        operators: vec![],
        trainable: vec![],
        positive_identity: None,
        random: Random(seed),
    };
    let original = b.rule(expr, arity)?;
    let body_trainable = b.trainable.clone();
    let mut rules = vec![original.clone()];
    if !shared {
        for _ in 1..uses.len() {
            let mut remap = BTreeMap::new();
            for old in &body_trainable {
                let index = b.operators.len();
                b.operators.push(Arc::new((*b.operators[*old]).clone()));
                b.trainable.push(index);
                remap.insert(*old, index);
            }
            let mut rule = original.clone();
            for node in &mut rule.nodes {
                if let Node::Affine { terms, bias } = node {
                    for (_, op) in terms {
                        *op = remap.get(op).copied().unwrap_or(*op);
                    }
                    if let Some(op) = bias {
                        *op = remap.get(op).copied().unwrap_or(*op);
                    }
                }
            }
            rules.push(rule);
        }
    }
    let mut nodes = Vec::new();
    let mut outputs = Vec::new();
    let mut groups = Vec::new();
    let mut start = 0usize;
    for (use_index, spec) in uses.iter().enumerate() {
        let raw = nodes.len();
        nodes.push(Node::Raw { slot: use_index });
        let mut arguments = Vec::new();
        for argument in 0..arity {
            let reader = b.dense(
                b.interface.clone(),
                Interface::native(spec.input_width).map_err(|e| e.to_string())?,
                false,
                format!("use {use_index} argument {argument} reader"),
            )?;
            let bias = b.dense(
                b.interface.clone(),
                Interface::constant(),
                true,
                format!("use {use_index} argument {argument} offset"),
            )?;
            arguments.push(nodes.len());
            nodes.push(Node::Affine {
                terms: vec![(raw, reader)],
                bias: Some(bias),
            });
        }
        let call = nodes.len();
        nodes.push(Node::Call {
            rule: if shared { 0 } else { use_index },
            arguments,
        });
        let out = Interface::native(spec.output_width).map_err(|e| e.to_string())?;
        let writer = b.dense(
            out.clone(),
            b.interface.clone(),
            false,
            format!("use {use_index} writer"),
        )?;
        let bias = b.dense(
            out,
            Interface::constant(),
            true,
            format!("use {use_index} output offset"),
        )?;
        outputs.push(nodes.len());
        nodes.push(Node::Affine {
            terms: vec![(call, writer)],
            bias: Some(bias),
        });
        let end = start
            .checked_add(spec.output_width)
            .ok_or("output shape overflow")?;
        groups.push(OutputGroup {
            label: format!("use {use_index}"),
            start,
            end,
        });
        start = end;
    }
    let output = nodes.len();
    nodes.push(Node::Concat { parts: outputs });
    let program = OperatorProgram {
        declarations: Declarations {
            domains: vec![],
            slots: uses
                .iter()
                .map(|u| Slot::Raw {
                    width: u.input_width,
                })
                .collect(),
            parameters: 0,
        },
        bases: vec![],
        operators: b.operators,
        rules,
        nodes,
        output,
    };
    program.interfaces().map_err(|e| e.to_string())?;
    Ok(Proposal {
        program,
        trainable: b.trainable,
        groups,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::Artifact,
        operator_program::{FamilyInputs, SlotValues},
    };

    #[test]
    fn compound_products_are_generated_without_singleton_acceptance() {
        let grammar = Grammar {
            arguments: 2,
            max_operations: 2,
            max_expressions: 1000,
            unary: vec![Unary::Relu],
            binary: vec![Binary::Multiply],
            affine: false,
        };
        let bank = enumerate(&grammar).unwrap();
        assert!(!bank.truncated);
        let x = Expr::Argument(0);
        let y = Expr::Argument(1);
        // Enum's ordering makes Argument precede Unary in the canonical commutative pair.
        let wanted = Expr::Binary(
            Binary::Multiply,
            Box::new(y),
            Box::new(Expr::Unary(Unary::Relu, Box::new(x))),
        );
        assert!(bank.expressions.contains(&wanted));
        assert!(bank.expressions.iter().all(|e| e.arity().is_ok()));
        let short = enumerate(&Grammar {
            max_expressions: 3,
            ..grammar
        })
        .unwrap();
        assert!(short.truncated);
        assert_eq!(short.intermediate_expressions, 3);
    }

    #[test]
    fn shared_and_untied_start_identically_but_store_distinct_trainable_bodies() {
        let expr = Expr::Unary(
            Unary::GeluTanh,
            Box::new(Expr::Affine(Box::new(Expr::Argument(0)))),
        );
        let uses = [UseSpec {
            input_width: 5,
            output_width: 5,
        }; 2];
        let shared = compile(&expr, 5, &uses, 31).unwrap();
        let untied = compile_untied(&expr, 5, &uses, 31).unwrap();
        assert_eq!(shared.program.rules.len(), 1);
        assert_eq!(untied.program.rules.len(), 2);
        assert_eq!(untied.trainable.len(), shared.trainable.len() + 2);
        let inputs = FamilyInputs {
            rows: 3,
            slots: (0..2)
                .map(|k| {
                    SlotValues::Raw(Array2::from_shape_fn((3, 5), |(i, j)| {
                        (i * 5 + j + k) as f64 / 7. - 1.
                    }))
                })
                .collect(),
            layout: None,
        };
        let a = shared.program.execute(&inputs, false).unwrap();
        let b = untied.program.execute(&inputs, false).unwrap();
        assert_eq!(
            a.values[shared.program.output],
            b.values[untied.program.output]
        );
        let first = function(&shared, 0).unwrap();
        let second = function(&shared, 1).unwrap();
        for (left, right) in first.operators.iter().zip(&second.operators) {
            assert!(Arc::ptr_eq(left, right));
        }
        for (slot, extracted) in [first, second].iter().enumerate() {
            let input = FamilyInputs {
                rows: inputs.rows,
                slots: vec![inputs.slots[slot].clone()],
                layout: None,
            };
            let trace = extracted.execute(&input, false).unwrap();
            assert_eq!(
                trace.values[extracted.output],
                a.values[shared.program.output].slice(ndarray::s![.., slot * 5..(slot + 1) * 5])
            );
        }
        assert!(function(&shared, 2).is_err());
        let artifact = Artifact::native(&shared.program)
            .unwrap()
            .f32_literals()
            .unwrap();
        let saved = artifact.to_bytes().unwrap();
        let decoded = Artifact::from_bytes(&saved, &shared.program.declarations).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), saved);
    }

    #[test]
    fn generated_programs_execute_and_preserve_repeated_subexpressions() {
        let x = Expr::Unary(Unary::Relu, Box::new(Expr::Argument(0)));
        let expr = Expr::Binary(Binary::Multiply, Box::new(x.clone()), Box::new(x));
        let proposal = compile(
            &expr,
            7,
            &[UseSpec {
                input_width: 7,
                output_width: 7,
            }],
            2,
        )
        .unwrap();
        assert_eq!(proposal.program.rules[0].nodes.len(), 3);
        let bank = enumerate(&Grammar {
            arguments: 2,
            max_operations: 2,
            max_expressions: 1000,
            unary: vec![Unary::Relu, Unary::GeluTanh],
            binary: vec![Binary::Add, Binary::Subtract, Binary::Multiply],
            affine: true,
        })
        .unwrap();
        for expr in bank.expressions {
            let proposal = compile(
                &expr,
                3,
                &[UseSpec {
                    input_width: 4,
                    output_width: 2,
                }; 2],
                3,
            )
            .unwrap();
            let inputs = FamilyInputs {
                rows: 2,
                slots: vec![SlotValues::Raw(Array2::ones((2, 4))); 2],
                layout: None,
            };
            let trace = proposal.program.execute(&inputs, false).unwrap();
            assert_eq!(trace.values[proposal.program.output].dim(), (2, 4));
        }
    }
    #[test]
    fn interior_sharing_distinguishes_pilot_forms_without_changing_inventory() {
        let mut grammar = Grammar {
            arguments: 1,
            max_operations: 2,
            max_expressions: 10000,
            unary: vec![Unary::GeluTanh],
            binary: vec![Binary::Multiply],
            affine: true,
        };
        let pilot = enumerate(&grammar).expect("pilot inventory");
        assert_eq!(
            pilot.expressions[6],
            Expr::Unary(
                Unary::GeluTanh,
                Box::new(Expr::Affine(Box::new(Expr::Argument(0))))
            )
        );
        let boundary = sharing_analysis(&pilot.expressions[6]);
        assert_eq!(boundary.learned_affine_expressions, 1);
        assert!(boundary.input_boundary_affine);
        assert!(!boundary.has_interior_learned_sharing);
        let product = sharing_analysis(&pilot.expressions[10]);
        assert_eq!(product.learned_affine_expressions, 0);
        assert!(!product.has_interior_learned_sharing);
        grammar.max_operations = 3;
        let inventory = enumerate(&grammar).expect("expanded inventory");
        let original = inventory.expressions.clone();
        let encoded = serde_json::to_vec(&inventory).expect("inventory bytes");
        let selection = interior_learned_selection(&inventory);
        let inner = Expr::Affine(Box::new(Expr::Unary(
            Unary::GeluTanh,
            Box::new(Expr::Argument(0)),
        )));
        for target in [
            Expr::Unary(Unary::GeluTanh, Box::new(inner.clone())),
            Expr::Binary(
                Binary::Multiply,
                Box::new(Expr::Argument(0)),
                Box::new(inner),
            ),
        ] {
            let id = inventory
                .expressions
                .iter()
                .position(|e| *e == target)
                .expect("generated interior form");
            assert!(selection.interior_indices.contains(&id));
            assert_eq!(
                selection.analyses[id].interior_learned_affine_expressions,
                1
            );
        }
        assert!(selection.baseline_indices.contains(&6));
        assert!(selection.baseline_indices.contains(&10));
        let mut all = selection.interior_indices.clone();
        all.extend(&selection.baseline_indices);
        all.sort_unstable();
        assert_eq!(all, (0..inventory.expressions.len()).collect::<Vec<_>>());
        assert_eq!(inventory.expressions, original);
        assert_eq!(
            serde_json::to_vec(&inventory).expect("unchanged bytes"),
            encoded
        );
        assert_eq!(
            enumerate(&grammar)
                .expect("same default enumeration")
                .expressions,
            original
        );
    }
    #[test]
    fn sharing_analysis_counts_cse_once_and_reports_redundancy_and_truncation() {
        let inner = Expr::Affine(Box::new(Expr::Unary(
            Unary::Relu,
            Box::new(Expr::Argument(0)),
        )));
        let repeated = Expr::Binary(
            Binary::Multiply,
            Box::new(inner.clone()),
            Box::new(inner.clone()),
        );
        let analysis = sharing_analysis(&repeated);
        assert_eq!(analysis.learned_affine_expressions, 1);
        assert_eq!(analysis.interior_learned_affine_expressions, 1);
        let linear = Expr::Affine(Box::new(Expr::Affine(Box::new(Expr::Argument(0)))));
        let analysis = sharing_analysis(&linear);
        assert_eq!(analysis.learned_affine_expressions, 2);
        assert!(analysis.adjacent_affine_redundancy);
        assert!(analysis.input_boundary_affine && analysis.output_boundary_affine);
        assert!(!analysis.has_interior_learned_sharing);
        // A shared subtree may have both linear and nonlinear readers. Any nonlinear
        // consumer path suffices; a root affine alone does not fabricate such a path.
        let mixed = Expr::Binary(
            Binary::Add,
            Box::new(inner.clone()),
            Box::new(Expr::Unary(Unary::Relu, Box::new(inner))),
        );
        assert_eq!(
            sharing_analysis(&mixed).interior_learned_affine_expressions,
            1
        );
        let selection = interior_learned_selection(&Inventory {
            expressions: vec![linear, repeated],
            truncated: true,
            intermediate_expressions: 2,
        });
        assert!(selection.source_inventory_truncated);
        assert_eq!(selection.interior_indices, vec![1]);
        assert_eq!(selection.baseline_indices, vec![0]);
    }
}
