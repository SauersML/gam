//! Adapter-free equation proposals on actual current-artifact boundary coordinates.
//! This finite coordinatewise grammar cannot express general linear mixing, learned
//! representations or persistent memory. A successful replacement is not by itself
//! recovery of computational organization.
use crate::{
    artifact::{Argument, Artifact, Callee},
    composed_rule_search::{self, Binary, Expr, Grammar, Unary},
    operator_program::{Coefficient, Interface, Law, Node, Operator, OperatorBody, Rule},
    program_joint_regions::{self as joint, Exit, JointBody},
    program_regions::{self, Region},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rejection {
    pub expression: Expr,
    pub reason: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Inventory {
    pub expressions: Vec<Expr>,
    pub rejections: Vec<Rejection>,
    pub truncated: bool,
    pub intermediate_expressions: usize,
}
fn boundaries(
    artifact: &Artifact,
    region: &Region,
    args: &[usize],
) -> Result<(Vec<Interface>, Interface), String> {
    // Reuse the established closed-cut and intervention-coverage validation.
    program_regions::extract(artifact, region)?;
    if args.len() != region.native_reads.len()
        || args.iter().copied().collect::<BTreeSet<_>>()
            != region.native_reads.iter().copied().collect()
    {
        return Err("expression arguments must be a bijection of complete region boundary".into());
    }
    let interfaces = artifact.program.interfaces().map_err(|e| e.to_string())?;
    let inputs = args
        .iter()
        .map(|native| {
            artifact
                .place(*native)
                .map(|current| interfaces[current].clone())
                .ok_or("expression native argument absent".into())
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok((inputs, interfaces[region.current_write].clone()))
}
struct Builder<'a> {
    artifact: &'a Artifact,
    inputs: Vec<Interface>,
    nodes: Vec<Node>,
    types: Vec<Interface>,
    operators: Vec<Operator>,
    identities: Vec<(Interface, usize)>,
    cache: BTreeMap<Expr, usize>,
}
impl Builder<'_> {
    fn identity(&mut self, interface: &Interface) -> usize {
        if let Some((_, id)) = self.identities.iter().find(|(ty, _)| ty == interface) {
            return *id;
        }
        let id = if let Some(id) = self.artifact.program.operators.iter().position(|op| {
            op.rows == *interface
                && op.cols == *interface
                && matches!(op.body, OperatorBody::Identity)
        }) {
            id
        } else {
            let id = self.artifact.program.operators.len() + self.operators.len();
            self.operators.push(Operator::identity(
                "expression fixed identity",
                interface.clone(),
            ));
            id
        };
        self.identities.push((interface.clone(), id));
        id
    }
    fn expr(&mut self, expr: &Expr) -> Result<usize, String> {
        if let Some(id) = self.cache.get(expr) {
            return Ok(*id);
        }
        let (node, ty) = match expr {
            Expr::Argument(index) => {
                let ty = self
                    .inputs
                    .get(*index)
                    .ok_or("expression references argument outside complete boundary")?
                    .clone();
                (Node::Param { index: *index }, ty)
            }
            Expr::Affine(_) => {
                return Err("learned Affine is forbidden in adapter-free synthesis".into());
            }
            Expr::Unary(op, input) => {
                let input = self.expr(input)?;
                let ty = self.types[input].clone();
                let law = match op {
                    Unary::Relu => Law::Relu,
                    Unary::Silu => Law::Silu,
                    Unary::Gelu => Law::Gelu,
                    Unary::GeluTanh => Law::GeluTanh,
                };
                (
                    Node::Pointwise {
                        input,
                        laws: vec![law; ty.group_count()],
                    },
                    ty,
                )
            }
            Expr::Binary(op, left, right) => {
                let left = self.expr(left)?;
                let mut right = self.expr(right)?;
                let ty = self.types[left].clone();
                if ty != self.types[right] {
                    return Err(
                        "coordinatewise binary operands require exactly equal interfaces".into(),
                    );
                }
                let node = match op {
                    Binary::Multiply => Node::Hadamard { left, right },
                    Binary::Add | Binary::Subtract => {
                        if *op == Binary::Subtract {
                            let id = self.nodes.len();
                            self.nodes.push(Node::Gain {
                                input: right,
                                coefficient: Coefficient::Number(-1.),
                            });
                            self.types.push(ty.clone());
                            right = id;
                        }
                        let identity = self.identity(&ty);
                        Node::Affine {
                            terms: vec![(left, identity), (right, identity)],
                            bias: None,
                        }
                    }
                };
                (node, ty)
            }
        };
        let id = self.nodes.len();
        self.nodes.push(node);
        self.types.push(ty);
        self.cache.insert(expr.clone(), id);
        Ok(id)
    }
}
fn compile(
    artifact: &Artifact,
    target: &Interface,
    expr: &Expr,
    inputs: Vec<Interface>,
) -> Result<(Rule, Vec<Operator>), String> {
    let mut builder = Builder {
        artifact,
        inputs: inputs.clone(),
        nodes: vec![],
        types: vec![],
        operators: vec![],
        identities: vec![],
        cache: BTreeMap::new(),
    };
    let output = builder.expr(expr)?;
    if &builder.types[output] != target {
        return Err("expression output interface differs from native write".into());
    }
    Ok((
        Rule {
            name: "adapter-free typed expression".into(),
            inputs,
            nodes: builder.nodes,
            output,
        },
        builder.operators,
    ))
}
pub fn enumerate(
    artifact: &Artifact,
    region: &Region,
    grammar: &Grammar,
    native_arguments: &[usize],
) -> Result<Inventory, String> {
    if grammar.affine {
        return Err("adapter-free grammar requires affine=false".into());
    }
    if grammar.arguments != region.native_reads.len() {
        return Err("grammar arguments must declare full region boundary arity".into());
    }
    let (inputs, target) = boundaries(artifact, region, native_arguments)?;
    let raw = composed_rule_search::enumerate(grammar)?;
    let mut result = Inventory {
        expressions: vec![],
        rejections: vec![],
        truncated: raw.truncated,
        intermediate_expressions: raw.intermediate_expressions,
    };
    for expression in raw.expressions {
        match compile(artifact, &target, &expression, inputs.clone()) {
            Ok(_) => result.expressions.push(expression),
            Err(reason) => result.rejections.push(Rejection { expression, reason }),
        }
    }
    Ok(result)
}
/// All cut inputs remain declared, including unused inputs (e.g. exact cancellation).
/// No donor/output/ambient state, coordinate adapter or learned matrix is supplied.
pub fn apply(
    artifact: &Artifact,
    region: &Region,
    expression: &Expr,
    native_arguments: &[usize],
) -> Result<Artifact, String> {
    let (inputs, target) = boundaries(artifact, region, native_arguments)?;
    let (rule, operators) = compile(artifact, &target, expression, inputs)?;
    let same = |stored: &Rule| {
        stored.inputs == rule.inputs && stored.nodes == rule.nodes && stored.output == rule.output
    };
    let callee = if operators.is_empty() {
        artifact
            .program
            .rules
            .iter()
            .position(same)
            .map_or(Callee::New(rule), Callee::Existing)
    } else {
        Callee::New(rule)
    };
    let result = artifact.replace_block(
        "adapter-free-expression",
        callee,
        native_arguments
            .iter()
            .copied()
            .map(Argument::Native)
            .collect(),
        region.native_write,
        operators,
    )?;
    if result.controls.len() != artifact.controls.len()
        || result.exceptions.len() != artifact.exceptions.len()
    {
        return Err("expression removes declared control/exception coverage".into());
    }
    crate::native_control::validate_shape(&result)?;
    Ok(result)
}

/// A tuple uses one exact-expression cache across every output. It has native
/// boundary inputs and fixed primitive operators only, with no fitted adapters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharedRejection {
    pub expressions: Vec<Expr>,
    pub reason: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharedInventory {
    pub expressions: Vec<Vec<Expr>>,
    pub rejections: Vec<SharedRejection>,
    pub truncated: bool,
    pub intermediate_expressions: usize,
    pub checked_tuples: usize,
    pub generated_tuple_states: usize,
    pub checked_anchor_pairs: usize,
    pub duplicate_tuples: usize,
    pub rejection_counts: BTreeMap<String, usize>,
    pub rejection_examples_limit: usize,
    pub compiled_node_counts: Vec<usize>,
    pub priority: String,
}
fn shared_boundaries(
    artifact: &Artifact,
    region: &joint::Region,
    args: &[usize],
) -> Result<(Vec<Interface>, Vec<Interface>), String> {
    joint::exact_body(artifact, region)?;
    if args.len() != region.native_reads.len()
        || args.iter().copied().collect::<BTreeSet<_>>()
            != region.native_reads.iter().copied().collect()
    {
        return Err("shared DAG arguments must biject the complete native boundary".into());
    }
    let types = artifact.program.interfaces().map_err(|e| e.to_string())?;
    let inputs = args
        .iter()
        .map(|n| {
            artifact
                .place(*n)
                .map(|c| types[c].clone())
                .ok_or("shared DAG argument absent".into())
        })
        .collect::<Result<Vec<_>, String>>()?;
    let outputs = region
        .native_writes
        .iter()
        .map(|n| {
            artifact
                .place(*n)
                .map(|c| types[c].clone())
                .ok_or("shared DAG exit absent".into())
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok((inputs, outputs))
}
fn non_argument_subexpressions(expr: &Expr, out: &mut BTreeSet<Expr>) {
    match expr {
        Expr::Argument(_) => (),
        Expr::Unary(_, a) | Expr::Affine(a) => {
            out.insert(expr.clone());
            non_argument_subexpressions(a, out);
        }
        Expr::Binary(_, a, b) => {
            out.insert(expr.clone());
            non_argument_subexpressions(a, out);
            non_argument_subexpressions(b, out);
        }
    }
}
fn genuinely_shared(expressions: &[Expr]) -> bool {
    let mut counts = BTreeMap::<Expr, usize>::new();
    for expression in expressions {
        let mut set = BTreeSet::new();
        non_argument_subexpressions(expression, &mut set);
        for expr in set {
            *counts.entry(expr).or_default() += 1;
        }
    }
    counts.values().any(|count| *count > 1)
}
fn compile_shared(
    artifact: &Artifact,
    region: &joint::Region,
    expressions: &[Expr],
    inputs: Vec<Interface>,
    targets: &[Interface],
) -> Result<JointBody, String> {
    if expressions.len() != targets.len() || targets.len() < 2 {
        return Err("shared DAG requires one expression for every joint exit and >=2 exits".into());
    }
    if !genuinely_shared(expressions) {
        return Err("tuple has no non-Argument computation shared across exits".into());
    }
    let mut builder = Builder {
        artifact,
        inputs: inputs.clone(),
        nodes: vec![],
        types: vec![],
        operators: vec![],
        identities: vec![],
        cache: BTreeMap::new(),
    };
    let mut exits = vec![];
    for ((expr, target), native_write) in expressions.iter().zip(targets).zip(&region.native_writes)
    {
        let node = builder.expr(expr)?;
        if &builder.types[node] != target {
            return Err("shared expression output differs from its native exit interface".into());
        }
        exits.push(Exit {
            native_write: *native_write,
            node,
        });
    }
    Ok(JointBody {
        inputs,
        nodes: builder.nodes,
        exits,
        operators: builder.operators,
    })
}
/// Unlike single-output enumeration, each tuple component may use any subset of
/// arguments; the complete boundary remains declared once for the entire DAG.
fn tuple_expression_pool(grammar: &Grammar) -> Result<(Vec<Expr>, bool, usize), String> {
    if grammar.affine || grammar.arguments == 0 || grammar.max_expressions < grammar.arguments {
        return Err("shared grammar needs boundary-covering budget and affine=false".into());
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
    enum Class {
        Unary {
            op: Unary,
            index: usize,
        },
        Binary {
            op: Binary,
            left_size: usize,
            left: usize,
            right: usize,
        },
    }
    'sizes: for size in 1..=grammar.max_operations {
        let mut next = vec![];
        let mut classes = std::collections::VecDeque::new();
        for op in &unary {
            classes.push_back(Class::Unary { op: *op, index: 0 });
        }
        for left_size in 0..size {
            for op in &binary {
                classes.push_back(Class::Binary {
                    op: *op,
                    left_size,
                    left: 0,
                    right: 0,
                });
            }
        }
        // Fair across operation and operand-size classes, rather than exhausting
        // all unbalanced products before any balanced shared-subexpression pair.
        while let Some(mut class) = classes.pop_front() {
            let expression = match &mut class {
                Class::Unary { op, index } => {
                    let Some(input) = by_size[size - 1].get(*index) else {
                        continue;
                    };
                    *index += 1;
                    Expr::Unary(*op, Box::new(input.clone()))
                }
                Class::Binary {
                    op,
                    left_size,
                    left,
                    right,
                } => {
                    let a_pool = &by_size[*left_size];
                    let b_pool = &by_size[size - 1 - *left_size];
                    let expression = loop {
                        if *left >= a_pool.len() || b_pool.is_empty() {
                            break None;
                        }
                        let a = &a_pool[*left];
                        let b = &b_pool[*right];
                        *right += 1;
                        if *right == b_pool.len() {
                            *right = 0;
                            *left += 1;
                        }
                        if *op != Binary::Subtract && a > b {
                            continue;
                        }
                        break Some(Expr::Binary(*op, Box::new(a.clone()), Box::new(b.clone())));
                    };
                    let Some(expression) = expression else {
                        continue;
                    };
                    expression
                }
            };
            classes.push_back(class);
            if seen.contains(&expression) {
                continue;
            }
            if seen.len() == grammar.max_expressions {
                truncated = true;
                by_size.push(next);
                break 'sizes;
            }
            seen.insert(expression.clone());
            next.push(expression);
        }
        if next.is_empty() {
            break;
        }
        by_size.push(next);
    }
    let count = seen.len();
    Ok((by_size.into_iter().flatten().collect(), truncated, count))
}
/// Index generic expressions by their non-Argument subexpressions, then search
/// tuple generators anchored on a shared computation in two distinct exits.
/// All other exits range over their typed expression pools. Anchor-pair checks
/// and popped states (including duplicates) each have the explicit check cap;
/// visited states are capped by checks * (exits + 1). No Cartesian list is built.
pub fn enumerate_shared(
    artifact: &Artifact,
    region: &joint::Region,
    grammar: &Grammar,
    native_arguments: &[usize],
    max_tuple_checks: usize,
    max_tuples: usize,
    max_body_nodes: usize,
) -> Result<SharedInventory, String> {
    if max_tuple_checks == 0 || max_tuples == 0 || max_body_nodes == 0 {
        return Err("positive shared tuple/check/body budgets required".into());
    }
    if grammar.arguments != region.native_reads.len() {
        return Err("shared grammar arguments must cover the whole native boundary".into());
    }
    let (inputs, targets) = shared_boundaries(artifact, region, native_arguments)?;
    if targets.len() < 2 {
        return Err("shared DAG requires multiple exits".into());
    }
    let state_limit = targets
        .len()
        .checked_add(1)
        .and_then(|n| max_tuple_checks.checked_mul(n))
        .ok_or("shared tuple state budget overflow")?;
    let (pool, truncated, count) = tuple_expression_pool(grammar)?;
    let mut result = SharedInventory {
        expressions: vec![],
        rejections: vec![],
        truncated,
        intermediate_expressions: count,
        checked_tuples: 0,
        generated_tuple_states: 0,
        checked_anchor_pairs: 0,
        duplicate_tuples: 0,
        rejection_counts: BTreeMap::new(),
        rejection_examples_limit: 16,
        compiled_node_counts: vec![],
        priority: "Ascending actual compiled shared-DAG node count, then lexicographic Expr tuple; enumeration coverage remains explicitly bounded before ranking".into(),
    };
    // Compile each single expression once to learn its exact output interface.
    // The shared compiler later checks the complete tuple and CSE node budget.
    let mut typed_pools = vec![vec![]; targets.len()];
    let mut inverted = BTreeMap::<Expr, Vec<usize>>::new();
    for (id, expression) in pool.iter().enumerate() {
        let mut builder = Builder {
            artifact,
            inputs: inputs.clone(),
            nodes: vec![],
            types: vec![],
            operators: vec![],
            identities: vec![],
            cache: BTreeMap::new(),
        };
        let Ok(node) = builder.expr(expression) else {
            continue;
        };
        for (axis, target) in targets.iter().enumerate() {
            if &builder.types[node] == target {
                typed_pools[axis].push(id);
            }
        }
        let mut shared = BTreeSet::new();
        non_argument_subexpressions(expression, &mut shared);
        for subexpression in shared {
            inverted.entry(subexpression).or_default().push(id);
        }
    }
    if typed_pools.iter().any(Vec::is_empty) {
        return Ok(result);
    }
    let typed_pools: Vec<_> = typed_pools.into_iter().map(std::sync::Arc::new).collect();
    let mut generators: Vec<Vec<std::sync::Arc<Vec<usize>>>> = vec![];
    let mut heap = std::collections::BinaryHeap::new();
    let mut visited = BTreeSet::new();
    'anchors: for containing in inverted.values() {
        for first in 0..targets.len() {
            for second in first + 1..targets.len() {
                if result.checked_anchor_pairs == max_tuple_checks {
                    result.truncated = true;
                    break 'anchors;
                }
                result.checked_anchor_pairs += 1;
                let mut choices = typed_pools.clone();
                for axis in [first, second] {
                    choices[axis] = std::sync::Arc::new(
                        typed_pools[axis]
                            .iter()
                            .copied()
                            .filter(|id| containing.binary_search(id).is_ok())
                            .collect(),
                    );
                }
                if choices[first].is_empty() || choices[second].is_empty() {
                    continue;
                }
                if visited.len() == state_limit {
                    result.truncated = true;
                    break 'anchors;
                }
                let priority = choices
                    .iter()
                    .try_fold(0usize, |sum, axis| sum.checked_add(axis[0]))
                    .ok_or("shared tuple priority overflow")?;
                let generator = generators.len();
                let positions = vec![0usize; targets.len()];
                visited.insert((generator, positions.clone()));
                heap.push(std::cmp::Reverse((priority, generator, positions)));
                generators.push(choices);
            }
        }
    }
    let mut tuple_seen = BTreeSet::new();
    let mut compiled = vec![];
    while let Some(std::cmp::Reverse((priority, generator, positions))) = heap.pop() {
        if result.checked_tuples == max_tuple_checks || compiled.len() == max_tuples {
            result.truncated = true;
            break;
        }
        result.checked_tuples += 1;
        let choices = &generators[generator];
        let indices = positions
            .iter()
            .enumerate()
            .map(|(axis, pos)| choices[axis][*pos])
            .collect::<Vec<_>>();
        if tuple_seen.insert(indices.clone()) {
            let expressions = indices.iter().map(|i| pool[*i].clone()).collect::<Vec<_>>();
            match compile_shared(artifact, region, &expressions, inputs.clone(), &targets) {
                Ok(body) if body.nodes.len() <= max_body_nodes => {
                    compiled.push((body.nodes.len(), expressions));
                }
                other => {
                    let reason = match other {
                        Ok(_) => "shared body node budget exceeded".to_string(),
                        Err(reason) => reason,
                    };
                    *result.rejection_counts.entry(reason.clone()).or_default() += 1;
                    if result.rejections.len() < result.rejection_examples_limit {
                        result.rejections.push(SharedRejection {
                            expressions,
                            reason,
                        });
                    }
                }
            }
        } else {
            result.duplicate_tuples += 1;
        }
        for axis in 0..positions.len() {
            if positions[axis] + 1 >= choices[axis].len() {
                continue;
            }
            let mut next = positions.clone();
            next[axis] += 1;
            let key = (generator, next.clone());
            if !visited.contains(&key) {
                if visited.len() == state_limit {
                    result.truncated = true;
                    continue;
                }
                let delta = choices[axis][next[axis]] - choices[axis][positions[axis]];
                visited.insert(key);
                heap.push(std::cmp::Reverse((
                    priority
                        .checked_add(delta)
                        .ok_or("shared tuple priority overflow")?,
                    generator,
                    next,
                )));
            }
        }
    }
    result.generated_tuple_states = visited.len();
    compiled.sort();
    for (node_count, expressions) in compiled {
        result.compiled_node_counts.push(node_count);
        result.expressions.push(expressions);
    }
    Ok(result)
}
/// Exact input-coordinate canonicalization for cross-binding deduplication.
/// Only the grammar's existing commutative operand exchange is normalized;
/// subtraction order, association and numerical identities remain untouched.
pub fn canonical_shared_expressions(
    region: &joint::Region,
    expressions: &[Expr],
    native_arguments: &[usize],
) -> Result<Vec<Expr>, String> {
    if native_arguments.len() != region.native_reads.len()
        || native_arguments.iter().copied().collect::<BTreeSet<_>>()
            != region.native_reads.iter().copied().collect()
    {
        return Err("shared canonical arguments must biject the native region boundary".into());
    }
    let permutation = native_arguments
        .iter()
        .map(|native| {
            region
                .native_reads
                .iter()
                .position(|n| n == native)
                .ok_or("canonical shared input absent")
        })
        .collect::<Result<Vec<_>, _>>()?;
    fn remap(expr: &Expr, permutation: &[usize]) -> Result<Expr, String> {
        Ok(match expr {
            Expr::Argument(index) => Expr::Argument(
                *permutation
                    .get(*index)
                    .ok_or("canonical expression input outside boundary")?,
            ),
            Expr::Unary(op, input) => Expr::Unary(*op, Box::new(remap(input, permutation)?)),
            Expr::Affine(_) => {
                return Err("shared canonicalization forbids learned affine adapters".into());
            }
            Expr::Binary(op, left, right) => {
                let mut left = remap(left, permutation)?;
                let mut right = remap(right, permutation)?;
                if *op != Binary::Subtract && left > right {
                    std::mem::swap(&mut left, &mut right);
                }
                Expr::Binary(*op, Box::new(left), Box::new(right))
            }
        })
    }
    expressions
        .iter()
        .map(|expr| remap(expr, &permutation))
        .collect()
}
pub fn apply_shared(
    artifact: &Artifact,
    region: &joint::Region,
    expressions: &[Expr],
    native_arguments: &[usize],
) -> Result<Artifact, String> {
    let (inputs, targets) = shared_boundaries(artifact, region, native_arguments)?;
    let mut body = compile_shared(artifact, region, expressions, inputs, &targets)?;
    // The tuple grammar uses the proposed argument order; joint::apply binds
    // Param indices in the region's canonical boundary order. Translate once.
    let permutation = native_arguments
        .iter()
        .map(|native| {
            region
                .native_reads
                .iter()
                .position(|n| n == native)
                .ok_or("shared argument missing from canonical boundary")
        })
        .collect::<Result<Vec<_>, _>>()?;
    for node in &mut body.nodes {
        if let Node::Param { index } = node {
            *index = permutation[*index];
        }
    }
    let mut canonical_inputs = body.inputs.clone();
    for (proposed, canonical) in permutation.iter().enumerate() {
        canonical_inputs[*canonical] = body.inputs[proposed].clone();
    }
    body.inputs = canonical_inputs;
    joint::apply(artifact, region, &body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_program::{
        Declarations, FamilyInputs, OperatorProgram, SequenceLayout, Slot, SlotValues,
    };
    use std::sync::Arc;
    fn base(nodes: Vec<Node>, slots: usize) -> Artifact {
        let p = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }; slots],
                parameters: 0,
            },
            bases: vec![],
            operators: vec![Arc::new(Operator::identity(
                "native add",
                Interface::native(2).expect("interface"),
            ))],
            rules: vec![],
            output: nodes.len() - 1,
            nodes,
        };
        Artifact::native(&p).expect("native artifact")
    }
    fn cut(a: &Artifact, internal: &[usize]) -> Region {
        program_regions::propose_regions(
            a,
            program_regions::Limits {
                max_internal_nodes: 16,
                max_inputs: 8,
                max_regions: 1000,
                max_states: 1000,
            },
        )
        .expect("inventory")
        .regions
        .into_iter()
        .find(|r| r.native_write == a.program.output && r.current_internal_nodes == internal)
        .expect("requested cut")
    }
    fn family(inputs: Vec<ndarray::Array2<f64>>) -> FamilyInputs {
        FamilyInputs {
            rows: 2,
            slots: inputs.into_iter().map(SlotValues::Raw).collect(),
            layout: Some(SequenceLayout {
                sequence: vec![0; 2],
                position: vec![0, 1],
            }),
        }
    }
    fn output(a: &Artifact, x: &FamilyInputs) -> ndarray::Array2<f64> {
        a.program.execute(x, false).expect("execute").values[a.program.output].clone()
    }
    #[test]
    fn alternate_factorization_not_literal_native_dag_with_controls_and_decode() {
        let a = base(
            vec![
                Node::Raw { slot: 0 },
                Node::Raw { slot: 1 },
                Node::Raw { slot: 2 },
                Node::Hadamard { left: 0, right: 1 },
                Node::Hadamard { left: 0, right: 2 },
                Node::Affine {
                    terms: vec![(3, 0), (4, 0)],
                    bias: None,
                },
            ],
            3,
        );
        let r = cut(&a, &[3, 4, 5]);
        let expression = Expr::Binary(
            Binary::Multiply,
            Box::new(Expr::Argument(0)),
            Box::new(Expr::Binary(
                Binary::Add,
                Box::new(Expr::Argument(1)),
                Box::new(Expr::Argument(2)),
            )),
        );
        let grammar = Grammar {
            arguments: 3,
            max_operations: 2,
            max_expressions: 1000,
            unary: vec![],
            binary: vec![Binary::Add, Binary::Multiply],
            affine: false,
        };
        assert!(
            enumerate(&a, &r, &grammar, &r.native_reads)
                .expect("equation inventory")
                .expressions
                .contains(&expression)
        );
        let b = apply(&a, &r, &expression, &r.native_reads).expect("new equation");
        assert!(
            b.program
                .operators
                .iter()
                .all(|o| matches!(o.body, OperatorBody::Identity))
        );
        let body = b.program.rules.last().expect("synthesized rule");
        assert!(body.nodes.iter().any(|n|matches!(n,Node::Hadamard{left:_,right} if matches!(body.nodes[*right],Node::Affine{..}))));
        assert!(a.program.nodes.iter().all(|n|!matches!(n,Node::Hadamard{left:_,right} if matches!(a.program.nodes[*right],Node::Affine{..}))));
        for gains in [
            [1., 1., 1.],
            [-1., 1., 1.],
            [1., 2., 1.],
            [1., 1., -2.],
            [2., -1., 0.5],
        ] {
            let x = family(vec![
                ndarray::array![[1., 2.], [3., 4.]] * gains[0],
                ndarray::array![[2., -1.], [1., 0.5]] * gains[1],
                ndarray::array![[1., 3.], [-2., 2.]] * gains[2],
            ]);
            assert_eq!(output(&a, &x), output(&b, &x));
            let bytes = b.to_bytes().expect("save");
            let replay = Artifact::from_bytes(&bytes, &a.program.declarations).expect("decode");
            assert_eq!(bytes, replay.to_bytes().expect("canonical"));
            assert_eq!(output(&a, &x), output(&replay, &x));
        }
    }
    #[test]
    fn unused_boundary_cancellation_and_independent_control_negative() {
        let a = base(
            vec![
                Node::Raw { slot: 0 },
                Node::Raw { slot: 1 },
                Node::Gain {
                    input: 1,
                    coefficient: Coefficient::Number(-1.),
                },
                Node::Affine {
                    terms: vec![(0, 0), (1, 0), (2, 0)],
                    bias: None,
                },
            ],
            2,
        );
        let r = cut(&a, &[2, 3]);
        let b = apply(&a, &r, &Expr::Argument(0), &r.native_reads).expect("cancelled unused input");
        assert_eq!(b.program.rules.last().expect("rule").inputs.len(), 2);
        for gain in [-2., 0., 1., 3.] {
            let x = family(vec![
                ndarray::array![[1., 2.], [3., 4.]],
                ndarray::array![[2., 1.], [4., 8.]] * gain,
            ]);
            assert_eq!(output(&a, &x), output(&b, &x));
        }
        let negative = base(
            vec![
                Node::Raw { slot: 0 },
                Node::Raw { slot: 1 },
                Node::Raw { slot: 2 },
                Node::Gain {
                    input: 2,
                    coefficient: Coefficient::Number(-1.),
                },
                Node::Affine {
                    terms: vec![(0, 0), (1, 0), (3, 0)],
                    bias: None,
                },
            ],
            3,
        );
        let r = cut(&negative, &[3, 4]);
        let wrong = apply(&negative, &r, &Expr::Argument(0), &r.native_reads)
            .expect("clean shortcut proposal");
        let branch = ndarray::array![[2., 1.], [4., 8.]];
        let clean = family(vec![
            ndarray::array![[1., 2.], [3., 4.]],
            branch.clone(),
            branch.clone(),
        ]);
        assert_eq!(output(&negative, &clean), output(&wrong, &clean));
        for gains in [[0.5, 1.], [1., -1.], [2., 0.5]] {
            let edited = family(vec![
                ndarray::array![[1., 2.], [3., 4.]],
                &branch * gains[0],
                &branch * gains[1],
            ]);
            assert_ne!(output(&negative, &edited), output(&wrong, &edited));
        }
    }
    fn joint_cut(a: &Artifact, internal: &[usize]) -> joint::Region {
        joint::propose_regions(
            a,
            joint::Limits {
                max_internal_nodes: 16,
                max_inputs: 8,
                max_exits: 8,
                max_regions: 1000,
                max_states: 1000,
            },
        )
        .expect("joint inventory")
        .regions
        .into_iter()
        .find(|r| r.current_internal_nodes == internal)
        .expect("joint cut")
    }
    fn shared_fixture() -> (Artifact, joint::Region, Vec<Expr>) {
        let mut program = base(vec![Node::Raw { slot: 0 }, Node::Raw { slot: 1 }], 2).program;
        let ty = Interface::native(2).expect("interface");
        program.operators.push(Arc::new(
            Operator::dense(
                "native subtraction",
                ty.clone(),
                ty,
                -ndarray::Array2::eye(2),
                crate::operator_program::exact_precision([-1., 0.]).expect("precision"),
                crate::operator_program::Provenance::native("fixture"),
            )
            .expect("negative identity"),
        ));
        program.nodes = vec![
            Node::Raw { slot: 0 },
            Node::Raw { slot: 1 },
            Node::Affine {
                terms: vec![(0, 0), (1, 1)],
                bias: None,
            },
            Node::Hadamard { left: 2, right: 2 },
            Node::Pointwise {
                input: 2,
                laws: vec![Law::Silu],
            },
            Node::Affine {
                terms: vec![(3, 0), (4, 0)],
                bias: None,
            },
        ];
        program.output = 5;
        let a = Artifact::native(&program).expect("native");
        let r = joint_cut(&a, &[2, 3, 4]);
        assert_eq!(r.native_writes, vec![3, 4]);
        let difference = Expr::Binary(
            Binary::Subtract,
            Box::new(Expr::Argument(0)),
            Box::new(Expr::Argument(1)),
        );
        let expressions = vec![
            Expr::Binary(
                Binary::Multiply,
                Box::new(difference.clone()),
                Box::new(difference.clone()),
            ),
            Expr::Unary(Unary::Silu, Box::new(difference)),
        ];
        (a, r, expressions)
    }
    #[test]
    fn shared_dag_executes_one_latent_node_and_keeps_both_exit_views() {
        let (a, r, expressions) = shared_fixture();
        let b = apply_shared(&a, &r, &expressions, &r.native_reads).expect("shared equation");
        assert!(b.program.rules.is_empty());
        assert!(
            b.place(2).is_none(),
            "ordinary producer replaced by unnamed shared state"
        );
        let square = b.place(3).expect("square exit");
        let silu = b.place(4).expect("silu exit");
        // Exit views may wrap the actual value node; find the shared computation.
        let square_input = b
            .program
            .nodes
            .iter()
            .find_map(|n| match n {
                Node::Hadamard { left, right } if left == right => Some(*left),
                _ => None,
            })
            .expect("square");
        let silu_input = b
            .program
            .nodes
            .iter()
            .find_map(|n| match n {
                Node::Pointwise { input, laws } if laws == &[Law::Silu] => Some(*input),
                _ => None,
            })
            .expect("silu");
        assert_eq!(square_input, silu_input);
        assert_ne!(square, silu);
        assert_eq!(
            b.program
                .nodes
                .iter()
                .filter(|n| matches!(
                    n,
                    Node::Gain {
                        coefficient: Coefficient::Number(-1.),
                        ..
                    }
                ))
                .count(),
            1
        );
        for gains in [[1., 1.], [-1., 1.], [1., -2.], [0.5, 3.]] {
            let x = family(vec![
                ndarray::array![[1., 2.], [3., 4.]] * gains[0],
                ndarray::array![[2., -1.], [0.5, 1.]] * gains[1],
            ]);
            assert_eq!(output(&a, &x), output(&b, &x));
            let replay =
                Artifact::from_bytes(&b.to_bytes().expect("save"), &a.program.declarations)
                    .expect("decode");
            assert_eq!(output(&b, &x), output(&replay, &x));
        }
    }
    #[test]
    fn shared_noncommutative_arguments_follow_proposed_order() {
        let (a, r, expressions) = shared_fixture();
        let reverse = r.native_reads.iter().rev().copied().collect::<Vec<_>>();
        let b = apply_shared(&a, &r, &expressions, &reverse).expect("reverse binding");
        let left = ndarray::array![[1., 2.], [3., 4.]];
        let right = ndarray::array![[2., -1.], [0.5, 1.]];
        let x = family(vec![left.clone(), right.clone()]);
        let swapped = family(vec![right, left]);
        assert_ne!(output(&a, &x), output(&b, &x));
        assert_eq!(output(&a, &swapped), output(&b, &x));
        let canonical =
            canonical_shared_expressions(&r, &expressions, &reverse).expect("canonical binding");
        assert_ne!(
            canonical, expressions,
            "subtraction direction must survive input remapping"
        );
        let c = apply_shared(&a, &r, &canonical, &r.native_reads).expect("canonical equation");
        assert_eq!(
            b.to_bytes().expect("proposed save"),
            c.to_bytes().expect("canonical save")
        );
        assert_eq!(output(&b, &x), output(&c, &x));

        assert!(apply_shared(&a, &r, &expressions, &[r.native_reads[0]; 2]).is_err());
    }
    #[test]
    fn canonicalization_removes_permutation_duplicates_without_algebraic_rewriting() {
        let (_, r, _) = shared_fixture();
        let reverse = r.native_reads.iter().rev().copied().collect::<Vec<_>>();
        let product = Expr::Binary(
            Binary::Multiply,
            Box::new(Expr::Argument(0)),
            Box::new(Expr::Argument(1)),
        );
        let sum = Expr::Binary(
            Binary::Add,
            Box::new(Expr::Argument(0)),
            Box::new(Expr::Argument(1)),
        );
        let tuple = vec![product, sum];
        assert_eq!(
            canonical_shared_expressions(&r, &tuple, &r.native_reads).unwrap(),
            canonical_shared_expressions(&r, &tuple, &reverse).unwrap()
        );
        let subtract = Expr::Binary(
            Binary::Subtract,
            Box::new(Expr::Argument(0)),
            Box::new(Expr::Argument(1)),
        );
        assert_ne!(
            canonical_shared_expressions(&r, &[subtract.clone()], &r.native_reads).unwrap(),
            canonical_shared_expressions(&r, &[subtract], &reverse).unwrap()
        );
        assert!(canonical_shared_expressions(&r, &[Expr::Argument(2)], &r.native_reads).is_err());
    }
    #[test]
    fn bounded_pool_covers_balanced_and_unbalanced_operation_classes() {
        let grammar = Grammar {
            arguments: 2,
            max_operations: 3,
            max_expressions: 400,
            unary: vec![Unary::Silu],
            binary: vec![Binary::Subtract, Binary::Multiply],
            affine: false,
        };
        let (pool, _, count) = tuple_expression_pool(&grammar).expect("pool");
        let difference = Expr::Binary(
            Binary::Subtract,
            Box::new(Expr::Argument(0)),
            Box::new(Expr::Argument(1)),
        );
        let square = Expr::Binary(
            Binary::Multiply,
            Box::new(difference.clone()),
            Box::new(difference.clone()),
        );
        assert!(
            pool.contains(&square),
            "balanced products must not follow every unbalanced product"
        );
        assert!(pool.contains(&Expr::Unary(Unary::Silu, Box::new(difference))));
        assert!(count <= 400);
    }
    #[test]
    fn tuple_search_anchors_shared_computation_and_bounds_every_attempt() {
        let (a, r, _) = shared_fixture();
        let grammar = Grammar {
            arguments: 2,
            max_operations: 1,
            max_expressions: 16,
            unary: vec![Unary::Silu],
            binary: vec![Binary::Subtract],
            affine: false,
        };
        let inventory =
            enumerate_shared(&a, &r, &grammar, &r.native_reads, 64, 8, 32).expect("tuple search");
        let right_only = Expr::Unary(Unary::Silu, Box::new(Expr::Argument(1)));
        assert!(
            inventory
                .expressions
                .contains(&vec![right_only.clone(), right_only])
        );
        assert!(inventory.checked_tuples <= 64);
        assert!(inventory.generated_tuple_states <= 64 * 3);
        assert!(inventory.checked_anchor_pairs <= 64);
        assert!(
            inventory
                .expressions
                .iter()
                .all(|tuple| genuinely_shared(tuple))
        );
        let capped = enumerate_shared(&a, &r, &grammar, &r.native_reads, 1, 8, 32).expect("cap");
        assert_eq!(capped.checked_tuples, 1);
        assert!(capped.truncated);
        assert!(
            capped
                .expressions
                .iter()
                .all(|tuple| genuinely_shared(tuple))
        );
        assert!(enumerate_shared(&a, &r, &grammar, &r.native_reads, usize::MAX, 8, 32).is_err());
    }
    #[test]
    fn adapters_and_hidden_arguments_refused() {
        let a = base(
            vec![Node::Raw { slot: 0 }, Node::Hadamard { left: 0, right: 0 }],
            1,
        );
        let r = cut(&a, &[1]);
        assert!(
            apply(
                &a,
                &r,
                &Expr::Affine(Box::new(Expr::Argument(0))),
                &r.native_reads
            )
            .is_err()
        );
        assert!(apply(&a, &r, &Expr::Argument(1), &r.native_reads).is_err());
        assert!(apply(&a, &r, &Expr::Argument(0), &[]).is_err());
        let grammar = Grammar {
            arguments: 1,
            max_operations: 1,
            max_expressions: 2,
            unary: vec![Unary::Relu],
            binary: vec![Binary::Add],
            affine: false,
        };
        assert!(
            enumerate(&a, &r, &grammar, &r.native_reads)
                .expect("bounded inventory")
                .truncated
        );
    }
}
