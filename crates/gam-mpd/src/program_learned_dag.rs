//! Bounded proposals for typed vector equations with explicitly named affine
//! parameters. Outputs can have different nonlinear equations and share actual
//! intermediate computations. This is a restricted fitted candidate family, not
//! identification of native variables or a guarantee of structure recovery.
use crate::{
    artifact::Artifact,
    composed_rule_search::{Binary, Unary},
    operator_program::{
        Coefficient, Declarations, Interface, Law, Node, Operator, OperatorProgram, Slot,
        exact_precision,
    },
    program_joint_regions::{self as joint, Exit, JointBody},
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::Arc,
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TypeRef {
    Input(usize),
    Exit(usize),
    Latent { width: usize },
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Expr {
    Argument(usize),
    Unary(Unary, Box<Expr>),
    Binary(Binary, Box<Expr>, Box<Expr>),
    Affine {
        parameter: usize,
        output: TypeRef,
        input: Box<Expr>,
        bias: bool,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub latent_widths: Vec<usize>,
    pub unary: Vec<Unary>,
    pub binary: Vec<Binary>,
    pub affine_bias: bool,
    pub require_shared: bool,
    /// Tree operations per individual output/value; shared DAG size is separately bounded.
    pub max_operations: usize,
    pub max_affine_parameters: usize,
    /// Matrix and optional bias scalar elements, counting each parameter identity once.
    pub max_parameter_elements: usize,
    /// Input skeletons plus every attempted expression extension, including rejected/duplicate work.
    pub max_expression_states: usize,
    /// Inspected output skeleton tuples plus every attempted partial parameter binding.
    /// Completed bindings are counted separately; invalid and duplicate work still consumes budget.
    pub max_tuple_checks: usize,
    pub max_tuples: usize,
    pub max_body_nodes: usize,
    pub seed: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proposal {
    pub expressions: Vec<Expr>,
    /// Syntactic affine-parameter ancestry for each supervised output. This
    /// flags available learned paths, not nonzero numerical derivatives,
    /// observability, or whether fitting can achieve the requested target.
    #[serde(default)]
    pub output_has_learned_ancestor: Vec<bool>,
    pub compiled_node_count: usize,
    pub parameter_elements: usize,
    pub trainable_operator_count: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Inventory {
    pub proposals: Vec<Proposal>,
    pub explored_states: usize,
    pub accepted_states: usize,
    pub checked_tuples: usize,
    pub checked_skeleton_tuples: usize,
    pub completed_parameter_bindings: usize,
    pub duplicate_states: usize,
    pub duplicate_tuples: usize,
    pub rejection_counts: BTreeMap<String, usize>,
    pub truncated: bool,
    pub priority: String,
}
pub struct Applied {
    pub artifact: Artifact,
    pub trainable_operator_ids: Vec<usize>,
    pub node_mapping: Vec<usize>,
}
pub struct Compiled {
    pub program: OperatorProgram,
    pub output_nodes: Vec<usize>,
    pub trainable_operator_ids: Vec<usize>,
}
#[derive(Clone, PartialEq, Eq)]
struct Parameter {
    rows: Interface,
    cols: Interface,
    bias: bool,
}
fn resolve(t: &TypeRef, inputs: &[Interface], outputs: &[Interface]) -> Result<Interface, String> {
    match t {
        TypeRef::Input(i) => inputs
            .get(*i)
            .cloned()
            .ok_or("affine input type reference absent".into()),
        TypeRef::Exit(i) => outputs
            .get(*i)
            .cloned()
            .ok_or("affine exit type reference absent".into()),
        TypeRef::Latent { width } => Interface::native(*width).map_err(|e| e.to_string()),
    }
}
fn choices(
    inputs: &[Interface],
    outputs: &[Interface],
    widths: &[usize],
) -> Result<Vec<(TypeRef, Interface)>, String> {
    let mut out = Vec::new();
    for t in (0..inputs.len())
        .map(TypeRef::Input)
        .chain((0..outputs.len()).map(TypeRef::Exit))
        .chain(
            widths
                .iter()
                .copied()
                .map(|width| TypeRef::Latent { width }),
        )
    {
        let ty = resolve(&t, inputs, outputs)?;
        if !out.iter().any(|(_, held)| *held == ty) {
            out.push((t, ty));
        }
    }
    Ok(out)
}
/// Canonical IDs preserve the parameter-sharing partition. Equal actual interfaces
/// have one spelling. Ordered expression syntax is preserved: this does not solve
/// commutative graph isomorphism, so equivalent operand orders can remain proposals.
/// No numerical equality test or algebraic rewrite identifies parameter owners.
pub fn canonicalize(
    inputs: &[Interface],
    outputs: &[Interface],
    expressions: &[Expr],
) -> Result<Vec<Expr>, String> {
    fn types(
        e: &Expr,
        inputs: &[Interface],
        outputs: &[Interface],
        aliases: &mut Vec<(TypeRef, Interface)>,
    ) -> Result<Expr, String> {
        Ok(match e {
            Expr::Argument(i) => Expr::Argument(*i),
            Expr::Unary(op, x) => Expr::Unary(*op, Box::new(types(x, inputs, outputs, aliases)?)),
            Expr::Binary(op, a, b) => {
                let a = types(a, inputs, outputs, aliases)?;
                let b = types(b, inputs, outputs, aliases)?;
                Expr::Binary(*op, Box::new(a), Box::new(b))
            }
            Expr::Affine {
                parameter,
                output,
                input,
                bias,
            } => {
                let ty = resolve(output, inputs, outputs)?;
                let t = if let Some((t, _)) = aliases.iter().find(|(_, held)| *held == ty) {
                    t.clone()
                } else {
                    aliases.push((output.clone(), ty));
                    output.clone()
                };
                Expr::Affine {
                    parameter: *parameter,
                    output: t,
                    input: Box::new(types(input, inputs, outputs, aliases)?),
                    bias: *bias,
                }
            }
        })
    }
    fn names(e: &Expr, ids: &mut BTreeMap<usize, usize>) -> Expr {
        match e {
            Expr::Argument(i) => Expr::Argument(*i),
            Expr::Unary(op, x) => Expr::Unary(*op, Box::new(names(x, ids))),
            Expr::Binary(op, a, b) => {
                Expr::Binary(*op, Box::new(names(a, ids)), Box::new(names(b, ids)))
            }
            Expr::Affine {
                parameter,
                output,
                input,
                bias,
            } => {
                let input = names(input, ids);
                let next = ids.len();
                let parameter = *ids.entry(*parameter).or_insert(next);
                Expr::Affine {
                    parameter,
                    output: output.clone(),
                    input: Box::new(input),
                    bias: *bias,
                }
            }
        }
    }
    let mut aliases = choices(inputs, outputs, &[])?;
    let expressions = expressions
        .iter()
        .map(|e| types(e, inputs, outputs, &mut aliases))
        .collect::<Result<Vec<_>, _>>()?;
    let mut ids = BTreeMap::new();
    Ok(expressions.iter().map(|e| names(e, &mut ids)).collect())
}
fn operations(e: &Expr) -> usize {
    match e {
        Expr::Argument(_) => 0,
        Expr::Unary(_, x) | Expr::Affine { input: x, .. } => 1 + operations(x),
        Expr::Binary(_, a, b) => 1 + operations(a) + operations(b),
    }
}
struct Checker<'a> {
    inputs: &'a [Interface],
    outputs: &'a [Interface],
    settings: &'a Settings,
    parameters: BTreeMap<usize, Parameter>,
    cache: BTreeMap<Expr, Interface>,
    extra_nodes: usize,
}
impl Checker<'_> {
    fn expr(&mut self, e: &Expr) -> Result<Interface, String> {
        if let Some(ty) = self.cache.get(e) {
            return Ok(ty.clone());
        }
        if operations(e) > self.settings.max_operations {
            return Err("max_operations exceeded".into());
        }
        let ty = match e {
            Expr::Argument(i) => self
                .inputs
                .get(*i)
                .cloned()
                .ok_or("argument outside complete boundary")?,
            Expr::Unary(op, x) => {
                if !self.settings.unary.contains(op) {
                    return Err("unary operation outside declared grammar".into());
                }
                self.expr(x)?
            }
            Expr::Binary(op, a, b) => {
                if !self.settings.binary.contains(op) {
                    return Err("binary operation outside declared grammar".into());
                }
                let a = self.expr(a)?;
                if a != self.expr(b)? {
                    return Err("binary operand interfaces differ".into());
                }
                if *op == Binary::Subtract {
                    self.extra_nodes += 1;
                }
                a
            }
            Expr::Affine {
                parameter,
                output,
                input,
                bias,
            } => {
                if *bias != self.settings.affine_bias {
                    return Err("affine bias outside declared grammar".into());
                }
                let cols = self.expr(input)?;
                let rows = resolve(output, self.inputs, self.outputs)?;
                if let TypeRef::Latent { width } = output {
                    if !self.settings.latent_widths.contains(width) {
                        return Err("latent width not declared".into());
                    }
                }
                let spec = Parameter {
                    rows: rows.clone(),
                    cols,
                    bias: *bias,
                };
                if let Some(held) = self.parameters.get(parameter) {
                    if *held != spec {
                        return Err("parameter ID has inconsistent interfaces or bias".into());
                    }
                } else {
                    self.parameters.insert(*parameter, spec);
                }
                rows
            }
        };
        self.cache.insert(e.clone(), ty.clone());
        Ok(ty)
    }
    fn limits(&self) -> Result<(usize, usize, usize), String> {
        if self.parameters.len() > self.settings.max_affine_parameters {
            return Err("max_affine_parameters exceeded".into());
        }
        let mut elements = 0usize;
        let mut operators = 0usize;
        for p in self.parameters.values() {
            elements = elements
                .checked_add(
                    p.rows
                        .width()
                        .checked_mul(p.cols.width())
                        .ok_or("parameter shape overflow")?,
                )
                .ok_or("parameter count overflow")?;
            if p.bias {
                elements = elements
                    .checked_add(p.rows.width())
                    .ok_or("parameter count overflow")?;
            }
            operators += 1 + usize::from(p.bias);
        }
        if elements > self.settings.max_parameter_elements {
            return Err("max_parameter_elements exceeded".into());
        }
        let nodes = self
            .cache
            .len()
            .checked_add(self.extra_nodes)
            .ok_or("body node count overflow")?;
        if nodes > self.settings.max_body_nodes {
            return Err("max_body_nodes exceeded".into());
        }
        Ok((nodes, elements, operators))
    }
}
fn checker<'a>(
    inputs: &'a [Interface],
    outputs: &'a [Interface],
    settings: &'a Settings,
) -> Checker<'a> {
    Checker {
        inputs,
        outputs,
        settings,
        parameters: BTreeMap::new(),
        cache: BTreeMap::new(),
        extra_nodes: 0,
    }
}
fn validate(inputs: &[Interface], outputs: &[Interface], s: &Settings) -> Result<(), String> {
    if inputs.is_empty()
        || outputs.len() < 2
        || s.latent_widths.contains(&0)
        || [
            s.max_operations,
            s.max_affine_parameters,
            s.max_parameter_elements,
            s.max_expression_states,
            s.max_tuple_checks,
            s.max_tuples,
            s.max_body_nodes,
        ]
        .contains(&0)
    {
        return Err("positive learned DAG budgets/widths and >=2 exits required".into());
    }
    choices(inputs, outputs, &s.latent_widths)?;
    Ok(())
}
fn subexpressions(e: &Expr, out: &mut BTreeSet<Expr>) {
    if matches!(e, Expr::Argument(_)) {
        return;
    }
    out.insert(e.clone());
    match e {
        Expr::Unary(_, x) | Expr::Affine { input: x, .. } => subexpressions(x, out),
        Expr::Binary(_, a, b) => {
            subexpressions(a, out);
            subexpressions(b, out);
        }
        Expr::Argument(_) => return,
    }
}
fn shares(expressions: &[Expr]) -> bool {
    let mut seen = BTreeSet::new();
    for e in expressions {
        let mut own = BTreeSet::new();
        subexpressions(e, &mut own);
        if own.iter().any(|x| seen.contains(x)) {
            return true;
        }
        seen.extend(own);
    }
    false
}
fn reject(counts: &mut BTreeMap<String, usize>, reason: String) {
    *counts.entry(reason).or_default() += 1;
}

/// Typed expression skeletons are generated by operation count. Parameter
/// sharing is then a separately searched restricted-growth partition of the
/// affine occurrences across the entire output tuple. Arbitrary ID permutations
/// are never introduced. Work counts include rejected skeletons and every partial
/// parameter-binding branch, not just fitted or admissible tuples.
pub fn enumerate_interfaces(
    inputs: &[Interface],
    outputs: &[Interface],
    s: &Settings,
) -> Result<Inventory, String> {
    validate(inputs, outputs, s)?;
    if s.max_expression_states < inputs.len() {
        return Err("expression budget cannot retain complete input boundary".into());
    }
    let output_choices = choices(inputs, outputs, &s.latent_widths)?;
    let unary = s.unary.iter().copied().collect::<BTreeSet<_>>();
    let binary = s.binary.iter().copied().collect::<BTreeSet<_>>();
    let mut result = Inventory { proposals: vec![], explored_states: inputs.len(), accepted_states: inputs.len(), checked_tuples: 0,
        checked_skeleton_tuples: 0, completed_parameter_bindings: 0,
        duplicate_states: 0, duplicate_tuples: 0, rejection_counts: BTreeMap::new(), truncated: false,
        priority: "Operation-count typed skeletons; round-robin maximum-operation output-tuple bands with four resumable restricted-growth partition searches per band; balanced learned-nonlinear tuple seeds when type-compatible; one attempted binding branch per turn; admitted proposals ranked by compiled DAG node count, parameter elements, then canonical ordered syntax. All attempted skeleton and partial binding work bounded; truncation is not exhaustive search.".into() };
    let mut by_size = vec![(0..inputs.len()).map(Expr::Argument).collect::<Vec<_>>()];
    let mut seen = by_size[0].iter().cloned().collect::<BTreeSet<_>>();
    enum Class {
        Unary {
            op: Unary,
            index: usize,
        },
        Affine {
            output: TypeRef,
            index: usize,
        },
        Binary {
            op: Binary,
            left_size: usize,
            left: usize,
            right: usize,
        },
    }
    'sizes: for size in 1..=s.max_operations {
        let mut classes = VecDeque::new();
        for op in &unary {
            classes.push_back(Class::Unary { op: *op, index: 0 });
        }
        for (output, _) in &output_choices {
            classes.push_back(Class::Affine {
                output: output.clone(),
                index: 0,
            });
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
        let mut next = Vec::new();
        while let Some(mut class) = classes.pop_front() {
            if result.explored_states == s.max_expression_states {
                result.truncated = true;
                break 'sizes;
            }
            let (expression, remains) = match &mut class {
                Class::Unary { op, index } => {
                    if *index == by_size[size - 1].len() {
                        continue;
                    }
                    let e = Expr::Unary(*op, Box::new(by_size[size - 1][*index].clone()));
                    *index += 1;
                    (e, *index < by_size[size - 1].len())
                }
                Class::Affine { output, index } => {
                    if *index == by_size[size - 1].len() {
                        continue;
                    }
                    let e = Expr::Affine {
                        parameter: 0,
                        output: output.clone(),
                        input: Box::new(by_size[size - 1][*index].clone()),
                        bias: s.affine_bias,
                    };
                    *index += 1;
                    (e, *index < by_size[size - 1].len())
                }
                Class::Binary {
                    op,
                    left_size,
                    left,
                    right,
                } => {
                    let l = &by_size[*left_size];
                    let r = &by_size[size - 1 - *left_size];
                    if *left >= l.len() || r.is_empty() {
                        continue;
                    }
                    let mut a = l[*left].clone();
                    let mut b = r[*right].clone();
                    if *op != Binary::Subtract && a > b {
                        std::mem::swap(&mut a, &mut b);
                    }
                    let e = Expr::Binary(*op, Box::new(a), Box::new(b));
                    *right += 1;
                    if *right == r.len() {
                        *right = 0;
                        *left += 1;
                    }
                    (e, *left < l.len())
                }
            };
            if remains {
                classes.push_back(class);
            }
            result.explored_states += 1;
            match sketch_type(&expression, inputs, outputs) {
                Err(reason) => reject(&mut result.rejection_counts, reason),
                Ok(_) => {
                    if seen.insert(expression.clone()) {
                        next.push(expression);
                        result.accepted_states += 1;
                    } else {
                        result.duplicate_states += 1;
                    }
                }
            }
        }
        next.sort();
        by_size.push(next);
    }
    // Include a partially generated final size rather than silently dropping it.
    let mut pool = seen.into_iter().collect::<Vec<_>>();
    pool.sort_by_key(|e| (operations(e), e.clone()));
    let types = pool
        .iter()
        .map(|e| sketch_type(e, inputs, outputs))
        .collect::<Result<Vec<_>, _>>()?;
    let axes = outputs
        .iter()
        .map(|target| {
            types
                .iter()
                .enumerate()
                .filter_map(|(i, ty)| (ty == target).then_some(i))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if axes.iter().any(|axis| axis.is_empty()) {
        return Ok(result);
    }
    let mut tuple_seen = BTreeSet::new();
    // Each maximum-operation band has its own frontier. Seeding each axis at
    // its first expression in the band reaches every tuple with that maximum,
    // without traversing the entire Cartesian product of shallower expressions.
    let mut bands = VecDeque::new();
    let largest_generated_size = pool.iter().map(operations).max().unwrap_or(0);
    for size in 0..=largest_generated_size {
        let mut band = TupleBand::default();
        for axis in 0..axes.len() {
            if let Some(index) = axes[axis]
                .iter()
                .position(|&id| operations(&pool[id]) == size)
            {
                let mut cursor = vec![0; axes.len()];
                cursor[axis] = index;
                if cursor
                    .iter()
                    .enumerate()
                    .all(|(a, &i)| operations(&pool[axes[a][i]]) <= size)
                    && band.seen.insert(cursor.clone())
                {
                    band.heap.push(std::cmp::Reverse((
                        tuple_cost(&cursor, &axes, &pool)?,
                        cursor,
                    )));
                }
            }
        }
        // Seed one balanced learned nonlinear tuple before asymmetric frontier
        // expansion when every exit has such a value at this complexity. This
        // is a declared scheduling preference, not a supplied native equation.
        let balanced = axes
            .iter()
            .map(|axis| {
                axis.iter().position(|&id| {
                    operations(&pool[id]) == size && has_affine(&pool[id]) && has_unary(&pool[id])
                })
            })
            .collect::<Option<Vec<_>>>();
        if let Some(cursor) = balanced {
            if band.seen.insert(cursor.clone()) {
                band.heap.push(std::cmp::Reverse((0, cursor)));
            }
        }
        if !band.heap.is_empty() {
            bands.push_back((size, band));
        }
    }
    while let Some((size, mut band)) = bands.pop_front() {
        if result.checked_tuples == s.max_tuple_checks || result.proposals.len() == s.max_tuples {
            result.truncated = true;
            break;
        }
        // A small fixed window bounds live partition searches independently of
        // the tuple budget. Rotate admission and one-branch binding work, rather
        // than finishing every sharing partition before admitting another tuple.
        if band.tasks.len() < 4
            && !band.heap.is_empty()
            && (band.admit_next || band.tasks.is_empty())
        {
            if let Some(std::cmp::Reverse((_, cursor))) = band.heap.pop() {
                result.checked_tuples += 1;
                result.checked_skeleton_tuples += 1;
                let mut occurrence = 0;
                let expressions = cursor
                    .iter()
                    .enumerate()
                    .map(|(axis, &i)| fresh_occurrences(&pool[axes[axis][i]], &mut occurrence))
                    .collect::<Vec<_>>();
                let mut c = checker(inputs, outputs, s);
                for e in &expressions {
                    c.expr(e)?;
                }
                let specs = (0..occurrence)
                    .map(|i| c.parameters[&i].clone())
                    .collect::<Vec<_>>();
                if specs.is_empty() {
                    reject(
                        &mut result.rejection_counts,
                        "tuple has no learned affine parameter".into(),
                    );
                } else {
                    band.tasks.push_back(BindingTask {
                        specs,
                        expressions,
                        stack: vec![BindingFrame {
                            owners: vec![],
                            names: vec![],
                            elements: 0,
                            next_owner: 0,
                        }],
                    });
                }
                for axis in 0..cursor.len() {
                    let mut next = cursor.clone();
                    next[axis] += 1;
                    if next[axis] < axes[axis].len()
                        && operations(&pool[axes[axis][next[axis]]]) <= size
                        && band.seen.insert(next.clone())
                    {
                        band.heap
                            .push(std::cmp::Reverse((tuple_cost(&next, &axes, &pool)?, next)));
                    }
                }
            }
            band.admit_next = false;
        } else if let Some(mut task) = band.tasks.pop_front() {
            BindingSearch {
                specs: &task.specs,
                expressions: &task.expressions,
                inputs,
                outputs,
                settings: s,
                seen: &mut tuple_seen,
                result: &mut result,
            }
            .step(&mut task.stack)?;
            if !task.stack.is_empty() {
                band.tasks.push_back(task);
            }
            band.admit_next = true;
        }
        if !band.heap.is_empty() || !band.tasks.is_empty() {
            bands.push_back((size, band));
        }
    }
    result.proposals.sort_by(|a, b| {
        (a.compiled_node_count, a.parameter_elements, &a.expressions).cmp(&(
            b.compiled_node_count,
            b.parameter_elements,
            &b.expressions,
        ))
    });
    Ok(result)
}
fn sketch_type(e: &Expr, inputs: &[Interface], outputs: &[Interface]) -> Result<Interface, String> {
    match e {
        Expr::Argument(i) => inputs
            .get(*i)
            .cloned()
            .ok_or("skeleton argument absent".into()),
        Expr::Unary(_, x) => sketch_type(x, inputs, outputs),
        Expr::Affine { output, input, .. } => {
            sketch_type(input, inputs, outputs)?;
            resolve(output, inputs, outputs)
        }
        Expr::Binary(_, a, b) => {
            let a = sketch_type(a, inputs, outputs)?;
            if a != sketch_type(b, inputs, outputs)? {
                Err("skeleton binary interfaces differ".into())
            } else {
                Ok(a)
            }
        }
    }
}
fn fresh_occurrences(e: &Expr, next: &mut usize) -> Expr {
    match e {
        Expr::Argument(i) => Expr::Argument(*i),
        Expr::Unary(op, x) => Expr::Unary(*op, Box::new(fresh_occurrences(x, next))),
        Expr::Binary(op, a, b) => Expr::Binary(
            *op,
            Box::new(fresh_occurrences(a, next)),
            Box::new(fresh_occurrences(b, next)),
        ),
        Expr::Affine {
            output,
            input,
            bias,
            ..
        } => {
            let input = fresh_occurrences(input, next);
            let parameter = *next;
            *next += 1;
            Expr::Affine {
                parameter,
                output: output.clone(),
                input: Box::new(input),
                bias: *bias,
            }
        }
    }
}
fn bind_names(e: &Expr, names: &[usize]) -> Expr {
    match e {
        Expr::Argument(i) => Expr::Argument(*i),
        Expr::Unary(op, x) => Expr::Unary(*op, Box::new(bind_names(x, names))),
        Expr::Binary(op, a, b) => Expr::Binary(
            *op,
            Box::new(bind_names(a, names)),
            Box::new(bind_names(b, names)),
        ),
        Expr::Affine {
            parameter,
            output,
            input,
            bias,
        } => Expr::Affine {
            parameter: names[*parameter],
            output: output.clone(),
            input: Box::new(bind_names(input, names)),
            bias: *bias,
        },
    }
}
#[derive(Default)]
struct TupleBand {
    heap: std::collections::BinaryHeap<std::cmp::Reverse<(usize, Vec<usize>)>>,
    seen: BTreeSet<Vec<usize>>,
    tasks: VecDeque<BindingTask>,
    admit_next: bool,
}
fn has_affine(e: &Expr) -> bool {
    match e {
        Expr::Argument(_) => false,
        Expr::Affine { .. } => true,
        Expr::Unary(_, x) => has_affine(x),
        Expr::Binary(_, a, b) => has_affine(a) || has_affine(b),
    }
}
fn has_unary(e: &Expr) -> bool {
    match e {
        Expr::Argument(_) => false,
        Expr::Unary(_, _) => true,
        Expr::Affine { input, .. } => has_unary(input),
        Expr::Binary(_, a, b) => has_unary(a) || has_unary(b),
    }
}
fn tuple_cost(cursor: &[usize], axes: &[Vec<usize>], pool: &[Expr]) -> Result<usize, String> {
    cursor.iter().enumerate().try_fold(0usize, |sum, (a, &i)| {
        sum.checked_add(operations(&pool[axes[a][i]]))
            .ok_or("tuple operation count overflow".into())
    })
}
struct BindingFrame {
    owners: Vec<Parameter>,
    names: Vec<usize>,
    elements: usize,
    next_owner: usize,
}
struct BindingTask {
    specs: Vec<Parameter>,
    expressions: Vec<Expr>,
    stack: Vec<BindingFrame>,
}
struct BindingSearch<'a> {
    specs: &'a [Parameter],
    expressions: &'a [Expr],
    inputs: &'a [Interface],
    outputs: &'a [Interface],
    settings: &'a Settings,
    seen: &'a mut BTreeSet<Vec<Expr>>,
    result: &'a mut Inventory,
}
impl BindingSearch<'_> {
    /// Resume one attempted restricted-growth owner binding. Completed and
    /// exhausted frames do not consume work; every attempted branch does.
    fn step(&mut self, stack: &mut Vec<BindingFrame>) -> Result<(), String> {
        while stack
            .last()
            .is_some_and(|f| f.names.len() < self.specs.len() && f.next_owner > f.owners.len())
        {
            stack.pop();
        }
        let Some(frame) = stack.last_mut() else {
            return Ok(());
        };
        let index = frame.names.len();
        if index == self.specs.len() {
            self.result.completed_parameter_bindings += 1;
            let candidate = self
                .expressions
                .iter()
                .map(|e| bind_names(e, &frame.names))
                .collect::<Vec<_>>();
            let candidate = canonicalize(self.inputs, self.outputs, &candidate)?;
            if !self.seen.insert(candidate.clone()) {
                self.result.duplicate_tuples += 1;
                stack.pop();
                return Ok(());
            }
            if self.settings.require_shared && !shares(&candidate) {
                reject(
                    &mut self.result.rejection_counts,
                    "tuple has no shared non-Argument computation".into(),
                );
                stack.pop();
                return Ok(());
            }
            let mut c = checker(self.inputs, self.outputs, self.settings);
            for e in &candidate {
                c.expr(e)?;
            }
            match c.limits() {
                Ok((nodes, elements, operators)) => self.result.proposals.push(Proposal {
                    output_has_learned_ancestor: candidate.iter().map(has_affine).collect(),
                    expressions: candidate,
                    compiled_node_count: nodes,
                    parameter_elements: elements,
                    trainable_operator_count: operators,
                }),
                Err(reason) => reject(&mut self.result.rejection_counts, reason),
            }
            stack.pop();
            return Ok(());
        }
        let spec = self.specs[index].clone();
        let owner = frame.next_owner;
        frame.next_owner += 1;
        self.result.checked_tuples += 1;
        let fresh = owner == frame.owners.len();
        if fresh && frame.owners.len() == self.settings.max_affine_parameters {
            reject(
                &mut self.result.rejection_counts,
                "max_affine_parameters exceeded".into(),
            );
            return Ok(());
        }
        if !fresh && frame.owners[owner] != spec {
            reject(
                &mut self.result.rejection_counts,
                "parameter binding interfaces differ".into(),
            );
            return Ok(());
        }
        let added = if fresh {
            spec.rows
                .width()
                .checked_mul(spec.cols.width())
                .and_then(|v| v.checked_add(if spec.bias { spec.rows.width() } else { 0 }))
                .ok_or("parameter element count overflow")?
        } else {
            0
        };
        let total = frame
            .elements
            .checked_add(added)
            .ok_or("parameter element count overflow")?;
        if total > self.settings.max_parameter_elements {
            reject(
                &mut self.result.rejection_counts,
                "max_parameter_elements exceeded".into(),
            );
            return Ok(());
        }
        let mut child = BindingFrame {
            owners: frame.owners.clone(),
            names: frame.names.clone(),
            elements: total,
            next_owner: 0,
        };
        if fresh {
            child.owners.push(spec);
        }
        child.names.push(owner);
        stack.push(child);
        Ok(())
    }
}

struct Builder {
    nodes: Vec<Node>,
    operators: Vec<Operator>,
    cache: BTreeMap<Expr, usize>,
    parameters: BTreeMap<usize, (usize, Option<usize>)>,
    identities: Vec<(Interface, usize)>,
    seed: u64,
}
fn initialized(spec: &Parameter, seed: u64, id: usize, bias: bool) -> Result<Operator, String> {
    let cols = if bias {
        Interface::constant()
    } else {
        spec.cols.clone()
    };
    let mut random = seed
        ^ (id as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ if bias { 0x51ed270b } else { 0x243f6a8885a308d3 };
    if random == 0 {
        random = 0x6a09e667f3bcc909;
    }
    // Zero-mean fan-in uniform initialization; no native coefficients or
    // preferred identity orientation. Biases start at zero.
    let amplitude = (3. / cols.width() as f64).sqrt();
    let values = Array2::from_shape_fn((spec.rows.width(), cols.width()), |_| {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        if bias {
            0.
        } else {
            amplitude * (2. * (random >> 11) as f64 / 9007199254740992. - 1.)
        }
    });
    Operator::dense(
        format!(
            "learned affine {id} {}",
            if bias { "bias" } else { "matrix" }
        ),
        spec.rows.clone(),
        cols,
        values.clone(),
        exact_precision(values.iter().copied()).map_err(|e| e.to_string())?,
        Default::default(),
    )
    .map_err(|e| e.to_string())
}
impl Builder {
    fn expr(&mut self, e: &Expr, checked: &Checker<'_>) -> Result<usize, String> {
        if let Some(id) = self.cache.get(e) {
            return Ok(*id);
        }
        let node = match e {
            Expr::Argument(index) => Node::Param { index: *index },
            Expr::Unary(op, x) => Node::Pointwise {
                input: self.expr(x, checked)?,
                laws: vec![
                    match op {
                        Unary::Relu => Law::Relu,
                        Unary::Silu => Law::Silu,
                        Unary::Gelu => Law::Gelu,
                        Unary::GeluTanh => Law::GeluTanh,
                    };
                    checked.cache[e].group_count()
                ],
            },
            Expr::Binary(op, a, b) => {
                let left = self.expr(a, checked)?;
                let mut right = self.expr(b, checked)?;
                if *op == Binary::Multiply {
                    Node::Hadamard { left, right }
                } else {
                    if *op == Binary::Subtract {
                        right = self.nodes.len();
                        self.nodes.push(Node::Gain {
                            input: self.cache[b.as_ref()],
                            coefficient: Coefficient::Number(-1.),
                        });
                    }
                    let ty = &checked.cache[e];
                    let identity = if let Some((_, id)) =
                        self.identities.iter().find(|(held, _)| held == ty)
                    {
                        *id
                    } else {
                        let id = self.operators.len();
                        self.operators
                            .push(Operator::identity("fixed addition identity", ty.clone()));
                        self.identities.push((ty.clone(), id));
                        id
                    };
                    Node::Affine {
                        terms: vec![(left, identity), (right, identity)],
                        bias: None,
                    }
                }
            }
            Expr::Affine {
                parameter, input, ..
            } => {
                let input = self.expr(input, checked)?;
                let (matrix, bias) = if let Some(held) = self.parameters.get(parameter) {
                    *held
                } else {
                    let spec = &checked.parameters[parameter];
                    let matrix = self.operators.len();
                    self.operators
                        .push(initialized(spec, self.seed, *parameter, false)?);
                    let bias = if spec.bias {
                        let id = self.operators.len();
                        self.operators
                            .push(initialized(spec, self.seed, *parameter, true)?);
                        Some(id)
                    } else {
                        None
                    };
                    self.parameters.insert(*parameter, (matrix, bias));
                    (matrix, bias)
                };
                Node::Affine {
                    terms: vec![(input, matrix)],
                    bias,
                }
            }
        };
        let id = self.nodes.len();
        self.nodes.push(node);
        self.cache.insert(e.clone(), id);
        Ok(id)
    }
}
fn build(
    inputs: &[Interface],
    outputs: &[Interface],
    expressions: &[Expr],
    s: &Settings,
) -> Result<(Builder, Vec<usize>), String> {
    validate(inputs, outputs, s)?;
    if expressions.len() != outputs.len() {
        return Err("one learned equation per actual exit required".into());
    }
    let expressions = canonicalize(inputs, outputs, expressions)?;
    let mut checked = checker(inputs, outputs, s);
    for (e, ty) in expressions.iter().zip(outputs) {
        if checked.expr(e)? != *ty {
            return Err("learned equation output interface mismatch".into());
        }
    }
    checked.limits()?;
    if checked.parameters.is_empty() {
        return Err("learned DAG has no affine parameters".into());
    }
    if s.require_shared && !shares(&expressions) {
        return Err("learned DAG has no shared non-Argument computation".into());
    }
    let mut builder = Builder {
        nodes: vec![],
        operators: vec![],
        cache: BTreeMap::new(),
        parameters: BTreeMap::new(),
        identities: vec![],
        seed: s.seed,
    };
    let exits = expressions
        .iter()
        .map(|e| builder.expr(e, &checked))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((builder, exits))
}
pub fn compile_program(
    inputs: &[Interface],
    outputs: &[Interface],
    expressions: &[Expr],
    s: &Settings,
) -> Result<Compiled, String> {
    if inputs
        .iter()
        .any(|ty| !matches!(Interface::native(ty.width()), Ok(ref held) if held == ty))
    {
        return Err("standalone learned program requires native Raw input grouping; use artifact apply for grouped boundaries".into());
    }
    let (mut builder, output_nodes) = build(inputs, outputs, expressions, s)?;
    for node in &mut builder.nodes {
        if let Node::Param { index } = node {
            *node = Node::Raw { slot: *index };
        }
    }
    let trainable_operator_ids = builder
        .parameters
        .values()
        .flat_map(|(a, b)| std::iter::once(*a).chain(*b))
        .collect();
    let output = builder.nodes.len();
    builder.nodes.push(Node::Concat {
        parts: output_nodes.clone(),
    });
    let program = OperatorProgram {
        declarations: Declarations {
            domains: vec![],
            slots: inputs
                .iter()
                .map(|ty| Slot::Raw { width: ty.width() })
                .collect(),
            parameters: 0,
        },
        bases: vec![],
        rules: vec![],
        operators: builder.operators.into_iter().map(Arc::new).collect(),
        nodes: builder.nodes,
        output,
    };
    program.interfaces().map_err(|e| e.to_string())?;
    Ok(Compiled {
        program,
        output_nodes,
        trainable_operator_ids,
    })
}
fn boundaries(
    a: &Artifact,
    r: &joint::Region,
    args: &[usize],
) -> Result<(Vec<Interface>, Vec<Interface>), String> {
    joint::exact_body(a, r)?;
    if args.len() != r.native_reads.len()
        || args.iter().copied().collect::<BTreeSet<_>>() != r.native_reads.iter().copied().collect()
    {
        return Err("learned arguments must biject the complete native boundary".into());
    }
    let types = a.program.interfaces().map_err(|e| e.to_string())?;
    let resolve = |native| {
        a.place(native)
            .map(|i| types[i].clone())
            .ok_or("learned native boundary absent".to_string())
    };
    Ok((
        args.iter()
            .copied()
            .map(resolve)
            .collect::<Result<_, _>>()?,
        r.native_writes
            .iter()
            .copied()
            .map(resolve)
            .collect::<Result<_, _>>()?,
    ))
}
pub fn enumerate(
    a: &Artifact,
    r: &joint::Region,
    s: &Settings,
    args: &[usize],
) -> Result<Inventory, String> {
    let (inputs, outputs) = boundaries(a, r, args)?;
    enumerate_interfaces(&inputs, &outputs, s)
}
pub fn apply(
    a: &Artifact,
    r: &joint::Region,
    expressions: &[Expr],
    args: &[usize],
    s: &Settings,
) -> Result<Applied, String> {
    let (inputs, outputs) = boundaries(a, r, args)?;
    let (builder, exits) = build(&inputs, &outputs, expressions, s)?;
    let local_trainables = builder
        .parameters
        .values()
        .flat_map(|(a, b)| std::iter::once(*a).chain(*b))
        .collect::<Vec<_>>();
    let mut body = JointBody {
        inputs,
        nodes: builder.nodes,
        operators: builder.operators,
        exits: r
            .native_writes
            .iter()
            .copied()
            .zip(exits)
            .map(|(native_write, node)| Exit { native_write, node })
            .collect(),
    };
    for node in &mut body.nodes {
        if let Node::Param { index } = node {
            *index = r
                .native_reads
                .iter()
                .position(|n| *n == args[*index])
                .ok_or("learned canonical input absent")?;
        }
        // JointBody operator references are into parent pool followed by appended operators.
        if let Node::Affine { terms, bias } = node {
            for (_, op) in terms {
                *op += a.program.operators.len();
            }
            if let Some(op) = bias {
                *op += a.program.operators.len();
            }
        }
    }
    let native_types = a.program.interfaces().map_err(|e| e.to_string())?;
    body.inputs = r
        .current_reads
        .iter()
        .map(|&i| native_types[i].clone())
        .collect();
    let mapped = joint::apply_mapped(a, r, &body)?;
    let trainable_operator_ids = local_trainables
        .iter()
        .map(|&id| {
            let id = mapped.operator_mapping[a.program.operators.len() + id];
            if id == usize::MAX {
                Err("learned parameter erased during joint compaction".into())
            } else {
                Ok(id)
            }
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Applied {
        artifact: mapped.artifact,
        trainable_operator_ids,
        node_mapping: mapped.node_mapping,
    })
}

#[cfg(test)]
#[path = "program_learned_dag_calibration_tests.rs"]
mod calibration_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_program::{FamilyInputs, SlotValues};
    use ndarray::array;

    fn settings() -> Settings {
        Settings {
            latent_widths: vec![1, 2, 3],
            unary: vec![Unary::Relu, Unary::Silu],
            binary: vec![Binary::Add, Binary::Subtract, Binary::Multiply],
            affine_bias: false,
            require_shared: false,
            max_operations: 8,
            max_affine_parameters: 4,
            max_parameter_elements: 200,
            max_expression_states: 1000,
            max_tuple_checks: 10000,
            max_tuples: 1000,
            max_body_nodes: 30,
            seed: 29,
        }
    }
    fn projection(parameter: usize, output: TypeRef, input: Expr) -> Expr {
        Expr::Affine {
            parameter,
            output,
            input: Box::new(input),
            bias: false,
        }
    }
    #[test]
    fn distinct_named_projections_and_shared_latent_survive_codec() {
        let ty = Interface::native(2).expect("native vector");
        let a = projection(71, TypeRef::Input(0), Expr::Argument(0));
        let b = projection(9, TypeRef::Exit(1), Expr::Argument(0));
        let z = Expr::Binary(Binary::Multiply, Box::new(a), Box::new(b));
        let exprs = vec![
            Expr::Unary(Unary::Silu, Box::new(z.clone())),
            Expr::Unary(Unary::Relu, Box::new(z)),
        ];
        let c = compile_program(&[ty.clone()], &[ty.clone(), ty], &exprs, &settings())
            .expect("typed learned equations");
        assert_eq!(c.trainable_operator_ids.len(), 2);
        assert_ne!(
            c.program.operators[c.trainable_operator_ids[0]].matrix(),
            c.program.operators[c.trainable_operator_ids[1]].matrix()
        );
        assert_eq!(
            c.program
                .nodes
                .iter()
                .filter(|n| matches!(n, Node::Hadamard { .. }))
                .count(),
            1
        );
        assert_eq!(
            c.program
                .nodes
                .iter()
                .filter(|n| matches!(n, Node::Affine { .. }))
                .count(),
            2
        );
        let native = Artifact::native(&c.program).expect("ordinary executable artifact");
        let saved = Artifact::from_bytes(
            &native.to_bytes().expect("codec bytes"),
            &c.program.declarations,
        )
        .expect("codec replay");
        let inputs = FamilyInputs {
            rows: 2,
            slots: vec![SlotValues::Raw(array![[1., -2.], [0.5, 3.]])],
            layout: None,
        };
        let before = c
            .program
            .execute(&inputs, false)
            .expect("original computation");
        let after = saved.execute(&inputs).expect("saved computation");
        assert_eq!(
            before.values[c.program.output],
            after.values[saved.program.output]
        );
        let Node::Pointwise { input: first, .. } = c.program.nodes[c.output_nodes[0]] else {
            panic!("first nonlinear exit")
        };
        let Node::Pointwise { input: second, .. } = c.program.nodes[c.output_nodes[1]] else {
            panic!("second nonlinear exit")
        };
        assert_eq!(
            first, second,
            "two different equations consume one actual latent node"
        );
    }
    #[test]
    fn parameter_renaming_is_invariant_and_type_aliases_cse() {
        let ty = Interface::native(2).expect("native vector");
        let make = |a, b| {
            vec![
                Expr::Binary(
                    Binary::Add,
                    Box::new(projection(a, TypeRef::Exit(0), Expr::Argument(0))),
                    Box::new(projection(
                        b,
                        TypeRef::Latent { width: 2 },
                        Expr::Argument(0),
                    )),
                ),
                projection(a, TypeRef::Input(0), Expr::Argument(0)),
            ]
        };
        let first = canonicalize(&[ty.clone()], &[ty.clone(), ty.clone()], &make(0, 1))
            .expect("canonical names");
        let renamed = canonicalize(&[ty.clone()], &[ty.clone(), ty.clone()], &make(19, 3))
            .expect("renamed canonical names");
        assert_eq!(first, renamed);
        assert_eq!(
            first,
            canonicalize(&[ty.clone()], &[ty.clone(), ty.clone()], &first)
                .expect("idempotent names")
        );
        let exprs = vec![
            projection(4, TypeRef::Input(0), Expr::Argument(0)),
            projection(4, TypeRef::Exit(1), Expr::Argument(0)),
        ];
        let c = compile_program(&[ty.clone()], &[ty.clone(), ty], &exprs, &settings())
            .expect("alias interface compiler");
        assert_eq!(c.trainable_operator_ids.len(), 1);
        assert_eq!(c.output_nodes[0], c.output_nodes[1]);
    }
    #[test]
    fn conflicting_parameter_specs_and_resource_overflow_are_refused() {
        let ty = Interface::native(2).expect("native vector");
        let scalar = Interface::native(1).expect("native scalar");
        let exprs = vec![
            projection(0, TypeRef::Input(0), Expr::Argument(0)),
            projection(0, TypeRef::Exit(1), Expr::Argument(0)),
        ];
        assert!(
            compile_program(&[ty.clone()], &[ty.clone(), scalar], &exprs, &settings())
                .err()
                .expect("inconsistent ID refused")
                .contains("inconsistent")
        );
        let mut s = settings();
        s.max_parameter_elements = 3;
        let exprs = vec![projection(0, TypeRef::Input(0), Expr::Argument(0)); 2];
        assert!(
            compile_program(&[ty.clone()], &[ty.clone(), ty], &exprs, &s)
                .err()
                .expect("finite element budget refused")
                .contains("max_parameter_elements")
        );
        s.latent_widths = vec![0];
        assert!(
            enumerate_interfaces(
                &[Interface::native(1).expect("input")],
                &vec![Interface::native(1).expect("output"); 2],
                &s
            )
            .is_err()
        );
    }
    #[test]
    fn finite_budget_admits_trainable_nonlinear_native_exit_and_shared_response() {
        let native = Interface::native(8).expect("native vector");
        let scalar = Interface::native(1).expect("scalar response");
        let mut s = settings();
        s.latent_widths = vec![1];
        s.unary = vec![Unary::Silu];
        s.binary.clear();
        s.require_shared = true;
        s.max_operations = 3;
        s.max_expression_states = 512;
        s.max_tuple_checks = 128;
        s.max_tuples = 12;
        let inputs = vec![native.clone()];
        let outputs = vec![native, scalar.clone(), scalar];
        let inventory =
            enumerate_interfaces(&inputs, &outputs, &s).expect("bounded fair inventory");
        assert!(
            inventory.proposals.iter().any(|p| {
                has_affine(&p.expressions[0])
                    && has_unary(&p.expressions[0])
                    && shares(&p.expressions[1..])
                    && p.expressions[1..].iter().any(has_unary)
            }),
            "finite budget must reach a trainable nonlinear clean path and shared nonlinear responses"
        );
        for p in &inventory.proposals {
            assert_eq!(
                p.output_has_learned_ancestor,
                p.expressions.iter().map(has_affine).collect::<Vec<_>>()
            );
        }
        assert!(inventory.checked_tuples <= s.max_tuple_checks);
        assert!(inventory.truncated);
        let replay = enumerate_interfaces(&inputs, &outputs, &s).expect("deterministic replay");
        assert_eq!(
            serde_json::to_string(&inventory).expect("inventory encoding"),
            serde_json::to_string(&replay).expect("replay encoding")
        );
    }

    #[test]
    fn unsupplied_projection_partitions_are_enumerated_with_explicit_work_caps() {
        let ty = Interface::native(1).expect("native scalar");
        let mut s = settings();
        s.latent_widths = vec![1];
        s.unary = vec![Unary::Silu, Unary::Gelu];
        s.binary.clear();
        s.max_operations = 2;
        s.max_affine_parameters = 3;
        s.max_expression_states = 4096;
        s.max_tuple_checks = 1000000;
        s.max_tuples = 100000;
        let inventory =
            enumerate_interfaces(&[ty.clone()], &vec![ty; 3], &s).expect("finite grammar search");
        let mut classes = BTreeSet::new();
        for p in &inventory.proposals {
            let mut names = Vec::new();
            let mut laws = Vec::new();
            for e in &p.expressions {
                if let Expr::Unary(law, inner) = e {
                    if let Expr::Affine {
                        parameter, input, ..
                    } = inner.as_ref()
                    {
                        if **input == Expr::Argument(0) {
                            names.push(*parameter);
                            laws.push(*law);
                        }
                    }
                }
            }
            if names.len() == 3 {
                classes.insert((names, laws));
            }
        }
        assert_eq!(
            classes.len(),
            40,
            "all five sharing partitions times eight nonlinear law tuples"
        );
        assert!(inventory.explored_states <= s.max_expression_states);
        assert!(inventory.checked_tuples <= s.max_tuple_checks);
        let mut scarce = s;
        scarce.max_expression_states = 4;
        scarce.max_tuple_checks = 8;
        scarce.max_tuples = 1;
        let tiny = enumerate_interfaces(
            &[Interface::native(1).expect("input")],
            &vec![Interface::native(1).expect("output"); 3],
            &scarce,
        )
        .expect("bounded truncated inventory");
        assert!(tiny.truncated);
        assert!(tiny.checked_tuples <= scarce.max_tuple_checks);
    }
}
