//! Bounded proposals for typed vector equations with explicitly named affine
//! parameters. Outputs can have different nonlinear equations and share actual
//! intermediate computations. This is a restricted fitted candidate family, not
//! identification of native variables or a guarantee of structure recovery.
#[path = "program_learned_dag_parent.rs"]
pub mod parent;

use crate::{
    artifact::Artifact,
    composed_rule_search::{Binary, Unary},
    operator_program::{
        exact_precision, remap_node, Coefficient, Declarations, Interface, Law, Node, Operator,
        OperatorBody, OperatorProgram, Slot,
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
    /// Inspected sharing-seed projection values, output skeleton tuples, and every attempted partial parameter binding.
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
    /// Generated learned-nonlinear projection values inspected for sharing-first seeds.
    #[serde(default)]
    pub sharing_seed_checks: usize,
    pub completed_parameter_bindings: usize,
    pub duplicate_states: usize,
    pub duplicate_tuples: usize,
    pub rejection_counts: BTreeMap<String, usize>,
    pub truncated: bool,
    pub priority: String,
}
pub struct Applied {
    pub local_fit: LocalFit,
    pub initialization: InitializationReport,
    pub artifact: Artifact,
    pub trainable_operator_ids: Vec<usize>,
    pub node_mapping: Vec<usize>,
}
/// Local replacement and exact compaction mapping, retained only for its fitting callback.
/// Inputs follow the actual argument binding; outputs follow declared native exits.
pub struct LocalFit {
    pub program: OperatorProgram,
    pub trainable_operator_ids: Vec<usize>,
    pub owner_mapping: Vec<(usize, usize)>,
    pub input_native_places: Vec<usize>,
    pub output_native_places: Vec<usize>,
    pub output_nodes: Vec<usize>,
}
impl LocalFit {
    /// Rebind after the caller has verified a codec roundtrip. This does not encode
    /// again or assert unrounded numerical equality; reference identities and mapped
    /// owner structures are checked here, and stale transfer checks remain strict.
    pub(crate) fn rebind_to_saved_candidate(
        &mut self,
        before: &Artifact,
        after: &Artifact,
    ) -> Result<(), String> {
        if before.program.nodes != after.program.nodes
            || before.program.output != after.program.output
            || before.program.operators.len() != after.program.operators.len()
            || before.places != after.places
            || before.program.declarations != after.program.declarations
            || before.program.rules.len() != after.program.rules.len()
            || before
                .program
                .rules
                .iter()
                .zip(&after.program.rules)
                .any(|(a, b)| a.nodes != b.nodes || a.inputs != b.inputs || a.output != b.output)
        {
            return Err(
                "local owner rebinding requires an identity-preserving canonical roundtrip".into(),
            );
        }
        for &(local, grafted) in &self.owner_mapping {
            let original = self
                .program
                .operators
                .get(local)
                .ok_or("local owner out of range")?;
            let old = before
                .program
                .operators
                .get(grafted)
                .ok_or("grafted owner out of range")?;
            let new = after
                .program
                .operators
                .get(grafted)
                .ok_or("canonical owner out of range")?;
            if original != old
                || old.rows != new.rows
                || old.cols != new.cols
                || !matches!((&old.body,&new.body),(OperatorBody::Dense {present:a,values:av,..},OperatorBody::Dense {present:b,values:bv,..}) if a==b && av.dim()==bv.dim())
            {
                return Err("canonical local owner mapping is incompatible or stale".into());
            }
        }
        for &(local, grafted) in &self.owner_mapping {
            self.program.operators[local] = Arc::clone(&after.program.operators[grafted]);
        }
        Ok(())
    }
    /// Transfer coefficients only, after checking every destination before mutation.
    pub fn transfer(
        &self,
        fitted: &OperatorProgram,
        candidate: &mut Artifact,
    ) -> Result<(), String> {
        let original = &self.program;
        if fitted.declarations != original.declarations
            || fitted.bases != original.bases
            || fitted.rules != original.rules
            || fitted.nodes != original.nodes
            || fitted.output != original.output
            || fitted.operators.len() != original.operators.len()
        {
            return Err("local fit changed replacement graph".into());
        }
        fitted.interfaces().map_err(|e| e.to_string())?;
        let owners = self
            .trainable_operator_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if self.owner_mapping.len() != owners.len()
            || self
                .owner_mapping
                .iter()
                .map(|(id, _)| *id)
                .collect::<BTreeSet<_>>()
                != owners
            || self
                .owner_mapping
                .iter()
                .map(|(_, id)| *id)
                .collect::<BTreeSet<_>>()
                .len()
                != self.owner_mapping.len()
        {
            return Err("local fit owner mapping is not a bijection".into());
        }
        for (id, (before, after)) in original.operators.iter().zip(&fitted.operators).enumerate() {
            if !owners.contains(&id) {
                if before != after {
                    return Err("local fit changed frozen operator".into());
                }
                continue;
            }
            let compatible = before.name == after.name
                && before.rows == after.rows
                && before.cols == after.cols
                && before.provenance == after.provenance
                && matches!((&before.body, &after.body),
                    (OperatorBody::Dense {present:a, values:av, ..}, OperatorBody::Dense {present:b, values:bv, ..}) if a == b && av.dim() == bv.dim() && bv.iter().all(|v| v.is_finite()));
            if !compatible {
                return Err("local fit changed dense owner structure".into());
            }
        }
        for &(local, grafted) in &self.owner_mapping {
            if local >= original.operators.len() || grafted >= candidate.program.operators.len() {
                return Err("local fit owner mapping is out of range".into());
            }
            if candidate.program.operators.get(grafted) != original.operators.get(local) {
                return Err("local fit grafted owner mapping is stale".into());
            }
        }
        for &(local, grafted) in &self.owner_mapping {
            candidate.program.operators[grafted] = Arc::clone(&fitted.operators[local]);
        }
        Ok(())
    }
}
fn standalone_local(
    inputs: &[Interface],
    nodes: &[Node],
    operators: &[Operator],
    exits: &[usize],
) -> Result<(OperatorProgram, Vec<usize>), String> {
    let mut output = Vec::new();
    let mut pool = operators.iter().cloned().map(Arc::new).collect::<Vec<_>>();
    let mut mapping = vec![usize::MAX; nodes.len()];
    let operator_ids = (0..operators.len()).collect::<Vec<_>>();
    // Preserve every declared argument, even when this proposal does not read it.
    // These are genuine supplied inputs, not dummy computational dependencies.
    let mut argument_nodes = Vec::with_capacity(inputs.len());
    for (index, ty) in inputs.iter().enumerate() {
        let native = Interface::native(ty.width()).map_err(|e| e.to_string())?;
        output.push(Node::Raw { slot: index });
        if native != *ty {
            let op = pool.len();
            pool.push(Arc::new(
                Operator::dense(
                    "fixed local input grouping",
                    ty.clone(),
                    native,
                    Array2::eye(ty.width()),
                    exact_precision([0., 1.]).map_err(|e| e.to_string())?,
                    Default::default(),
                )
                .map_err(|e| e.to_string())?,
            ));
            output.push(Node::Affine {
                terms: vec![(output.len() - 1, op)],
                bias: None,
            });
        }
        argument_nodes.push(output.len() - 1);
    }
    for (id, node) in nodes.iter().enumerate() {
        if let Node::Param { index } = node {
            mapping[id] = *argument_nodes
                .get(*index)
                .ok_or("local parameter outside declared arguments")?;
        } else {
            let mut copied = node.clone();
            remap_node(&mut copied, &mapping, &operator_ids, &[], &[]);
            output.push(copied);
            mapping[id] = output.len() - 1;
        }
    }
    let output_nodes = exits.iter().map(|id| mapping[*id]).collect::<Vec<_>>();
    let output_id = output.len();
    output.push(Node::Concat {
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
        operators: pool,
        nodes: output,
        output: output_id,
    };
    program.interfaces().map_err(|e| e.to_string())?;
    Ok((program, output_nodes))
}
pub struct Compiled {
    pub initialization: InitializationReport,
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
        checked_skeleton_tuples: 0, sharing_seed_checks: 0, completed_parameter_bindings: 0,
        duplicate_states: 0, duplicate_tuples: 0, rejection_counts: BTreeMap::new(), truncated: false,
        priority: "Operation-count typed skeletons; round-robin maximum-operation output-tuple bands with four resumable restricted-growth partition searches per ordinary band and sixteen per sharing-seed band; separate round-robin projected and direct-exit seed bands, each starting with the widest generated intermediate types; sharing-first tuples of generated learned-nonlinear intermediates and generated affine exit projections, including the unprojected intermediate at type-compatible exits (seed inspection capped at one eighth of tuple work); balanced learned-nonlinear tuple seeds when type-compatible; one attempted binding branch per turn; admitted proposals ranked by compiled DAG node count, parameter elements, then canonical ordered syntax. All attempted skeleton and partial binding work bounded; truncation is not exhaustive search.".into() };
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
        let mut band = TupleBand {
            expand_frontier: true,
            task_limit: 4,
            ..TupleBand::default()
        };
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
    // Build sharing seeds by indexing generated projection values, never by
    // inventing an equation or taking another Cartesian product. Discovery has
    // an explicit work slice; ordinary tuple/partition search retains the rest.
    let seed_budget = s.max_tuple_checks / 8;
    let mut projections: BTreeMap<Expr, Vec<Option<usize>>> = BTreeMap::new();
    for (id, expression) in pool.iter().enumerate() {
        let Expr::Affine { input, .. } = expression else {
            continue;
        };
        if !matches!(input.as_ref(), Expr::Unary(_, _)) || !has_affine(input) {
            continue;
        }
        if result.sharing_seed_checks == seed_budget {
            result.truncated = true;
            break;
        }
        result.sharing_seed_checks += 1;
        result.checked_tuples += 1;
        let entry = projections
            .entry((**input).clone())
            .or_insert_with(|| vec![None; outputs.len()]);
        for (axis, target) in outputs.iter().enumerate() {
            if &types[id] == target && entry[axis].is_none() {
                entry[axis] = Some(
                    axes[axis]
                        .binary_search(&id)
                        .map_err(|_| "sharing projection absent from typed output axis")?,
                );
            }
        }
    }
    let mut sharing_projected = TupleBand {
        task_limit: 16,
        ..TupleBand::default()
    };
    let mut sharing_direct = TupleBand {
        task_limit: 16,
        ..TupleBand::default()
    };
    let mut projected_seeds = Vec::new();
    let mut direct_seeds = Vec::new();
    for (intermediate, positions) in projections {
        // The already generated nonlinear value may itself be a declared exit.
        // Retain the projected seed too: no identity adapter is inserted and no
        // Cartesian product of direct/projected choices is opened. Each seed's
        // later admission and parameter binding consume the ordinary tuple work.
        // These lookups are bounded by the charged projection inspections above.
        let projected = positions.iter().copied().collect::<Option<Vec<_>>>();
        let intermediate_id = pool.binary_search_by(|e| {
            (operations(e), e).cmp(&(operations(&intermediate), &intermediate))
        }).map_err(|_| "generated sharing intermediate absent from expression pool")?;
        let direct = positions.iter().enumerate().map(|(axis, projection)| {
            if types[intermediate_id] == outputs[axis] {
                axes[axis].binary_search(&intermediate_id).ok()
            } else {
                *projection
            }
        }).collect::<Option<Vec<_>>>();
        // A cheaper direct exit must not displace the entire projected family.
        // Rotate the two families as separate bands, starting each with the
        // widest generated intermediates before narrower ones.
        // Width is read from the generated type, never from native weights.
        let width = std::cmp::Reverse(types[intermediate_id].width());
        if let Some(cursor) = &projected {
            if sharing_projected.seen.insert(cursor.clone()) {
                projected_seeds.push((width, tuple_cost(cursor, &axes, &pool)?, cursor.clone()));
            }
        }
        if direct != projected {
            if let Some(cursor) = direct {
                if sharing_direct.seen.insert(cursor.clone()) {
                    direct_seeds.push((width, tuple_cost(&cursor, &axes, &pool)?, cursor));
                }
            }
        }
    }
    projected_seeds.sort();
    direct_seeds.sort();
    for (rank, (_, _, cursor)) in projected_seeds.into_iter().enumerate() {
        sharing_projected.heap.push(std::cmp::Reverse((rank, cursor)));
    }
    for (rank, (_, _, cursor)) in direct_seeds.into_iter().enumerate() {
        sharing_direct.heap.push(std::cmp::Reverse((rank, cursor)));
    }
    if !sharing_direct.heap.is_empty() {
        bands.push_front((largest_generated_size, sharing_direct));
    }
    if !sharing_projected.heap.is_empty() {
        bands.push_front((largest_generated_size, sharing_projected));
    }
    while let Some((size, mut band)) = bands.pop_front() {
        if result.checked_tuples == s.max_tuple_checks || result.proposals.len() == s.max_tuples {
            result.truncated = true;
            break;
        }
        // A small fixed window bounds live partition searches independently of
        // the tuple budget. Rotate admission and one-branch binding work, rather
        // than finishing every sharing partition before admitting another tuple.
        if band.tasks.len() < band.task_limit
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
                for axis in 0..if band.expand_frontier {
                    cursor.len()
                } else {
                    0
                } {
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
    expand_frontier: bool,
    task_limit: usize,
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InitializationReport {
    pub owners: Vec<OwnerInitialization>,
    pub inherited_elements: usize,
    pub zero_filled_bias_elements: usize,
    pub random_elements: usize,
    pub scope: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OwnerInitialization {
    pub parameter: usize,
    pub matrix_operator: usize,
    pub bias_operator: Option<usize>,
    pub parent_matrices: Vec<usize>,
    pub parent_biases: Vec<usize>,
    pub inherited_elements: usize,
    pub zero_filled_bias_elements: usize,
    pub random_elements: usize,
    pub status: String,
    pub interfaces_relabelled: bool,
}
fn unary_law(op: Unary) -> Law {
    match op {
        Unary::Relu => Law::Relu,
        Unary::Silu => Law::Silu,
        Unary::Gelu => Law::Gelu,
        Unary::GeluTanh => Law::GeluTanh,
    }
}
fn dense_complete(op: &Operator) -> bool {
    matches!(&op.body, OperatorBody::Dense { values, present, .. } if present.iter().all(|v| *v) && values.iter().all(|v| v.is_finite()))
}
fn same_coefficients(a: &Operator, b: &Operator) -> bool {
    a.rows.width() == b.rows.width()
        && a.cols.width() == b.cols.width()
        && match (&a.body, &b.body) {
            (
                OperatorBody::Dense {
                    values: a,
                    present: ap,
                    ..
                },
                OperatorBody::Dense {
                    values: b,
                    present: bp,
                    ..
                },
            ) => {
                a.dim() == b.dim()
                    && ap.iter().all(|v| *v)
                    && bp.iter().all(|v| *v)
                    && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
            }
            _ => false,
        }
}
fn zero_bias(parent: &OperatorProgram, bias: Option<usize>) -> bool {
    bias.is_none_or(|id| matches!(&parent.operators[id].body, OperatorBody::Dense { values, present, .. } if present.iter().all(|v| *v) && values.iter().all(|v| *v==0.)))
}
struct ParentMatch<'a> {
    parent: &'a OperatorProgram,
    inputs: &'a [usize],
    types: Vec<Interface>,
    checked: Checker<'a>,
}
impl ParentMatch<'_> {
    fn matches(&self, e: &Expr, node: usize) -> bool {
        if self.checked.cache[e].width() != self.types[node].width() {
            return false;
        }
        match (e, &self.parent.nodes[node]) {
            (Expr::Argument(i), _) => self.inputs[*i] == node,
            (Expr::Unary(op, x), Node::Pointwise { input, laws }) => {
                laws.iter().all(|law| *law == unary_law(*op)) && self.matches(x, *input)
            }
            (Expr::Affine { input, bias, .. }, Node::Affine { terms, bias: pb })
                if terms.len() == 1 && (*bias || zero_bias(self.parent, *pb)) =>
            {
                dense_complete(&self.parent.operators[terms[0].1])
                    && pb.is_none_or(|id| dense_complete(&self.parent.operators[id]))
                    && self.matches(input, terms[0].0)
            }
            (Expr::Binary(Binary::Multiply, a, b), Node::Hadamard { left, right }) => {
                self.matches(a, *left) && self.matches(b, *right)
            }
            _ => false,
        }
    }
    fn record_proven(
        &self,
        e: &Expr,
        node: usize,
        inherited: &mut BTreeMap<usize, Vec<(usize, Option<usize>)>>,
    ) {
        if self.matches(e, node) {
            self.record(e, node, inherited);
            return;
        }
        // Align unchanged child roles even when an outer law/map changes. This
        // retains only children whose complete boundary-anchored subtree matches.
        match (e, &self.parent.nodes[node]) {
            (Expr::Unary(_, x), Node::Pointwise { input, .. }) => {
                self.record_proven(x, *input, inherited)
            }
            (Expr::Affine { input, .. }, Node::Affine { terms, .. }) if terms.len() == 1 => {
                self.record_proven(input, terms[0].0, inherited)
            }
            (Expr::Binary(Binary::Multiply, a, b), Node::Hadamard { left, right }) => {
                self.record_proven(a, *left, inherited);
                self.record_proven(b, *right, inherited);
            }
            _ => return,
        }
    }
    fn record(
        &self,
        e: &Expr,
        node: usize,
        inherited: &mut BTreeMap<usize, Vec<(usize, Option<usize>)>>,
    ) {
        match (e, &self.parent.nodes[node]) {
            (
                Expr::Affine {
                    parameter, input, ..
                },
                Node::Affine { terms, bias },
            ) => {
                inherited
                    .entry(*parameter)
                    .or_default()
                    .push((terms[0].1, *bias));
                self.record(input, terms[0].0, inherited);
            }
            (Expr::Unary(_, x), Node::Pointwise { input, .. }) => self.record(x, *input, inherited),
            (Expr::Binary(Binary::Multiply, a, b), Node::Hadamard { left, right }) => {
                self.record(a, *left, inherited);
                self.record(b, *right, inherited);
            }
            _ => return,
        }
    }
}
fn initialize_from_parent(
    builder: &mut Builder,
    inputs: &[Interface],
    outputs: &[Interface],
    expressions: &[Expr],
    s: &Settings,
    parent: Option<(&OperatorProgram, &[usize], &[usize])>,
) -> Result<InitializationReport, String> {
    let expressions = canonicalize(inputs, outputs, expressions)?;
    let mut inherited: BTreeMap<usize, Vec<(usize, Option<usize>)>> = BTreeMap::new();
    if let Some((parent, parent_inputs, parent_outputs)) = parent {
        let types = parent.interfaces().map_err(|e| e.to_string())?;
        if parent_inputs.len() != inputs.len()
            || parent_outputs.len() != outputs.len()
            || parent_inputs
                .iter()
                .zip(inputs)
                .any(|(&n, ty)| types.get(n).is_none_or(|held| held.width() != ty.width()))
            || parent_outputs
                .iter()
                .zip(outputs)
                .any(|(&n, ty)| types.get(n).is_none_or(|held| held.width() != ty.width()))
        {
            return Err("parent initialization boundary/interface mismatch".into());
        }
        let mut checked = checker(inputs, outputs, s);
        for e in &expressions {
            checked.expr(e)?;
        }
        let matcher = ParentMatch {
            parent,
            inputs: parent_inputs,
            types,
            checked,
        };
        // First respect semantic exit correspondence, including different parent
        // owners that happen to have the same shape. A proposed tie must agree.
        for (e, &node) in expressions.iter().zip(parent_outputs) {
            matcher.record_proven(e, node, &mut inherited);
        }
        let mut ancestry = BTreeSet::new();
        let mut pending = parent_outputs.to_vec();
        while let Some(node) = pending.pop() {
            if parent_inputs.contains(&node) || !ancestry.insert(node) {
                continue;
            }
            pending.extend(parent.nodes[node].arguments());
        }
        // Changed outer equations can still retain exact unchanged subtrees.
        // Without exit correspondence, multiple parent owners are ambiguous;
        // retain random initialization rather than choosing by matrix shape.
        for e in builder.cache.keys() {
            let Expr::Affine { parameter, .. } = e else {
                continue;
            };
            if inherited.contains_key(parameter) {
                continue;
            }
            let matches = ancestry
                .iter()
                .filter(|&&n| matcher.matches(e, n))
                .copied()
                .collect::<Vec<_>>();
            let pairs = matches
                .iter()
                .filter_map(|&n| match &parent.nodes[n] {
                    Node::Affine { terms, bias } => Some((terms[0].1, *bias)),
                    _ => None,
                })
                .collect::<BTreeSet<_>>();
            if pairs.len() == 1 {
                if let Some(&node) = matches.first() {
                    matcher.record(e, node, &mut inherited);
                }
            }
        }
    }
    let mut report = InitializationReport { scope:"Coefficient inheritance is initialization, not discovery. Exact ordered parent subtrees require matching boundary arguments, coordinate-preserving width-compatible interfaces (explicit boundary order; grouping labels may be adapted for complete dense and uniform pointwise operations), supported operators, nonlinear laws. Extra candidate biases remain explicitly zero; omitted parent biases must be exactly zero. Semantic exit matches take priority; ambiguous unmatched subtrees stay random. Shared candidate owners must have bit-identical inherited matrices and biases across all matched parent uses. Changed/unsupported pieces retain deterministic fan-in initialization (biases zero). No numerical sensitivity, fit success, or preservation claim for changed equations.".into(), ..InitializationReport::default() };
    for (&parameter, &(matrix, bias)) in &builder.parameters {
        let elements = builder.operators[matrix].rows.width()
            * builder.operators[matrix].cols.width()
            + bias.map_or(0, |id| builder.operators[id].rows.width());
        let sources = inherited.get(&parameter).cloned().unwrap_or_default();
        let mut owner = OwnerInitialization {
            parameter,
            matrix_operator: matrix,
            bias_operator: bias,
            parent_matrices: vec![],
            parent_biases: vec![],
            inherited_elements: 0,
            zero_filled_bias_elements: 0,
            random_elements: elements,
            status: "random: no unambiguous exact parent subtree".into(),
            interfaces_relabelled: false,
        };
        if let (Some((parent, _, _)), Some(&(source, source_bias))) = (parent, sources.first()) {
            for &(other, other_bias) in &sources {
                if !same_coefficients(&parent.operators[source], &parent.operators[other])
                    || match (source_bias, other_bias) {
                        (None, None) => false,
                        (Some(a), Some(b)) => {
                            !same_coefficients(&parent.operators[a], &parent.operators[b])
                        }
                        _ => !zero_bias(parent, source_bias) || !zero_bias(parent, other_bias),
                    }
                {
                    return Err(format!(
                        "incompatible parent coefficient inheritance for shared candidate owner {parameter}: parent matrix/bias owners disagree"
                    ));
                }
            }
            if !same_operator_interfaces(&builder.operators[matrix], &parent.operators[source]) {
                return Err("inherited matrix interface mismatch".into());
            }
            owner.interfaces_relabelled |= builder.operators[matrix].rows
                != parent.operators[source].rows
                || builder.operators[matrix].cols != parent.operators[source].cols;
            builder.operators[matrix] =
                inherited_operator(&builder.operators[matrix], &parent.operators[source])?;
            match (bias, source_bias) {
                (Some(target), Some(source)) => {
                    if !same_operator_interfaces(
                        &builder.operators[target],
                        &parent.operators[source],
                    ) {
                        return Err("inherited bias interface mismatch".into());
                    }
                    owner.interfaces_relabelled |= builder.operators[target].rows
                        != parent.operators[source].rows
                        || builder.operators[target].cols != parent.operators[source].cols;
                    builder.operators[target] =
                        inherited_operator(&builder.operators[target], &parent.operators[source])?;
                }
                (Some(target), None) => {
                    // A zero candidate bias is a function-preserving embedding
                    // of an exactly corresponding bias-free parent affine.
                    owner.zero_filled_bias_elements = builder.operators[target].rows.width();
                }
                (None, parent_bias) if zero_bias(parent, parent_bias) => (),
                _ => return Err("nonzero parent bias cannot be omitted".into()),
            }
            owner.parent_matrices = sources
                .iter()
                .map(|&(id, _)| id)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            owner.parent_biases = sources
                .iter()
                .filter_map(|&(_, id)| id)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            owner.inherited_elements = elements - owner.zero_filled_bias_elements;
            owner.random_elements = 0;
            owner.status = "inherited: exact parent subtree correspondence".into();
        }
        report.inherited_elements += owner.inherited_elements;
        report.zero_filled_bias_elements += owner.zero_filled_bias_elements;
        report.random_elements += owner.random_elements;
        report.owners.push(owner);
    }
    Ok(report)
}
fn inherited_operator(target: &Operator, source: &Operator) -> Result<Operator, String> {
    let mut result = source.clone();
    result.rows = target.rows.clone();
    result.cols = target.cols.clone();
    let OperatorBody::Dense { present, .. } = &mut result.body else {
        return Err("inheritance requires complete dense coefficients".into());
    };
    *present = Array2::from_elem((result.rows.group_count(), result.cols.group_count()), true);
    Ok(result)
}
fn same_operator_interfaces(a: &Operator, b: &Operator) -> bool {
    a.rows.width() == b.rows.width() && a.cols.width() == b.cols.width()
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
/// Inherit only proven parent subtrees. Parent boundary nodes and semantic exits
/// correspond to the supplied input/output interface order.
pub fn compile_program_inheriting(
    inputs: &[Interface],
    outputs: &[Interface],
    expressions: &[Expr],
    s: &Settings,
    parent: &OperatorProgram,
    parent_inputs: &[usize],
    parent_outputs: &[usize],
) -> Result<Compiled, String> {
    compile_program_initialized(
        inputs,
        outputs,
        expressions,
        s,
        Some((parent, parent_inputs, parent_outputs)),
    )
}
fn compile_program_initialized(
    inputs: &[Interface],
    outputs: &[Interface],
    expressions: &[Expr],
    s: &Settings,
    parent: Option<(&OperatorProgram, &[usize], &[usize])>,
) -> Result<Compiled, String> {
    if inputs
        .iter()
        .any(|ty| !matches!(Interface::native(ty.width()), Ok(ref held) if held == ty))
    {
        return Err("standalone learned program requires native Raw input grouping; use artifact apply for grouped boundaries".into());
    }
    let (mut builder, output_nodes) = build(inputs, outputs, expressions, s)?;
    let initialization =
        initialize_from_parent(&mut builder, inputs, outputs, expressions, s, parent)?;
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
        initialization,
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
    let (mut builder, exits) = build(&inputs, &outputs, expressions, s)?;
    let parent_inputs = args
        .iter()
        .map(|&n| a.place(n).ok_or("inheritance parent input absent"))
        .collect::<Result<Vec<_>, _>>()?;
    let mut initialization = initialize_from_parent(
        &mut builder,
        &inputs,
        &outputs,
        expressions,
        s,
        Some((&a.program, &parent_inputs, &r.current_writes)),
    )?;
    let local_trainables = builder
        .parameters
        .values()
        .flat_map(|(a, b)| std::iter::once(*a).chain(*b))
        .collect::<Vec<_>>();
    let (mut local_program, local_outputs) =
        standalone_local(&inputs, &builder.nodes, &builder.operators, &exits)?;
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
    for owner in &mut initialization.owners {
        owner.matrix_operator =
            mapped.operator_mapping[a.program.operators.len() + owner.matrix_operator];
        owner.bias_operator = owner
            .bias_operator
            .map(|id| mapped.operator_mapping[a.program.operators.len() + id]);
    }
    // Reuse the exact grafted owner Arcs after compaction; adapters exist only in the training wrapper.
    for local in 0..body.operators.len() {
        let grafted = mapped.operator_mapping[a.program.operators.len() + local];
        if grafted != usize::MAX {
            local_program.operators[local] =
                Arc::clone(&mapped.artifact.program.operators[grafted]);
        }
    }
    let local_fit = LocalFit {
        program: local_program,
        trainable_operator_ids: local_trainables.clone(),
        owner_mapping: local_trainables
            .iter()
            .copied()
            .zip(trainable_operator_ids.iter().copied())
            .collect(),
        input_native_places: args.to_vec(),
        output_native_places: r.native_writes.clone(),
        output_nodes: local_outputs,
    };
    Ok(Applied {
        local_fit,
        initialization,
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
    #[test]
    fn local_unused_arguments_remain_real_raw_inputs_through_fitter_inlining() {
        let scalar = Interface::native(1).expect("type");
        let (program, _) = standalone_local(
            &[scalar.clone(), scalar],
            &[
                Node::Param { index: 1 },
                Node::Pointwise {
                    input: 0,
                    laws: vec![Law::Relu],
                },
            ],
            &[],
            &[1],
        )
        .expect("unused first boundary");
        let (flat, _) = crate::artifact_device::mapped_inlined(&program).expect("fitter expansion");
        for slot in 0..2 {
            assert_eq!(
                flat.nodes
                    .iter()
                    .filter(|node| matches!(node,Node::Raw {slot:s} if *s==slot))
                    .count(),
                1
            );
        }
        let inputs = vec![array![[99.], [98.]], array![[1.], [2.]]];
        let measured = crate::resident_rule_fit::measure_grouped(
            &gam_gpu::tensor::Device::host(),
            &program,
            &inputs,
            &inputs[1],
            &[crate::resident_rule_fit::OutputGroup {
                label: "native exit".into(),
                start: 0,
                end: 1,
            }],
            1 << 24,
            2,
        )
        .expect("production local fitter accepts unused input");
        assert_eq!(measured.maximum, 0.);
    }

    #[test]
    fn local_training_wrapper_preserves_grouped_boundary_laws() {
        let grouped = Interface::uniform(2, 1, crate::operator_program::LabelKind::Native, 3)
            .expect("groups");
        let (program, outputs) = standalone_local(
            &[grouped.clone()],
            &[
                Node::Param { index: 0 },
                Node::Pointwise {
                    input: 0,
                    laws: vec![Law::Relu, Law::Identity],
                },
            ],
            &[],
            &[1],
        )
        .expect("grouped local wrapper");
        assert_eq!(program.interfaces().expect("types")[outputs[0]], grouped);
        let family = FamilyInputs {
            rows: 1,
            slots: vec![SlotValues::Raw(array![[-1., -2.]])],
            layout: None,
        };
        assert_eq!(
            program.execute(&family, false).expect("laws").values[program.output],
            array![[0., -2.]]
        );
        assert_eq!(
            program.operators.len(),
            1,
            "only fixed training regrouping adapter"
        );
    }

    #[test]
    fn generated_sharing_seeds_allow_unprojected_native_width_exit() {
        let inputs = [Interface::native(768).unwrap()];
        let outputs = [Interface::native(3072).unwrap(), Interface::native(768).unwrap()];
        let mut s = settings();
        s.latent_widths = vec![768, 3072];
        s.unary = vec![Unary::Silu, Unary::GeluTanh];
        s.binary = vec![Binary::Add, Binary::Multiply];
        s.affine_bias = false;
        s.require_shared = true;
        s.max_operations = 3;
        s.max_affine_parameters = 3;
        s.max_parameter_elements = 6_000_000;
        s.max_expression_states = 4096;
        s.max_tuple_checks = 8192;
        s.max_tuples = 128;
        s.max_body_nodes = 12;
        let inventory = enumerate_interfaces(&inputs, &outputs, &s).unwrap();
        assert!(inventory.proposals.iter().any(|p| {
            let Expr::Unary(Unary::GeluTanh, up) = &p.expressions[0] else { return false; };
            let Expr::Affine { output, input, .. } = up.as_ref() else { return false; };
            resolve(output, &inputs, &outputs).is_ok_and(|ty| ty.width() == 3072)
                && **input == Expr::Argument(0)
                && matches!(&p.expressions[1], Expr::Affine { input, .. } if **input == p.expressions[0])
                && p.parameter_elements == 2 * 768 * 3072
                && p.trainable_operator_count == 2
        }), "generated intermediate and its projection must share the full-width native equation without an identity adapter");
        assert!(inventory.explored_states <= s.max_expression_states);
        assert!(inventory.checked_tuples <= s.max_tuple_checks);
        assert!(inventory.sharing_seed_checks <= s.max_tuple_checks / 8);
        assert!(inventory.proposals.len() <= s.max_tuples);
    }

    #[test]
    fn generated_sharing_seeds_reach_native_width_class_without_supplied_equations() {
        let input = Interface::native(768).expect("native input");
        let scalar = Interface::native(1).expect("response");
        let mut s = settings();
        s.latent_widths = vec![3072];
        s.unary = vec![Unary::GeluTanh];
        s.binary.clear();
        s.require_shared = true;
        s.max_operations = 3;
        s.max_expression_states = 4096;
        s.max_tuple_checks = 512;
        s.max_tuples = 32;
        s.max_parameter_elements = 6_000_000;
        let inventory =
            enumerate_interfaces(&[input.clone()], &[input, scalar.clone(), scalar], &s)
                .expect("typed enumeration only");
        assert!(
            inventory.proposals.iter().any(|p| {
                let Expr::Affine {
                    input: activation, ..
                } = &p.expressions[0]
                else {
                    return false;
                };
                let Expr::Unary(Unary::GeluTanh, up) = activation.as_ref() else {
                    return false;
                };
                let Expr::Affine {
                    output: TypeRef::Latent { width: 3072 },
                    input,
                    ..
                } = up.as_ref()
                else {
                    return false;
                };
                **input == Expr::Argument(0)
                    && p.expressions[1..]
                        .iter()
                        .all(|e| matches!(e, Expr::Affine { input, .. } if input == activation))
            }),
            "generated shared nonlinear latent class must reach finite bank"
        );
        assert!(inventory.sharing_seed_checks > 0);
        assert!(inventory.sharing_seed_checks <= s.max_tuple_checks / 8);
        assert!(inventory.checked_tuples <= s.max_tuple_checks);
    }

    #[test]
    fn broad_128_bank_includes_shared_native_law_latent_class() {
        let input = Interface::native(768).expect("native input");
        let scalar = Interface::native(1).expect("response");
        let mut s = settings();
        s.latent_widths = vec![3072];
        s.unary = vec![Unary::Relu, Unary::Silu, Unary::Gelu, Unary::GeluTanh];
        s.binary = vec![Binary::Add, Binary::Subtract, Binary::Multiply];
        s.affine_bias = true;
        s.require_shared = true;
        s.max_operations = 3;
        s.max_expression_states = 4096;
        s.max_tuple_checks = 2000;
        s.max_tuples = 128;
        s.max_parameter_elements = 6_000_000;
        let inventory =
            enumerate_interfaces(&[input.clone()], &[input, scalar.clone(), scalar], &s)
                .expect("broad generated bank");
        assert!(inventory.proposals.iter().any(|p| {
            let Expr::Affine { input: activation, bias: true, .. } = &p.expressions[0] else { return false; };
            let Expr::Unary(Unary::GeluTanh, up) = activation.as_ref() else { return false; };
            let Expr::Affine { output: TypeRef::Latent { width: 3072 }, input, bias: true, .. } = up.as_ref() else { return false; };
            **input == Expr::Argument(0) && p.expressions[1..].iter().all(|e| matches!(e, Expr::Affine { input, bias: true, .. } if input == activation))
        }), "bounded broad law/operator search must cover native-law shared latent class");
        assert_eq!(inventory.proposals.len(), s.max_tuples);
        assert!(inventory.checked_tuples <= s.max_tuple_checks);
        assert!(inventory.sharing_seed_checks <= s.max_tuple_checks / 8);
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
