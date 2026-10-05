//! Parent-derived typed DAG edits. The exact copy is a preservation control;
//! hypotheses change an existing operand edge and inherit every surviving owner.
//! No new basis, random coefficients, architecture template, or fitter is used.
//!
//! This is a child module of `program_learned_dag`, sharing its standalone local
//! compiler and LocalFit contract. Native operator identities are kept separately
//! from expression identity: a matrix and its bias need not share an owner.
use super::{Applied, InitializationReport, LocalFit, standalone_local};
use crate::{
    artifact::Artifact,
    operator_program::{Coefficient, Interface, Node, Operator, OperatorBody, remap_node},
    program_joint_regions::{self as joint, Exit, JointBody, Region},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    /// Every inspected (consumer, operand, earlier producer), even if rejected.
    pub max_edit_checks: usize,
    /// Nonidentity hypotheses only; the preservation control is separate.
    pub max_proposals: usize,
    pub max_body_nodes: usize,
    /// Surviving operator reals, counted once per exact parent owner.
    pub max_parameter_elements: usize,
}

/// Current-parent node identities, not new native place names. Operand positions
/// follow Node::arguments(); term positions in a multi-term affine remain ordered.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Reuse {
    pub consumer: usize,
    pub operand: usize,
    pub previous: usize,
    pub replacement: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnerSource {
    pub parent_operator: usize,
    /// Only complete finite dense owners exclusive to this region can be fitted.
    /// An edited hypothesis additionally requires every surviving use to be an
    /// affected affine input/bias. Upstream, donor, and clean sibling uses freeze
    /// their whole shared owner. Controls retain all otherwise eligible owners.
    pub trainable: bool,
    pub coefficient_elements: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proposal {
    /// Deterministic stale-source guard over the extracted graph and all source
    /// operator bits. This is not a cryptographic identity or equivalence proof.
    pub source_fingerprint: [u64; 2],
    /// None means exact native-copy control, never a discovery proposal.
    pub edit: Option<Reuse>,
    pub owners: Vec<OwnerSource>,
    pub compiled_node_count: usize,
    pub parameter_elements: usize,
    /// Indices into the complete region input boundary. Unused arguments remain
    /// supplied in the ABI but are not computational dependencies of the exits.
    pub used_arguments: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Inventory {
    pub control: Proposal,
    pub proposals: Vec<Proposal>,
    pub checked_edits: usize,
    pub duplicate_edits: usize,
    pub rejection_counts: BTreeMap<String, usize>,
    pub truncated: bool,
    pub scope: String,
}

struct Extracted {
    source_fingerprint: [u64; 2],
    inputs: Vec<Interface>,
    nodes: Vec<Node>,
    /// Extracted local node index -> current parent node index.
    parent_nodes: Vec<usize>,
    types: Vec<Interface>,
    exits: Vec<usize>,
    owners: BTreeMap<usize, OwnerSource>,
}

fn fingerprint(a: &Artifact, r: &Region, body: &JointBody) -> [u64; 2] {
    fn word(state: &mut u64, value: u64) {
        *state = (*state ^ value)
            .rotate_left(27)
            .wrapping_mul(0x9e3779b185ebca87);
    }
    fn bytes(state: &mut u64, text: &str) {
        word(state, text.len() as u64);
        for value in text.bytes() {
            word(state, u64::from(value));
        }
    }
    let mut structure = 0x6a09e667f3bcc909;
    let mut coefficients = 0xbb67ae8584caa73b;
    bytes(
        &mut structure,
        &format!("{r:?}/{:?}/{:?}/{:?}", body.inputs, body.nodes, body.exits),
    );
    for id in body
        .nodes
        .iter()
        .flat_map(Node::operators)
        .collect::<BTreeSet<_>>()
    {
        let op = &a.program.operators[id];
        word(&mut structure, id as u64);
        bytes(
            &mut structure,
            &format!(
                "{:?}/{:?}/{:?}/{:?}",
                op.name, op.rows, op.cols, op.provenance
            ),
        );
        word(&mut coefficients, id as u64);
        match &op.body {
            OperatorBody::Identity => word(&mut structure, 0),
            OperatorBody::Dense {
                values,
                present,
                precision,
            } => {
                word(&mut structure, 1);
                bytes(
                    &mut structure,
                    &format!("{:?}/{:?}/{precision:?}", values.dim(), present.dim()),
                );
                for &keep in present {
                    word(&mut structure, u64::from(keep));
                }
                for &value in values {
                    word(&mut coefficients, value.to_bits());
                }
            }
            OperatorBody::LowRank {
                left,
                right,
                precision,
            } => {
                word(&mut structure, 2);
                bytes(
                    &mut structure,
                    &format!("{:?}/{:?}/{precision:?}", left.dim(), right.dim()),
                );
                for &value in left.iter().chain(right) {
                    word(&mut coefficients, value.to_bits());
                }
            }
            OperatorBody::Diagonal { values, precision } => {
                word(&mut structure, 3);
                bytes(&mut structure, &format!("{}/{precision:?}", values.len()));
                for &value in values {
                    word(&mut coefficients, value.to_bits());
                }
            }
        }
    }
    [structure, coefficients]
}

fn supported(node: &Node) -> bool {
    matches!(
        node,
        Node::Param { .. }
            | Node::Affine { .. }
            | Node::Pointwise { .. }
            | Node::Hadamard { .. }
            | Node::Concat { .. }
            | Node::Gain {
                coefficient: Coefficient::Number(_),
                ..
            }
    )
}

fn extract(a: &Artifact, r: &Region) -> Result<Extracted, String> {
    let body = joint::exact_body(a, r)?;
    if body.nodes.iter().any(|n| !supported(n)) {
        return Err("parent edit region contains an unsupported native node".into());
    }
    let parent_nodes = r
        .current_reads
        .iter()
        .chain(&r.current_internal_nodes)
        .copied()
        .collect::<Vec<_>>();
    if parent_nodes.len() != body.nodes.len() {
        return Err("parent extraction node correspondence differs".into());
    }
    let native_types = a.program.interfaces().map_err(|e| e.to_string())?;
    let inside = r
        .current_internal_nodes
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut external_owners = BTreeSet::new();
    for (index, node) in a.program.nodes.iter().enumerate() {
        if !inside.contains(&index) {
            external_owners.extend(a.program.node_operators(node));
        }
    }
    // A retained library rule can be called again by a later candidate. Preserve
    // its coefficient identities even if it currently has no live invocation.
    for rule in &a.program.rules {
        for node in &rule.nodes {
            external_owners.extend(a.program.node_operators(node));
        }
    }
    // Artifact-level derivations retain both their targets and coefficient
    // sources independently of ordinary graph uses.
    external_owners.extend(a.derived_operators());
    let owners = body
        .nodes
        .iter()
        .flat_map(Node::operators)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|id| {
            let op = &a.program.operators[id];
            let finite_dense = matches!(&op.body, OperatorBody::Dense { values, present, .. }
                if present.iter().all(|keep| *keep) && values.iter().all(|v| v.is_finite()));
            (
                id,
                OwnerSource {
                    parent_operator: id,
                    trainable: finite_dense && !external_owners.contains(&id),
                    coefficient_elements: op.real_count(),
                },
            )
        })
        .collect();
    let source_fingerprint = fingerprint(a, r, &body);
    Ok(Extracted {
        source_fingerprint,
        inputs: body.inputs,
        nodes: body.nodes,
        types: parent_nodes
            .iter()
            .map(|&n| native_types[n].clone())
            .collect(),
        parent_nodes,
        exits: body.exits.iter().map(|e| e.node).collect(),
        owners,
    })
}

fn set_operand(node: &mut Node, operand: usize, replacement: usize) -> Result<(), String> {
    let target = match node {
        Node::Affine { terms, .. } => terms.get_mut(operand).map(|(n, _)| n),
        Node::Pointwise { input, .. } | Node::Gain { input, .. } if operand == 0 => Some(input),
        Node::Hadamard { left, .. } if operand == 0 => Some(left),
        Node::Hadamard { right, .. } if operand == 1 => Some(right),
        Node::Concat { parts } => parts.get_mut(operand),
        _ => None,
    }
    .ok_or("parent edit operand is not supported")?;
    *target = replacement;
    Ok(())
}

#[derive(Debug)]
struct Materialized {
    nodes: Vec<Node>,
    exits: Vec<usize>,
    proposal: Proposal,
}

/// Owners eligible for compensation after a single edge change. Values become
/// dirty at the edited consumer and through its descendants, never through the
/// unchanged donor's ancestors. Eligibility is per affine term, not per node:
/// clean sibling terms remain native even in an affine with a dirty output.
fn affected_owners(
    nodes: &[Node],
    live: &BTreeSet<usize>,
    consumer: usize,
    operand: usize,
) -> BTreeSet<usize> {
    let mut dirty = BTreeSet::new();
    let mut affected = BTreeSet::new();
    let mut clean = BTreeSet::new();
    for &id in live {
        let changed = id == consumer || nodes[id].arguments().iter().any(|n| dirty.contains(n));
        if changed {
            dirty.insert(id);
        }
        if let Node::Affine { terms, bias } = &nodes[id] {
            for (position, &(input, owner)) in terms.iter().enumerate() {
                // The substituted input value is dirty on this edge even though
                // the donor node itself is an unchanged native computation.
                if dirty.contains(&input) || (id == consumer && position == operand) {
                    affected.insert(owner);
                } else {
                    clean.insert(owner);
                }
            }
            if let Some(owner) = bias {
                if changed {
                    affected.insert(*owner);
                } else {
                    clean.insert(*owner);
                }
            }
        }
    }
    // One clean use freezes the entire owner; do not break a native tie to make
    // its dirty use independently trainable.
    affected.retain(|id| !clean.contains(id));
    affected
}

fn materialize(
    a: &Artifact,
    parent: &Extracted,
    edit: Option<&Reuse>,
) -> Result<Materialized, String> {
    let mut nodes = parent.nodes.clone();
    if let Some(edit) = edit {
        let find = |id| {
            parent
                .parent_nodes
                .iter()
                .position(|&n| n == id)
                .ok_or("parent edit references a node outside the region boundary/body")
        };
        let consumer = find(edit.consumer)?;
        let replacement = find(edit.replacement)?;
        let previous = find(edit.previous)?;
        if replacement >= consumer {
            return Err("parent edit creates a forward reference".into());
        }
        if nodes[consumer].arguments().get(edit.operand) != Some(&previous) {
            return Err("parent edit previous operand is stale".into());
        }
        if previous == replacement {
            return Err("parent edit does not change an operand".into());
        }
        if parent.types[previous] != parent.types[replacement] {
            return Err("parent edit operand interfaces differ".into());
        }
        set_operand(&mut nodes[consumer], edit.operand, replacement)?;
    }
    let mut live = BTreeSet::new();
    let mut pending = parent.exits.clone();
    while let Some(n) = pending.pop() {
        if live.insert(n) {
            pending.extend(nodes[n].arguments());
        }
    }
    if let Some(edit) = edit {
        let consumer = parent
            .parent_nodes
            .iter()
            .position(|&n| n == edit.consumer)
            .unwrap();
        if !live.contains(&consumer) {
            return Err("parent edit consumer is not an exit ancestor".into());
        }
    }
    let used_arguments = live
        .iter()
        .filter_map(|&n| match nodes[n] {
            Node::Param { index } => Some(index),
            _ => None,
        })
        .collect::<Vec<_>>();
    // Keep the complete input ABI independently of computational support.
    live.extend(0..parent.inputs.len());
    let affected = edit.map(|edit| {
        let consumer = parent
            .parent_nodes
            .iter()
            .position(|&n| n == edit.consumer)
            .expect("validated parent edit consumer");
        affected_owners(&nodes, &live, consumer, edit.operand)
    });
    let owner_ids = live
        .iter()
        .flat_map(|&n| nodes[n].operators())
        .collect::<BTreeSet<_>>();
    let mut owners = Vec::new();
    let mut operator_map = vec![usize::MAX; a.program.operators.len()];
    let mut elements = 0usize;
    for id in owner_ids {
        let source = &parent.owners[&id];
        let count = source.coefficient_elements;
        elements = elements
            .checked_add(count)
            .ok_or("parent coefficient count overflow")?;
        let mut owner = source.clone();
        owner.trainable &= affected.as_ref().is_none_or(|ids| ids.contains(&id));
        owners.push(owner);
        operator_map[id] = owners.len() - 1;
    }
    let mut node_map = vec![usize::MAX; nodes.len()];
    let mut compact = Vec::new();
    for &id in &live {
        node_map[id] = compact.len();
        let mut node = nodes[id].clone();
        remap_node(&mut node, &node_map, &operator_map, &[], &[]);
        compact.push(node);
    }
    let exits = parent
        .exits
        .iter()
        .map(|&n| node_map[n])
        .collect::<Vec<_>>();
    let proposal = Proposal {
        source_fingerprint: parent.source_fingerprint,
        edit: edit.cloned(),
        owners,
        compiled_node_count: compact.len(),
        parameter_elements: elements,
        used_arguments,
    };
    Ok(Materialized {
        nodes: compact,
        exits,
        proposal,
    })
}

fn limits(p: &Proposal, s: &Settings) -> Result<(), String> {
    if p.compiled_node_count > s.max_body_nodes {
        return Err("parent max_body_nodes exceeded".into());
    }
    if p.parameter_elements > s.max_parameter_elements {
        return Err("parent max_parameter_elements exceeded".into());
    }
    Ok(())
}

fn signature(m: &Materialized) -> String {
    // Coefficients are identified by their source owners, never by value/shape.
    // Debug formatting is used only for bounded in-memory duplicate detection.
    format!(
        "{:?}/{:?}/{:?}",
        m.nodes,
        m.exits,
        m.proposal
            .owners
            .iter()
            .map(|o| o.parent_operator)
            .collect::<Vec<_>>()
    )
}

/// Enumerate one-edge changes from the actual native/current region. Typed
/// producer reuse has no grammar expansion or parameter-partition search.
pub fn enumerate(a: &Artifact, r: &Region, s: &Settings) -> Result<Inventory, String> {
    if [
        s.max_edit_checks,
        s.max_proposals,
        s.max_body_nodes,
        s.max_parameter_elements,
    ]
    .contains(&0)
    {
        return Err("positive parent edit budgets required".into());
    }
    let parent = extract(a, r)?;
    let control = materialize(a, &parent, None)?;
    // The parent is the baseline, not an admitted edit. Requiring its cost to
    // meet the target cap would block exactly the simplifications being sought.
    let mut seen = BTreeSet::from([signature(&control)]);
    let mut out = Inventory {
        control: control.proposal,
        proposals: Vec::new(), checked_edits: 0, duplicate_edits: 0,
        rejection_counts: BTreeMap::new(), truncated: false,
        scope: "Exact parent-copy control separate from one-edge typed producer-reuse hypotheses. Surviving independent operator identities and coefficients are inherited, including matrices above edited descendants. Hypotheses fit only dirty-input affine terms and affected affine biases; upstream/donor computations, clean sibling terms, any owner with a clean surviving use, and external shared owners stay frozen. No new native variables, neuron-subset discovery, numerical equivalence, or semantic interpretation is asserted.".into(),
    };
    // Rotate consumer/operand sites while scanning earlier producers so one
    // large fan-in node cannot exhaust every edit check before other sites.
    let sites = parent
        .nodes
        .iter()
        .enumerate()
        .flat_map(|(consumer, n)| {
            n.arguments()
                .into_iter()
                .enumerate()
                .map(move |(operand, previous)| (consumer, operand, previous))
        })
        .collect::<Vec<_>>();
    for replacement in 0..parent.nodes.len() {
        for &(consumer, operand, previous) in &sites {
            if replacement >= consumer {
                continue;
            }
            if out.checked_edits == s.max_edit_checks || out.proposals.len() == s.max_proposals {
                out.truncated = true;
                return Ok(out);
            }
            out.checked_edits += 1;
            let edit = Reuse {
                consumer: parent.parent_nodes[consumer],
                operand,
                previous: parent.parent_nodes[previous],
                replacement: parent.parent_nodes[replacement],
            };
            let result = materialize(a, &parent, Some(&edit)).and_then(|m| {
                limits(&m.proposal, s)?;
                Ok(m)
            });
            match result {
                Ok(m) => {
                    if seen.insert(signature(&m)) {
                        out.proposals.push(m.proposal);
                    } else {
                        out.duplicate_edits += 1;
                    }
                }
                Err(reason) => *out.rejection_counts.entry(reason).or_default() += 1,
            }
        }
    }
    Ok(out)
}

/// Compile with existing local-wrapper and atomic joint-graft machinery. Every
/// surviving coefficient is copied from its explicit parent owner; changed child
/// expressions do not invalidate inheritance of an unchanged consumer matrix.
pub fn apply(a: &Artifact, r: &Region, p: &Proposal, s: &Settings) -> Result<Applied, String> {
    let parent = extract(a, r)?;
    let built = materialize(a, &parent, p.edit.as_ref())?;
    if built.proposal != *p {
        return Err("parent proposal provenance or structure is stale".into());
    }
    if p.edit.is_some() {
        limits(p, s)?;
    }
    let local_trainables = p
        .owners
        .iter()
        .enumerate()
        .filter_map(|(id, source)| source.trainable.then_some(id))
        .collect::<Vec<_>>();
    // Enumeration never clones large matrices; materialize coefficients only for
    // the chosen candidate entering the existing compiler/fitter path.
    let operators = p
        .owners
        .iter()
        .map(|source| a.program.operators[source.parent_operator].as_ref().clone())
        .collect::<Vec<Operator>>();
    let (mut local_program, local_outputs) =
        standalone_local(&parent.inputs, &built.nodes, &operators, &built.exits)?;
    let mut appended = Vec::new();
    let graft_operators = p
        .owners
        .iter()
        .enumerate()
        .map(|(id, source)| {
            if source.trainable {
                let destination = a.program.operators.len() + appended.len();
                appended.push(operators[id].clone());
                destination
            } else {
                source.parent_operator
            }
        })
        .collect::<Vec<_>>();
    let node_ids = (0..built.nodes.len()).collect::<Vec<_>>();
    let mut nodes = built.nodes;
    for node in &mut nodes {
        remap_node(node, &node_ids, &graft_operators, &[], &[]);
    }
    let body = JointBody {
        inputs: parent.inputs,
        nodes,
        operators: appended,
        exits: r
            .native_writes
            .iter()
            .zip(&built.exits)
            .map(|(&native_write, &node)| Exit { native_write, node })
            .collect(),
    };
    let mapped = joint::apply_mapped(a, r, &body)?;
    let mut destinations = Vec::new();
    for (local, &id) in graft_operators.iter().enumerate() {
        let destination = mapped.operator_mapping[id];
        if destination == usize::MAX {
            return Err("live parent owner erased during joint compaction".into());
        }
        local_program.operators[local] =
            Arc::clone(&mapped.artifact.program.operators[destination]);
        destinations.push(destination);
    }
    let trainable_operator_ids = local_trainables
        .iter()
        .map(|&id| destinations[id])
        .collect::<Vec<_>>();
    let initialization = InitializationReport {
        // Parent provenance is per independent operator in Proposal::owners.
        // The legacy owners field describes bundled Expr affine parameters and
        // cannot faithfully encode independently shared matrix/bias ownership.
        inherited_elements: p.parameter_elements,
        scope: "Explicit parent operator inheritance; per-operator provenance is Proposal::owners, not the legacy bundled-affine owners list. Zero random elements. Unchanged consumer coefficients survive edited descendants. This is initialization, not a preservation claim for nonidentity edits.".into(),
        ..InitializationReport::default()
    };
    let local_fit = LocalFit {
        program: local_program,
        trainable_operator_ids: local_trainables.clone(),
        owner_mapping: local_trainables
            .iter()
            .copied()
            .zip(trainable_operator_ids.iter().copied())
            .collect(),
        input_native_places: r.native_reads.clone(),
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

/// Search-facing entry point; a forged/control mutation never enters the
/// hypothesis fitting callback even if constructed without enumeration.
pub fn apply_hypothesis(
    a: &Artifact,
    r: &Region,
    p: &Proposal,
    s: &Settings,
) -> Result<Applied, String> {
    if p.edit.is_none() {
        return Err("native preservation control is not a discovery hypothesis".into());
    }
    apply(a, r, p, s)
}

#[cfg(test)]
#[path = "program_learned_dag_parent_tests.rs"]
mod tests;
