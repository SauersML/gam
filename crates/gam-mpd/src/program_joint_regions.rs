//! Atomic typed multi-exit rewrites of a shared current-artifact DAG.
//! Enumeration/extraction is structural support, not identification of a learned law.
use crate::{
    artifact::{Artifact, Binding, compact_with_roots},
    operator_program::{Coefficient, Interface, Node, Operator, remap_node},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::Arc,
};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Limits {
    pub max_internal_nodes: usize,
    pub max_inputs: usize,
    pub max_exits: usize,
    pub max_regions: usize,
    pub max_states: usize,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AnchorMode {
    #[default]
    Interior,
    Boundary,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Region {
    #[serde(default)]
    pub anchor_mode: AnchorMode,
    pub native_reads: Vec<usize>,
    pub current_reads: Vec<usize>,
    pub native_writes: Vec<usize>,
    pub current_writes: Vec<usize>,
    pub current_internal_nodes: Vec<usize>,
    pub producer_native: usize,
    pub erased_native_places: Vec<usize>,
    pub source_program_nodes: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Skipped {
    #[serde(default)]
    pub anchor_mode: AnchorMode,
    pub producer_native: usize,
    pub current_internal_nodes: Vec<usize>,
    pub reason: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Inventory {
    pub regions: Vec<Region>,
    pub skipped: Vec<Skipped>,
    pub truncated: bool,
    /// Total description and successor attempts, including duplicate successors.
    pub explored_states: usize,
    #[serde(default)]
    pub attempted_expansions: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Exit {
    pub native_write: usize,
    pub node: usize,
}
#[derive(Clone, Debug)]
pub struct JointBody {
    pub inputs: Vec<Interface>,
    pub nodes: Vec<Node>,
    pub exits: Vec<Exit>,
    pub operators: Vec<Operator>,
}
fn ambient(n: &Node) -> bool {
    matches!(
        n,
        Node::Raw { .. } | Node::Feature { .. } | Node::Param { .. }
    )
}
fn aliases(a: &Artifact) -> Result<BTreeMap<usize, usize>, String> {
    let mut out = BTreeMap::new();
    for &(native, current) in &a.places {
        if native >= a.native_nodes
            || current >= a.program.nodes.len()
            || out.insert(current, native).is_some()
        {
            return Err("invalid/ambiguous native place for joint boundary".into());
        }
    }
    Ok(out)
}
fn describe_anchored(
    a: &Artifact,
    producer: usize,
    anchor_mode: AnchorMode,
    mut inside: BTreeSet<usize>,
    limits: &Limits,
) -> Result<Region, String> {
    let names = aliases(a)?;
    let root = a.place(producer).ok_or("joint producer place absent")?;
    if anchor_mode == AnchorMode::Interior && !inside.contains(&root) {
        return Err("joint producer absent from region".into());
    }
    if anchor_mode == AnchorMode::Boundary && inside.contains(&root) {
        return Err("joint boundary producer must remain outside region".into());
    }
    let boundary = loop {
        if inside.len() > limits.max_internal_nodes {
            return Err("max_internal_nodes exceeded".into());
        }
        let mut missing = BTreeSet::new();
        let mut boundary = BTreeSet::new();
        for &i in &inside {
            let node = a
                .program
                .nodes
                .get(i)
                .ok_or("joint internal node missing")?;
            if ambient(node) {
                return Err("ambient Raw/Feature/Param inside joint body".into());
            }
            for p in node.arguments() {
                if !inside.contains(&p) {
                    if names.contains_key(&p) {
                        boundary.insert(p);
                    } else {
                        missing.insert(p);
                    }
                }
            }
        }
        if missing.is_empty() {
            break boundary;
        }
        inside.extend(missing);
    };
    let mut adjacency = BTreeMap::<usize, BTreeSet<usize>>::new();
    for &i in &inside {
        for p in a.program.nodes[i].arguments() {
            if inside.contains(&p) || (anchor_mode == AnchorMode::Boundary && p == root) {
                adjacency.entry(i).or_default().insert(p);
                adjacency.entry(p).or_default().insert(i);
            }
        }
    }
    if anchor_mode == AnchorMode::Boundary
        && (!boundary.contains(&root)
            || inside
                .iter()
                .filter(|&&i| a.program.nodes[i].arguments().contains(&root))
                .count()
                < 2)
    {
        return Err("joint boundary producer requires two direct internal consumers".into());
    }
    let mut connected = BTreeSet::from([root]);
    let mut pending = vec![root];
    while let Some(i) = pending.pop() {
        for &neighbor in adjacency.get(&i).into_iter().flatten() {
            if connected.insert(neighbor) {
                pending.push(neighbor);
            }
        }
    }
    if anchor_mode == AnchorMode::Boundary {
        connected.remove(&root);
    }
    if connected != inside {
        return Err("joint region is not connected to its producer".into());
    }
    if boundary.len() > limits.max_inputs {
        return Err("max_inputs exceeded".into());
    }
    if boundary
        .iter()
        .any(|&i| matches!(a.program.nodes[i], Node::Feature { .. }))
    {
        return Err("gathered Feature boundary is not an ordinary matrix input".into());
    }
    let mut exits = BTreeSet::new();
    if inside.contains(&a.program.output) {
        exits.insert(a.program.output);
    }
    for (i, node) in a.program.nodes.iter().enumerate() {
        if !inside.contains(&i) {
            for p in node.arguments() {
                if inside.contains(&p) {
                    exits.insert(p);
                }
            }
        }
    }
    for c in &a.controls {
        if inside.contains(&c.write) {
            exits.insert(c.write);
        }
    }
    for e in &a.exceptions {
        if inside.contains(&e.node) {
            exits.insert(e.node);
        }
    }
    // A block whose write survives must retain each read's original typed state.
    loop {
        let before = exits.len();
        for b in &a.blocks {
            if !inside.contains(&b.write) || exits.contains(&b.write) {
                for &r in &b.reads {
                    if inside.contains(&r) {
                        exits.insert(r);
                    }
                }
            }
        }
        if before == exits.len() {
            break;
        }
    }
    if exits.is_empty() || exits.len() > limits.max_exits {
        return Err("empty exits or max_exits exceeded".into());
    }
    let mut writes = exits
        .iter()
        .map(|i| {
            names
                .get(i)
                .copied()
                .map(|n| (n, *i))
                .ok_or("external joint exit lacks native place".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    writes.sort();
    let mut reads = boundary.iter().map(|i| (names[i], *i)).collect::<Vec<_>>();
    reads.sort();
    if reads.iter().any(|(native, _)| *native >= writes[0].0) {
        return Err(
            "joint boundary input is not before every native exit (per-exit subsets unsupported)"
                .into(),
        );
    }
    Ok(Region {
        anchor_mode,
        native_reads: reads.iter().map(|x| x.0).collect(),
        current_reads: reads.iter().map(|x| x.1).collect(),
        native_writes: writes.iter().map(|x| x.0).collect(),
        current_writes: writes.iter().map(|x| x.1).collect(),
        erased_native_places: inside
            .iter()
            .filter(|i| !exits.contains(i))
            .filter_map(|i| names.get(i).copied())
            .collect(),
        current_internal_nodes: inside.into_iter().collect(),
        producer_native: producer,
        source_program_nodes: a.program.nodes.len(),
    })
}
pub fn propose_regions(a: &Artifact, limits: Limits) -> Result<Inventory, String> {
    if [
        limits.max_internal_nodes,
        limits.max_exits,
        limits.max_regions,
        limits.max_states,
    ]
    .contains(&0)
    {
        return Err("positive joint node/exit/region/state budgets required".into());
    }
    a.program.interfaces().map_err(|e| e.to_string())?;
    let names = aliases(a)?;
    let mut users = vec![BTreeSet::new(); a.program.nodes.len()];
    for (i, n) in a.program.nodes.iter().enumerate() {
        for p in n.arguments() {
            users[p].insert(i);
        }
    }
    enum Work {
        Describe(BTreeSet<usize>),
        Expand {
            inside: BTreeSet<usize>,
            next: BTreeSet<usize>,
        },
    }
    let mut anchors = VecDeque::new();
    let mut seen = BTreeSet::new();
    let mut seed_truncated = false;
    // Each native anchor/mode has its own lazy frontier and receives one work
    // attempt per turn. A large fan-out cannot consume another anchor's budget
    // in a single turn. Successors are generated one at a time, never as pairs.
    for (&current, &native) in &names {
        if users[current].len() < 2 || matches!(a.program.nodes[current], Node::Feature { .. }) {
            continue;
        }
        for mode in [AnchorMode::Interior, AnchorMode::Boundary] {
            if mode == AnchorMode::Interior && ambient(&a.program.nodes[current]) {
                continue;
            }
            if seen.len() >= limits.max_states {
                seed_truncated = true;
                break;
            }
            let set = if mode == AnchorMode::Interior {
                BTreeSet::from([current])
            } else {
                BTreeSet::new()
            };
            seen.insert((native, mode, set.clone()));
            anchors.push_back((native, current, mode, VecDeque::from([Work::Describe(set)])));
        }
    }
    let mut out = Inventory {
        regions: vec![],
        skipped: vec![],
        truncated: seed_truncated,
        explored_states: 0,
        attempted_expansions: 0,
    };
    while let Some((producer, current, mode, mut frontier)) = anchors.pop_front() {
        if out.explored_states >= limits.max_states || out.regions.len() >= limits.max_regions {
            out.truncated = true;
            break;
        }
        let work = frontier.pop_front().expect("active anchor has work");
        out.explored_states += 1;
        match work {
            Work::Describe(inside) => {
                match describe_anchored(a, producer, mode, inside.clone(), &limits) {
                    Ok(r) => {
                        if users[current]
                            .iter()
                            .filter(|i| r.current_internal_nodes.contains(i))
                            .count()
                            >= 2
                        {
                            out.regions.push(r);
                        }
                    }
                    Err(reason) => out.skipped.push(Skipped {
                        anchor_mode: mode,
                        producer_native: producer,
                        current_internal_nodes: inside.iter().copied().collect(),
                        reason,
                    }),
                }
                let next = inside
                    .iter()
                    .flat_map(|i| users[*i].iter().copied())
                    .chain(
                        (mode == AnchorMode::Boundary)
                            .then_some(&users[current])
                            .into_iter()
                            .flatten()
                            .copied(),
                    )
                    .filter(|i| !inside.contains(i))
                    .collect::<BTreeSet<_>>();
                if inside.len() < limits.max_internal_nodes && !next.is_empty() {
                    frontier.push_back(Work::Expand { inside, next });
                } else if !next.is_empty() {
                    out.truncated = true;
                }
            }
            Work::Expand { inside, mut next } => {
                out.attempted_expansions += 1;
                let i = next.pop_first().expect("successor cursor is nonempty");
                let mut expanded = inside.clone();
                expanded.insert(i);
                // Duplicate attempts still consume work: graph symmetry cannot
                // create unaccounted enumeration loops.
                if seen.insert((producer, mode, expanded.clone())) {
                    frontier.push_back(Work::Describe(expanded));
                }
                if !next.is_empty() {
                    frontier.push_back(Work::Expand { inside, next });
                }
            }
        }
        if !frontier.is_empty() {
            anchors.push_back((producer, current, mode, frontier));
        }
    }
    Ok(out)
}
fn validate_region(a: &Artifact, r: &Region) -> Result<Vec<Interface>, String> {
    if r.source_program_nodes != a.program.nodes.len() || r.current_internal_nodes.is_empty() {
        return Err("stale or empty joint region".into());
    }
    let generous = Limits {
        max_internal_nodes: usize::MAX,
        max_inputs: usize::MAX,
        max_exits: usize::MAX,
        max_regions: 1,
        max_states: 1,
    };
    let expected = describe_anchored(
        a,
        r.producer_native,
        r.anchor_mode,
        r.current_internal_nodes.iter().copied().collect(),
        &generous,
    )?;
    if &expected != r {
        return Err("joint region boundaries/protected exits differ from current artifact".into());
    }
    let interfaces = a.program.interfaces().map_err(|e| e.to_string())?;
    Ok(r.current_reads
        .iter()
        .map(|&i| interfaces[i].clone())
        .collect())
}
pub fn exact_body(a: &Artifact, r: &Region) -> Result<JointBody, String> {
    let inputs = validate_region(a, r)?;
    let mut mapping = vec![usize::MAX; a.program.nodes.len()];
    let mut nodes = vec![];
    for (index, &read) in r.current_reads.iter().enumerate() {
        mapping[read] = nodes.len();
        nodes.push(Node::Param { index });
    }
    let ops = (0..a.program.operators.len()).collect::<Vec<_>>();
    let bases = (0..a.program.bases.len()).collect::<Vec<_>>();
    let rules = (0..a.program.rules.len()).collect::<Vec<_>>();
    for &i in &r.current_internal_nodes {
        if a.program.nodes[i]
            .arguments()
            .iter()
            .any(|&p| mapping[p] == usize::MAX)
        {
            return Err("joint exact body is not ancestor closed".into());
        }
        let mut node = a.program.nodes[i].clone();
        remap_node(&mut node, &mapping, &ops, &bases, &rules);
        mapping[i] = nodes.len();
        nodes.push(node);
    }
    Ok(JointBody {
        inputs,
        nodes,
        exits: r
            .native_writes
            .iter()
            .zip(&r.current_writes)
            .map(|(&native, &current)| Exit {
                native_write: native,
                node: mapping[current],
            })
            .collect(),
        operators: vec![],
    })
}
/// Replace all declared exits together with ONE materialized shared DAG.
/// No native name is assigned to a latent node. Observable exits get distinct identity views.
/// Pools/metadata are compacted by Artifact's existing compactor, not a second codec.
pub fn apply(a: &Artifact, r: &Region, body: &JointBody) -> Result<Artifact, String> {
    Ok(apply_mapped(a, r, body)?.artifact)
}

/// Exact identities after topological ordering and compaction. Operator indices
/// address the parent pool followed by `body.operators`; node indices address
/// `body.nodes`. Removed entries are `usize::MAX`. These maps identify actual
/// candidate values/parameters, without assigning latent values native names.
#[derive(Clone, Debug)]
pub struct Applied {
    pub artifact: Artifact,
    pub operator_mapping: Vec<usize>,
    pub node_mapping: Vec<usize>,
}
pub fn apply_mapped(a: &Artifact, r: &Region, body: &JointBody) -> Result<Applied, String> {
    let input_types = validate_region(a, r)?;
    if body.inputs != input_types || body.exits.len() != r.native_writes.len() {
        return Err("joint body input/exits mismatch".into());
    }
    let exit_ids = body
        .exits
        .iter()
        .map(|e| e.native_write)
        .collect::<BTreeSet<_>>();
    if exit_ids.len() != body.exits.len() || exit_ids != r.native_writes.iter().copied().collect() {
        return Err("joint exits must biject declared native writes".into());
    }
    let mut exit_views = BTreeMap::<usize, Vec<usize>>::new();
    for (index, exit) in body.exits.iter().enumerate() {
        if exit.node >= body.nodes.len() {
            return Err("joint exit node absent".into());
        }
        exit_views.entry(exit.node).or_default().push(index);
    }
    for (&node, views) in &exit_views {
        if views.len() > 1 && body.nodes.iter().any(|n| n.arguments().contains(&node)) {
            return Err(
                "equal exits with internal consumers have ambiguous native control dependence"
                    .into(),
            );
        }
    }
    let n = a.program.nodes.len();
    let size = n
        .checked_add(body.nodes.len())
        .and_then(|n| n.checked_add(body.exits.len()))
        .ok_or("joint node count overflow")?;
    let internal = r
        .current_internal_nodes
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut graph = vec![None; size];
    let mut local = vec![usize::MAX; body.nodes.len()];
    let mut program = a.program.clone();
    program
        .operators
        .extend(body.operators.iter().cloned().map(Arc::new));
    let ops = (0..program.operators.len()).collect::<Vec<_>>();
    let bases = (0..program.bases.len()).collect::<Vec<_>>();
    let rules = (0..program.rules.len()).collect::<Vec<_>>();
    for (i, node) in body.nodes.iter().enumerate() {
        match node {
            Node::Param { index } => {
                local[i] = *r
                    .current_reads
                    .get(*index)
                    .ok_or("joint Param outside boundary")?
            }
            Node::Raw { .. } | Node::Feature { .. } => {
                return Err("ambient input inside candidate joint body".into());
            }
            other => {
                if other.arguments().iter().any(|&p| p >= i) {
                    return Err("joint body forward reference/cycle".into());
                }
                if other.operators().iter().any(|&o| o >= ops.len()) {
                    return Err("joint body unknown operator".into());
                }
                if matches!(other, Node::Call { rule, .. } if *rule >= rules.len())
                    || matches!(other, Node::Readout { basis, .. } if *basis >= bases.len())
                {
                    return Err("joint body unknown rule/basis".into());
                }
                let mut copy = other.clone();
                remap_node(&mut copy, &local, &ops, &bases, &rules);
                local[i] = n + i;
                graph[n + i] = Some(copy);
            }
        }
        if let Some(views) = exit_views.get(&i) {
            let source = local[i];
            for &index in views {
                graph[n + body.nodes.len() + index] = Some(Node::Gain {
                    input: source,
                    coefficient: Coefficient::Number(1.),
                });
            }
            if views.len() == 1 {
                local[i] = n + body.nodes.len() + views[0];
            }
        }
    }
    let mut changed = BTreeMap::new();
    for (index, e) in body.exits.iter().enumerate() {
        let old = *r
            .current_writes
            .get(
                r.native_writes
                    .iter()
                    .position(|id| *id == e.native_write)
                    .ok_or("unknown joint exit")?,
            )
            .ok_or("missing joint exit")?;
        changed.insert(old, n + body.nodes.len() + index);
    }
    let mut old_to_provisional = (0..n).collect::<Vec<_>>();
    for &i in &internal {
        old_to_provisional[i] = changed.get(&i).copied().unwrap_or(usize::MAX);
    }
    for (i, node) in a.program.nodes.iter().enumerate() {
        if !internal.contains(&i) {
            if node
                .arguments()
                .iter()
                .any(|&p| old_to_provisional[p] == usize::MAX)
            {
                return Err("external consumer of erased joint interior".into());
            }
            let mut copy = node.clone();
            remap_node(&mut copy, &old_to_provisional, &ops, &bases, &rules);
            graph[i] = Some(copy);
        }
    }
    // Deterministic topological ordering also permits independent later boundary inputs.
    let mut pending = vec![0; size];
    let mut users = vec![vec![]; size];
    let mut ready = BTreeSet::new();
    let mut count = 0;
    for (i, node) in graph.iter().enumerate() {
        if let Some(node) = node {
            count += 1;
            let parents = node.arguments().into_iter().collect::<BTreeSet<_>>();
            for p in &parents {
                if *p >= size || graph[*p].is_none() {
                    return Err("joint DAG missing parent".into());
                }
                users[*p].push(i);
            }
            pending[i] = parents.len();
            if parents.is_empty() {
                ready.insert(i);
            }
        }
    }
    let mut order = vec![];
    while let Some(i) = ready.pop_first() {
        order.push(i);
        for &u in &users[i] {
            pending[u] -= 1;
            if pending[u] == 0 {
                ready.insert(u);
            }
        }
    }
    if order.len() != count {
        return Err("joint replacement creates cycle through boundary inputs".into());
    }
    let mut topo = vec![usize::MAX; size];
    for (index, &old) in order.iter().enumerate() {
        topo[old] = index;
    }
    program.nodes = order
        .iter()
        .map(|&i| {
            let mut node = graph[i].clone().expect("ordered active node");
            remap_node(&mut node, &topo, &ops, &bases, &rules);
            node
        })
        .collect();
    program.output = topo[*old_to_provisional
        .get(a.program.output)
        .ok_or("missing output")?];
    let old_map = old_to_provisional
        .iter()
        .map(|&i| if i == usize::MAX { usize::MAX } else { topo[i] })
        .collect::<Vec<_>>();
    let interfaces = program.interfaces().map_err(|e| e.to_string())?;
    let native_types = a.program.interfaces().map_err(|e| e.to_string())?;
    for (&old, &new) in &changed {
        if interfaces[topo[new]] != native_types[old] {
            return Err("joint exit interface differs from native write".into());
        }
    }
    let mut out = a.clone();
    out.program = program;
    out.places = a
        .places
        .iter()
        .filter_map(|&(native, current)| {
            if old_map[current] == usize::MAX {
                None
            } else {
                Some((native, old_map[current]))
            }
        })
        .collect();
    out.blocks = vec![];
    for b in &a.blocks {
        if internal.contains(&b.write) {
            continue;
        }
        if old_map[b.write] == usize::MAX || b.reads.iter().any(|&i| old_map[i] == usize::MAX) {
            return Err("surviving block lost its read/write boundary".into());
        }
        let mut copy = b.clone();
        copy.write = old_map[b.write];
        copy.reads = b.reads.iter().map(|&i| old_map[i]).collect();
        out.blocks.push(copy);
    }
    for (&native, &current) in r.native_writes.iter().zip(&r.current_writes) {
        out.blocks.push(Binding {
            name: "joint observable exit".into(),
            native_reads: r.native_reads.clone(),
            native_write: native,
            reads: r.current_reads.iter().map(|&i| old_map[i]).collect(),
            write: old_map[current],
        });
    }
    for e in &mut out.exceptions {
        if old_map[e.node] == usize::MAX {
            return Err("joint rewrite erased exception boundary".into());
        }
        e.node = old_map[e.node];
    }
    for c in &mut out.controls {
        if old_map[c.write] == usize::MAX {
            return Err("joint rewrite erased control boundary".into());
        }
        c.write = old_map[c.write];
    }
    // Declared interfaces remain real executable nodes even if an equation ignores
    // an argument, matching ordinary Call argument semantics. They remain priced.
    let roots = out
        .blocks
        .iter()
        .flat_map(|b| b.reads.iter().copied().chain(std::iter::once(b.write)))
        .chain(out.controls.iter().map(|c| c.write))
        .chain(out.exceptions.iter().map(|e| e.node))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let (live, operator_map) = compact_with_roots(&mut out.program, &a.derived_operators(), &roots);
    out.derived = a.renumbered_derived(&operator_map)?;
    for b in &mut out.blocks {
        if live[b.write] == usize::MAX || b.reads.iter().any(|&i| live[i] == usize::MAX) {
            return Err("compaction erased a declared joint/block boundary".into());
        }
        b.write = live[b.write];
        for read in &mut b.reads {
            *read = live[*read];
        }
    }
    for e in &mut out.exceptions {
        if live[e.node] == usize::MAX {
            return Err("compaction erased exception".into());
        }
        e.node = live[e.node];
    }
    for c in &mut out.controls {
        if live[c.write] == usize::MAX {
            return Err("compaction erased control".into());
        }
        c.write = live[c.write];
    }
    out.places = out
        .places
        .into_iter()
        .filter_map(|(native, current)| {
            if live[current] == usize::MAX {
                None
            } else {
                Some((native, live[current]))
            }
        })
        .collect();
    crate::native_control::validate_shape(&out)?;
    out.program.interfaces().map_err(|e| e.to_string())?;
    let node_mapping = local
        .iter()
        .map(|&node| {
            if node == usize::MAX || topo[node] == usize::MAX {
                usize::MAX
            } else {
                live[topo[node]]
            }
        })
        .collect();
    Ok(Applied {
        artifact: out,
        operator_mapping: operator_map,
        node_mapping,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::Exception,
        operator_program::{
            Declarations, FamilyInputs, Law, OperatorProgram, Rule, Scale, SequenceLayout, Slot,
            SlotValues,
        },
    };
    use ndarray::{Array2, array};
    fn describe(
        a: &Artifact,
        producer: usize,
        inside: BTreeSet<usize>,
        limits: &Limits,
    ) -> Result<Region, String> {
        describe_anchored(a, producer, AnchorMode::Interior, inside, limits)
    }
    fn limits() -> Limits {
        Limits {
            max_internal_nodes: 12,
            max_inputs: 8,
            max_exits: 8,
            max_regions: 2000,
            max_states: 3000,
        }
    }
    fn base(nodes: Vec<Node>, slots: usize) -> Artifact {
        let p = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }; slots],
                parameters: 0,
            },
            nodes,
            operators: vec![Arc::new(Operator::identity(
                "I",
                Interface::native(2).expect("interface"),
            ))],
            bases: vec![],
            rules: vec![],
            output: 0,
        };
        let mut p = p;
        p.output = p.nodes.len() - 1;
        Artifact::native(&p).expect("native")
    }
    fn source() -> Artifact {
        base(
            vec![
                Node::Raw { slot: 0 },
                Node::Raw { slot: 1 },
                Node::Affine {
                    terms: vec![(0, 0), (1, 0)],
                    bias: None,
                },
                Node::Hadamard { left: 2, right: 0 },
                Node::Gain {
                    input: 2,
                    coefficient: Coefficient::Number(2.),
                },
                Node::Affine {
                    terms: vec![(2, 0), (1, 0)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(3, 0), (4, 0), (5, 0)],
                    bias: None,
                },
            ],
            2,
        )
    }
    fn region(a: &Artifact, inside: &[usize]) -> Region {
        describe(a, 2, inside.iter().copied().collect(), &limits()).expect("joint cut")
    }
    #[test]
    fn scarce_budget_round_robins_high_and_low_fanout_anchor_modes() {
        let mut nodes = vec![
            Node::Raw { slot: 0 },
            Node::Raw { slot: 1 },
            Node::RmsNorm {
                input: 0,
                epsilon: 1e-5,
            },
        ];
        for _branch in 0..18 {
            nodes.push(Node::Gain {
                input: 2,
                coefficient: Coefficient::Number(2.),
            });
        }
        let later = nodes.len();
        nodes.push(Node::RmsNorm {
            input: 1,
            epsilon: 1e-5,
        });
        let left = nodes.len();
        nodes.push(Node::Gain {
            input: later,
            coefficient: Coefficient::Number(3.),
        });
        let right = nodes.len();
        nodes.push(Node::Gain {
            input: later,
            coefficient: Coefficient::Number(4.),
        });
        nodes.push(Node::Affine {
            terms: (3..21).chain([left, right]).map(|i| (i, 0)).collect(),
            bias: None,
        });
        let a = base(nodes, 2);
        let inventory = propose_regions(
            &a,
            Limits {
                max_states: 40,
                ..limits()
            },
        )
        .expect("fair bounded inventory");
        assert!(inventory.truncated);
        assert_eq!(inventory.explored_states, 40);
        assert!(inventory.attempted_expansions > 0);
        for producer in [2, later] {
            for mode in [AnchorMode::Interior, AnchorMode::Boundary] {
                let initial_len = usize::from(mode == AnchorMode::Interior);
                assert!(
                    inventory
                        .regions
                        .iter()
                        .any(|r| r.producer_native == producer && r.anchor_mode == mode)
                        || inventory
                            .skipped
                            .iter()
                            .any(|r| r.producer_native == producer
                                && r.anchor_mode == mode
                                && r.current_internal_nodes.len() > initial_len),
                    "each anchor mode must describe an admitted successor"
                );
            }
        }
        assert!(inventory.regions.iter().any(|r| r.producer_native == later
            && r.anchor_mode == AnchorMode::Boundary
            && r.current_internal_nodes == vec![left, right]));
    }
    #[test]
    fn boundary_anchor_retains_normalization_and_raw_forks() {
        for normalized in [false, true] {
            let mut nodes = vec![Node::Raw { slot: 0 }, Node::Raw { slot: 1 }];
            let anchor = if normalized {
                nodes.push(Node::RmsNorm {
                    input: 0,
                    epsilon: 1e-5,
                });
                2
            } else {
                0
            };
            let left = nodes.len();
            nodes.push(Node::Gain {
                input: anchor,
                coefficient: Coefficient::Number(2.),
            });
            let right = nodes.len();
            nodes.push(Node::Hadamard {
                left: anchor,
                right: 1,
            });
            nodes.push(Node::Affine {
                terms: vec![(left, 0), (right, 0)],
                bias: None,
            });
            let a = base(nodes, 2);
            let inventory = propose_regions(&a, limits()).expect("bounded fork inventory");
            let r = inventory
                .regions
                .iter()
                .find(|r| {
                    r.anchor_mode == AnchorMode::Boundary
                        && r.producer_native == anchor
                        && r.current_internal_nodes == vec![left, right]
                })
                .expect("two-branch boundary cut is enumerated");
            assert!(r.current_reads.contains(&anchor));
            assert!(!r.current_internal_nodes.contains(&anchor));
            assert_eq!(r.native_writes, vec![left, right]);
            let body = exact_body(&a, r).expect("boundary extraction");
            let patched = replay(&apply(&a, r, &body).expect("independent exits patch"));
            let trace = patched
                .program
                .execute(&panel(), false)
                .expect("patched execution");
            assert!(trace.values.iter().flatten().all(|x| x.is_finite()));
            assert_eq!(result(&patched, &panel()), result(&a, &panel()));
            assert!(patched.place(anchor).is_some());
            let left_place = patched.place(left).expect("first decoded exit");
            let right_place = patched.place(right).expect("second decoded exit");
            let edited = patched
                .program
                .execute_edited(&panel(), |i, value, _| {
                    if i == left_place {
                        *value *= 0.;
                    }
                    Ok(())
                })
                .expect("independent boundary exit intervention");
            assert_eq!(edited.values[right_place], trace.values[right_place]);
            assert!(edited.values[left_place].iter().all(|&x| x == 0.));
        }
    }
    #[test]
    fn boundary_anchor_refuses_unread_and_disconnected_branches() {
        let a = source();
        assert!(
            describe_anchored(
                &a,
                0,
                AnchorMode::Boundary,
                BTreeSet::from([4, 5]),
                &limits()
            )
            .is_err()
        );
        let a = base(
            vec![
                Node::Raw { slot: 0 },
                Node::Raw { slot: 1 },
                Node::Gain {
                    input: 0,
                    coefficient: Coefficient::Number(2.),
                },
                Node::Gain {
                    input: 0,
                    coefficient: Coefficient::Number(3.),
                },
                Node::Gain {
                    input: 1,
                    coefficient: Coefficient::Number(4.),
                },
                Node::Affine {
                    terms: vec![(2, 0), (3, 0), (4, 0)],
                    bias: None,
                },
            ],
            2,
        );
        assert!(
            describe_anchored(
                &a,
                0,
                AnchorMode::Boundary,
                BTreeSet::from([2, 3, 4]),
                &limits()
            )
            .is_err()
        );
        let r = describe_anchored(
            &a,
            0,
            AnchorMode::Boundary,
            BTreeSet::from([2, 3]),
            &limits(),
        )
        .expect("connected boundary cut");
        let mut forged = r.clone();
        forged.anchor_mode = AnchorMode::Interior;
        assert!(exact_body(&a, &forged).is_err());
        let mut legacy = serde_json::to_value(&region(&source(), &[2, 3, 4])).expect("region JSON");
        legacy
            .as_object_mut()
            .expect("region object")
            .remove("anchor_mode");
        let restored: Region = serde_json::from_value(legacy).expect("legacy anchor default");
        assert_eq!(restored.anchor_mode, AnchorMode::Interior);
    }
    #[test]
    fn mapped_rewrite_preserves_exact_parameter_and_value_identities() {
        let a = source();
        let r = region(&a, &[2, 3, 4]);
        let mut body = exact_body(&a, &r).expect("body");
        let first = a.program.operators.len();
        body.operators = vec![
            Operator::identity("new used", Interface::native(2).unwrap()),
            Operator::identity("new unused", Interface::native(2).unwrap()),
        ];
        for node in &mut body.nodes {
            if let Node::Affine { terms, .. } = node {
                for (_, operator) in terms {
                    if *operator == 0 {
                        *operator = first;
                    }
                }
            }
        }
        let mapped = apply_mapped(&a, &r, &body).expect("mapped rewrite");
        assert_ne!(mapped.operator_mapping[first], usize::MAX);
        assert_eq!(mapped.operator_mapping[first + 1], usize::MAX);
        assert_eq!(
            mapped.artifact.program,
            apply(&a, &r, &body).unwrap().program
        );
        let trace = mapped.artifact.program.execute(&panel(), false).unwrap();
        let native = a.program.execute(&panel(), false).unwrap();
        for exit in &body.exits {
            let node = mapped.node_mapping[exit.node];
            assert_eq!(Some(node), mapped.artifact.place(exit.native_write));
            assert_eq!(trace.values[node], native.values[exit.native_write]);
        }
        assert_eq!(result(&mapped.artifact, &panel()), result(&a, &panel()));
    }
    fn panel() -> FamilyInputs {
        FamilyInputs {
            rows: 2,
            slots: vec![
                SlotValues::Raw(array![[1., 2.], [2., -1.]]),
                SlotValues::Raw(array![[3., 1.], [0.5, 2.]]),
            ],
            layout: Some(SequenceLayout {
                sequence: vec![0, 0],
                position: vec![0, 1],
            }),
        }
    }
    fn result(a: &Artifact, x: &FamilyInputs) -> Array2<f64> {
        a.program.execute(x, false).expect("execute").values[a.program.output].clone()
    }
    fn replay(a: &Artifact) -> Artifact {
        let b = a.to_bytes().expect("encode");
        let d = Artifact::from_bytes(&b, &a.program.declarations).expect("decode");
        assert_eq!(d.to_bytes().expect("canonical"), b);
        d
    }
    #[test]
    fn atomic_producer_two_consumers_outside_reader_and_native_patch() {
        let a = source();
        let r = region(&a, &[2, 3, 4]);
        let mut body = exact_body(&a, &r).expect("extract");
        body.nodes[2] = Node::Gain {
            input: 1,
            coefficient: Coefficient::Number(-1.),
        };
        body.nodes.insert(
            3,
            Node::Affine {
                terms: vec![(0, 0), (2, 0)],
                bias: None,
            },
        );
        body.nodes[4] = Node::Hadamard { left: 3, right: 0 };
        body.nodes[5] = Node::Gain {
            input: 3,
            coefficient: Coefficient::Number(2.),
        };
        body.exits = vec![
            Exit {
                native_write: 2,
                node: 3,
            },
            Exit {
                native_write: 3,
                node: 4,
            },
            Exit {
                native_write: 4,
                node: 5,
            },
        ];
        let b = replay(&apply(&a, &r, &body).expect("atomic replacement"));
        let x = panel();
        assert_eq!(result(&b, &x), array![[-5., 6.], [8., -4.]]);
        let producer = b.place(2).expect("retained producer");
        let edited = b
            .program
            .execute_edited(&x, |i, v, _| {
                if i == producer {
                    *v *= 2.;
                }
                Ok(())
            })
            .expect("producer patch");
        // All three consumers (two inside, one outside) see the patched producer.
        assert_eq!(
            edited.values[b.program.output],
            array![[-13., 11.], [15.5, -10.]]
        );
        assert_eq!(b.program.nodes.iter().filter(|n|matches!(n,Node::Affine{terms,..}if terms.len()==2&&terms.iter().any(|(i,_)|matches!(b.program.nodes[*i],Node::Gain{coefficient:Coefficient::Number(-1.),..})))).count(),1);
    }
    #[test]
    fn equal_leaf_exits_are_independently_patchable_after_saved_decode() {
        let a = source();
        let r = region(&a, &[2, 3, 4]);
        let body = JointBody {
            inputs: validate_region(&a, &r).expect("types"),
            nodes: vec![Node::Param { index: 0 }, Node::Param { index: 1 }],
            exits: vec![
                Exit {
                    native_write: 2,
                    node: 1,
                },
                Exit {
                    native_write: 3,
                    node: 0,
                },
                Exit {
                    native_write: 4,
                    node: 0,
                },
            ],
            operators: vec![],
        };
        let b = replay(&apply(&a, &r, &body).expect("distinct identity views"));
        let x = panel();
        assert_ne!(b.place(3), b.place(4));
        assert_ne!(b.place(3), b.place(r.native_reads[0]));
        let p3 = b.place(3).expect("exit3");
        let p4 = b.place(4).expect("exit4");
        let clean = b.program.execute(&x, false).expect("clean");
        let edited = b
            .program
            .execute_edited(&x, |i, v, _| {
                if i == p3 {
                    *v *= 0.;
                }
                Ok(())
            })
            .expect("independent leaf patch");
        assert_eq!(edited.values[p4], clean.values[p4]);
        assert!(edited.values[p3].iter().all(|&v| v == 0.));
        let mut ambiguous = body;
        ambiguous.nodes.push(Node::Gain {
            input: 0,
            coefficient: Coefficient::Number(2.),
        });
        assert!(
            apply(&a, &r, &ambiguous)
                .expect_err("shared exit internal dependence ambiguous")
                .contains("ambiguous")
        );
    }
    #[test]
    fn automatic_inventory_exact_extraction_and_protected_metadata() {
        let a = source();
        let inv = propose_regions(&a, limits()).expect("inventory");
        assert!(
            inv.regions
                .iter()
                .any(|r| r.producer_native == 2 && r.current_internal_nodes == vec![2, 3, 4])
        );
        let r = region(&a, &[2, 3, 4, 5]);
        assert!(!r.native_writes.contains(&2));
        assert!(r.erased_native_places.contains(&2));
        let b = apply(&a, &r, &exact_body(&a, &r).expect("body")).expect("exact multiwrite");
        assert_eq!(result(&a, &panel()), result(&replay(&b), &panel()));
        assert!(b.place(2).is_none());
        let mut protected = a.clone();
        protected.exceptions.push(Exception {
            context: vec![],
            node: 2,
            column: 0,
            value: 0.5,
        });
        let guarded = region(&protected, &[2, 3, 4, 5]);
        assert!(guarded.native_writes.contains(&2));
        let mut body = exact_body(&protected, &guarded).expect("guarded body");
        body.exits.retain(|e| e.native_write != 2);
        assert!(apply(&protected, &guarded, &body).is_err());
        let tiny = propose_regions(
            &a,
            Limits {
                max_states: 1,
                ..limits()
            },
        )
        .expect("bounded");
        assert!(tiny.truncated);
        assert_eq!(tiny.explored_states, 1);
    }
    #[test]
    fn attention_and_nested_calls_use_existing_execution() {
        let mut a = base(
            vec![
                Node::Raw { slot: 0 },
                Node::Raw { slot: 1 },
                Node::Raw { slot: 2 },
                Node::RmsNorm {
                    input: 2,
                    epsilon: 1e-6,
                },
                Node::Attend {
                    query: 3,
                    key: 0,
                    value: 1,
                    scale: Scale::One,
                    rotary: None,
                    causal: true,
                },
                Node::Gain {
                    input: 3,
                    coefficient: Coefficient::Number(1.),
                },
                Node::Affine {
                    terms: vec![(4, 0), (5, 0)],
                    bias: None,
                },
            ],
            3,
        );
        a.program.rules.push(Rule {
            name: "nonlinear".into(),
            inputs: vec![Interface::native(2).expect("interface")],
            nodes: vec![
                Node::Param { index: 0 },
                Node::Pointwise {
                    input: 0,
                    laws: vec![Law::Relu],
                },
            ],
            output: 1,
        });
        a.program.nodes[5] = Node::Call {
            rule: 0,
            arguments: vec![3],
        };
        let r = describe(&a, 3, BTreeSet::from([3, 4, 5]), &limits()).expect("attention joint cut");
        let body = exact_body(&a, &r).expect("attention body");
        let b = apply(&a, &r, &body).expect("attention apply");
        let mut x = panel();
        x.slots.push(SlotValues::Raw(array![[0.5, -1.], [2., 3.]]));
        assert_eq!(result(&a, &x), result(&replay(&b), &x));
    }
    #[test]
    fn forward_hidden_reads_stale_regions_and_late_native_boundaries_rejected() {
        let a = source();
        let r = region(&a, &[2, 3, 4]);
        let mut body = exact_body(&a, &r).expect("body");
        body.nodes.push(Node::Raw { slot: 0 });
        assert!(apply(&a, &r, &body).is_err());
        let mut body = exact_body(&a, &r).expect("body");
        body.nodes[2] = Node::Gain {
            input: 3,
            coefficient: Coefficient::Number(1.),
        };
        assert!(apply(&a, &r, &body).is_err());
        let mut stale = r.clone();
        stale.source_program_nodes += 1;
        assert!(exact_body(&a, &stale).is_err());
        let late = base(
            vec![
                Node::Raw { slot: 0 },
                Node::Gain {
                    input: 0,
                    coefficient: Coefficient::Number(2.),
                },
                Node::Raw { slot: 1 },
                Node::Affine {
                    terms: vec![(1, 0), (2, 0)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(1, 0), (3, 0)],
                    bias: None,
                },
            ],
            2,
        );
        assert!(
            describe(&late, 1, BTreeSet::from([1, 3]), &limits())
                .expect_err("late input not available to earlier exit")
                .contains("not before")
        );
    }
    #[test]
    fn paid_controls_blocks_and_derived_operator_compaction_preserved() {
        let mut a = source();
        a.program.nodes[4] = Node::Affine {
            terms: vec![(3, 0)],
            bias: None,
        };
        a.program.nodes[6] = Node::Affine {
            terms: vec![(4, 0), (5, 0)],
            bias: None,
        };
        let native = a.program.clone();
        a.places.retain(|(native, _)| *native != 3);
        a = a
            .bind("linear control", &[0, 1], 4)
            .expect("old binding")
            .with_uniform_scale_control(&native, 3, 4)
            .expect("paid native control");
        let ty = Interface::native(2).expect("interface");
        a.program.operators.push(Arc::new(
            Operator::dense(
                "unused",
                ty.clone(),
                ty.clone(),
                Array2::ones((2, 2)),
                crate::precision::DeclaredPrecision::new(0).expect("precision"),
                Default::default(),
            )
            .expect("unused operator"),
        ));
        a.program
            .operators
            .push(Arc::new(Operator::identity("derived target", ty)));
        let matrix = crate::matrix_rule::MatrixRule {
            inputs: vec![crate::matrix_rule::Type::Matrix { rows: 2, cols: 2 }],
            nodes: vec![crate::matrix_rule::Node::Param { index: 0 }],
            output: 0,
        };
        a = a
            .derive(
                2,
                crate::artifact::OperatorLaw::Expression {
                    body: Arc::new(matrix),
                    sources: vec![0],
                },
                1.,
                vec![],
            )
            .expect("derived metadata");
        let r = region(&a, &[2, 3, 4, 5]);
        let b =
            apply(&a, &r, &exact_body(&a, &r).expect("exact body")).expect("guarded joint body");
        assert_eq!(b.controls.len(), 1);
        assert_eq!(b.controls[0].write, b.place(4).expect("held control write"));
        assert_eq!(b.blocks.iter().filter(|b| b.native_write == 4).count(), 1);
        assert_eq!(b.derived[0].operator, 1);
        assert_eq!(
            b.program.operators.len(),
            2,
            "unused payload removed, derived source/target retained"
        );
        assert!(Arc::ptr_eq(
            &a.program.operators[0],
            &b.program.operators[0]
        ));
        b.validate_coverage(&native)
            .expect("native controls/coverage");
        let decoded = replay(&b);
        assert_eq!(decoded.controls, b.controls);
        assert_eq!(decoded.derived, b.derived);
        assert_eq!(result(&a, &panel()), result(&decoded, &panel()));
    }
    #[test]
    fn unused_declared_boundary_survives_compaction_and_codec() {
        let a = base(
            vec![
                Node::Raw { slot: 0 },
                Node::Raw { slot: 1 },
                Node::Affine {
                    terms: vec![(0, 0), (1, 0)],
                    bias: None,
                },
                Node::Hadamard { left: 2, right: 0 },
                Node::Gain {
                    input: 2,
                    coefficient: Coefficient::Number(2.),
                },
                Node::Affine {
                    terms: vec![(3, 0), (4, 0)],
                    bias: None,
                },
            ],
            2,
        );
        let r = region(&a, &[2, 3, 4]);
        let body = JointBody {
            inputs: validate_region(&a, &r).expect("types"),
            nodes: vec![
                Node::Param { index: 0 },
                Node::Param { index: 1 },
                Node::Gain {
                    input: 0,
                    coefficient: Coefficient::Number(2.),
                },
            ],
            exits: vec![
                Exit {
                    native_write: 3,
                    node: 0,
                },
                Exit {
                    native_write: 4,
                    node: 2,
                },
            ],
            operators: vec![],
        };
        let b = replay(&apply(&a, &r, &body).expect("unused input retained"));
        let unused = b.place(1).expect("retained unused interface");
        assert!(b.blocks.iter().all(|b| b.reads.contains(&unused)));
        let trace = b
            .program
            .execute(&panel(), false)
            .expect("ordinary autonomous execution");
        assert_eq!(trace.values[unused], array![[3., 1.], [0.5, 2.]]);
        for binding in &b.blocks {
            assert!(binding.reads.iter().all(|&read| read < binding.write));
        }
        b.validate_coverage(&a.program)
            .expect("native Local bindings remain valid");
    }
}
