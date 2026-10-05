//! Bounded cuts in the current artifact graph. Exact extraction is a structural move,
//! not evidence of discovery, compression, or preserved erased interventions.
use crate::{
    artifact::{Argument, Artifact, Callee},
    operator_program::{Node, Rule, remap_node},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Limits {
    pub max_internal_nodes: usize,
    pub max_inputs: usize,
    pub max_regions: usize,
    pub max_states: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Region {
    pub native_reads: Vec<usize>,
    pub native_write: usize,
    pub current_reads: Vec<usize>,
    pub current_write: usize,
    pub current_internal_nodes: Vec<usize>,
    pub internal_native_places: Vec<usize>,
    pub source_program_nodes: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Skipped {
    pub native_write: usize,
    pub current_internal_nodes: Vec<usize>,
    pub reason: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Inventory {
    pub regions: Vec<Region>,
    pub skipped: Vec<Skipped>,
    pub truncated: bool,
    pub explored_states: usize,
}
fn ambient(node: &Node) -> bool {
    matches!(
        node,
        Node::Raw { .. } | Node::Feature { .. } | Node::Param { .. }
    )
}
fn aliases(artifact: &Artifact) -> Result<BTreeMap<usize, usize>, String> {
    let mut reverse = BTreeMap::new();
    for (native, current) in &artifact.places {
        if *native >= artifact.native_nodes || *current >= artifact.program.nodes.len() {
            return Err("invalid native place in region source".into());
        }
        if reverse.insert(*current, *native).is_some() {
            return Err("ambiguous native aliases at one region boundary".into());
        }
    }
    Ok(reverse)
}
// Non-native intermediate parents cannot become invisible inputs: include them or fail.
fn close(
    artifact: &Artifact,
    mut internal: BTreeSet<usize>,
    places: &BTreeMap<usize, usize>,
    limit: usize,
) -> Result<(BTreeSet<usize>, BTreeSet<usize>), String> {
    loop {
        if internal.len() > limit {
            return Err("max_internal_nodes exceeded".into());
        }
        let mut missing = BTreeSet::new();
        let mut boundary = BTreeSet::new();
        for id in &internal {
            let node = artifact
                .program
                .nodes
                .get(*id)
                .ok_or("region node missing")?;
            if ambient(node) {
                return Err(format!(
                    "ambient input node {id} cannot be inside extracted body"
                ));
            }
            for parent in node.arguments() {
                if !internal.contains(&parent) {
                    if places.contains_key(&parent) {
                        boundary.insert(parent);
                    } else {
                        missing.insert(parent);
                    }
                }
            }
        }
        if boundary
            .iter()
            .any(|id| matches!(artifact.program.nodes[*id], Node::Feature { .. }))
        {
            return Err("Feature boundary unsupported: gathered tokens are not ordinary explicit matrix arguments".into());
        }
        if missing.is_empty() {
            return Ok((internal, boundary));
        }
        internal.extend(missing);
    }
}
pub fn propose_regions(artifact: &Artifact, limits: Limits) -> Result<Inventory, String> {
    if limits.max_internal_nodes == 0 || limits.max_regions == 0 || limits.max_states == 0 {
        return Err("positive region/node/state limits required".into());
    }
    artifact.program.interfaces().map_err(|e| e.to_string())?;
    let places = aliases(artifact)?;
    let mut result = Inventory {
        regions: vec![],
        skipped: vec![],
        truncated: false,
        explored_states: 0,
    };
    // One initial cut at every native write precedes every deeper ancestor expansion.
    let mut queue = artifact
        .places
        .iter()
        .map(|(native, current)| (*native, *current, BTreeSet::from([*current])))
        .collect::<VecDeque<_>>();
    let mut seen = BTreeMap::<usize, BTreeSet<Vec<usize>>>::new();
    while let Some((native_write, current_write, cut)) = queue.pop_front() {
        if result.regions.len() >= limits.max_regions || result.explored_states >= limits.max_states
        {
            result.truncated = true;
            queue.push_front((native_write, current_write, cut));
            for (native_write, _, cut) in queue {
                result.skipped.push(Skipped {
                    native_write,
                    current_internal_nodes: cut.into_iter().collect(),
                    reason: "queued cut unresolved: declared global region/state budget exhausted"
                        .into(),
                });
            }
            break;
        }
        let visited = seen.entry(native_write).or_default();
        if !visited.insert(cut.iter().copied().collect()) {
            continue;
        }
        result.explored_states += 1;
        if artifact.program.declarations.parameters != 0 {
            result.skipped.push(Skipped {
                native_write,
                current_internal_nodes: cut.into_iter().collect(),
                reason: "external scalar parameters unsupported by exact region extraction".into(),
            });
            continue;
        }
        let (internal, boundary) =
            match close(artifact, cut.clone(), &places, limits.max_internal_nodes) {
                Ok(v) => v,
                Err(reason) => {
                    result.skipped.push(Skipped {
                        native_write,
                        current_internal_nodes: cut.into_iter().collect(),
                        reason,
                    });
                    continue;
                }
            };
        if internal != cut && !visited.insert(internal.iter().copied().collect()) {
            continue;
        }
        if boundary.len() <= limits.max_inputs {
            let current_reads = boundary.iter().copied().collect::<Vec<_>>();
            result.regions.push(Region {
                native_reads: current_reads.iter().map(|n| places[n]).collect(),
                native_write,
                current_reads,
                current_write,
                current_internal_nodes: internal.iter().copied().collect(),
                internal_native_places: internal
                    .iter()
                    .filter_map(|n| places.get(n).copied())
                    .collect(),
                source_program_nodes: artifact.program.nodes.len(),
            });
        } else {
            result.skipped.push(Skipped {
                native_write,
                current_internal_nodes: internal.iter().copied().collect(),
                reason: "cut exceeds max_inputs; ancestor expansions still explored".into(),
            });
        }
        if internal.len() < limits.max_internal_nodes {
            for parent in boundary {
                if !ambient(&artifact.program.nodes[parent]) {
                    let mut expanded = internal.clone();
                    expanded.insert(parent);
                    queue.push_back((native_write, current_write, expanded));
                }
            }
        }
    }
    Ok(result)
}
fn checked_rule(artifact: &Artifact, region: &Region) -> Result<Rule, String> {
    if region.source_program_nodes != artifact.program.nodes.len()
        || artifact.place(region.native_write) != Some(region.current_write)
        || region.native_reads.len() != region.current_reads.len()
        || region
            .native_reads
            .iter()
            .zip(&region.current_reads)
            .any(|(native, current)| artifact.place(*native) != Some(*current))
    {
        return Err("region source places changed".into());
    }
    if region
        .current_reads
        .iter()
        .any(|id| matches!(artifact.program.nodes.get(*id), Some(Node::Feature { .. })))
    {
        return Err("Feature boundary unsupported for explicit Rule arguments".into());
    }
    let interfaces = artifact.program.interfaces().map_err(|e| e.to_string())?;
    let internal = region
        .current_internal_nodes
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let boundaries = region
        .current_reads
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if internal.len() != region.current_internal_nodes.len()
        || boundaries.len() != region.current_reads.len()
        || !internal.contains(&region.current_write)
        || internal.intersection(&boundaries).next().is_some()
        || region
            .current_reads
            .iter()
            .any(|n| *n >= region.current_write)
    {
        return Err("invalid region cut ordering/duplicates".into());
    }
    let mut reachable = BTreeSet::new();
    let mut todo = vec![region.current_write];
    while let Some(id) = todo.pop() {
        if internal.contains(&id) && reachable.insert(id) {
            todo.extend(artifact.program.nodes[id].arguments());
        }
    }
    if reachable != internal {
        return Err("region contains nodes outside write ancestor closure".into());
    }
    if artifact
        .exceptions
        .iter()
        .any(|e| internal.contains(&e.node) && e.node != region.current_write)
    {
        return Err("region would hide an internal exception activation intervention".into());
    }
    if artifact
        .controls
        .iter()
        .any(|c| internal.contains(&c.write) && c.write != region.current_write)
    {
        return Err("region would hide an internal declared native control".into());
    }
    if artifact.program.declarations.parameters != 0 {
        return Err("external scalar parameters unsupported in extracted regions".into());
    }
    let mut nodes = region
        .current_reads
        .iter()
        .enumerate()
        .map(|(index, _)| Node::Param { index })
        .collect::<Vec<_>>();
    let mut map = vec![usize::MAX; artifact.program.nodes.len()];
    for (index, current) in region.current_reads.iter().enumerate() {
        map[*current] = index
    }
    let ops = (0..artifact.program.operators.len()).collect::<Vec<_>>();
    let bases = (0..artifact.program.bases.len()).collect::<Vec<_>>();
    let rules = (0..artifact.program.rules.len()).collect::<Vec<_>>();
    for id in &internal {
        let mut node = artifact
            .program
            .nodes
            .get(*id)
            .ok_or("region node missing")?
            .clone();
        if ambient(&node)
            || node
                .arguments()
                .iter()
                .any(|parent| *parent >= map.len() || map[*parent] == usize::MAX)
        {
            return Err("region body is not closed over declared boundaries".into());
        }
        if let Node::Call { rule, .. } = &node {
            check_nested(artifact, *rule, &mut BTreeSet::new())?;
        }
        remap_node(&mut node, &map, &ops, &bases, &rules);
        map[*id] = nodes.len();
        nodes.push(node);
    }
    Ok(Rule {
        name: format!("region-native-{}", region.native_write),
        inputs: region
            .current_reads
            .iter()
            .map(|n| interfaces[*n].clone())
            .collect(),
        output: map[region.current_write],
        nodes,
    })
}
fn check_nested(
    artifact: &Artifact,
    rule: usize,
    seen: &mut BTreeSet<usize>,
) -> Result<(), String> {
    if !seen.insert(rule) {
        return Ok(());
    }
    for node in &artifact
        .program
        .rules
        .get(rule)
        .ok_or("nested rule missing")?
        .nodes
    {
        if matches!(node, Node::Raw { .. } | Node::Feature { .. }) {
            return Err("nested rule uses ambient inputs".into());
        }
        if let Node::Call { rule, .. } = node {
            check_nested(artifact, *rule, seen)?;
        }
    }
    Ok(())
}
fn apply(
    artifact: &Artifact,
    region: &Region,
    callee: Callee,
    native_arguments: &[usize],
) -> Result<Artifact, String> {
    let result = artifact.replace_block(
        &format!("region-native-{}", region.native_write),
        callee,
        native_arguments
            .iter()
            .copied()
            .map(Argument::Native)
            .collect(),
        region.native_write,
        vec![],
    )?;
    if result.controls.len() != artifact.controls.len() {
        return Err("region removes a declared native intervention control".into());
    }
    for control in &artifact.controls {
        if !result.controls.iter().any(|c| {
            c.native_source == control.native_source
                && c.native_write == control.native_write
                && c.width == control.width
        }) {
            return Err("region changes declared native intervention control coverage".into());
        }
    }
    if result.exceptions.len() != artifact.exceptions.len() {
        return Err("region removes declared exception coverage".into());
    }
    crate::native_control::validate_shape(&result)?;
    Ok(result)
}
pub fn extract(artifact: &Artifact, region: &Region) -> Result<Artifact, String> {
    let rule = checked_rule(artifact, region)?;
    apply(artifact, region, Callee::New(rule), &region.native_reads)
}
pub fn reuse_rule(artifact: &Artifact, region: &Region, ruleid: usize) -> Result<Artifact, String> {
    reuse_rule_with_binding(artifact, region, ruleid, &region.native_reads)
}
/// Bind an existing body to an explicit typed permutation of the complete cut boundary.
pub fn reuse_rule_with_binding(
    artifact: &Artifact,
    region: &Region,
    ruleid: usize,
    native_arguments: &[usize],
) -> Result<Artifact, String> {
    checked_rule(artifact, region)?;
    if native_arguments.len() != region.native_reads.len()
        || native_arguments.iter().copied().collect::<BTreeSet<_>>()
            != region.native_reads.iter().copied().collect()
    {
        return Err("reuse arguments must be a bijection of region native reads".into());
    }
    let stored = artifact
        .program
        .rules
        .get(ruleid)
        .ok_or("reuse rule missing")?;
    let interfaces = artifact.program.interfaces().map_err(|e| e.to_string())?;
    let ordered = native_arguments
        .iter()
        .map(|native| {
            artifact
                .place(*native)
                .map(|current| interfaces[current].clone())
                .ok_or("reuse native argument is absent")
        })
        .collect::<Result<Vec<_>, _>>()?;
    if stored.inputs != ordered {
        return Err("reuse rule ordered input interfaces differ".into());
    }
    check_nested(artifact, ruleid, &mut BTreeSet::new())?;
    // Existing replacement checks the call's output interface and preserves native bindings.
    apply(artifact, region, Callee::Existing(ruleid), native_arguments)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_program::{
        Declarations, FamilyInputs, Interface, Operator, OperatorProgram, Rule, Scale,
        SequenceLayout, Slot, SlotValues,
    };
    use std::sync::Arc;
    fn source() -> OperatorProgram {
        OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }],
                parameters: 0,
            },
            bases: vec![],
            operators: vec![Arc::new(Operator::identity(
                "shared",
                Interface::native(2).expect("interface"),
            ))],
            rules: vec![],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Hadamard { left: 1, right: 1 },
                Node::Affine {
                    terms: vec![(1, 0), (2, 0)],
                    bias: None,
                },
                Node::Hadamard { left: 1, right: 3 },
            ],
            output: 4,
        }
    }
    fn input() -> FamilyInputs {
        FamilyInputs {
            rows: 2,
            slots: vec![SlotValues::Raw(ndarray::array![[0.5, 1.], [2., -1.]])],
            layout: Some(SequenceLayout {
                sequence: vec![0; 2],
                position: vec![0, 1],
            }),
        }
    }
    fn limits() -> Limits {
        Limits {
            max_internal_nodes: 8,
            max_inputs: 4,
            max_regions: 100,
            max_states: 200,
        }
    }
    fn region(a: &Artifact, write: usize, internal: &[usize]) -> Region {
        propose_regions(a, limits())
            .expect("inventory")
            .regions
            .into_iter()
            .find(|r| r.native_write == write && r.current_internal_nodes == internal)
            .expect("requested exact cut")
    }
    #[test]
    fn reconvergent_external_consumer_and_serialization() {
        let p = source();
        let a = Artifact::native(&p).expect("native");
        let r = region(&a, 3, &[1, 2, 3]);
        assert_eq!(r.native_reads, vec![0]);
        let candidate = extract(&a, &r).expect("exact extraction");
        assert!(
            candidate.place(1).is_some(),
            "external consumer retains original native place"
        );
        let x = input();
        assert_eq!(
            p.execute(&x, false).expect("native execute").values[p.output],
            candidate
                .program
                .execute(&x, false)
                .expect("extracted execute")
                .values[candidate.program.output]
        );
        assert!(
            candidate
                .program
                .operators
                .iter()
                .any(|o| Arc::ptr_eq(o, &p.operators[0]))
        );
        let bytes = candidate.to_bytes().expect("encode");
        let decoded = Artifact::from_bytes(&bytes, &p.declarations).expect("decode");
        assert_eq!(bytes, decoded.to_bytes().expect("canonical"));
        assert_eq!(
            candidate
                .program
                .execute(&x, false)
                .expect("candidate")
                .values[candidate.program.output],
            decoded.program.execute(&x, false).expect("replay").values[decoded.program.output]
        );
    }
    #[test]
    fn crossrow_attention_and_nested_call() {
        let mut p = source();
        p.nodes = vec![
            Node::Raw { slot: 0 },
            Node::Affine {
                terms: vec![(0, 0)],
                bias: None,
            },
            Node::Attend {
                query: 1,
                key: 1,
                value: 1,
                scale: Scale::InverseSqrt(2),
                rotary: None,
                causal: true,
            },
        ];
        p.nodes.push(Node::RmsNorm {
            input: 2,
            epsilon: 1e-5,
        });
        p.output = 3;
        let a = Artifact::native(&p).expect("native attention");
        let r = region(&a, 3, &[1, 2, 3]);
        let extracted = extract(&a, &r).expect("crossrow extract");
        let q = extracted.place(3).expect("call place");
        assert!(matches!(extracted.program.nodes[q], Node::Call { .. }));
        let again = region(&extracted, 3, &[q]);
        let nested = extract(&extracted, &again).expect("existing nested call");
        let x = input();
        assert_eq!(
            p.execute(&x, false).expect("native").values[3],
            nested.program.execute(&x, false).expect("nested").values[nested.program.output]
        );
    }
    #[test]
    fn limits_and_ambient_sources_are_explicit() {
        let a = Artifact::native(&source()).expect("native");
        let inventory = propose_regions(
            &a,
            Limits {
                max_regions: 1,
                max_states: 2,
                ..limits()
            },
        )
        .expect("bounded");
        assert!(inventory.truncated);
        assert!(!inventory.skipped.is_empty());
        assert!(
            inventory
                .skipped
                .iter()
                .any(|s| s.reason.contains("ambient"))
        );
        let mut invalid = region(&a, 3, &[1, 2, 3]);
        invalid.current_internal_nodes.insert(0, 0);
        assert!(extract(&a, &invalid).is_err());
    }
    #[test]
    fn internal_exception_cannot_be_hidden_by_external_consumer() {
        let mut a = Artifact::native(&source()).expect("native");
        a.exceptions.push(crate::artifact::Exception {
            context: vec![0],
            node: 1,
            column: 0,
            value: 0.5,
        });
        let r = region(&a, 3, &[1, 2, 3]);
        assert!(
            extract(&a, &r)
                .expect_err("hidden internal exception")
                .contains("exception")
        );
        let boundary = region(&a, 3, &[2, 3]);
        let kept = extract(&a, &boundary).expect("exception stays at boundary");
        assert_eq!(kept.exceptions.len(), 1);
    }
    #[test]
    fn paid_control_preserved_or_explicitly_refused() {
        let mut p = source();
        p.nodes = vec![
            Node::Raw { slot: 0 },
            Node::Affine {
                terms: vec![(0, 0)],
                bias: None,
            },
            Node::Affine {
                terms: vec![(1, 0)],
                bias: None,
            },
            Node::Affine {
                terms: vec![(2, 0)],
                bias: None,
            },
        ];
        p.output = 3;
        let mut a = Artifact::native(&p)
            .expect("native")
            .bind("existing", &[0], 2)
            .expect("binding");
        a.places.retain(|(native, _)| *native != 1);
        a.controls.push(crate::native_control::UniformScaleBinding {
            native_source: 1,
            native_write: 2,
            write: 2,
            width: 2,
        });
        crate::native_control::validate_shape(&a).expect("declared control shape");
        let single = region(&a, 3, &[3]);
        let kept = extract(&a, &single).expect("outside control retained");
        assert_eq!(kept.controls.len(), 1);
        let whole = region(&a, 3, &[1, 2, 3]);
        assert!(
            extract(&a, &whole)
                .expect_err("internal control erased")
                .contains("control")
        );
    }
    #[test]
    fn initial_cuts_visit_all_writes_before_expansion() {
        let a = Artifact::native(&source()).expect("native");
        let inventory = propose_regions(
            &a,
            Limits {
                max_states: 5,
                ..limits()
            },
        )
        .expect("bounded roots");
        assert_eq!(
            inventory
                .regions
                .iter()
                .map(|r| r.native_write)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert!(inventory.truncated);
    }
    #[test]
    fn explicit_argument_permutation_changes_noncommutative_rule() {
        let mut p = source();
        p.nodes = vec![
            Node::Raw { slot: 0 },
            Node::Affine {
                terms: vec![(0, 0)],
                bias: None,
            },
            Node::Hadamard { left: 1, right: 1 },
            Node::Hadamard { left: 1, right: 2 },
        ];
        p.output = 3;
        let mut a = Artifact::native(&p).expect("native");
        a.program.rules.push(Rule {
            name: "choose-first".into(),
            inputs: vec![Interface::native(2).expect("input"); 2],
            nodes: vec![Node::Param { index: 0 }, Node::Param { index: 1 }],
            output: 0,
        });
        let r = region(&a, 3, &[3]);
        assert_eq!(r.native_reads, vec![1, 2]);
        let ab = reuse_rule_with_binding(&a, &r, 0, &[1, 2]).expect("AB");
        let ba = reuse_rule_with_binding(&a, &r, 0, &[2, 1]).expect("BA");
        let x = input();
        assert_ne!(
            ab.program.execute(&x, false).expect("AB output").values[ab.program.output],
            ba.program.execute(&x, false).expect("BA output").values[ba.program.output]
        );
        assert!(reuse_rule_with_binding(&a, &r, 0, &[1, 1]).is_err());
    }
    #[test]
    fn token_feature_cut_is_explicitly_skipped_but_embedding_state_is_supported() {
        use crate::operator_program::{Basis, Domain, LabelKind, Provenance};
        use crate::precision::DeclaredPrecision;
        let op = Operator::dense(
            "embedding",
            Interface::native(2).expect("hidden"),
            Interface::uniform(3, 1, LabelKind::Token, 0).expect("tokens"),
            ndarray::array![[0.5, 1., 2.], [1., -1., 0.5]],
            DeclaredPrecision::new(32).expect("precision"),
            Provenance::default(),
        )
        .expect("embedding");
        let p = OperatorProgram {
            declarations: Declarations {
                domains: vec![Domain { size: 3 }],
                slots: vec![Slot::Token { domain: 0 }],
                parameters: 0,
            },
            bases: vec![Basis::Indicator { domain: 0 }],
            operators: vec![Arc::new(op)],
            rules: vec![],
            nodes: vec![
                Node::Feature { slot: 0, basis: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Hadamard { left: 1, right: 1 },
            ],
            output: 2,
        };
        let a = Artifact::native(&p).expect("native");
        let inventory = propose_regions(&a, limits()).expect("inventory");
        assert!(
            inventory
                .skipped
                .iter()
                .any(|s| s.native_write == 1 && s.reason.contains("Feature boundary"))
        );
        let r = region(&a, 2, &[2]);
        let extracted = extract(&a, &r).expect("embedding activation boundary");
        let x = FamilyInputs {
            rows: 2,
            slots: vec![SlotValues::Tokens(vec![0, 2])],
            layout: Some(SequenceLayout {
                sequence: vec![0; 2],
                position: vec![0, 1],
            }),
        };
        let native = p.execute(&x, false).expect("native");
        assert_eq!(native.values[0].ncols(), 0);
        assert_eq!(
            native.values[2],
            extracted
                .program
                .execute(&x, false)
                .expect("extracted")
                .values[extracted.program.output]
        );
    }
    #[test]
    fn typed_reuse_and_internal_activation_loss() {
        let p = source();
        let a = Artifact::native(&p).expect("native");
        let r = region(&a, 2, &[1, 2]);
        let extracted = extract(&a, &r).expect("extract");
        assert!(extracted.place(1).is_some());
        let mut candidate = a.clone();
        candidate.program.rules.push(Rule {
            name: "square".into(),
            inputs: vec![Interface::native(2).expect("interface")],
            nodes: vec![
                Node::Param { index: 0 },
                Node::Hadamard { left: 0, right: 0 },
            ],
            output: 1,
        });
        let single = region(&candidate, 2, &[2]);
        let reused = reuse_rule(&candidate, &single, 0).expect("typed reuse");
        let x = input();
        assert_eq!(
            p.execute(&x, false).expect("native").values[p.output],
            reused.program.execute(&x, false).expect("reuse").values[reused.program.output]
        );
        assert!(reuse_rule(&candidate, &single, 10).is_err());
    }
}
