//! Bounded candidate-to-candidate search over typed executable artifact DAGs.
//! Extraction supplies structural scaffolding, not scientific discovery. Every
//! archived child is the fitted, measured, decoded object returned by the caller.
//! Pareto fidelity/description-cost objectives are research objectives, not an
//! interpretability score or a certificate of recovered computational organization.
use crate::{
    artifact::Artifact,
    canonical_artifact::CanonicalArtifactCache,
    composed_rule_search::{Expr, Grammar},
    operator_program::{Node, OperatorBody, OperatorProgram},
    program_expression_search, program_joint_regions, program_learned_dag,
    program_regions::{self, Inventory, Limits, Region},
};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub value: f64,
}
/// All axes are explicitly named, nonnegative losses/errors to minimize. Names
/// and order must match the declared constraint vector throughout the search.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Evaluation {
    pub fidelity: Vec<Metric>,
    pub local_errors: Vec<Metric>,
    pub intervention_errors: Vec<Metric>,
    pub description_bits: f64,
}
#[derive(Clone, Debug)]
pub struct EvaluatedArtifact {
    pub artifact: Artifact,
    pub evaluation: Evaluation,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Constraints {
    pub max_fidelity: Vec<Metric>,
    pub max_local_errors: Vec<Metric>,
    pub max_intervention_errors: Vec<Metric>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub region_limits: Limits,
    pub max_depth: usize,
    pub max_callback_calls: usize,
    pub max_move_attempts: usize,
    pub beam_width: usize,
    pub max_frontier: usize,
    /// Maximum complete argument permutations attempted for each region/rule.
    pub max_argument_bindings: usize,
    /// Ordered source/target pairs inspected per parent before filtering by type.
    pub max_compound_pairs: usize,
    /// Explicit intervention boundaries required by the caller, never site selection.
    pub preserve_native_places: Vec<usize>,
    /// Optional original-state observables used to seed joint nonlinear windows.
    /// None uses native syntax defaults; an empty list retains fan-out-only search.
    /// These do not force every observable to survive every accepted abstraction.
    #[serde(default)]
    pub joint_observation_places: Option<Vec<usize>>,
    /// Off preserves the original scheduling. When enabled, its work and callback
    /// budgets are reserved within the global totals rather than added to them.
    #[serde(default)]
    pub expression_search: Option<ExpressionSettings>,
    #[serde(default)]
    pub shared_dag_search: Option<SharedDAGSettings>,
    #[serde(default)]
    pub learned_dag_search: Option<LearnedDAGSettings>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearnedDAGSettings {
    /// Derive bounded connection edits from the current native computation.
    /// When present, replaces blind expression enumeration for this family.
    /// The unchanged parent is recorded as a control, never a discovery proposal.
    #[serde(default)]
    pub native_parent: Option<program_learned_dag::parent::Settings>,
    pub region_limits: program_joint_regions::Limits,
    pub grammar: program_learned_dag::Settings,
    pub max_enumerations_per_parent: usize,
    pub max_move_attempts: usize,
    pub max_callback_calls: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpressionSettings {
    /// Operation template; arguments are derived from each actual region boundary.
    /// Affine proposals are forbidden: synthesis introduces no learned adapters.
    pub grammar: Grammar,
    /// Region/argument-binding enumerations inspected per fitted parent.
    pub max_enumerations_per_parent: usize,
    pub max_move_attempts: usize,
    pub max_callback_calls: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharedDAGSettings {
    pub region_limits: program_joint_regions::Limits,
    pub grammar: Grammar,
    pub max_enumerations_per_parent: usize,
    /// All inspected tuples, including nonsharing/type-incompatible tuples.
    pub max_tuple_checks: usize,
    pub max_tuples_per_enumeration: usize,
    pub max_body_nodes: usize,
    pub max_move_attempts: usize,
    pub max_callback_calls: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Mutation {
    ReuseNativeExpression {
        region: program_joint_regions::Region,
        proposal: program_learned_dag::parent::Proposal,
    },
    #[serde(rename = "synthesize_learned_dag")]
    SynthesizeLearnedDAG {
        region: program_joint_regions::Region,
        expressions: Vec<program_learned_dag::Expr>,
        native_arguments: Vec<usize>,
    },
    ExactExtract {
        region: Region,
    },
    ReuseExistingRule {
        region: Region,
        rule_id: usize,
        native_arguments: Vec<usize>,
    },
    /// Introduce a persistent body and a distinct invocation in one measured move.
    ExtractAndReuse {
        source_region: Region,
        target_region: Region,
        native_arguments: Vec<usize>,
    },
    #[serde(rename = "synthesize_shared_dag")]
    SynthesizeSharedDAG {
        region: program_joint_regions::Region,
        expressions: Vec<Expr>,
        native_arguments: Vec<usize>,
    },
    SynthesizeExpression {
        region: Region,
        expression: Expr,
        native_arguments: Vec<usize>,
    },
}
impl Mutation {
    pub fn region(&self) -> Option<&Region> {
        match self {
            Self::ExactExtract { region }
            | Self::ReuseExistingRule { region, .. }
            | Self::SynthesizeExpression { region, .. } => Some(region),
            Self::ExtractAndReuse { target_region, .. } => Some(target_region),
            Self::SynthesizeSharedDAG { .. } | Self::SynthesizeLearnedDAG { .. } | Self::ReuseNativeExpression { .. } => None,
        }
    }
}
/// Reuse trainables use the candidate's current operator IDs after compaction
/// and decoding. Shared invocations of eligible persistent rules remain tied.
/// Learned DAG synthesis exposes only its newly compiled affine parameters,
/// using the exact compaction map; surviving native operators stay frozen.
/// Exact extraction and parameter-free expression synthesis have no trainables.
/// The caller can further freeze controlled operators; fitting and joint
/// global/local/intervention measurement are external.
pub struct FitRequest<'a> {
    pub local_fit: Option<&'a program_learned_dag::LocalFit>,
    pub initialization: Option<&'a program_learned_dag::InitializationReport>,
    pub candidate: &'a Artifact,
    pub mutation: &'a Mutation,
    pub parent: &'a EvaluatedArtifact,
    pub trainable_operator_ids: &'a [usize],
    pub attempt_id: usize,
    pub parent_id: usize,
    pub depth: usize,
}
#[derive(Clone, Debug)]
pub struct Candidate {
    pub id: usize,
    pub parent_id: Option<usize>,
    pub depth: usize,
    pub mutation: Option<Mutation>,
    pub evaluated: EvaluatedArtifact,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateRecord {
    pub id: usize,
    pub parent_id: Option<usize>,
    pub depth: usize,
    pub mutation: Option<Mutation>,
    pub evaluation: Evaluation,
    pub encoded_size_bytes: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptStatus {
    StructuralRejected,
    CallbackFailed,
    MeasurementRejected,
    ArtifactRejected,
    Duplicate,
    FittedAdmitted,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attempt {
    pub attempt_id: usize,
    pub parent_id: usize,
    pub depth: usize,
    pub mutation: Mutation,
    pub trainable_operator_ids: Vec<usize>,
    pub status: AttemptStatus,
    pub reason: Option<String>,
    pub candidate_id: Option<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Enumeration {
    pub parent_id: usize,
    pub depth: usize,
    pub inventory: Option<Inventory>,
    pub error: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompoundPair {
    pub source_native_write: usize,
    pub target_native_write: usize,
    pub source_internal_native_places: Vec<usize>,
    pub target_internal_native_places: Vec<usize>,
    pub proposed_bindings: usize,
    pub rejection: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompoundEnumeration {
    pub parent_id: usize,
    pub depth: usize,
    pub checked_pairs: Vec<CompoundPair>,
    pub truncated: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpressionEnumeration {
    pub parent_id: usize,
    pub depth: usize,
    pub region: Region,
    pub native_arguments: Vec<usize>,
    pub effective_grammar: Grammar,
    pub inventory: Option<program_expression_search::Inventory>,
    pub error: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharedDAGEnumeration {
    pub parent_id: usize,
    pub depth: usize,
    pub region: program_joint_regions::Region,
    pub native_arguments: Vec<usize>,
    pub effective_grammar: Grammar,
    pub inventory: Option<program_expression_search::SharedInventory>,
    pub error: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharedDAGSchedule {
    pub parent_id: usize,
    pub depth: usize,
    pub region: program_joint_regions::Region,
    pub enumerated_proposals: usize,
    pub duplicate_binding_proposals: usize,
    pub unique_proposals: usize,
    pub priority: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointRegionEnumeration {
    pub parent_id: usize,
    pub depth: usize,
    pub inventory: Option<program_joint_regions::Inventory>,
    pub error: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearnedDAGEnumeration {
    pub parent_id: usize,
    pub depth: usize,
    pub region: Option<program_joint_regions::Region>,
    pub inventory: Option<program_learned_dag::Inventory>,
    #[serde(default)]
    pub native_inventory: Option<program_learned_dag::parent::Inventory>,
    pub error: Option<String>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalFamily {
    Structural,
    Expression,
    SharedDAG,
    LearnedDAG,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FamilyBudgetOmission {
    pub parent_id: usize,
    pub depth: usize,
    pub family: ProposalFamily,
    pub reason: String,
    pub omitted_proposals: usize,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Counts {
    pub enumerations: usize,
    pub enumeration_failures: usize,
    pub inventories_truncated: usize,
    pub region_states_explored: usize,
    pub regions_proposed: usize,
    pub argument_binding_sets_truncated: usize,
    pub compound_pairs_checked: usize,
    pub compound_pairs_type_rejected: usize,
    pub compound_pair_sets_truncated: usize,
    pub expression_enumerations: usize,
    pub expression_enumeration_failures: usize,
    pub expression_enumeration_sets_truncated: usize,
    pub expression_binding_sets_truncated: usize,
    pub expression_inventories_truncated: usize,
    pub expression_intermediate_expressions: usize,
    pub expression_proposals: usize,
    pub expression_move_attempts: usize,
    pub expression_callback_calls: usize,
    pub shared_dag_enumerations: usize,
    pub shared_dag_enumeration_failures: usize,
    pub shared_dag_enumeration_sets_truncated: usize,
    pub shared_dag_inventories_truncated: usize,
    pub shared_dag_tuple_checks: usize,
    pub shared_dag_proposals: usize,
    pub shared_dag_duplicate_binding_proposals: usize,
    pub shared_dag_unique_proposals: usize,
    pub shared_dag_move_attempts: usize,
    pub shared_dag_callback_calls: usize,
    #[serde(default)]
    pub learned_dag_move_attempts: usize,
    #[serde(default)]
    pub learned_dag_callback_calls: usize,
    #[serde(default)]
    pub learned_dag_enumerations: usize,
    #[serde(default)]
    pub learned_dag_proposals: usize,
    #[serde(default)]
    pub learned_dag_parameterless_proposals: usize,
    #[serde(default)]
    pub learned_dag_enumerations_truncated: usize,
    pub joint_region_states_explored: usize,
    pub joint_regions_proposed: usize,
    pub joint_region_inventories_truncated: usize,
    pub shared_dag_binding_sets_truncated: usize,
    pub move_attempts: usize,
    pub structural_rejections: usize,
    pub callback_calls: usize,
    pub callback_failures: usize,
    pub measurement_rejections: usize,
    pub artifact_rejections: usize,
    pub duplicates: usize,
    pub fitted_children: usize,
    pub beam_pruned: usize,
    pub frontier_truncations: usize,
    pub completed_depth: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub settings: Settings,
    pub constraints: Constraints,
    pub counts: Counts,
    pub attempts: Vec<Attempt>,
    pub enumerations: Vec<Enumeration>,
    pub compound_enumerations: Vec<CompoundEnumeration>,
    pub expression_enumerations: Vec<ExpressionEnumeration>,
    pub shared_dag_enumerations: Vec<SharedDAGEnumeration>,
    pub shared_dag_schedules: Vec<SharedDAGSchedule>,
    pub joint_region_enumerations: Vec<JointRegionEnumeration>,
    #[serde(default)]
    pub learned_dag_enumerations: Vec<LearnedDAGEnumeration>,
    pub family_budget_omissions: Vec<FamilyBudgetOmission>,
    pub admitted_candidates: Vec<CandidateRecord>,
    pub stop_reason: String,
}
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub candidates: Vec<Candidate>,
    pub frontier: Vec<usize>,
    pub beam: Vec<usize>,
    pub report: Report,
}
fn axes(e: &Evaluation) -> Vec<f64> {
    e.fidelity
        .iter()
        .chain(&e.local_errors)
        .chain(&e.intervention_errors)
        .map(|m| m.value)
        .chain([e.description_bits])
        .collect()
}
fn validate_metrics(measured: &[Metric], limits: &[Metric], family: &str) -> Result<(), String> {
    if measured.len() != limits.len() {
        return Err(format!("{family} measurement dimensions differ"));
    }
    for (metric, limit) in measured.iter().zip(limits) {
        if metric.name != limit.name {
            return Err(format!("{family} measurement names/order differ"));
        }
        if !metric.value.is_finite() || metric.value < 0. {
            return Err(format!("{family} {} is nonfinite or negative", metric.name));
        }
        if metric.value > limit.value {
            return Err(format!(
                "{family} {}={} exceeds {}",
                metric.name, metric.value, limit.value
            ));
        }
    }
    Ok(())
}
/// Check a measured stage against the same named TRAIN limits used for admission.
/// Callers must supply the validated constraints declared for their search.
pub fn validate_evaluation(e: &Evaluation, constraints: &Constraints) -> Result<(), String> {
    validate_metrics(&e.fidelity, &constraints.max_fidelity, "fidelity")?;
    validate_metrics(&e.local_errors, &constraints.max_local_errors, "local")?;
    validate_metrics(
        &e.intervention_errors,
        &constraints.max_intervention_errors,
        "intervention",
    )?;
    if !e.description_bits.is_finite() || e.description_bits < 0. {
        return Err("nonfinite/negative description cost".into());
    }
    Ok(())
}
fn control_signature(a: &Artifact) -> Vec<(usize, usize, usize)> {
    a.controls
        .iter()
        .map(|c| (c.native_source, c.native_write, c.width))
        .collect()
}
fn validate_artifact(
    a: &Artifact,
    native: &OperatorProgram,
    parent: Option<&Artifact>,
    settings: &Settings,
) -> Result<(), String> {
    a.validate_coverage(native)?;
    for place in &settings.preserve_native_places {
        if a.place(*place).is_none() {
            return Err(format!(
                "required native intervention boundary {place} erased"
            ));
        }
    }
    if let Some(parent) = parent {
        if control_signature(a) != control_signature(parent) {
            return Err("declared native controls dropped or changed".into());
        }
    }
    Ok(())
}
fn canonical(a: &Artifact) -> Result<(Artifact, Vec<u8>), String> {
    let bytes = a.to_bytes()?;
    let decoded = Artifact::from_bytes(&bytes, &a.program.declarations)?;
    if decoded.to_bytes()? != bytes {
        return Err("artifact codec is not stable under decoding".into());
    }
    if decoded.program.nodes != a.program.nodes
        || decoded.program.output != a.program.output
        || decoded.program.operators.len() != a.program.operators.len()
        || decoded
            .program
            .operators
            .iter()
            .zip(&a.program.operators)
            .any(|(x, y)| x.rows != y.rows || x.cols != y.cols)
        || decoded.program.rules.len() != a.program.rules.len()
        || decoded
            .program
            .rules
            .iter()
            .zip(&a.program.rules)
            .any(|(x, y)| x.nodes != y.nodes || x.inputs != y.inputs || x.output != y.output)
        || decoded.places != a.places
    {
        return Err("artifact codec changed executable reference identities".into());
    }
    Ok((decoded, bytes))
}
fn canonical_with_codec(a: &Artifact, codec: Option<&CanonicalArtifactCache>) -> Result<(Artifact, Vec<u8>), String> {
    match codec {
        Some(codec) => codec.canonical_exact(a).map(|c| (c.decoded, c.bytes)),
        None => canonical(a),
    }
}
fn serialized_measured(e: &EvaluatedArtifact, codec: Option<&CanonicalArtifactCache>) -> Result<Vec<u8>, String> {
    let (decoded, bytes) = canonical_with_codec(&e.artifact, codec)?;
    if decoded != e.artifact {
        return Err(
            "measurements must describe the actual decoded artifact returned by the callback"
                .into(),
        );
    }
    Ok(bytes)
}
/// Parameter fitting cannot silently introduce an unlogged structural mutation.
/// Numerical updates and precision declarations are measured by the callback;
/// the executable graph, typed interfaces, and native/control coverage stay fixed.
fn validate_fit_structure(
    proposed: &Artifact,
    fitted: &Artifact,
    trainable_ids: &[usize],
) -> Result<(), String> {
    let p = &proposed.program;
    let f = &fitted.program;
    if p.declarations != f.declarations
        || p.bases != f.bases
        || p.nodes != f.nodes
        || p.rules != f.rules
        || p.output != f.output
        || p.operators.len() != f.operators.len()
        || proposed.native_nodes != fitted.native_nodes
        || proposed.blocks != fitted.blocks
        || proposed.places != fitted.places
        || proposed.exceptions != fitted.exceptions
        || proposed.derived != fitted.derived
        || proposed.controls != fitted.controls
    {
        return Err("fitting changed the proposed graph or native/control provenance".into());
    }
    for (id, (p, f)) in p.operators.iter().zip(&f.operators).enumerate() {
        if !trainable_ids.contains(&id) {
            if p != f {
                return Err(format!("fitting changed frozen operator {id}"));
            }
            continue;
        }
        if p.rows != f.rows || p.cols != f.cols {
            return Err("fitting changed an operator's typed interface".into());
        }
        let same_family = match (&p.body, &f.body) {
            (OperatorBody::Dense { present: a, .. }, OperatorBody::Dense { present: b, .. }) => {
                a == b
            }
            _ => false,
        };
        if !same_family {
            return Err("fitting changed an eligible Dense family or block support".into());
        }
    }
    Ok(())
}
fn dominates(a: &Evaluation, b: &Evaluation) -> bool {
    let a = axes(a);
    let b = axes(b);
    a.iter().zip(&b).all(|(a, b)| a <= b) && a.iter().zip(&b).any(|(a, b)| a < b)
}
fn nondominated(ids: &[usize], candidates: &BTreeMap<usize, Candidate>) -> Vec<usize> {
    ids.iter()
        .copied()
        .filter(|id| {
            !ids.iter().any(|other| {
                other != id
                    && dominates(
                        &candidates[other].evaluated.evaluation,
                        &candidates[id].evaluated.evaluation,
                    )
            })
        })
        .collect()
}
fn bound(
    ids: &[usize],
    candidates: &BTreeMap<usize, Candidate>,
    maximum: usize,
) -> (Vec<usize>, bool) {
    let mut keep = nondominated(ids, candidates);
    keep.sort_by(|a, b| compare(&candidates[a], &candidates[b]));
    let truncated = keep.len() > maximum;
    keep.truncate(maximum);
    (keep, truncated)
}
fn compare(a: &Candidate, b: &Candidate) -> Ordering {
    for (a, b) in axes(&a.evaluated.evaluation)
        .iter()
        .zip(axes(&b.evaluated.evaluation))
    {
        let order = a.total_cmp(&b);
        if order != Ordering::Equal {
            return order;
        }
    }
    a.id.cmp(&b.id)
}
fn rule_closure(program: &OperatorProgram, rule: usize) -> Result<BTreeSet<usize>, String> {
    let mut pending = vec![rule];
    let mut visited = BTreeSet::new();
    while let Some(rule) = pending.pop() {
        if !visited.insert(rule) {
            continue;
        }
        for node in &program
            .rules
            .get(rule)
            .ok_or("callee missing after compaction")?
            .nodes
        {
            if let Node::Call { rule, .. } = node {
                pending.push(*rule);
            }
        }
    }
    Ok(visited)
}
fn trainables(a: &Artifact, mutation: &Mutation) -> Result<Vec<usize>, String> {
    if matches!(
        mutation,
        Mutation::ExactExtract { .. }
            | Mutation::SynthesizeExpression { .. }
            | Mutation::SynthesizeSharedDAG { .. }
    ) {
        return Ok(vec![]);
    }
    let write = a
        .place(
            mutation
                .region()
                .ok_or("single-output rule mutation expected")?
                .native_write,
        )
        .ok_or("replacement write lost native provenance")?;
    let callee = match a.program.nodes.get(write) {
        Some(Node::Call { rule, .. }) => *rule,
        _ => return Err("replacement write is not an executable library call".into()),
    };
    let eligible = rule_closure(&a.program, callee)?;
    let mut inside = BTreeSet::new();
    let mut outside = BTreeSet::new();
    for rule in &eligible {
        for node in &a.program.rules[*rule].nodes {
            inside.extend(node.operators());
        }
    }
    for node in &a.program.nodes {
        outside.extend(node.operators());
    }
    // Other calls of eligible persistent rules are intentional ties. Direct
    // operator occurrences in other reachable rule bodies are native/background uses.
    let mut reachable = BTreeSet::new();
    for node in &a.program.nodes {
        if let Node::Call { rule, .. } = node {
            reachable.extend(rule_closure(&a.program, *rule)?);
        }
    }
    for rule in reachable.difference(&eligible) {
        for node in &a.program.rules[*rule].nodes {
            outside.extend(node.operators());
        }
    }
    let derived = a
        .derived
        .iter()
        .map(|d| d.operator)
        .collect::<BTreeSet<_>>();
    Ok(inside
        .difference(&outside)
        .copied()
        .filter(|id| {
            !derived.contains(id)
                && matches!(
                    a.program.operators.get(*id).map(|o| &o.body),
                    Some(OperatorBody::Dense { .. })
                )
        })
        .collect())
}
/// Bounded permutations of the existing native boundary only; no new inputs.
fn bindings(region: &Region, maximum: usize) -> (Vec<Vec<usize>>, bool) {
    boundary_bindings(&region.native_reads, maximum)
}
fn boundary_bindings(native_reads: &[usize], maximum: usize) -> (Vec<Vec<usize>>, bool) {
    fn visit(
        prefix: &mut Vec<usize>,
        remaining: &mut Vec<usize>,
        maximum: usize,
        out: &mut Vec<Vec<usize>>,
        truncated: &mut bool,
    ) {
        if remaining.is_empty() {
            if out.len() == maximum {
                *truncated = true;
            } else {
                out.push(prefix.clone());
            }
            return;
        }
        for i in 0..remaining.len() {
            let input = remaining.remove(i);
            prefix.push(input);
            visit(prefix, remaining, maximum, out, truncated);
            prefix.pop();
            remaining.insert(i, input);
            if *truncated {
                break;
            }
        }
    }
    let mut result = vec![];
    let mut truncated = false;
    let mut remaining = native_reads.to_vec();
    remaining.sort_unstable();
    visit(
        &mut vec![],
        &mut remaining,
        maximum,
        &mut result,
        &mut truncated,
    );
    (result, truncated)
}

fn compound_moves(
    artifact: &Artifact,
    regions: &[Region],
    settings: &Settings,
    counts: &mut Counts,
) -> Result<(Vec<Mutation>, Vec<CompoundPair>, bool), String> {
    let interfaces = artifact.program.interfaces().map_err(|e| e.to_string())?;
    let mut moves = vec![];
    let mut checked = vec![];
    // Round-robin source coverage instead of exhausting every target of the first source.
    for distance in 1..regions.len() {
        for source_id in 0..regions.len() {
            if checked.len() == settings.max_compound_pairs {
                counts.compound_pair_sets_truncated += 1;
                return Ok((moves, checked, true));
            }
            let source = &regions[source_id];
            let target = &regions[(source_id + distance) % regions.len()];
            counts.compound_pairs_checked += 1;
            let mut record = CompoundPair {
                source_native_write: source.native_write,
                target_native_write: target.native_write,
                source_internal_native_places: source.internal_native_places.clone(),
                target_internal_native_places: target.internal_native_places.clone(),
                proposed_bindings: 0,
                rejection: None,
            };
            if source.native_reads.len() != target.native_reads.len()
                || interfaces[source.current_write] != interfaces[target.current_write]
            {
                counts.compound_pairs_type_rejected += 1;
                record.rejection = Some("input arity or output interface differs".into());
            } else {
                let (ordered, truncated) = bindings(target, settings.max_argument_bindings);
                counts.argument_binding_sets_truncated += usize::from(truncated);
                for native_arguments in ordered {
                    let compatible = source.current_reads.iter().zip(&native_arguments).all(
                        |(source, native)| {
                            artifact
                                .place(*native)
                                .is_some_and(|target| interfaces[*source] == interfaces[target])
                        },
                    );
                    if compatible {
                        record.proposed_bindings += 1;
                        moves.push(Mutation::ExtractAndReuse {
                            source_region: source.clone(),
                            target_region: target.clone(),
                            native_arguments,
                        });
                    }
                }
                if record.proposed_bindings == 0 {
                    counts.compound_pairs_type_rejected += 1;
                    record.rejection = Some("no compatible binding within argument budget".into());
                }
            }
            checked.push(record);
        }
    }
    Ok((moves, checked, false))
}
fn extract_and_reuse(
    parent: &Artifact,
    source: &Region,
    target: &Region,
    arguments: &[usize],
) -> Result<Artifact, String> {
    if source
        .current_internal_nodes
        .iter()
        .any(|n| target.current_internal_nodes.contains(n))
    {
        return Err("compound source and target internal nodes overlap".into());
    }
    if target.internal_native_places.len() != target.current_internal_nodes.len() {
        return Err(
            "compound target contains intermediate nodes without native boundary labels".into(),
        );
    }
    let extracted = program_regions::extract(parent, source)?;
    let source_write = extracted
        .place(source.native_write)
        .ok_or("compound source write lost")?;
    let rule_id = match extracted.program.nodes.get(source_write) {
        Some(Node::Call { rule, .. }) => *rule,
        _ => return Err("compound extracted source is not a persistent call".into()),
    };
    let remap = |native| {
        extracted.place(native).ok_or_else(|| {
            format!("compound target native boundary {native} lost after extraction")
        })
    };
    let mut current_reads = target
        .native_reads
        .iter()
        .map(|n| remap(*n).map(|c| (*n, c)))
        .collect::<Result<Vec<_>, _>>()?;
    current_reads.sort_by_key(|(_, current)| *current);
    let mut current_internal_nodes = target
        .internal_native_places
        .iter()
        .map(|n| remap(*n))
        .collect::<Result<Vec<_>, _>>()?;
    current_internal_nodes.sort_unstable();
    let retargeted = Region {
        native_reads: current_reads.iter().map(|(native, _)| *native).collect(),
        native_write: target.native_write,
        current_reads: current_reads.iter().map(|(_, current)| *current).collect(),
        current_write: remap(target.native_write)?,
        current_internal_nodes,
        internal_native_places: target.internal_native_places.clone(),
        source_program_nodes: extracted.program.nodes.len(),
    };
    program_regions::reuse_rule_with_binding(&extracted, &retargeted, rule_id, arguments)
}
fn expression_moves(
    artifact: &Artifact,
    regions: &[(Region, Vec<Vec<usize>>)],
    settings: &Settings,
    parent_id: usize,
    depth: usize,
    report: &mut Report,
) -> Vec<Mutation> {
    let Some(settings) = &settings.expression_search else {
        return vec![];
    };
    let mut proposal_lists = std::collections::VecDeque::new();
    let mut checked = 0;
    // A first binding at every proposed region precedes second bindings.
    let maximum_bindings = regions
        .iter()
        .map(|(_, args)| args.len())
        .max()
        .unwrap_or(0);
    'enumerations: for binding_id in 0..maximum_bindings {
        for (region, arguments) in regions {
            let Some(arguments) = arguments.get(binding_id) else {
                continue;
            };
            if checked == settings.max_enumerations_per_parent {
                report.counts.expression_enumeration_sets_truncated += 1;
                break 'enumerations;
            }
            checked += 1;
            report.counts.expression_enumerations += 1;
            let mut grammar = settings.grammar.clone();
            grammar.arguments = region.native_reads.len();
            let mut enumeration = ExpressionEnumeration {
                parent_id,
                depth,
                region: region.clone(),
                native_arguments: arguments.clone(),
                effective_grammar: grammar.clone(),
                inventory: None,
                error: None,
            };
            match program_expression_search::enumerate(artifact, region, &grammar, arguments) {
                Ok(inventory) => {
                    report.counts.expression_inventories_truncated +=
                        usize::from(inventory.truncated);
                    report.counts.expression_intermediate_expressions +=
                        inventory.intermediate_expressions;
                    report.counts.expression_proposals += inventory.expressions.len();
                    let proposals = inventory
                        .expressions
                        .iter()
                        .map(|expression| Mutation::SynthesizeExpression {
                            region: region.clone(),
                            expression: expression.clone(),
                            native_arguments: arguments.clone(),
                        })
                        .collect::<Vec<_>>();
                    proposal_lists.push_back(proposals.into_iter());
                    enumeration.inventory = Some(inventory);
                }
                Err(reason) => {
                    report.counts.expression_enumeration_failures += 1;
                    enumeration.error = Some(reason);
                }
            }
            report.expression_enumerations.push(enumeration);
        }
    }
    // Each region/binding's first expression precedes its second. Enumeration's
    // operation-count ordering is retained within every list, including when the
    // enumeration budget cuts off later regions or argument bindings.
    let mut moves = vec![];
    while let Some(mut proposals) = proposal_lists.pop_front() {
        if let Some(proposal) = proposals.next() {
            moves.push(proposal);
            proposal_lists.push_back(proposals);
        }
    }
    moves
}
fn joint_regions(
    artifact: &Artifact,
    limits: program_joint_regions::Limits,
    settings: &Settings,
) -> Result<program_joint_regions::Inventory, String> {
    if let Some(observations) = &settings.joint_observation_places {
        let remaining = observations
            .iter()
            .copied()
            .filter(|n| artifact.place(*n).is_some())
            .collect::<Vec<_>>();
        program_joint_regions::propose_regions_with_observations(artifact, limits, &remaining)
    } else {
        program_joint_regions::propose_regions(artifact, limits)
    }
}
fn shared_dag_moves(
    artifact: &Artifact,
    settings: &Settings,
    parent_id: usize,
    depth: usize,
    report: &mut Report,
) -> Vec<Mutation> {
    let Some(shared) = &settings.shared_dag_search else {
        return vec![];
    };
    let inventory = match joint_regions(artifact, shared.region_limits.clone(), settings) {
        Ok(inventory) => inventory,
        Err(error) => {
            report.counts.shared_dag_enumeration_failures += 1;
            report
                .joint_region_enumerations
                .push(JointRegionEnumeration {
                    parent_id,
                    depth,
                    inventory: None,
                    error: Some(error),
                });
            return vec![];
        }
    };
    report.counts.joint_region_states_explored += inventory.explored_states;
    report.counts.joint_regions_proposed += inventory.regions.len();
    report.counts.joint_region_inventories_truncated += usize::from(inventory.truncated);
    report
        .joint_region_enumerations
        .push(JointRegionEnumeration {
            parent_id,
            depth,
            inventory: Some(inventory.clone()),
            error: None,
        });
    let regions = inventory
        .regions
        .into_iter()
        .map(|region| {
            let (bindings, truncated) =
                boundary_bindings(&region.native_reads, settings.max_argument_bindings);
            report.counts.shared_dag_binding_sets_truncated += usize::from(truncated);
            (region, bindings)
        })
        .collect::<Vec<_>>();
    struct RegionProposals {
        region: program_joint_regions::Region,
        tuples: BTreeMap<Vec<Expr>, usize>,
        enumerated: usize,
        duplicates: usize,
    }
    let mut grouped: Vec<RegionProposals> = vec![];
    let mut region_ids = BTreeMap::new();
    let maximum_bindings = regions
        .iter()
        .map(|(_, args)| args.len())
        .max()
        .unwrap_or(0);
    let mut checked = 0;
    'enumerations: for binding_id in 0..maximum_bindings {
        for (region, bindings) in &regions {
            let Some(arguments) = bindings.get(binding_id) else {
                continue;
            };
            if checked == shared.max_enumerations_per_parent {
                report.counts.shared_dag_enumeration_sets_truncated += 1;
                break 'enumerations;
            }
            checked += 1;
            report.counts.shared_dag_enumerations += 1;
            let mut grammar = shared.grammar.clone();
            grammar.arguments = region.native_reads.len();
            let mut enumeration = SharedDAGEnumeration {
                parent_id,
                depth,
                region: region.clone(),
                native_arguments: arguments.clone(),
                effective_grammar: grammar.clone(),
                inventory: None,
                error: None,
            };
            match program_expression_search::enumerate_shared(
                artifact,
                region,
                &grammar,
                arguments,
                shared.max_tuple_checks,
                shared.max_tuples_per_enumeration,
                shared.max_body_nodes,
            ) {
                Ok(inventory) => {
                    report.counts.shared_dag_inventories_truncated +=
                        usize::from(inventory.truncated);
                    report.counts.shared_dag_tuple_checks += inventory.checked_tuples;
                    report.counts.shared_dag_proposals += inventory.expressions.len();
                    // Same native cut may be reached from different producer seeds.
                    // Group by the actual boundary/cut, excluding the seed label.
                    let key = (
                        region.native_reads.clone(),
                        region.native_writes.clone(),
                        region.current_internal_nodes.clone(),
                    );
                    let group_id = *region_ids.entry(key).or_insert_with(|| {
                        let id = grouped.len();
                        grouped.push(RegionProposals {
                            region: region.clone(),
                            tuples: BTreeMap::new(),
                            enumerated: 0,
                            duplicates: 0,
                        });
                        id
                    });
                    let group = &mut grouped[group_id];
                    for (expressions, node_count) in inventory
                        .expressions
                        .iter()
                        .zip(&inventory.compiled_node_counts)
                    {
                        group.enumerated += 1;
                        match program_expression_search::canonical_shared_expressions(
                            region,
                            expressions,
                            arguments,
                        ) {
                            Ok(canonical) => {
                                if group.tuples.contains_key(&canonical) {
                                    group.duplicates += 1;
                                    report.counts.shared_dag_duplicate_binding_proposals += 1;
                                } else {
                                    group.tuples.insert(canonical, *node_count);
                                    report.counts.shared_dag_unique_proposals += 1;
                                }
                            }
                            Err(error) => {
                                report.counts.shared_dag_enumeration_failures += 1;
                                enumeration.error = Some(error);
                            }
                        }
                    }
                    enumeration.inventory = Some(inventory);
                }
                Err(error) => {
                    report.counts.shared_dag_enumeration_failures += 1;
                    enumeration.error = Some(error);
                }
            }
            report.shared_dag_enumerations.push(enumeration);
        }
    }
    let mut lists = std::collections::VecDeque::new();
    for group in grouped {
        let mut ranked = group
            .tuples
            .into_iter()
            .map(|(expressions, nodes)| (nodes, expressions))
            .collect::<Vec<_>>();
        ranked.sort();
        report.shared_dag_schedules.push(SharedDAGSchedule {parent_id,depth,region:group.region.clone(),enumerated_proposals:group.enumerated,
            duplicate_binding_proposals:group.duplicates,unique_proposals:ranked.len(),
            priority:"Exact Arg remapping and commutative operand canonicalization across inspected bindings; unique native-coordinate tuples by compiled DAG node count then Expr tuple, round robin across native regions; no outcome reuse".into()});
        let moves = ranked
            .into_iter()
            .map(|(_, expressions)| Mutation::SynthesizeSharedDAG {
                native_arguments: group.region.native_reads.clone(),
                region: group.region.clone(),
                expressions,
            })
            .collect::<Vec<_>>();
        lists.push_back(moves.into_iter());
    }
    let mut moves = vec![];
    while let Some(mut list) = lists.pop_front() {
        if let Some(proposal) = list.next() {
            moves.push(proposal);
            lists.push_back(list);
        }
    }
    moves
}

/// Preserve cost order within structural strata, but rotate strata before
/// spending another callback on the same kind of equation. Inventory cost order
/// alone can bury full-width nonlinear equations behind many affine proposals.
/// This is a syntactic scheduling policy, not evidence of native correspondence.
fn learned_proposal_order(
    proposals: &[program_learned_dag::Proposal],
    input_widths: &[usize],
    output_widths: &[usize],
) -> Vec<usize> {
    use program_learned_dag::{Expr as LearnedExpr, TypeRef};
    use std::cmp::Reverse;
    // (operations, has affine, learned nonlinear, affine/nonlinear/affine chain,
    // widest affine output, widest affine below a nonlinearity). Interfaces,
    // not latent spelling, determine width.
    fn shape(e: &LearnedExpr, inputs: &[usize], outputs: &[usize]) -> (usize, bool, bool, bool, usize, usize) {
        match e {
            LearnedExpr::Argument(_) => (0, false, false, false, 0, 0),
            LearnedExpr::Unary(_, x) => {
                let (n, affine, nonlinear, chain, width, nonlinear_width) = shape(x, inputs, outputs);
                (n + 1, affine, nonlinear || affine, chain, width, nonlinear_width.max(width))
            }
            LearnedExpr::Affine { output, input, .. } => {
                let (n, _, nonlinear, chain, width, nonlinear_width) = shape(input, inputs, outputs);
                let own_width = match output {
                    TypeRef::Input(i) => inputs[*i],
                    TypeRef::Exit(i) => outputs[*i],
                    TypeRef::Latent { width } => *width,
                };
                (n + 1, true, nonlinear, chain || nonlinear, width.max(own_width), nonlinear_width)
            }
            LearnedExpr::Binary(op, a, b) => {
                let a = shape(a, inputs, outputs);
                let b = shape(b, inputs, outputs);
                let product = *op == crate::composed_rule_search::Binary::Multiply && (a.1 || b.1);
                let nonlinear_width = a.5.max(b.5).max(if product { a.4.max(b.4) } else { 0 });
                (a.0 + b.0 + 1, a.1 || b.1, a.2 || b.2 || product, a.3 || b.3, a.4.max(b.4), nonlinear_width)
            }
        }
    }
    fn laws(e: &LearnedExpr, unary: &mut BTreeSet<crate::composed_rule_search::Unary>, binary: &mut BTreeSet<crate::composed_rule_search::Binary>) {
        match e {
            LearnedExpr::Argument(_) => {},
            LearnedExpr::Unary(op, x) => { unary.insert(*op); laws(x, unary, binary); },
            LearnedExpr::Affine { input, .. } => laws(input, unary, binary),
            LearnedExpr::Binary(op, a, b) => { binary.insert(*op); laws(a, unary, binary); laws(b, unary, binary); },
        }
    }
    let mut strata = BTreeMap::new();
    for (index, proposal) in proposals.iter().enumerate() {
        if proposal.trainable_operator_count == 0 {
            continue;
        }
        let shapes = proposal.expressions.iter()
            .map(|e| shape(e, input_widths, output_widths)).collect::<Vec<_>>();
        let mut nonlinear_laws = BTreeSet::new();
        let mut binary_laws = BTreeSet::new();
        for e in &proposal.expressions { laws(e, &mut nonlinear_laws, &mut binary_laws); }
        let key = (
            Reverse(shapes.iter().any(|s| s.3)),
            Reverse(shapes.iter().any(|s| s.2)),
            Reverse(shapes.iter().all(|s| s.1)),
            Reverse(shapes.iter().map(|s| s.0).max().unwrap_or(0)),
            Reverse(shapes.iter().map(|s| s.5).max().unwrap_or(0)),
            Reverse(shapes.iter().map(|s| s.4).max().unwrap_or(0)),
            // Rotate single-law equations before mixed-law variants; lexical
            // set order otherwise puts {Silu, GeluTanh} ahead of {GeluTanh}.
            nonlinear_laws.len() + binary_laws.len(),
            nonlinear_laws,
            binary_laws,
        );
        strata.entry(key).or_insert_with(std::collections::VecDeque::new).push_back(index);
    }
    let mut queues = strata.into_values().collect::<std::collections::VecDeque<_>>();
    let mut order = Vec::new();
    while let Some(mut queue) = queues.pop_front() {
        if let Some(index) = queue.pop_front() {
            order.push(index);
            if !queue.is_empty() {
                queues.push_back(queue);
            }
        }
    }
    order
}

fn learned_dag_moves(
    artifact: &Artifact,
    settings: &Settings,
    parent_id: usize,
    depth: usize,
    report: &mut Report,
) -> Vec<Mutation> {
    let Some(config) = &settings.learned_dag_search else {
        return vec![];
    };
    let regions = match joint_regions(artifact, config.region_limits.clone(), settings) {
        Ok(inventory) => {
            report.counts.learned_dag_enumerations_truncated += usize::from(inventory.truncated);
            inventory.regions
        }
        Err(error) => {
            report.learned_dag_enumerations.push(LearnedDAGEnumeration {
                parent_id,
                depth,
                region: None,
                inventory: None,
                native_inventory: None,
                error: Some(error),
            });
            return vec![];
        }
    };
    let mut seen = BTreeSet::new();
    let mut lists = std::collections::VecDeque::new();
    let mut checked = 0;
    for region in regions {
        let key = (
            region.native_reads.clone(),
            region.native_writes.clone(),
            region.current_internal_nodes.clone(),
        );
        if !seen.insert(key) {
            continue;
        }
        if checked == config.max_enumerations_per_parent {
            report.counts.learned_dag_enumerations_truncated += 1;
            break;
        }
        checked += 1;
        report.counts.learned_dag_enumerations += 1;
        let mut record = LearnedDAGEnumeration {
            parent_id,
            depth,
            region: Some(region.clone()),
            inventory: None,
            native_inventory: None,
            error: None,
        };
        if let Some(parent_settings) = &config.native_parent {
            match program_learned_dag::parent::enumerate(artifact, &region, parent_settings) {
                Ok(inventory) => {
                    report.counts.learned_dag_proposals += inventory.proposals.len();
                    report.counts.learned_dag_enumerations_truncated += usize::from(inventory.truncated);
                    lists.push_back(inventory.proposals.iter().map(|proposal| {
                        Mutation::ReuseNativeExpression {
                            region: region.clone(), proposal: proposal.clone(),
                        }
                    }).collect::<Vec<_>>().into_iter());
                    record.native_inventory = Some(inventory);
                }
                Err(error) => record.error = Some(error),
            }
            report.learned_dag_enumerations.push(record);
            continue;
        }
        match program_learned_dag::enumerate(
            artifact,
            &region,
            &config.grammar,
            &region.native_reads,
        ) {
            Ok(inventory) => {
                report.counts.learned_dag_proposals += inventory.proposals.len();
                report.counts.learned_dag_parameterless_proposals += inventory
                    .proposals
                    .iter()
                    .filter(|p| p.trainable_operator_count == 0)
                    .count();
                report.counts.learned_dag_enumerations_truncated +=
                    usize::from(inventory.truncated);
                // Parameter-free expressions have their own proposal family. Do
                // not spend this family's reserved fitting budget on them.
                let interfaces = artifact.program.interfaces().expect("enumerated typed artifact");
                let input_widths = region.current_reads.iter().map(|n| interfaces[*n].width()).collect::<Vec<_>>();
                let output_widths = region.current_writes.iter().map(|n| interfaces[*n].width()).collect::<Vec<_>>();
                lists.push_back(
                    learned_proposal_order(&inventory.proposals, &input_widths, &output_widths)
                        .into_iter()
                        .map(|i| &inventory.proposals[i])
                        .map(|p| Mutation::SynthesizeLearnedDAG {
                            region: region.clone(),
                            expressions: p.expressions.clone(),
                            native_arguments: region.native_reads.clone(),
                        })
                        .collect::<Vec<_>>()
                        .into_iter(),
                );
                record.inventory = Some(inventory);
            }
            Err(error) => record.error = Some(error),
        }
        report.learned_dag_enumerations.push(record);
    }
    let mut moves = vec![];
    while let Some(mut list) = lists.pop_front() {
        if let Some(proposal) = list.next() {
            moves.push(proposal);
            lists.push_back(list);
        }
    }
    moves
}

fn omit_family(
    report: &mut Report,
    parent_id: usize,
    depth: usize,
    family: ProposalFamily,
    reason: &str,
) {
    if let Some(omission) = report.family_budget_omissions.iter_mut().find(|o| {
        o.parent_id == parent_id && o.depth == depth && o.family == family && o.reason == reason
    }) {
        omission.omitted_proposals += 1;
    } else {
        report.family_budget_omissions.push(FamilyBudgetOmission {
            parent_id,
            depth,
            family,
            reason: reason.into(),
            omitted_proposals: 1,
        });
    }
}

/// Expand each fitted parent, including its persistent calls/rules. The depth
/// beam is Pareto among *new children*, not against their parents: exact structural
/// scaffolding can survive a temporary cost increase before enabling reuse.
/// If that frontier exceeds beam_width, bound it lexicographically in declared
/// fidelity/local/intervention axis order, then description bits, then candidate ID.
/// A separate capped visited frontier describes measured outcomes, using the
/// same ranking when max_frontier is exceeded. This is finite budget search,
/// with all unresolved region/binding cuts explicit, not global optimization.
pub fn search<F>(
    native: &OperatorProgram,
    initial: EvaluatedArtifact,
    settings: &Settings,
    constraints: &Constraints,
    fit_and_measure: F,
) -> Result<SearchResult, String>
where
    F: FnMut(FitRequest<'_>) -> Result<EvaluatedArtifact, String>,
{
    search_with_codec(native, initial, settings, constraints, None, fit_and_measure)
}

/// Same search, scheduling, literal coefficients and standalone artifact bytes.
/// The optional bounded cache witnesses immutable native codewords only; it
/// never rounds proposal literals or caches measurements/acceptance decisions.
pub fn search_with_codec<F>(
    native: &OperatorProgram,
    initial: EvaluatedArtifact,
    settings: &Settings,
    constraints: &Constraints,
    codec: Option<&CanonicalArtifactCache>,
    mut fit_and_measure: F,
) -> Result<SearchResult, String>
where
    F: FnMut(FitRequest<'_>) -> Result<EvaluatedArtifact, String>,
{
    if let Some(observations) = &settings.joint_observation_places {
        let unique = observations.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != observations.len()
            || unique.iter().any(|n| *n >= initial.artifact.native_nodes)
        {
            return Err(
                "joint observation places must be unique original native node identities".into(),
            );
        }
    }
    if settings.max_depth == 0
        || settings.max_callback_calls == 0
        || settings.max_move_attempts == 0
        || settings.beam_width == 0
        || settings.max_frontier == 0
        || settings.max_argument_bindings == 0
        || settings.max_compound_pairs == 0
    {
        return Err("positive search depth/work/beam/binding budgets required".into());
    }
    if let Some(expression) = &settings.expression_search {
        if expression.grammar.affine {
            return Err("expression synthesis forbids learned affine adapters".into());
        }
        if expression.max_enumerations_per_parent == 0
            || expression.max_move_attempts == 0
            || expression.max_callback_calls == 0
            || expression.grammar.max_expressions == 0
            || expression.max_move_attempts > settings.max_move_attempts
            || expression.max_callback_calls > settings.max_callback_calls
            || expression.max_callback_calls > expression.max_move_attempts
        {
            return Err(
                "positive expression budgets must fit within global work/callback budgets".into(),
            );
        }
    }
    if let Some(shared) = &settings.shared_dag_search {
        if shared.grammar.affine {
            return Err("shared DAG synthesis forbids learned affine adapters".into());
        }
        if shared.max_enumerations_per_parent == 0
            || shared.max_tuple_checks == 0
            || shared.max_tuples_per_enumeration == 0
            || shared.max_body_nodes == 0
            || shared.max_move_attempts == 0
            || shared.max_callback_calls == 0
            || shared.max_callback_calls > shared.max_move_attempts
            || shared.grammar.max_expressions == 0
        {
            return Err("positive shared DAG enumeration/work/callback budgets required".into());
        }
        shared
            .max_tuple_checks
            .checked_mul(shared.region_limits.max_exits)
            .and_then(|n| n.checked_add(1))
            .ok_or("shared tuple budget overflow")?;
    }
    if let Some(learned) = &settings.learned_dag_search {
        if learned.max_enumerations_per_parent == 0
            || learned.max_move_attempts == 0
            || learned.max_callback_calls == 0
            || learned.max_callback_calls > learned.max_move_attempts
        {
            return Err("positive learned DAG enumeration/work/callback budgets required".into());
        }
    }
    let reserved_moves = settings
        .expression_search
        .as_ref()
        .map_or(0, |s| s.max_move_attempts);
    let reserved_callbacks = settings
        .expression_search
        .as_ref()
        .map_or(0, |s| s.max_callback_calls);
    let shared_reserved_moves = settings
        .shared_dag_search
        .as_ref()
        .map_or(0, |s| s.max_move_attempts);
    let shared_reserved_callbacks = settings
        .shared_dag_search
        .as_ref()
        .map_or(0, |s| s.max_callback_calls);
    let total_reserved_moves = reserved_moves
        .checked_add(shared_reserved_moves)
        .and_then(|n| {
            n.checked_add(
                settings
                    .learned_dag_search
                    .as_ref()
                    .map_or(0, |s| s.max_move_attempts),
            )
        })
        .ok_or("proposal move budget overflow")?;
    let total_reserved_callbacks = reserved_callbacks
        .checked_add(shared_reserved_callbacks)
        .and_then(|n| {
            n.checked_add(
                settings
                    .learned_dag_search
                    .as_ref()
                    .map_or(0, |s| s.max_callback_calls),
            )
        })
        .ok_or("proposal callback budget overflow")?;
    if total_reserved_moves > settings.max_move_attempts
        || total_reserved_callbacks > settings.max_callback_calls
    {
        return Err(
            "expression and shared DAG budgets must sum within global work/callback budgets".into(),
        );
    }
    if constraints.max_fidelity.is_empty()
        || constraints.max_local_errors.is_empty()
        || constraints.max_intervention_errors.is_empty()
    {
        return Err("explicit fidelity/local/intervention measurement axes required".into());
    }
    for group in [
        &constraints.max_fidelity,
        &constraints.max_local_errors,
        &constraints.max_intervention_errors,
    ] {
        let mut names = BTreeSet::new();
        for metric in group {
            if metric.name.is_empty()
                || !names.insert(&metric.name)
                || !metric.value.is_finite()
                || metric.value < 0.
            {
                return Err("invalid/duplicate measurement constraints".into());
            }
        }
    }
    if settings
        .preserve_native_places
        .iter()
        .any(|id| *id >= native.nodes.len())
    {
        return Err("required native place outside source model".into());
    }
    validate_evaluation(&initial.evaluation, constraints)?;
    validate_artifact(&initial.artifact, native, None, settings)?;
    let initial_size = serialized_measured(&initial, codec)?.len();
    let initial_record = CandidateRecord {
        id: 0,
        parent_id: None,
        depth: 0,
        mutation: None,
        evaluation: initial.evaluation.clone(),
        encoded_size_bytes: initial_size,
    };
    let mut resident = BTreeMap::from([(
        0,
        Candidate {
            id: 0,
            parent_id: None,
            depth: 0,
            mutation: None,
            evaluated: initial,
        },
    )]);
    let mut result = SearchResult {
        candidates: vec![],
        frontier: vec![0],
        beam: vec![0],
        report: Report {
            settings: settings.clone(),
            constraints: constraints.clone(),
            counts: Counts::default(),
            attempts: vec![],
            enumerations: vec![],
            compound_enumerations: vec![],
            expression_enumerations: vec![],
            shared_dag_enumerations: vec![],
            shared_dag_schedules: vec![],
            joint_region_enumerations: vec![],
            learned_dag_enumerations: vec![],
            family_budget_omissions: vec![],
            admitted_candidates: vec![initial_record],
            stop_reason: "depth_budget".into(),
        },
    };
    'depths: for depth in 1..=settings.max_depth {
        let mut children = vec![];
        let parents = result.beam.clone();
        for parent_id in parents {
            result.report.counts.enumerations += 1;
            let (inventory, region_error) = match program_regions::propose_regions(
                &resident[&parent_id].evaluated.artifact,
                settings.region_limits.clone(),
            ) {
                Ok(inventory) => (inventory, None),
                Err(reason) => {
                    result.report.counts.enumeration_failures += 1;
                    // Joint proposals have their own enumeration and remain reachable.
                    (
                        Inventory {
                            regions: vec![],
                            skipped: vec![],
                            truncated: false,
                            explored_states: 0,
                        },
                        Some(reason),
                    )
                }
            };
            result.report.counts.regions_proposed += inventory.regions.len();
            result.report.counts.region_states_explored += inventory.explored_states;
            result.report.counts.inventories_truncated += usize::from(inventory.truncated);
            result.report.enumerations.push(Enumeration {
                parent_id,
                depth,
                inventory: region_error.is_none().then(|| inventory.clone()),
                error: region_error,
            });
            let rule_count = resident[&parent_id].evaluated.artifact.program.rules.len();
            let (compound, checked_pairs, truncated) = compound_moves(
                &resident[&parent_id].evaluated.artifact,
                &inventory.regions,
                settings,
                &mut result.report.counts,
            )?;
            result
                .report
                .compound_enumerations
                .push(CompoundEnumeration {
                    parent_id,
                    depth,
                    checked_pairs,
                    truncated,
                });
            let regions = inventory
                .regions
                .into_iter()
                .map(|region| {
                    let (arguments, truncated) = bindings(&region, settings.max_argument_bindings);
                    result.report.counts.argument_binding_sets_truncated +=
                        usize::from(truncated) * rule_count;
                    result.report.counts.expression_binding_sets_truncated +=
                        usize::from(truncated && settings.expression_search.is_some());
                    (region, arguments)
                })
                .collect::<Vec<_>>();
            let expression = expression_moves(
                &resident[&parent_id].evaluated.artifact,
                &regions,
                settings,
                parent_id,
                depth,
                &mut result.report,
            );
            let shared = shared_dag_moves(
                &resident[&parent_id].evaluated.artifact,
                settings,
                parent_id,
                depth,
                &mut result.report,
            );
            let learned = learned_dag_moves(
                &resident[&parent_id].evaluated.artifact,
                settings,
                parent_id,
                depth,
                &mut result.report,
            );
            let singles = regions.into_iter().flat_map(|(region, arguments)| {
                let mut moves = vec![Mutation::ExactExtract {
                    region: region.clone(),
                }];
                for rule_id in 0..rule_count {
                    for native_arguments in &arguments {
                        moves.push(Mutation::ReuseExistingRule {
                            region: region.clone(),
                            rule_id,
                            native_arguments: native_arguments.clone(),
                        });
                    }
                }
                moves
            });
            let structural = compound.into_iter().chain(singles).collect::<Vec<_>>();
            let mut families = std::collections::VecDeque::new();
            if settings.learned_dag_search.is_some() {
                families.push_back(learned.into_iter());
            }
            if settings.shared_dag_search.is_some() {
                families.push_back(shared.into_iter());
            }
            families.push_back(expression.into_iter());
            families.push_back(structural.into_iter());
            let moves = std::iter::from_fn(move || {
                while let Some(mut family) = families.pop_front() {
                    if let Some(proposal) = family.next() {
                        families.push_back(family);
                        return Some(proposal);
                    }
                }
                None
            });
            for mutation in moves {
                if result.report.counts.move_attempts == settings.max_move_attempts
                    || result.report.counts.callback_calls == settings.max_callback_calls
                {
                    result.report.stop_reason =
                        if result.report.counts.move_attempts == settings.max_move_attempts {
                            "move_attempt_budget"
                        } else {
                            "callback_budget"
                        }
                        .into();
                    result.beam = children;
                    break 'depths;
                }
                let is_expression = matches!(mutation, Mutation::SynthesizeExpression { .. });
                let is_shared = matches!(mutation, Mutation::SynthesizeSharedDAG { .. });
                let is_learned = matches!(mutation, Mutation::SynthesizeLearnedDAG { .. } | Mutation::ReuseNativeExpression { .. });
                let counts = &result.report.counts;
                let family = if is_learned {
                    ProposalFamily::LearnedDAG
                } else if is_shared {
                    ProposalFamily::SharedDAG
                } else if is_expression {
                    ProposalFamily::Expression
                } else {
                    ProposalFamily::Structural
                };
                let (moves_used, callbacks_used, allocated_moves, allocated_callbacks) =
                    if is_learned {
                        let config = settings
                            .learned_dag_search
                            .as_ref()
                            .ok_or("learned DAG configuration missing")?;
                        (
                            counts.learned_dag_move_attempts,
                            counts.learned_dag_callback_calls,
                            config.max_move_attempts,
                            config.max_callback_calls,
                        )
                    } else if is_shared {
                        (
                            counts.shared_dag_move_attempts,
                            counts.shared_dag_callback_calls,
                            shared_reserved_moves,
                            shared_reserved_callbacks,
                        )
                    } else if is_expression {
                        (
                            counts.expression_move_attempts,
                            counts.expression_callback_calls,
                            reserved_moves,
                            reserved_callbacks,
                        )
                    } else {
                        (
                            counts.move_attempts
                                - counts.expression_move_attempts
                                - counts.shared_dag_move_attempts
                                - counts.learned_dag_move_attempts,
                            counts.callback_calls
                                - counts.expression_callback_calls
                                - counts.shared_dag_callback_calls
                                - counts.learned_dag_callback_calls,
                            settings.max_move_attempts - total_reserved_moves,
                            settings.max_callback_calls - total_reserved_callbacks,
                        )
                    };
                let omission = if moves_used == allocated_moves {
                    Some("proposal family move allocation exhausted")
                } else if callbacks_used == allocated_callbacks {
                    Some("proposal family callback allocation exhausted")
                } else {
                    None
                };
                if let Some(reason) = omission {
                    omit_family(&mut result.report, parent_id, depth, family, reason);
                    continue;
                }
                let attempt_id = result.report.counts.move_attempts;
                result.report.counts.move_attempts += 1;
                result.report.counts.expression_move_attempts += usize::from(is_expression);
                result.report.counts.shared_dag_move_attempts += usize::from(is_shared);
                result.report.counts.learned_dag_move_attempts += usize::from(is_learned);
                let parent = &resident[&parent_id].evaluated;
                let mut learned_parameters = None;
                let mut initialization = None;
                let mut local_fit = None;
                let proposed = match &mutation {
                    Mutation::ReuseNativeExpression { region, proposal } => {
                        let config = settings.learned_dag_search.as_ref()
                            .and_then(|config| config.native_parent.as_ref())
                            .ok_or("native parent proposal configuration missing")?;
                        program_learned_dag::parent::apply_hypothesis(&parent.artifact, region, proposal, config)
                            .map(|applied| {
                                local_fit = Some(applied.local_fit);
                                learned_parameters = Some(applied.trainable_operator_ids);
                                initialization = Some(applied.initialization);
                                applied.artifact
                            })
                    }
                    Mutation::SynthesizeLearnedDAG {
                        region,
                        expressions,
                        native_arguments,
                    } => {
                        let config = settings
                            .learned_dag_search
                            .as_ref()
                            .ok_or("learned DAG configuration missing")?;
                        program_learned_dag::apply(
                            &parent.artifact,
                            region,
                            expressions,
                            native_arguments,
                            &config.grammar,
                        )
                        .map(|applied| {
                            local_fit = Some(applied.local_fit);
                            learned_parameters = Some(applied.trainable_operator_ids);
                            initialization = Some(applied.initialization);
                            applied.artifact
                        })
                    }
                    Mutation::ExactExtract { region } => {
                        program_regions::extract(&parent.artifact, region)
                    }
                    Mutation::ReuseExistingRule {
                        region,
                        rule_id,
                        native_arguments,
                    } => program_regions::reuse_rule_with_binding(
                        &parent.artifact,
                        region,
                        *rule_id,
                        native_arguments,
                    ),
                    Mutation::ExtractAndReuse {
                        source_region,
                        target_region,
                        native_arguments,
                    } => extract_and_reuse(
                        &parent.artifact,
                        source_region,
                        target_region,
                        native_arguments,
                    ),
                    Mutation::SynthesizeSharedDAG {
                        region,
                        expressions,
                        native_arguments,
                    } => program_expression_search::apply_shared(
                        &parent.artifact,
                        region,
                        expressions,
                        native_arguments,
                    ),
                    Mutation::SynthesizeExpression {
                        region,
                        expression,
                        native_arguments,
                    } => program_expression_search::apply(
                        &parent.artifact,
                        region,
                        expression,
                        native_arguments,
                    ),
                }
                .and_then(|a| {
                    validate_artifact(&a, native, Some(&parent.artifact), settings)?;
                    let (decoded, _) = canonical_with_codec(&a, codec)?;
                    if let Some(local) = &mut local_fit {
                        local.rebind_to_saved_candidate(&a, &decoded)?;
                    }
                    Ok(decoded)
                });
                let mut attempt = Attempt {
                    attempt_id,
                    parent_id,
                    depth,
                    mutation: mutation.clone(),
                    trainable_operator_ids: vec![],
                    status: AttemptStatus::StructuralRejected,
                    reason: None,
                    candidate_id: None,
                };
                let proposed = match proposed {
                    Ok(a) => a,
                    Err(reason) => {
                        result.report.counts.structural_rejections += 1;
                        attempt.reason = Some(reason);
                        result.report.attempts.push(attempt);
                        continue;
                    }
                };
                let operators = match learned_parameters
                    .map(Ok)
                    .unwrap_or_else(|| trainables(&proposed, &mutation))
                {
                    Ok(ids) => ids,
                    Err(reason) => {
                        result.report.counts.structural_rejections += 1;
                        attempt.reason = Some(reason);
                        result.report.attempts.push(attempt);
                        continue;
                    }
                };
                attempt.trainable_operator_ids = operators.clone();
                result.report.counts.callback_calls += 1;
                result.report.counts.expression_callback_calls += usize::from(is_expression);
                result.report.counts.shared_dag_callback_calls += usize::from(is_shared);
                result.report.counts.learned_dag_callback_calls += usize::from(is_learned);
                let fitted = match fit_and_measure(FitRequest {
                    local_fit: local_fit.as_ref(),
                    initialization: initialization.as_ref(),
                    candidate: &proposed,
                    mutation: &mutation,
                    parent,
                    trainable_operator_ids: &operators,
                    attempt_id,
                    parent_id,
                    depth,
                }) {
                    Ok(fitted) => fitted,
                    Err(reason) => {
                        result.report.counts.callback_failures += 1;
                        attempt.status = AttemptStatus::CallbackFailed;
                        attempt.reason = Some(reason);
                        result.report.attempts.push(attempt);
                        continue;
                    }
                };
                if let Err(reason) = validate_evaluation(&fitted.evaluation, constraints) {
                    result.report.counts.measurement_rejections += 1;
                    attempt.status = AttemptStatus::MeasurementRejected;
                    attempt.reason = Some(reason);
                    result.report.attempts.push(attempt);
                    continue;
                }
                let verified = validate_fit_structure(&proposed, &fitted.artifact, &operators)
                    .and_then(|_| {
                        validate_artifact(
                            &fitted.artifact,
                            native,
                            Some(&parent.artifact),
                            settings,
                        )
                    })
                    .and_then(|_| serialized_measured(&fitted, codec));
                let bytes = match verified {
                    Ok(bytes) => bytes,
                    Err(reason) => {
                        result.report.counts.artifact_rejections += 1;
                        attempt.status = AttemptStatus::ArtifactRejected;
                        attempt.reason = Some(reason);
                        result.report.attempts.push(attempt);
                        continue;
                    }
                };
                if let Some(id) = resident
                    .iter()
                    .find_map(|(id, c)| (c.evaluated.artifact == fitted.artifact).then_some(*id))
                {
                    result.report.counts.duplicates += 1;
                    attempt.status = AttemptStatus::Duplicate;
                    attempt.candidate_id = Some(id);
                    result.report.attempts.push(attempt);
                    continue;
                }
                let id = result.report.admitted_candidates.len();
                result.report.admitted_candidates.push(CandidateRecord {
                    id,
                    parent_id: Some(parent_id),
                    depth,
                    mutation: Some(mutation.clone()),
                    evaluation: fitted.evaluation.clone(),
                    encoded_size_bytes: bytes.len(),
                });
                drop(bytes);
                resident.insert(
                    id,
                    Candidate {
                        id,
                        parent_id: Some(parent_id),
                        depth,
                        mutation: Some(mutation),
                        evaluated: fitted,
                    },
                );
                children.push(id);
                let (bounded, _) = bound(&children, &resident, settings.beam_width);
                result.report.counts.beam_pruned += children.len() - bounded.len();
                children = bounded;
                result.frontier.push(id);
                let (frontier, truncated) =
                    bound(&result.frontier, &resident, settings.max_frontier);
                result.report.counts.frontier_truncations += usize::from(truncated);
                result.frontier = frontier;
                // Numeric state is bounded: old parent beam + new child beam +
                // capped visited frontier. Historical lineage holds metadata only.
                resident.retain(|id, _| {
                    result.beam.contains(id)
                        || children.contains(id)
                        || result.frontier.contains(id)
                });
                result.report.counts.fitted_children += 1;
                attempt.status = AttemptStatus::FittedAdmitted;
                attempt.candidate_id = Some(id);
                result.report.attempts.push(attempt);
            }
        }
        if children.is_empty() {
            result.report.stop_reason = "no_fitted_children".into();
            result.beam.clear();
            break;
        }
        result.beam = children;
        result.report.counts.completed_depth = depth;
        resident.retain(|id, _| result.beam.contains(id) || result.frontier.contains(id));
    }
    resident.retain(|id, _| result.beam.contains(id) || result.frontier.contains(id));
    result.candidates = resident.into_values().collect();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_program::{
        exact_precision, Declarations, FamilyInputs, Interface, Operator,
        Provenance, Slot, SlotValues,
    };
    use ndarray::array;
    use std::sync::Arc;
    fn dense(value: f64) -> Arc<Operator> {
        let interface = Interface::native(1).unwrap();
        let values = array![[value]];
        Arc::new(
            Operator::dense(
                "native",
                interface.clone(),
                interface,
                values,
                exact_precision([value]).unwrap(),
                Provenance::native("fixture"),
            )
            .unwrap(),
        )
    }
    fn ordered() -> OperatorProgram {
        OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 1 }, Slot::Raw { width: 1 }],
                parameters: 0,
            },
            bases: vec![],
            operators: vec![dense(2.), dense(3.), dense(1.)],
            rules: vec![],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Raw { slot: 1 },
                Node::Affine {
                    terms: vec![(0, 0), (1, 1)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(1, 0), (0, 1)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(2, 2), (3, 2)],
                    bias: None,
                },
            ],
            output: 4,
        }
    }
    #[test]
    fn learned_scheduler_treats_learned_products_as_nonlinear() {
        use crate::composed_rule_search::Binary;
        use program_learned_dag::{Expr as E, Proposal, TypeRef};
        let affine = E::Affine { parameter: 0, output: TypeRef::Exit(0),
            input: Box::new(E::Argument(0)), bias: false };
        let product = E::Binary(Binary::Multiply, Box::new(affine.clone()), Box::new(affine.clone()));
        let sum = E::Binary(Binary::Add, Box::new(affine.clone()), Box::new(affine.clone()));
        let proposals = [affine, sum, product].into_iter().map(|e| Proposal {
            expressions: vec![e.clone(), e], output_has_learned_ancestor: vec![true, true],
            compiled_node_count: 4, parameter_elements: 4, trainable_operator_count: 1,
        }).collect::<Vec<_>>();
        let order = learned_proposal_order(&proposals, &[2], &[2, 2]);
        assert_eq!(order[0], 2, "shared product must not be scheduled as an affine-only equation");
        assert_eq!(order.iter().copied().collect::<BTreeSet<_>>(), BTreeSet::from([0, 1, 2]));
    }

    #[test]
    fn learned_scheduler_rotates_single_laws_before_mixed_law_sets() {
        use crate::composed_rule_search::Unary;
        use program_learned_dag::{Expr as E, Proposal, TypeRef};
        let value = |law| E::Unary(law, Box::new(E::Affine {
            parameter: 0, output: TypeRef::Exit(0),
            input: Box::new(E::Argument(0)), bias: false,
        }));
        let silu = value(Unary::Silu);
        let gelu = value(Unary::GeluTanh);
        let proposals = [vec![silu.clone(), silu.clone()], vec![silu, gelu.clone()], vec![gelu.clone(), gelu]]
            .into_iter().map(|expressions| Proposal {
                expressions, output_has_learned_ancestor: vec![true, true],
                compiled_node_count: 3, parameter_elements: 4, trainable_operator_count: 1,
            }).collect::<Vec<_>>();
        assert_eq!(learned_proposal_order(&proposals, &[2], &[2, 2]), vec![0, 2, 1]);
    }

    #[test]
    fn learned_scheduler_reaches_generated_full_width_nonlinear_equations_early() {
        use crate::composed_rule_search::{Binary, Unary};
        use crate::operator_program::Interface;
        use program_learned_dag::{Expr as E, TypeRef};
        let grammar = program_learned_dag::Settings {
            latent_widths: vec![768, 3072],
            unary: vec![Unary::Silu, Unary::GeluTanh],
            binary: vec![Binary::Add, Binary::Multiply],
            affine_bias: false, require_shared: true,
            max_operations: 3, max_affine_parameters: 3,
            max_parameter_elements: 6_000_000, max_expression_states: 4096,
            max_tuple_checks: 8192, max_tuples: 128, max_body_nodes: 12, seed: 17,
        };
        let inputs = [Interface::native(768).unwrap()];
        let outputs = [Interface::native(3072).unwrap(), Interface::native(768).unwrap()];
        let inventory = program_learned_dag::enumerate_interfaces(&inputs, &outputs, &grammar).unwrap();
        let native_shape = |p: &program_learned_dag::Proposal| {
            let E::Unary(Unary::GeluTanh, up) = &p.expressions[0] else { return false; };
            let E::Affine { output, input, .. } = up.as_ref() else { return false; };
            let width = match output {
                TypeRef::Input(i) => inputs[*i].width(),
                TypeRef::Exit(i) => outputs[*i].width(),
                TypeRef::Latent { width } => *width,
            };
            width == 3072 && **input == E::Argument(0)
                && matches!(&p.expressions[1], E::Affine { input, .. } if **input == p.expressions[0])
        };
        assert!(inventory.proposals.iter().any(native_shape),
            "bounded grammar must generate full-width native-shape equation before scheduling");
        let order = learned_proposal_order(&inventory.proposals, &[768], &[3072, 768]);
        assert!(order.iter().take(2).any(|i| native_shape(&inventory.proposals[*i])),
            "two callbacks for this region must reach full-width shared native-law equation");
        assert_eq!(order.len(), inventory.proposals.iter().filter(|p| p.trainable_operator_count > 0).count());
        assert_eq!(order.iter().copied().collect::<BTreeSet<_>>().len(), order.len());
        assert_eq!(order, learned_proposal_order(&inventory.proposals, &[768], &[3072, 768]));
        let first = &inventory.proposals[order[0]];
        assert!(first.output_has_learned_ancestor.iter().all(|b| *b));
    }

    fn settings(depth: usize) -> Settings {
        Settings {
            region_limits: Limits {
                max_internal_nodes: 1,
                max_inputs: 3,
                max_regions: 32,
                max_states: 64,
            },
            max_depth: depth,
            max_callback_calls: 200,
            max_move_attempts: 300,
            beam_width: 1,
            max_frontier: 2,
            max_argument_bindings: 8,
            max_compound_pairs: 32,
            preserve_native_places: vec![0],
            joint_observation_places: None,
            expression_search: None,
            shared_dag_search: None,
            learned_dag_search: None,
        }
    }
    #[test]
    fn native_operator_uses_outside_reused_rules_are_frozen() {
        let mut native = ordered();
        native.nodes.push(Node::Affine {
            terms: vec![(4, 0)],
            bias: None,
        });
        native.output = 5;
        let seed = Artifact::native(&native).unwrap();
        let inventory = program_regions::propose_regions(&seed, settings(1).region_limits).unwrap();
        let region = inventory
            .regions
            .iter()
            .find(|r| r.native_write == 2)
            .unwrap();
        let extracted = program_regions::extract(&seed, region).unwrap();
        let inventory =
            program_regions::propose_regions(&extracted, settings(1).region_limits).unwrap();
        let region = inventory
            .regions
            .iter()
            .find(|r| r.native_write == 3)
            .unwrap()
            .clone();
        let proposed =
            program_regions::reuse_rule_with_binding(&extracted, &region, 0, &[1, 0]).unwrap();
        let (decoded, _) = canonical(&proposed).unwrap();
        let mutation = Mutation::ReuseExistingRule {
            region,
            rule_id: 0,
            native_arguments: vec![1, 0],
        };
        let ids = trainables(&decoded, &mutation).unwrap();
        assert_eq!(ids.len(), 1);
        assert!(ids
            .iter()
            .all(|id| decoded.program.operators[*id].matrix()[(0, 0)] == 3.));
    }
    #[test]
    fn shared_rule_preserves_independent_raw_control_arguments() {
        // This is an ordinary executable controlled graph, taking (x, a, b).
        // Both sites share the same coefficient but receive separate control values.
        let native = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 1 }; 3],
                parameters: 0,
            },
            bases: vec![],
            operators: vec![dense(2.), dense(1.)],
            rules: vec![],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Raw { slot: 1 },
                Node::Raw { slot: 2 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Hadamard { left: 3, right: 1 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Hadamard { left: 5, right: 2 },
                Node::Affine {
                    terms: vec![(4, 1), (6, 1)],
                    bias: None,
                },
            ],
            output: 7,
        };
        let seed = Artifact::native(&native).unwrap();
        let mut limits = settings(1).region_limits;
        limits.max_internal_nodes = 2;
        let inventory = program_regions::propose_regions(&seed, limits).unwrap();
        let source = inventory
            .regions
            .iter()
            .find(|r| r.native_write == 4 && r.native_reads == [0, 1])
            .unwrap();
        let target = inventory
            .regions
            .iter()
            .find(|r| r.native_write == 6 && r.native_reads == [0, 2])
            .unwrap();
        let mutation = Mutation::ExtractAndReuse {
            source_region: source.clone(),
            target_region: target.clone(),
            native_arguments: vec![0, 2],
        };
        let (mut candidate, _) =
            canonical(&extract_and_reuse(&seed, source, target, &[0, 2]).unwrap()).unwrap();
        let ids = trainables(&candidate, &mutation).unwrap();
        assert_eq!(ids.len(), 1);
        let calls = candidate
            .program
            .nodes
            .iter()
            .filter_map(|n| {
                if let Node::Call { rule, arguments } = n {
                    Some((*rule, arguments.clone()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, calls[1].0);
        assert_ne!(calls[0].1[1], calls[1].1[1]);
        let family = FamilyInputs {
            rows: 4,
            slots: vec![
                SlotValues::Raw(array![[1.], [1.], [1.], [1.]]),
                SlotValues::Raw(array![[1.], [2.], [1.], [-1.]]),
                SlotValues::Raw(array![[1.], [1.], [2.], [3.]]),
            ],
            layout: None,
        };
        let y = candidate.execute(&family).unwrap();
        assert_eq!(
            y.values[candidate.program.output],
            array![[4.], [6.], [6.], [4.]]
        );
        let proposed = candidate.clone();
        candidate.program.operators[ids[0]] = dense(3.);
        let (candidate, _) = canonical(&candidate).unwrap();
        validate_fit_structure(&proposed, &candidate, &ids).unwrap();
        let bytes = candidate.to_bytes().unwrap();
        let replay = Artifact::from_bytes(&bytes, &native.declarations)
            .unwrap()
            .execute(&family)
            .unwrap();
        assert_eq!(
            replay.values[candidate.program.output],
            array![[6.], [9.], [9.], [6.]]
        );
    }
    #[test]
    fn cached_search_preserves_callback_order_artifacts_and_non_f32_updates() {
        let native = ordered();
        let source = canonical(&Artifact::native(&native).unwrap()).unwrap().0;
        let cache = CanonicalArtifactCache::new(&source, 1 << 20).unwrap();
        let mut config = settings(2);
        config.max_callback_calls = 24;
        config.max_move_attempts = 64;
        let axis = |name: &str, value| vec![Metric { name:name.into(), value }];
        let initial_evaluation = Evaluation {
            fidelity:axis("run", 1.), local_errors:axis("local", 1.),
            intervention_errors:axis("intervention", 1.),
            description_bits:(source.to_bytes().unwrap().len() * 8) as f64,
        };
        let constraints = Constraints {
            max_fidelity:axis("run", 1.), max_local_errors:axis("local", 1.),
            max_intervention_errors:axis("intervention", 1.),
        };
        let run = |cached: bool| {
            let mut callbacks = Vec::new();
            let mut changed = 0;
            let mut callback = |request: FitRequest<'_>| {
                let mut artifact = request.candidate.clone();
                if let Some(&id) = request.trainable_operator_ids.first() {
                    let OperatorBody::Dense { values, precision, .. } = &mut Arc::make_mut(&mut artifact.program.operators[id]).body
                        else { panic!("fixture trainable must be dense"); };
                    values[[0, 0]] = 1.0 + 2.0_f64.powi(-40);
                    *precision = exact_precision(values.iter().copied()).unwrap();
                    changed += 1;
                }
                // Literal test updates, not a fitted scientific measurement.
                let (artifact, bytes) = canonical(&artifact).unwrap();
                callbacks.push((request.attempt_id, request.parent_id,
                    serde_json::to_value(request.mutation).unwrap(),
                    request.trainable_operator_ids.to_vec(), bytes.clone()));
                Ok(EvaluatedArtifact { artifact, evaluation:Evaluation {
                    fidelity:axis("run", 0.5), local_errors:axis("local", 0.5),
                    intervention_errors:axis("intervention", 0.5), description_bits:(bytes.len() * 8) as f64,
                }})
            };
            let initial = EvaluatedArtifact { artifact:source.clone(), evaluation:initial_evaluation.clone() };
            let result = if cached {
                search_with_codec(&native, initial, &config, &constraints, Some(&cache), &mut callback)
            } else {
                search(&native, initial, &config, &constraints, &mut callback)
            }.unwrap();
            assert!(changed > 0, "test must exercise changed numerical owners");
            assert!(callbacks.iter().any(|(_, _, _, _, bytes)| {
                let decoded = Artifact::from_bytes(bytes, &native.declarations).unwrap();
                !decoded.has_f32_literals()
            }), "non-f32 callback coefficients must survive canonical replay");
            let retained = result.candidates.iter().map(|c|
                (c.id, c.evaluated.artifact.to_bytes().unwrap())).collect::<Vec<_>>();
            (callbacks, serde_json::to_value(result.report).unwrap(), retained)
        };
        assert_eq!(run(false), run(true));
        assert!(cache.usage().encoded_native_operator_hits > 0);
        assert!(cache.usage().decoded_native_operator_hits > 0);
    }

    #[test]
    fn native_parent_search_calls_existing_fitter_only_for_changed_programs() {
        use crate::operator_program::Law;
        let native = OperatorProgram {
            declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width: 1 }], parameters: 0 },
            bases: vec![], rules: vec![], operators: vec![dense(2.), dense(3.)],
            nodes: vec![Node::Raw { slot: 0 },
                Node::Affine { terms: vec![(0, 0)], bias: None },
                Node::Pointwise { input: 1, laws: vec![Law::Relu] },
                Node::Affine { terms: vec![(0, 0)], bias: None },
                Node::Pointwise { input: 3, laws: vec![Law::Relu] },
                Node::Affine { terms: vec![(2, 1)], bias: None },
                Node::Affine { terms: vec![(4, 1)], bias: None },
                Node::Concat { parts: vec![5, 6] }], output: 7,
        };
        let source = canonical(&Artifact::native(&native).unwrap()).unwrap().0;
        let mut config = settings(1);
        config.max_callback_calls = 8;
        config.max_move_attempts = 64;
        config.learned_dag_search = Some(LearnedDAGSettings {
            native_parent: Some(program_learned_dag::parent::Settings {
                max_edit_checks: 128, max_proposals: 16, max_body_nodes: 16, max_parameter_elements: 32,
            }),
            region_limits: program_joint_regions::Limits {
                max_internal_nodes: 6, max_inputs: 4, max_exits: 4, max_regions: 32, max_states: 512,
            },
            // Deliberately incapable blind grammar. The native mode must use its
            // parent graph and its own declared edit budget instead.
            grammar: program_learned_dag::Settings {
                latent_widths: vec![], unary: vec![], binary: vec![], affine_bias: false,
                require_shared: false, max_operations: 0, max_affine_parameters: 0,
                max_parameter_elements: 0, max_expression_states: 0, max_tuple_checks: 0,
                max_tuples: 0, max_body_nodes: 0, seed: 0,
            },
            max_enumerations_per_parent: 16, max_move_attempts: 64, max_callback_calls: 8,
        });
        // Native-vs-itself is exactly zero on every declared axis. The callback
        // below deliberately refuses to fabricate measurements for any edit.
        let axis = |name: &str, value| vec![Metric { name: name.into(), value }];
        let evaluation = Evaluation { fidelity: axis("run", 0.), local_errors: axis("local", 0.),
            intervention_errors: axis("intervention", 0.), description_bits: 0. };
        let constraints = Constraints { max_fidelity: axis("run", 1.),
            max_local_errors: axis("local", 1.), max_intervention_errors: axis("intervention", 1.) };
        let mut calls = 0;
        let result = search(&native, EvaluatedArtifact { artifact: source, evaluation },
            &config, &constraints, |request| {
                let Mutation::ReuseNativeExpression { proposal, .. } = request.mutation
                    else { panic!("reserved callbacks must use native edits"); };
                assert!(proposal.edit.is_some(), "exact parent controls must not be fitted as discoveries");
                let local = request.local_fit.expect("same standalone local fitter contract");
                assert_eq!(local.owner_mapping.len(), request.trainable_operator_ids.len());
                assert_eq!(request.initialization.unwrap().random_elements, 0);
                assert_eq!(request.initialization.unwrap().inherited_elements, proposal.parameter_elements);
                calls += 1;
                // This test verifies scheduling and provenance, not fidelity or
                // a scientific result. A failed fitting callback must stay failed.
                Err("deliberately unmeasured hypothesis".into())
            }).unwrap();
        assert!(calls > 0, "native reuse hypotheses must reach the existing fitting callback");
        assert_eq!(calls, result.report.counts.learned_dag_callback_calls);
        assert!(calls <= 8);
        assert_eq!(result.report.counts.fitted_children, 0);
        assert!(result.report.learned_dag_enumerations.iter().any(|r|
            r.native_inventory.as_ref().is_some_and(|i| i.control.edit.is_none() && !i.proposals.is_empty())));
    }

}
