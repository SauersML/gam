//! Bounded candidate-to-candidate search over typed executable artifact DAGs.
//! Extraction supplies structural scaffolding, not scientific discovery. Every
//! archived child is the fitted, measured, decoded object returned by the caller.
//! Pareto fidelity/description-cost objectives are research objectives, not an
//! interpretability score or a certificate of recovered computational organization.
use crate::{
    artifact::Artifact,
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
            Self::SynthesizeSharedDAG { .. } | Self::SynthesizeLearnedDAG { .. } => None,
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
pub struct StoredCandidate {
    pub id: usize,
    pub parent_id: Option<usize>,
    pub depth: usize,
    pub mutation: Option<Mutation>,
    pub evaluation: Evaluation,
    pub artifact_bytes: Vec<u8>,
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub candidates: Vec<StoredCandidate>,
    pub frontier: Vec<usize>,
    pub beam: Vec<usize>,
    pub report: Report,
}
impl SearchResult {
    /// Standalone codec bytes of actual fitted artifacts, not descriptions of
    /// hypothetical calls. Decode with the native declarations alone.
    pub fn snapshot(&self) -> Result<Snapshot, String> {
        let candidates = self
            .candidates
            .iter()
            .map(|c| {
                Ok(StoredCandidate {
                    id: c.id,
                    parent_id: c.parent_id,
                    depth: c.depth,
                    mutation: c.mutation.clone(),
                    evaluation: c.evaluated.evaluation.clone(),
                    artifact_bytes: serialized_measured(&c.evaluated)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Snapshot {
            candidates,
            frontier: self.frontier.clone(),
            beam: self.beam.clone(),
            report: self.report.clone(),
        })
    }
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
fn validate_evaluation(e: &Evaluation, constraints: &Constraints) -> Result<(), String> {
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
fn serialized_measured(e: &EvaluatedArtifact) -> Result<Vec<u8>, String> {
    let (decoded, bytes) = canonical(&e.artifact)?;
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
    let inventory =
        match program_joint_regions::propose_regions(artifact, shared.region_limits.clone()) {
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
    let regions =
        match program_joint_regions::propose_regions(artifact, config.region_limits.clone()) {
            Ok(inventory) => {
                report.counts.learned_dag_enumerations_truncated +=
                    usize::from(inventory.truncated);
                inventory.regions
            }
            Err(error) => {
                report.learned_dag_enumerations.push(LearnedDAGEnumeration {
                    parent_id,
                    depth,
                    region: None,
                    inventory: None,
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
            error: None,
        };
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
                lists.push_back(
                    inventory
                        .proposals
                        .iter()
                        .filter(|p| p.trainable_operator_count > 0)
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
    mut fit_and_measure: F,
) -> Result<SearchResult, String>
where
    F: FnMut(FitRequest<'_>) -> Result<EvaluatedArtifact, String>,
{
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
    let initial_size = serialized_measured(&initial)?.len();
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
                let is_learned = matches!(mutation, Mutation::SynthesizeLearnedDAG { .. });
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
                let proposed = match &mutation {
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
                            learned_parameters = Some(applied.trainable_operator_ids);
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
                    canonical(&a).map(|(a, _)| a)
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
                    .and_then(|_| serialized_measured(&fitted));
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
        Coefficient, Declarations, FamilyInputs, Interface, Law, Operator, Provenance, Slot,
        SlotValues, exact_precision,
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
    fn heterogeneous() -> OperatorProgram {
        OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 1 }],
                parameters: 0,
            },
            bases: vec![],
            operators: vec![dense(1.)],
            rules: vec![],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Pointwise {
                    input: 0,
                    laws: vec![Law::Relu],
                },
                Node::Pointwise {
                    input: 0,
                    laws: vec![Law::Silu],
                },
                Node::Pointwise {
                    input: 0,
                    laws: vec![Law::Relu],
                },
                Node::Affine {
                    terms: vec![(1, 0), (2, 0), (3, 0)],
                    bias: None,
                },
            ],
            output: 4,
        }
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
    fn inputs(program: &OperatorProgram) -> FamilyInputs {
        FamilyInputs {
            rows: 3,
            slots: (0..program.declarations.slots.len())
                .map(|i| {
                    SlotValues::Raw(if i == 0 {
                        array![[-1.], [0.2], [1.]]
                    } else {
                        array![[0.3], [-0.7], [0.5]]
                    })
                })
                .collect(),
            layout: None,
        }
    }
    fn metric(name: &str, value: f64) -> Metric {
        Metric {
            name: name.into(),
            value,
        }
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
            expression_search: None,
            shared_dag_search: None,
            learned_dag_search: None,
        }
    }
    fn constraints() -> Constraints {
        Constraints {
            max_fidelity: vec![metric("output_squared_error", 1e-20)],
            max_local_errors: vec![metric("local_squared_error", 1e-20)],
            max_intervention_errors: vec![metric("scaled_input_output_squared_error", 1e-20)],
        }
    }
    fn evaluated(a: &Artifact, native: &OperatorProgram) -> EvaluatedArtifact {
        let (artifact, _) = canonical(a).unwrap();
        let family = inputs(native);
        let original = native.execute(&family, false).unwrap();
        let candidate = artifact.execute(&family).unwrap();
        let error = (&candidate.values[artifact.program.output] - &original.values[native.output])
            .mapv(|v| v * v)
            .sum();
        let local = if artifact.blocks.is_empty() {
            0.
        } else {
            let (local, _) = artifact.local_program(native).unwrap();
            let values = local.execute(&family, false).unwrap();
            values.values[local.output].mapv(|v| v * v).sum()
        };
        let native_changed = native
            .execute_edited(&family, |id, v, _| {
                if id == 0 {
                    *v *= 0.75;
                }
                Ok(())
            })
            .unwrap();
        let changed = artifact
            .execute_edited(&family, |id, v, _| {
                if id == artifact.place(0).unwrap() {
                    *v *= 0.75;
                }
                Ok(())
            })
            .unwrap();
        let intervention = (&changed.values[artifact.program.output]
            - &native_changed.values[native.output])
            .mapv(|v| v * v)
            .sum();
        let bits = artifact.encode().unwrap().len_bits() as f64;
        EvaluatedArtifact {
            artifact,
            evaluation: Evaluation {
                fidelity: vec![metric("output_squared_error", error)],
                local_errors: vec![metric("local_squared_error", local)],
                intervention_errors: vec![metric(
                    "scaled_input_output_squared_error",
                    intervention,
                )],
                description_bits: bits,
            },
        }
    }
    #[test]
    fn fitted_parent_composition_builds_heterogeneous_persistent_library() {
        let native = heterogeneous();
        let initial = evaluated(&Artifact::native(&native).unwrap(), &native);
        let mut observed = vec![];
        let result = search(&native, initial, &settings(3), &constraints(), |request| {
            let expected = match &request.mutation {
                Mutation::ExactExtract { region }
                    if request.depth == 1 && region.native_write == 1 =>
                {
                    true
                }
                Mutation::ExactExtract { region }
                    if request.depth == 2 && region.native_write == 2 =>
                {
                    true
                }
                Mutation::ReuseExistingRule {
                    region, rule_id, ..
                } if request.depth == 3 && region.native_write == 3 && *rule_id == 0 => true,
                _ => false,
            };
            if !expected {
                return Err("fixture constrains the required multi-step path".into());
            }
            observed.push((request.depth, request.parent.artifact.program.rules.len()));
            if matches!(request.mutation, Mutation::ExactExtract { .. }) {
                assert!(request.trainable_operator_ids.is_empty());
            }
            Ok(evaluated(request.candidate, &native))
        })
        .unwrap();
        assert_eq!(observed, vec![(1, 0), (2, 1), (3, 2)]);
        let final_state = result.candidates.iter().find(|c| c.depth == 3).unwrap();
        assert_eq!(final_state.evaluated.artifact.program.rules.len(), 2);
        assert_eq!(
            final_state
                .evaluated
                .artifact
                .program
                .nodes
                .iter()
                .filter(|n| matches!(n, Node::Call { .. }))
                .count(),
            3
        );
        assert_eq!(result.report.admitted_candidates.len(), 4);
        assert!(result.candidates.len() <= settings(3).beam_width + settings(3).max_frontier);
    }
    #[test]
    fn typed_argument_permutation_finds_noncommutative_reuse() {
        let native = ordered();
        let initial = evaluated(&Artifact::native(&native).unwrap(), &native);
        let mut chosen = vec![];
        let result = search(&native, initial, &settings(2), &constraints(), |request| {
            match request.mutation {
                Mutation::ExactExtract { region }
                    if request.depth == 1 && region.native_write == 2 => {}
                Mutation::ReuseExistingRule {
                    region,
                    rule_id,
                    native_arguments,
                } if request.depth == 2 && region.native_write == 3 && *rule_id == 0 => {
                    chosen.push(native_arguments.clone());
                    if native_arguments == &[1, 0] {
                        assert_eq!(request.trainable_operator_ids.len(), 2);
                    }
                }
                _ => return Err("fixture path".into()),
            }
            Ok(evaluated(request.candidate, &native))
        })
        .unwrap();
        assert_eq!(chosen, vec![vec![0, 1], vec![1, 0]]);
        assert!(result.report.counts.measurement_rejections > 0);
        assert!(result.candidates.iter().any(|c|matches!(&c.mutation,Some(Mutation::ReuseExistingRule{native_arguments,..}) if native_arguments==&[1,0])));
    }
    #[test]
    fn finite_intervention_failure_never_enters_archive() {
        let native = heterogeneous();
        let initial = evaluated(&Artifact::native(&native).unwrap(), &native);
        let result = search(&native, initial, &settings(1), &constraints(), |request| {
            let mut measured = evaluated(request.candidate, &native);
            measured.evaluation.intervention_errors[0].value = 0.1;
            Ok(measured)
        })
        .unwrap();
        assert_eq!(result.report.admitted_candidates.len(), 1);
        assert_eq!(result.frontier, vec![0]);
        assert!(result.report.counts.measurement_rejections > 0);
    }
    #[test]
    fn callback_failures_and_work_budget_are_deterministic() {
        let native = heterogeneous();
        let mut configuration = settings(3);
        configuration.max_callback_calls = 2;
        let run = || {
            search(
                &native,
                evaluated(&Artifact::native(&native).unwrap(), &native),
                &configuration,
                &constraints(),
                |_| Err("fit failed".into()),
            )
            .unwrap()
        };
        let first = run();
        let second = run();
        assert_eq!(first.report.counts.callback_calls, 2);
        assert_eq!(first.report.counts.callback_failures, 2);
        assert_eq!(first.report.stop_reason, "callback_budget");
        assert_eq!(
            serde_json::to_vec(&first.report).unwrap(),
            serde_json::to_vec(&second.report).unwrap()
        );
    }
    #[test]
    fn snapshots_decode_actual_returned_fitted_operator() {
        let native = ordered();
        let mut permissive = constraints();
        permissive.max_fidelity[0].value = 100.;
        permissive.max_local_errors[0].value = 100.;
        permissive.max_intervention_errors[0].value = 100.;
        let result = search(
            &native,
            evaluated(&Artifact::native(&native).unwrap(), &native),
            &settings(2),
            &permissive,
            |request| match request.mutation {
                Mutation::ExactExtract { region }
                    if request.depth == 1 && region.native_write == 2 =>
                {
                    Ok(evaluated(request.candidate, &native))
                }
                Mutation::ReuseExistingRule {
                    region,
                    native_arguments,
                    ..
                } if request.depth == 2
                    && region.native_write == 3
                    && native_arguments == &[1, 0] =>
                {
                    let mut fitted = request.candidate.clone();
                    let id = request.trainable_operator_ids[0];
                    let old = &fitted.program.operators[id];
                    let values = array![[2.5]];
                    fitted.program.operators[id] = Arc::new(
                        Operator::dense(
                            "jointly fitted",
                            old.rows.clone(),
                            old.cols.clone(),
                            values,
                            exact_precision([2.5]).unwrap(),
                            old.provenance.clone(),
                        )
                        .unwrap(),
                    );
                    Ok(evaluated(&fitted, &native))
                }
                _ => Err("fixture path".into()),
            },
        )
        .unwrap();
        let snapshot = result.snapshot().unwrap();
        let serialized = serde_json::to_vec(&snapshot).unwrap();
        let recovered: Snapshot = serde_json::from_slice(&serialized).unwrap();
        let final_state = recovered.candidates.iter().find(|c| c.depth == 2).unwrap();
        let decoded =
            Artifact::from_bytes(&final_state.artifact_bytes, &native.declarations).unwrap();
        assert!(
            decoded
                .program
                .operators
                .iter()
                .any(|o| matches!(&o.body,OperatorBody::Dense{values,..} if values[(0,0)]==2.5))
        );
        assert!(
            result
                .report
                .admitted_candidates
                .iter()
                .any(|c| c.depth == 2 && c.evaluation.fidelity[0].value > 0.)
        );
    }
    #[test]
    fn noncanonical_measurements_are_rejected() {
        let native = heterogeneous();
        let initial = evaluated(&Artifact::native(&native).unwrap(), &native);
        let result = search(&native, initial, &settings(1), &constraints(), |request| {
            let mut fitted = evaluated(request.candidate, &native);
            fitted.artifact.program.rules[0].name = "unmeasured metadata".into();
            Ok(fitted)
        })
        .unwrap();
        assert_eq!(result.report.admitted_candidates.len(), 1);
        assert!(result.report.counts.artifact_rejections > 0);
    }
    #[test]
    fn required_native_intervention_boundary_cannot_disappear() {
        let native = heterogeneous();
        let initial = evaluated(&Artifact::native(&native).unwrap(), &native);
        let mut configuration = settings(1);
        configuration.region_limits.max_internal_nodes = 2;
        configuration.preserve_native_places = vec![0, 2];
        let result = search(
            &native,
            initial,
            &configuration,
            &constraints(),
            |request| Ok(evaluated(request.candidate, &native)),
        )
        .unwrap();
        assert!(
            result
                .candidates
                .iter()
                .all(|c| c.evaluated.artifact.place(2).is_some())
        );
        assert!(result.report.attempts.iter().any(|a| {
            a.reason
                .as_deref()
                .is_some_and(|r| r.contains("required native intervention boundary"))
        }));
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
        assert!(
            ids.iter()
                .all(|id| decoded.program.operators[*id].matrix()[(0, 0)] == 3.)
        );
    }
    #[test]
    fn compound_reuse_reaches_fitting_before_small_budget_expires() {
        let native = ordered();
        let mut configuration = settings(1);
        configuration.max_callback_calls = 2;
        let result = search(
            &native,
            evaluated(&Artifact::native(&native).unwrap(), &native),
            &configuration,
            &constraints(),
            |request| {
                assert!(matches!(request.mutation, Mutation::ExtractAndReuse { .. }));
                Ok(evaluated(request.candidate, &native))
            },
        )
        .unwrap();
        assert_eq!(result.report.counts.callback_calls, 2);
        assert_eq!(result.report.counts.fitted_children, 1);
        assert!(result.candidates.iter().any(|c| matches!(&c.mutation,
            Some(Mutation::ExtractAndReuse { native_arguments, .. }) if native_arguments == &[1, 0])));
        assert_eq!(
            result.report.compound_enumerations[0].checked_pairs[0].source_native_write,
            2
        );
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
    fn callback_cannot_fit_unlisted_native_background_operator() {
        let native = ordered();
        let mut permissive = constraints();
        for limits in [
            &mut permissive.max_fidelity,
            &mut permissive.max_local_errors,
            &mut permissive.max_intervention_errors,
        ] {
            limits[0].value = 1e6;
        }
        let result = search(
            &native,
            evaluated(&Artifact::native(&native).unwrap(), &native),
            &settings(1),
            &permissive,
            |request| {
                if !matches!(request.mutation, Mutation::ExtractAndReuse { .. }) {
                    return Err("compound only".into());
                }
                let mut candidate = request.candidate.clone();
                let id = (0..candidate.program.operators.len())
                    .find(|id| !request.trainable_operator_ids.contains(id))
                    .unwrap();
                candidate.program.operators[id] = dense(4.);
                Ok(evaluated(&candidate, &native))
            },
        )
        .unwrap();
        assert_eq!(result.report.admitted_candidates.len(), 1);
        assert!(result.report.attempts.iter().any(|a| {
            a.reason
                .as_deref()
                .is_some_and(|r| r.contains("frozen operator"))
        }));
    }
    #[test]
    fn callback_cannot_silently_change_a_logged_rule_body() {
        let native = heterogeneous();
        let mut permissive = constraints();
        for limits in [
            &mut permissive.max_fidelity,
            &mut permissive.max_local_errors,
            &mut permissive.max_intervention_errors,
        ] {
            limits[0].value = 1e6;
        }
        let result = search(
            &native,
            evaluated(&Artifact::native(&native).unwrap(), &native),
            &settings(1),
            &permissive,
            |request| {
                let mut candidate = request.candidate.clone();
                let body = &mut candidate.program.rules[0];
                let mut changed = false;
                for node in &mut body.nodes {
                    if let Node::Pointwise { laws, .. } = node {
                        *laws = vec![Law::Gelu];
                        changed = true;
                    }
                }
                if !changed {
                    return Err("fixture requires a pointwise body".into());
                }
                Ok(evaluated(&candidate, &native))
            },
        )
        .unwrap();
        assert_eq!(result.report.admitted_candidates.len(), 1);
        assert!(result.report.attempts.iter().any(|a| {
            a.reason
                .as_deref()
                .is_some_and(|r| r.contains("proposed graph"))
        }));
    }
    #[test]
    fn synthesized_expression_absent_from_native_dag_is_measured_and_saved() {
        use crate::{composed_rule_search::Binary, operator_program::Scale};
        let native = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 1 }],
                parameters: 0,
            },
            bases: vec![],
            operators: vec![],
            rules: vec![],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Bilinear {
                    left: 0,
                    right: 0,
                    scale: Scale::One,
                },
            ],
            output: 1,
        };
        assert!(
            !native
                .nodes
                .iter()
                .any(|n| matches!(n, Node::Hadamard { .. }))
        );
        let mut configuration = settings(1);
        configuration.max_callback_calls = 3;
        configuration.max_move_attempts = 3;
        configuration.expression_search = Some(ExpressionSettings {
            grammar: Grammar {
                arguments: 0,
                max_operations: 1,
                max_expressions: 4,
                unary: vec![],
                binary: vec![Binary::Multiply],
                affine: false,
            },
            max_enumerations_per_parent: 1,
            max_move_attempts: 2,
            max_callback_calls: 2,
        });
        let mut saved = vec![];
        let result = search(
            &native,
            evaluated(&Artifact::native(&native).unwrap(), &native),
            &configuration,
            &constraints(),
            |request| {
                let measured = evaluated(request.candidate, &native);
                if let Mutation::SynthesizeExpression { expression, .. } = request.mutation {
                    assert!(request.trainable_operator_ids.is_empty());
                    assert!(
                        measured
                            .artifact
                            .program
                            .operators
                            .iter()
                            .all(|o| !matches!(o.body, OperatorBody::Dense { .. }))
                    );
                    saved.push((expression.clone(), measured.artifact.to_bytes().unwrap()));
                }
                Ok(measured)
            },
        )
        .unwrap();
        let wanted = Expr::Binary(
            Binary::Multiply,
            Box::new(Expr::Argument(0)),
            Box::new(Expr::Argument(0)),
        );
        assert_eq!(result.report.counts.expression_callback_calls, 2);
        assert_eq!(result.report.counts.callback_calls, 3);
        assert!(result.report.admitted_candidates.iter().any(|c| matches!(&c.mutation, Some(Mutation::SynthesizeExpression { expression, .. }) if expression == &wanted)));
        let bytes = &saved.iter().find(|(e, _)| e == &wanted).unwrap().1;
        let artifact = Artifact::from_bytes(bytes, &native.declarations).unwrap();
        assert!(
            artifact
                .program
                .rules
                .iter()
                .any(|r| r.nodes.iter().any(|n| matches!(n, Node::Hadamard { .. })))
        );
        let output = artifact.execute(&inputs(&native)).unwrap();
        let expected = native.execute(&inputs(&native), false).unwrap();
        assert_eq!(
            output.values[artifact.program.output],
            expected.values[native.output]
        );
        assert_eq!(
            result.report.expression_enumerations[0]
                .effective_grammar
                .arguments,
            1
        );
    }
    fn joint_fixture() -> OperatorProgram {
        let mut native = heterogeneous();
        native.operators = vec![std::sync::Arc::new(Operator::identity(
            "sum",
            Interface::native(1).unwrap(),
        ))];
        native.nodes = vec![
            Node::Raw { slot: 0 },
            Node::Pointwise {
                input: 0,
                laws: vec![Law::Silu],
            },
            Node::Gain {
                input: 1,
                coefficient: Coefficient::Number(1.),
            },
            Node::Gain {
                input: 1,
                coefficient: Coefficient::Number(1.),
            },
            Node::Affine {
                terms: vec![(2, 0), (3, 0)],
                bias: None,
            },
        ];
        native.output = 4;
        native
    }
    fn joint_settings() -> SharedDAGSettings {
        SharedDAGSettings {
            region_limits: program_joint_regions::Limits {
                max_internal_nodes: 3,
                max_inputs: 1,
                max_exits: 2,
                max_regions: 8,
                max_states: 32,
            },
            grammar: Grammar {
                arguments: 0,
                max_operations: 1,
                max_expressions: 4,
                unary: vec![crate::composed_rule_search::Unary::Silu],
                binary: vec![],
                affine: false,
            },
            max_enumerations_per_parent: 4,
            max_tuple_checks: 8,
            max_tuples_per_enumeration: 4,
            max_body_nodes: 8,
            max_move_attempts: 1,
            max_callback_calls: 1,
        }
    }
    #[test]
    fn learned_dag_search_reserves_callbacks_and_only_fits_new_parameters() {
        let native = joint_fixture();
        let mut configuration = settings(1);
        configuration.max_callback_calls = 2;
        configuration.max_move_attempts = 2;
        configuration.learned_dag_search = Some(LearnedDAGSettings {
            region_limits: joint_settings().region_limits,
            grammar: program_learned_dag::Settings {
                latent_widths: vec![1],
                unary: vec![],
                binary: vec![],
                affine_bias: false,
                require_shared: false,
                max_operations: 1,
                max_affine_parameters: 2,
                max_parameter_elements: 2,
                max_expression_states: 128,
                max_tuple_checks: 128,
                max_tuples: 16,
                max_body_nodes: 8,
                seed: 19,
            },
            max_enumerations_per_parent: 1,
            max_move_attempts: 1,
            max_callback_calls: 1,
        });
        let mut inspected = false;
        let result = search(
            &native,
            evaluated(
                &Artifact::native(&native).expect("native artifact"),
                &native,
            ),
            &configuration,
            &constraints(),
            |request| {
                if matches!(request.mutation, Mutation::SynthesizeLearnedDAG { .. }) {
                    assert!(!request.trainable_operator_ids.is_empty());
                    let mut fitted = request.candidate.clone();
                    for &id in request.trainable_operator_ids {
                        assert!(matches!(
                            fitted.program.operators[id].body,
                            OperatorBody::Dense { .. }
                        ));
                        fitted.program.operators[id] = dense(2.5);
                    }
                    let (fitted, _) = canonical(&fitted).expect("stable parameter identities");
                    validate_fit_structure(
                        request.candidate,
                        &fitted,
                        request.trainable_operator_ids,
                    )
                    .expect("new affine parameters may be fitted");
                    let frozen = fitted
                        .program
                        .operators
                        .iter()
                        .enumerate()
                        .find(|(id, _)| !request.trainable_operator_ids.contains(id))
                        .map(|(id, _)| id)
                        .expect("native output operator remains frozen");
                    let mut invalid = fitted.clone();
                    invalid.program.operators[frozen] = dense(4.);
                    assert!(
                        validate_fit_structure(
                            request.candidate,
                            &invalid,
                            request.trainable_operator_ids
                        )
                        .is_err()
                    );
                    let encoded =
                        serde_json::to_value(request.mutation).expect("mutation encoding");
                    assert_eq!(encoded["kind"], "synthesize_learned_dag");
                    let decoded: Mutation =
                        serde_json::from_value(encoded).expect("mutation decoding");
                    assert_eq!(&decoded, request.mutation);
                    inspected = true;
                }
                Err("provenance and scheduling regression, no fitted scientific candidate".into())
            },
        )
        .expect("bounded search");
        assert!(inspected);
        assert_eq!(result.report.counts.learned_dag_callback_calls, 1);
        assert_eq!(result.report.counts.callback_calls, 2);
        assert!(result.report.counts.learned_dag_proposals > 0);
        assert!(!result.report.learned_dag_enumerations.is_empty());
        assert!(matches!(
            result.report.attempts[0].mutation,
            Mutation::SynthesizeLearnedDAG { .. }
        ));
    }
    #[test]
    fn joint_synthesis_search_saves_actual_shared_state_and_distinct_native_exits() {
        let native = joint_fixture();
        let mut configuration = settings(1);
        configuration.shared_dag_search = Some(joint_settings());
        configuration.max_callback_calls = 2;
        configuration.max_move_attempts = 2;
        let mut saved = None;
        let result = search(
            &native,
            evaluated(&Artifact::native(&native).unwrap(), &native),
            &configuration,
            &constraints(),
            |request| {
                let measured = evaluated(request.candidate, &native);
                if let Mutation::SynthesizeSharedDAG {
                    region,
                    expressions,
                    ..
                } = request.mutation
                {
                    assert!(request.trainable_operator_ids.is_empty());
                    assert_eq!(region.native_writes, vec![2, 3]);
                    assert_eq!(expressions.len(), 2);
                    assert!(measured.artifact.place(1).is_none());
                    assert_ne!(measured.artifact.place(2), measured.artifact.place(3));
                    saved = Some(measured.artifact.to_bytes().unwrap());
                }
                Ok(measured)
            },
        )
        .unwrap();
        assert_eq!(result.report.counts.shared_dag_callback_calls, 1);
        assert!(
            result
                .report
                .admitted_candidates
                .iter()
                .any(|c| matches!(c.mutation, Some(Mutation::SynthesizeSharedDAG { .. })))
        );
        let decoded = Artifact::from_bytes(&saved.unwrap(), &native.declarations).unwrap();
        assert_eq!(
            decoded
                .program
                .nodes
                .iter()
                .filter(|n| matches!(n, Node::Pointwise { .. }))
                .count(),
            1
        );
        assert_eq!(
            decoded.execute(&inputs(&native)).unwrap().values[decoded.program.output],
            native.execute(&inputs(&native), false).unwrap().values[native.output]
        );
        let mutation = &result
            .report
            .attempts
            .iter()
            .find(|a| matches!(a.mutation, Mutation::SynthesizeSharedDAG { .. }))
            .unwrap()
            .mutation;
        let value = serde_json::to_value(mutation).unwrap();
        assert_eq!(value["kind"], "synthesize_shared_dag");
        let roundtrip: Mutation = serde_json::from_value(value).unwrap();
        assert_eq!(&roundtrip, mutation);
    }
    #[test]
    fn joint_callbacks_are_reserved_and_finite_intervention_failures_are_rejected() {
        let native = joint_fixture();
        let mut configuration = settings(1);
        configuration.max_callback_calls = 3;
        configuration.max_move_attempts = 3;
        configuration.shared_dag_search = Some(joint_settings());
        configuration.expression_search = Some(ExpressionSettings {
            grammar: joint_settings().grammar,
            max_enumerations_per_parent: 1,
            max_move_attempts: 1,
            max_callback_calls: 1,
        });
        let result = search(
            &native,
            evaluated(&Artifact::native(&native).unwrap(), &native),
            &configuration,
            &constraints(),
            |request| {
                if matches!(request.mutation, Mutation::SynthesizeSharedDAG { .. }) {
                    let mut measured = evaluated(request.candidate, &native);
                    measured.evaluation.intervention_errors[0].value = 0.1;
                    Ok(measured)
                } else {
                    Err("records family scheduling".into())
                }
            },
        )
        .unwrap();
        assert_eq!(result.report.counts.shared_dag_callback_calls, 1);
        assert_eq!(result.report.counts.expression_callback_calls, 1);
        assert_eq!(result.report.counts.callback_calls, 3);
        assert!(matches!(
            result.report.attempts[0].mutation,
            Mutation::SynthesizeSharedDAG { .. }
        ));
        assert!(matches!(
            result.report.attempts[0].status,
            AttemptStatus::MeasurementRejected
        ));
        assert_eq!(result.report.admitted_candidates.len(), 1);
        configuration.max_callback_calls = 1;
        assert!(
            search(
                &native,
                evaluated(&Artifact::native(&native).unwrap(), &native),
                &configuration,
                &constraints(),
                |_| panic!("invalid sums must not run")
            )
            .unwrap_err()
            .contains("budgets")
        );
        let mut legacy = serde_json::to_value(settings(1)).unwrap();
        legacy.as_object_mut().unwrap().remove("shared_dag_search");
        assert!(
            serde_json::from_value::<Settings>(legacy)
                .unwrap()
                .shared_dag_search
                .is_none()
        );
    }
    #[test]
    fn equivalent_permutation_proposals_are_deduplicated_before_failed_callbacks() {
        let mut native = ordered();
        native.operators = vec![std::sync::Arc::new(Operator::identity(
            "sum",
            Interface::native(1).unwrap(),
        ))];
        native.nodes = vec![
            Node::Raw { slot: 0 },
            Node::Raw { slot: 1 },
            Node::Affine {
                terms: vec![(0, 0), (1, 0)],
                bias: None,
            },
            Node::Gain {
                input: 2,
                coefficient: Coefficient::Number(1.),
            },
            Node::Gain {
                input: 2,
                coefficient: Coefficient::Number(1.),
            },
            Node::Affine {
                terms: vec![(3, 0), (4, 0)],
                bias: None,
            },
        ];
        native.output = 5;
        let mut configuration = settings(1);
        configuration.max_callback_calls = 7;
        configuration.max_move_attempts = 7;
        let mut joint = joint_settings();
        joint.region_limits.max_inputs = 2;
        joint.grammar.unary = vec![];
        joint.grammar.binary = vec![crate::composed_rule_search::Binary::Add];
        joint.grammar.max_expressions = 16;
        joint.max_tuple_checks = 64;
        joint.max_tuples_per_enumeration = 16;
        joint.max_callback_calls = 6;
        joint.max_move_attempts = 6;
        configuration.shared_dag_search = Some(joint);
        let mut seen = BTreeSet::new();
        let mut per_cut = BTreeMap::<(Vec<usize>, Vec<usize>, Vec<usize>), usize>::new();
        let result = search(
            &native,
            evaluated(&Artifact::native(&native).unwrap(), &native),
            &configuration,
            &constraints(),
            |request| {
                if let Mutation::SynthesizeSharedDAG {
                    region,
                    expressions,
                    native_arguments,
                } = request.mutation
                {
                    assert_eq!(native_arguments, &region.native_reads);
                    let cut = (
                        region.native_reads.clone(),
                        region.native_writes.clone(),
                        region.current_internal_nodes.clone(),
                    );
                    assert!(
                        seen.insert((
                            cut.clone(),
                            request.candidate.to_bytes().expect("actual proposed artifact bytes"),
                        )),
                        "even failed proposals for the same native cut must be deduplicated before callback"
                    );
                    *per_cut.entry(cut).or_default() += 1;
                    assert_eq!(expressions.len(), region.native_writes.len());
                }
                Err("intentionally failed measurement to audit duplicate work".into())
            },
        )
        .unwrap();
        assert_eq!(result.report.counts.shared_dag_proposals, 7);
        assert_eq!(result.report.counts.shared_dag_unique_proposals, 4);
        assert_eq!(
            result.report.counts.shared_dag_duplicate_binding_proposals,
            3
        );
        assert_eq!(result.report.counts.shared_dag_callback_calls, 4);
        assert_eq!(seen.len(), 4);
        // Interior extraction still searches three equations, each exactly once
        // despite two input permutations. Boundary anchoring contributes a distinct
        // one-input cut; identical expression syntax there means a different native
        // input and must not be discarded as the earlier hypothesis.
        assert_eq!(per_cut[&(vec![0, 1], vec![3, 4], vec![2, 3, 4])], 3);
        assert_eq!(per_cut[&(vec![2], vec![3, 4], vec![3, 4])], 1);
        assert_eq!(result.report.shared_dag_schedules.len(), 2);
        assert_eq!(
            result.report
                .shared_dag_schedules
                .iter()
                .map(|s| s.unique_proposals)
                .sum::<usize>(),
            4
        );
        assert_eq!(result.report.admitted_candidates.len(), 1);
    }
    #[test]
    fn expression_setting_defaults_off_and_rejects_unallocated_or_adapter_work() {
        let mut json = serde_json::to_value(settings(1)).unwrap();
        json.as_object_mut().unwrap().remove("expression_search");
        let old_settings: Settings = serde_json::from_value(json).unwrap();
        assert!(old_settings.expression_search.is_none());
        let native = heterogeneous();
        let mut configuration = old_settings;
        configuration.expression_search = Some(ExpressionSettings {
            grammar: Grammar {
                arguments: 0,
                max_operations: 1,
                max_expressions: 4,
                unary: vec![],
                binary: vec![],
                affine: true,
            },
            max_enumerations_per_parent: 1,
            max_move_attempts: 2,
            max_callback_calls: 2,
        });
        let run = |configuration: &Settings| {
            search(
                &native,
                evaluated(&Artifact::native(&native).unwrap(), &native),
                configuration,
                &constraints(),
                |_| panic!("invalid settings must not invoke fitting"),
            )
        };
        assert!(run(&configuration).unwrap_err().contains("affine adapters"));
        configuration
            .expression_search
            .as_mut()
            .unwrap()
            .grammar
            .affine = false;
        configuration
            .expression_search
            .as_mut()
            .unwrap()
            .max_callback_calls = configuration.max_callback_calls + 1;
        assert!(
            run(&configuration)
                .unwrap_err()
                .contains("global work/callback budgets")
        );
    }
    #[test]
    fn scarce_expression_callbacks_cover_multiple_native_writes_first() {
        use crate::composed_rule_search::Unary;
        let native = heterogeneous();
        let mut configuration = settings(1);
        configuration.max_callback_calls = 4;
        configuration.max_move_attempts = 4;
        configuration.expression_search = Some(ExpressionSettings {
            grammar: Grammar {
                arguments: 0,
                max_operations: 1,
                max_expressions: 8,
                unary: vec![Unary::Relu, Unary::Silu],
                binary: vec![],
                affine: false,
            },
            max_enumerations_per_parent: 3,
            max_move_attempts: 3,
            max_callback_calls: 3,
        });
        let mut expression_writes = vec![];
        let result = search(
            &native,
            evaluated(&Artifact::native(&native).unwrap(), &native),
            &configuration,
            &constraints(),
            |request| {
                if let Mutation::SynthesizeExpression {
                    region, expression, ..
                } = request.mutation
                {
                    assert_eq!(expression, &Expr::Argument(0));
                    expression_writes.push(region.native_write);
                }
                Err("fixture records attempts without accepting unfit hypotheses".into())
            },
        )
        .unwrap();
        assert_eq!(expression_writes, vec![1, 2, 3]);
        assert_eq!(result.report.counts.expression_callback_calls, 3);
        assert_eq!(result.report.counts.callback_calls, 4);
        assert_eq!(
            result.report.counts.expression_enumeration_sets_truncated,
            1
        );
        assert_eq!(result.report.admitted_candidates.len(), 1);
    }
}
