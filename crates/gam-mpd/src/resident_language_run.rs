//! Opt-in resident execution below LanguageRun's original imported node/action map.
//! The native teacher is the actual f64 imported program, never rounded candidate code.
use super::*;
use crate::fixed_metric_device::{
    Budget, Group, HostSpotcheck, Outcome, RawStream, Timing, mean_outcomes,
};
use gam_gpu::tensor::{Arithmetic, Device, Op, Tensor};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct ResidentBudget {
    /// Explicit cap on the declared numeric allocation plan (not total CUDA memory).
    pub aggregate_numeric_bytes: usize,
    pub teacher_bytes: usize,
    pub operator_bytes: usize,
    pub edit_bytes: usize,
    pub metric: Budget,
}
#[derive(serde::Serialize)]
pub struct ResidentEpisode {
    pub id: String,
    pub group: String,
    pub scored_from: usize,
    pub scored_until: usize,
    pub kl: Outcome,
    pub native_effect_raw: Outcome,
    pub top1_agree: f64,
    pub top1_defined: bool,
    pub unheld: usize,
    pub host_analytic_spotcheck: Option<HostSpotcheck>,
    pub metric_timing: Timing,
}
#[derive(Default, Clone, serde::Serialize)]
pub struct Transfers {
    /// Uploads during this measure call, excluding candidate preparation.
    pub edit_constant_upload_bytes: u64,
    pub residual_upload_bytes: u64,
    pub raw_oracle_download_bytes: u64,
    pub final_residual_download_bytes: u64,
    pub donor_download_bytes: u64,
}
#[derive(serde::Serialize)]
pub struct ResidentMeasure {
    pub episodes: Vec<ResidentEpisode>,
    pub groups: Vec<Group>,
    pub scope: &'static str,
    pub declared_numeric_plan_bytes: usize,
    pub teacher_numeric_bytes: usize,
    pub teacher_initialization_peak_reference_bytes: usize,
    pub native_operator_initialization_bytes: usize,
    /// Conservative retained candidate edit allocation bound; includes temporary donor rows.
    pub prepared_edit_numeric_bytes: usize,
    pub native_edit_initialization_peak_bytes: usize,
    pub candidate_preparation_upload_bytes: u64,
    pub head_numeric_resident_bytes: usize,
    pub metric_numeric_workspace_bytes: usize,
    pub transfers: Transfers,
    pub teacher_initialization_seconds: f64,
    pub preparation_seconds: f64,
    pub wall_seconds: f64,
}
struct MaskCache {
    masks: BTreeMap<String, Arc<Tensor>>,
    bytes: usize,
    limit: usize,
    uploaded: u64,
}
impl MaskCache {
    fn new(limit: usize) -> Self {
        Self {
            masks: BTreeMap::new(),
            bytes: 0,
            limit,
            uploaded: 0,
        }
    }
    fn mask(
        &mut self,
        device: &Device,
        rows: usize,
        width: usize,
        affected: Rows,
        columns: Range<usize>,
        value: f64,
        rest: f64,
    ) -> Result<Arc<Tensor>, String> {
        if columns.end > width || columns.start > columns.end || !value.is_finite() {
            return Err("resident edit outside finite interface".into());
        }
        let key = format!(
            "{rows}:{width}:{affected:?}:{}:{}:{:016x}:{:016x}",
            columns.start,
            columns.end,
            value.to_bits(),
            rest.to_bits()
        );
        if let Some(mask) = self.masks.get(&key) {
            return Ok(Arc::clone(mask));
        }
        let bytes = rows
            .checked_mul(width)
            .and_then(|n| n.checked_mul(8))
            .ok_or("resident mask overflow")?;
        if self.bytes.checked_add(bytes).is_none_or(|n| n > self.limit) {
            return Err("resident edit constants exceed declared budget".into());
        }
        let mut values = Array2::from_elem((rows, width), rest);
        for row in 0..rows {
            if affected == Rows::All || affected == Rows::One(row) {
                values.slice_mut(s![row, columns.clone()]).fill(value);
            }
        }
        let mask = Arc::new(device.upload(values.view()).map_err(|e| e.to_string())?);
        self.bytes += bytes;
        self.uploaded += bytes as u64;
        self.masks.insert(key, Arc::clone(&mask));
        Ok(mask)
    }
}
enum PreparedEdit {
    Scale(Arc<Tensor>),
    Mix {
        mask: Arc<Tensor>,
        addition: Tensor,
        alpha: f64,
    },
    Add {
        input: usize,
        left: Tensor,
        right: Tensor,
    },
}
struct Plan {
    resident: crate::artifact_device::Resident,
    family: FamilyInputs,
    residual: usize,
    edits: BTreeMap<usize, Vec<PreparedEdit>>,
    unheld: usize,
}
fn apply_resident<'a>(
    device: &Device,
    node: usize,
    root: impl Fn(usize) -> Result<&'a Tensor, String>,
    edits: &BTreeMap<usize, Vec<PreparedEdit>>,
) -> Result<Option<Tensor>, String> {
    let Some(actions) = edits.get(&node) else {
        return Ok(None);
    };
    let mut value = device.copy(root(node)?).map_err(|e| e.to_string())?;
    for action in actions {
        match action {
            PreparedEdit::Scale(mask) => {
                let mut next = device
                    .zeros(value.rows(), value.cols())
                    .map_err(|e| e.to_string())?;
                device
                    .hadamard(&mut next, &value, mask, false)
                    .map_err(|e| e.to_string())?;
                value = next;
            }
            PreparedEdit::Mix {
                mask,
                addition,
                alpha,
            } => {
                let mut next = device
                    .zeros(value.rows(), value.cols())
                    .map_err(|e| e.to_string())?;
                device
                    .hadamard(&mut next, &value, mask, false)
                    .map_err(|e| e.to_string())?;
                device
                    .axpy(&mut next, *alpha, addition)
                    .map_err(|e| e.to_string())?;
                value = next;
            }
            PreparedEdit::Add { input, left, right } => {
                let source = root(*input)?;
                let mut hidden = device
                    .zeros(source.rows(), right.cols())
                    .map_err(|e| e.to_string())?;
                device
                    .gemm(
                        &mut hidden,
                        1.,
                        source,
                        Op::N,
                        right,
                        Op::N,
                        0.,
                        Arithmetic::F64,
                    )
                    .map_err(|e| e.to_string())?;
                let mut addition = device
                    .zeros(source.rows(), left.rows())
                    .map_err(|e| e.to_string())?;
                device
                    .gemm(
                        &mut addition,
                        1.,
                        &hidden,
                        Op::N,
                        left,
                        Op::T,
                        0.,
                        Arithmetic::F64,
                    )
                    .map_err(|e| e.to_string())?;
                device
                    .axpy(&mut value, 1., &addition)
                    .map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(Some(value))
}
fn forward(device: &Device, plan: &Plan, trace_limit: usize) -> Result<Tensor, String> {
    let estimated = plan
        .resident
        .estimated_intermediate_bytes(plan.family.rows)?;
    if estimated > trace_limit {
        return Err("resident forward exceeds intermediate budget".into());
    }
    let trace = plan
        .resident
        .forward_edited_intermediates(&plan.family, |node, trace| {
            apply_resident(
                device,
                node,
                |root| plan.resident.root_value(trace, root),
                &plan.edits,
            )
        })?;
    plan.resident.take_root_value(trace, plan.residual)
}
fn donor_rows(
    device: &Device,
    resident: &crate::artifact_device::Resident,
    family: &FamilyInputs,
    wanted: &[DonorKey],
    trace_limit: usize,
) -> Result<BTreeMap<DonorKey, Tensor>, String> {
    if resident.estimated_intermediate_bytes(family.rows)? > trace_limit {
        return Err("resident donor exceeds trace budget".into());
    }
    let trace = resident.forward_edited_intermediates(family, |_, _| Ok(None))?;
    wanted
        .iter()
        .map(|key| {
            Ok((
                *key,
                device
                    .rows_of(resident.root_value(&trace, key.node)?, key.row, 1)
                    .map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}
fn prepare_edits(
    device: &Device,
    artifact: &Artifact,
    rows: usize,
    edits: &BTreeMap<usize, Vec<NodeEdit>>,
    donor: &BTreeMap<DonorKey, Tensor>,
    cache: &mut MaskCache,
) -> Result<BTreeMap<usize, Vec<PreparedEdit>>, String> {
    let mut prepared = BTreeMap::new();
    for (node, actions) in edits {
        let width = artifact
            .program
            .node_interface(*node)
            .map_err(|e| e.to_string())?
            .width();
        let mut list = Vec::new();
        for action in actions {
            match action {
                NodeEdit::Scale {
                    rows: affected,
                    columns,
                    scale,
                } => list.push(PreparedEdit::Scale(cache.mask(
                    device,
                    rows,
                    width,
                    *affected,
                    columns.clone(),
                    *scale,
                    1.,
                )?)),
                NodeEdit::Mix {
                    row,
                    columns,
                    alpha,
                    donor: key,
                } => {
                    if *row >= rows || !alpha.is_finite() {
                        return Err("resident mix outside finite domain".into());
                    }
                    let d = donor.get(key).ok_or("resident donor absent")?;
                    if d.dim() != (1, columns.len()) {
                        return Err("resident donor width mismatch".into());
                    }
                    let mask = cache.mask(
                        device,
                        rows,
                        width,
                        Rows::One(*row),
                        columns.clone(),
                        1. - alpha,
                        1.,
                    )?;
                    let active = cache.mask(
                        device,
                        rows,
                        width,
                        Rows::One(*row),
                        columns.clone(),
                        1.,
                        0.,
                    )?;
                    let mut wide = device.zeros(1, width).map_err(|e| e.to_string())?;
                    device
                        .set_columns(&mut wide, columns.start, d)
                        .map_err(|e| e.to_string())?;
                    let repeated = device
                        .broadcast_rows(&wide, rows)
                        .map_err(|e| e.to_string())?;
                    let mut addition = device.zeros(rows, width).map_err(|e| e.to_string())?;
                    // Preserve the original multiply-by-zero/sign semantics on other rows.
                    device
                        .hadamard(&mut addition, &repeated, &active, false)
                        .map_err(|e| e.to_string())?;
                    let bytes = addition.bytes();
                    if cache
                        .bytes
                        .checked_add(bytes)
                        .is_none_or(|n| n > cache.limit)
                    {
                        return Err("resident donor additions exceed edit budget".into());
                    }
                    cache.bytes += bytes;
                    list.push(PreparedEdit::Mix {
                        mask,
                        addition,
                        alpha: *alpha,
                    });
                }
                NodeEdit::AddMap { input, left, right } => {
                    let bytes = left
                        .len()
                        .checked_add(right.len())
                        .and_then(|n| n.checked_mul(8))
                        .ok_or("resident delta overflow")?;
                    if cache
                        .bytes
                        .checked_add(bytes)
                        .is_none_or(|n| n > cache.limit)
                    {
                        return Err("resident deltas exceed edit budget".into());
                    }
                    cache.bytes += bytes;
                    cache.uploaded += bytes as u64;
                    list.push(PreparedEdit::Add {
                        input: *input,
                        left: device.upload(left.view()).map_err(|e| e.to_string())?,
                        right: device.upload(right.view()).map_err(|e| e.to_string())?,
                    });
                }
            }
        }
        prepared.insert(*node, list);
    }
    Ok(prepared)
}
fn prepare_plans(
    run: &LanguageRun<'_>,
    artifact: &Artifact,
    base: &crate::artifact_device::Resident,
    residuals: &[usize],
    values: bool,
    budget: ResidentBudget,
) -> Result<(Vec<Plan>, usize, u64), String> {
    let device = run.device.as_ref().ok_or("resident CUDA backend absent")?;
    let (planned, wanted) = run.resident_plans(artifact, residuals)?;
    let mut donors = BTreeMap::new();
    let mut donor_bytes = 0usize;
    for (passage, keys) in wanted {
        let rows = donor_rows(
            device,
            base,
            &run.family(passage),
            &keys,
            run.trace_bytes_limit,
        )?;
        donor_bytes = rows.values().try_fold(donor_bytes, |n, t| {
            n.checked_add(t.bytes()).ok_or("resident donor overflow")
        })?;
        if donor_bytes > budget.edit_bytes {
            return Err("resident donor cache exceeds edit budget".into());
        }
        donors.insert(passage, rows);
    }
    let mut cache = MaskCache::new(budget.edit_bytes - donor_bytes);
    let empty = BTreeMap::new();
    let mut plans = Vec::new();
    for (episode, (program, residuals, edits, unheld)) in run.spec.episodes.iter().zip(planned) {
        let resident = if values {
            crate::artifact_device::Resident::from_decoded_values_sharing_bounded(
                base,
                &program,
                budget.operator_bytes,
            )?
        } else {
            crate::artifact_device::Resident::from_decoded_sharing(base, &program)?
        };
        let own = episode.donor.and_then(|d| donors.get(&d)).unwrap_or(&empty);
        let edits = prepare_edits(device, &program, run.spec.rows, &edits, own, &mut cache)?;
        plans.push(Plan {
            resident,
            family: run.family(episode.passage),
            residual: *residuals.last().ok_or("resident final place absent")?,
            edits,
            unheld,
        });
    }
    Ok((plans, cache.bytes + donor_bytes, cache.uploaded))
}
pub(super) struct Teachers {
    references: Vec<Tensor>,
    effects: Vec<Outcome>,
    budget: ResidentBudget,
    numeric_bytes: usize,
    peak_reference_bytes: usize,
    operator_bytes: usize,
    edit_bytes: usize,
    edit_uploaded: u64,
    seconds: f64,
}
// Conservative sums of configured limits. Three operator allowances cover imported
// sharing source, true-f64 teacher and candidate; edit limits can overlap cold.
// Attention/library workspaces, edit-construction scratch and allocator overhead
// are not enclosed by this numeric plan and must retain separate device headroom.
fn numeric_plan_bytes(budget: ResidentBudget, trace: usize, head: usize) -> Result<usize, String> {
    [
        budget.teacher_bytes,
        budget.operator_bytes,
        budget.operator_bytes,
        budget.operator_bytes,
        budget.edit_bytes,
        budget.edit_bytes,
        budget.metric.workspace_bytes,
        trace,
        head,
    ]
    .into_iter()
    .try_fold(0usize, |n, v| {
        n.checked_add(v)
            .ok_or_else(|| "resident aggregate byte overflow".into())
    })
}
fn same_budget(a: ResidentBudget, b: ResidentBudget) -> bool {
    a.aggregate_numeric_bytes == b.aggregate_numeric_bytes
        && a.teacher_bytes == b.teacher_bytes
        && a.operator_bytes == b.operator_bytes
        && a.edit_bytes == b.edit_bytes
        && a.metric.batch_rows == b.metric.batch_rows
        && a.metric.workspace_bytes == b.metric.workspace_bytes
}
fn raw_rows(
    run: &LanguageRun<'_>,
    p: &Tensor,
    q: &Tensor,
    from: usize,
    budget: Budget,
    oracle: bool,
) -> Result<crate::fixed_metric_device::RawEvidence, String> {
    let head = run.native_readout.as_ref().ok_or("resident head absent")?;
    if p.dim() != q.dim() || from >= p.rows() {
        return Err("resident metric domain mismatch".into());
    }
    let guard = run
        .readout_lock
        .lock()
        .map_err(|_| "resident paired head lock poisoned")?;
    let mut stream = RawStream::new(oracle);
    let mut start = from;
    while start < p.rows() {
        let rows = budget.batch_rows.min(p.rows() - start);
        stream.append(
            head.checked_raw_metrics_device(p, q, start, rows, budget.workspace_bytes, oracle)?,
            start,
        )?;
        start += rows;
    }
    drop(guard);
    stream.finish_evidence(from, p.rows())
}
impl<'a> LanguageRun<'a> {
    fn resident_teacher_cache(&self, budget: ResidentBudget) -> Result<&Teachers, String> {
        let teachers = self
            .resident_teachers
            .get_or_init(|| {
                let timer = std::time::Instant::now();
                let device = self
                    .device
                    .as_ref()
                    .ok_or("resident teacher CUDA backend absent")?;
                let last = self
                    .layers
                    .last()
                    .ok_or("resident native layer absent")?
                    .residual;
                let actual = Artifact::native(self.native)?.truncated(last)?;
                let residuals: Vec<_> = self
                    .layers
                    .iter()
                    .map(|layer| {
                        actual
                            .place(layer.residual)
                            .ok_or("resident native place absent")
                    })
                    .collect::<Result<_, _>>()?;
                let width = actual
                    .program
                    .node_interface(*residuals.last().ok_or("native final place absent")?)
                    .map_err(|e| e.to_string())?
                    .width();
                let passage_count = self
                    .spec
                    .episodes
                    .iter()
                    .map(|e| e.passage)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len();
                let required = self
                    .spec
                    .episodes
                    .len()
                    .checked_add(passage_count)
                    .and_then(|n| n.checked_mul(self.spec.rows))
                    .and_then(|n| n.checked_mul(width))
                    .and_then(|n| n.checked_mul(8))
                    .ok_or("resident teacher cache overflow")?;
                if required > budget.teacher_bytes {
                    return Err(format!(
                        "resident native cache {required} exceeds {}",
                        budget.teacher_bytes
                    ));
                }
                let base = crate::artifact_device::Resident::from_decoded_values_bounded(
                    device,
                    &actual,
                    budget.operator_bytes,
                )?;
                let operator_bytes = base.operator_numeric_bytes()?;
                let (plans, edit_bytes, edit_uploaded) =
                    prepare_plans(self, &actual, &base, &residuals, true, budget)?;
                if plans.iter().any(|p| p.unheld != 0) {
                    return Err("native CUDA teacher omitted a declared interface".into());
                }
                let mut clean = BTreeMap::new();
                for passage in self
                    .spec
                    .episodes
                    .iter()
                    .map(|e| e.passage)
                    .collect::<std::collections::BTreeSet<_>>()
                {
                    let trace =
                        base.forward_edited_intermediates(&self.family(passage), |_, _| Ok(None))?;
                    clean.insert(
                        passage,
                        base.take_root_value(
                            trace,
                            *residuals.last().ok_or("native final place absent")?,
                        )?,
                    );
                }
                let mut references = Vec::new();
                let mut effects = Vec::new();
                for (episode, plan) in self.spec.episodes.iter().zip(&plans) {
                    let reference = forward(device, plan, self.trace_bytes_limit)?;
                    let from = episode
                        .actions
                        .iter()
                        .map(Action::first_row)
                        .min()
                        .unwrap_or(0);
                    let effect = raw_rows(
                        self,
                        &reference,
                        clean
                            .get(&episode.passage)
                            .ok_or("native clean cache absent")?,
                        from,
                        budget.metric,
                        false,
                    )?
                    .kl;
                    references.push(reference);
                    effects.push(effect);
                }
                let numeric_bytes = references.iter().map(Tensor::bytes).sum();
                Ok(Teachers {
                    references,
                    effects,
                    budget,
                    numeric_bytes,
                    peak_reference_bytes: required,
                    operator_bytes,
                    edit_bytes,
                    edit_uploaded,
                    seconds: timer.elapsed().as_secs_f64(),
                })
            })
            .as_ref()
            .map_err(Clone::clone)?;
        if !same_budget(teachers.budget, budget) {
            return Err("resident native cache belongs to another declared budget".into());
        }
        Ok(teachers)
    }
    /// Prepare immutable compiled episode graphs and device edit constants once.
    /// The borrow prevents mutation of the artifact while its prepared executable is reused.
    pub fn prepare_resident_candidate<'r, 'p>(
        &'r self,
        artifact: &'p Artifact,
        budget: ResidentBudget,
    ) -> Result<PreparedResidentCandidate<'r, 'p, 'a>, String> {
        let timer = std::time::Instant::now();
        if budget.teacher_bytes == 0 || budget.operator_bytes == 0 || budget.edit_bytes == 0 {
            return Err("resident budgets must be positive".into());
        }
        let head = self
            .native_readout
            .as_ref()
            .ok_or("resident candidate needs explicit native CUDA readout")?;
        if head.checked_raw_workspace_bytes(budget.metric.batch_rows)?
            > budget.metric.workspace_bytes
        {
            return Err("resident metric budget insufficient".into());
        }
        let planned = numeric_plan_bytes(budget, self.trace_bytes_limit, head.resident_bytes())?;
        if planned > budget.aggregate_numeric_bytes {
            return Err(format!(
                "declared numeric plan {planned} exceeds aggregate budget {}",
                budget.aggregate_numeric_bytes
            ));
        }
        let device = self.device.as_ref().ok_or("resident CUDA backend absent")?;
        let (validated, _) = self.truncated(artifact)?;
        let last = self.layers.last().ok_or("resident layer absent")?.residual;
        let mut program = validated.truncated(last)?;
        let residuals: Vec<_> = self
            .layers
            .iter()
            .map(|layer| {
                program
                    .place(layer.residual)
                    .ok_or("candidate residual absent")
            })
            .collect::<Result<_, _>>()?;
        let base = match &self.native_device_source {
            Some(source) => {
                source.interner.intern(&mut program);
                crate::artifact_device::Resident::from_decoded_values_sharing_bounded(
                    &source.resident,
                    &program,
                    budget.operator_bytes,
                )?
            }
            None => crate::artifact_device::Resident::from_decoded_values_bounded(
                device,
                &program,
                budget.operator_bytes,
            )?,
        };
        if base.operator_numeric_bytes()? > budget.operator_bytes {
            return Err("resident candidate operators exceed budget".into());
        }
        let (plans, edit_bytes, edit_uploaded) =
            prepare_plans(self, &program, &base, &residuals, true, budget)?;
        Ok(PreparedResidentCandidate {
            run: self,
            artifact,
            budget,
            plans,
            edit_bytes,
            edit_uploaded,
            preparation_seconds: timer.elapsed().as_secs_f64(),
        })
    }
}
/// One immutable candidate, with compiled episode graphs and GPU edit constants retained.
pub struct PreparedResidentCandidate<'r, 'p, 'a> {
    run: &'r LanguageRun<'a>,
    artifact: &'p Artifact,
    budget: ResidentBudget,
    plans: Vec<Plan>,
    edit_bytes: usize,
    edit_uploaded: u64,
    preparation_seconds: f64,
}
impl PreparedResidentCandidate<'_, '_, '_> {
    pub fn measure_resident(&self, host_spotchecks: bool) -> Result<ResidentMeasure, String> {
        let timer = std::time::Instant::now();
        let head = self
            .run
            .native_readout
            .as_ref()
            .ok_or("resident head absent")?;
        let before = head.raw_transfer_counts();
        let teacher_was_ready = self.run.resident_teachers.get().is_some();
        let teachers = self.run.resident_teacher_cache(self.budget)?;
        let device = self
            .run
            .device
            .as_ref()
            .ok_or("resident CUDA backend absent")?;
        let mut episodes = Vec::new();
        if teachers.references.len() != self.plans.len()
            || self.plans.len() != self.run.spec.episodes.len()
        {
            return Err("resident native episode count mismatch".into());
        }
        for (index, (episode, plan)) in self.run.spec.episodes.iter().zip(&self.plans).enumerate() {
            let explained = forward(device, plan, self.run.trace_bytes_limit)?;
            let from = episode
                .actions
                .iter()
                .map(Action::first_row)
                .min()
                .unwrap_or(0);
            let evidence = raw_rows(
                self.run,
                &teachers.references[index],
                &explained,
                from,
                self.budget.metric,
                host_spotchecks,
            )?;
            episodes.push(ResidentEpisode {
                id: episode.id.clone(),
                group: episode.group.clone(),
                scored_from: from,
                scored_until: explained.rows(),
                kl: evidence.kl,
                native_effect_raw: teachers.effects[index].clone(),
                top1_agree: evidence.top1_agree,
                top1_defined: evidence.top1_defined,
                unheld: plan.unheld,
                host_analytic_spotcheck: evidence.host_analytic_spotcheck,
                metric_timing: evidence.metric_timing,
            });
        }
        let mut grouped = BTreeMap::<String, Vec<&Outcome>>::new();
        for episode in &episodes {
            grouped
                .entry(episode.group.clone())
                .or_default()
                .push(&episode.kl);
        }
        let groups = grouped
            .into_iter()
            .map(|(name, values)| Group {
                name,
                episodes: values.len(),
                mean_kl: mean_outcomes(&values),
            })
            .collect();
        let after = head.raw_transfer_counts();
        Ok(ResidentMeasure {
            episodes,
            groups,
            scope: "True imported f64 native teacher and candidate execute mapped interventions on CUDA. Neural residuals/donors stay resident. Checked intervals certify exact resulting raw binary64 logits only, excluding network/RMS/gain/GEMM rounding. GPU native effects are independently labelled raw-logit intervals; no CPU-effect substitution. Warm metric residual/vocabulary transfers zero with host oracle disabled. Token/layout/edit preparation is retained per immutable prepared candidate; forward intermediates/workspaces remain dynamic. Teacher operator/edit byte counts are initialization allocations, dropped after references are cached; teacher_numeric_bytes counts retained reference tensors, peak-reference bytes additionally includes transient clean references. Edit budget is per preparation (native and candidate may overlap during initialization). Numeric buffers exclude CUDA context, cuBLAS workspace, kernel registers/spills and allocator overhead.",
            declared_numeric_plan_bytes: numeric_plan_bytes(
                self.budget,
                self.run.trace_bytes_limit,
                head.resident_bytes(),
            )?,
            teacher_numeric_bytes: teachers.numeric_bytes,
            teacher_initialization_peak_reference_bytes: teachers.peak_reference_bytes,
            native_operator_initialization_bytes: teachers.operator_bytes,
            prepared_edit_numeric_bytes: self.edit_bytes,
            native_edit_initialization_peak_bytes: teachers.edit_bytes,
            candidate_preparation_upload_bytes: self.edit_uploaded,
            head_numeric_resident_bytes: head.resident_bytes(),
            metric_numeric_workspace_bytes: head
                .checked_raw_workspace_bytes(self.budget.metric.batch_rows)?,
            transfers: Transfers {
                edit_constant_upload_bytes: if teacher_was_ready {
                    0
                } else {
                    teachers.edit_uploaded
                },
                residual_upload_bytes: after.0 - before.0,
                raw_oracle_download_bytes: after.1 - before.1,
                final_residual_download_bytes: 0,
                donor_download_bytes: 0,
            },
            teacher_initialization_seconds: if teacher_was_ready {
                0.
            } else {
                teachers.seconds
            },
            preparation_seconds: self.preparation_seconds,
            wall_seconds: timer.elapsed().as_secs_f64(),
        })
    }
}
fn band(outcome: &Outcome) -> Result<(f64, f64), String> {
    let Outcome::Bounded { lower, upper } = outcome else {
        return Err("resident raw evidence unresolved".into());
    };
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err("resident evidence invalid".into());
    }
    if lower == upper {
        return Ok((*lower, 0.));
    }
    let midpoint = lower / 2. + upper / 2.;
    let error = (midpoint - lower).max(upper - midpoint).next_up();
    Ok((midpoint, error))
}
fn run_measure(measure: &ResidentMeasure) -> Result<crate::acceptance::RunMeasure, String> {
    let mut episodes = Vec::new();
    for episode in &measure.episodes {
        let (kl, numerical_error) = band(&episode.kl)?;
        let (native_effect, _) = band(&episode.native_effect_raw)?;
        if !episode.top1_defined {
            return Err("resident top1 undefined".into());
        }
        episodes.push(EpisodeScore {
            id: episode.id.clone(),
            group: episode.group.clone(),
            kl,
            numerical_error,
            native_effect,
            top1_agree: episode.top1_agree,
            unheld: episode.unheld,
        });
    }
    let mut groups = Vec::new();
    for group in &measure.groups {
        let related: Vec<_> = measure
            .episodes
            .iter()
            .filter(|e| e.group == group.name)
            .collect();
        if related.len() != group.episodes || related.is_empty() {
            return Err("resident group domain mismatch".into());
        }
        let (value, error) = band(&group.mean_kl)?;
        let effects: Vec<_> = related.iter().map(|e| &e.native_effect_raw).collect();
        let (effect, _) = band(&mean_outcomes(&effects))?;
        groups.push((group.name.clone(), value, error, effect, group.episodes));
    }
    Ok(crate::acceptance::RunMeasure { episodes, groups })
}
impl RunCheck for PreparedResidentCandidate<'_, '_, '_> {
    fn episodes(&self, artifact: &Artifact) -> Result<Vec<EpisodeScore>, String> {
        Ok(self.measure(artifact)?.episodes)
    }
    fn measure(&self, artifact: &Artifact) -> Result<crate::acceptance::RunMeasure, String> {
        if !std::ptr::eq(artifact, self.artifact) {
            return Err("prepared resident candidate identity differs".into());
        }
        run_measure(&self.measure_resident(false)?)
    }
}

/// Acceptance adapter: prepare precisely the decoded candidate passed by assessment.
/// CUDA execution and fixed-logit bounds remain opt-in; unresolved rows return an error.
pub struct ResidentLanguageRun<'r, 'a> {
    run: &'r LanguageRun<'a>,
    budget: ResidentBudget,
    telemetry: std::sync::Mutex<ResidentTelemetry>,
}
impl<'a> LanguageRun<'a> {
    pub fn resident_run(&self, budget: ResidentBudget) -> ResidentLanguageRun<'_, 'a> {
        ResidentLanguageRun {
            run: self,
            budget,
            telemetry: std::sync::Mutex::new(ResidentTelemetry::default()),
        }
    }
}
#[derive(Clone, Default, serde::Serialize)]
pub struct ResidentTelemetry {
    pub calls: usize,
    pub completed: usize,
    pub errors: usize,
    pub preparation_seconds: f64,
    pub teacher_initialization_seconds: f64,
    pub measure_wall_seconds: f64,
    pub transfers: Transfers,
    pub metric_timing: Timing,
    pub latest_teacher_numeric_bytes: usize,
    pub latest_prepared_edit_numeric_bytes: usize,
    pub latest_head_numeric_resident_bytes: usize,
    pub latest_metric_workspace_bytes: usize,
}
impl ResidentLanguageRun<'_, '_> {
    pub fn telemetry(&self) -> Result<ResidentTelemetry, String> {
        Ok(self
            .telemetry
            .lock()
            .map_err(|_| "resident telemetry lock poisoned")?
            .clone())
    }
    pub fn backend_name(&self) -> &'static str {
        "CUDA f64 teacher/candidate/head; analytic fixed-raw-logit intervals; device reference/donor cache"
    }
}
impl RunCheck for ResidentLanguageRun<'_, '_> {
    fn episodes(&self, artifact: &Artifact) -> Result<Vec<EpisodeScore>, String> {
        Ok(self.measure(artifact)?.episodes)
    }
    fn measure(&self, artifact: &Artifact) -> Result<crate::acceptance::RunMeasure, String> {
        let result = (|| {
            let prepared = self.run.prepare_resident_candidate(artifact, self.budget)?;
            let report = prepared.measure_resident(false)?;
            let measure = run_measure(&report)?;
            Ok::<_, String>((measure, report))
        })();
        let mut telemetry = self
            .telemetry
            .lock()
            .map_err(|_| "resident telemetry lock poisoned")?;
        telemetry.calls += 1;
        match result {
            Ok((measure, report)) => {
                telemetry.completed += 1;
                telemetry.preparation_seconds += report.preparation_seconds;
                telemetry.teacher_initialization_seconds += report.teacher_initialization_seconds;
                telemetry.measure_wall_seconds += report.wall_seconds;
                telemetry.transfers.edit_constant_upload_bytes +=
                    report.transfers.edit_constant_upload_bytes
                        + report.candidate_preparation_upload_bytes;
                telemetry.transfers.residual_upload_bytes += report.transfers.residual_upload_bytes;
                telemetry.transfers.raw_oracle_download_bytes +=
                    report.transfers.raw_oracle_download_bytes;
                telemetry.transfers.final_residual_download_bytes +=
                    report.transfers.final_residual_download_bytes;
                telemetry.transfers.donor_download_bytes += report.transfers.donor_download_bytes;
                for episode in &report.episodes {
                    let t = &episode.metric_timing;
                    let total = &mut telemetry.metric_timing;
                    total.packing_and_upload_seconds += t.packing_and_upload_seconds;
                    total.checked_metric_seconds += t.checked_metric_seconds;
                    total.cpu_reference_seconds += t.cpu_reference_seconds;
                    total.cpu_top1_seconds += t.cpu_top1_seconds;
                    total.resident_head_seconds += t.resident_head_seconds;
                    total.gpu_top1_seconds += t.gpu_top1_seconds;
                    total.raw_oracle_download_seconds += t.raw_oracle_download_seconds;
                }
                telemetry.latest_teacher_numeric_bytes = report.teacher_numeric_bytes;
                telemetry.latest_prepared_edit_numeric_bytes = report.prepared_edit_numeric_bytes;
                telemetry.latest_head_numeric_resident_bytes = report.head_numeric_resident_bytes;
                telemetry.latest_metric_workspace_bytes = report.metric_numeric_workspace_bytes;
                Ok(measure)
            }
            Err(error) => {
                telemetry.errors += 1;
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_device_edits_match_existing_order_and_partial_mix() {
        use crate::operator_program::{Declarations, Interface, Operator, Slot};
        let program = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }],
                parameters: 0,
            },
            operators: vec![Arc::new(Operator::identity(
                "id",
                Interface::native(2).unwrap(),
            ))],
            bases: vec![],
            rules: vec![],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
            ],
            output: 1,
        };
        let artifact = Artifact::native(&program).unwrap();
        let device = Device::host();
        let input = ndarray::array![[0.2, -0.7], [1.1, 0.4]];
        let value = ndarray::array![[3.4, -0.0], [-0.3, 1.7]];
        let key = DonorKey { node: 7, row: 1 };
        let d = ndarray::array![[2.3]];
        let edits = BTreeMap::from([(
            1,
            vec![
                NodeEdit::Scale {
                    rows: Rows::One(0),
                    columns: 0..1,
                    scale: 0.3,
                },
                NodeEdit::Mix {
                    row: 1,
                    columns: 1..2,
                    alpha: 0.4,
                    donor: key,
                },
                NodeEdit::AddMap {
                    input: 0,
                    left: Arc::new(ndarray::array![[0.6], [-0.2]]),
                    right: Arc::new(ndarray::array![[0.3], [0.8]]),
                },
            ],
        )]);
        let roots = [
            device.upload(input.view()).unwrap(),
            device.upload(value.view()).unwrap(),
        ];
        let host_donor = BTreeMap::from([(key, d.row(0).to_owned())]);
        let expected =
            apply_device_node_edits(&device, 1, 2, |n| Ok(&roots[n]), &edits, &host_donor)
                .unwrap()
                .unwrap();
        let donor = BTreeMap::from([(key, device.upload(d.view()).unwrap())]);
        let mut cache = MaskCache::new(4096);
        let prepared = prepare_edits(&device, &artifact, 2, &edits, &donor, &mut cache).unwrap();
        let actual = apply_resident(&device, 1, |n| Ok(&roots[n]), &prepared)
            .unwrap()
            .unwrap();
        let expected = device.download(&expected).unwrap();
        let actual = device.download(&actual).unwrap();
        assert_eq!(expected.mapv(f64::to_bits), actual.mapv(f64::to_bits));
        let uploaded = cache.uploaded;
        let reused = cache
            .mask(&device, 2, 2, Rows::One(0), 0..1, 0.3, 1.)
            .unwrap();
        assert_eq!(cache.uploaded, uploaded);
        assert_eq!(reused.dim(), (2, 2));
        assert!(
            prepare_edits(
                &device,
                &artifact,
                2,
                &edits,
                &donor,
                &mut MaskCache::new(1)
            )
            .is_err()
        );
    }
    #[test]
    fn conversion_outward_encloses_and_rejects_unknown() {
        for (lower, upper) in [
            (0., 0.),
            (0.029781106383897322, 0.029781106383984228),
            (f64::from_bits(1), f64::from_bits(3)),
        ] {
            let (value, error) = band(&Outcome::Bounded { lower, upper }).unwrap();
            assert!(value - error <= lower && value + error >= upper);
        }
        assert!(
            band(&Outcome::Unresolved {
                reason: "overflow".into()
            })
            .is_err()
        );
        assert!(
            band(&Outcome::Bounded {
                lower: f64::NAN,
                upper: 1.
            })
            .is_err()
        );
    }
}
