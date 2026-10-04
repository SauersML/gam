//! Optional generic Run backend caching native hidden states rather than vocabulary logits.
//! The native prefix and metrics remain CPU; candidate execution/head are CUDA f64.
//! This changes storage and GEMM tiling, not intervention scope or fidelity tolerances.
use super::{apply, validate_edits, Timer, Timers, Timing};
use crate::{
    acceptance::{Edit, EpisodeScore, FamilyRun, RunCheck, kl_logits},
    artifact::Artifact,
    artifact_device::Resident,
    operator_program::{Basis, Node, OperatorBody, OperatorProgram},
};
use gam_gpu::tensor::Device;
use ndarray::{Array2, ArrayView2, s};
use std::{collections::BTreeMap, sync::OnceLock};

/// Explicit numeric budgets; neither is a total process/device-memory estimate.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct StreamedBudget {
    /// Retained CPU clean/edited hidden-state arrays. Excludes prefix temporaries,
    /// operator storage, allocator overhead and the original model.
    pub teacher_bytes: usize,
    /// Four logits tiles and two hidden-width tiles, across host and device.
    /// Excludes library-private GEMM workspaces and retained forward states.
    pub tile_bytes: usize,
    /// Retained candidate intermediates, excluding the streamed vocabulary head.
    pub intermediate_bytes: usize,
}

#[derive(Clone, Copy)]
struct Head {
    hidden: usize,
    operator: usize,
    transposed: bool,
    classes: usize,
    width: usize,
}
impl Head {
    fn of(program: &OperatorProgram) -> Result<Self, String> {
        let logits = match program.nodes.get(program.output).ok_or("native output absent")? {
            Node::Readout {input,basis} if matches!(program.bases.get(*basis),Some(Basis::Indicator {..})) => *input,
            Node::Readout {..} => return Err("streamed native head requires indicator readout".into()),
            _ => program.output,
        };
        let (hidden,operator,transposed) = match program.nodes.get(logits).ok_or("native logits absent")? {
            Node::Transposed {input,operator} => (*input,*operator,true),
            Node::Affine {terms,bias:None} if terms.len()==1 => (terms[0].0,terms[0].1,false),
            _ => return Err("streamed native output must be one bias-free dense linear head".into()),
        };
        let op = program.operators.get(operator).ok_or("native head operator absent")?;
        if !matches!(op.body,OperatorBody::Dense {..}) || op.diagonal().is_some() {
            return Err("streamed native head must be dense".into());
        }
        let (classes,width) = if transposed {(op.cols.width(),op.rows.width())} else {(op.rows.width(),op.cols.width())};
        if hidden>=logits || classes==0 || width==0 || program.node_interface(hidden).map_err(|e|e.to_string())?.width()!=width {
            return Err("native head shape/order mismatch".into());
        }
        Ok(Self {hidden,operator,transposed,classes,width})
    }
    fn prefix(&self, program:&OperatorProgram) -> Result<OperatorProgram,String> {
        let mut prefix=program.clone();
        // Nodes are topologically ordered. Keep original prefix indices so edits
        // retain their identity; pruning would remap those indices.
        prefix.nodes.truncate(self.hidden+1);
        prefix.output=self.hidden;
        prefix.interfaces().map_err(|e|e.to_string())?;
        Ok(prefix)
    }
    fn logits(&self, program:&OperatorProgram, hidden:ArrayView2<'_,f64>) -> Result<Array2<f64>,String> {
        if hidden.ncols()!=self.width {return Err("cached native hidden width changed".into());}
        let matrix=program.operators[self.operator].matrix_cow();
        Ok(if self.transposed {hidden.dot(&matrix.view())} else {hidden.dot(&matrix.t())})
    }
    fn allows_edits(&self, edits:&[Edit]) -> Result<(),String> {
        if edits.iter().any(|e|e.node>self.hidden) {
            Err("streamed native teacher cannot skip a declared terminal-head intervention".into())
        } else {Ok(())}
    }
}

fn layout(head:&Head, rows:usize, episodes:usize, budget:StreamedBudget) -> Result<(usize,usize),String> {
    let teacher=episodes.checked_add(1).and_then(|n|n.checked_mul(rows)).and_then(|n|n.checked_mul(head.width)).and_then(|n|n.checked_mul(8)).ok_or("teacher byte count overflow")?;
    let per_row=head.classes.checked_mul(4).and_then(|n|head.width.checked_mul(2).and_then(|h|n.checked_add(h))).and_then(|n|n.checked_mul(8)).ok_or("head tile byte count overflow")?;
    if teacher>budget.teacher_bytes || per_row==0 || budget.tile_bytes<per_row || budget.intermediate_bytes==0 {
        return Err(format!("streamed numeric budgets insufficient: teachers={teacher}, minimum tile={per_row}"));
    }
    Ok((teacher,(budget.tile_bytes/per_row).min(rows)))
}

/// Opt-in storage backend. Unsupported head edits are rejected before execution.
/// CPU/GPU parity and GEMM tiling error are not proved by the metric's conditional
/// exp/log comparison bands; callers must report this backend and test it.
pub struct StreamedFamilyRun<'a> {
    run:&'a FamilyRun<'a>,
    device:Device,
    head:Head,
    prefix:OperatorProgram,
    budget:StreamedBudget,
    teacher_bytes:usize,
    tile_rows:usize,
    teachers:OnceLock<Result<(Array2<f64>,Vec<Array2<f64>>),String>>,
    timers:Timers,
}
impl<'a> StreamedFamilyRun<'a> {
    pub fn new(run:&'a FamilyRun<'a>,device:Device,budget:StreamedBudget)->Result<Self,String> {
        if !cfg!(target_os="linux") || device.is_host() || !device.float64() || run.family.rows==0 || run.episodes.is_empty() || run.readouts==0 {
            return Err("streamed Run requires Linux f64 CUDA and nonempty declared family".into());
        }
        let head=Head::of(run.model)?;
        if head.classes%run.readouts!=0 {return Err("native vocabulary/readout mismatch".into());}
        let interfaces=run.model.interfaces().map_err(|e|e.to_string())?;
        for episode in &run.episodes {
            validate_edits(&episode.edits,run.family.rows,|n|interfaces.get(n).map(|i|i.width()))?;
            head.allows_edits(&episode.edits)?;
        }
        let (teacher_bytes,tile_rows)=layout(&head,run.family.rows,run.episodes.len(),budget)?;
        let prefix=head.prefix(run.model)?;
        Ok(Self {run,device,head,prefix,budget,teacher_bytes,tile_rows,teachers:OnceLock::new(),timers:Timers::default()})
    }
    pub fn teacher_numeric_bytes(&self)->usize {self.teacher_bytes}
    pub fn tile_rows(&self)->usize {self.tile_rows}
    pub fn backend_name(&self)->&'static str {"hybrid: CPU native hidden teachers and tiled native head; CUDA candidate intermediates/tiled head; CPU conditional KL"}
    pub fn timing(&self)->Timing {
        let seconds=|v:&std::sync::atomic::AtomicU64|v.load(std::sync::atomic::Ordering::Relaxed) as f64*1e-9;
        Timing {native_teacher_seconds:seconds(&self.timers.teacher),candidate_construction_seconds:seconds(&self.timers.construction),cuda_forward_hooks_download_seconds:seconds(&self.timers.forward),cpu_metric_seconds:seconds(&self.timers.metric)}
    }
    fn teachers(&self)->Result<&(Array2<f64>,Vec<Array2<f64>>),String> {
        self.teachers.get_or_init(|| {
            let teacher_timer=Timer::start(&self.timers.teacher);
            let execute=|edits:&[Edit]|->Result<Array2<f64>,String>{
                let trace=self.prefix.execute_edited(&self.run.family,|node,value,_| {
                    apply(&edits.iter().filter(|e|e.node==node).collect::<Vec<_>>(),value);Ok(())
                }).map_err(|e|e.to_string())?;
                let value=trace.values[self.prefix.output].clone();
                if value.dim()!=(self.run.family.rows,self.head.width) || value.iter().any(|x|!x.is_finite()) {
                    return Err("native hidden cache shape/nonfinite mismatch".into());
                }
                Ok(value)
            };
            let result=Ok((execute(&[])?,self.run.episodes.iter().map(|e|execute(&e.edits)).collect::<Result<Vec<_>,_>>()?));
            drop(teacher_timer);result
        }).as_ref().map_err(Clone::clone)
    }
}

impl RunCheck for StreamedFamilyRun<'_> {
    fn episodes(&self,artifact:&Artifact)->Result<Vec<EpisodeScore>,String> {
        artifact.validate_coverage(self.run.model)?;
        let interfaces=artifact.program.interfaces().map_err(|e|e.to_string())?;
        if interfaces[artifact.program.output].width()!=self.head.classes {return Err("candidate output width differs from native vocabulary".into());}
        let (clean,references)=self.teachers()?;
        let resident={let timer=Timer::start(&self.timers.construction);let value=Resident::from_decoded(&self.device,artifact)?;drop(timer);value};
        let (candidate_width,classes)=resident.streamed_head_dimensions();
        if classes!=self.head.classes {return Err("compiled candidate vocabulary differs from native".into());}
        let tile_head=Head {width:self.head.width.max(candidate_width),..self.head};
        // A candidate may expose a differently sized sufficient interface. Count
        // its actual hidden tile width before any forward/head allocation.
        let per_row=tile_head.classes.checked_mul(4).and_then(|n|tile_head.width.checked_mul(2).and_then(|h|n.checked_add(h))).and_then(|n|n.checked_mul(8)).ok_or("candidate tile byte count overflow")?;
        let tile_rows=(self.budget.tile_bytes/per_row).min(self.tile_rows);
        if tile_rows==0 {return Err("candidate hidden interface exceeds declared tile budget".into());}
        let estimate=resident.estimated_intermediate_bytes(self.run.family.rows)?;
        if estimate>self.budget.intermediate_bytes {return Err(format!("streamed intermediate estimate {estimate} exceeds {}",self.budget.intermediate_bytes));}
        let mut scores=Vec::new();
        for (episode,reference) in self.run.episodes.iter().zip(references) {
            let mut held:BTreeMap<usize,Vec<&Edit>>=BTreeMap::new();let mut unheld=0;
            for e in &episode.edits {
                match artifact.place(e.node) {
                    Some(node)=>{
                        validate_edits(std::slice::from_ref(e),self.run.family.rows,|_|interfaces.get(node).map(|i|i.width()))?;
                        if resident.is_streamed_head_root(node)? {return Err("candidate maps an intervention to an unmaterialized head".into());}
                        held.entry(node).or_default().push(e);
                    },
                    None=>unheld+=1,
                }
            }
            let trace={
                let forward_timer=Timer::start(&self.timers.forward);
                let value=resident.forward_edited_intermediates(&self.run.family,|node,trace| {
                    let Some(edits)=held.get(&node) else {return Ok(None)};
                    let mut values=self.device.download(resident.root_value(trace,node)?).map_err(|e|e.to_string())?;
                    apply(edits,&mut values);
                    Ok(Some(self.device.upload(values.view()).map_err(|e|e.to_string())?))
                })?;
                drop(forward_timer);value
            };
            let (mut kl,mut error,mut effect,mut agree,mut count)=(0.,0.,0.,0.,0.);
            for start in (0..self.run.family.rows).step_by(tile_rows) {
                let end=(start+tile_rows).min(self.run.family.rows);
                let explained={let timer=Timer::start(&self.timers.forward);let value=self.device.download(&resident.logits_rows(&trace,start,end-start)?).map_err(|e|e.to_string())?;drop(timer);value};
                let metric_timer=Timer::start(&self.timers.metric);
                let z=self.head.logits(self.run.model,reference.slice(s![start..end,..]))?;
                let c=self.head.logits(self.run.model,clean.slice(s![start..end,..]))?;
                if explained.dim()!=z.dim() || z.dim()!=c.dim() {return Err("streamed output shape mismatch".into());}
                let classes=self.head.classes/self.run.readouts;
                for row in 0..z.nrows() {for readout in 0..self.run.readouts {
                    let range=readout*classes..(readout+1)*classes;
                    let (teacher,candidate,original)=(z.slice(s![row,range.clone()]),explained.slice(s![row,range.clone()]),c.slice(s![row,range]));
                    if teacher.iter().chain(candidate.iter()).chain(original.iter()).any(|v|!v.is_finite()) {return Err("nonfinite streamed output".into());}
                    let (value,rounding)=kl_logits(teacher,candidate);kl+=value;error+=rounding;effect+=kl_logits(teacher,original).0;
                    let argmax=|values:ndarray::ArrayView1<'_,f64>|values.iter().enumerate().fold((0,f64::NEG_INFINITY),|best,(i,&v)|if v>best.1{(i,v)}else{best}).0;
                    agree+=f64::from(u8::from(argmax(teacher)==argmax(candidate)));count+=1.;
                }}
                drop(metric_timer);
            }
            scores.push(EpisodeScore {id:episode.id.clone(),group:episode.group.clone(),kl:kl/count,numerical_error:(error/count).next_up(),native_effect:effect/count,top1_agree:agree/count,unheld});
        }
        Ok(scores)
    }
}

#[cfg(test)]
#[path = "streamed_family_run_tests.rs"]
mod tests;
