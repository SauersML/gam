//! Optional generic Run backend caching native hidden states rather than vocabulary logits.
//! Native prefix/head execution is optionally CUDA; comparison metrics remain CPU.
//! This changes storage and GEMM tiling, not intervention scope or fidelity tolerances.
use super::{apply, validate_edits, Timer, Timers, Timing};
use crate::{
    acceptance::{Edit, EpisodeScore, FamilyRun, RunCheck, kl_logits},
    artifact::Artifact,
    artifact_device::Resident,
    operator_program::{Basis, Node, OperatorBody, OperatorProgram},
};
use gam_gpu::tensor::{Arithmetic, Device, Op, Tensor};
use ndarray::{Array2, ArrayView2, s};
use std::sync::OnceLock;

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
    fn logits_device(&self,device:&Device,head:&Tensor,hidden:ArrayView2<'_,f64>)->Result<Array2<f64>,String> {
        if hidden.ncols()!=self.width {return Err("native CUDA hidden width mismatch".into());}
        let input=device.upload(hidden).map_err(|e|e.to_string())?;
        let mut output=device.zeros(hidden.nrows(),self.classes).map_err(|e|e.to_string())?;
        device.gemm(&mut output,1.,&input,Op::N,head,if self.transposed {Op::N}else{Op::T},0.,Arithmetic::F64).map_err(|e|e.to_string())?;
        device.download(&output).map_err(|e|e.to_string())
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

struct CudaNative {
    prefix:Resident,
    head:Tensor,
    numeric_bytes:usize,
    budget:usize,
    initialization_seconds:f64,
    prefix_seconds:std::sync::atomic::AtomicU64,
    head_seconds:std::sync::atomic::AtomicU64,
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
    cuda_native:Option<CudaNative>,
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
        Ok(Self {run,device,head,prefix,budget,teacher_bytes,tile_rows,teachers:OnceLock::new(),timers:Timers::default(),cuda_native:None})
    }
    /// Optional actual-f64 native prefix/head execution. The budget counts retained
    /// numeric operator buffers only, excluding traces, tiles and CUDA workspaces.
    pub fn with_cuda_native(mut self,numeric_bytes:usize)->Result<Self,String> {
        if self.cuda_native.is_some() || self.teachers.get().is_some() {return Err("native CUDA must be configured before teacher initialization".into());}
        for episode in &self.run.episodes {
            if episode.edits.iter().any(|e|matches!(self.prefix.nodes.get(e.node),Some(Node::Feature {..}))) {
                return Err("native CUDA cannot edit an implicit feature node".into());
            }
        }
        let begun=std::time::Instant::now();
        let head_bytes=self.head.classes.checked_mul(self.head.width).and_then(|n|n.checked_mul(8)).ok_or("native head byte overflow")?;
        let remaining=numeric_bytes.checked_sub(head_bytes).ok_or("native CUDA head exceeds numeric budget")?;
        let prefix=Resident::from_decoded_values_bounded(&self.device,&Artifact::native(&self.prefix).map_err(|e|e.to_string())?,remaining)?;
        if prefix.estimated_intermediate_bytes(self.run.family.rows)?>self.budget.intermediate_bytes {return Err("native CUDA prefix exceeds intermediate budget".into());}
        let retained=prefix.operator_numeric_bytes()?.checked_add(head_bytes).ok_or("native CUDA byte overflow")?;
        let matrix=self.run.model.operators[self.head.operator].matrix_cow();
        let head=self.device.upload(matrix.view()).map_err(|e|e.to_string())?;
        self.cuda_native=Some(CudaNative {prefix,head,numeric_bytes:retained,budget:numeric_bytes,initialization_seconds:begun.elapsed().as_secs_f64(),prefix_seconds:Default::default(),head_seconds:Default::default()});
        Ok(self)
    }
    pub fn native_cuda_report(&self)->Option<serde_json::Value> {
        self.cuda_native.as_ref().map(|c|serde_json::json!({"retained_numeric_bytes":c.numeric_bytes,"numeric_budget_bytes":c.budget,"initialization_seconds":c.initialization_seconds,"excludes":"traces, hidden caches, tiles, allocator and GEMM workspaces","native_values":"original f64; no codec rounding","prefix_forward_seconds":c.prefix_seconds.load(std::sync::atomic::Ordering::Relaxed) as f64*1e-9,"head_seconds":c.head_seconds.load(std::sync::atomic::Ordering::Relaxed) as f64*1e-9,"intervention_hooks":"CPU ordered edits on original native nodes"}))
    }
    fn native_logits(&self,hidden:ArrayView2<'_,f64>)->Result<Array2<f64>,String> {
        match &self.cuda_native {
            None=>self.head.logits(self.run.model,hidden),
            Some(cuda)=>{
                let head_timer=Timer::start(&cuda.head_seconds);
                let value=self.head.logits_device(&self.device,&cuda.head,hidden);
                drop(head_timer);value
            }
        }
    }
    pub fn teacher_numeric_bytes(&self)->usize {self.teacher_bytes}
    pub fn tile_rows(&self)->usize {self.tile_rows}
    pub fn backend_name(&self)->&'static str {if self.cuda_native.is_some() {"CUDA native/candidate prefixes and tiled heads; CPU ordered intervention hooks and conditional KL"}else{"hybrid: CPU native hidden teachers and tiled native head; CUDA candidate intermediates/tiled head; CPU conditional KL"}}
    pub fn timing(&self)->Timing {
        let seconds=|v:&std::sync::atomic::AtomicU64|v.load(std::sync::atomic::Ordering::Relaxed) as f64*1e-9;
        Timing {native_teacher_seconds:seconds(&self.timers.teacher),candidate_construction_seconds:seconds(&self.timers.construction),cuda_forward_hooks_download_seconds:seconds(&self.timers.forward),cpu_metric_seconds:seconds(&self.timers.metric)}
    }
    fn teachers(&self)->Result<&(Array2<f64>,Vec<Array2<f64>>),String> {
        self.teachers.get_or_init(|| {
            let teacher_timer=Timer::start(&self.timers.teacher);
            let execute=|edits:&[Edit]|->Result<Array2<f64>,String>{
                if let Some(cuda)=&self.cuda_native {
                    let prefix_timer=Timer::start(&cuda.prefix_seconds);
                    let trace=cuda.prefix.forward_edited(&self.run.family,|node,trace| {
                        let edits=edits.iter().filter(|e|e.node==node).collect::<Vec<_>>();
                        if edits.is_empty() {return Ok(None);}
                        let mut value=self.device.download(cuda.prefix.root_value(trace,node)?).map_err(|e|e.to_string())?;
                        apply(&edits,&mut value);
                        Ok(Some(self.device.upload(value.view()).map_err(|e|e.to_string())?))
                    })?;
                    let value=self.device.download(&cuda.prefix.output(&trace)?).map_err(|e|e.to_string())?;
                    if value.dim()!=(self.run.family.rows,self.head.width) || value.iter().any(|x|!x.is_finite()) {return Err("native CUDA hidden cache shape/nonfinite mismatch".into());}
                    drop(prefix_timer);
                    return Ok(value);
                }
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
            let mapped=crate::native_control::map_edits(artifact,&episode.edits)?;
            let held=mapped.edits;
            let unheld=mapped.unheld;
            for (&node,edits) in &held {
                validate_edits(edits,self.run.family.rows,|_|interfaces.get(node).map(|i|i.width()))?;
                if resident.is_streamed_head_root(node)? {return Err("candidate maps an intervention to an unmaterialized head".into());}
            }
            let trace={
                let forward_timer=Timer::start(&self.timers.forward);
                let value=resident.forward_edited_intermediates(&self.run.family,|node,trace| {
                    let Some(edits)=held.get(&node) else {return Ok(None)};
                    let mut values=self.device.download(resident.root_value(trace,node)?).map_err(|e|e.to_string())?;
                    apply(&edits.iter().collect::<Vec<_>>(),&mut values);
                    Ok(Some(self.device.upload(values.view()).map_err(|e|e.to_string())?))
                })?;
                drop(forward_timer);value
            };
            let (mut kl,mut error,mut effect,mut agree,mut count)=(0.,0.,0.,0.,0.);
            for start in (0..self.run.family.rows).step_by(tile_rows) {
                let end=(start+tile_rows).min(self.run.family.rows);
                let explained={let timer=Timer::start(&self.timers.forward);let value=self.device.download(&resident.logits_rows(&trace,start,end-start)?).map_err(|e|e.to_string())?;drop(timer);value};
                let cpu_head_timer=if self.cuda_native.is_none() {Some(Timer::start(&self.timers.metric))}else{None};
                let z=self.native_logits(reference.slice(s![start..end,..]))?;
                let c=self.native_logits(clean.slice(s![start..end,..]))?;
                let metric_timer=if cpu_head_timer.is_none() {Some(Timer::start(&self.timers.metric))}else{None};
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
                drop(cpu_head_timer);
            }
            scores.push(EpisodeScore {id:episode.id.clone(),group:episode.group.clone(),kl:kl/count,numerical_error:(error/count).next_up(),native_effect:effect/count,top1_agree:agree/count,unheld});
        }
        Ok(scores)
    }
}

#[cfg(test)]
#[path = "streamed_family_run_tests.rs"]
mod tests;
