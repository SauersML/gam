//! Generic hybrid FamilyRun: immutable CPU native teachers, CUDA f64 candidate execution,
//! CPU KL comparison. No specialized language-model decoder or CPU execution fallback.
use super::{
    acceptance::{Change, Edit, EpisodeScore, FamilyRun, RunCheck, kl_logits},
    artifact::Artifact,
    artifact_device::Resident,
};
use gam_gpu::tensor::Device;
use ndarray::{Array2, s};
use std::{collections::BTreeMap, sync::OnceLock};

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Timing {
    pub native_teacher_seconds: f64,
    pub candidate_construction_seconds: f64,
    pub cuda_forward_hooks_download_seconds: f64,
    pub cpu_metric_seconds: f64,
}
#[derive(Default)]
struct Timers {
    teacher: std::sync::atomic::AtomicU64,
    construction: std::sync::atomic::AtomicU64,
    forward: std::sync::atomic::AtomicU64,
    metric: std::sync::atomic::AtomicU64,
}
struct Timer<'a>(std::time::Instant, &'a std::sync::atomic::AtomicU64);
impl<'a> Timer<'a> {
    fn start(counter: &'a std::sync::atomic::AtomicU64) -> Self {
        Self(std::time::Instant::now(), counter)
    }
}
impl Drop for Timer<'_> {
    fn drop(&mut self) {
        self.1.fetch_add(
            self.0.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}
type Teachers = (Array2<f64>, Vec<Array2<f64>>);
pub struct DeviceFamilyRun<'a> {
    run: &'a FamilyRun<'a>,
    device: Device,
    intermediate_bytes_limit: usize,
    teachers: OnceLock<Result<Teachers, String>>,
    timers: Timers,
}
fn validate_edits(
    edits: &[Edit],
    rows: usize,
    width: impl Fn(usize) -> Option<usize>,
) -> Result<(), String> {
    for e in edits {
        let w = width(e.node).ok_or("intervention node absent")?;
        if e.columns.start >= e.columns.end
            || e.columns.end > w
            || e.rows
                .as_ref()
                .is_some_and(|r| r.is_empty() || r.iter().any(|&i| i >= rows))
        {
            return Err("intervention columns or rows outside native value".into());
        }
        let v = match e.change {
            Change::Scale(v) | Change::Add(v) => v,
        };
        if !v.is_finite() {
            return Err("nonfinite intervention".into());
        }
    }
    Ok(())
}
fn apply(edits: &[&Edit], value: &mut Array2<f64>) {
    for e in edits {
        for row in e
            .rows
            .clone()
            .unwrap_or_else(|| (0..value.nrows()).collect())
        {
            for column in e.columns.clone() {
                let v = &mut value[[row, column]];
                *v = match e.change {
                    Change::Scale(x) => *v * x,
                    Change::Add(x) => *v + x,
                };
            }
        }
    }
}
impl<'a> DeviceFamilyRun<'a> {
    /// Limit covers estimated retained intermediate tensors, including materialized logits.
    /// It excludes operators, CUDA workspaces, hook transfers, teachers and host allocations.
    pub fn new(
        run: &'a FamilyRun<'a>,
        device: Device,
        intermediate_bytes_limit: usize,
    ) -> Result<Self, String> {
        if !cfg!(target_os = "linux")
            || device.is_host()
            || !device.float64()
            || intermediate_bytes_limit == 0
        {
            return Err("DeviceFamilyRun requires Linux float64 CUDA and explicit positive intermediate byte limit".into());
        }
        if run.readouts == 0 || run.family.rows == 0 || run.episodes.is_empty() {
            return Err("nonempty family/episodes and positive readouts required".into());
        }
        let interfaces = run.model.interfaces().map_err(|e| e.to_string())?;
        for e in &run.episodes {
            validate_edits(&e.edits, run.family.rows, |n| {
                interfaces.get(n).map(|i| i.width())
            })?;
        }
        Ok(Self {
            run,
            device,
            intermediate_bytes_limit,
            teachers: OnceLock::new(),
            timers: Timers::default(),
        })
    }
    /// Diagnostic host wall intervals. Forward includes intervention transfers and
    /// output download synchronization; no new synchronization is introduced.
    pub fn timing(&self) -> Timing {
        let seconds = |x: &std::sync::atomic::AtomicU64| {
            x.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9
        };
        Timing {
            native_teacher_seconds: seconds(&self.timers.teacher),
            candidate_construction_seconds: seconds(&self.timers.construction),
            cuda_forward_hooks_download_seconds: seconds(&self.timers.forward),
            cpu_metric_seconds: seconds(&self.timers.metric),
        }
    }
    pub fn backend_name(&self) -> &'static str {
        "hybrid: explained CUDA f64; cached native teacher and output KL CPU"
    }
    fn teachers(&self) -> Result<&Teachers, String> {
        self.teachers
            .get_or_init(|| {
                let teacher_timer = Timer::start(&self.timers.teacher);
                let native = |edits: &[Edit]| -> Result<Array2<f64>, String> {
                    let t = self
                        .run
                        .model
                        .execute_edited(&self.run.family, |node, value, _| {
                            apply(
                                &edits.iter().filter(|e| e.node == node).collect::<Vec<_>>(),
                                value,
                            );
                            Ok(())
                        })
                        .map_err(|e| e.to_string())?;
                    Ok(t.values[self.run.model.output].clone())
                };
                let result = Ok((
                    native(&[])?,
                    self.run
                        .episodes
                        .iter()
                        .map(|e| native(&e.edits))
                        .collect::<Result<_, _>>()?,
                ));
                drop(teacher_timer);
                result
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}
impl RunCheck for DeviceFamilyRun<'_> {
    fn episodes(&self, artifact: &Artifact) -> Result<Vec<EpisodeScore>, String> {
        artifact.validate_coverage(self.run.model)?;
        let (clean, references) = self.teachers()?;
        let resident = {
            let construction_timer = Timer::start(&self.timers.construction);
            let resident = Resident::from_decoded(&self.device, artifact)?;
            drop(construction_timer);
            resident
        };
        let estimate = resident.estimated_resident_bytes(self.run.family.rows)?;
        if estimate > self.intermediate_bytes_limit {
            return Err(format!(
                "estimated retained CUDA intermediates {estimate} exceed declared limit {}; estimate excludes operators/workspaces/hooks/host",
                self.intermediate_bytes_limit
            ));
        }
        let interfaces = artifact.program.interfaces().map_err(|e| e.to_string())?;
        let mut scores = Vec::new();
        for (episode, reference) in self.run.episodes.iter().zip(references) {
            let mut held: BTreeMap<usize, Vec<&Edit>> = BTreeMap::new();
            let mut unheld = 0;
            for e in &episode.edits {
                match artifact.place(e.node) {
                    Some(n) => {
                        validate_edits(std::slice::from_ref(e), self.run.family.rows, |_| {
                            interfaces.get(n).map(|i| i.width())
                        })?;
                        held.entry(n).or_default().push(e);
                    }
                    None => unheld += 1,
                }
            }
            let forward_timer = Timer::start(&self.timers.forward);
            let trace = resident.forward_edited(&self.run.family, |node, trace| {
                let Some(edits) = held.get(&node) else {
                    return Ok(None);
                };
                let mut value = self
                    .device
                    .download(resident.root_value(trace, node)?)
                    .map_err(|e| e.to_string())?;
                apply(edits, &mut value);
                Ok(Some(
                    self.device
                        .upload(value.view())
                        .map_err(|e| e.to_string())?,
                ))
            })?;
            let explained = self
                .device
                .download(&resident.output(&trace)?)
                .map_err(|e| e.to_string())?;
            drop(forward_timer);
            let metric_timer = Timer::start(&self.timers.metric);
            if explained.dim() != reference.dim() || reference.ncols() % self.run.readouts != 0 {
                return Err("candidate output/readout dimensions differ".into());
            }
            let classes = reference.ncols() / self.run.readouts;
            if classes == 0 {
                return Err("empty output classes".into());
            }
            let (mut kl, mut error, mut effect, mut agree, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for row in 0..reference.nrows() {
                for k in 0..self.run.readouts {
                    let range = k * classes..(k + 1) * classes;
                    let (z, w, c) = (
                        reference.slice(s![row, range.clone()]),
                        explained.slice(s![row, range.clone()]),
                        clean.slice(s![row, range]),
                    );
                    if z.iter()
                        .chain(w.iter())
                        .chain(c.iter())
                        .any(|v| !v.is_finite())
                    {
                        return Err("nonfinite counterfactual output".into());
                    }
                    let (value, rounding) = kl_logits(z, w);
                    kl += value;
                    error += rounding;
                    effect += kl_logits(z, c).0;
                    let argmax = |a: ndarray::ArrayView1<'_, f64>| {
                        a.iter()
                            .enumerate()
                            .fold((0, f64::NEG_INFINITY), |best, (i, &v)| {
                                if v > best.1 { (i, v) } else { best }
                            })
                            .0
                    };
                    agree += f64::from(u8::from(argmax(z) == argmax(w)));
                    n += 1.0;
                }
            }
            scores.push(EpisodeScore {
                id: episode.id.clone(),
                group: episode.group.clone(),
                kl: kl / n,
                numerical_error: (error / n).next_up(),
                native_effect: effect / n,
                top1_agree: agree / n,
                unheld,
            });
            drop(metric_timer);
        }
        Ok(scores)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_outside_interventions_before_execution() {
        let mut e = Edit {
            node: 2,
            rows: Some(vec![1]),
            columns: 0..2,
            change: Change::Scale(0.0),
        };
        assert!(validate_edits(&[e.clone()], 2, |n| (n == 2).then_some(2)).is_ok());
        e.rows = Some(vec![2]);
        assert!(validate_edits(&[e.clone()], 2, |_| Some(2)).is_err());
        e.rows = None;
        e.columns = 0..3;
        assert!(validate_edits(&[e.clone()], 2, |_| Some(2)).is_err());
        e.columns = 0..2;
        e.change = Change::Add(f64::NAN);
        assert!(validate_edits(&[e], 2, |_| Some(2)).is_err());
    }
    #[test]
    fn edits_preserve_declared_sequence() {
        let mut value = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let a = Edit {
            node: 0,
            rows: Some(vec![0]),
            columns: 0..1,
            change: Change::Add(2.0),
        };
        let b = Edit {
            node: 0,
            rows: None,
            columns: 0..1,
            change: Change::Scale(3.0),
        };
        apply(&[&a, &b], &mut value);
        assert_eq!(
            value,
            Array2::from_shape_vec((2, 2), vec![9.0, 2.0, 9.0, 4.0]).unwrap()
        );
        let repeated = Edit {
            rows: Some(vec![0, 0]),
            change: Change::Add(1.0),
            ..a
        };
        apply(&[&repeated], &mut value);
        assert_eq!(value[[0, 0]], 11.0);
    }
}
