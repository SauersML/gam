//! Optional CUDA intervals on the same finite binary64 arrays supplied to CPU KL.
//! These normalized arrays are reinterpreted as logits and softmax-normalized
//! again, exactly as acceptance::kl_logits does. No upstream arithmetic claim.
use crate::fixed_logit_interval::Interval;
use gam_gpu::tensor::{CheckedInterval, Device, checked_interval_output_bytes};
use ndarray::Array2;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Budget {
    pub batch_rows: usize,
    pub workspace_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Outcome {
    Bounded { lower: f64, upper: f64 },
    Unresolved { reason: String },
}
impl Outcome {
    pub(crate) fn of(interval: Interval) -> Self {
        if interval.lo.is_finite() && interval.hi.is_finite() && interval.lo <= interval.hi {
            Self::Bounded {
                lower: interval.lo,
                upper: interval.hi,
            }
        } else {
            Self::Unresolved {
                reason: "nonfinite interval aggregation".into(),
            }
        }
    }
    pub(crate) fn interval(&self) -> Option<Interval> {
        match self {
            Self::Bounded { lower, upper } => Some(Interval::new(*lower, *upper)),
            Self::Unresolved { .. } => None,
        }
    }
}

#[derive(Default, Clone, Debug, Serialize)]
pub struct Timing {
    pub packing_and_upload_seconds: f64,
    pub checked_metric_seconds: f64,
    pub cpu_reference_seconds: f64,
    pub cpu_top1_seconds: f64,
    pub resident_head_seconds: f64,
    pub gpu_top1_seconds: f64,
    pub raw_oracle_download_seconds: f64,
}

/// CPU comparisons retain the existing conditional ULP model, not a proof oracle.
#[derive(Default, Clone, Debug, Serialize)]
pub struct CpuComparison {
    pub mean: f64,
    pub conditional_error: f64,
    pub maximum_midpoint_difference: f64,
    pub maximum_distance_from_checked_interval: f64,
    pub maximum_distance_row: usize,
    pub row_values_outside_checked_interval: usize,
    pub conditional_intervals_disjoint: usize,
}

/// Independent host analytic spotchecks on exactly the downloaded raw logits.
#[derive(Default, Clone, Debug, Serialize)]
pub struct HostSpotcheck {
    pub rows: usize,
    pub disjoint: usize,
    pub unresolved: usize,
    pub maximum_host_width: f64,
    pub top1_mismatches: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Episode {
    pub id: String,
    pub group: String,
    pub scored_from: usize,
    pub scored_until: usize,
    pub kl: Outcome,
    pub native_effect_cpu_metric: f64,
    pub top1_agree: f64,
    pub top1_defined: bool,
    pub unheld: usize,
    pub cpu_comparison: Option<CpuComparison>,
    pub metric_timing: Timing,
    pub host_analytic_spotcheck: Option<HostSpotcheck>,
}

#[derive(Debug, Serialize)]
pub struct Group {
    pub name: String,
    pub episodes: usize,
    pub mean_kl: Outcome,
}

#[derive(Debug, Serialize)]
pub struct Measure {
    pub episodes: Vec<Episode>,
    pub groups: Vec<Group>,
    pub fixed_input_scope: &'static str,
    pub workspace_numeric_bound_bytes: usize,
    pub readout_numeric_resident_bytes: Option<usize>,
    pub workspace_budget: Budget,
    pub result_numeric_storage_estimate_bytes: usize,
}
impl Measure {
    pub fn of(episodes: Vec<Episode>, resident: &Resident) -> Self {
        Self::from_parts(
            episodes,
            resident.budget,
            resident.required_bytes,
            "Exact fixed binary64 CPU-normalized arrays reinterpreted as logits and independently softmax-normalized; no readout/GEMM/network arithmetic guarantee; no default acceptance integration; CPU ULP comparison model is conditional",
        )
    }
    pub(crate) fn from_parts(
        episodes: Vec<Episode>,
        budget: Budget,
        required_bytes: usize,
        scope: &'static str,
    ) -> Self {
        let mut grouped = std::collections::BTreeMap::<String, Vec<&Outcome>>::new();
        for episode in &episodes {
            grouped
                .entry(episode.group.clone())
                .or_default()
                .push(&episode.kl);
        }
        let groups: Vec<Group> = grouped
            .into_iter()
            .map(|(name, outcomes)| {
                let count = outcomes.len();
                Group {
                    name,
                    episodes: count,
                    mean_kl: mean_outcomes(&outcomes),
                }
            })
            .collect();
        let result_numeric_storage_estimate_bytes = episodes
            .len()
            .saturating_mul(std::mem::size_of::<Episode>())
            .saturating_add(groups.len().saturating_mul(std::mem::size_of::<Group>()))
            .saturating_add(std::mem::size_of::<Self>());
        Self {
            episodes,
            groups,
            fixed_input_scope: scope,
            workspace_numeric_bound_bytes: required_bytes,
            readout_numeric_resident_bytes: None,
            workspace_budget: budget,
            result_numeric_storage_estimate_bytes,
        }
    }
}
pub(crate) fn mean_outcomes(outcomes: &[&Outcome]) -> Outcome {
    if outcomes.is_empty() {
        return Outcome::Unresolved {
            reason: "empty group".into(),
        };
    }
    if outcomes.len() as u128 > (1_u128 << 53) {
        return Outcome::Unresolved {
            reason: "group denominator not exactly representable".into(),
        };
    }
    let mut sum = Interval::point(0.0);
    for outcome in outcomes {
        let Some(interval) = outcome.interval() else {
            return (*outcome).clone();
        };
        sum = sum.add(interval);
    }
    Outcome::of(sum.div_positive(Interval::point(outcomes.len() as f64)))
}

pub struct Resident {
    device: Device,
    columns: usize,
    readout_rows: usize,
    budget: Budget,
    required_bytes: usize,
    compare_cpu: bool,
}

pub fn workspace_bytes(
    columns: usize,
    batch_rows: usize,
    readout_rows: usize,
) -> Result<usize, String> {
    if columns == 0 || batch_rows == 0 || readout_rows == 0 {
        return Err("positive checked metric dimensions required".into());
    }
    // Pending teacher/candidate, two CUDA operands, one transient upload clone,
    // plus both incoming CPU-normalized readout arrays. No full-model duplicate.
    let elements = batch_rows
        .checked_mul(5)
        .and_then(|n| readout_rows.checked_mul(2).and_then(|r| n.checked_add(r)))
        .and_then(|n| n.checked_mul(columns))
        .ok_or("checked metric numeric count overflow")?;
    elements
        .checked_mul(8)
        .and_then(|n| {
            checked_interval_output_bytes(batch_rows)
                .ok()
                .and_then(|o| n.checked_add(o))
        })
        .and_then(|n| n.checked_add(readout_rows))
        .and_then(|n| n.checked_add(std::mem::size_of::<Stream>()))
        .ok_or("checked metric numeric byte overflow".into())
}
impl Resident {
    pub fn new(
        device: Device,
        columns: usize,
        readout_rows: usize,
        budget: Budget,
        compare_cpu: bool,
    ) -> Result<Self, String> {
        if !cfg!(target_os = "linux") || device.is_host() || !device.float64() {
            return Err("fixed metric intervals require CUDA f64 without fallback".into());
        }
        let required_bytes = workspace_bytes(columns, budget.batch_rows, readout_rows)?;
        if required_bytes > budget.workspace_bytes {
            return Err(format!(
                "checked metric numeric workspace {required_bytes} exceeds {}",
                budget.workspace_bytes
            ));
        }
        Ok(Self {
            device,
            columns,
            readout_rows,
            budget,
            required_bytes,
            compare_cpu,
        })
    }
    pub fn stream(&self) -> Stream<'_> {
        Stream {
            resident: self,
            p: Vec::with_capacity(self.columns * self.budget.batch_rows),
            q: Vec::with_capacity(self.columns * self.budget.batch_rows),
            sum: Outcome::Bounded {
                lower: 0.0,
                upper: 0.0,
            },
            count: 0,
            agree: 0,
            pending_from: 0,
            domain_start: None,
            domain_end: None,
            comparison: self.compare_cpu.then(CpuComparison::default),
            timing: Timing::default(),
        }
    }
}

pub struct Stream<'a> {
    resident: &'a Resident,
    p: Vec<f64>,
    q: Vec<f64>,
    sum: Outcome,
    count: usize,
    agree: usize,
    pending_from: usize,
    domain_start: Option<usize>,
    domain_end: Option<usize>,
    comparison: Option<CpuComparison>,
    timing: Timing,
}
impl Stream<'_> {
    pub fn append(&mut self, p: &Array2<f64>, q: &Array2<f64>, from: usize) -> Result<(), String> {
        if p.dim() != q.dim() || p.ncols() != self.resident.columns {
            return Err("checked metric tile shape mismatch".into());
        }
        if p.nrows() > self.resident.readout_rows {
            return Err("incoming normalized tile exceeds declared metric budget rows".into());
        }
        if self.domain_end.is_some_and(|end| end != from) {
            return Err("checked metric row-domain gap or overlap".into());
        }
        self.domain_start.get_or_insert(from);
        self.domain_end = Some(
            from.checked_add(p.nrows())
                .ok_or("scored row index overflow")?,
        );
        let timer = std::time::Instant::now();
        self.agree += crate::counterfactual::top1_rows(p, q)
            .into_iter()
            .filter(|a| *a)
            .count();
        self.timing.cpu_top1_seconds += timer.elapsed().as_secs_f64();
        for r in 0..p.nrows() {
            if self.p.is_empty() {
                self.pending_from = from + r;
            }
            let timer = std::time::Instant::now();
            self.p.extend(p.row(r).iter().copied());
            self.q.extend(q.row(r).iter().copied());
            self.timing.packing_and_upload_seconds += timer.elapsed().as_secs_f64();
            if self.p.len() / self.resident.columns == self.resident.budget.batch_rows {
                self.flush()?;
            }
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<(), String> {
        if self.p.is_empty() {
            return Ok(());
        }
        let rows = self.p.len() / self.resident.columns;
        let shape = (rows, self.resident.columns);
        let p = ndarray::ArrayView2::from_shape(shape, &self.p).map_err(|e| e.to_string())?;
        let q = ndarray::ArrayView2::from_shape(shape, &self.q).map_err(|e| e.to_string())?;
        let timer = std::time::Instant::now();
        let teacher = self.resident.device.upload(p).map_err(|e| e.to_string())?;
        let candidate = self.resident.device.upload(q).map_err(|e| e.to_string())?;
        self.timing.packing_and_upload_seconds += timer.elapsed().as_secs_f64();
        let timer = std::time::Instant::now();
        let intervals = self
            .resident
            .device
            .checked_kl_intervals(
                &teacher,
                &candidate,
                checked_interval_output_bytes(rows).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
        self.timing.checked_metric_seconds += timer.elapsed().as_secs_f64();
        for (row, outcome) in intervals.into_iter().enumerate() {
            let reference = if let Some(comparison) = &mut self.comparison {
                let timer = std::time::Instant::now();
                let (value, error) = crate::acceptance::kl_logits(p.row(row), q.row(row));
                self.timing.cpu_reference_seconds += timer.elapsed().as_secs_f64();
                if !value.is_finite() || !error.is_finite() {
                    return Err("CPU comparison nonfinite".into());
                }
                comparison.mean += value;
                comparison.conditional_error += error;
                Some((value, error))
            } else {
                None
            };
            match outcome {
                CheckedInterval::Bounded { lower, upper } => {
                    if let (Some(comparison), Some((value, error))) =
                        (&mut self.comparison, reference)
                    {
                        comparison.maximum_midpoint_difference = comparison
                            .maximum_midpoint_difference
                            .max((value - (lower / 2.0 + upper / 2.0)).abs());
                        let distance = (lower - value).max(value - upper).max(0.0);
                        if distance > 0.0 {
                            comparison.row_values_outside_checked_interval += 1;
                        }
                        if distance > comparison.maximum_distance_from_checked_interval {
                            comparison.maximum_distance_from_checked_interval = distance;
                            comparison.maximum_distance_row = self.pending_from + row;
                        }
                        if value + error < lower || value - error > upper {
                            comparison.conditional_intervals_disjoint += 1;
                        }
                    }
                    if let Some(sum) = self.sum.interval() {
                        self.sum = Outcome::of(sum.add(Interval::new(lower, upper)));
                    }
                }
                CheckedInterval::Unresolved(reason) => {
                    self.sum = Outcome::Unresolved {
                        reason: format!("row {}: {reason:?}", self.pending_from + row),
                    }
                }
            }
        }
        self.count += rows;
        self.p.clear();
        self.q.clear();
        Ok(())
    }
    pub fn finish(
        mut self,
        id: String,
        group: String,
        from: usize,
        until: usize,
        native_effect: f64,
        unheld: usize,
    ) -> Result<Episode, String> {
        self.flush()?;
        if self.count != until.saturating_sub(from)
            || self.count == 0
            || self.domain_start != Some(from)
            || self.domain_end != Some(until)
        {
            return Err("checked metric scored-row domain mismatch".into());
        }
        if self.count as u128 > (1_u128 << 53) {
            return Err("episode denominator not exactly representable".into());
        }
        let n = self.count as f64;
        let kl = match self.sum.interval() {
            Some(sum) => Outcome::of(sum.div_positive(Interval::point(n))),
            None => self.sum,
        };
        if let Some(comparison) = &mut self.comparison {
            comparison.mean /= n;
            comparison.conditional_error = (comparison.conditional_error / n).next_up();
        }
        Ok(Episode {
            id,
            group,
            scored_from: from,
            scored_until: until,
            kl,
            native_effect_cpu_metric: native_effect,
            top1_agree: self.agree as f64 / n,
            top1_defined: true,
            unheld,
            cpu_comparison: self.comparison,
            metric_timing: self.timing,
            host_analytic_spotcheck: None,
        })
    }
}

pub(crate) struct RawEvidence {
    pub kl: Outcome,
    pub top1_agree: f64,
    pub top1_defined: bool,
    pub host_analytic_spotcheck: Option<HostSpotcheck>,
    pub metric_timing: Timing,
}

/// Directed accumulation of already-resident raw-logit metric rows. No vocabulary buffers.
pub(crate) struct RawStream {
    top1_defined: bool,
    sum: Outcome,
    count: usize,
    agree: usize,
    start: Option<usize>,
    end: Option<usize>,
    oracle: Option<HostSpotcheck>,
    timing: Timing,
}
impl RawStream {
    pub(crate) fn new(oracle: bool) -> Self {
        Self {
            top1_defined: true,
            sum: Outcome::Bounded {
                lower: 0.0,
                upper: 0.0,
            },
            count: 0,
            agree: 0,
            start: None,
            end: None,
            oracle: oracle.then(HostSpotcheck::default),
            timing: Timing::default(),
        }
    }
    pub(crate) fn append(
        &mut self,
        tile: crate::native_readout::RawTile,
        from: usize,
    ) -> Result<(), String> {
        if tile.intervals.is_empty() || tile.intervals.len() != tile.top1_equal.len() {
            return Err("raw checked metric tile shape mismatch".into());
        }
        if self.end.is_some_and(|end| end != from) {
            return Err("raw checked row-domain gap or overlap".into());
        }
        let until = from
            .checked_add(tile.intervals.len())
            .ok_or("raw row index overflow")?;
        self.start.get_or_insert(from);
        self.end = Some(until);
        for (row, interval) in tile.intervals.iter().enumerate() {
            match interval {
                CheckedInterval::Bounded { lower, upper } => {
                    if let Some(sum) = self.sum.interval() {
                        self.sum = Outcome::of(sum.add(Interval::new(*lower, *upper)));
                    }
                }
                CheckedInterval::Unresolved(reason) => {
                    self.sum = Outcome::Unresolved {
                        reason: format!("raw row {}: {reason:?}", from + row),
                    };
                }
            }
        }
        self.top1_defined &= tile.top1_defined;
        self.count += tile.intervals.len();
        self.agree += tile.top1_equal.iter().filter(|x| **x).count();
        if let Some(ours) = &mut self.oracle {
            let other = tile.oracle.ok_or("raw analytic oracle absent")?;
            ours.rows += other.rows;
            ours.disjoint += other.disjoint;
            ours.unresolved += other.unresolved;
            ours.top1_mismatches += other.top1_mismatches;
            ours.maximum_host_width = ours.maximum_host_width.max(other.maximum_host_width);
        }
        self.timing.resident_head_seconds += tile.timing.resident_head_seconds;
        self.timing.checked_metric_seconds += tile.timing.checked_metric_seconds;
        self.timing.gpu_top1_seconds += tile.timing.gpu_top1_seconds;
        self.timing.raw_oracle_download_seconds += tile.timing.raw_oracle_download_seconds;
        self.timing.cpu_reference_seconds += tile.timing.cpu_reference_seconds;
        Ok(())
    }
    pub(crate) fn finish_evidence(self,from:usize,until:usize)->Result<RawEvidence,String> {
        if self.count == 0
            || self.count != until.saturating_sub(from)
            || self.start != Some(from)
            || self.end != Some(until)
            || self.count as u128 > (1_u128 << 53)
        {
            return Err("raw checked scored-row domain mismatch".into());
        }
        let n = self.count as f64;
        let kl = match self.sum.interval() {
            Some(sum) => Outcome::of(sum.div_positive(Interval::point(n))),
            None => self.sum,
        };
        Ok(RawEvidence {kl,top1_agree:self.agree as f64/n,top1_defined:self.top1_defined,
             host_analytic_spotcheck:self.oracle,metric_timing:self.timing})
    }
    pub(crate) fn finish(self,id:String,group:String,from:usize,until:usize,native_effect:f64,unheld:usize)->Result<Episode,String> {
        let evidence=self.finish_evidence(from,until)?;
        Ok(Episode {id,group,scored_from:from,scored_until:until,kl:evidence.kl,native_effect_cpu_metric:native_effect,
             top1_agree:evidence.top1_agree,top1_defined:evidence.top1_defined,unheld,cpu_comparison:None,
             metric_timing:evidence.metric_timing,host_analytic_spotcheck:evidence.host_analytic_spotcheck})
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn group_mean_retains_directed_intervals_and_unknowns() {
        let a = Outcome::Bounded {
            lower: 0.1,
            upper: 0.2,
        };
        let b = Outcome::Bounded {
            lower: 0.3,
            upper: 0.4,
        };
        let mean = mean_outcomes(&[&a, &b]);
        assert!(matches!(mean,Outcome::Bounded {lower,upper} if lower<=0.2 && upper>=0.3));
        let unknown = Outcome::Unresolved {
            reason: "guard".into(),
        };
        assert_eq!(mean_outcomes(&[&a, &unknown]), unknown);
        assert!(matches!(mean_outcomes(&[]), Outcome::Unresolved { .. }));
    }
    fn raw_tile(bounds: Vec<CheckedInterval>, valid: bool) -> crate::native_readout::RawTile {
        crate::native_readout::RawTile {
            top1_equal: vec![true; bounds.len()],
            top1_defined: valid,
            intervals: bounds,
            oracle: None,
            timing: Timing::default(),
        }
    }
    #[test]
    fn raw_rows_preserve_directed_domains_and_unknowns() {
        let mut stream = RawStream::new(false);
        stream
            .append(
                raw_tile(
                    vec![CheckedInterval::Bounded {
                        lower: 0.1,
                        upper: 0.2,
                    }],
                    true,
                ),
                7,
            )
            .expect("first");
        assert!(
            stream
                .append(
                    raw_tile(
                        vec![CheckedInterval::Bounded {
                            lower: 0.3,
                            upper: 0.4
                        }],
                        true
                    ),
                    9
                )
                .is_err()
        );
        stream
            .append(
                raw_tile(
                    vec![CheckedInterval::Bounded {
                        lower: 0.3,
                        upper: 0.4,
                    }],
                    true,
                ),
                8,
            )
            .expect("contiguous");
        let episode = stream
            .finish("a".into(), "b".into(), 7, 9, 3.0, 2)
            .expect("domain");
        assert!(matches!(episode.kl,Outcome::Bounded {lower,upper} if lower<=0.2 && upper>=0.3));
        assert!(episode.top1_defined);
        assert_eq!(episode.top1_agree, 1.0);
        assert_eq!(episode.native_effect_cpu_metric, 3.0);
        let mut unknown = RawStream::new(false);
        unknown
            .append(
                raw_tile(
                    vec![CheckedInterval::Unresolved(
                        gam_gpu::tensor::CheckedIntervalReason::NonFiniteInput,
                    )],
                    false,
                ),
                0,
            )
            .expect("unknown");
        let episode = unknown
            .finish("u".into(), "b".into(), 0, 1, 0.0, 0)
            .expect("unknown domain");
        assert!(matches!(episode.kl, Outcome::Unresolved { .. }));
        assert!(!episode.top1_defined);
    }
    #[test]
    fn explicit_workspace_covers_upload_clone_and_incoming_tiles() {
        let bytes = workspace_bytes(50277, 128, 64).expect("bounded");
        assert!(bytes > 5 * 128 * 50277 * 8 + 2 * 64 * 50277 * 8);
        assert!(bytes < 1 << 30);
        assert!(workspace_bytes(usize::MAX, 128, 64).is_err());
        assert!(workspace_bytes(50277, 0, 64).is_err());
        assert!(
            Resident::new(
                Device::host(),
                50277,
                64,
                Budget {
                    batch_rows: 128,
                    workspace_bytes: 1 << 30
                },
                false
            )
            .is_err()
        );
    }
}
