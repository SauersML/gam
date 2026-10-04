//! Opt-in head-only f64 CUDA logits; normalization and acceptance metrics remain CPU.
//!
//! The CPU metric is an independent operational reference with the existing
//! conditional exp/log ULP model; Rust does not guarantee those transcendental
//! errors. Neither CPU nor GPU proposal bands certify full head/network arithmetic.
use crate::counterfactual::Decoder;
use gam_gpu::tensor::{
    Arithmetic, CheckedInterval, Device, Op, Tensor, checked_interval_output_bytes,
};
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use ndarray::{Array2, Axis};

/// Numeric-buffer budgets. Excludes allocator metadata, CUDA library workspaces and context.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub resident_bytes: usize,
    pub workspace_bytes: usize,
}

/// A proposal statistic only. Never feed this estimate to acceptance as a
/// certified interval: CPU metric replay remains the independent verdict oracle.
#[derive(Clone, Debug, serde::Serialize)]
pub struct MetricProposal {
    pub kl_estimate: f64,
    pub conditional_reduction_error_estimate: f64,
    pub magnitude: f64,
    pub spread: f64,
    pub max_log_difference: f64,
    pub underflow_allowance: f64,
    pub top1_equal: bool,
}

/// Diagnostic adaptation of the CPU KL comparison model to raw GPU logits.
/// g=gamma_(V+4), eta_t=(g+4u)(|max_t|+|log sum exp|+1), and likewise q.
/// E=g*magnitude+eta_t+eta_q+(5u+eta_t)*spread+V*min_subnormal*max_log_difference.
/// The extra operation terms model exp/divide probability weights; the last term
/// records an absolute lost-tail allowance, not a relative underflow assumption.
/// This is NOT a proven bound: CUDA12.2's exp/log ULP table is based on
/// non-exhaustive tests and expressly not guaranteed. No GEMM/network error is
/// covered, and neither acceptance nor near-boundary rejection may use E.
/// https://docs.nvidia.com/cuda/archive/12.2.0/cuda-c-programming-guide/index.html#mathematical-functions-appendix
pub fn proposal_comparison_terms(
    row: &gam_gpu::tensor::KlProposalRow,
    classes: usize,
) -> Result<(f64, f64), String> {
    if [
        row.value,
        row.magnitude,
        row.spread,
        row.max_log_difference,
        row.teacher_max,
        row.teacher_log_sum,
        row.explained_max,
        row.explained_log_sum,
    ]
    .iter()
    .any(|x| !x.is_finite())
        || row.magnitude < 0.0
        || row.spread < 0.0
        || row.max_log_difference < 0.0
    {
        return Err("invalid GPU proposal statistics".into());
    }
    let count = classes
        .checked_add(4)
        .ok_or("GPU metric class count overflow")?;
    let growth = accumulation_growth(count);
    let eta = |m: f64, l: f64| (growth + 4.0 * UNIT_ROUNDOFF) * (m.abs() + l.abs() + 1.0);
    let (p, q) = (
        eta(row.teacher_max, row.teacher_log_sum),
        eta(row.explained_max, row.explained_log_sum),
    );
    let tail = (classes as f64 * f64::from_bits(1)) * row.max_log_difference;
    let estimate =
        (growth * row.magnitude + p + q + (5.0 * UNIT_ROUNDOFF + p) * row.spread + tail).next_up();
    if classes == 0 || !estimate.is_finite() || estimate < 0.0 || !tail.is_finite() {
        return Err("invalid GPU proposal comparison terms".into());
    }
    Ok((estimate, tail))
}

/// Raw fixed-binary64 head outputs; no upstream arithmetic certificate.
pub(crate) struct RawTile {
    pub intervals: Vec<CheckedInterval>,
    pub top1_equal: Vec<bool>,
    pub top1_defined: bool,
    pub oracle: Option<crate::fixed_metric_device::HostSpotcheck>,
    pub timing: crate::fixed_metric_device::Timing,
}

/// Immutable native head only: no blocks or candidate parameters are uploaded.
pub struct Resident {
    device: Device,
    embedding: Tensor,
    gain: Tensor,
    eps: f64,
    width: usize,
    vocab: usize,
    tile_rows: usize,
    resident_bytes: usize,
    raw_residual_upload_bytes: std::sync::atomic::AtomicU64,
    raw_oracle_download_bytes: std::sync::atomic::AtomicU64,
}

fn bytes(elements: usize) -> Result<usize, String> {
    elements
        .checked_mul(8)
        .ok_or_else(|| "readout byte count overflow".into())
}

/// Conservative numeric buffers: four logits tiles (including caller's paired
/// returned tile), six residual-width tiles, and forty reduction/diagnostic scalars per row.
/// Both host and device buffers count. Library-private GEMM workspace is excluded.
pub fn tile_bytes(width: usize, vocab: usize) -> Result<usize, String> {
    let n = vocab
        .checked_mul(4)
        .and_then(|n| width.checked_mul(6).and_then(|d| n.checked_add(d)))
        .and_then(|n| n.checked_add(40))
        .ok_or("readout tile count overflow")?;
    bytes(n)
}

pub fn normalize_logits(mut logits: Array2<f64>) -> Result<Array2<f64>, String> {
    if logits.ncols() == 0 || logits.iter().any(|x| !x.is_finite()) {
        return Err("readout logits must be finite and nonempty".into());
    }
    for mut row in logits.outer_iter_mut() {
        let max = row.fold(f64::NEG_INFINITY, |m, v| m.max(*v));
        let total = row.iter().map(|v| (v - max).exp()).sum::<f64>().ln() + max;
        if !total.is_finite() {
            return Err("readout log normalization is nonfinite".into());
        }
        row.mapv_inplace(|v| v - total);
    }
    if logits.iter().any(|x| !x.is_finite()) {
        return Err("normalized readout is nonfinite".into());
    }
    Ok(logits)
}

impl Resident {
    pub fn new(device: Device, decoder: &Decoder, budget: Budget) -> Result<Self, String> {
        if !cfg!(target_os = "linux") || device.is_host() || !device.float64() {
            return Err("native readout requires CUDA f64, without fallback".into());
        }
        Self::from_head(
            device,
            decoder.embedding(),
            decoder.final_gain(),
            decoder.eps(),
            budget,
        )
    }
    fn from_head(
        device: Device,
        embedding: &Array2<f64>,
        gain: &ndarray::Array1<f64>,
        eps: f64,
        budget: Budget,
    ) -> Result<Self, String> {
        let (vocab, width) = embedding.dim();
        if width == 0
            || vocab == 0
            || gain.len() != width
            || !eps.is_finite()
            || eps <= 0.0
            || embedding.iter().chain(gain.iter()).any(|x| !x.is_finite())
        {
            return Err("invalid native head shape or finite parameters".into());
        }
        let resident_bytes = bytes(
            embedding
                .len()
                .checked_add(width)
                .ok_or("readout resident count overflow")?,
        )?;
        if resident_bytes > budget.resident_bytes {
            return Err(format!(
                "native readout resident numeric bytes {resident_bytes} exceed budget {}",
                budget.resident_bytes
            ));
        }
        let tile_rows = budget.workspace_bytes / tile_bytes(width, vocab)?;
        if tile_rows == 0 {
            return Err("native readout workspace cannot hold one paired tile row".into());
        }
        let embedding = device.upload(embedding.view()).map_err(|e| e.to_string())?;
        let gain = device
            .upload(gain.view().insert_axis(Axis(0)))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            device,
            embedding,
            gain,
            eps,
            width,
            vocab,
            tile_rows,
            resident_bytes,
            raw_residual_upload_bytes: Default::default(),
            raw_oracle_download_bytes: Default::default(),
        })
    }
    pub fn tile_rows(&self) -> usize {
        self.tile_rows
    }
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
    fn logits(&self, residual: &Array2<f64>) -> Result<Tensor, String> {
        if residual.ncols() != self.width
            || residual.nrows() > self.tile_rows
            || residual.iter().any(|x| !x.is_finite())
        {
            return Err(
                "native readout residual shape, finite values or tile budget violated".into(),
            );
        }
        let d = &self.device;
        let x = d.upload(residual.view()).map_err(|e| e.to_string())?;
        self.logits_device(&x)
    }

    /// Head execution on supplied resident values; no host residual or vocabulary transfer.
    /// Nonfinite raw results are rejected by checked metrics as typed Unresolved.
    fn logits_device(&self, x: &Tensor) -> Result<Tensor, String> {
        if x.cols() != self.width || x.rows() == 0 || x.rows() > self.tile_rows {
            return Err("native resident head shape or tile budget violated".into());
        }
        let d = &self.device;
        let norm = d.rms_norm(x, self.eps).map_err(|e| e.to_string())?;
        let mut scaled = d.zeros(x.rows(), self.width).map_err(|e| e.to_string())?;
        d.scale_columns(&mut scaled, &norm, &self.gain, false)
            .map_err(|e| e.to_string())?;
        drop(norm);
        let mut out = d.zeros(x.rows(), self.vocab).map_err(|e| e.to_string())?;
        d.gemm(
            &mut out,
            1.0,
            &scaled,
            Op::N,
            &self.embedding,
            Op::T,
            0.0,
            Arithmetic::F64,
        )
        .map_err(|e| e.to_string())?;
        Ok(out)
    }

    pub fn log_probs(&self, residual: &Array2<f64>) -> Result<Array2<f64>, String> {
        self.log_probs_profiled(residual).map(|pair| pair.0)
    }

    /// Synchronous upload/norm/GEMM/download time, then CPU normalization time.
    /// These are elapsed call stages, not isolated GPU kernel measurements.
    pub(crate) fn log_probs_profiled(
        &self,
        residual: &Array2<f64>,
    ) -> Result<(Array2<f64>, [u64; 2]), String> {
        let gpu_start = std::time::Instant::now();
        let out = self.logits(residual)?;
        let logits = self.device.download(&out).map_err(|e| e.to_string())?;
        let gpu_ns = gpu_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let cpu_start = std::time::Instant::now();
        let normalized = normalize_logits(logits)?;
        let cpu_ns = cpu_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        Ok((normalized, [gpu_ns, cpu_ns]))
    }

    /// Cumulative host transfer bytes in raw metric endpoints only. Device row
    /// copies and O(rows) interval/argmax result downloads are excluded.
    pub fn raw_transfer_counts(&self) -> (u64, u64) {
        use std::sync::atomic::Ordering;
        (
            self.raw_residual_upload_bytes.load(Ordering::Relaxed),
            self.raw_oracle_download_bytes.load(Ordering::Relaxed),
        )
    }

    /// Complete conservative live numeric workspace for a paired raw tile.
    /// Includes two resident logits, optional exact-value downloads, head transient
    /// arrays, checked outputs and argmax arrays, and 2KiB static shared memory per
    /// checked row/block. Registers/spills, context and library workspace excluded.
    pub fn checked_raw_workspace_bytes(&self, rows: usize) -> Result<usize, String> {
        if rows == 0 || rows > self.tile_rows {
            return Err("raw head tile exceeds declared readout budget".into());
        }
        tile_bytes(self.width, self.vocab)?
            .checked_mul(rows)
            .and_then(|n| {
                checked_interval_output_bytes(rows)
                    .ok()
                    .and_then(|o| n.checked_add(o))
            })
            .and_then(|n| rows.checked_mul(64).and_then(|o| n.checked_add(o)))
            .and_then(|n| rows.checked_mul(2048).and_then(|o| n.checked_add(o)))
            .and_then(|n| n.checked_add(std::mem::size_of::<RawTile>()))
            .ok_or("raw head numeric budget overflow".into())
    }

    /// Opt-in checked KL directly on resident raw head logits. Production mode
    /// downloads only O(rows) bounds/argmax. Oracle mode explicitly downloads the
    /// same raw arrays and checks first/last row of every tile with host analytic
    /// intervals; these are spotchecks, not a full-network arithmetic guarantee.
    pub(crate) fn checked_raw_metrics(
        &self,
        teacher: &Array2<f64>,
        explained: &Array2<f64>,
        workspace_bytes: usize,
        oracle: bool,
    ) -> Result<RawTile, String> {
        if !cfg!(target_os = "linux") || self.device.is_host() || !self.device.float64() {
            return Err("resident raw checked metrics require CUDA f64 without fallback".into());
        }
        if teacher.dim() != explained.dim()
            || teacher.ncols() != self.width
            || teacher
                .iter()
                .chain(explained.iter())
                .any(|v| !v.is_finite())
        {
            return Err("raw head residual shape or finite values violated".into());
        }
        let required = self.checked_raw_workspace_bytes(teacher.nrows())?;
        if required > workspace_bytes {
            return Err(format!(
                "raw head numeric workspace {required} exceeds {workspace_bytes}"
            ));
        }
        let upload_timer = std::time::Instant::now();
        let uploaded = teacher
            .len()
            .checked_mul(16)
            .ok_or("raw residual transfer overflow")?;
        let p_input = self
            .device
            .upload(teacher.view())
            .map_err(|e| e.to_string())?;
        let q_input = self
            .device
            .upload(explained.view())
            .map_err(|e| e.to_string())?;
        self.raw_residual_upload_bytes
            .fetch_add(uploaded as u64, std::sync::atomic::Ordering::Relaxed);
        let upload_seconds = upload_timer.elapsed().as_secs_f64();
        let mut tile = self.checked_raw_metrics_device(
            &p_input,
            &q_input,
            0,
            teacher.nrows(),
            workspace_bytes,
            oracle,
        )?;
        tile.timing.resident_head_seconds += upload_seconds;
        Ok(tile)
    }

    /// Checked raw head metrics from immutable resident residuals. Only explicit
    /// oracle mode transfers vocabulary values; production downloads row evidence.
    pub(crate) fn checked_raw_metrics_device(
        &self,
        teacher: &Tensor,
        explained: &Tensor,
        start: usize,
        rows: usize,
        workspace_bytes: usize,
        oracle: bool,
    ) -> Result<RawTile, String> {
        if !cfg!(target_os = "linux") || self.device.is_host() || !self.device.float64() {
            return Err("resident raw checked metrics require CUDA f64 without fallback".into());
        }
        let end = start
            .checked_add(rows)
            .ok_or("raw resident row domain overflow")?;
        if teacher.rows() != explained.rows()
            || teacher.cols() != self.width
            || explained.cols() != self.width
            || end > teacher.rows()
        {
            return Err("raw resident head residual shape/domain mismatch".into());
        }
        let required = self.checked_raw_workspace_bytes(rows)?;
        if required > workspace_bytes {
            return Err(format!(
                "raw head numeric workspace {required} exceeds {workspace_bytes}"
            ));
        }
        let mut timing = crate::fixed_metric_device::Timing::default();
        let timer = std::time::Instant::now();
        let teacher_rows = self
            .device
            .rows_of(teacher, start, rows)
            .map_err(|e| e.to_string())?;
        let explained_rows = self
            .device
            .rows_of(explained, start, rows)
            .map_err(|e| e.to_string())?;
        let p = self.logits_device(&teacher_rows)?;
        let q = self.logits_device(&explained_rows)?;
        drop(teacher_rows);
        drop(explained_rows);
        self.device.synchronize().map_err(|e| e.to_string())?;
        timing.resident_head_seconds = timer.elapsed().as_secs_f64();
        let timer = std::time::Instant::now();
        let intervals = self
            .device
            .checked_kl_intervals(
                &p,
                &q,
                checked_interval_output_bytes(rows).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
        timing.checked_metric_seconds = timer.elapsed().as_secs_f64();
        let top1_defined = intervals
            .iter()
            .all(|x| matches!(x, CheckedInterval::Bounded { .. }));
        let timer = std::time::Instant::now();
        let a = self.device.argmax_rows(&p).map_err(|e| e.to_string())?;
        let b = self.device.argmax_rows(&q).map_err(|e| e.to_string())?;
        let top1_equal: Vec<bool> = a.iter().zip(&b).map(|(a, b)| a == b).collect();
        timing.gpu_top1_seconds = timer.elapsed().as_secs_f64();
        let oracle = if oracle {
            let timer = std::time::Instant::now();
            let bytes = rows
                .checked_mul(self.vocab)
                .and_then(|n| n.checked_mul(16))
                .ok_or("raw oracle transfer overflow")?;
            let hp = self.device.download(&p).map_err(|e| e.to_string())?;
            let hq = self.device.download(&q).map_err(|e| e.to_string())?;
            self.raw_oracle_download_bytes
                .fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
            timing.raw_oracle_download_seconds = timer.elapsed().as_secs_f64();
            let timer = std::time::Instant::now();
            let mut check = crate::fixed_metric_device::HostSpotcheck::default();
            for row in (0..hp.nrows()).filter(|r| *r == 0 || *r + 1 == hp.nrows()) {
                check.rows += 1;
                let first_max = |values: ndarray::ArrayView1<f64>| {
                    values
                        .iter()
                        .enumerate()
                        .fold((0, f64::NEG_INFINITY), |best, (c, v)| {
                            if *v > best.1 { (c, *v) } else { best }
                        })
                        .0
                };
                let cpu_top = first_max(hp.row(row)) == first_max(hq.row(row));
                if cpu_top != top1_equal[row] {
                    check.top1_mismatches += 1;
                }
                match crate::fixed_logit_interval::kl_logits(
                    hp.row(row).as_slice().ok_or("raw p layout")?,
                    hq.row(row).as_slice().ok_or("raw q layout")?,
                ) {
                    crate::fixed_logit_interval::Enclosure::Bounded(host) => {
                        check.maximum_host_width = check.maximum_host_width.max(host.hi - host.lo);
                        match intervals[row] {
                            CheckedInterval::Bounded { lower, upper } => {
                                if host.lo > upper || lower > host.hi {
                                    check.disjoint += 1;
                                }
                            }
                            CheckedInterval::Unresolved(_) => check.unresolved += 1,
                        }
                    }
                    crate::fixed_logit_interval::Enclosure::Unresolved(_) => check.unresolved += 1,
                }
            }
            timing.cpu_reference_seconds = timer.elapsed().as_secs_f64();
            Some(check)
        } else {
            None
        };
        Ok(RawTile {
            intervals,
            top1_equal,
            top1_defined,
            oracle,
            timing,
        })
    }

    /// Explicit proposal-only GPU reductions. Downloads O(rows) statistics and
    /// first-maximum indices; never downloads a full vocabulary logits tile.
    /// No fallback or accepted verdict is produced. CPU replay is mandatory.
    pub fn proposal_metrics(
        &self,
        teacher: &Array2<f64>,
        explained: &Array2<f64>,
    ) -> Result<Vec<MetricProposal>, String> {
        if teacher.dim() != explained.dim() {
            return Err("GPU metric proposal residual shapes differ".into());
        }
        let p = self.logits(teacher)?;
        let q = self.logits(explained)?;
        let rows = self
            .device
            .kl_proposal_rows(&p, &q)
            .map_err(|e| e.to_string())?;
        let a = self.device.argmax_rows(&p).map_err(|e| e.to_string())?;
        let b = self.device.argmax_rows(&q).map_err(|e| e.to_string())?;
        rows.iter()
            .zip(a.iter().zip(&b))
            .map(|(r, (a, b))| {
                let (estimate, tail) = proposal_comparison_terms(r, self.vocab)?;
                Ok(MetricProposal {
                    kl_estimate: r.value,
                    conditional_reduction_error_estimate: estimate,
                    magnitude: r.magnitude,
                    spread: r.spread,
                    max_log_difference: r.max_log_difference,
                    underflow_allowance: tail,
                    top1_equal: a == b,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization_extremes_and_finite_errors() {
        let p = normalize_logits(ndarray::array![
            [1000., 999., -1000.],
            [-1000., -1001., -3000.]
        ])
        .unwrap();
        for i in 0..3 {
            assert!((p[[0, i]] - p[[1, i]]).abs() < 1e-12);
        }
        assert!(normalize_logits(ndarray::array![[f64::INFINITY]]).is_err());
        assert!(normalize_logits(ndarray::array![[f64::NAN]]).is_err());
        assert!(tile_bytes(usize::MAX, 1).is_err());
    }
    #[test]
    fn proposal_statistics_match_independent_cpu_kl_and_keep_oracle_separate() {
        let e = ndarray::array![[1., 2.], [3., 4.], [-5., 6.]];
        let g = ndarray::array![1., 1.];
        let b = Budget {
            resident_bytes: 64,
            workspace_bytes: tile_bytes(2, 3).unwrap() * 2,
        };
        let h = Resident::from_head(Device::host(), &e, &g, 1e-6, b).unwrap();
        let p = ndarray::array![[1., -1.], [0., 0.]];
        let q = ndarray::array![[-2., 1.], [0., 0.]];
        let proposal = h.proposal_metrics(&p, &q).unwrap();
        let a = h.log_probs(&p).unwrap();
        let z = h.log_probs(&q).unwrap();
        for i in 0..2 {
            let (cpu, band) = crate::acceptance::kl_logits(a.row(i), z.row(i));
            assert!((cpu - proposal[i].kl_estimate).abs() < 1e-12);
            assert!(
                band.is_finite() && proposal[i].conditional_reduction_error_estimate.is_finite()
            );
        }
        assert!(proposal[1].top1_equal);
        assert_eq!(proposal[1].kl_estimate, 0.0);
        assert!(h.proposal_metrics(&p, &Array2::zeros((1, 2))).is_err());
        assert!(
            h.proposal_metrics(&p, &Array2::from_elem((2, 2), f64::NAN))
                .is_err()
        );
    }

    #[test]
    fn resident_head_matches_existing_arrays_without_host_transfers() {
        let device = Device::host();
        let embedding = ndarray::array![[1., 2.], [-3., 4.], [5., -6.]];
        let gain = ndarray::array![0.75, 1.25];
        let head = Resident::from_head(
            device.clone(),
            &embedding,
            &gain,
            1e-6,
            Budget {
                resident_bytes: 64,
                workspace_bytes: tile_bytes(2, 3).unwrap() * 2,
            },
        )
        .unwrap();
        let full = ndarray::array![[9., 8.], [1., -1.], [0., -0.]];
        let tensor = device.upload(full.view()).unwrap();
        let tile = device.rows_of(&tensor, 1, 2).unwrap();
        let resident = head.logits_device(&tile).unwrap();
        let arrays = head
            .logits(&full.slice(ndarray::s![1.., ..]).to_owned())
            .unwrap();
        assert_eq!(
            device.download(&resident).unwrap(),
            device.download(&arrays).unwrap()
        );
        assert_eq!(device.download(&tensor).unwrap(), full);
        assert_eq!(head.raw_transfer_counts(), (0, 0));
        assert!(head.logits_device(&device.zeros(3, 2).unwrap()).is_err());
        assert!(head.logits_device(&device.zeros(1, 3).unwrap()).is_err());
        assert!(head.logits_device(&device.zeros(0, 2).unwrap()).is_err());
        // No host fallback for the checked resident endpoint.
        assert!(
            head.checked_raw_metrics_device(&tensor, &tensor, 1, 2, usize::MAX, false)
                .is_err()
        );
    }

    #[test]
    fn head_only_host_structure_and_budget() {
        let e = ndarray::array![[1., 2.], [3., 4.], [5., 6.]];
        let g = ndarray::array![1., 1.];
        let b = Budget {
            resident_bytes: 64,
            workspace_bytes: tile_bytes(2, 3).unwrap() * 2,
        };
        let h = Resident::from_head(Device::host(), &e, &g, 1e-6, b).unwrap();
        assert_eq!(h.resident_bytes(), 64);
        assert_eq!(h.tile_rows(), 2);
        assert!(h.checked_raw_workspace_bytes(2).unwrap() > 2 * tile_bytes(2, 3).unwrap());
        assert!(h.checked_raw_workspace_bytes(0).is_err());
        assert!(h.checked_raw_workspace_bytes(3).is_err());
        assert!(
            h.checked_raw_metrics(
                &Array2::zeros((1, 2)),
                &Array2::zeros((1, 2)),
                1 << 20,
                false
            )
            .is_err()
        );
        assert!(h.log_probs(&Array2::zeros((3, 2))).is_err());
        assert!(
            Resident::from_head(
                Device::host(),
                &e,
                &g,
                1e-6,
                Budget {
                    resident_bytes: 63,
                    ..b
                }
            )
            .is_err()
        );
        let p = h.log_probs(&ndarray::array![[1., -1.]]).unwrap();
        assert!(p.iter().all(|x| x.is_finite()));
    }
}
