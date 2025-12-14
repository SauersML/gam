use super::progress::{
    FitProgressObserver, FitProgressStage, NoopFitProgress, StageProgressHandle,
};
use super::variant_filter::VariantKey;
use core::cmp::{Ordering, min};
use core::fmt;
use core::marker::PhantomData;
use dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::col::Col;
use faer::linalg::matmul::matmul;
use faer::linalg::matmul::triangular as triangular_matmul;
use faer::linalg::solvers::{Llt as FaerLlt, Solve as FaerSolve};
use faer::linalg::{temp_mat_scratch, temp_mat_uninit};
use faer::mat::AsMatMut;
use faer::matrix_free::LinOp;
use faer::matrix_free::eigen::{
    PartialEigenInfo, PartialEigenParams, partial_eigen_scratch, partial_self_adjoint_eigen,
};
use faer::prelude::ReborrowMut;
use faer::{
    Accum, ColMut, Mat, MatMut, MatRef, Par, Side, get_global_parallelism, set_global_parallelism,
    unzip, zip,
};
use rayon::prelude::*;
use serde::de::Error as DeError;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::convert::Infallible;
use std::error::Error;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::simd::num::SimdFloat;
use std::simd::{LaneCount, Simd, SupportedLaneCount};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use sysinfo::System;

pub const HWE_VARIANCE_EPSILON: f64 = 1.0e-12;
pub const HWE_SCALE_FLOOR: f64 = 1.0e-6;
pub const EIGENVALUE_EPSILON: f64 = 1.0e-9;
pub const DEFAULT_BLOCK_WIDTH: usize = 2_048;
const DENSE_EIGEN_FALLBACK_THRESHOLD: usize = 64;
const MAX_PARTIAL_COMPONENTS: usize = 512;
const FALLBACK_GRAM_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MIN_GRAM_BUDGET_BYTES: u64 = 512 * 1024 * 1024;
pub const DEFAULT_LD_WINDOW: usize = 51;
const DEFAULT_LD_RIDGE: f64 = 1.0e-3;
const MIN_LD_WEIGHT: f64 = 1.0e-6;

#[inline]
fn select_top_k_desc(ordering: &mut [(usize, f64)], k: usize) -> usize {
    let mid = k.min(ordering.len());
    if mid == 0 {
        return 0;
    }
    let desc =
        |a: &(usize, f64), b: &(usize, f64)| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal);
    ordering.select_nth_unstable_by(mid - 1, desc);
    ordering[..mid].sort_by(desc);
    mid
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CovarianceComputationMode {
    Dense,
    Partial,
}

fn default_gram_budget_usize() -> usize {
    default_gram_budget_bytes()
        .min(usize::MAX as u64)
        .try_into()
        .unwrap()
}

fn default_gram_budget_bytes() -> u64 {
    static DEFAULT_GRAM_BUDGET_BYTES: OnceLock<u64> = OnceLock::new();

    *DEFAULT_GRAM_BUDGET_BYTES.get_or_init(compute_default_gram_budget_bytes)
}

fn compute_default_gram_budget_bytes() -> u64 {
    match detect_total_memory_bytes() {
        Some(total) if total > 0 => {
            let target = total.saturating_mul(3) / 4;
            target.max(MIN_GRAM_BUDGET_BYTES).min(total).max(1)
        }
        _ => FALLBACK_GRAM_BUDGET_BYTES,
    }
}

fn detect_total_memory_bytes() -> Option<u64> {
    let mut system = System::new_all();
    system.refresh_memory();
    system.total_memory().checked_mul(1024)
}

fn gram_matrix_budget_bytes() -> usize {
    match std::env::var("GNOMON_GRAM_BUDGET_BYTES") {
        Ok(value) => match value.parse::<u64>() {
            Ok(parsed) if parsed == 0 => usize::MAX,
            Ok(parsed) => (parsed.min(usize::MAX as u64)) as usize,
            Err(_) => default_gram_budget_usize(),
        },
        Err(_) => default_gram_budget_usize(),
    }
}

fn gram_matrix_size_bytes(n: usize) -> Option<usize> {
    n.checked_mul(n)?.checked_mul(core::mem::size_of::<f64>())
}

fn covariance_computation_mode(n: usize, budget_bytes: usize) -> CovarianceComputationMode {
    match gram_matrix_size_bytes(n) {
        Some(bytes) if bytes <= budget_bytes => CovarianceComputationMode::Dense,
        _ => CovarianceComputationMode::Partial,
    }
}

#[derive(Clone, Debug, Default)]
pub struct FitOptions {
    pub ld: Option<LdConfig>,
}

#[derive(Clone, Debug)]
pub enum LdWindow {
    Sites(usize),
    BasePairs(u64),
}

#[derive(Clone, Debug, Default)]
pub struct LdConfig {
    pub window: Option<LdWindow>,
    pub ridge: Option<f64>,
    pub variant_keys: Option<Arc<Vec<VariantKey>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LdWeights {
    pub weights: Vec<f64>,
    pub window: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bp_window: Option<u64>,
    pub ridge: f64,
}

#[derive(Clone, Debug)]
struct LdBpWindowRange {
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
enum LdResolvedWindow {
    Sites {
        size: usize,
    },
    BasePairs {
        span_bp: u64,
        ranges: Arc<[LdBpWindowRange]>,
        capacity: usize,
    },
}

#[derive(Clone, Debug)]
struct LdResolvedConfig {
    window: LdResolvedWindow,
    ridge: f64,
}

impl LdResolvedConfig {
    fn window_capacity(&self) -> usize {
        match &self.window {
            LdResolvedWindow::Sites { size } => *size,
            LdResolvedWindow::BasePairs { capacity, .. } => *capacity,
        }
    }

    fn bp_window(&self) -> Option<u64> {
        match &self.window {
            LdResolvedWindow::Sites { .. } => None,
            LdResolvedWindow::BasePairs { span_bp, .. } => Some(*span_bp),
        }
    }
}

impl FitOptions {
    fn resolved_ld(
        &self,
        observed_variants: usize,
    ) -> Result<Option<LdResolvedConfig>, HwePcaError> {
        let Some(cfg) = &self.ld else {
            return Ok(None);
        };

        if observed_variants == 0 {
            return Ok(None);
        }

        let window_spec = cfg
            .window
            .clone()
            .unwrap_or(LdWindow::Sites(DEFAULT_LD_WINDOW));

        let ridge = cfg.ridge.unwrap_or(DEFAULT_LD_RIDGE);
        if !(ridge.is_finite() && ridge > 0.0) {
            return Err(HwePcaError::InvalidInput(
                "LD weighting ridge must be positive and finite",
            ));
        }

        let window = match window_spec {
            LdWindow::Sites(mut window) => {
                if window == 0 {
                    return Err(HwePcaError::InvalidInput(
                        "LD weighting window must be at least one variant",
                    ));
                }
                window = window.min(observed_variants.max(1));
                if window == 0 {
                    window = 1;
                }
                if window % 2 == 0 {
                    window = window.saturating_sub(1);
                    if window == 0 {
                        window = 1;
                    }
                }
                LdResolvedWindow::Sites { size: window }
            }
            LdWindow::BasePairs(span_bp) => {
                let keys = cfg.variant_keys.as_ref().ok_or_else(|| {
                    HwePcaError::InvalidInput("LD base-pair window requires variant positions")
                })?;
                if keys.len() != observed_variants {
                    return Err(HwePcaError::InvalidInput(
                        "LD base-pair window requires positions for all variants",
                    ));
                }
                let (ranges, capacity) = compute_ld_bp_ranges(keys, span_bp)?;
                LdResolvedWindow::BasePairs {
                    span_bp,
                    ranges,
                    capacity,
                }
            }
        };

        Ok(Some(LdResolvedConfig { window, ridge }))
    }
}

fn compute_ld_bp_ranges(
    keys: &[VariantKey],
    span_bp: u64,
) -> Result<(Arc<[LdBpWindowRange]>, usize), HwePcaError> {
    if keys.is_empty() {
        return Err(HwePcaError::InvalidInput(
            "LD base-pair window requires at least one variant",
        ));
    }

    let half_span = span_bp / 2;

    let mut left_bounds = vec![0usize; keys.len()];
    let mut start = 0usize;

    for (idx, key) in keys.iter().enumerate() {
        if idx == 0 {
            left_bounds[idx] = 0;
            continue;
        }

        if keys[idx - 1].chromosome != key.chromosome {
            start = idx;
        }

        while start < idx {
            let candidate = &keys[start];
            if candidate.chromosome != key.chromosome {
                start += 1;
                continue;
            }
            let delta = key.position.saturating_sub(candidate.position);
            if delta > half_span {
                start += 1;
                continue;
            }
            break;
        }

        left_bounds[idx] = start.min(idx);
    }

    let mut ranges = Vec::with_capacity(keys.len());
    let mut capacity = 1usize;
    let mut right = 0usize;

    for (idx, key) in keys.iter().enumerate() {
        if right < idx {
            right = idx;
        }

        while right < keys.len() {
            let candidate = &keys[right];
            if candidate.chromosome != key.chromosome {
                break;
            }
            let delta = candidate.position.saturating_sub(key.position);
            if delta > half_span {
                break;
            }
            right += 1;
        }

        let start_idx = left_bounds[idx].min(idx);
        let end_idx = right.max(idx + 1);
        let width = end_idx - start_idx;
        capacity = capacity.max(width.max(1));
        ranges.push(LdBpWindowRange {
            start: start_idx,
            end: end_idx,
        });
    }

    Ok((ranges.into_boxed_slice().into(), capacity.max(1)))
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f64);

// SAFETY: `SendPtr` is only ever constructed from buffers that are owned by the
// current thread and remain alive for the entire duration of the scoped thread
// in which the pointer is sent. We only move the raw pointer between threads to
// avoid borrow checker restrictions; the pointed-to memory is still uniquely
// owned, and channel coordination guarantees that at most one thread accesses a
// given buffer at a time.
unsafe impl Send for SendPtr {}

struct ParallelismGuard {
    previous: Par,
}

impl ParallelismGuard {
    fn new() -> Self {
        let previous = get_global_parallelism();
        let desired = Par::rayon(rayon::current_num_threads());
        set_global_parallelism(desired);
        Self { previous }
    }

    fn active_parallelism(&self) -> Par {
        get_global_parallelism()
    }
}

impl Drop for ParallelismGuard {
    fn drop(&mut self) {
        set_global_parallelism(self.previous);
    }
}

#[derive(Debug)]
struct OperatorError;

pub trait VariantBlockSource {
    type Error;

    fn n_samples(&self) -> usize;
    fn n_variants(&self) -> usize;
    fn reset(&mut self) -> Result<(), Self::Error>;
    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error>;

    fn progress_bytes(&self) -> Option<(u64, Option<u64>)> {
        None
    }

    fn progress_variants(&self) -> Option<(usize, Option<usize>)> {
        None
    }
}

pub struct DenseBlockSource<'a> {
    data: &'a [f64],
    dims: (usize, usize),
    cursor: usize,
}

impl<'a> DenseBlockSource<'a> {
    pub fn new(data: &'a [f64], n_samples: usize, n_variants: usize) -> Result<Self, HwePcaError> {
        if n_samples == 0 {
            return Err(HwePcaError::InvalidInput(
                "DenseBlockSource: n_samples must be positive",
            ));
        }
        if n_variants == 0 {
            return Err(HwePcaError::InvalidInput(
                "DenseBlockSource: n_variants must be positive",
            ));
        }
        let expected = n_samples
            .checked_mul(n_variants)
            .ok_or_else(|| HwePcaError::InvalidInput("DenseBlockSource: dimension overflow"))?;
        if data.len() != expected {
            return Err(HwePcaError::InvalidInput(
                "DenseBlockSource: data length does not match dimensions",
            ));
        }
        Ok(Self {
            data,
            dims: (n_samples, n_variants),
            cursor: 0,
        })
    }
}

impl<'a> VariantBlockSource for DenseBlockSource<'a> {
    type Error = Infallible;

    fn n_samples(&self) -> usize {
        self.dims.0
    }

    fn n_variants(&self) -> usize {
        self.dims.1
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.cursor = 0;
        Ok(())
    }

    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error> {
        if max_variants == 0 {
            return Ok(0);
        }
        let remaining = self.n_variants().saturating_sub(self.cursor);
        if remaining == 0 {
            return Ok(0);
        }
        let ncols = min(max_variants, remaining);
        let nrows = self.n_samples();
        let len = nrows * ncols;
        let start = self.cursor * nrows;
        let end = start + len;
        storage[..len].copy_from_slice(&self.data[start..end]);
        self.cursor += ncols;
        Ok(ncols)
    }
}

#[derive(Debug)]
pub enum HwePcaError {
    InvalidInput(&'static str),
    Source(Box<dyn Error + Send + Sync + 'static>),
    Eigen(String),
}

impl fmt::Display for HwePcaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HwePcaError::InvalidInput(msg) => f.write_str(msg),
            HwePcaError::Source(err) => write!(f, "source error: {err}"),
            HwePcaError::Eigen(msg) => f.write_str(msg),
        }
    }
}

impl Error for HwePcaError {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HweScaler {
    frequencies: Vec<f64>,
    scales: Vec<f64>,
}

impl HweScaler {
    fn new(frequencies: Vec<f64>, scales: Vec<f64>) -> Self {
        Self {
            frequencies,
            scales,
        }
    }

    pub fn allele_frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    pub fn variant_scales(&self) -> &[f64] {
        &self.scales
    }

    pub(crate) fn standardize_block(
        &self,
        block: MatMut<'_, f64>,
        variant_range: Range<usize>,
        par: Par,
    ) {
        let start = variant_range.start;
        let end = variant_range.end;
        let freqs = &self.frequencies[start..end];
        let scales = &self.scales[start..end];
        standardize_block_impl(block, freqs, scales, par);
    }

    pub(crate) fn standardize_block_with_mask(
        &self,
        block: MatMut<'_, f64>,
        variant_range: Range<usize>,
        presence_out: MatMut<'_, f64>,
        par: Par,
    ) {
        let start = variant_range.start;
        let end = variant_range.end;
        let freqs = &self.frequencies[start..end];
        let filled = freqs.len();

        debug_assert_eq!(filled, block.ncols());
        debug_assert_eq!(filled, presence_out.ncols());
        debug_assert_eq!(block.nrows(), presence_out.nrows());

        let scales = &self.scales[start..end];
        debug_assert_eq!(filled, scales.len());

        let mut block = block.subcols_mut(0, filled);
        let mut presence_out = presence_out.subcols_mut(0, filled);

        let apply_standardization =
            |column: ColMut<'_, f64>, presence_col: ColMut<'_, f64>, mean: f64, inv: f64| {
                let contiguous_values = column
                    .try_as_col_major_mut()
                    .expect("projection block column must be contiguous");
                let contiguous_mask = presence_col
                    .try_as_col_major_mut()
                    .expect("projection mask column must be contiguous");
                let values = contiguous_values.as_slice_mut();
                let mask = contiguous_mask.as_slice_mut();
                standardize_column_with_mask(values, mask, mean, inv);
            };

        let use_parallel = filled >= 32 && par.degree() > 1;

        if use_parallel {
            presence_out
                .par_col_iter_mut()
                .zip(block.par_col_iter_mut())
                .enumerate()
                .for_each(|(idx, (presence_col, column))| {
                    let freq = freqs[idx];
                    let scale = scales[idx];
                    let mean = 2.0 * freq;
                    let denom = scale.max(HWE_SCALE_FLOOR);
                    let inv = if denom > 0.0 { denom.recip() } else { 0.0 };
                    apply_standardization(column, presence_col, mean, inv);
                });
        } else {
            for idx in 0..filled {
                let presence_col = presence_out.rb_mut().col_mut(idx);
                let column = block.rb_mut().col_mut(idx);
                let freq = freqs[idx];
                let scale = scales[idx];
                let mean = 2.0 * freq;
                let denom = scale.max(HWE_SCALE_FLOOR);
                let inv = if denom > 0.0 { denom.recip() } else { 0.0 };
                apply_standardization(column, presence_col, mean, inv);
            }
        }
    }
}

fn standardize_block_impl(block: MatMut<'_, f64>, freqs: &[f64], scales: &[f64], par: Par) {
    let filled = freqs.len();

    debug_assert_eq!(filled, block.ncols());
    debug_assert_eq!(filled, scales.len());
    let block = block.subcols_mut(0, filled);

    let apply_standardization = |mut column: ColMut<'_, f64>, mean: f64, inv: f64| {
        if inv == 0.0 {
            column.fill(0.0);
            return;
        }

        let contiguous = column
            .try_as_col_major_mut()
            .expect("projection block column must be contiguous");
        let values = contiguous.as_slice_mut();
        standardize_column_simd(values, mean, inv);
    };

    let use_parallel = filled >= 32 && par.degree() > 1;

    if use_parallel {
        block
            .par_col_iter_mut()
            .enumerate()
            .for_each(|(idx, column)| {
                let freq = freqs[idx];
                let scale = scales[idx];
                let mean = 2.0 * freq;
                let denom = scale.max(HWE_SCALE_FLOOR);
                let inv = if denom > 0.0 { denom.recip() } else { 0.0 };
                apply_standardization(column, mean, inv);
            });
    } else {
        for (idx, column) in block.col_iter_mut().enumerate() {
            let freq = freqs[idx];
            let scale = scales[idx];
            let mean = 2.0 * freq;
            let denom = scale.max(HWE_SCALE_FLOOR);
            let inv = if denom > 0.0 { denom.recip() } else { 0.0 };
            apply_standardization(column, mean, inv);
        }
    }
}

pub(crate) fn apply_ld_weights(
    block: MatMut<'_, f64>,
    variant_range: Range<usize>,
    weights: &[f64],
) {
    let start = variant_range.start.min(weights.len());
    let end = variant_range.end.min(weights.len());
    if end <= start {
        return;
    }
    let slice = &weights[start..end];
    let columns = block.subcols_mut(0, slice.len());
    for (column, &weight) in columns.col_iter_mut().zip(slice.iter()) {
        if (weight - 1.0).abs() < f64::EPSILON {
            continue;
        }
        zip!(column).for_each(|unzip!(value)| {
            *value *= weight;
        });
    }
}

struct VariantStatsCache {
    frequencies: Vec<f64>,
    scales: Vec<f64>,
    block_sums: Vec<f64>,
    block_calls: Vec<usize>,
    finalized_len: Option<usize>,
    write_pos: usize,
}

impl VariantStatsCache {
    fn new(block_capacity: usize, variant_capacity_hint: usize) -> Self {
        let frequencies = Vec::with_capacity(variant_capacity_hint);
        let scales = Vec::with_capacity(variant_capacity_hint);
        Self {
            frequencies,
            scales,
            block_sums: vec![0.0; block_capacity],
            block_calls: vec![0usize; block_capacity],
            finalized_len: None,
            write_pos: 0,
        }
    }

    fn is_finalized(&self) -> bool {
        self.finalized_len.is_some()
    }

    fn ensure_statistics(&mut self, block: MatRef<'_, f64>, variant_range: Range<usize>, par: Par) {
        if self.is_finalized() {
            return;
        }

        debug_assert!(variant_range.start == self.write_pos);

        let filled = block.ncols();
        {
            let sums_slice = &mut self.block_sums[..filled];
            let calls_slice = &mut self.block_calls[..filled];

            let use_parallel = filled >= 32 && par.degree() > 1;

            if use_parallel {
                sums_slice
                    .par_iter_mut()
                    .zip(calls_slice.par_iter_mut())
                    .zip(block.par_col_iter())
                    .for_each(|((sum_slot, calls_slot), column)| {
                        let contiguous = column
                            .try_as_col_major()
                            .expect("variant block column must be contiguous");
                        let (sum, calls) = sum_and_count_finite(contiguous.as_slice());
                        *sum_slot = sum;
                        *calls_slot = calls;
                    });
            } else {
                sums_slice
                    .iter_mut()
                    .zip(calls_slice.iter_mut())
                    .zip(block.col_iter())
                    .for_each(|((sum_slot, calls_slot), column)| {
                        let contiguous = column
                            .try_as_col_major()
                            .expect("variant block column must be contiguous");
                        let (sum, calls) = sum_and_count_finite(contiguous.as_slice());
                        *sum_slot = sum;
                        *calls_slot = calls;
                    });
            }
        }

        let end = variant_range.end;
        self.ensure_capacity(end);

        let freq_slice = &mut self.frequencies[variant_range.clone()];
        let scale_slice = &mut self.scales[variant_range.clone()];
        let sums_slice = &self.block_sums[..filled];
        let calls_slice = &self.block_calls[..filled];

        for idx in 0..filled {
            let sum = sums_slice[idx];
            let calls = calls_slice[idx];
            if calls == 0 {
                freq_slice[idx] = 0.0;
                scale_slice[idx] = HWE_SCALE_FLOOR;

                continue;
            }

            let mean_genotype = sum / (calls as f64);
            let allele_freq = (mean_genotype / 2.0).clamp(0.0, 1.0);
            let variance = (2.0 * allele_freq * (1.0 - allele_freq)).max(HWE_VARIANCE_EPSILON);
            let derived_scale = variance.sqrt();

            freq_slice[idx] = allele_freq;
            scale_slice[idx] = if derived_scale < HWE_SCALE_FLOOR {
                HWE_SCALE_FLOOR
            } else {
                derived_scale
            };
        }

        self.write_pos = end;

        debug_assert!(self.frequencies.len() >= end);
        debug_assert!(self.scales.len() >= end);
    }

    fn finalize(&mut self) {
        self.frequencies.truncate(self.write_pos);
        self.scales.truncate(self.write_pos);
        self.finalized_len = Some(self.write_pos);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.finalized_len.unwrap_or(self.write_pos)
    }

    fn into_scaler(self) -> Option<HweScaler> {
        if self.finalized_len.is_some() {
            Some(HweScaler::new(self.frequencies, self.scales))
        } else {
            None
        }
    }

    fn ensure_capacity(&mut self, required: usize) {
        if self.frequencies.len() >= required {
            return;
        }

        let freq_capacity = self.frequencies.capacity();
        if freq_capacity < required {
            let block_capacity = self.block_sums.len();
            let growth_from_capacity = freq_capacity + freq_capacity / 2;
            let growth_from_block = self.write_pos.saturating_add(block_capacity);
            let mut target = required
                .max(growth_from_capacity)
                .max(growth_from_block)
                .max(1);

            if target <= freq_capacity {
                target = required;
            }

            let additional_capacity = target - freq_capacity;
            self.frequencies.reserve_exact(additional_capacity);
            self.scales.reserve_exact(additional_capacity);
        }

        let additional = required - self.frequencies.len();
        self.frequencies
            .extend(std::iter::repeat(0.0).take(additional));
        self.scales.extend(std::iter::repeat(0.0).take(additional));
    }
}

#[derive(Clone, Copy, Debug)]
enum SimdLaneSelection {
    Lanes4,
    Lanes2,
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
const _: () = {
    // Ensure the two-lane variant stays in use on targets that never select it at runtime.
    let _ = SimdLaneSelection::Lanes2;
};

#[inline(always)]
fn record_simd_lane_diagnostic(
    stage: &'static str,
    selection: SimdLaneSelection,
) -> SimdLaneSelection {
    log::debug!("SIMD lane selection stage {stage} -> {selection:?}");
    selection
}

fn detected_simd_lane_selection() -> SimdLaneSelection {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        static DETECTED: OnceLock<SimdLaneSelection> = OnceLock::new();
        return *DETECTED.get_or_init(|| {
            let selection =
                if cfg!(target_feature = "avx") && std::arch::is_x86_feature_detected!("avx") {
                    SimdLaneSelection::Lanes4
                } else {
                    SimdLaneSelection::Lanes2
                };
            record_simd_lane_diagnostic("x86 runtime detection", selection)
        });
    }

    #[cfg(all(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        any(target_arch = "aarch64", target_arch = "wasm32")
    ))]
    {
        return record_simd_lane_diagnostic(
            "default lanes4 architecture",
            SimdLaneSelection::Lanes4,
        );
    }

    #[cfg(all(not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    ))))]
    {
        return record_simd_lane_diagnostic("portable fallback", SimdLaneSelection::Lanes2);
    }
}

#[inline(always)]
fn standardize_column_simd(values: &mut [f64], mean: f64, inv: f64) {
    match detected_simd_lane_selection() {
        #[cfg(any(
            target_feature = "avx",
            target_arch = "aarch64",
            target_arch = "wasm32"
        ))]
        SimdLaneSelection::Lanes4 => {
            standardize_column_simd_lanes4(values, mean, inv);
        }
        _ => standardize_column_simd_impl::<2>(values, mean, inv),
    }
}

#[cfg(any(
    target_feature = "avx",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[inline(always)]
fn standardize_column_simd_lanes4(values: &mut [f64], mean: f64, inv: f64) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // SAFETY: The AVX-specific implementation is only compiled on x86 targets
    // and this branch is taken after `detected_simd_lane_selection` confirmed
    // that the CPU supports the required feature set.
    unsafe {
        standardize_column_simd_avx(values, mean, inv);
    }

    #[cfg(all(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        any(target_arch = "aarch64", target_arch = "wasm32")
    ))]
    {
        standardize_column_simd_impl::<4>(values, mean, inv);
    }
}

#[cfg(any(
    target_feature = "avx",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
/// # Safety
/// The caller must ensure the current CPU supports AVX instructions. All call
/// sites guard this by checking `std::arch::is_x86_feature_detected!("avx")` or
/// by only invoking it in configurations where AVX is guaranteed to be present.
unsafe fn standardize_column_simd_avx(values: &mut [f64], mean: f64, inv: f64) {
    standardize_column_simd_impl::<4>(values, mean, inv);
}

#[inline(always)]
fn standardize_column_simd_impl<const LANES: usize>(values: &mut [f64], mean: f64, inv: f64)
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mean_simd = Simd::<f64, LANES>::splat(mean);
    let inv_simd = Simd::<f64, LANES>::splat(inv);
    let zero = Simd::<f64, LANES>::splat(0.0);

    let (chunks, remainder) = values.as_chunks_mut::<LANES>();
    for chunk in chunks {
        let lane = Simd::<f64, LANES>::from_array(*chunk);
        let mask = lane.is_finite();
        let standardized = (lane - mean_simd) * inv_simd;
        let result = mask.select(standardized, zero);
        *chunk = result.to_array();
    }

    for value in remainder {
        let raw = *value;
        *value = if raw.is_finite() {
            (raw - mean) * inv
        } else {
            0.0
        };
    }
}

#[inline(always)]
fn standardize_column_with_mask(values: &mut [f64], mask: &mut [f64], mean: f64, inv: f64) {
    debug_assert_eq!(values.len(), mask.len());

    if inv == 0.0 {
        for (value, mask_value) in values.iter_mut().zip(mask.iter_mut()) {
            let raw = *value;
            *mask_value = if raw.is_finite() { 1.0 } else { 0.0 };
            *value = 0.0;
        }
        return;
    }

    standardize_column_with_mask_simd(values, mask, mean, inv);
}

#[inline(always)]
fn standardize_column_with_mask_simd(values: &mut [f64], mask: &mut [f64], mean: f64, inv: f64) {
    match detected_simd_lane_selection() {
        #[cfg(any(
            target_feature = "avx",
            target_arch = "aarch64",
            target_arch = "wasm32"
        ))]
        SimdLaneSelection::Lanes4 => {
            standardize_column_with_mask_simd_lanes4(values, mask, mean, inv);
        }
        _ => standardize_column_with_mask_simd_impl::<2>(values, mask, mean, inv),
    }
}

#[cfg(any(
    target_feature = "avx",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[inline(always)]
fn standardize_column_with_mask_simd_lanes4(
    values: &mut [f64],
    mask: &mut [f64],
    mean: f64,
    inv: f64,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // SAFETY: On x86 we only reach this branch when runtime detection selects
    // the four-lane configuration, which implies AVX availability.
    unsafe {
        standardize_column_with_mask_simd_avx(values, mask, mean, inv);
    }

    #[cfg(all(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        any(target_arch = "aarch64", target_arch = "wasm32")
    ))]
    {
        standardize_column_with_mask_simd_impl::<4>(values, mask, mean, inv);
    }
}

#[cfg(any(
    target_feature = "avx",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
/// # Safety
/// The caller must ensure the CPU supports AVX instructions. All invocations
/// are conditioned on runtime feature detection or target configurations that
/// guarantee AVX availability.
unsafe fn standardize_column_with_mask_simd_avx(
    values: &mut [f64],
    mask: &mut [f64],
    mean: f64,
    inv: f64,
) {
    standardize_column_with_mask_simd_impl::<4>(values, mask, mean, inv);
}

#[inline(always)]
fn standardize_column_with_mask_simd_impl<const LANES: usize>(
    values: &mut [f64],
    mask: &mut [f64],
    mean: f64,
    inv: f64,
) where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mean_simd = Simd::<f64, LANES>::splat(mean);
    let inv_simd = Simd::<f64, LANES>::splat(inv);
    let zero = Simd::<f64, LANES>::splat(0.0);
    let one = Simd::<f64, LANES>::splat(1.0);

    let (value_chunks, value_remainder) = values.as_chunks_mut::<LANES>();
    let (mask_chunks, mask_remainder) = mask.as_chunks_mut::<LANES>();

    debug_assert_eq!(value_chunks.len(), mask_chunks.len());
    debug_assert_eq!(value_remainder.len(), mask_remainder.len());

    for (value_chunk, mask_chunk) in value_chunks.iter_mut().zip(mask_chunks.iter_mut()) {
        let lane = Simd::<f64, LANES>::from_array(*value_chunk);
        let finite_mask = lane.is_finite();
        let standardized = (lane - mean_simd) * inv_simd;
        let result = finite_mask.select(standardized, zero);
        *value_chunk = result.to_array();
        let mask_values = finite_mask.select(one, zero);
        *mask_chunk = mask_values.to_array();
    }

    for (value, mask_value) in value_remainder.iter_mut().zip(mask_remainder.iter_mut()) {
        let raw = *value;
        if raw.is_finite() {
            *mask_value = 1.0;
            *value = (raw - mean) * inv;
        } else {
            *mask_value = 0.0;
            *value = 0.0;
        }
    }
}

#[cfg(test)]
#[inline(always)]
fn standardize_column_simd_full(values: &mut [f64], mean: f64, inv: f64) {
    match detected_simd_lane_selection() {
        #[cfg(any(
            target_feature = "avx",
            target_arch = "aarch64",
            target_arch = "wasm32"
        ))]
        SimdLaneSelection::Lanes4 => {
            standardize_column_simd_full_lanes4(values, mean, inv);
        }
        _ => standardize_column_simd_full_impl::<2>(values, mean, inv),
    }
}

#[cfg(test)]
#[inline(always)]
fn standardize_column_simd_full_lanes4(values: &mut [f64], mean: f64, inv: f64) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // SAFETY: The AVX path is only taken on x86 targets when the runtime lane
    // selection confirmed AVX support.
    unsafe {
        standardize_column_simd_full_avx(values, mean, inv);
    }

    #[cfg(all(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        any(target_arch = "aarch64", target_arch = "wasm32")
    ))]
    {
        standardize_column_simd_full_impl::<4>(values, mean, inv);
    }
}

#[cfg(test)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
/// # Safety
/// Callers must guarantee AVX availability; runtime dispatch ensures that the
/// function is only invoked when the CPU advertises the capability.
unsafe fn standardize_column_simd_full_avx(values: &mut [f64], mean: f64, inv: f64) {
    standardize_column_simd_full_impl::<4>(values, mean, inv);
}

#[cfg(test)]
#[inline(always)]
fn standardize_column_simd_full_impl<const LANES: usize>(values: &mut [f64], mean: f64, inv: f64)
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mean_simd = Simd::<f64, LANES>::splat(mean);
    let inv_simd = Simd::<f64, LANES>::splat(inv);

    let (chunks, remainder) = values.as_chunks_mut::<LANES>();
    for chunk in chunks {
        let lane = Simd::<f64, LANES>::from_array(*chunk);
        let standardized = (lane - mean_simd) * inv_simd;
        *chunk = standardized.to_array();
    }

    for value in remainder {
        *value = (*value - mean) * inv;
    }
}

#[inline(always)]
fn sum_and_count_finite(values: &[f64]) -> (f64, usize) {
    match detected_simd_lane_selection() {
        #[cfg(any(
            target_feature = "avx",
            target_arch = "aarch64",
            target_arch = "wasm32"
        ))]
        SimdLaneSelection::Lanes4 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: Runtime detection guaranteed AVX is present before
            // dispatching to the specialized implementation.
            unsafe {
                log::debug!("Invoking AVX sum_and_count_finite implementation");
                return sum_and_count_finite_avx(values);
            }

            #[cfg(all(
                not(any(target_arch = "x86", target_arch = "x86_64")),
                any(target_arch = "aarch64", target_arch = "wasm32")
            ))]
            {
                log::debug!(
                    "Using generic four-lane sum_and_count_finite implementation for non-x86 architecture"
                );
                return sum_and_count_finite_impl::<4>(values);
            }

            #[cfg(all(not(any(
                target_arch = "x86",
                target_arch = "x86_64",
                target_arch = "aarch64",
                target_arch = "wasm32"
            ))))]
            {
                log::warn!(
                    "Falling back to two-lane sum_and_count_finite implementation despite four-lane selection"
                );
                return sum_and_count_finite_impl::<2>(values);
            }
        }
        _ => return sum_and_count_finite_impl::<2>(values),
    }
}

#[inline(always)]
fn sum_and_count_finite_impl<const LANES: usize>(values: &[f64]) -> (f64, usize)
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    let zero = Simd::<f64, LANES>::splat(0.0);

    let (chunks, remainder) = values.as_chunks::<LANES>();
    for chunk in chunks {
        let lane = Simd::<f64, LANES>::from_array(*chunk);
        let mask = lane.is_finite();
        let finite = mask.select(lane, zero);
        sum += finite.reduce_sum();
        count += mask.to_bitmask().count_ones() as usize;
    }

    for &value in remainder {
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }

    (sum, count)
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx"
))]
#[target_feature(enable = "avx")]
/// # Safety
/// Callers must ensure AVX is supported by the running CPU. Runtime feature
/// checks protect all invocations of this function.
unsafe fn sum_and_count_finite_avx(values: &[f64]) -> (f64, usize) {
    sum_and_count_finite_impl::<4>(values)
}

#[derive(Clone, Debug)]
pub struct HwePcaModel {
    n_samples: usize,
    n_variants: usize,
    scaler: HweScaler,
    eigenvalues: Vec<f64>,
    singular_values: Vec<f64>,
    sample_basis: Mat<f64>,
    sample_scores: Mat<f64>,
    loadings: Mat<f64>,
    component_weighted_norms_sq: Vec<f64>,
    variant_keys: Option<Vec<VariantKey>>,
    ld: Option<LdWeights>,
}

impl HwePcaModel {
    pub fn fit_k<S>(source: &mut S, components: usize) -> Result<Self, HwePcaError>
    where
        S: VariantBlockSource + Send,
        S::Error: Error + Send + Sync + 'static,
    {
        let progress = Arc::new(NoopFitProgress::default());
        Self::fit_k_with_options_and_progress(source, components, &FitOptions::default(), &progress)
    }

    pub fn fit_k_with_progress<S, P>(
        source: &mut S,
        components: usize,
        progress: &Arc<P>,
    ) -> Result<Self, HwePcaError>
    where
        S: VariantBlockSource + Send,
        S::Error: Error + Send + Sync + 'static,
        P: FitProgressObserver + Send + Sync + 'static,
    {
        Self::fit_k_with_options_and_progress(source, components, &FitOptions::default(), progress)
    }

    pub fn fit_k_with_options_and_progress<S, P>(
        source: &mut S,
        components: usize,
        options: &FitOptions,
        progress: &Arc<P>,
    ) -> Result<Self, HwePcaError>
    where
        S: VariantBlockSource + Send,
        S::Error: Error + Send + Sync + 'static,
        P: FitProgressObserver + Send + Sync + 'static,
    {
        let n_samples = source.n_samples();
        let n_variants_hint = source.n_variants();

        if n_samples < 2 {
            return Err(HwePcaError::InvalidInput(
                "HWE PCA requires at least two samples",
            ));
        }

        if components == 0 {
            return Err(HwePcaError::InvalidInput(
                "Requested component count must be at least one",
            ));
        }

        let max_rank = n_samples.saturating_sub(1);
        let target_components = components.min(max_rank);

        let parallelism_guard = ParallelismGuard::new();
        let par = parallelism_guard.active_parallelism();
        let block_capacity = if n_variants_hint > 0 {
            min(DEFAULT_BLOCK_WIDTH.max(1), n_variants_hint)
        } else {
            DEFAULT_BLOCK_WIDTH.max(1)
        };
        progress.on_stage_start(FitProgressStage::AlleleStatistics, n_variants_hint);
        let stats_progress =
            StageProgressHandle::new(Arc::clone(progress), FitProgressStage::AlleleStatistics);
        let (scaler, observed_variants) = compute_variant_statistics(
            source,
            block_capacity,
            par,
            stats_progress,
            n_variants_hint,
        )?;

        let ld_config = options.resolved_ld(observed_variants)?;
        let ld_weights = if let Some(ld_cfg) = ld_config {
            Some(compute_ld_weights(
                source,
                &scaler,
                observed_variants,
                block_capacity,
                ld_cfg,
                n_variants_hint,
                progress,
                par,
            )?)
        } else {
            None
        };

        let ld_weights_arc = ld_weights
            .as_ref()
            .map(|ld| Arc::<[f64]>::from(ld.weights.clone().into_boxed_slice()));

        let operator = StandardizedCovarianceOp::new(
            source,
            block_capacity,
            n_variants_hint,
            observed_variants,
            scaler,
            ld_weights_arc.clone(),
        );

        let gram_budget = gram_matrix_budget_bytes();
        let gram_bytes = gram_matrix_size_bytes(n_samples);
        let gram_mode = covariance_computation_mode(n_samples, gram_budget);
        let gram_progress_handle = match gram_mode {
            CovarianceComputationMode::Dense => {
                progress.on_stage_start(FitProgressStage::GramMatrix, n_variants_hint);
                Some(StageProgressHandle::new(
                    Arc::clone(progress),
                    FitProgressStage::GramMatrix,
                ))
            }
            CovarianceComputationMode::Partial => {
                progress.on_stage_start(FitProgressStage::GramMatrix, 0);
                if n_variants_hint > 0 {
                    progress.on_stage_estimate(FitProgressStage::GramMatrix, n_variants_hint);
                }
                None
            }
        };

        if matches!(gram_mode, CovarianceComputationMode::Partial) {
            if let Some(bytes) = gram_bytes {
                log::info!(
                    "Skipping explicit Gram matrix ({} bytes exceeds budget of {} bytes); using matrix-free solver",
                    bytes,
                    gram_budget
                );
            } else {
                log::info!(
                    "Skipping explicit Gram matrix due to size overflow; using matrix-free solver"
                );
            }
        }

        let decomposition_result = compute_covariance_eigenpairs(
            &operator,
            par,
            gram_mode,
            target_components,
            gram_progress_handle.as_ref(),
        );
        let (source, scaler) = operator.into_parts();
        let decomposition = decomposition_result?;

        let variant_count = scaler.variant_scales().len();
        debug_assert_eq!(variant_count, observed_variants);
        if variant_count == 0 {
            return Err(HwePcaError::InvalidInput(
                "HWE PCA requires at least one variant",
            ));
        }

        log::info!("Observed {} variants during first pass", variant_count);

        if gram_progress_handle.is_none() {
            progress.on_stage_total(FitProgressStage::GramMatrix, variant_count);
        }
        progress.on_stage_finish(FitProgressStage::GramMatrix);

        if decomposition.values.is_empty() {
            return Err(HwePcaError::Eigen(
                "All eigenvalues are numerically zero; increase cohort size or review input data"
                    .into(),
            ));
        }

        let (mut singular_values, mut sample_scores) =
            build_sample_scores(n_samples, &decomposition);

        let mut loadings = compute_variant_loadings(
            source,
            &scaler,
            variant_count,
            block_capacity,
            decomposition.vectors.as_ref(),
            &singular_values,
            ld_weights_arc.as_deref(),
            progress,
            par,
        )?;

        let component_weighted_norms_sq = renormalize_variant_loadings(
            loadings.as_mut(),
            &mut singular_values,
            sample_scores.as_mut(),
            ld_weights_arc.as_deref(),
        );

        Ok(Self {
            n_samples,
            n_variants: variant_count,
            scaler,
            eigenvalues: decomposition.values,
            singular_values,
            sample_basis: decomposition.vectors,
            sample_scores,
            loadings,
            component_weighted_norms_sq,
            variant_keys: None,
            ld: ld_weights,
        })
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn n_variants(&self) -> usize {
        self.n_variants
    }

    pub fn scaler(&self) -> &HweScaler {
        &self.scaler
    }

    pub fn components(&self) -> usize {
        self.eigenvalues.len()
    }

    pub fn explained_variance(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Returns the singular values after renormalizing the variant loadings.
    ///
    /// These values remain consistent with [`HwePcaModel::variant_loadings`] and
    /// [`HwePcaModel::sample_scores`] after the post-fit rescaling step. Use
    /// [`HwePcaModel::canonical_singular_values`] or [`HwePcaModel::explained_variance`]
    /// when deriving explained variance in the classical PCA metric.
    pub fn singular_values(&self) -> &[f64] {
        &self.singular_values
    }

    /// Returns the canonical singular values that satisfy σ²/(n−1)=λ.
    pub fn canonical_singular_values(&self) -> Vec<f64> {
        let scale = self.n_samples.saturating_sub(1) as f64;
        if scale == 0.0 {
            return vec![0.0; self.eigenvalues.len()];
        }

        self.eigenvalues
            .iter()
            .map(|&lambda| (lambda * scale).max(0.0).sqrt())
            .collect()
    }

    pub fn explained_variance_ratio(&self) -> Vec<f64> {
        let total: f64 = self.eigenvalues.iter().copied().sum();
        if total > 0.0 {
            self.eigenvalues
                .iter()
                .map(|&lambda| lambda / total)
                .collect()
        } else {
            vec![0.0; self.eigenvalues.len()]
        }
    }

    pub fn sample_basis(&self) -> MatRef<'_, f64> {
        self.sample_basis.as_ref()
    }

    pub fn sample_scores(&self) -> MatRef<'_, f64> {
        self.sample_scores.as_ref()
    }

    pub fn variant_loadings(&self) -> MatRef<'_, f64> {
        self.loadings.as_ref()
    }

    pub fn component_weighted_norms_sq(&self) -> &[f64] {
        &self.component_weighted_norms_sq
    }

    pub fn set_variant_keys(&mut self, keys: Option<Vec<VariantKey>>) {
        self.variant_keys = keys;
    }

    pub fn variant_keys(&self) -> Option<&[VariantKey]> {
        self.variant_keys.as_deref()
    }

    pub fn ld(&self) -> Option<&LdWeights> {
        self.ld.as_ref()
    }
}

struct Eigenpairs {
    values: Vec<f64>,
    vectors: Mat<f64>,
}

#[derive(Debug)]
struct DenseSymmetricOp<'a> {
    matrix: MatRef<'a, f64>,
}

impl<'a> LinOp<f64> for DenseSymmetricOp<'a> {
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        let _ = (rhs_ncols, par);
        StackReq::empty()
    }

    fn nrows(&self) -> usize {
        self.matrix.nrows()
    }

    fn ncols(&self) -> usize {
        self.matrix.ncols()
    }

    fn apply(&self, mut out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, par: Par, _: &mut MemStack) {
        matmul(out.rb_mut(), Accum::Replace, self.matrix, rhs, 1.0, par);
    }

    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}

struct StandardizedCovarianceOp<'a, S, P>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync + 'static,
{
    apply_lock: Mutex<()>,
    source: Mutex<&'a mut S>,
    n_samples: usize,
    n_variants_hint: usize,
    block_capacity: usize,
    scale: f64,
    scaler: HweScaler,
    observed_variants: usize,
    ld_weights: Option<Arc<[f64]>>,
    error: Mutex<Option<HwePcaError>>,
    marker: PhantomData<P>,
}

impl<'a, S, P> StandardizedCovarianceOp<'a, S, P>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync + 'static,
{
    fn new(
        source: &'a mut S,
        block_capacity: usize,
        n_variants_hint: usize,
        observed_variants: usize,
        scaler: HweScaler,
        ld_weights: Option<Arc<[f64]>>,
    ) -> Self {
        let n_samples = source.n_samples();
        let scale = 1.0 / ((n_samples - 1) as f64);
        Self {
            apply_lock: Mutex::new(()),
            source: Mutex::new(source),
            n_samples,
            n_variants_hint,
            block_capacity,
            scale,
            scaler,
            observed_variants,
            ld_weights,
            error: Mutex::new(None),
            marker: PhantomData,
        }
    }

    fn into_parts(self) -> (&'a mut S, HweScaler) {
        let Self {
            apply_lock: _,
            source,
            n_samples: _,
            n_variants_hint: _,
            block_capacity: _,
            scale: _,
            scaler,
            observed_variants: _,
            ld_weights: _,
            error: _,
            marker: _,
        } = self;
        (
            source
                .into_inner()
                .expect("covariance source mutex poisoned during teardown"),
            scaler,
        )
    }

    fn take_error(&self) -> Option<HwePcaError> {
        self.error
            .lock()
            .expect("operator error mutex poisoned")
            .take()
    }

    fn fail_invalid(&self, msg: &'static str) -> ! {
        self.record_error(HwePcaError::InvalidInput(msg))
    }

    fn record_error(&self, err: HwePcaError) -> ! {
        let mut slot = self.error.lock().expect("operator error mutex poisoned");
        if slot.is_none() {
            *slot = Some(err);
        }
        std::panic::panic_any(OperatorError);
    }

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn standardize_block_in_place(
        &self,
        mut block: MatMut<'_, f64>,
        variant_range: Range<usize>,
        par: Par,
    ) {
        self.scaler
            .standardize_block(block.as_mut(), variant_range.clone(), par);
        if let Some(weights) = &self.ld_weights {
            apply_ld_weights(block, variant_range, weights);
        }
    }
}

impl<'a, S, P> fmt::Debug for StandardizedCovarianceOp<'a, S, P>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StandardizedCovarianceOp")
            .field("n_samples", &self.n_samples)
            .field("n_variants_hint", &self.n_variants_hint)
            .field("observed_variants", &self.observed_variants)
            .field("block_capacity", &self.block_capacity)
            .finish()
    }
}

impl<'a, S, P> LinOp<f64> for StandardizedCovarianceOp<'a, S, P>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync + 'static,
{
    fn apply_scratch(&self, rhs_ncols: usize, _: Par) -> StackReq {
        let block_len = self.n_samples * self.block_capacity;
        let block_req = StackReq::new::<f64>(block_len);
        let proj_req = temp_mat_scratch::<f64>(self.block_capacity, rhs_ncols);
        block_req.and(block_req).and(proj_req)
    }

    fn nrows(&self) -> usize {
        self.n_samples
    }

    fn ncols(&self) -> usize {
        self.n_samples
    }

    fn apply(
        &self,
        mut out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        let apply_guard = self
            .apply_lock
            .lock()
            .expect("covariance apply lock poisoned");
        let _ = &apply_guard;

        debug_assert_eq!(out.nrows(), self.n_samples);
        debug_assert_eq!(rhs.nrows(), self.n_samples);

        out.fill(0.0);

        if rhs.ncols() == 0 {
            return;
        }

        let block_len = self.n_samples * self.block_capacity;
        let (buf0_uninit, stack) = stack.make_uninit::<f64>(block_len);
        // SAFETY: `buf0_uninit` was allocated with capacity `block_len` and lives
        // for the entire scope of `apply`. We immediately coerce it to
        // `&mut [f64]` so it can be passed to `VariantBlockSource::next_block_into`,
        // which writes the first `n_samples * filled` entries before we read
        // them. Any remaining slots stay uninitialized but are never observed.
        let buf0 = unsafe {
            std::slice::from_raw_parts_mut(buf0_uninit.as_mut_ptr() as *mut f64, block_len)
        };
        let (buf1_uninit, stack) = stack.make_uninit::<f64>(block_len);
        // SAFETY: Same reasoning as above for `buf0`; the buffer lives long
        // enough and only the initialized prefix is observed.
        let buf1 = unsafe {
            std::slice::from_raw_parts_mut(buf1_uninit.as_mut_ptr() as *mut f64, block_len)
        };
        let mut buffer_slices = [buf0, buf1];
        let [first_slice, second_slice] = &mut buffer_slices;
        let buffer_ptrs = [
            SendPtr(first_slice.as_mut_ptr()),
            SendPtr(second_slice.as_mut_ptr()),
        ];

        let (mut proj_uninit, _) =
            // SAFETY: `temp_mat_uninit` returns an uninitialized matrix view that
            // must not be read before being written. We only use it as the
            // destination of GEMM calls with `Accum::Replace`, which overwrite
            // every element touched. The stack capacity was sized by
            // `apply_scratch`, so the backing storage stays live for the entire
            // duration of `apply`.
            unsafe { temp_mat_uninit::<f64, _, _>(self.block_capacity, rhs.ncols(), stack) };
        let mut proj_storage = proj_uninit.as_mat_mut();

        enum PrefetchMessage {
            Data {
                id: usize,
                filled: usize,
                start: usize,
            },
            End,
            Error(HwePcaError),
        }

        let buffer_count = buffer_ptrs.len();
        let (filled_tx, filled_rx) = sync_channel::<PrefetchMessage>(buffer_count);
        let (free_tx, free_rx) = sync_channel::<usize>(buffer_count);
        for id in 0..buffer_count {
            free_tx.send(id).expect("failed to seed prefetch buffers");
        }

        let source_mutex = &self.source;
        let n_samples = self.n_samples;
        let block_capacity = self.block_capacity;
        let observed_total = self.observed_variants;
        let scale = self.scale;
        let block_len = block_len;

        let processed = thread::scope(|scope| {
            let buffer_ptrs_prefetch = buffer_ptrs;
            let filled_sender = filled_tx;
            let free_receiver = free_rx;
            scope.spawn(move || {
                if let Err(err) = {
                    let mut guard = source_mutex
                        .lock()
                        .expect("covariance source mutex poisoned");
                    let source: &mut S = &mut **guard;
                    source.reset().map_err(|e| HwePcaError::Source(Box::new(e)))
                } {
                    let _ = filled_sender.send(PrefetchMessage::Error(err));
                    return;
                }

                let mut start = 0usize;
                while let Ok(id) = free_receiver.recv() {
                    // SAFETY: The raw pointer originated from a mutable slice
                    // in `buffer_slices` and remains valid until the scoped
                    // thread exits. Channel ownership ensures each `id`
                    // corresponds to a single borrower, so the reconstructed
                    // slice is never aliased. Only the prefix written by
                    // `next_block_into` is observed after this call.
                    let buffer_slice = unsafe {
                        std::slice::from_raw_parts_mut(buffer_ptrs_prefetch[id].0, block_len)
                    };
                    let filled_res = {
                        let mut guard = source_mutex
                            .lock()
                            .expect("covariance source mutex poisoned");
                        let source: &mut S = &mut **guard;
                        source.next_block_into(block_capacity, buffer_slice)
                    };

                    match filled_res {
                        Ok(filled) => {
                            if filled == 0 {
                                let _ = filled_sender.send(PrefetchMessage::End);
                                break;
                            }

                            let _ = filled_sender.send(PrefetchMessage::Data { id, filled, start });
                            start += filled;
                        }
                        Err(err) => {
                            let _ = filled_sender
                                .send(PrefetchMessage::Error(HwePcaError::Source(Box::new(err))));
                            break;
                        }
                    }
                }
            });

            let free_sender = free_tx;
            let mut processed = 0usize;
            let buffer_ptrs_compute = buffer_ptrs;
            while let Ok(message) = filled_rx.recv() {
                match message {
                    PrefetchMessage::Data { id, filled, start } => {
                        if start != processed {
                            self.fail_invalid("prefetch produced out-of-order variant ranges");
                        }
                        if start + filled > observed_total {
                            self.fail_invalid(
                                "VariantBlockSource returned more variants than observed",
                            );
                        }

                        // SAFETY: `buffer_ptrs_compute[id]` points to the same
                        // uniquely-owned buffer handed to the prefetch thread.
                        // Scoped threads guarantee the memory outlives this use,
                        // channel coordination prevents concurrent access, and we
                        // restrict all reads to the portion initialized by the
                        // source.
                        let block_slice = unsafe {
                            std::slice::from_raw_parts_mut(buffer_ptrs_compute[id].0, block_len)
                        };
                        let mut block = MatMut::from_column_major_slice_mut(
                            &mut block_slice[..n_samples * filled],
                            n_samples,
                            filled,
                        );
                        let variant_range = start..start + filled;
                        self.standardize_block_in_place(block.rb_mut(), variant_range.clone(), par);

                        let mut proj_block = proj_storage.rb_mut().subrows_mut(0, filled);

                        matmul(
                            proj_block.as_mut(),
                            Accum::Replace,
                            block.as_ref().transpose(),
                            rhs,
                            1.0,
                            par,
                        );

                        matmul(
                            out.rb_mut(),
                            Accum::Add,
                            block.as_ref(),
                            proj_block.as_ref(),
                            scale,
                            par,
                        );

                        processed = start + filled;

                        if free_sender.send(id).is_err() {
                            break;
                        }
                    }
                    PrefetchMessage::End => {
                        break;
                    }
                    PrefetchMessage::Error(err) => {
                        self.record_error(err);
                    }
                }
            }

            processed
        });

        if processed != self.observed_variants {
            self.fail_invalid("VariantBlockSource terminated early during covariance accumulation");
        }
    }

    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}

fn compute_covariance_eigenpairs<S, P>(
    operator: &StandardizedCovarianceOp<'_, S, P>,
    par: Par,
    mode: CovarianceComputationMode,
    top_k: usize,
    progress: Option<&StageProgressHandle<P>>,
) -> Result<Eigenpairs, HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync + 'static,
{
    let n = operator.n_samples();
    if n == 0 || top_k == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(0, 0),
        });
    }

    let max_rank = n.saturating_sub(1);
    if max_rank == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(n, 0),
        });
    }

    let desired = top_k.min(max_rank);
    if desired == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(n, 0),
        });
    }

    if matches!(mode, CovarianceComputationMode::Dense) {
        return compute_covariance_eigenpairs_dense(operator, par, desired, progress);
    }

    let upper_target = max_rank.min(MAX_PARTIAL_COMPONENTS.max(desired));
    if upper_target == 0 {
        return compute_covariance_eigenpairs_dense(operator, par, desired, progress);
    }

    let normalization = (n as f64).sqrt();
    let v0 = Col::from_fn(n, |_| 1.0 / normalization);
    let mut target = desired.min(upper_target).max(1);

    loop {
        let params = partial_solver_params(n, target);
        let (info, eigvals, eigvecs) = run_partial_eigensolver(operator, target, par, &v0, params)?;

        let n_converged = info.n_converged_eigen.min(target);

        let positive = eigvals
            .iter()
            .take(n_converged)
            .filter(|&&v| v > EIGENVALUE_EPSILON)
            .count();

        if positive == 0 {
            return Ok(Eigenpairs {
                values: Vec::new(),
                vectors: Mat::zeros(n, 0),
            });
        }

        let keep = positive.min(desired);
        let mut ordering = Vec::with_capacity(n_converged);
        for idx in 0..n_converged {
            ordering.push((idx, eigvals[idx]));
        }

        let mid = select_top_k_desc(&mut ordering, keep);

        let mut values = Vec::with_capacity(mid);
        let mut vectors = Mat::zeros(n, mid);
        for (out_idx, (src_idx, value)) in ordering[..mid].iter().copied().enumerate() {
            values.push(value);
            for row in 0..n {
                vectors[(row, out_idx)] = eigvecs[(row, src_idx)];
            }
        }

        if keep >= desired || target >= upper_target {
            return Ok(Eigenpairs { values, vectors });
        }

        let next_target = (target * 2).min(upper_target);
        if next_target == target {
            return Ok(Eigenpairs { values, vectors });
        }

        target = next_target;
    }
}

fn compute_covariance_eigenpairs_dense<S, P>(
    operator: &StandardizedCovarianceOp<'_, S, P>,
    par: Par,
    top_k: usize,
    progress: Option<&StageProgressHandle<P>>,
) -> Result<Eigenpairs, HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync + 'static,
{
    let mut covariance = accumulate_covariance_matrix(operator, par, progress)?;
    let n = covariance.nrows();

    if n == 0 || top_k == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(n, 0),
        });
    }

    if n <= DENSE_EIGEN_FALLBACK_THRESHOLD || top_k + 8 >= n {
        let eig = covariance.self_adjoint_eigen(Side::Lower).map_err(|err| {
            HwePcaError::Eigen(format!("dense eigendecomposition failed: {err:?}"))
        })?;

        let diag = eig.S();
        let basis = eig.U();

        let positive = (0..n).filter(|&i| diag[i] > EIGENVALUE_EPSILON).count();

        let keep = positive.min(top_k);
        if keep == 0 {
            return Ok(Eigenpairs {
                values: Vec::new(),
                vectors: Mat::zeros(n, 0),
            });
        }

        let mut ordering = Vec::with_capacity(n);
        for idx in 0..n {
            ordering.push((idx, diag[idx]));
        }

        let mid = select_top_k_desc(&mut ordering, keep);

        let mut values = Vec::with_capacity(mid);
        let mut vectors = Mat::zeros(n, mid);
        for (out_idx, (src_idx, value)) in ordering[..mid].iter().copied().enumerate() {
            values.push(value);
            for row in 0..n {
                vectors[(row, out_idx)] = basis[(row, src_idx)];
            }
        }

        return Ok(Eigenpairs { values, vectors });
    }

    mirror_lower_to_upper(&mut covariance);

    let gram = covariance.as_ref();
    let upper_target = n.min(MAX_PARTIAL_COMPONENTS.max(top_k));
    let mut target = top_k.min(upper_target).max(1);

    let normalization = (n as f64).sqrt();
    let v0 = Col::from_fn(n, |_| 1.0 / normalization);
    let op = DenseSymmetricOp { matrix: gram };

    loop {
        let params = partial_solver_params(n, target);
        let mut eigvecs = Mat::zeros(n, target);
        let mut eigvals = vec![0.0f64; target];
        let scratch = partial_eigen_scratch(&op, params.max_dim, par, params);
        let mut mem = MemBuffer::new(scratch);
        let info = {
            let mut stack = MemStack::new(&mut mem);
            partial_self_adjoint_eigen(
                eigvecs.as_mut(),
                &mut eigvals,
                &op,
                v0.as_ref(),
                f64::EPSILON * 128.0,
                par,
                &mut stack,
                params,
            )
        };

        let n_converged = info.n_converged_eigen.min(target);

        let positive = (0..n_converged)
            .filter(|&i| eigvals[i] > EIGENVALUE_EPSILON)
            .count();

        if positive == 0 {
            return Ok(Eigenpairs {
                values: Vec::new(),
                vectors: Mat::zeros(n, 0),
            });
        }

        let keep = positive.min(top_k);
        let mut ordering = Vec::with_capacity(n_converged);
        for idx in 0..n_converged {
            ordering.push((idx, eigvals[idx]));
        }

        let mid = select_top_k_desc(&mut ordering, keep);

        let mut values = Vec::with_capacity(mid);
        let mut vectors = Mat::zeros(n, mid);
        for (out_idx, (src_idx, value)) in ordering[..mid].iter().copied().enumerate() {
            values.push(value);
            for row in 0..n {
                vectors[(row, out_idx)] = eigvecs[(row, src_idx)];
            }
        }

        if keep >= top_k || target >= upper_target {
            return Ok(Eigenpairs { values, vectors });
        }

        let next_target = (target * 2).min(upper_target);
        if next_target == target {
            return Ok(Eigenpairs { values, vectors });
        }

        target = next_target;
    }
}

fn accumulate_covariance_matrix<S, P>(
    operator: &StandardizedCovarianceOp<'_, S, P>,
    par: Par,
    progress: Option<&StageProgressHandle<P>>,
) -> Result<Mat<f64>, HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync + 'static,
{
    let apply_guard = operator
        .apply_lock
        .lock()
        .expect("covariance apply lock poisoned");
    let _ = &apply_guard;

    let n_samples = operator.n_samples;
    let cross_products = Mat::zeros(n_samples, n_samples);

    if n_samples == 0 {
        return Ok(Mat::zeros(n_samples, n_samples));
    }

    let block_capacity = operator.block_capacity;
    let block_len = n_samples * block_capacity;
    let buffer_req = StackReq::new::<f64>(block_len).and(StackReq::new::<f64>(block_len));
    let mut mem = MemBuffer::new(buffer_req);
    let stack = MemStack::new(&mut mem);
    let (buf0_uninit, stack) = stack.make_uninit::<f64>(block_len);
    let buf0 =
        // SAFETY: `buf0_uninit` owns `block_len` contiguous `f64`s that live for
        // the duration of this function. We immediately coerce it to
        // `&mut [f64]` so `VariantBlockSource::next_block_into` can stream data
        // into the prefix `n_samples * filled`. That prefix is fully written
        // before being read; any trailing capacity remains untouched.
        unsafe { std::slice::from_raw_parts_mut(buf0_uninit.as_mut_ptr() as *mut f64, block_len) };
    let (buf1_uninit, _) = stack.make_uninit::<f64>(block_len);
    let buf1 =
        // SAFETY: Mirroring the reasoning for `buf0`, the allocation stays alive
        // for the entire call and only the written prefix is ever read.
        unsafe { std::slice::from_raw_parts_mut(buf1_uninit.as_mut_ptr() as *mut f64, block_len) };
    let mut buffer_slices = [buf0, buf1];
    let [first_slice, second_slice] = &mut buffer_slices;
    let buffer_ptrs = [
        SendPtr(first_slice.as_mut_ptr()),
        SendPtr(second_slice.as_mut_ptr()),
    ];

    enum PrefetchMessage {
        Data {
            id: usize,
            filled: usize,
            start: usize,
        },
        End,
        Error(HwePcaError),
    }

    let buffer_count = buffer_ptrs.len();
    let (filled_tx, filled_rx) = sync_channel::<PrefetchMessage>(buffer_count);
    let (free_tx, free_rx) = sync_channel::<usize>(buffer_count);
    for id in 0..buffer_count {
        free_tx.send(id).expect("failed to seed covariance buffers");
    }

    let source_mutex = &operator.source;
    let n_variants_hint = operator.n_variants_hint;
    let observed_total = operator.observed_variants;
    let block_capacity = operator.block_capacity;
    let block_len = block_len;
    let progress_handle = progress;

    let result = thread::scope(|scope| {
        let buffer_ptrs_prefetch = buffer_ptrs;
        let filled_sender = filled_tx;
        let free_receiver = free_rx;
        let progress_handle = progress_handle;
        scope.spawn(move || {
            if let Err(err) = {
                let mut guard = source_mutex
                    .lock()
                    .expect("covariance source mutex poisoned");
                let source: &mut S = &mut **guard;
                source.reset().map_err(|e| HwePcaError::Source(Box::new(e)))
            } {
                let _ = filled_sender.send(PrefetchMessage::Error(err));
                return;
            }

            let mut start = 0usize;
            let mut used_source_progress = false;
            while let Ok(id) = free_receiver.recv() {
                // SAFETY: The pointer was derived from a uniquely-owned slice
                // stored in `buffer_slices` and the scoped threads ensure the
                // backing storage outlives this reconstruction. Channel
                // ownership gives each `id` a single borrower, and we only
                // consume the prefix populated by `next_block_into`.
                let buffer_slice = unsafe {
                    std::slice::from_raw_parts_mut(buffer_ptrs_prefetch[id].0, block_len)
                };
                let (filled_res, progress_bytes, progress_variants) = {
                    let mut guard = source_mutex
                        .lock()
                        .expect("covariance source mutex poisoned");
                    let source: &mut S = &mut **guard;
                    let filled = source.next_block_into(block_capacity, buffer_slice);
                    let bytes = source.progress_bytes();
                    let variants = source.progress_variants();
                    (filled, bytes, variants)
                };

                match filled_res {
                    Ok(filled) => {
                        if filled == 0 {
                            if let Some(handle) = progress_handle {
                                if let Some((_, Some(total))) = progress_variants {
                                    handle.set_total(total);
                                } else if !used_source_progress {
                                    handle.set_total(start);
                                }
                            }
                            let _ = filled_sender.send(PrefetchMessage::End);
                            break;
                        }

                        if let Some(handle) = progress_handle {
                            if let Some((bytes_read, total_bytes)) = progress_bytes {
                                used_source_progress = true;
                                handle.advance_bytes(bytes_read, total_bytes);
                            } else if let Some((work_done, total_work)) = progress_variants {
                                used_source_progress = true;
                                if let Some(total) = total_work {
                                    handle.set_total(total);
                                } else if n_variants_hint > 0 {
                                    handle.estimate(n_variants_hint);
                                }
                                handle.advance(work_done);
                            } else {
                                handle.advance(start + filled);
                            }
                        }

                        let _ = filled_sender.send(PrefetchMessage::Data { id, filled, start });
                        start += filled;
                    }
                    Err(err) => {
                        let _ = filled_sender
                            .send(PrefetchMessage::Error(HwePcaError::Source(Box::new(err))));
                        break;
                    }
                }
            }
        });

        let free_sender = free_tx;
        let mut processed = 0usize;
        let buffer_ptrs_compute = buffer_ptrs;
        let mut cross_products = cross_products;
        while let Ok(message) = filled_rx.recv() {
            match message {
                PrefetchMessage::Data { id, filled, start } => {
                    if start != processed {
                        return Err(HwePcaError::InvalidInput(
                            "prefetch produced out-of-order variant ranges",
                        ));
                    }
                    if start + filled > observed_total {
                        return Err(HwePcaError::InvalidInput(
                            "VariantBlockSource returned more variants than observed",
                        ));
                    }

                    // SAFETY: Each `id` corresponds to a single buffer whose
                    // ownership is passed through the channel. The pointer
                    // remains valid and uniquely borrowed until we send `id`
                    // back on the free list, and all reads stay within the
                    // initialized prefix.
                    let block_slice = unsafe {
                        std::slice::from_raw_parts_mut(buffer_ptrs_compute[id].0, block_len)
                    };
                    let mut block = MatMut::from_column_major_slice_mut(
                        &mut block_slice[..n_samples * filled],
                        n_samples,
                        filled,
                    );
                    let variant_range = start..start + filled;
                    operator.standardize_block_in_place(block.rb_mut(), variant_range.clone(), par);

                    let block_ref = block.as_ref();

                    triangular_matmul::matmul(
                        cross_products.as_mut(),
                        triangular_matmul::BlockStructure::TriangularLower,
                        Accum::Add,
                        block_ref,
                        triangular_matmul::BlockStructure::Rectangular,
                        block_ref.transpose(),
                        triangular_matmul::BlockStructure::Rectangular,
                        1.0,
                        par,
                    );

                    processed = start + filled;

                    if free_sender.send(id).is_err() {
                        break;
                    }
                }
                PrefetchMessage::End => {
                    break;
                }
                PrefetchMessage::Error(err) => {
                    return Err(err);
                }
            }
        }

        if processed == 0 {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource yielded no variants",
            ));
        }

        if processed != observed_total {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource terminated early during covariance accumulation",
            ));
        }

        Ok(cross_products)
    });

    let mut covariance = result?;
    let scale = operator.scale;
    for col in 0..n_samples {
        for row in col..n_samples {
            covariance[(row, col)] *= scale;
        }
    }
    mirror_lower_to_upper(&mut covariance);

    Ok(covariance)
}

fn mirror_lower_to_upper(matrix: &mut Mat<f64>) {
    debug_assert_eq!(matrix.nrows(), matrix.ncols());
    let n = matrix.nrows();
    for col in 0..n {
        for row in 0..col {
            let value = matrix[(col, row)];
            matrix[(row, col)] = value;
        }
    }
}

fn partial_solver_params(n: usize, target: usize) -> PartialEigenParams {
    let mut params = PartialEigenParams::default();
    let max_available = n.saturating_sub(1);
    params.min_dim = target.max(64).min(max_available); // let Faer clamp internally
    params.max_dim = (2 * target).max(128).min(max_available);
    if params.max_dim < params.min_dim {
        params.max_dim = params.min_dim;
    }
    params.max_restarts = 2048;
    params
}

fn run_partial_eigensolver<S, P>(
    operator: &StandardizedCovarianceOp<'_, S, P>,
    target: usize,
    par: Par,
    v0: &Col<f64>,
    params: PartialEigenParams,
) -> Result<(PartialEigenInfo, Vec<f64>, Mat<f64>), HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync + 'static,
{
    let n = operator.n_samples();
    let mut eigvecs = Mat::zeros(n, target);
    let mut eigvals = vec![0.0f64; target];

    // The partial eigensolver can increase its working subspace dimension up to
    // `params.max_dim` during restarts. The scratch allocator must therefore be
    // sized for the worst-case dimension rather than the requested target, or
    // faer's internal workspace requests will overflow the provided buffer.
    let scratch = partial_eigen_scratch(operator, params.max_dim, par, params);
    let mut mem = MemBuffer::new(scratch);

    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut stack = MemStack::new(&mut mem);
        partial_self_adjoint_eigen(
            eigvecs.as_mut(),
            &mut eigvals,
            operator,
            v0.as_ref(),
            f64::EPSILON * 128.0,
            par,
            &mut stack,
            params,
        )
    }));

    let info = match result {
        Ok(info) => info,
        Err(payload) => {
            if payload.downcast_ref::<OperatorError>().is_some() {
                let err = operator.take_error().unwrap_or_else(|| {
                    HwePcaError::Eigen(
                        "matrix-free eigensolver aborted with an internal error".into(),
                    )
                });
                return Err(err);
            }
            std::panic::resume_unwind(payload);
        }
    };

    if let Some(err) = operator.take_error() {
        return Err(err);
    }

    Ok((info, eigvals, eigvecs))
}

fn build_sample_scores(n_samples: usize, decomposition: &Eigenpairs) -> (Vec<f64>, Mat<f64>) {
    let mut singular_values = Vec::with_capacity(decomposition.values.len());
    let mut sample_scores = decomposition.vectors.clone();

    for (&lambda, mut column) in decomposition
        .values
        .iter()
        .zip(sample_scores.col_iter_mut())
    {
        let scaled = (n_samples - 1) as f64 * lambda;
        let sigma = if scaled > 0.0 { scaled.sqrt() } else { 0.0 };
        singular_values.push(sigma);
        zip!(&mut column).for_each(|unzip!(value)| {
            *value *= sigma;
        });
    }

    (singular_values, sample_scores)
}

fn compute_variant_statistics<S, P>(
    source: &mut S,
    block_capacity: usize,
    par: Par,
    progress: StageProgressHandle<P>,
    n_variants_hint: usize,
) -> Result<(HweScaler, usize), HwePcaError>
where
    S: VariantBlockSource,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync,
{
    let n_samples = source.n_samples();
    let mut stats = VariantStatsCache::new(block_capacity, n_variants_hint);
    let mut block_storage = vec![0.0f64; n_samples * block_capacity];

    source
        .reset()
        .map_err(|err| HwePcaError::Source(Box::new(err)))?;

    let mut processed = 0usize;
    let mut used_source_progress = false;

    loop {
        let filled = source
            .next_block_into(block_capacity, &mut block_storage[..])
            .map_err(|err| HwePcaError::Source(Box::new(err)))?;

        if filled == 0 {
            break;
        }

        if n_variants_hint > 0 && processed + filled > n_variants_hint {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource returned more variants than reported hint",
            ));
        }

        let block = MatMut::from_column_major_slice_mut(
            &mut block_storage[..n_samples * filled],
            n_samples,
            filled,
        );

        let variant_range = processed..processed + filled;
        stats.ensure_statistics(block.as_ref(), variant_range.clone(), par);

        processed += filled;

        if let Some((bytes_read, total_bytes)) = source.progress_bytes() {
            used_source_progress = true;
            progress.advance_bytes(bytes_read, total_bytes);
        } else if let Some((work_done, total_work)) = source.progress_variants() {
            used_source_progress = true;
            if let Some(total) = total_work {
                progress.set_total(total);
            } else if n_variants_hint > 0 {
                progress.estimate(n_variants_hint);
            }
            progress.advance(work_done);
        } else {
            progress.advance(processed);
        }
    }

    if processed == 0 {
        progress.finish();
        return Err(HwePcaError::InvalidInput(
            "VariantBlockSource yielded no variants",
        ));
    }

    if let Some((_, Some(total))) = source.progress_variants() {
        progress.set_total(total);
    } else if !used_source_progress {
        progress.set_total(processed);
    }

    stats.finalize();
    let scaler = stats
        .into_scaler()
        .expect("finalized statistics must produce a scaler");
    progress.finish();

    Ok((scaler, processed))
}

struct LdRingBuffer {
    values: Mat<f64>,
    masks: Vec<u8>,
    n_samples: usize,
    indices: Vec<usize>,
    start: usize,
    len: usize,
}

impl LdRingBuffer {
    fn new(n_samples: usize, capacity: usize) -> Self {
        Self {
            values: Mat::zeros(n_samples, capacity),
            masks: vec![0u8; n_samples * capacity],
            n_samples,
            indices: vec![usize::MAX; capacity],
            start: 0,
            len: 0,
        }
    }

    fn capacity(&self) -> usize {
        self.indices.len()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn values(&self) -> MatRef<'_, f64> {
        self.values.as_ref()
    }

    fn values_mut(&mut self) -> MatMut<'_, f64> {
        self.values.as_mut()
    }

    fn mask_slice(&self, slot: usize) -> &[u8] {
        let start = slot * self.n_samples;
        &self.masks[start..start + self.n_samples]
    }

    fn mask_slice_mut(&mut self, slot: usize) -> &mut [u8] {
        let start = slot * self.n_samples;
        &mut self.masks[start..start + self.n_samples]
    }

    fn indices_mut(&mut self) -> &mut [usize] {
        &mut self.indices
    }

    fn push_slot(&mut self) -> usize {
        let capacity = self.capacity();
        if capacity == 0 {
            return 0;
        }
        let slot = if self.len < capacity {
            let slot = (self.start + self.len) % capacity;
            self.len += 1;
            slot
        } else {
            let slot = self.start;
            self.start = (self.start + 1) % capacity;
            slot
        };
        slot
    }

    fn position_of(&self, index: usize) -> Option<usize> {
        let capacity = self.capacity();
        if capacity == 0 {
            return None;
        }
        for offset in 0..self.len {
            let slot = (self.start + offset) % capacity;
            if self.indices[slot] == index {
                return Some(offset);
            }
        }
        None
    }

    fn slot_at(&self, offset: usize) -> usize {
        let capacity = self.capacity();
        if capacity == 0 {
            0
        } else {
            (self.start + offset) % capacity
        }
    }

    fn truncate_front(&mut self, keep_from: usize) {
        let capacity = self.capacity();
        if capacity == 0 {
            return;
        }
        while self.len > 0 {
            let slot = self.start;
            if self.indices[slot] < keep_from {
                self.indices[slot] = usize::MAX;
                self.start = (self.start + 1) % capacity;
                self.len -= 1;
            } else {
                break;
            }
        }
    }
}

struct LdWindowStats {
    gram: Mat<f64>,
    counts: Mat<f64>,
    sums: Mat<f64>,
    squared_sums: Mat<f64>,
    len: usize,
    head: usize,
    tail: usize,
    n_samples: usize,
}

impl LdWindowStats {
    fn new(n_samples: usize, capacity: usize) -> Self {
        Self {
            gram: Mat::zeros(capacity, capacity),
            counts: Mat::zeros(capacity, capacity),
            sums: Mat::zeros(capacity, capacity),
            squared_sums: Mat::zeros(capacity, capacity),
            len: 0,
            head: 0,
            tail: 0,
            n_samples,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn capacity(&self) -> usize {
        self.gram.nrows()
    }

    fn slot_at(&self, offset: usize) -> usize {
        let capacity = self.capacity();
        if capacity == 0 {
            0
        } else {
            (self.head + offset) % capacity
        }
    }

    fn update_tail(&mut self) {
        let capacity = self.capacity();
        if capacity == 0 {
            self.tail = 0;
        } else {
            self.tail = (self.head + self.len) % capacity;
        }
    }

    fn zero_slot(&mut self, slot: usize) {
        let capacity = self.capacity();
        if capacity == 0 {
            return;
        }
        for idx in 0..capacity {
            self.gram[(slot, idx)] = 0.0;
            self.gram[(idx, slot)] = 0.0;
            self.counts[(slot, idx)] = 0.0;
            self.counts[(idx, slot)] = 0.0;
            self.sums[(slot, idx)] = 0.0;
            self.sums[(idx, slot)] = 0.0;
            self.squared_sums[(slot, idx)] = 0.0;
            self.squared_sums[(idx, slot)] = 0.0;
        }
    }

    fn add_variant(&mut self, ring: &LdRingBuffer) {
        let ring_len = ring.len();
        if ring_len == 0 {
            return;
        }

        debug_assert_eq!(ring_len, self.len + 1);

        let new_offset = ring_len - 1;
        let new_ring_slot = ring.slot_at(new_offset);
        let new_stats_slot = self.slot_at(new_offset);
        let values = ring.values().col(new_ring_slot);
        let mask = ring.mask_slice(new_ring_slot);

        let mut gram_diag = 0.0;
        let mut count_diag = 0.0;
        let mut sum_diag = 0.0;
        let mut sq_diag = 0.0;

        for row in 0..self.n_samples {
            let value = values[row];
            let mask_value = f64::from(mask[row]);
            gram_diag += value * value;
            count_diag += mask_value;
            sum_diag += value * mask_value;
            sq_diag += value * value * mask_value;
        }

        self.gram[(new_stats_slot, new_stats_slot)] = gram_diag;
        self.counts[(new_stats_slot, new_stats_slot)] = count_diag;
        self.sums[(new_stats_slot, new_stats_slot)] = sum_diag;
        self.squared_sums[(new_stats_slot, new_stats_slot)] = sq_diag;

        for existing in 0..self.len {
            let existing_ring_slot = ring.slot_at(existing);
            let existing_stats_slot = self.slot_at(existing);
            let existing_values = ring.values().col(existing_ring_slot);
            let existing_mask = ring.mask_slice(existing_ring_slot);

            let mut gram_value = 0.0;
            let mut count_value = 0.0;
            let mut sum_row = 0.0;
            let mut sum_col = 0.0;
            let mut sq_row = 0.0;
            let mut sq_col = 0.0;

            for row in 0..self.n_samples {
                let value_new = values[row];
                let value_existing = existing_values[row];
                let mask_new = f64::from(mask[row]);
                let mask_existing = f64::from(existing_mask[row]);

                gram_value += value_new * value_existing;
                count_value += mask_new * mask_existing;
                sum_row += value_new * mask_existing;
                sum_col += value_existing * mask_new;
                sq_row += value_new * value_new * mask_existing;
                sq_col += value_existing * value_existing * mask_new;
            }

            self.gram[(new_stats_slot, existing_stats_slot)] = gram_value;
            self.gram[(existing_stats_slot, new_stats_slot)] = gram_value;
            self.counts[(new_stats_slot, existing_stats_slot)] = count_value;
            self.counts[(existing_stats_slot, new_stats_slot)] = count_value;
            self.sums[(new_stats_slot, existing_stats_slot)] = sum_row;
            self.sums[(existing_stats_slot, new_stats_slot)] = sum_col;
            self.squared_sums[(new_stats_slot, existing_stats_slot)] = sq_row;
            self.squared_sums[(existing_stats_slot, new_stats_slot)] = sq_col;
        }

        self.len += 1;
        self.update_tail();
    }

    fn remove_front(&mut self) {
        if self.len == 0 {
            return;
        }

        let capacity = self.capacity();
        if capacity == 0 {
            self.len = 0;
            self.head = 0;
            self.tail = 0;
            return;
        }

        let slot = self.head;
        self.zero_slot(slot);
        self.head = (self.head + 1) % capacity;
        self.len -= 1;
        self.update_tail();
    }

    fn truncate_front(&mut self, count: usize) {
        let to_remove = count.min(self.len);
        let capacity = self.capacity();
        if capacity == 0 {
            self.len = 0;
            self.head = 0;
            self.tail = 0;
            return;
        }

        for _ in 0..to_remove {
            let slot = self.head;
            self.zero_slot(slot);
            self.head = (self.head + 1) % capacity;
        }
        self.len -= to_remove;
        self.update_tail();
    }

    fn view<'a>(
        &'a self,
        start: usize,
        len: usize,
        scratch: &'a mut LdWindowStatsScratch,
    ) -> LdWindowStatsView<'a> {
        debug_assert!(start <= self.len);
        debug_assert!(start + len <= self.len);
        if len == 0 {
            return LdWindowStatsView {
                gram: scratch.gram.as_ref().submatrix(0, 0, 0, 0),
                counts: scratch.counts.as_ref().submatrix(0, 0, 0, 0),
                sums: scratch.sums.as_ref().submatrix(0, 0, 0, 0),
                squared_sums: scratch.squared_sums.as_ref().submatrix(0, 0, 0, 0),
            };
        }

        let mut contiguous = true;
        let mut prev_slot = self.slot_at(start);
        for offset in (start + 1)..(start + len) {
            let slot = self.slot_at(offset);
            if slot != prev_slot + 1 {
                contiguous = false;
                break;
            }
            prev_slot = slot;
        }

        if contiguous {
            let start_slot = self.slot_at(start);
            return LdWindowStatsView {
                gram: self
                    .gram
                    .as_ref()
                    .submatrix(start_slot, start_slot, len, len),
                counts: self
                    .counts
                    .as_ref()
                    .submatrix(start_slot, start_slot, len, len),
                sums: self
                    .sums
                    .as_ref()
                    .submatrix(start_slot, start_slot, len, len),
                squared_sums: self
                    .squared_sums
                    .as_ref()
                    .submatrix(start_slot, start_slot, len, len),
            };
        }

        {
            let mut gram_scratch = scratch.gram.as_mut().submatrix_mut(0, 0, len, len);
            let mut count_scratch = scratch.counts.as_mut().submatrix_mut(0, 0, len, len);
            let mut sum_scratch = scratch.sums.as_mut().submatrix_mut(0, 0, len, len);
            let mut sq_scratch = scratch.squared_sums.as_mut().submatrix_mut(0, 0, len, len);

            for row in 0..len {
                let row_slot = self.slot_at(start + row);
                for col in 0..len {
                    let col_slot = self.slot_at(start + col);
                    gram_scratch[(row, col)] = self.gram[(row_slot, col_slot)];
                    count_scratch[(row, col)] = self.counts[(row_slot, col_slot)];
                    sum_scratch[(row, col)] = self.sums[(row_slot, col_slot)];
                    sq_scratch[(row, col)] = self.squared_sums[(row_slot, col_slot)];
                }
            }
        }

        let scratch_ref: &'a LdWindowStatsScratch = &*scratch;
        LdWindowStatsView {
            gram: scratch_ref.gram.as_ref().submatrix(0, 0, len, len),
            counts: scratch_ref.counts.as_ref().submatrix(0, 0, len, len),
            sums: scratch_ref.sums.as_ref().submatrix(0, 0, len, len),
            squared_sums: scratch_ref.squared_sums.as_ref().submatrix(0, 0, len, len),
        }
    }
}

struct LdWindowStatsView<'a> {
    gram: MatRef<'a, f64>,
    counts: MatRef<'a, f64>,
    sums: MatRef<'a, f64>,
    squared_sums: MatRef<'a, f64>,
}

struct LdWindowStatsScratch {
    gram: Mat<f64>,
    counts: Mat<f64>,
    sums: Mat<f64>,
    squared_sums: Mat<f64>,
}

impl LdWindowStatsScratch {
    fn new(capacity: usize) -> Self {
        Self {
            gram: Mat::zeros(capacity, capacity),
            counts: Mat::zeros(capacity, capacity),
            sums: Mat::zeros(capacity, capacity),
            squared_sums: Mat::zeros(capacity, capacity),
        }
    }
}

fn compute_ld_weights<S, P>(
    source: &mut S,
    scaler: &HweScaler,
    observed_variants: usize,
    block_capacity: usize,
    config: LdResolvedConfig,
    n_variants_hint: usize,
    progress: &Arc<P>,
    par: Par,
) -> Result<LdWeights, HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync + 'static,
{
    let mut weights = vec![1.0; observed_variants];
    progress.on_stage_start(FitProgressStage::LdWeights, observed_variants);
    let stage_progress =
        StageProgressHandle::new(Arc::clone(progress), FitProgressStage::LdWeights);

    if observed_variants == 0 {
        stage_progress.finish();
        return Ok(LdWeights {
            weights,
            window: config.window_capacity().max(1),
            bp_window: config.bp_window(),
            ridge: config.ridge,
        });
    }

    let n_samples = source.n_samples();
    let mut block_storage = vec![0.0f64; n_samples * block_capacity];
    let mut presence_storage = vec![0.0f64; n_samples * block_capacity];
    let window_capacity = config.window_capacity().max(1);
    let mut ring = LdRingBuffer::new(n_samples, window_capacity);
    let mut next_weight = 0usize;

    let mut window_stats = LdWindowStats::new(n_samples, window_capacity);
    let mut stats_scratch = LdWindowStatsScratch::new(window_capacity);
    let mut system_scratch = Mat::zeros(window_capacity, window_capacity);
    let mut rhs_scratch = Mat::zeros(window_capacity, 1);

    source
        .reset()
        .map_err(|err| HwePcaError::Source(Box::new(err)))?;

    let mut processed = 0usize;
    loop {
        let filled = source
            .next_block_into(block_capacity, &mut block_storage[..])
            .map_err(|err| HwePcaError::Source(Box::new(err)))?;

        if filled == 0 {
            break;
        }

        if n_variants_hint > 0 && processed + filled > n_variants_hint {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource returned more variants than reported hint",
            ));
        }

        let mut block = MatMut::from_column_major_slice_mut(
            &mut block_storage[..n_samples * filled],
            n_samples,
            filled,
        );
        let mut presence = MatMut::from_column_major_slice_mut(
            &mut presence_storage[..n_samples * filled],
            n_samples,
            filled,
        );

        let variant_range = processed..processed + filled;
        scaler.standardize_block_with_mask(
            block.as_mut(),
            variant_range.clone(),
            presence.as_mut(),
            par,
        );

        for (local_idx, (column, present)) in block
            .as_ref()
            .col_iter()
            .zip(presence.as_ref().col_iter())
            .enumerate()
        {
            let was_full = ring.len() == ring.capacity();
            let slot = ring.push_slot();
            if was_full {
                window_stats.remove_front();
            }
            {
                let dst_col = ring.values_mut().col_mut(slot);
                zip!(dst_col, column).for_each(|unzip!(dst, src)| {
                    *dst = *src;
                });
            }
            {
                let mask_slice = ring.mask_slice_mut(slot);
                for (dst, &src) in mask_slice.iter_mut().zip(present.iter()) {
                    *dst = if src != 0.0 { 1u8 } else { 0u8 };
                }
            }
            ring.indices_mut()[slot] = processed + local_idx;
            window_stats.add_variant(&ring);
            assign_ready_weights(
                &mut ring,
                &mut weights,
                &mut next_weight,
                &config,
                &mut window_stats,
                &mut stats_scratch,
                &mut system_scratch,
                &mut rhs_scratch,
                &stage_progress,
            )?;
        }

        processed += filled;
    }

    assign_ready_weights(
        &mut ring,
        &mut weights,
        &mut next_weight,
        &config,
        &mut window_stats,
        &mut stats_scratch,
        &mut system_scratch,
        &mut rhs_scratch,
        &stage_progress,
    )?;

    stage_progress.set_total(observed_variants);
    stage_progress.finish();

    Ok(LdWeights {
        weights,
        window: config.window_capacity().max(1),
        bp_window: config.bp_window(),
        ridge: config.ridge,
    })
}

fn assign_ready_weights<P: FitProgressObserver>(
    ring: &mut LdRingBuffer,
    weights: &mut [f64],
    next_weight: &mut usize,
    config: &LdResolvedConfig,
    stats: &mut LdWindowStats,
    stats_scratch: &mut LdWindowStatsScratch,
    system_scratch: &mut Mat<f64>,
    rhs_scratch: &mut Mat<f64>,
    progress: &StageProgressHandle<P>,
) -> Result<(), HwePcaError> {
    while *next_weight < weights.len() {
        debug_assert_eq!(stats.len(), ring.len());

        let position = match ring.position_of(*next_weight) {
            Some(pos) => pos,
            None => break,
        };

        let available = ring.len();
        if available == 0 {
            break;
        }

        let window_params = match &config.window {
            LdResolvedWindow::Sites { size } => {
                let window_size = (*size).min(available).max(1);
                let half_window = window_size / 2;
                let start = if available <= window_size {
                    0
                } else if position <= half_window {
                    0
                } else {
                    let tail_start = available.saturating_sub(window_size);
                    min(position - half_window, tail_start)
                };
                let end = min(start + window_size, available);
                let window_len = end - start;
                if window_len == 0 {
                    None
                } else {
                    let center = position - start;
                    let keep_from = next_weight.saturating_sub(window_size / 2);
                    Some((start, window_len, center, keep_from))
                }
            }
            LdResolvedWindow::BasePairs { ranges, .. } => {
                let range = &ranges[*next_weight];
                if range.end <= range.start {
                    return Err(HwePcaError::InvalidInput(
                        "LD base-pair window produced an empty range",
                    ));
                }
                let last_index = range.end - 1;
                match (ring.position_of(range.start), ring.position_of(last_index)) {
                    (Some(start_offset), Some(end_offset)) => {
                        let window_len = end_offset.saturating_sub(start_offset) + 1;
                        if window_len == 0 {
                            None
                        } else {
                            let center = position.saturating_sub(start_offset);
                            Some((start_offset, window_len, center, range.start))
                        }
                    }
                    _ => None,
                }
            }
        };

        let Some((start, window_len, center, keep_from)) = window_params else {
            break;
        };

        debug_assert!(start + window_len <= stats.len());

        let stats_view = stats.view(start, window_len, stats_scratch);
        let mut system_view = system_scratch
            .as_mut()
            .submatrix_mut(0, 0, window_len, window_len);
        let mut rhs_view = rhs_scratch.as_mut().submatrix_mut(0, 0, window_len, 1);
        let weight = solve_ld_window_from_stats(
            stats_view.gram,
            stats_view.sums,
            stats_view.squared_sums,
            stats_view.counts,
            center,
            config.ridge,
            system_view.as_mut(),
            rhs_view.as_mut(),
        );
        weights[*next_weight] = weight;
        progress.advance(*next_weight + 1);
        *next_weight += 1;

        let before_len = ring.len();
        ring.truncate_front(keep_from);
        let removed = before_len.saturating_sub(ring.len());
        if removed > 0 {
            stats.truncate_front(removed);
        }
    }

    Ok(())
}

fn solve_ld_window_from_stats(
    gram: MatRef<'_, f64>,
    sums: MatRef<'_, f64>,
    squared_sums: MatRef<'_, f64>,
    counts: MatRef<'_, f64>,
    center: usize,
    ridge: f64,
    mut system: MatMut<'_, f64>,
    mut rhs: MatMut<'_, f64>,
) -> f64 {
    let size = gram.nrows();
    if size == 0 || center >= size {
        return 1.0;
    }

    let mut adjusted_ridge = ridge;
    for attempt in 0..2 {
        for i in 0..size {
            system[(i, i)] = 1.0 + adjusted_ridge;
            for j in 0..i {
                let count = counts[(i, j)];
                let value = if count.is_finite() && count >= 2.0 {
                    let sum_x = sums[(i, j)];
                    let sum_y = sums[(j, i)];
                    let cov = gram[(i, j)] - (sum_x * sum_y) / count;
                    let var_x = squared_sums[(i, j)] - (sum_x * sum_x) / count;
                    let var_y = squared_sums[(j, i)] - (sum_y * sum_y) / count;

                    if !cov.is_finite() || !var_x.is_finite() || !var_y.is_finite() {
                        0.0
                    } else if var_x <= 0.0 || var_y <= 0.0 {
                        0.0
                    } else {
                        let corr = (cov / (var_x * var_y).sqrt()).clamp(-1.0, 1.0);
                        if !corr.is_finite() {
                            0.0
                        } else if count <= 2.0 {
                            0.0
                        } else {
                            let corr_sq = corr * corr;
                            let numerator = (count - 1.0) * corr_sq - 1.0;
                            let denominator = count - 2.0;
                            if denominator <= 0.0 {
                                0.0
                            } else {
                                let estimate = numerator / denominator;
                                estimate.max(0.0).min(1.0)
                            }
                        }
                    }
                } else {
                    0.0
                };
                system[(i, j)] = value;
                system[(j, i)] = value;
            }
        }

        for i in 0..size {
            rhs[(i, 0)] = 1.0;
        }

        match FaerLlt::new(system.as_ref(), Side::Lower) {
            Ok(factor) => {
                let solution = factor.solve(rhs.as_ref());
                let mut weight_sq = solution[(center, 0)];
                if !weight_sq.is_finite() || weight_sq <= 0.0 {
                    weight_sq = 1.0;
                }
                return weight_sq.sqrt().max(MIN_LD_WEIGHT);
            }
            Err(_) => {
                if attempt == 0 {
                    adjusted_ridge *= 10.0;
                    continue;
                } else {
                    return 1.0;
                }
            }
        }
    }

    1.0
}

fn compute_component_weighted_norms_sq(
    loadings: MatRef<'_, f64>,
    ld_weights: Option<&[f64]>,
) -> Vec<f64> {
    let weights = ld_weights.unwrap_or(&[]);
    let n_weights = weights.len();
    let n_components = loadings.ncols();
    let mut norms_sq = vec![0.0f64; n_components];

    for component in 0..n_components {
        let column_ref = loadings.col(component);
        let mut sum = 0.0f64;
        let mut compensation = 0.0f64;

        if n_weights > 0 {
            let mut idx = 0usize;
            zip!(column_ref).for_each(|unzip!(value)| {
                let weight = if idx < n_weights { weights[idx] } else { 1.0 };
                let weighted = weight * *value;
                let square = weighted * weighted;
                let y = square - compensation;
                let t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
                idx += 1;
            });
        } else {
            zip!(column_ref).for_each(|unzip!(value)| {
                let square = *value * *value;
                let y = square - compensation;
                let t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            });
        }

        let sum = if sum.is_finite() && sum >= 0.0 {
            sum
        } else {
            0.0
        };
        norms_sq[component] = sum;
    }

    norms_sq
}

fn renormalize_variant_loadings(
    mut loadings: MatMut<'_, f64>,
    singular_values: &mut [f64],
    mut sample_scores: MatMut<'_, f64>,
    ld_weights: Option<&[f64]>,
) -> Vec<f64> {
    debug_assert_eq!(loadings.ncols(), singular_values.len());
    debug_assert_eq!(sample_scores.ncols(), singular_values.len());

    let mut norms_sq = compute_component_weighted_norms_sq(loadings.as_ref(), ld_weights);

    for (component, norm_sq) in norms_sq.iter_mut().enumerate() {
        let norm = (*norm_sq).sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            *norm_sq = 0.0;
            continue;
        }

        let inv = norm.recip();
        let column_mut = loadings.rb_mut().col_mut(component);
        zip!(column_mut).for_each(|unzip!(value)| {
            *value *= inv;
        });

        singular_values[component] *= norm;

        let score_col = sample_scores.rb_mut().col_mut(component);
        zip!(score_col).for_each(|unzip!(value)| {
            *value *= norm;
        });

        *norm_sq = 1.0;
    }

    norms_sq
}

fn compute_variant_loadings<S, P>(
    source: &mut S,
    scaler: &HweScaler,
    expected_variants: usize,
    block_capacity: usize,
    sample_basis: MatRef<'_, f64>,
    singular_values: &[f64],
    ld_weights: Option<&[f64]>,
    progress: &Arc<P>,
    par: Par,
) -> Result<Mat<f64>, HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync + 'static,
{
    let n_samples = source.n_samples();
    let n_components = singular_values.len();
    let loadings = Mat::zeros(expected_variants, n_components);
    let mut chunk_storage = vec![0.0f64; block_capacity * n_components];
    let inverse_singular: Vec<f64> = singular_values
        .iter()
        .map(|&sigma| if sigma > 0.0 { 1.0 / sigma } else { 0.0 })
        .collect();

    progress.on_stage_start(FitProgressStage::Loadings, expected_variants);

    let block_len = n_samples * block_capacity;
    let buffer_req = StackReq::new::<f64>(block_len).and(StackReq::new::<f64>(block_len));
    let mut mem = MemBuffer::new(buffer_req);
    let stack = MemStack::new(&mut mem);
    let (buf0_uninit, stack) = stack.make_uninit::<f64>(block_len);
    let buf0 =
        // SAFETY: `buf0_uninit` was allocated with exactly `block_len` elements
        // and lives until the end of this function. We convert it to `&mut [f64]`
        // solely to let `VariantBlockSource::next_block_into` fill the
        // `n_samples * filled` prefix before it is observed.
        unsafe { std::slice::from_raw_parts_mut(buf0_uninit.as_mut_ptr() as *mut f64, block_len) };
    let (buf1_uninit, _) = stack.make_uninit::<f64>(block_len);
    let buf1 =
        // SAFETY: Identical justification as for `buf0`; only the filled prefix
        // is ever accessed after writing.
        unsafe { std::slice::from_raw_parts_mut(buf1_uninit.as_mut_ptr() as *mut f64, block_len) };
    let mut buffer_slices = [buf0, buf1];
    let [first_slice, second_slice] = &mut buffer_slices;
    let buffer_ptrs = [
        SendPtr(first_slice.as_mut_ptr()),
        SendPtr(second_slice.as_mut_ptr()),
    ];

    enum PrefetchMessage {
        Data {
            id: usize,
            filled: usize,
            start: usize,
        },
        End,
        Error(HwePcaError),
    }

    let buffer_count = buffer_ptrs.len();
    let (filled_tx, filled_rx) = sync_channel::<PrefetchMessage>(buffer_count);
    let (free_tx, free_rx) = sync_channel::<usize>(buffer_count);
    for id in 0..buffer_count {
        free_tx.send(id).expect("failed to seed loading buffers");
    }

    let observer = Arc::clone(progress);
    let block_capacity = block_capacity;
    let block_len = block_len;
    let expected_variants = expected_variants;

    let result = thread::scope(|scope| {
        let buffer_ptrs_prefetch = buffer_ptrs;
        let filled_sender = filled_tx;
        let free_receiver = free_rx;
        scope.spawn(move || {
            if let Err(err) = source.reset().map_err(|e| HwePcaError::Source(Box::new(e))) {
                let _ = filled_sender.send(PrefetchMessage::Error(err));
                return;
            }

            let mut start = 0usize;
            while let Ok(id) = free_receiver.recv() {
                // SAFETY: Each pointer came from a mutable slice backed by the
                // stack allocation above and remains valid throughout the
                // scoped thread. Distinct `id`s ensure no concurrent aliasing,
                // and we only consume the portion that `next_block_into`
                // initialized.
                let buffer_slice = unsafe {
                    std::slice::from_raw_parts_mut(buffer_ptrs_prefetch[id].0, block_len)
                };
                let filled = match source.next_block_into(block_capacity, buffer_slice) {
                    Ok(filled) => filled,
                    Err(err) => {
                        let _ = filled_sender
                            .send(PrefetchMessage::Error(HwePcaError::Source(Box::new(err))));
                        break;
                    }
                };

                if filled == 0 {
                    let _ = filled_sender.send(PrefetchMessage::End);
                    break;
                }

                observer.on_stage_advance(
                    FitProgressStage::Loadings,
                    (start + filled).min(expected_variants),
                );

                if filled_sender
                    .send(PrefetchMessage::Data { id, filled, start })
                    .is_err()
                {
                    break;
                }
                start += filled;
            }
        });

        let free_sender = free_tx;
        let mut processed = 0usize;
        let buffer_ptrs_compute = buffer_ptrs;
        let mut loadings = loadings;
        while let Ok(message) = filled_rx.recv() {
            match message {
                PrefetchMessage::Data { id, filled, start } => {
                    if start != processed {
                        return Err(HwePcaError::InvalidInput(
                            "prefetch produced out-of-order variant ranges",
                        ));
                    }
                    if start + filled > expected_variants {
                        return Err(HwePcaError::InvalidInput(
                            "VariantBlockSource returned more variants than reported",
                        ));
                    }

                    // SAFETY: The pointer corresponds to a unique buffer owned
                    // by this worker. It stays valid until the `id` is returned
                    // via `free_sender`, preventing simultaneous mutable
                    // borrows, and we only touch the prefix filled with new
                    // samples.
                    let block_slice = unsafe {
                        std::slice::from_raw_parts_mut(buffer_ptrs_compute[id].0, block_len)
                    };
                    let mut block = MatMut::from_column_major_slice_mut(
                        &mut block_slice[..n_samples * filled],
                        n_samples,
                        filled,
                    );
                    scaler.standardize_block(block.as_mut(), start..start + filled, par);
                    if let Some(weights) = ld_weights {
                        apply_ld_weights(block.as_mut(), start..start + filled, weights);
                    }

                    let block_ref = block.as_ref();
                    let mut chunk = MatMut::from_column_major_slice_mut(
                        &mut chunk_storage[..filled * n_components],
                        filled,
                        n_components,
                    );

                    matmul(
                        chunk.as_mut(),
                        Accum::Replace,
                        block_ref.transpose(),
                        sample_basis,
                        1.0,
                        par,
                    );

                    {
                        let chunk_view = chunk.as_mut();
                        for (column, &inv_sigma) in
                            chunk_view.col_iter_mut().zip(inverse_singular.iter())
                        {
                            zip!(column).for_each(|unzip!(value)| {
                                *value *= inv_sigma;
                            });
                        }
                    }

                    loadings
                        .submatrix_mut(start, 0, filled, n_components)
                        .copy_from(chunk.as_ref());

                    processed = start + filled;

                    if free_sender.send(id).is_err() {
                        break;
                    }
                }
                PrefetchMessage::End => {
                    break;
                }
                PrefetchMessage::Error(err) => {
                    return Err(err);
                }
            }
        }

        if processed != expected_variants {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource terminated early while computing loadings",
            ));
        }

        progress.on_stage_finish(FitProgressStage::Loadings);

        Ok(loadings)
    });

    result
}

#[derive(Serialize, Deserialize)]
struct MatrixData {
    nrows: usize,
    ncols: usize,
    data: Vec<f64>,
}

impl MatrixData {
    fn from_mat(mat: MatRef<'_, f64>) -> Self {
        let mut data = Vec::with_capacity(mat.nrows() * mat.ncols());
        for col in 0..mat.ncols() {
            for row in 0..mat.nrows() {
                data.push(mat[(row, col)]);
            }
        }
        Self {
            nrows: mat.nrows(),
            ncols: mat.ncols(),
            data,
        }
    }

    fn into_mat(self) -> Result<Mat<f64>, String> {
        let MatrixData { nrows, ncols, data } = self;
        if data.len() != nrows * ncols {
            return Err("matrix data length does not match dimensions".into());
        }
        let mut mat = Mat::zeros(nrows, ncols);
        for col in 0..ncols {
            for row in 0..nrows {
                mat[(row, col)] = data[col * nrows + row];
            }
        }
        Ok(mat)
    }
}

impl Serialize for HwePcaModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("HwePcaModel", 11)?;
        state.serialize_field("n_samples", &self.n_samples)?;
        state.serialize_field("n_variants", &self.n_variants)?;
        state.serialize_field("scaler", &self.scaler)?;
        state.serialize_field("eigenvalues", &self.eigenvalues)?;
        state.serialize_field("singular_values", &self.singular_values)?;
        state.serialize_field(
            "sample_basis",
            &MatrixData::from_mat(self.sample_basis.as_ref()),
        )?;
        state.serialize_field(
            "sample_scores",
            &MatrixData::from_mat(self.sample_scores.as_ref()),
        )?;
        state.serialize_field("loadings", &MatrixData::from_mat(self.loadings.as_ref()))?;
        state.serialize_field(
            "component_weighted_norms_sq",
            &self.component_weighted_norms_sq,
        )?;
        state.serialize_field("variant_keys", &self.variant_keys)?;
        state.serialize_field("ld", &self.ld)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for HwePcaModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ModelData {
            n_samples: usize,
            n_variants: usize,
            scaler: HweScaler,
            eigenvalues: Vec<f64>,
            singular_values: Vec<f64>,
            sample_basis: MatrixData,
            sample_scores: MatrixData,
            loadings: MatrixData,
            #[serde(default)]
            component_weighted_norms_sq: Vec<f64>,
            #[serde(default)]
            variant_keys: Option<Vec<VariantKey>>,
            #[serde(default)]
            ld: Option<LdWeights>,
        }

        let raw = ModelData::deserialize(deserializer)?;
        let singular_values_len = raw.singular_values.len();
        let sample_basis = raw.sample_basis.into_mat().map_err(DeError::custom)?;
        let sample_scores = raw.sample_scores.into_mat().map_err(DeError::custom)?;
        let loadings = raw.loadings.into_mat().map_err(DeError::custom)?;
        let ld = raw.ld;
        let component_weighted_norms_sq =
            if raw.component_weighted_norms_sq.len() == singular_values_len {
                raw.component_weighted_norms_sq
            } else {
                compute_component_weighted_norms_sq(
                    loadings.as_ref(),
                    ld.as_ref().map(|ld| ld.weights.as_slice()),
                )
            };

        Ok(HwePcaModel {
            n_samples: raw.n_samples,
            n_variants: raw.n_variants,
            scaler: raw.scaler,
            eigenvalues: raw.eigenvalues,
            singular_values: raw.singular_values,
            sample_basis,
            sample_scores,
            loadings,
            component_weighted_norms_sq,
            variant_keys: raw.variant_keys,
            ld,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::io::GenotypeDataset;
    use std::path::Path;
    use std::sync::Arc;

    fn compute_reference_ld_weights(
        data: &[f64],
        n_samples: usize,
        scaler: &HweScaler,
        config: &LdResolvedConfig,
    ) -> Result<Vec<f64>, HwePcaError> {
        let observed_variants = data.len() / n_samples;
        assert_eq!(observed_variants * n_samples, data.len());

        let mut standardized = data.to_vec();
        let mut mask = vec![0.0f64; data.len()];
        {
            let mut block = MatMut::from_column_major_slice_mut(
                &mut standardized,
                n_samples,
                observed_variants,
            );
            let mut mask_mat =
                MatMut::from_column_major_slice_mut(&mut mask, n_samples, observed_variants);
            scaler.standardize_block_with_mask(
                block.as_mut(),
                0..observed_variants,
                mask_mat.as_mut(),
                Par::Seq,
            );
        }

        let capacity = config.window_capacity().max(1);
        let mut weights = vec![1.0; observed_variants];
        let mut ring = LdRingBuffer::new(n_samples, capacity);
        let mut next_weight = 0usize;

        let mut values_scratch = Mat::zeros(n_samples, capacity);
        let mut mask_scratch = Mat::zeros(n_samples, capacity);
        let mut square_scratch = Mat::zeros(n_samples, capacity);
        let mut gram_scratch = Mat::zeros(capacity, capacity);
        let mut count_scratch = Mat::zeros(capacity, capacity);
        let mut sum_scratch = Mat::zeros(capacity, capacity);
        let mut sq_scratch = Mat::zeros(capacity, capacity);
        let mut system_scratch = Mat::zeros(capacity, capacity);
        let mut rhs_scratch = Mat::zeros(capacity, 1);

        let block =
            MatMut::from_column_major_slice_mut(&mut standardized, n_samples, observed_variants);
        let presence = MatMut::from_column_major_slice_mut(&mut mask, n_samples, observed_variants);

        for idx in 0..observed_variants {
            let column = block.as_ref().col(idx);
            let present = presence.as_ref().col(idx);
            let slot = ring.push_slot();
            {
                let dst_col = ring.values_mut().col_mut(slot);
                zip!(dst_col, column).for_each(|unzip!(dst, src)| {
                    *dst = *src;
                });
            }
            {
                let mask_slice = ring.mask_slice_mut(slot);
                for (dst, &src) in mask_slice.iter_mut().zip(present.iter()) {
                    *dst = if src != 0.0 { 1u8 } else { 0u8 };
                }
            }
            ring.indices_mut()[slot] = idx;
            assign_ready_weights_naive(
                &mut ring,
                &mut weights,
                &mut next_weight,
                config,
                &mut values_scratch,
                &mut mask_scratch,
                &mut square_scratch,
                &mut gram_scratch,
                &mut count_scratch,
                &mut sum_scratch,
                &mut sq_scratch,
                &mut system_scratch,
                &mut rhs_scratch,
            )?;
        }

        assign_ready_weights_naive(
            &mut ring,
            &mut weights,
            &mut next_weight,
            config,
            &mut values_scratch,
            &mut mask_scratch,
            &mut square_scratch,
            &mut gram_scratch,
            &mut count_scratch,
            &mut sum_scratch,
            &mut sq_scratch,
            &mut system_scratch,
            &mut rhs_scratch,
        )?;

        Ok(weights)
    }

    fn assign_ready_weights_naive(
        ring: &mut LdRingBuffer,
        weights: &mut [f64],
        next_weight: &mut usize,
        config: &LdResolvedConfig,
        values_scratch: &mut Mat<f64>,
        mask_scratch: &mut Mat<f64>,
        square_scratch: &mut Mat<f64>,
        gram_scratch: &mut Mat<f64>,
        count_scratch: &mut Mat<f64>,
        sum_scratch: &mut Mat<f64>,
        sq_scratch: &mut Mat<f64>,
        system_scratch: &mut Mat<f64>,
        rhs_scratch: &mut Mat<f64>,
    ) -> Result<(), HwePcaError> {
        let par = Par::Seq;
        while *next_weight < weights.len() {
            let position = match ring.position_of(*next_weight) {
                Some(pos) => pos,
                None => break,
            };

            let available = ring.len();
            if available == 0 {
                break;
            }

            let window_params = match &config.window {
                LdResolvedWindow::Sites { size } => {
                    let window_size = (*size).min(available).max(1);
                    let half_window = window_size / 2;
                    let start = if available <= window_size {
                        0
                    } else if position <= half_window {
                        0
                    } else {
                        let tail_start = available.saturating_sub(window_size);
                        min(position - half_window, tail_start)
                    };
                    let end = min(start + window_size, available);
                    let window_len = end - start;
                    if window_len == 0 {
                        None
                    } else {
                        let center = position - start;
                        let keep_from = next_weight.saturating_sub(window_size / 2);
                        Some((start, window_len, center, keep_from))
                    }
                }
                LdResolvedWindow::BasePairs { ranges, .. } => {
                    let range = &ranges[*next_weight];
                    if range.end <= range.start {
                        return Err(HwePcaError::InvalidInput(
                            "LD base-pair window produced an empty range",
                        ));
                    }
                    let last_index = range.end - 1;
                    match (ring.position_of(range.start), ring.position_of(last_index)) {
                        (Some(start_offset), Some(end_offset)) => {
                            let window_len = end_offset.saturating_sub(start_offset) + 1;
                            if window_len == 0 {
                                None
                            } else {
                                let center = position.saturating_sub(start_offset);
                                Some((start_offset, window_len, center, range.start))
                            }
                        }
                        _ => None,
                    }
                }
            };

            let Some((start, window_len, center, keep_from)) = window_params else {
                break;
            };

            let values_ref = ring.values();
            let first_slot = ring.slot_at(start);
            let contiguous = first_slot + window_len <= ring.capacity();
            let n_samples = values_ref.nrows();

            for col in 0..window_len {
                let slot = ring.slot_at(start + col);
                let src = ring.mask_slice(slot);
                let dst_col = mask_scratch.as_mut().col_mut(col);
                for (dst_value, &src_value) in dst_col.iter_mut().zip(src.iter()) {
                    *dst_value = f64::from(src_value);
                }
            }
            let mask_view = mask_scratch.as_ref().submatrix(0, 0, n_samples, window_len);

            let values_view = if contiguous {
                values_ref.submatrix(0, first_slot, n_samples, window_len)
            } else {
                for col in 0..window_len {
                    let slot = ring.slot_at(start + col);
                    let src = values_ref.col(slot);
                    let dst_col = values_scratch.as_mut().col_mut(col);
                    zip!(dst_col, src).for_each(|unzip!(dst, src)| {
                        *dst = *src;
                    });
                }
                values_scratch
                    .as_ref()
                    .submatrix(0, 0, n_samples, window_len)
            };

            let mut gram_view = gram_scratch
                .as_mut()
                .submatrix_mut(0, 0, window_len, window_len);
            matmul(
                gram_view.as_mut(),
                Accum::Replace,
                values_view.transpose(),
                values_view,
                1.0,
                par,
            );

            let mut squared_view = square_scratch
                .as_mut()
                .submatrix_mut(0, 0, n_samples, window_len);
            for col in 0..window_len {
                let src = values_view.col(col);
                let dst_col = squared_view.rb_mut().col_mut(col);
                zip!(dst_col, src).for_each(|unzip!(dst, src)| {
                    *dst = src * src;
                });
            }

            let mut count_view = count_scratch
                .as_mut()
                .submatrix_mut(0, 0, window_len, window_len);
            matmul(
                count_view.as_mut(),
                Accum::Replace,
                mask_view.transpose(),
                mask_view,
                1.0,
                par,
            );

            let mut sum_view = sum_scratch
                .as_mut()
                .submatrix_mut(0, 0, window_len, window_len);
            matmul(
                sum_view.as_mut(),
                Accum::Replace,
                values_view.transpose(),
                mask_view,
                1.0,
                par,
            );

            let mut sq_view = sq_scratch
                .as_mut()
                .submatrix_mut(0, 0, window_len, window_len);
            matmul(
                sq_view.as_mut(),
                Accum::Replace,
                squared_view.as_ref().transpose(),
                mask_view,
                1.0,
                par,
            );

            let mut system_view = system_scratch
                .as_mut()
                .submatrix_mut(0, 0, window_len, window_len);
            let mut rhs_view = rhs_scratch.as_mut().submatrix_mut(0, 0, window_len, 1);
            let weight = solve_ld_window_from_stats(
                gram_view.as_ref(),
                sum_view.as_ref(),
                sq_view.as_ref(),
                count_view.as_ref(),
                center,
                config.ridge,
                system_view.as_mut(),
                rhs_view.as_mut(),
            );
            weights[*next_weight] = weight;
            *next_weight += 1;

            ring.truncate_front(keep_from);
        }

        Ok(())
    }

    fn make_simple_scaler(data: &[f64], n_samples: usize) -> HweScaler {
        let observed_variants = data.len() / n_samples;
        let mut freqs = Vec::with_capacity(observed_variants);
        let mut scales = Vec::with_capacity(observed_variants);
        for variant in 0..observed_variants {
            let mut sum = 0.0;
            for sample in 0..n_samples {
                sum += data[variant * n_samples + sample];
            }
            let mean = sum / (n_samples as f64);
            let freq = (mean / 2.0).clamp(0.0, 1.0);
            let variance = (2.0 * freq * (1.0 - freq)).max(HWE_VARIANCE_EPSILON);
            let scale = variance.sqrt().max(HWE_SCALE_FLOOR);
            freqs.push(freq);
            scales.push(scale);
        }
        HweScaler::new(freqs, scales)
    }

    #[test]
    fn ld_weights_sites_match_reference() {
        let n_samples = 4;
        let observed_variants = 6;
        let data: Vec<f64> = vec![
            0.0, 1.0, 2.0, 1.0, // variant 0
            1.0, 2.0, 0.0, 2.0, // variant 1
            2.0, 1.0, 1.0, 0.0, // variant 2
            0.0, 2.0, 2.0, 1.0, // variant 3
            1.0, 0.0, 2.0, 2.0, // variant 4
            2.0, 2.0, 1.0, 0.0, // variant 5
        ];
        assert_eq!(data.len(), n_samples * observed_variants);

        let scaler = make_simple_scaler(&data, n_samples);
        let mut source =
            DenseBlockSource::new(&data, n_samples, observed_variants).expect("dense source");
        let progress = Arc::new(NoopFitProgress);
        let config = LdResolvedConfig {
            window: LdResolvedWindow::Sites { size: 3 },
            ridge: DEFAULT_LD_RIDGE,
        };

        let weights = compute_ld_weights(
            &mut source,
            &scaler,
            observed_variants,
            observed_variants,
            config.clone(),
            observed_variants,
            &progress,
            Par::Seq,
        )
        .expect("ld weights")
        .weights;

        let reference = compute_reference_ld_weights(&data, n_samples, &scaler, &config)
            .expect("reference weights");

        assert_eq!(weights.len(), reference.len());
        let max_diff = weights
            .iter()
            .zip(reference.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        assert!(max_diff < 1.0e-9, "max difference was {max_diff}");
    }

    #[test]
    fn ld_weights_bp_window_matches_reference() {
        let n_samples = 4;
        let observed_variants = 5;
        let data: Vec<f64> = vec![
            0.0, 1.0, 2.0, 1.0, // variant 0
            1.0, 2.0, 0.0, 2.0, // variant 1
            2.0, 1.0, 1.0, 0.0, // variant 2
            0.0, 2.0, 2.0, 1.0, // variant 3
            1.0, 0.0, 2.0, 2.0, // variant 4
        ];
        assert_eq!(data.len(), n_samples * observed_variants);

        let keys = vec![
            VariantKey::new("1", 100),
            VariantKey::new("1", 140),
            VariantKey::new("1", 200),
            VariantKey::new("1", 260),
            VariantKey::new("1", 320),
        ];
        let (ranges, capacity) = compute_ld_bp_ranges(&keys, 120).expect("ranges");
        let window = LdResolvedWindow::BasePairs {
            span_bp: 120,
            ranges: Arc::clone(&ranges),
            capacity,
        };
        let config = LdResolvedConfig {
            window,
            ridge: DEFAULT_LD_RIDGE,
        };

        let scaler = make_simple_scaler(&data, n_samples);
        let mut source =
            DenseBlockSource::new(&data, n_samples, observed_variants).expect("dense source");
        let progress = Arc::new(NoopFitProgress);

        let weights = compute_ld_weights(
            &mut source,
            &scaler,
            observed_variants,
            observed_variants,
            config.clone(),
            observed_variants,
            &progress,
            Par::Seq,
        )
        .expect("ld weights")
        .weights;

        let reference = compute_reference_ld_weights(&data, n_samples, &scaler, &config)
            .expect("reference weights");

        assert_eq!(weights.len(), reference.len());
        let max_diff = weights
            .iter()
            .zip(reference.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        assert!(max_diff < 1.0e-9, "max difference was {max_diff}");
    }

    #[test]
    fn ld_weights_sites_large_window_matches_reference() {
        let n_samples = 8;
        let observed_variants = 32;
        let mut data = Vec::with_capacity(n_samples * observed_variants);
        for variant in 0..observed_variants {
            for sample in 0..n_samples {
                let value = ((variant * 3 + sample * 5) % 4) as f64;
                data.push(value);
            }
        }

        let scaler = make_simple_scaler(&data, n_samples);
        let mut source =
            DenseBlockSource::new(&data, n_samples, observed_variants).expect("dense source");
        let progress = Arc::new(NoopFitProgress);
        let config = LdResolvedConfig {
            window: LdResolvedWindow::Sites { size: 17 },
            ridge: DEFAULT_LD_RIDGE,
        };

        let weights = compute_ld_weights(
            &mut source,
            &scaler,
            observed_variants,
            observed_variants,
            config.clone(),
            observed_variants,
            &progress,
            Par::Seq,
        )
        .expect("ld weights")
        .weights;

        let reference = compute_reference_ld_weights(&data, n_samples, &scaler, &config)
            .expect("reference weights");

        assert_eq!(weights.len(), reference.len());
        let max_diff = weights
            .iter()
            .zip(reference.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        assert!(max_diff < 1.0e-9, "max difference was {max_diff}");
    }

    #[test]
    fn ld_weights_bp_large_window_matches_reference() {
        let n_samples = 6;
        let observed_variants = 28;
        let mut data = Vec::with_capacity(n_samples * observed_variants);
        for variant in 0..observed_variants {
            for sample in 0..n_samples {
                let value = ((variant * 5 + sample * 7) % 6) as f64;
                data.push(value);
            }
        }

        let keys = (0..observed_variants)
            .map(|idx| VariantKey::new("1", 10 + (idx as u64) * 37))
            .collect::<Vec<_>>();
        let span_bp = 240;
        let (ranges, capacity) = compute_ld_bp_ranges(&keys, span_bp).expect("ranges");
        let window = LdResolvedWindow::BasePairs {
            span_bp,
            ranges: Arc::clone(&ranges),
            capacity,
        };
        let config = LdResolvedConfig {
            window,
            ridge: DEFAULT_LD_RIDGE,
        };

        let scaler = make_simple_scaler(&data, n_samples);
        let mut source =
            DenseBlockSource::new(&data, n_samples, observed_variants).expect("dense source");
        let progress = Arc::new(NoopFitProgress);

        let weights = compute_ld_weights(
            &mut source,
            &scaler,
            observed_variants,
            observed_variants,
            config.clone(),
            observed_variants,
            &progress,
            Par::Seq,
        )
        .expect("ld weights")
        .weights;

        let reference = compute_reference_ld_weights(&data, n_samples, &scaler, &config)
            .expect("reference weights");

        assert_eq!(weights.len(), reference.len());
        let max_diff = weights
            .iter()
            .zip(reference.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        assert!(max_diff < 1.0e-9, "max difference was {max_diff}");
    }

    #[test]
    fn bp_window_treated_as_total_span() {
        let keys = vec![
            VariantKey::new("1", 100),
            VariantKey::new("1", 140),
            VariantKey::new("1", 180),
            VariantKey::new("1", 220),
        ];

        let (ranges, capacity) = compute_ld_bp_ranges(&keys, 100).expect("ranges");

        assert_eq!(capacity, 3);
        assert_eq!(ranges.len(), keys.len());

        assert_eq!((ranges[1].start, ranges[1].end), (0, 3));
        assert_eq!((ranges[2].start, ranges[2].end), (1, 4));
    }

    #[test]
    fn bp_window_uses_total_span_for_odds() {
        let keys = vec![VariantKey::new("1", 1), VariantKey::new("1", 52)];

        let (ranges, capacity) = compute_ld_bp_ranges(&keys, 101).expect("ranges");

        assert_eq!(capacity, 1);
        assert_eq!(ranges.len(), keys.len());

        assert_eq!((ranges[0].start, ranges[0].end), (0, 1));
        assert_eq!((ranges[1].start, ranges[1].end), (1, 2));
    }

    #[test]
    fn fast_path_matches_masked_when_no_missingness() {
        let mut masked = (0..128)
            .map(|i| f64::from(i % 7) * 0.25)
            .collect::<Vec<_>>();
        let mut fast = masked.clone();
        let mean = 0.75;
        let inv = 0.5;

        standardize_column_simd(masked.as_mut_slice(), mean, inv);
        standardize_column_simd_full(fast.as_mut_slice(), mean, inv);

        for (lhs, rhs) in masked.iter().zip(fast.iter()) {
            assert!((lhs - rhs).abs() < 1e-15);
        }
    }

    #[test]
    fn negative_eigenvalues_do_not_produce_nan_scores() {
        let n_samples = 3;
        let eigenpairs = Eigenpairs {
            values: vec![-1.0e-12, 0.5],
            vectors: Mat::from_fn(n_samples, 2, |row, col| if row == col { 1.0 } else { 0.0 }),
        };

        let (singular_values, scores) = build_sample_scores(n_samples, &eigenpairs);

        assert_eq!(singular_values.len(), 2);
        assert!(singular_values.iter().all(|value| value.is_finite()));
        for row in 0..scores.nrows() {
            for col in 0..scores.ncols() {
                assert!(scores[(row, col)].is_finite());
            }
        }

        assert_eq!(singular_values[0], 0.0);
        for row in 0..scores.nrows() {
            assert_eq!(scores[(row, 0)], 0.0);
        }
    }

    #[test]
    fn variant_stats_cache_grows_lazily() {
        let block_capacity = 8;
        let hint = 1 << 15;
        let mut cache = VariantStatsCache::new(block_capacity, hint);
        let par = get_global_parallelism();
        let n_samples = 4;

        assert_eq!(cache.frequencies.len(), 0);
        assert_eq!(cache.scales.len(), 0);

        let first_block = Mat::from_fn(n_samples, 3, |row, col| (row + col) as f64);
        cache.ensure_statistics(first_block.as_ref(), 0..3, par);
        assert_eq!(cache.frequencies.len(), 3);
        assert_eq!(cache.scales.len(), 3);
        assert_eq!(cache.len(), 3);

        let second_block = Mat::from_fn(n_samples, 2, |row, col| (row + col + 1) as f64);
        cache.ensure_statistics(second_block.as_ref(), 3..5, par);
        assert_eq!(cache.frequencies.len(), 5);
        assert_eq!(cache.scales.len(), 5);
        assert_eq!(cache.len(), 5);

        cache.finalize();
        assert_eq!(cache.len(), 5);
        assert_eq!(cache.frequencies.len(), 5);
        assert_eq!(cache.scales.len(), 5);
    }

    #[test]
    fn variant_stats_cache_handles_zero_hint() {
        let block_capacity = 4;
        let mut cache = VariantStatsCache::new(block_capacity, 0);
        let par = get_global_parallelism();
        let n_samples = 3;
        let block = Mat::from_fn(n_samples, 2, |row, col| (row * 2 + col) as f64);

        cache.ensure_statistics(block.as_ref(), 0..2, par);
        assert_eq!(cache.frequencies.len(), 2);
        assert_eq!(cache.scales.len(), 2);

        cache.ensure_statistics(block.as_ref(), 2..4, par);
        assert_eq!(cache.frequencies.len(), 4);
        assert_eq!(cache.scales.len(), 4);
    }

    #[test]
    fn ld_weights_are_applied_during_standardization() {
        use std::sync::Arc;

        let scaler = HweScaler::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let dense_data = vec![0.0; 8];
        let mut source = DenseBlockSource::new(&dense_data, 4, 2).unwrap();
        let weights = Arc::from(vec![0.5, 2.0].into_boxed_slice());
        let operator: StandardizedCovarianceOp<'_, DenseBlockSource<'_>, NoopFitProgress> =
            StandardizedCovarianceOp::new(&mut source, 2, 2, 2, scaler, Some(weights));

        let mut block_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        {
            let mut block = MatMut::from_column_major_slice_mut(&mut block_data, 4, 2);
            operator.standardize_block_in_place(block.as_mut(), 0..2, get_global_parallelism());
        }

        let expected = vec![0.5, 1.0, 1.5, 2.0, 10.0, 12.0, 14.0, 16.0];
        assert_eq!(block_data, expected);
    }

    #[test]
    fn ld_weights_are_ignored_when_absent() {
        let scaler = HweScaler::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let dense_data = vec![0.0; 8];
        let mut source = DenseBlockSource::new(&dense_data, 4, 2).unwrap();
        let operator: StandardizedCovarianceOp<'_, DenseBlockSource<'_>, NoopFitProgress> =
            StandardizedCovarianceOp::new(&mut source, 2, 2, 2, scaler, None);

        let mut block_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        {
            let mut block = MatMut::from_column_major_slice_mut(&mut block_data, 4, 2);
            operator.standardize_block_in_place(block.as_mut(), 0..2, get_global_parallelism());
        }

        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(block_data, expected);
    }

    const TEST_VCF_URL: &str = "https://raw.githubusercontent.com/SauersML/genomic_pca/refs/heads/main/tests/chr22_chunk.vcf.gz";
    const MAX_TEST_VARIANTS: usize = 32;
    const MAX_TEST_SAMPLES: usize = 8;
    const TEST_COMPONENTS: usize = 4;

    struct LimitedBlockSource<T> {
        inner: T,
        sample_limit: usize,
        variant_limit: usize,
        remaining_variants: usize,
        inner_samples: usize,
        scratch: Vec<f64>,
    }

    impl<T> LimitedBlockSource<T>
    where
        T: VariantBlockSource,
    {
        fn new(inner: T, max_samples: usize, max_variants: usize) -> Self {
            let inner_samples = inner.n_samples();
            let inner_variants = inner.n_variants();
            let sample_limit = max_samples.max(2).min(inner_samples);
            let variant_limit = if inner_variants == 0 {
                max_variants.max(1)
            } else {
                max_variants.max(1).min(inner_variants)
            };
            let scratch = vec![0.0; inner_samples * variant_limit];
            Self {
                inner,
                sample_limit,
                variant_limit,
                remaining_variants: variant_limit,
                inner_samples,
                scratch,
            }
        }
    }

    impl<T> VariantBlockSource for LimitedBlockSource<T>
    where
        T: VariantBlockSource,
    {
        type Error = T::Error;

        fn n_samples(&self) -> usize {
            self.sample_limit
        }

        fn n_variants(&self) -> usize {
            self.variant_limit
        }

        fn reset(&mut self) -> Result<(), Self::Error> {
            self.inner.reset()?;
            self.remaining_variants = self.variant_limit;
            Ok(())
        }

        fn next_block_into(
            &mut self,
            max_variants: usize,
            storage: &mut [f64],
        ) -> Result<usize, Self::Error> {
            if self.remaining_variants == 0 {
                return Ok(0);
            }

            let request = max_variants.min(self.remaining_variants);
            if request == 0 {
                return Ok(0);
            }

            let inner_len = self.inner_samples * request;
            let read = self
                .inner
                .next_block_into(request, &mut self.scratch[..inner_len])?;
            let consumed = read.min(self.remaining_variants);
            self.remaining_variants -= consumed;

            let samples = self.sample_limit;
            for variant_idx in 0..read {
                let inner_offset = variant_idx * self.inner_samples;
                let outer_offset = variant_idx * samples;
                let src = &self.scratch[inner_offset..inner_offset + samples];
                let dst = &mut storage[outer_offset..outer_offset + samples];
                dst.copy_from_slice(src);
            }

            Ok(read)
        }
    }

    #[test]
    fn fit_hwe_pca_from_http_vcf_stream() {
        let path = Path::new(TEST_VCF_URL);
        let dataset = GenotypeDataset::open(path)
            .unwrap_or_else(|err| panic!("Failed to open dataset: {err}"));

        let block_source = dataset
            .block_source()
            .unwrap_or_else(|err| panic!("Failed to create block source: {err}"));

        let mut limited_source =
            LimitedBlockSource::new(block_source, MAX_TEST_SAMPLES, MAX_TEST_VARIANTS);
        let expected_variants = limited_source.n_variants();
        let expected_samples = limited_source.n_samples();

        let model = HwePcaModel::fit_k(&mut limited_source, TEST_COMPONENTS)
            .unwrap_or_else(|err| panic!("Failed to fit PCA model: {err}"));

        assert_eq!(expected_samples, model.n_samples());
        assert_eq!(expected_variants, model.n_variants());
        assert!(model.components() > 0);
    }
}
