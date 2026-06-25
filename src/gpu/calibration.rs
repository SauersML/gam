use crate::gpu::device::GpuDeviceInfo;
use crate::gpu::gpu_error::GpuError;
use crate::gpu::policy::GpuDispatchPolicy;
use crate::linalg::faer_ndarray::FaerCholesky;
use faer::Side;
use gam_runtime::warm_start::{Fingerprint, Fingerprinter};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const SCHEMA_VERSION: u32 = 1;
const CACHE_ROOT_COMPONENTS: [&str; 4] = ["gam", "gpu", "policy", "v1"];
const GEMM_DIMS: [usize; 3] = [64, 128, 256];
const POTRF_DIMS: [usize; 3] = [64, 128, 256];
const XTWX_DIMS: [(usize, usize); 3] = [(2048, 32), (4096, 64), (8192, 96)];
const GPU_WIN_RATIO: f64 = 0.95;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CachedCalibration {
    schema_version: u32,
    device_fingerprint: String,
    policy: GpuDispatchPolicy,
    measurements: Vec<MeasurementRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MeasurementRecord {
    operation: String,
    rows: usize,
    cols: usize,
    inner: usize,
    flops: usize,
    cpu_seconds: f64,
    gpu_seconds: f64,
}

#[derive(Clone, Debug)]
struct Measurement {
    operation: &'static str,
    rows: usize,
    cols: usize,
    inner: usize,
    flops: usize,
    cpu_seconds: f64,
    gpu_seconds: f64,
}

pub(crate) fn calibrated_policy_for_device(device: &GpuDeviceInfo) -> GpuDispatchPolicy {
    let fingerprint = device_fingerprint(device);
    if let Some(cached) = load_cached_policy(fingerprint) {
        log::info!(
            "[GPU] loaded calibrated dispatch policy for {} ({fingerprint})",
            device.name
        );
        return cached;
    }

    match calibrate_device(device, fingerprint) {
        Ok(record) => {
            let policy = record.policy.clone();
            store_cached_policy(fingerprint, &record);
            policy
        }
        Err(err) => {
            log::warn!(
                "[GPU] dispatch calibration unavailable for {}: {}; using default policy",
                device.name,
                err
            );
            GpuDispatchPolicy::default()
        }
    }
}

fn calibrate_device(
    device: &GpuDeviceInfo,
    fingerprint: Fingerprint,
) -> Result<CachedCalibration, GpuError> {
    let mut measurements = Vec::new();
    measurements.extend(measure_gemm(device.ordinal)?);
    measurements.extend(measure_potrf(device.ordinal)?);
    measurements.extend(measure_xtwx(device.ordinal)?);
    if measurements.is_empty() {
        return Err(GpuError::CalibrationFailed {
            reason: "no GPU calibration measurements completed".to_string(),
        });
    }

    let mut policy = GpuDispatchPolicy::default();
    if let Some(flops) = crossover_flops(&measurements, "gemm", policy.gemm_min_flops) {
        policy.gemm_min_flops = flops;
    }
    if let Some(flops) = crossover_flops(&measurements, "xtwx", policy.xtwx_flops_min) {
        policy.xtwx_flops_min = flops;
    }
    if let Some(rows) = crossover_rows(&measurements, "xtwx", policy.xtwx_n_min) {
        policy.xtwx_n_min = rows;
        policy.row_kernel_min_n = rows;
        policy.fused_kernel_min_n = rows.saturating_mul(2);
    }
    if let Some(p) = crossover_rows(&measurements, "potrf", policy.potrf_min_p) {
        policy.potrf_min_p = p;
        policy.prefer_gpu_factorization_min_p = p;
    }

    log::info!(
        "[GPU] calibrated dispatch policy for {} ({fingerprint}) from {} measurements",
        device.name,
        measurements.len()
    );

    Ok(CachedCalibration {
        schema_version: SCHEMA_VERSION,
        device_fingerprint: fingerprint.to_hex(),
        policy,
        measurements: measurements
            .into_iter()
            .map(Measurement::into_record)
            .collect(),
    })
}

fn measure_gemm(ordinal: usize) -> Result<Vec<Measurement>, GpuError> {
    let mut out = Vec::with_capacity(GEMM_DIMS.len());
    for dim in GEMM_DIMS {
        let a = deterministic_matrix(dim, dim, 0.13);
        let b = deterministic_matrix(dim, dim, 0.37);
        let cpu_seconds = time_cpu(|| a.dot(&b))?;
        let gpu_seconds = time_gpu(|| {
            crate::gpu::blas::gemm_on_ordinal_cuda(ordinal, a.view(), b.view(), false, false)
        })?;
        out.push(Measurement {
            operation: "gemm",
            rows: dim,
            cols: dim,
            inner: dim,
            flops: 2usize
                .saturating_mul(dim)
                .saturating_mul(dim)
                .saturating_mul(dim),
            cpu_seconds,
            gpu_seconds,
        });
    }
    Ok(out)
}

fn measure_potrf(ordinal: usize) -> Result<Vec<Measurement>, GpuError> {
    let mut out = Vec::with_capacity(POTRF_DIMS.len());
    for dim in POTRF_DIMS {
        let a = deterministic_spd_matrix(dim);
        let cpu_seconds = time_gpu_result(|| {
            a.cholesky(Side::Lower)
                .map(|factor| factor.lower_triangular())
                .map_err(|err| format!("cpu POTRF failed: {err}"))
        })?;
        let gpu_seconds = time_gpu_result(|| {
            crate::gpu::solver::cholesky_lower_on_ordinal_gpu(ordinal, a.view())
        })?;
        out.push(Measurement {
            operation: "potrf",
            rows: dim,
            cols: dim,
            inner: dim,
            flops: dim.saturating_mul(dim).saturating_mul(dim) / 3,
            cpu_seconds,
            gpu_seconds,
        });
    }
    Ok(out)
}

fn measure_xtwx(ordinal: usize) -> Result<Vec<Measurement>, GpuError> {
    let mut out = Vec::with_capacity(XTWX_DIMS.len());
    for (n, p) in XTWX_DIMS {
        let x = deterministic_matrix(n, p, 0.61);
        let w = deterministic_weights(n);
        let cpu_seconds = time_cpu(|| cpu_xtwx(&x, &w))?;
        let gpu_seconds =
            time_gpu(|| crate::gpu::blas::xt_diag_x_on_ordinal_cuda(ordinal, x.view(), w.view()))?;
        out.push(Measurement {
            operation: "xtwx",
            rows: n,
            cols: p,
            inner: p,
            flops: 2usize.saturating_mul(n).saturating_mul(p).saturating_mul(p),
            cpu_seconds,
            gpu_seconds,
        });
    }
    Ok(out)
}

fn time_cpu<F>(mut f: F) -> Result<f64, GpuError>
where
    F: FnMut() -> Array2<f64>,
{
    time_gpu_result(|| Result::<Array2<f64>, GpuError>::Ok(f()))
}

fn time_gpu<F>(mut f: F) -> Result<f64, GpuError>
where
    F: FnMut() -> Option<Array2<f64>>,
{
    time_gpu_result(|| {
        f().ok_or_else(|| GpuError::CalibrationFailed {
            reason: "GPU calibration kernel returned no result".to_string(),
        })
    })
}

fn time_gpu_result<F, E>(mut f: F) -> Result<f64, GpuError>
where
    F: FnMut() -> Result<Array2<f64>, E>,
    E: std::fmt::Display,
{
    let start = Instant::now();
    let out = f().map_err(|err| GpuError::CalibrationFailed {
        reason: err.to_string(),
    })?;
    let elapsed = start.elapsed().as_secs_f64();
    let checksum = out.iter().fold(0.0, |acc, value| acc + value.abs());
    if elapsed.is_finite() && elapsed > 0.0 && checksum.is_finite() {
        Ok(elapsed)
    } else {
        Err(GpuError::CalibrationFailed {
            reason: format!(
                "invalid calibration timing/checksum: elapsed={elapsed}, checksum={checksum}"
            ),
        })
    }
}

fn crossover_flops(
    measurements: &[Measurement],
    operation: &'static str,
    fallback: usize,
) -> Option<usize> {
    crossover_measurement(measurements, operation)
        .map(|measurement| measurement.flops.max(1))
        .or_else(|| {
            measurements
                .iter()
                .filter(|measurement| measurement.operation == operation)
                .map(|measurement| measurement.flops)
                .max()
                .map(|max_seen| fallback.max(max_seen.saturating_mul(2)))
        })
}

fn crossover_rows(
    measurements: &[Measurement],
    operation: &'static str,
    fallback: usize,
) -> Option<usize> {
    crossover_measurement(measurements, operation)
        .map(|measurement| measurement.rows.max(1))
        .or_else(|| {
            measurements
                .iter()
                .filter(|measurement| measurement.operation == operation)
                .map(|measurement| measurement.rows)
                .max()
                .map(|max_seen| fallback.max(max_seen.saturating_mul(2)))
        })
}

fn crossover_measurement<'a>(
    measurements: &'a [Measurement],
    operation: &'static str,
) -> Option<&'a Measurement> {
    measurements
        .iter()
        .filter(|measurement| measurement.operation == operation)
        .find(|measurement| measurement.gpu_seconds <= measurement.cpu_seconds * GPU_WIN_RATIO)
}

fn deterministic_matrix(rows: usize, cols: usize, phase: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(row, col)| {
        let x = (row as f64 + 1.0) * 0.017 + (col as f64 + 1.0) * 0.031 + phase;
        x.sin() + 0.25 * (2.0 * x).cos()
    })
}

fn deterministic_spd_matrix(dim: usize) -> Array2<f64> {
    let a = deterministic_matrix(dim, dim, 0.89);
    let mut spd = a.t().dot(&a);
    for idx in 0..dim {
        spd[[idx, idx]] += dim as f64;
    }
    spd
}

fn deterministic_weights(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n, |idx| 0.5 + ((idx as f64 + 1.0) * 0.019).sin().abs())
}

fn cpu_xtwx(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    let mut weighted = x.clone();
    for (mut row, weight) in weighted.outer_iter_mut().zip(w.iter()) {
        row *= *weight;
    }
    x.t().dot(&weighted)
}

fn load_cached_policy(fingerprint: Fingerprint) -> Option<GpuDispatchPolicy> {
    let path = cache_path(fingerprint);
    let bytes = fs::read(path).ok()?;
    let record: CachedCalibration = serde_json::from_slice(&bytes).ok()?;
    if record.schema_version == SCHEMA_VERSION && record.device_fingerprint == fingerprint.to_hex()
    {
        Some(record.policy)
    } else {
        None
    }
}

fn store_cached_policy(fingerprint: Fingerprint, record: &CachedCalibration) {
    let path = cache_path(fingerprint);
    if let Some(parent) = path.parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            log::warn!("[GPU] unable to create calibration cache dir: {err}");
            return;
        }
    }
    let tmp = path.with_extension("json.tmp");
    let bytes = match serde_json::to_vec_pretty(record) {
        Ok(bytes) => bytes,
        Err(err) => {
            log::warn!("[GPU] unable to serialize calibration cache: {err}");
            return;
        }
    };
    if let Err(err) = fs::write(&tmp, bytes).and_then(|_| fs::rename(&tmp, &path)) {
        log::warn!("[GPU] unable to write calibration cache: {err}");
    }
}

fn cache_path(fingerprint: Fingerprint) -> PathBuf {
    let mut root = std::env::temp_dir();
    for component in CACHE_ROOT_COMPONENTS {
        root.push(component);
    }
    root.push(format!("{fingerprint}.json"));
    root
}

fn device_fingerprint(device: &GpuDeviceInfo) -> Fingerprint {
    let mut fp = Fingerprinter::new();
    fp.absorb_tag(b"gpu-dispatch-calibration");
    fp.absorb_u64(b"schema-version", u64::from(SCHEMA_VERSION));
    fp.absorb_str(b"name", &device.name);
    fp.absorb_u64(
        b"compute-major",
        u64::try_from(device.capability.compute_major).unwrap_or(0),
    );
    fp.absorb_u64(
        b"compute-minor",
        u64::try_from(device.capability.compute_minor).unwrap_or(0),
    );
    fp.absorb_u64(b"sm-count", u64::try_from(device.sm_count).unwrap_or(0));
    fp.absorb_u64(
        b"max-threads-per-sm",
        u64::try_from(device.max_threads_per_sm).unwrap_or(0),
    );
    fp.absorb_u64(
        b"max-shared-mem-per-block",
        device.max_shared_mem_per_block as u64,
    );
    fp.absorb_u64(b"l2-cache-bytes", device.l2_cache_bytes as u64);
    fp.absorb_u64(b"total-mem-bytes", device.total_mem_bytes as u64);
    fp.absorb_u64(b"ecc-enabled", bool_fingerprint_value(device.ecc_enabled));
    fp.absorb_u64(b"integrated", bool_fingerprint_value(device.integrated));
    fp.absorb_u64(b"mig-mode", bool_fingerprint_value(device.mig_mode));
    fp.finalize()
}

const fn bool_fingerprint_value(value: bool) -> u64 {
    if value { 1 } else { 0 }
}

impl Measurement {
    fn into_record(self) -> MeasurementRecord {
        MeasurementRecord {
            operation: self.operation.to_string(),
            rows: self.rows,
            cols: self.cols,
            inner: self.inner,
            flops: self.flops,
            cpu_seconds: self.cpu_seconds,
            gpu_seconds: self.gpu_seconds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::GpuCapability;

    fn measurement(
        operation: &'static str,
        rows: usize,
        cols: usize,
        flops: usize,
        cpu_seconds: f64,
        gpu_seconds: f64,
    ) -> Measurement {
        Measurement {
            operation,
            rows,
            cols,
            inner: cols,
            flops,
            cpu_seconds,
            gpu_seconds,
        }
    }

    #[test]
    fn calibration_crossover_uses_first_measured_gpu_win() {
        let measurements = vec![
            measurement("gemm", 64, 64, 524_288, 0.001, 0.004),
            measurement("gemm", 128, 128, 4_194_304, 0.010, 0.009),
            measurement("gemm", 256, 256, 33_554_432, 0.080, 0.010),
        ];

        assert_eq!(
            crossover_flops(&measurements, "gemm", 100_000_000),
            Some(4_194_304)
        );
    }

    #[test]
    fn calibration_crossover_raises_threshold_when_gpu_never_wins() {
        let measurements = vec![
            measurement("xtwx", 2_048, 32, 4_194_304, 0.001, 0.004),
            measurement("xtwx", 4_096, 64, 33_554_432, 0.010, 0.040),
            measurement("xtwx", 8_192, 96, 150_994_944, 0.080, 0.400),
        ];

        assert_eq!(
            crossover_flops(&measurements, "xtwx", 100_000_000),
            Some(301_989_888)
        );
        assert_eq!(crossover_rows(&measurements, "xtwx", 50_000), Some(50_000));
    }

    #[test]
    fn calibration_cache_key_tracks_device_fingerprint() {
        let device = GpuDeviceInfo {
            ordinal: 0,
            name: "unit-test GPU".to_string(),
            capability: GpuCapability::from_compute_capability(8, 0),
            sm_count: 108,
            max_threads_per_sm: 2048,
            max_shared_mem_per_block: 99_328,
            l2_cache_bytes: 40 * 1024 * 1024,
            total_mem_bytes: 80 * 1024 * 1024 * 1024,
            free_mem_bytes: 70 * 1024 * 1024 * 1024,
            ecc_enabled: true,
            integrated: false,
            mig_mode: false,
        };

        let fingerprint = device_fingerprint(&device);
        let path = cache_path(fingerprint);
        assert!(path.ends_with(format!("{}.json", fingerprint.to_hex())));
        assert!(
            path.components()
                .map(|component| component.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .windows(CACHE_ROOT_COMPONENTS.len())
                .any(|window| window == CACHE_ROOT_COMPONENTS)
        );
    }
}
