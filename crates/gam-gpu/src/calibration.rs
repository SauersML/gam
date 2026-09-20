use crate::device::GpuDeviceInfo;
use crate::gpu_error::GpuError;
use crate::policy::GpuDispatchPolicy;
use faer::Side;
use gam_linalg::faer_ndarray::FaerCholesky;
use gam_runtime::warm_start::{Fingerprint, Fingerprinter};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const SCHEMA_VERSION: u32 = 2;
const CACHE_ROOT_COMPONENTS: [&str; 4] = ["gam", "gpu", "policy", "v1"];
const GEMM_DIMS: [usize; 3] = [64, 128, 256];
const POTRF_DIMS: [usize; 3] = [64, 128, 256];
const XTWX_DIMS: [(usize, usize); 3] = [(2048, 32), (4096, 64), (8192, 96)];
const GPU_WIN_RATIO: f64 = 0.95;

// Pin the pre-probe admission floors to the smallest calibration measurements:
// `crossover_flops` / `crossover_rows` can lower `gemm_min_flops` /
// `potrf_min_p` at most to the smallest measured GEMM's flop count / smallest
// POTRF dimension. `DispatchOp::admissible_under_any_policy` (and every
// pre-probe size gate built on these constants) relies on that bound, so a
// change to the measurement grid must consciously update the constants.
const _: () = assert!(
    2 * (GEMM_DIMS[0] as u128) * (GEMM_DIMS[0] as u128) * (GEMM_DIMS[0] as u128)
        == GpuDispatchPolicy::MIN_CALIBRATABLE_GEMM_FLOPS
);
const _: () = assert!(POTRF_DIMS[0] == GpuDispatchPolicy::MIN_CALIBRATABLE_POTRF_P);

/// What the calibration cache persists: the device's CPU/GPU timings at each
/// point of the fixed measurement grid, in grid order. The policy is NOT
/// cached. It is rebuilt from these timings and the current build's
/// [`GpuDispatchPolicy::default`] on every load, so the never-calibrated
/// fields always carry this build's defaults and the calibrated fields can only
/// land on a current grid point (the bound the pre-probe gates rely on).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct CachedCalibration {
    schema_version: u32,
    device_fingerprint: String,
    timings: GridTimings,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct GridTimings {
    gemm: [Timing; GEMM_DIMS.len()],
    potrf: [Timing; POTRF_DIMS.len()],
    xtwx: [Timing; XTWX_DIMS.len()],
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Timing {
    cpu_seconds: f64,
    gpu_seconds: f64,
}

#[derive(Clone, Debug)]
struct Measurement {
    operation: &'static str,
    rows: usize,
    flops: usize,
    timing: Timing,
}

pub(crate) fn calibrated_policy_for_device(device: &GpuDeviceInfo) -> GpuDispatchPolicy {
    let fingerprint = device_fingerprint(device);
    if let Some(cached) = load_cached_policy(fingerprint) {
        log::debug!(
            "[GPU] loaded calibrated dispatch policy for {} ({fingerprint})",
            device.name
        );
        return cached;
    }

    match calibrate_device(device.ordinal) {
        Ok(timings) => {
            let policy = policy_from_timings(&timings);
            log::debug!(
                "[GPU] calibrated dispatch policy for {} ({fingerprint})",
                device.name
            );
            store_cached_timings(fingerprint, timings);
            policy
        }
        Err(err) => {
            log::debug!(
                "[GPU] dispatch calibration unavailable for {}: {}; using default policy",
                device.name,
                err
            );
            GpuDispatchPolicy::default()
        }
    }
}

fn calibrate_device(ordinal: usize) -> Result<GridTimings, GpuError> {
    Ok(GridTimings {
        gemm: time_grid(GEMM_DIMS, |dim| {
            let a = deterministic_matrix(dim, dim, 0.13);
            let b = deterministic_matrix(dim, dim, 0.37);
            Ok(Timing {
                cpu_seconds: time_cpu(|| a.dot(&b))?,
                gpu_seconds: time_gpu(|| {
                    crate::blas::gemm_on_ordinal_cuda(ordinal, a.view(), b.view(), false, false)
                })?,
            })
        })?,
        potrf: time_grid(POTRF_DIMS, |dim| {
            let a = deterministic_spd_matrix(dim);
            Ok(Timing {
                cpu_seconds: time_gpu_result(|| {
                    a.cholesky(Side::Lower)
                        .map(|factor| factor.lower_triangular())
                        .map_err(|err| format!("cpu POTRF failed: {err}"))
                })?,
                gpu_seconds: time_gpu_result(|| {
                    crate::solver::cholesky_lower_on_ordinal_gpu(ordinal, a.view())
                })?,
            })
        })?,
        xtwx: time_grid(XTWX_DIMS, |(n, p)| {
            let x = deterministic_matrix(n, p, 0.61);
            let w = deterministic_weights(n);
            Ok(Timing {
                cpu_seconds: time_cpu(|| cpu_xtwx(&x, &w))?,
                gpu_seconds: time_gpu(|| {
                    crate::blas::xt_diag_x_on_ordinal_cuda(ordinal, x.view(), w.view())
                })?,
            })
        })?,
    })
}

fn time_grid<D: Copy, const N: usize>(
    dims: [D; N],
    mut time: impl FnMut(D) -> Result<Timing, GpuError>,
) -> Result<[Timing; N], GpuError> {
    let mut out = [Timing {
        cpu_seconds: 0.0,
        gpu_seconds: 0.0,
    }; N];
    for (slot, dim) in out.iter_mut().zip(dims) {
        *slot = time(dim)?;
    }
    Ok(out)
}

impl GridTimings {
    /// Pair each timing with its grid point's shape and flop count.
    fn measurements(&self) -> Vec<Measurement> {
        let gemm = GEMM_DIMS
            .iter()
            .zip(self.gemm)
            .map(|(&dim, timing)| Measurement {
                operation: "gemm",
                rows: dim,
                flops: 2usize
                    .saturating_mul(dim)
                    .saturating_mul(dim)
                    .saturating_mul(dim),
                timing,
            });
        let potrf = POTRF_DIMS
            .iter()
            .zip(self.potrf)
            .map(|(&dim, timing)| Measurement {
                operation: "potrf",
                rows: dim,
                flops: dim.saturating_mul(dim).saturating_mul(dim) / 3,
                timing,
            });
        let xtwx = XTWX_DIMS
            .iter()
            .zip(self.xtwx)
            .map(|(&(n, p), timing)| Measurement {
                operation: "xtwx",
                rows: n,
                flops: 2usize.saturating_mul(n).saturating_mul(p).saturating_mul(p),
                timing,
            });
        gemm.chain(potrf).chain(xtwx).collect()
    }

    /// Every timing is a finite positive duration, as `time_gpu_result`
    /// guarantees for a fresh measurement.
    fn is_valid(&self) -> bool {
        self.gemm
            .iter()
            .chain(&self.potrf)
            .chain(&self.xtwx)
            .all(|timing| {
                [timing.cpu_seconds, timing.gpu_seconds]
                    .into_iter()
                    .all(|seconds| seconds.is_finite() && seconds > 0.0)
            })
    }
}

fn policy_from_timings(timings: &GridTimings) -> GpuDispatchPolicy {
    let measurements = timings.measurements();
    let mut policy = GpuDispatchPolicy::default();
    if let Some(flops) = crossover_flops(&measurements, "gemm", policy.gemm_min_flops) {
        policy.gemm_min_flops = flops;
    }
    if let Some(flops) = crossover_flops(&measurements, "xtwx", policy.xtwx_flops_min) {
        policy.xtwx_flops_min = flops;
    }
    if let Some(p) = crossover_rows(&measurements, "potrf", policy.potrf_min_p) {
        policy.potrf_min_p = p;
    }
    policy
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
        .find(|measurement| {
            measurement.timing.gpu_seconds <= measurement.timing.cpu_seconds * GPU_WIN_RATIO
        })
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
    (record.schema_version == SCHEMA_VERSION
        && record.device_fingerprint == fingerprint.to_hex()
        && record.timings.is_valid())
    .then(|| policy_from_timings(&record.timings))
}

fn store_cached_timings(fingerprint: Fingerprint, timings: GridTimings) {
    let record = CachedCalibration {
        schema_version: SCHEMA_VERSION,
        device_fingerprint: fingerprint.to_hex(),
        timings,
    };
    let path = cache_path(fingerprint);
    if let Some(parent) = path.parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            log::debug!("[GPU] unable to create calibration cache dir: {err}");
            return;
        }
    }
    let tmp = path.with_extension("json.tmp");
    let bytes = match serde_json::to_vec_pretty(&record) {
        Ok(bytes) => bytes,
        Err(err) => {
            log::debug!("[GPU] unable to serialize calibration cache: {err}");
            return;
        }
    };
    if let Err(err) = fs::write(&tmp, bytes).and_then(|_| fs::rename(&tmp, &path)) {
        log::debug!("[GPU] unable to write calibration cache: {err}");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuCapability;

    fn measurement(
        operation: &'static str,
        rows: usize,
        flops: usize,
        cpu_seconds: f64,
        gpu_seconds: f64,
    ) -> Measurement {
        Measurement {
            operation,
            rows,
            flops,
            timing: Timing {
                cpu_seconds,
                gpu_seconds,
            },
        }
    }

    #[test]
    fn calibration_crossover_uses_first_measured_gpu_win() {
        let measurements = vec![
            measurement("gemm", 64, 524_288, 0.001, 0.004),
            measurement("gemm", 128, 4_194_304, 0.010, 0.009),
            measurement("gemm", 256, 33_554_432, 0.080, 0.010),
        ];

        assert_eq!(
            crossover_flops(&measurements, "gemm", 100_000_000),
            Some(4_194_304)
        );
    }

    #[test]
    fn calibration_crossover_raises_threshold_when_gpu_never_wins() {
        let measurements = vec![
            measurement("xtwx", 2_048, 4_194_304, 0.001, 0.004),
            measurement("xtwx", 4_096, 33_554_432, 0.010, 0.040),
            measurement("xtwx", 8_192, 150_994_944, 0.080, 0.400),
        ];

        assert_eq!(
            crossover_flops(&measurements, "xtwx", 100_000_000),
            Some(301_989_888)
        );
        assert_eq!(crossover_rows(&measurements, "xtwx", 50_000), Some(50_000));
    }

    fn uniform_timings(cpu_seconds: f64, gpu_seconds: f64) -> GridTimings {
        let timing = Timing {
            cpu_seconds,
            gpu_seconds,
        };
        GridTimings {
            gemm: [timing; GEMM_DIMS.len()],
            potrf: [timing; POTRF_DIMS.len()],
            xtwx: [timing; XTWX_DIMS.len()],
        }
    }

    #[test]
    fn cached_timings_rebuild_policy_on_current_grid_and_defaults() {
        let record = CachedCalibration {
            schema_version: SCHEMA_VERSION,
            device_fingerprint: "unit-test".to_string(),
            timings: uniform_timings(1.0, 0.5),
        };
        let bytes = serde_json::to_vec(&record).expect("serialize calibration record");
        let loaded: CachedCalibration =
            serde_json::from_slice(&bytes).expect("deserialize calibration record");
        assert!(loaded.timings.is_valid());

        let policy = policy_from_timings(&loaded.timings);
        let seed = GpuDispatchPolicy::default();
        assert_eq!(
            policy.gemm_min_flops as u128,
            GpuDispatchPolicy::MIN_CALIBRATABLE_GEMM_FLOPS
        );
        assert_eq!(
            policy.potrf_min_p,
            GpuDispatchPolicy::MIN_CALIBRATABLE_POTRF_P
        );
        assert_eq!(
            policy.xtwx_flops_min,
            2 * XTWX_DIMS[0].0 * XTWX_DIMS[0].1 * XTWX_DIMS[0].1
        );
        assert_eq!(
            GpuDispatchPolicy {
                xtwx_flops_min: seed.xtwx_flops_min,
                gemm_min_flops: seed.gemm_min_flops,
                potrf_min_p: seed.potrf_min_p,
                ..policy
            },
            seed
        );
    }

    #[test]
    fn cached_timings_reject_non_positive_or_non_finite_durations() {
        assert!(uniform_timings(1.0, 0.5).is_valid());
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(!uniform_timings(bad, 0.5).is_valid());
            let mut timings = uniform_timings(1.0, 0.5);
            timings.xtwx[2].gpu_seconds = bad;
            assert!(!timings.is_valid());
        }
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
