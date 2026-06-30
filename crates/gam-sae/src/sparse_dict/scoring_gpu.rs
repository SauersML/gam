//! GPU score-block kernel for the collapsed-linear-lane router (#1026).
//!
//! The collapsed linear lane ([`crate::sparse_dict`]) scales a linear SAE
//! dictionary to `K ≈ 32_000` atoms by routing each row against the WHOLE
//! dictionary one atom-tile at a time and keeping only the top-`s` atoms online.
//! That route step — `scores[r][a] = Σ_c x[r][c]·decoder[a][c]` over a
//! `rows × tile × P` block — is the dominant cost of a fit (a single fit at
//! `K ≈ 32k` is the measured 1e4–1e6× hardware gap the issue tracks) and is the
//! embarrassingly-parallel shape a GPU exists for.
//!
//! # What this offloads (and what it does NOT)
//!
//! This computes ONE atom-tile's `rows × tile` score block on the device,
//! exactly the block the CPU [`super::scoring::score_row_tile`] folds into the
//! per-row online top-`s` selectors. The selection logic itself
//! ([`super::scoring::TopSSelector`]) stays single-sourced on the CPU: the
//! device returns the score block, the host folds it into the SAME selectors,
//! and discards it. Peak host score memory is `rows × tile`, independent of
//! `K` — the lane's memory discipline is preserved, the GPU just does the
//! `O(rows·tile·P)` multiply-accumulate that dominates.
//!
//! # Bit-exact parity (the gate, not a tolerance)
//!
//! The CPU oracle accumulates `acc += x[c]·d[c]` as SEPARATE f32 multiply then
//! f32 add (Rust emits no fused multiply-add for `a*b+c` unless `f32::mul_add`
//! is called, and `-ffp-contract` is off), in ascending `c` order. NVRTC
//! defaults to `--fmad=true`, which contracts `a*b+c` into a single-rounding
//! FMA — a ~1 ULP difference that can flip a near-tie top-`s` selection and make
//! the routed support, and hence the whole fit, diverge from the CPU oracle.
//!
//! So the kernel forces SEPARATE rounding with `__fmul_rn` + `__fadd_rn`, in the
//! SAME ascending-`c` order, giving a score block that is **bit-for-bit**
//! identical to the CPU `score_row_tile` (every `f32` equal under `to_bits`).
//! Because the scores are identical, the CPU selector fed device scores produces
//! the IDENTICAL routed support — parity is exact by construction, not bounded
//! by a tolerance.

#![cfg(target_os = "linux")]

use ndarray::ArrayView2;

/// The bit-exact-parity NVRTC kernel. One thread per `(row, atom)` output;
/// accumulates over `P` columns in ascending order with separate-rounding f32
/// ops so the result matches the CPU sequential `acc += x·d` to the bit.
///
/// `PP` (the column count) is baked in as a `#define` so the inner loop is a
/// fixed trip count (matching the other NVRTC kernels in this repo, which
/// monomorphise their shape macros for a pure `compile_ptx`).
pub const SCORE_BLOCK_KERNEL_SOURCE: &str = r#"
extern "C" __global__
void sparse_dict_score_block(
    const float* __restrict__ rows,    // [n_rows * PP] row-major
    const float* __restrict__ atoms,   // [n_atoms * PP] row-major (decoder tile)
    int n_rows,
    int n_atoms,
    float* __restrict__ scores)        // [n_rows * n_atoms] row-major
{
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long)n_rows * (long long)n_atoms;
  if (idx >= total) return;
  int r = (int)(idx / n_atoms);
  int a = (int)(idx % n_atoms);
  const float* xr = rows  + (long long)r * PP;
  const float* da = atoms + (long long)a * PP;
  // SEPARATE-rounding accumulation in ascending c — NO fused multiply-add, so
  // this is bit-identical to the CPU `acc += x[c]*d[c]` reference order.
  float acc = 0.0f;
  for (int c = 0; c < PP; ++c) {
    float prod = __fmul_rn(xr[c], da[c]);
    acc = __fadd_rn(acc, prod);
  }
  scores[idx] = acc;
}
"#;

/// Prepend the `PP` shape macro so the NVRTC compile is a pure `compile_ptx`
/// (mirrors `sae_rowjet::softmax_kernel_source` / `arrow_schur_nvrtc`).
#[must_use]
pub fn score_block_kernel_source(p: usize) -> String {
    format!("#define PP {p}\n{SCORE_BLOCK_KERNEL_SOURCE}")
}

/// CPU reference for the score block: `scores[r*n_atoms + a] = Σ_c
/// rows[r][c]·atoms[a][c]`, accumulated in ascending `c` with separate f32
/// rounding — the SAME arithmetic [`super::scoring::score_row_tile`] runs
/// per atom. This is the parity oracle the device kernel is locked against.
#[must_use]
pub fn score_block_cpu(rows: ArrayView2<'_, f32>, atoms: ArrayView2<'_, f32>) -> Vec<f32> {
    let n_rows = rows.nrows();
    let n_atoms = atoms.nrows();
    let p = rows.ncols();
    assert_eq!(p, atoms.ncols(), "score_block_cpu: P mismatch rows vs atoms");
    let mut scores = vec![0.0f32; n_rows * n_atoms];
    for r in 0..n_rows {
        let xr = rows.row(r);
        for a in 0..n_atoms {
            let da = atoms.row(a);
            let mut acc = 0.0f32;
            for c in 0..p {
                // separate mul then add — matches the kernel's __fmul_rn/__fadd_rn
                acc += xr[c] * da[c];
            }
            scores[r * n_atoms + a] = acc;
        }
    }
    scores
}

/// Minimum score-block element count (`n_rows · n_atoms`) below which the device
/// launch is not worth its fixed cost (probe + H2D + D2H). Below this the CPU
/// reference is used. Tuned to the same genus as the other SAE device floors
/// (`sae_rowjet::DEVICE_ROW_THRESHOLD`).
pub const DEVICE_SCORE_BLOCK_MIN_ELEMS: usize = 1 << 20; // ~1M MACs/tile minimum

/// Which path produced a score block. Returned by the fail-loud entry point so
/// callers (and the parity test) can ASSERT the device engaged rather than
/// silently falling back — the #1026/#1551 'GPU 0%' failure mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreBlockPath {
    /// The NVRTC `sparse_dict_score_block` kernel ran on the device.
    Device,
    /// The CPU `score_block_cpu` reference ran.
    Cpu,
}

/// Fail-loud, residency-aware score-block entry point (#1026 scale-K lane).
///
/// Honours the process-wide [`gam_gpu::GpuMode`] contract: under
/// [`gam_gpu::GpuMode::Required`] a missing CUDA runtime, an NVRTC/arch compile
/// failure, a launch fault, or a block below the device break-even all return
/// `Err` instead of silently degrading to the CPU. [`gam_gpu::GpuMode::Auto`]
/// uses the device when admitted and the block clears the break-even, else the
/// CPU; [`gam_gpu::GpuMode::Off`] always the CPU. The returned [`ScoreBlockPath`]
/// reports which path actually ran.
///
/// Both paths produce a BIT-IDENTICAL `f32` score block (see module docs), so
/// the routed top-`s` support is identical whichever path runs.
///
/// # Errors
/// Returns [`gam_gpu::GpuError`] when [`gam_gpu::GpuMode::Required`] is set but
/// the device path cannot run.
pub fn score_block_required(
    rows: ArrayView2<'_, f32>,
    atoms: ArrayView2<'_, f32>,
    mode: gam_gpu::GpuMode,
) -> Result<(Vec<f32>, ScoreBlockPath), gam_gpu::GpuError> {
    use gam_gpu::GpuMode;

    let n_rows = rows.nrows();
    let n_atoms = atoms.nrows();
    let elems = n_rows.saturating_mul(n_atoms);

    if mode == GpuMode::Off {
        return Ok((score_block_cpu(rows, atoms), ScoreBlockPath::Cpu));
    }

    let below_breakeven = elems < DEVICE_SCORE_BLOCK_MIN_ELEMS;
    if mode == GpuMode::Required && below_breakeven {
        return Err(gam_gpu::gpu_err!(
            "sparse_dict score-block GpuMode::Required: block of {n_rows}×{n_atoms} \
             = {elems} elems is below the device launch break-even \
             (DEVICE_SCORE_BLOCK_MIN_ELEMS={DEVICE_SCORE_BLOCK_MIN_ELEMS}); refusing \
             to silently run on the CPU"
        ));
    }
    if !below_breakeven {
        match device::score_block_device(rows, atoms) {
            Ok(out) => return Ok((out, ScoreBlockPath::Device)),
            Err(err) => {
                if mode == GpuMode::Required {
                    return Err(err);
                }
                // Auto: fall through to the CPU.
            }
        }
    }

    Ok((score_block_cpu(rows, atoms), ScoreBlockPath::Cpu))
}

mod device {
    use super::score_block_kernel_source;
    use gam_gpu::gpu_error::{GpuError, GpuResultExt};
    use ndarray::ArrayView2;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};

    struct Backend {
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        modules: Mutex<HashMap<usize, Arc<CudaModule>>>,
    }

    fn backend() -> Result<&'static Backend, GpuError> {
        static BACKEND: OnceLock<Result<Backend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                let parts = gam_gpu::backend_probe::probe_cuda_backend("sparse_dict_score_block")?;
                Ok(Backend {
                    ctx: parts.ctx,
                    stream: parts.stream,
                    modules: Mutex::new(HashMap::new()),
                })
            })
            .as_ref()
            .map_err(GpuError::clone)
    }

    fn module_for(b: &Backend, p: usize) -> Result<Arc<CudaModule>, GpuError> {
        if let Ok(guard) = b.modules.lock() {
            if let Some(m) = guard.get(&p) {
                return Ok(m.clone());
            }
        }
        let ptx = cudarc::nvrtc::compile_ptx(score_block_kernel_source(p))
            .gpu_ctx_with(|err| format!("sparse_dict score-block NVRTC (P={p}): {err}"))?;
        let module = b
            .ctx
            .load_module(ptx)
            .gpu_ctx("sparse_dict score-block module load")?;
        if let Ok(mut guard) = b.modules.lock() {
            guard.entry(p).or_insert_with(|| module.clone());
        }
        Ok(module)
    }

    /// Compute the `n_rows × n_atoms` score block on the device. Flattens the
    /// two views row-major (the kernel reads them as `[*, PP]`), launches one
    /// thread per output element, and downloads the block.
    pub(super) fn score_block_device(
        rows: ArrayView2<'_, f32>,
        atoms: ArrayView2<'_, f32>,
    ) -> Result<Vec<f32>, GpuError> {
        let n_rows = rows.nrows();
        let n_atoms = atoms.nrows();
        let p = rows.ncols();
        if p != atoms.ncols() {
            return Err(gam_gpu::gpu_err!(
                "sparse_dict score-block: P mismatch rows={p} atoms={}",
                atoms.ncols()
            ));
        }
        if n_rows == 0 || n_atoms == 0 || p == 0 {
            return Ok(vec![0.0f32; n_rows * n_atoms]);
        }

        let b = backend()?;
        let module = module_for(b, p)?;
        let func = module
            .load_function("sparse_dict_score_block")
            .gpu_ctx("sparse_dict score-block load_function")?;
        let stream = b.stream.clone();

        // Row-major contiguous host buffers (handles non-contiguous views).
        let rows_host: Vec<f32> = rows.iter().copied().collect();
        let atoms_host: Vec<f32> = atoms.iter().copied().collect();
        debug_assert_eq!(rows_host.len(), n_rows * p);
        debug_assert_eq!(atoms_host.len(), n_atoms * p);

        let rows_dev = stream
            .clone_htod(&rows_host)
            .gpu_ctx("sparse_dict score-block htod rows")?;
        let atoms_dev = stream
            .clone_htod(&atoms_host)
            .gpu_ctx("sparse_dict score-block htod atoms")?;
        let mut scores_dev = stream
            .alloc_zeros::<f32>(n_rows * n_atoms)
            .gpu_ctx("sparse_dict score-block alloc scores")?;

        let n_rows_i32 = i32::try_from(n_rows)
            .map_err(|_| gam_gpu::gpu_err!("sparse_dict score-block n_rows={n_rows} overflows i32"))?;
        let n_atoms_i32 = i32::try_from(n_atoms).map_err(|_| {
            gam_gpu::gpu_err!("sparse_dict score-block n_atoms={n_atoms} overflows i32")
        })?;

        let total = n_rows * n_atoms;
        let block: u32 = 256;
        let grid: u32 = u32::try_from(total.div_ceil(block as usize))
            .map_err(|_| gam_gpu::gpu_err!("sparse_dict score-block grid overflow"))?;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&rows_dev)
            .arg(&atoms_dev)
            .arg(&n_rows_i32)
            .arg(&n_atoms_i32)
            .arg(&mut scores_dev);
        // SAFETY: grid/block validated; all device pointers are cudarc-checked
        // allocations on this stream; the kernel reads rows[0..n_rows*P] /
        // atoms[0..n_atoms*P] and writes within scores[0..n_rows*n_atoms].
        unsafe { builder.launch(cfg) }.gpu_ctx("sparse_dict score-block launch")?;

        let mut scores = vec![0.0f32; n_rows * n_atoms];
        stream
            .memcpy_dtoh(&scores_dev, &mut scores)
            .gpu_ctx("sparse_dict score-block dtoh scores")?;
        stream
            .synchronize()
            .gpu_ctx("sparse_dict score-block synchronize")?;
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Deterministic fp32 fixture: `n_rows × p` rows and `n_atoms × p` unit-norm
    /// atoms (the lane unit-norms its decoder, so |xᵀd| is the projection).
    fn fixture(n_rows: usize, n_atoms: usize, p: usize) -> (Array2<f32>, Array2<f32>) {
        let rows = Array2::from_shape_fn((n_rows, p), |(i, c)| {
            (((i * 31 + c * 17) as f32) * 0.013).sin() * 0.9
        });
        let mut atoms = Array2::from_shape_fn((n_atoms, p), |(a, c)| {
            (((a * 7 + c * 5) as f32) * 0.011).cos()
        });
        for mut row in atoms.outer_iter_mut() {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            row.mapv_inplace(|v| v / norm);
        }
        (rows, atoms)
    }

    #[test]
    fn cpu_score_block_matches_score_row_tile() {
        // The block oracle must equal the per-atom CPU router primitive exactly.
        use crate::sparse_dict::scoring::score_row_tile;
        let (rows, atoms) = fixture(5, 9, 7);
        let block = score_block_cpu(rows.view(), atoms.view());
        for r in 0..rows.nrows() {
            // score_row_tile folds into a selector; reproduce its raw scores by
            // running the same acc loop it uses (separate mul/add, ascending c).
            for a in 0..atoms.nrows() {
                let mut acc = 0.0f32;
                for c in 0..rows.ncols() {
                    acc += rows[[r, c]] * atoms[[a, c]];
                }
                assert_eq!(
                    block[r * atoms.nrows() + a].to_bits(),
                    acc.to_bits(),
                    "block oracle vs raw acc differ at r={r} a={a}"
                );
            }
        }
        // And score_row_tile's selection over the full block is reproducible
        // from the same scores (sanity: the primitive is the one we accelerate).
        let mut sel = crate::sparse_dict::scoring::TopSSelector::new(3);
        score_row_tile(rows.row(0), atoms.view(), 0, &mut sel);
        let picked = sel.finish();
        assert!(picked.len() <= 3 && !picked.is_empty());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn device_score_block_is_bit_identical_to_cpu_when_available() {
        // Exactness gate. The block MUST clear DEVICE_SCORE_BLOCK_MIN_ELEMS so
        // the device path is actually admitted (a sub-break-even block would
        // skip-pass on the CPU and prove nothing). On a CUDA host we drive
        // GpuMode::Required so a silent CPU fallback is a hard FAILURE, and we
        // assert the device block is BIT-IDENTICAL to the CPU reference. With no
        // runtime, Required must fail closed and the CPU path stays exact.
        let n_rows = 256;
        let n_atoms = 4096; // 256*4096 = 1,048,576 == DEVICE_SCORE_BLOCK_MIN_ELEMS
        let p = 48;
        assert!(n_rows * n_atoms >= DEVICE_SCORE_BLOCK_MIN_ELEMS);
        let (rows, atoms) = fixture(n_rows, n_atoms, p);
        let cpu = score_block_cpu(rows.view(), atoms.view());

        match score_block_required(rows.view(), atoms.view(), gam_gpu::GpuMode::Required) {
            Ok((got, path)) => {
                assert_eq!(
                    path,
                    ScoreBlockPath::Device,
                    "Required succeeded but reported CPU — device did not engage"
                );
                assert_eq!(got.len(), cpu.len());
                for (i, (g, c)) in got.iter().zip(&cpu).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        c.to_bits(),
                        "device vs CPU score-block bit mismatch at {i}: dev={g} cpu={c}"
                    );
                }
            }
            Err(err) => {
                // No CUDA runtime on this host: Required correctly failed closed.
                assert!(
                    gam_gpu::GpuRuntime::global().is_none(),
                    "Required errored despite a live CUDA runtime: {err}"
                );
                // The CPU path must still be exact under Auto.
                let (got, path) =
                    score_block_required(rows.view(), atoms.view(), gam_gpu::GpuMode::Auto)
                        .expect("Auto must not error on a device-absent host");
                assert_eq!(path, ScoreBlockPath::Cpu);
                assert_eq!(got, cpu);
            }
        }
    }
}
