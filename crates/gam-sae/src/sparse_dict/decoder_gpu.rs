//! Device-resident block-CG backend for the giant-component decoder refresh
//! (#1017).
//!
//! The sparse-dictionary decoder refresh solves ONE co-firing normal-equation
//! operator against every decoder column. [`super::update`] runs that solve
//! through the shared multi-RHS recurrence `gam_linalg::pcg::pcg_multi_core`;
//! this module supplies the CUDA implementation of its block backend: the CSR
//! operator, the right-hand-side block, and ALL CG iterate state (`X`, `R`,
//! `P`, `AP`) are uploaded once and stay resident on the device for the whole
//! solve. Per iteration the host exchanges only the per-column scalars the
//! recurrence itself needs (`alpha`/`beta` up, the two column-dot vectors
//! down) — never a block. The solution block is downloaded once at the end.
//!
//! # Bit parity with the CPU backend (a gate, not a tolerance)
//!
//! The recurrence's scalar decisions live in `pcg_multi_core` and are shared
//! verbatim with the CPU path, so parity reduces to the four block primitives.
//! Each is implemented with EXACTLY the CPU backend's per-column arithmetic:
//!
//! * the CSR application accumulates `diag·x` first, then the stored
//!   neighbors in ascending CSR order, with SEPARATE `__dmul_rn`/`__dadd_rn`
//!   roundings (NVRTC's default `fmad` contraction would fuse `a·b + c` into
//!   a single-rounding FMA and drift ~1 ulp per term — Rust emits no FMA for
//!   `a * b + c`, so the kernel must not either);
//! * the per-column inner products are strict ascending-row folds in ONE
//!   thread per column (adjacent threads read adjacent addresses at each row,
//!   so the walk is coalesced despite being sequential per column);
//! * the `X`/`R` and `P` updates perform the same multiply-then-add per
//!   element, gated by the same per-column active mask.
//!
//! The `device_block_cg_matches_cpu_bitwise` test pins `to_bits` equality of
//! the full solve against the CPU backend on a giant-scale fixture, so a CUDA
//! box and a CPU box produce the SAME fit, bit for bit.
//!
//! # Policy
//!
//! `Off` never builds the backend. `Auto` builds it when a CUDA device is
//! available AND the dense block state (`rows × columns`) clears the
//! dictionary-lane device break-even [`gam_gpu::DEFAULT_DICTIONARY_SCORE_MIN_ELEMS`]
//! (below it the staging + launch tax outruns the traversal amortization the
//! device exists to provide); an `Auto` decline is an ordinary fall-back to
//! the CPU backend. `Required` resolves the device unconditionally (no size
//! gate — the caller demanded residency) and surfaces absence as a typed
//! error. After admission, ANY device fault panics loudly (#1551 discipline:
//! a post-admission failure must never be laundered into a silent CPU retry).

#![cfg(target_os = "linux")]

use gam_linalg::pcg::PcgBlockBackend;
use ndarray::Array2;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use gam_gpu::gpu_error::{GpuError, GpuResultExt};

/// Threads per block over the column (fast) dimension for every kernel here.
/// Purely a launch-geometry choice: no kernel's arithmetic order depends on
/// it, so it can never change a result bit.
const COLUMN_BLOCK_THREADS: u32 = 128;

/// CUDA `gridDim.y` hard limit; rows beyond it are covered by the kernels'
/// row-stride loops.
const MAX_GRID_Y: usize = 65_535;

/// The four block primitives, in one NVRTC module. All arithmetic is f64 with
/// separate roundings (`__dmul_rn`/`__dadd_rn`); see the module docs for why
/// contraction must be suppressed.
const BLOCK_CG_KERNELS: &str = r#"
extern "C" __global__ void sae_decoder_cg_spmm(
    const double* __restrict__ diag,
    const unsigned int* __restrict__ row_ptr,
    const unsigned int* __restrict__ cols,
    const double* __restrict__ vals,
    const double* __restrict__ p_blk,
    double* __restrict__ ap_blk,
    int m,
    int t)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= t) return;
    for (int i = blockIdx.y; i < m; i += gridDim.y) {
        double acc = __dmul_rn(diag[i], p_blk[(size_t)i * t + c]);
        unsigned int e_end = row_ptr[i + 1];
        for (unsigned int e = row_ptr[i]; e < e_end; ++e) {
            acc = __dadd_rn(acc, __dmul_rn(vals[e], p_blk[(size_t)cols[e] * t + c]));
        }
        ap_blk[(size_t)i * t + c] = acc;
    }
}

extern "C" __global__ void sae_decoder_cg_dot(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ out,
    int m,
    int t)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= t) return;
    double acc = 0.0;
    for (int i = 0; i < m; ++i) {
        acc = __dadd_rn(acc, __dmul_rn(a[(size_t)i * t + c], b[(size_t)i * t + c]));
    }
    out[c] = acc;
}

extern "C" __global__ void sae_decoder_cg_update_xr(
    double* __restrict__ x_blk,
    double* __restrict__ r_blk,
    const double* __restrict__ p_blk,
    const double* __restrict__ ap_blk,
    const double* __restrict__ alpha,
    const unsigned int* __restrict__ active,
    int m,
    int t)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= t || active[c] == 0u) return;
    double al = alpha[c];
    double nal = -al;
    for (int i = blockIdx.y; i < m; i += gridDim.y) {
        size_t idx = (size_t)i * t + c;
        x_blk[idx] = __dadd_rn(x_blk[idx], __dmul_rn(al, p_blk[idx]));
        r_blk[idx] = __dadd_rn(r_blk[idx], __dmul_rn(nal, ap_blk[idx]));
    }
}

extern "C" __global__ void sae_decoder_cg_update_p(
    double* __restrict__ p_blk,
    const double* __restrict__ r_blk,
    const double* __restrict__ beta,
    const unsigned int* __restrict__ active,
    int m,
    int t)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= t || active[c] == 0u) return;
    double be = beta[c];
    for (int i = blockIdx.y; i < m; i += gridDim.y) {
        size_t idx = (size_t)i * t + c;
        p_blk[idx] = __dadd_rn(r_blk[idx], __dmul_rn(be, p_blk[idx]));
    }
}
"#;

struct Backend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Mutex<Option<Arc<CudaModule>>>,
}

fn backend() -> Result<&'static Backend, GpuError> {
    static BACKEND: OnceLock<Result<Backend, GpuError>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
            let parts = gam_gpu::backend_probe::probe_cuda_backend("sparse_dict_decoder_cg")?;
            Ok(Backend {
                ctx: parts.ctx,
                stream: parts.stream,
                module: Mutex::new(None),
            })
        })
        .as_ref()
        .map_err(GpuError::clone)
}

fn module_for(b: &Backend) -> Result<Arc<CudaModule>, GpuError> {
    if let Ok(guard) = b.module.lock() {
        if let Some(m) = guard.as_ref() {
            return Ok(m.clone());
        }
    }
    let ptx = gam_gpu::device_cache::compile_ptx_arch(BLOCK_CG_KERNELS.to_string())
        .gpu_ctx_with(|err| format!("sparse_dict decoder block-CG NVRTC: {err}"))?;
    let module = b
        .ctx
        .load_module(ptx)
        .gpu_ctx("sparse_dict decoder block-CG module load")?;
    if let Ok(mut guard) = b.module.lock() {
        guard.get_or_insert_with(|| module.clone());
    }
    Ok(module)
}

/// A post-admission device fault is a fault, never an Auto decline: the
/// backend was already admitted, so a `None`-shaped continuation would
/// silently re-run the refresh on the CPU and misreport the residency the
/// caller was promised.
#[track_caller]
fn complete<T>(operation: &str, result: Result<T, GpuError>) -> T {
    match result {
        Ok(value) => value,
        // SAFETY: policy admission already committed this solve to the device;
        // this hook has no error channel that would not be read as an ordinary
        // pre-admission decline, and returning would silently re-run the
        // refresh on the CPU while misreporting device residency.
        Err(err) => panic!("sparse_dict decoder block-CG '{operation}' device fault: {err}"),
    }
}

/// Device-resident implementation of [`PcgBlockBackend`] for one column tile.
pub(super) struct DeviceBlockCgBackend {
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    m: usize,
    t: usize,
    diag: CudaSlice<f64>,
    row_ptr: CudaSlice<u32>,
    cols: CudaSlice<u32>,
    vals: CudaSlice<f64>,
    x: CudaSlice<f64>,
    r: CudaSlice<f64>,
    p: CudaSlice<f64>,
    ap: CudaSlice<f64>,
    dot_out: CudaSlice<f64>,
    scalars: CudaSlice<f64>,
    active: CudaSlice<u32>,
    dot_host: Vec<f64>,
    active_host: Vec<u32>,
}

impl DeviceBlockCgBackend {
    /// Build the resident backend when platform, policy, and workload admit
    /// it. `Ok(None)` is an ordinary Auto decline (absent device or a block
    /// below the break-even); `Err` is a Required-policy failure. The CG
    /// entry state is uploaded here: `X = 0`, `R = P = rhs_block`.
    pub(super) fn try_new(
        gpu: gam_gpu::GpuPolicy,
        row_ptr: &[u32],
        csr_cols: &[u32],
        csr_vals: &[f64],
        diag_ridge: &[f64],
        rhs_block: &Array2<f64>,
    ) -> Result<Option<Self>, String> {
        let (m, t) = rhs_block.dim();
        match gpu {
            gam_gpu::GpuPolicy::Off => return Ok(None),
            gam_gpu::GpuPolicy::Auto => {
                if m * t < gam_gpu::DEFAULT_DICTIONARY_SCORE_MIN_ELEMS {
                    return Ok(None);
                }
                match gam_gpu::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto) {
                    Ok(Some(_)) => {}
                    Ok(None) => return Ok(None),
                    Err(err) => {
                        return Err(format!(
                            "sparse_dict decoder block-CG availability probe failed: {err}"
                        ));
                    }
                }
            }
            gam_gpu::GpuPolicy::Required => {
                gam_gpu::GpuRuntime::require().map_err(|err| {
                    format!("sparse_dict decoder block-CG gpu=required: {err}")
                })?;
            }
        }

        let b = backend().map_err(|err| {
            format!("sparse_dict decoder block-CG backend probe failed: {err}")
        })?;
        let module = module_for(b).map_err(|err| {
            format!("sparse_dict decoder block-CG module build failed: {err}")
        })?;
        let stream = b.stream.clone();

        let rhs_host = rhs_block
            .as_slice()
            .expect("decoder block-CG rhs block is standard layout");
        let upload = || -> Result<Self, GpuError> {
            let diag = stream.clone_htod(diag_ridge).gpu_ctx("htod diag")?;
            let row_ptr = stream.clone_htod(row_ptr).gpu_ctx("htod row_ptr")?;
            let cols = stream.clone_htod(csr_cols).gpu_ctx("htod cols")?;
            let vals = stream.clone_htod(csr_vals).gpu_ctx("htod vals")?;
            let x = stream.alloc_zeros::<f64>(m * t).gpu_ctx("alloc x")?;
            let r = stream.clone_htod(rhs_host).gpu_ctx("htod r")?;
            let p = stream.clone_htod(rhs_host).gpu_ctx("htod p")?;
            let ap = stream.alloc_zeros::<f64>(m * t).gpu_ctx("alloc ap")?;
            let dot_out = stream.alloc_zeros::<f64>(t).gpu_ctx("alloc dot_out")?;
            let scalars = stream.alloc_zeros::<f64>(t).gpu_ctx("alloc scalars")?;
            let active = stream.alloc_zeros::<u32>(t).gpu_ctx("alloc active")?;
            Ok(Self {
                stream: stream.clone(),
                module,
                m,
                t,
                diag,
                row_ptr,
                cols,
                vals,
                x,
                r,
                p,
                ap,
                dot_out,
                scalars,
                active,
                dot_host: vec![0.0; t],
                active_host: vec![0; t],
            })
        };
        upload()
            .map(Some)
            .map_err(|err| format!("sparse_dict decoder block-CG operand upload failed: {err}"))
    }

    /// Download the solution block once, at the end of the solve.
    pub(super) fn take_solution(self) -> Result<Array2<f64>, String> {
        let mut host = vec![0.0f64; self.m * self.t];
        self.stream
            .memcpy_dtoh(&self.x, &mut host)
            .gpu_ctx("sparse_dict decoder block-CG dtoh solution")
            .and_then(|_| {
                self.stream
                    .synchronize()
                    .gpu_ctx("sparse_dict decoder block-CG solution synchronize")
            })
            .map_err(|err| format!("sparse_dict decoder block-CG solution download failed: {err}"))?;
        Array2::from_shape_vec((self.m, self.t), host)
            .map_err(|err| format!("sparse_dict decoder block-CG solution shape: {err}"))
    }

    fn launch_grid(&self, rows_span: bool) -> LaunchConfig {
        let grid_x = u32::try_from(self.t.div_ceil(COLUMN_BLOCK_THREADS as usize))
            .expect("decoder block-CG column grid overflows u32");
        let grid_y = if rows_span {
            u32::try_from(self.m.min(MAX_GRID_Y).max(1))
                .expect("decoder block-CG row grid overflows u32")
        } else {
            1
        };
        LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (COLUMN_BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn dims_i32(&self) -> (i32, i32) {
        (
            i32::try_from(self.m).expect("decoder block-CG rows overflow i32"),
            i32::try_from(self.t).expect("decoder block-CG columns overflow i32"),
        )
    }

    fn run_dot(&mut self, which: DotOperands, out: &mut [f64]) {
        let func = complete(
            "dot load_function",
            self.module
                .load_function("sae_decoder_cg_dot")
                .gpu_ctx("load sae_decoder_cg_dot"),
        );
        let (m_i32, t_i32) = self.dims_i32();
        let cfg = self.launch_grid(false);
        {
            let mut builder = self.stream.launch_builder(&func);
            match which {
                DotOperands::PAp => builder.arg(&self.p).arg(&self.ap),
                DotOperands::RR => builder.arg(&self.r).arg(&self.r),
            }
            .arg(&mut self.dot_out)
            .arg(&m_i32)
            .arg(&t_i32);
            // SAFETY: geometry covers exactly `t` columns; the kernel reads the
            // two `m*t` blocks and writes only `dot_out[0..t]`, all live
            // allocations on this stream.
            complete("dot launch", unsafe { builder.launch(cfg) }.gpu_ctx("launch dot"));
        }
        complete(
            "dot download",
            self.stream
                .memcpy_dtoh(&self.dot_out, &mut self.dot_host)
                .gpu_ctx("dtoh dot_out")
                .and_then(|_| self.stream.synchronize().gpu_ctx("dot synchronize")),
        );
        out.copy_from_slice(&self.dot_host);
    }

    fn upload_scalars(&mut self, scalars: &[f64], active: &[bool]) {
        for (slot, &flag) in self.active_host.iter_mut().zip(active.iter()) {
            *slot = u32::from(flag);
        }
        complete(
            "scalar upload",
            self.stream
                .memcpy_htod(scalars, &mut self.scalars)
                .gpu_ctx("htod scalars")
                .and_then(|_| {
                    self.stream
                        .memcpy_htod(&self.active_host, &mut self.active)
                        .gpu_ctx("htod active")
                }),
        );
    }
}

enum DotOperands {
    PAp,
    RR,
}

impl PcgBlockBackend for DeviceBlockCgBackend {
    fn rows(&self) -> usize {
        self.m
    }

    fn columns(&self) -> usize {
        self.t
    }

    fn apply_block(&mut self) {
        let func = complete(
            "spmm load_function",
            self.module
                .load_function("sae_decoder_cg_spmm")
                .gpu_ctx("load sae_decoder_cg_spmm"),
        );
        let (m_i32, t_i32) = self.dims_i32();
        let cfg = self.launch_grid(true);
        let mut builder = self.stream.launch_builder(&func);
        builder
            .arg(&self.diag)
            .arg(&self.row_ptr)
            .arg(&self.cols)
            .arg(&self.vals)
            .arg(&self.p)
            .arg(&mut self.ap)
            .arg(&m_i32)
            .arg(&t_i32);
        // SAFETY: the kernel reads diag[0..m], the CSR arrays within
        // row_ptr[m] bounds, and p[0..m*t]; it writes only ap[0..m*t]. All are
        // live allocations on this stream and the grid covers every (row,
        // column) exactly once via the row-stride loop.
        complete("spmm launch", unsafe { builder.launch(cfg) }.gpu_ctx("launch spmm"));
    }

    fn dot_p_ap(&mut self, out: &mut [f64]) {
        self.run_dot(DotOperands::PAp, out);
    }

    fn dot_r_r(&mut self, out: &mut [f64]) {
        self.run_dot(DotOperands::RR, out);
    }

    fn update_x_r(&mut self, alpha: &[f64], active: &[bool]) {
        self.upload_scalars(alpha, active);
        let func = complete(
            "update_xr load_function",
            self.module
                .load_function("sae_decoder_cg_update_xr")
                .gpu_ctx("load sae_decoder_cg_update_xr"),
        );
        let (m_i32, t_i32) = self.dims_i32();
        let cfg = self.launch_grid(true);
        let mut builder = self.stream.launch_builder(&func);
        builder
            .arg(&mut self.x)
            .arg(&mut self.r)
            .arg(&self.p)
            .arg(&self.ap)
            .arg(&self.scalars)
            .arg(&self.active)
            .arg(&m_i32)
            .arg(&t_i32);
        // SAFETY: reads p/ap/scalars/active within bounds, writes x/r within
        // m*t; masked columns are untouched, matching the CPU backend.
        complete(
            "update_xr launch",
            unsafe { builder.launch(cfg) }.gpu_ctx("launch update_xr"),
        );
    }

    fn update_p(&mut self, beta: &[f64], active: &[bool]) {
        self.upload_scalars(beta, active);
        let func = complete(
            "update_p load_function",
            self.module
                .load_function("sae_decoder_cg_update_p")
                .gpu_ctx("load sae_decoder_cg_update_p"),
        );
        let (m_i32, t_i32) = self.dims_i32();
        let cfg = self.launch_grid(true);
        let mut builder = self.stream.launch_builder(&func);
        builder
            .arg(&mut self.p)
            .arg(&self.r)
            .arg(&self.scalars)
            .arg(&self.active)
            .arg(&m_i32)
            .arg(&t_i32);
        // SAFETY: reads r/scalars/active within bounds, writes p within m*t;
        // masked columns are untouched, matching the CPU backend.
        complete(
            "update_p launch",
            unsafe { builder.launch(cfg) }.gpu_ctx("launch update_p"),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::pcg::{CpuPcgBlockBackend, pcg_multi_core};
    use rayon::prelude::*;

    fn cuda_available_for_test(label: &str) -> bool {
        match gam_gpu::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto) {
            Ok(Some(_)) => true,
            Ok(None) => {
                log::warn!("[{label}] no CUDA device; device parity not exercised here");
                false
            }
            Err(err) => panic!("[{label}] CUDA availability probe failed: {err}"),
        }
    }

    /// Deterministic sparse SPD giant-component fixture in CSR form: a ring
    /// plus pseudo-random chords, diagonally dominant, with a heterogeneous
    /// right-hand-side block (including a zero column, which the recurrence
    /// must freeze at zero on both backends).
    fn fixture(m: usize, t: usize, seed: u64) -> (Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>, Array2<f64>) {
        let mut state = seed.max(1);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) - 0.5
        };
        let mut neigh: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
        let link = |neigh: &mut Vec<Vec<(usize, f64)>>, i: usize, j: usize, v: f64| {
            neigh[i].push((j, v));
            neigh[j].push((i, v));
        };
        for i in 0..m {
            link(&mut neigh, i, (i + 1) % m, next());
            if i % 3 == 0 {
                link(&mut neigh, i, (i + m / 2 + 1) % m, next());
            }
        }
        for list in neigh.iter_mut() {
            list.sort_by_key(|&(j, _)| j);
        }
        let mut row_ptr = vec![0u32];
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        let mut diag = Vec::with_capacity(m);
        for (i, list) in neigh.iter().enumerate() {
            let mut row_abs = 0.0f64;
            for &(j, v) in list {
                cols.push(j as u32);
                vals.push(v);
                row_abs += v.abs();
            }
            row_ptr.push(cols.len() as u32);
            diag.push(row_abs + 0.75 + next().abs() + (i % 7) as f64 * 0.01);
        }
        let mut rhs = Array2::<f64>::zeros((m, t));
        for c in 0..t {
            if c == t / 2 {
                continue; // exact-zero column
            }
            let scale = 10f64.powi((c % 5) as i32 - 2);
            for i in 0..m {
                rhs[[i, c]] = scale * (((i * 13 + c * 7 + 3) as f64).sin());
            }
        }
        (row_ptr, cols, vals, diag, rhs)
    }

    fn cpu_solve(
        row_ptr: &[u32],
        cols: &[u32],
        vals: &[f64],
        diag: &[f64],
        rhs: &Array2<f64>,
        rel_tol: f64,
        cap: usize,
    ) -> (Vec<gam_linalg::pcg::PcgCoreResult>, Array2<f64>) {
        let apply = |pblk: &Array2<f64>, apblk: &mut Array2<f64>| {
            let t = pblk.ncols();
            let ps = pblk.as_slice().expect("standard layout");
            apblk
                .as_slice_mut()
                .expect("standard layout")
                .par_chunks_mut(t)
                .enumerate()
                .for_each(|(i, out_row)| {
                    let d = diag[i];
                    let base_i = i * t;
                    for (c, slot) in out_row.iter_mut().enumerate() {
                        *slot = d * ps[base_i + c];
                    }
                    for e in row_ptr[i] as usize..row_ptr[i + 1] as usize {
                        let v = vals[e];
                        let base_j = cols[e] as usize * t;
                        for (c, slot) in out_row.iter_mut().enumerate() {
                            *slot += v * ps[base_j + c];
                        }
                    }
                });
        };
        let mut backend = CpuPcgBlockBackend::new(rhs.clone(), apply);
        let results = pcg_multi_core(&mut backend, rel_tol, cap, true);
        (results, backend.into_solution())
    }

    fn device_solve(
        row_ptr: &[u32],
        cols: &[u32],
        vals: &[f64],
        diag: &[f64],
        rhs: &Array2<f64>,
        rel_tol: f64,
        cap: usize,
    ) -> (Vec<gam_linalg::pcg::PcgCoreResult>, Array2<f64>) {
        let mut backend = DeviceBlockCgBackend::try_new(
            gam_gpu::GpuPolicy::Required,
            row_ptr,
            cols,
            vals,
            diag,
            rhs,
        )
        .expect("device backend build")
        .expect("device backend admitted under Required");
        let results = pcg_multi_core(&mut backend, rel_tol, cap, true);
        let solution = backend.take_solution().expect("solution download");
        (results, solution)
    }

    /// The device-resident block CG must reproduce the CPU backend BIT-FOR-BIT:
    /// same per-column stop/iterations, same alpha/beta traces, same solution
    /// bits — and a second device run must reproduce itself exactly.
    #[test]
    fn device_block_cg_matches_cpu_bitwise() {
        if !cuda_available_for_test("decoder_gpu::device_block_cg_matches_cpu_bitwise") {
            return;
        }
        let (m, t) = (997, 33);
        let (row_ptr, cols, vals, diag, rhs) = fixture(m, t, 0x1017_2026);
        let rel_tol = f64::EPSILON.sqrt();
        let cap = m;

        let (cpu_results, cpu_solution) =
            cpu_solve(&row_ptr, &cols, &vals, &diag, &rhs, rel_tol, cap);
        let (dev_results, dev_solution) =
            device_solve(&row_ptr, &cols, &vals, &diag, &rhs, rel_tol, cap);
        let (dev2_results, dev2_solution) =
            device_solve(&row_ptr, &cols, &vals, &diag, &rhs, rel_tol, cap);

        let mut converged = 0usize;
        for c in 0..t {
            let cpu = &cpu_results[c];
            let dev = &dev_results[c];
            assert_eq!(cpu.stop, dev.stop, "column {c} stop");
            assert_eq!(cpu.iterations, dev.iterations, "column {c} iterations");
            assert_eq!(
                cpu.final_residual_norm.to_bits(),
                dev.final_residual_norm.to_bits(),
                "column {c} final residual"
            );
            if cpu.stop == gam_linalg::pcg::PcgStop::Converged && cpu.rhs_norm > 0.0 {
                converged += 1;
            }
            let dc = cpu.diagnostics.as_ref().expect("cpu diagnostics");
            let dd = dev.diagnostics.as_ref().expect("device diagnostics");
            assert_eq!(dc.alpha.len(), dd.alpha.len(), "column {c} alpha trace length");
            for (k, (a, b)) in dc.alpha.iter().zip(dd.alpha.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "column {c} alpha[{k}]");
            }
            for (k, (a, b)) in dc.beta.iter().zip(dd.beta.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "column {c} beta[{k}]");
            }
            assert_eq!(dev2_results[c].iterations, dev.iterations, "column {c} rerun");
        }
        assert!(
            converged >= t - 1,
            "fixture must exercise real convergence (got {converged}/{t})"
        );
        for i in 0..m {
            for c in 0..t {
                assert_eq!(
                    cpu_solution[[i, c]].to_bits(),
                    dev_solution[[i, c]].to_bits(),
                    "solution [{i},{c}]"
                );
                assert_eq!(
                    dev2_solution[[i, c]].to_bits(),
                    dev_solution[[i, c]].to_bits(),
                    "rerun solution [{i},{c}]"
                );
            }
        }
    }
}
