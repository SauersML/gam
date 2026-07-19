//! GPU **block-gate (curved) scoring** for the block-sparse lane
//! ([`crate::sparse_dict::block`]).
//!
//! The block-sparse lane groups the `K` atoms into `G` blocks of `b` rows each
//! (`K = G·b`, `b` small — typically 2–4) and routes whole blocks by their group
//! ℓ₂ **gate** `gate_g = ‖x D_gᵀ‖₂`. The hot loop
//! ([`super::block::route_block_minibatch`]) is exactly the atom lane's score
//! GEMM followed by a per-block ℓ₂ reduction: one `(minibatch × P)·(P × b·G)`
//! matrix multiply produces the raw projections `z` (`minibatch × b·G`), then
//! each adjacent `b`-group of `z` columns is reduced to its ℓ₂ norm to give the
//! `minibatch × G` gate block, whose per-row top-`k` blocks route.
//!
//! This module offloads that curved score to the device by REUSING the atom
//! lane's bit-exact score-GEMM ([`super::scoring_gpu::SCORE_BLOCK_KERNEL_SOURCE`]:
//! `sparse_dict_score_block_offset`) to form `z` a block-tile at a time, then a
//! fused epilogue kernel (`sparse_dict_block_gate`, below) reduces each adjacent
//! `b`-group to `gate_g`, then the atom lane's resident top-`s` fold
//! (`sparse_dict_fold_top_s`) selects the top-`k` blocks online. The whole `m × G`
//! gate stream is never downloaded: only the final `(block, gate)` shortlists
//! (`m × k`) cross PCIe — the same shortlist-only discipline the atom lane keeps.
//!
//! # Precision of the gate (why f32 on the device is sufficient)
//!
//! The gate is `gate_g = sqrt(Σ_{r<b} z_{g,r}²)` over only `b ∈ {2,3,4}` terms.
//! The device forms each `z_{g,r}` bit-identically to the CPU reference (the
//! score GEMM forbids FMA contraction and accumulates in ascending `c` with
//! separate-rounding `__fmul_rn`/`__fadd_rn`), and reduces the `b`-group with the
//! same separate-rounding f32 ops plus an IEEE round-to-nearest `sqrtf` (which
//! matches Rust's `f32::sqrt`). So the device gate equals the CPU gate to the
//! bit, and the online fold uses the identical `(gate desc, block asc)` order as
//! [`super::block::route_row_blocks`] — the routed block support is IDENTICAL to
//! the CPU oracle by construction.
//!
//! Even without that bit-for-bit coincidence f32 would suffice for SELECTION:
//! the accumulation error of a `b ≤ 4`-term sum of squares is a few ULP of the
//! largest term, whereas a routing decision only changes when two DISTINCT
//! blocks' gates fall within that few-ULP window — and two blocks whose subspace
//! energies agree to f32 rounding contribute interchangeably to the
//! reconstruction, with the tie broken deterministically by ascending block
//! index. f64 on the device would move no selection boundary that a downstream
//! consumer can observe. The selection-level equivalence to the CPU path is the
//! contract (SPEC 20); the bit-identity is a bonus the shared arithmetic gives.

use ndarray::{ArrayView1, ArrayView2};

use super::block::{block_gates, block_projections_row, route_row_blocks};

/// Which path produced a block route. Returned by the fail-loud entry point so
/// callers (and the parity test) can ASSERT the device engaged rather than
/// silently falling back — the #1026/#1551 'GPU 0%' failure mode. Mirrors
/// [`super::scoring::ScoreRoutePath`] but is defined here so the CPU-side
/// dispatch compiles on non-CUDA hosts too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockRoutePath {
    /// The CUDA block-gate router (score GEMM + gate epilogue + top-`k` fold) ran.
    Device,
    /// The CPU block-gate reference ran.
    Cpu,
}

/// The fused ℓ₂-gate epilogue kernel. Given the raw projection block
/// `z` (`n_rows × (n_blocks·b)` row-major, block `g` occupying columns
/// `[g·b, g·b+b)`), one thread per `(row, block)` output reduces that block's `b`
/// adjacent `z` columns to `gate = sqrt(Σ_r z_r²)`, writing the `n_rows ×
/// n_blocks` gate block. Separate-rounding f32 ops + IEEE `sqrtf` match the CPU
/// reference ([`block_gate_block_cpu`]) to the bit, so the downstream fold
/// selects the identical block support.
///
/// `b` is a runtime argument (blocks are 2–4 rows and the width varies per fit),
/// so unlike the score GEMM's `PP` this kernel is not monomorphised on it.
#[cfg(target_os = "linux")]
pub const BLOCK_GATE_KERNEL_SOURCE: &str = r#"
extern "C" __global__
void sparse_dict_block_gate(
    const float* __restrict__ z,   // [n_rows * (n_blocks*b)] row-major
    int n_rows,
    int n_blocks,
    int b,
    float* __restrict__ gates)     // [n_rows * n_blocks] row-major
{
  const long long total = (long long)n_rows * (long long)n_blocks;
  const long long idx = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
  if (idx >= total) return;
  const int row = (int)(idx / (long long)n_blocks);
  const int block = (int)(idx - (long long)row * (long long)n_blocks);
  const long long zbase =
      (long long)row * ((long long)n_blocks * (long long)b) + (long long)block * (long long)b;
  // Separate-rounding accumulation of the b squared projections, ascending r —
  // identical arithmetic to the CPU `e += v*v`. b is tiny (2-4).
  float acc = 0.0f;
  for (int r = 0; r < b; ++r) {
    const float v = z[zbase + (long long)r];
    acc = __fadd_rn(acc, __fmul_rn(v, v));
  }
  gates[(long long)row * (long long)n_blocks + (long long)block] = sqrtf(acc);
}
"#;

/// CPU reference for one row's group ℓ₂ **gate block**: `gate_g = ‖x D_gᵀ‖₂` for
/// every block `g`, built from the same ascending-`c`, separate-rounding
/// projection arithmetic ([`block_projections_row`]) the device score GEMM
/// reproduces bit-for-bit, then the group ℓ₂ ([`block_gates`]). This is the
/// parity oracle the device gate is locked against.
#[must_use]
pub fn block_gate_row_cpu(
    row: ArrayView1<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    n_blocks: usize,
    b: usize,
) -> Vec<f32> {
    let w = block_projections_row(row, decoder, n_blocks, b);
    block_gates(w.view())
}

/// CPU reference for a whole minibatch's gate block: `gates[r*n_blocks + g] =
/// ‖x_r D_gᵀ‖₂`, row-major. The bit-exact oracle for the device gate block.
#[must_use]
pub fn block_gate_block_cpu(
    rows: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    n_blocks: usize,
    b: usize,
) -> Vec<f32> {
    let n_rows = rows.nrows();
    let mut gates = vec![0.0f32; n_rows * n_blocks];
    for r in 0..n_rows {
        let g = block_gate_row_cpu(rows.row(r), decoder, n_blocks, b);
        gates[r * n_blocks..(r + 1) * n_blocks].copy_from_slice(&g);
    }
    gates
}

/// CPU oracle for the block route: each row's top-`k` `(block, gate)` shortlist,
/// selected by `(gate desc, block asc)`. Bit-identical to
/// [`super::block::route_block_minibatch`] up to f32 ties (that path forms `z`
/// with a blocked GEMM; this one and the device path share the ascending-`c`
/// scalar dot), which are interchangeable for the reconstruction. This is the
/// per-row-independent selection the device path must reproduce.
#[must_use]
pub fn route_blocks_cpu(
    rows: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    n_blocks: usize,
    b: usize,
    k: usize,
) -> Vec<Vec<(u32, f32)>> {
    rows.outer_iter()
        .map(|row| {
            let gates = block_gate_row_cpu(row, decoder, n_blocks, b);
            route_row_blocks(&gates, k)
        })
        .collect()
}

/// Minimum gate-block element count (`n_rows · K`, `K = G·b`) below which the
/// device launch is not worth its fixed cost. The GEMM cost is set by the `K`
/// atom-columns of `z`, so admission uses the same `n_rows × K` score-element
/// floor as the atom lane rather than `n_rows × G`.
#[cfg(target_os = "linux")]
pub const DEVICE_BLOCK_GATE_MIN_ELEMS: usize = gam_gpu::DEFAULT_DICTIONARY_SCORE_MIN_ELEMS;

/// Peak `z` score elements per device launch. The router walks `G` in
/// block-tiles sized so each launch's `n_rows × (tile_blocks·b)` `z` block stays
/// under this cap, keeping peak score memory bounded independent of `G`.
#[cfg(target_os = "linux")]
const GPU_BLOCK_ROUTE_TILE_ELEMS: usize = gam_gpu::DEFAULT_DICTIONARY_SCORE_TILE_ELEMS;

/// Fail-loud, residency-aware block-route entry point for the block-sparse lane.
///
/// Honours the process-wide [`gam_gpu::GpuPolicy`] contract: under
/// [`gam_gpu::GpuPolicy::Required`] a missing CUDA runtime, a compile/launch fault,
/// or an `n_rows × K` block below the device break-even all return `Err` instead
/// of silently degrading to the CPU. [`gam_gpu::GpuPolicy::Auto`] uses the device
/// when admitted and above break-even, else the CPU oracle; [`gam_gpu::GpuPolicy::Off`]
/// always the CPU oracle. The returned [`BlockRoutePath`] reports which ran, and
/// the `usize` is the device→host transfer in bytes (0 on the CPU path).
///
/// Both paths select the identical top-`k` block support (the device gate is
/// bit-identical to [`route_blocks_cpu`]'s), while the device downloads only the
/// `m × k` `(block, gate)` shortlists.
///
/// # Errors
/// Returns [`gam_gpu::GpuError`] when CUDA admission or execution fails.
/// One-shot engagement report for the block-gate router, mirroring the atom
/// lane's [`super::scoring_gpu`] `note_route_engagement` (#1551 "GPU 0%" class:
/// an `Auto` run that silently declines the device and falls back to the CPU
/// otherwise leaves no trace of WHY). Warns once per category per process — the
/// route is per-minibatch, so an unconditional line would spam thousands of
/// identical entries.
#[cfg(target_os = "linux")]
fn note_block_route_engagement(engaged: bool, detail: &str) {
    use std::sync::Once;
    static ENGAGED_ONCE: Once = Once::new();
    static DECLINED_ONCE: Once = Once::new();
    let once = if engaged {
        &ENGAGED_ONCE
    } else {
        &DECLINED_ONCE
    };
    once.call_once(|| {
        let verdict = if engaged {
            "device ENGAGED"
        } else {
            "device DECLINED - falling back to CPU"
        };
        log::warn!("[gam-sae sparse_dict block-gate router] {verdict}: {detail}");
    });
}

#[cfg(target_os = "linux")]
pub fn route_blocks_required(
    rows: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    b: usize,
    k: usize,
    mode: gam_gpu::GpuPolicy,
) -> Result<(Vec<Vec<(u32, f32)>>, BlockRoutePath, usize), gam_gpu::GpuError> {
    use gam_gpu::GpuPolicy;

    let m = rows.nrows();
    let krows = decoder.nrows();
    if b == 0 || krows == 0 || krows % b != 0 {
        return Err(gam_gpu::gpu_err!(
            "block-gate route: decoder K={krows} rows not a positive multiple of block_size b={b}"
        ));
    }
    let g = krows / b;
    let active = k.max(1).min(g.max(1));

    // Production CPU fallback: the blocked-GEMM router — the same top-`k`
    // support as the scalar oracle (`route_blocks_cpu`, which stays as the
    // device parity reference) but ~2 orders of magnitude faster. Under the
    // default `Auto` policy every below-break-even minibatch and every
    // CUDA-less Linux host lands here, so this closure IS the hot CPU path
    // (#2242: the scalar per-row oracle was burning 75% of block-lane cycles).
    let cpu_tile_blocks =
        (GPU_BLOCK_ROUTE_TILE_ELEMS / (rows.nrows().max(1) * b.max(1))).clamp(1, g.max(1));
    let cpu_route =
        || super::block::route_block_minibatch(rows, decoder, g, b, active, cpu_tile_blocks);

    if mode == GpuPolicy::Off {
        return Ok((cpu_route(), BlockRoutePath::Cpu, 0));
    }

    // Admission on the GEMM work `m × K` (K = G·b): that is what justifies the
    // launch. The launches themselves are G-tiled so buffers never grow with G.
    let plan = gam_gpu::DictionaryScoreRoutePlan::with_limits(
        m,
        krows,
        decoder.ncols(),
        DEVICE_BLOCK_GATE_MIN_ELEMS,
        GPU_BLOCK_ROUTE_TILE_ELEMS,
    );
    if !plan.device_admitted {
        if mode == GpuPolicy::Required {
            return Err(gam_gpu::gpu_err!(
                "block-gate route GpuPolicy::Required: block of {m}×{krows} = {} elems is below the \
                 device launch break-even (DEVICE_BLOCK_GATE_MIN_ELEMS={DEVICE_BLOCK_GATE_MIN_ELEMS}); \
                 refusing to silently run on the CPU",
                m.saturating_mul(krows)
            ));
        }
        note_block_route_engagement(
            false,
            &format!(
                "block {m}x{krows} = {} elems below the device launch break-even \
                 (DEVICE_BLOCK_GATE_MIN_ELEMS={DEVICE_BLOCK_GATE_MIN_ELEMS})",
                m.saturating_mul(krows)
            ),
        );
        return Ok((cpu_route(), BlockRoutePath::Cpu, 0));
    }
    if m == 0 || g == 0 {
        return Ok((cpu_route(), BlockRoutePath::Cpu, 0));
    }

    let runtime = if mode == GpuPolicy::Required {
        Some(gam_gpu::GpuRuntime::require()?)
    } else {
        gam_gpu::GpuRuntime::resolve(mode)?
    };
    if runtime.is_none() {
        note_block_route_engagement(false, "Auto admission found no CUDA device");
        return Ok((cpu_route(), BlockRoutePath::Cpu, 0));
    }

    // Blocks per launch: bound the per-launch `z` block `m × (tile_blocks·b)` to
    // GPU_BLOCK_ROUTE_TILE_ELEMS, at least one block, never more than G.
    let tile_blocks = (plan.tile_items / b.max(1)).clamp(1, g);

    let out = device::route_blocks_device(rows, decoder, b, g, active, tile_blocks)?;
    note_block_route_engagement(
        true,
        &format!("block {m}x{krows}, tile_blocks={tile_blocks}, active={active}"),
    );
    Ok((
        out.selections,
        BlockRoutePath::Device,
        out.device_dtoh_bytes,
    ))
}

#[cfg(target_os = "linux")]
mod device {
    use gam_gpu::gpu_error::{GpuError, GpuResultExt};
    use ndarray::ArrayView2;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};

    struct Backend {
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        modules: Mutex<HashMap<usize, Arc<CudaModule>>>,
        max_shared_mem_per_block: usize,
    }

    fn backend() -> Result<&'static Backend, GpuError> {
        static BACKEND: OnceLock<Result<Backend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                let parts = gam_gpu::backend_probe::probe_cuda_backend("sparse_dict_block_gate")?;
                Ok(Backend {
                    ctx: parts.ctx,
                    stream: parts.stream,
                    modules: Mutex::new(HashMap::new()),
                    max_shared_mem_per_block: gam_gpu::GpuRuntime::require()?
                        .selected_device()
                        .max_shared_mem_per_block,
                })
            })
            .as_ref()
            .map_err(GpuError::clone)
    }

    /// Combined NVRTC source: the atom lane's bit-exact score GEMM + top-`s` fold
    /// ([`super::super::scoring_gpu::score_block_kernel_source`], which bakes `PP`)
    /// plus this lane's ℓ₂-gate epilogue ([`super::BLOCK_GATE_KERNEL_SOURCE`]).
    fn combined_kernel_source(p: usize) -> String {
        format!(
            "{}\n{}",
            super::super::scoring_gpu::score_block_kernel_source(p),
            super::BLOCK_GATE_KERNEL_SOURCE
        )
    }

    fn module_for(b: &Backend, p: usize) -> Result<Arc<CudaModule>, GpuError> {
        if let Ok(guard) = b.modules.lock() {
            if let Some(m) = guard.get(&p) {
                return Ok(m.clone());
            }
        }
        let ptx = gam_gpu::device_cache::compile_ptx_arch(combined_kernel_source(p))
            .gpu_ctx_with(|err| format!("sparse_dict block-gate NVRTC (P={p}): {err}"))?;
        let module = b
            .ctx
            .load_module(ptx)
            .gpu_ctx("sparse_dict block-gate module load")?;
        if let Ok(mut guard) = b.modules.lock() {
            guard.entry(p).or_insert_with(|| module.clone());
        }
        Ok(module)
    }

    const TOP_S_FOLD_THREADS: u32 = 32;
    const GATE_KERNEL_THREADS: u32 = 256;

    /// Number of bounded-progress checkpoints across one route's block-tile walk
    /// (#2227). A telemetry/backlog cadence, not a numerical tuning knob: the tile
    /// loop synchronises `min(tile_count, this)` times so the in-flight async
    /// launch backlog is bounded to `ceil(tile_count/this)` tiles and any device
    /// fault or stall is attributed to the tile window that produced it, rather
    /// than surfacing (if at all) as one unattributed block in the terminal
    /// synchronize with no telemetry for the whole high-`K` route.
    const ROUTE_PROGRESS_CHECKPOINTS: usize = 16;

    pub(super) struct BlockRouteDeviceOutput {
        pub(super) selections: Vec<Vec<(u32, f32)>>,
        pub(super) device_dtoh_bytes: usize,
    }

    fn fold_shared_bytes(
        active: usize,
        threads: u32,
        max_shared_mem_per_block: usize,
    ) -> Result<u32, GpuError> {
        let slots = (threads as usize)
            .checked_add(1)
            .and_then(|v| v.checked_mul(active))
            .ok_or_else(|| {
                gam_gpu::gpu_err!("sparse_dict block-gate fold shared-memory overflow")
            })?;
        let bytes = slots
            .checked_mul(
                std::mem::size_of::<u32>()
                    + std::mem::size_of::<f32>()
                    + std::mem::size_of::<f32>(),
            )
            .ok_or_else(|| {
                gam_gpu::gpu_err!("sparse_dict block-gate fold shared-memory overflow")
            })?;
        if max_shared_mem_per_block > 0 && bytes > max_shared_mem_per_block {
            return Err(gam_gpu::gpu_err!(
                "sparse_dict block-gate fold requires {bytes} shared-memory bytes per row block \
                 (active={active}, threads={threads}) but the selected device reports \
                 max_shared_mem_per_block={max_shared_mem_per_block}"
            ));
        }
        u32::try_from(bytes).map_err(|_| {
            gam_gpu::gpu_err!("sparse_dict block-gate fold shared-memory bytes overflow")
        })
    }

    /// Route a whole minibatch's blocks on the device: rows and the whole decoder
    /// stay resident; per block-tile one score GEMM forms `z` (`m × tile_blocks·b`),
    /// the gate epilogue reduces it to `m × tile_blocks` gates, and the resident
    /// top-`k` fold folds those into per-row `(block, gate)` shortlists. Only the
    /// final `m × k` shortlists cross PCIe.
    pub(super) fn route_blocks_device(
        rows: ArrayView2<'_, f32>,
        decoder: ArrayView2<'_, f32>,
        b: usize,
        n_blocks: usize,
        active: usize,
        tile_blocks: usize,
    ) -> Result<BlockRouteDeviceOutput, GpuError> {
        let m = rows.nrows();
        let p = rows.ncols();
        let krows = decoder.nrows();
        if p != decoder.ncols() {
            return Err(gam_gpu::gpu_err!(
                "sparse_dict block-gate: P mismatch rows={p} decoder={}",
                decoder.ncols()
            ));
        }
        if b == 0 || krows != n_blocks * b {
            return Err(gam_gpu::gpu_err!(
                "sparse_dict block-gate: decoder K={krows} != G*b = {n_blocks}*{b}"
            ));
        }
        if m == 0 || n_blocks == 0 || p == 0 {
            return Ok(BlockRouteDeviceOutput {
                selections: vec![Vec::new(); m],
                device_dtoh_bytes: 0,
            });
        }
        let active = active.max(1).min(n_blocks);
        if n_blocks > u32::MAX as usize {
            return Err(gam_gpu::gpu_err!(
                "sparse_dict block-gate G={n_blocks} exceeds u32 block-index storage"
            ));
        }

        let backend = backend()?;
        let module = module_for(backend, p)?;
        let score_func = module
            .load_function("sparse_dict_score_block_offset")
            .gpu_ctx("sparse_dict block-gate score load_function")?;
        let gate_func = module
            .load_function("sparse_dict_block_gate")
            .gpu_ctx("sparse_dict block-gate gate load_function")?;
        let fold_func = module
            .load_function("sparse_dict_fold_top_s")
            .gpu_ctx("sparse_dict block-gate fold load_function")?;
        let stream = backend.stream.clone();

        let rows_storage: Vec<f32>;
        let rows_host: &[f32] = if let Some(slice) = rows.as_slice() {
            slice
        } else {
            rows_storage = rows.iter().copied().collect();
            rows_storage.as_slice()
        };
        assert_eq!(rows_host.len(), m * p, "block-gate rows flatten length");
        let rows_dev = stream
            .clone_htod(rows_host)
            .gpu_ctx("sparse_dict block-gate htod rows")?;

        let decoder_storage: Vec<f32>;
        let decoder_host: &[f32] = if let Some(slice) = decoder.as_slice() {
            slice
        } else {
            decoder_storage = decoder.iter().copied().collect();
            decoder_storage.as_slice()
        };
        assert_eq!(
            decoder_host.len(),
            krows * p,
            "block-gate decoder flatten length"
        );
        let decoder_dev = stream
            .clone_htod(decoder_host)
            .gpu_ctx("sparse_dict block-gate htod decoder")?;

        let m_i32 =
            i32::try_from(m).map_err(|_| gam_gpu::gpu_err!("block-gate m={m} overflows i32"))?;
        let active_i32 = i32::try_from(active)
            .map_err(|_| gam_gpu::gpu_err!("block-gate active={active} overflows i32"))?;
        let b_i32 =
            i32::try_from(b).map_err(|_| gam_gpu::gpu_err!("block-gate b={b} overflows i32"))?;

        let tile_blocks = tile_blocks.clamp(1, n_blocks);
        let max_tile_atoms = tile_blocks * b;
        let mut z_dev = stream
            .alloc_zeros::<f32>(m * max_tile_atoms)
            .gpu_ctx("sparse_dict block-gate alloc z")?;
        let mut gate_dev = stream
            .alloc_zeros::<f32>(m * tile_blocks)
            .gpu_ctx("sparse_dict block-gate alloc gates")?;
        let mut top_blocks_dev = stream
            .alloc_zeros::<u32>(m * active)
            .gpu_ctx("sparse_dict block-gate alloc top blocks")?;
        let mut top_gates_dev = stream
            .alloc_zeros::<f32>(m * active)
            .gpu_ctx("sparse_dict block-gate alloc top gates")?;
        let mut top_mags_dev = stream
            .alloc_zeros::<f32>(m * active)
            .gpu_ctx("sparse_dict block-gate alloc top mags")?;
        let fold_shared =
            fold_shared_bytes(active, TOP_S_FOLD_THREADS, backend.max_shared_mem_per_block)?;

        let tile_m = super::super::scoring_gpu::SCORE_BLOCK_TILE_M;
        let tile_n = super::super::scoring_gpu::SCORE_BLOCK_TILE_N;

        // Bounded-progress checkpoints (#2227). The block-tile walk enqueues every
        // score+gate+fold launch on one stream; without intermediate
        // synchronisation a device fault or stall in any tile surfaces only at the
        // terminal synchronize, as a single unattributed block with no telemetry
        // for the whole high-`G` route. Synchronise on a cadence derived from the
        // tile count so the async backlog is bounded and each fault is attributed
        // to its tile window; the heartbeat is `log::debug!` so an ordinary
        // (info-level) per-minibatch run is not flooded.
        let tile_count = n_blocks.div_ceil(tile_blocks.max(1));
        let checkpoint_stride = tile_count
            .div_ceil(ROUTE_PROGRESS_CHECKPOINTS.max(1))
            .max(1);
        let route_started = std::time::Instant::now();
        let mut tiles_done = 0usize;
        let mut checkpoint_lo = 0usize;
        let mut g0 = 0usize;
        while g0 < n_blocks {
            let g1 = (g0 + tile_blocks).min(n_blocks);
            let tile_g = g1 - g0;
            let n_atoms = tile_g * b; // z columns this tile
            let atom_offset = u32::try_from(g0 * b)
                .map_err(|_| gam_gpu::gpu_err!("block-gate atom offset overflows u32"))?;
            let block_offset = u32::try_from(g0)
                .map_err(|_| gam_gpu::gpu_err!("block-gate block offset overflows u32"))?;
            let n_atoms_i32 = i32::try_from(n_atoms)
                .map_err(|_| gam_gpu::gpu_err!("block-gate n_atoms={n_atoms} overflows i32"))?;
            let tile_g_i32 = i32::try_from(tile_g)
                .map_err(|_| gam_gpu::gpu_err!("block-gate tile_g={tile_g} overflows i32"))?;

            // (1) score GEMM: z[m × n_atoms] over decoder rows [g0·b, g1·b).
            let grid_x: u32 = u32::try_from(n_atoms.div_ceil(tile_n as usize))
                .map_err(|_| gam_gpu::gpu_err!("block-gate score grid_x overflow"))?;
            let grid_y: u32 = u32::try_from(m.div_ceil(tile_m as usize))
                .map_err(|_| gam_gpu::gpu_err!("block-gate score grid_y overflow"))?;
            let score_cfg = LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (
                    super::super::scoring_gpu::SCORE_BLOCK_THREADS_N,
                    super::super::scoring_gpu::SCORE_BLOCK_THREADS_M,
                    1,
                ),
                shared_mem_bytes: 0,
            };
            let mut score = stream.launch_builder(&score_func);
            score
                .arg(&rows_dev)
                .arg(&decoder_dev)
                .arg(&m_i32)
                .arg(&n_atoms_i32)
                .arg(&atom_offset)
                .arg(&mut z_dev);
            // SAFETY: grid/block validated; device pointers are cudarc-checked
            // allocations on this stream. The GEMM reads the resident rows and the
            // resident decoder slice [atom_offset, atom_offset + n_atoms) and writes
            // exactly m*n_atoms z values.
            unsafe { score.launch(score_cfg) }.gpu_ctx("sparse_dict block-gate score launch")?;

            // (2) gate epilogue: reduce each adjacent b-group of z to its ℓ₂ norm.
            let gate_elems = m.saturating_mul(tile_g);
            let gate_grid: u32 = u32::try_from(gate_elems.div_ceil(GATE_KERNEL_THREADS as usize))
                .map_err(|_| gam_gpu::gpu_err!("block-gate gate grid overflow"))?;
            let gate_cfg = LaunchConfig {
                grid_dim: (gate_grid, 1, 1),
                block_dim: (GATE_KERNEL_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut gate = stream.launch_builder(&gate_func);
            gate.arg(&z_dev)
                .arg(&m_i32)
                .arg(&tile_g_i32)
                .arg(&b_i32)
                .arg(&mut gate_dev);
            // SAFETY: one thread per (row, block) output within m*tile_g; reads the
            // z block just written on this stream, writes m*tile_g gates.
            unsafe { gate.launch(gate_cfg) }.gpu_ctx("sparse_dict block-gate gate launch")?;

            // (3) fold gates into resident per-row top-k block shortlists. The fold
            // treats the gate as a non-negative "score" and the block as its "atom":
            // (gate desc, block asc) — the identical order route_row_blocks uses.
            let fold_cfg = LaunchConfig {
                grid_dim: (
                    u32::try_from(m)
                        .map_err(|_| gam_gpu::gpu_err!("block-gate fold grid overflow"))?,
                    1,
                    1,
                ),
                block_dim: (TOP_S_FOLD_THREADS, 1, 1),
                shared_mem_bytes: fold_shared,
            };
            let mut fold = stream.launch_builder(&fold_func);
            fold.arg(&gate_dev)
                .arg(&m_i32)
                .arg(&tile_g_i32)
                .arg(&block_offset)
                .arg(&active_i32)
                .arg(&mut top_blocks_dev)
                .arg(&mut top_gates_dev)
                .arg(&mut top_mags_dev);
            // SAFETY: one block per row; reads the gate tile just written on this
            // stream, updates exactly m*active shortlist slots.
            unsafe { fold.launch(fold_cfg) }.gpu_ctx("sparse_dict block-gate fold launch")?;

            g0 = g1;
            tiles_done += 1;
            if tiles_done % checkpoint_stride == 0 || g0 >= n_blocks {
                stream.synchronize().gpu_ctx_with(|err| {
                    format!(
                        "sparse_dict block-gate route progress checkpoint (tiles {checkpoint_lo}..{tiles_done} of {tile_count}, blocks 0..{g0} of {n_blocks}): {err}"
                    )
                })?;
                log::debug!(
                    "[SAE block route] tiles {tiles_done}/{tile_count} blocks {g0}/{n_blocks} \
                     elapsed {:.2}s",
                    route_started.elapsed().as_secs_f64(),
                );
                checkpoint_lo = tiles_done;
            }
        }

        let mut top_blocks = vec![0u32; m * active];
        let mut top_gates = vec![0.0f32; m * active];
        stream
            .memcpy_dtoh(&top_blocks_dev, &mut top_blocks)
            .gpu_ctx("sparse_dict block-gate dtoh blocks")?;
        stream
            .memcpy_dtoh(&top_gates_dev, &mut top_gates)
            .gpu_ctx("sparse_dict block-gate dtoh gates")?;
        stream
            .synchronize()
            .gpu_ctx("sparse_dict block-gate synchronize")?;

        let mut selections = Vec::with_capacity(m);
        for r in 0..m {
            let mut row = Vec::with_capacity(active);
            let base = r * active;
            for j in 0..active {
                let block = top_blocks[base + j];
                if block != u32::MAX {
                    row.push((block, top_gates[base + j]));
                }
            }
            selections.push(row);
        }
        Ok(BlockRouteDeviceOutput {
            selections,
            device_dtoh_bytes: m
                .saturating_mul(active)
                .saturating_mul(std::mem::size_of::<u32>() + std::mem::size_of::<f32>()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // Both callers are `cfg(target_os = "linux")` device-parity tests, so this
    // admission helper is dead off-Linux and `-D dead-code` rejects the test
    // target there (a wheel-blocking break class). Gate it with its callers.
    #[cfg(target_os = "linux")]
    fn cuda_available_for_test(label: &str) -> bool {
        match gam_gpu::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto) {
            Ok(Some(_)) => true,
            Ok(None) => {
                eprintln!("[sparse_dict block-gate test] no CUDA device; skipping {label}");
                false
            }
            Err(error) => panic!("[sparse_dict block-gate test] CUDA admission failed: {error}"),
        }
    }

    /// Deterministic fp32 fixture: `n_rows × p` rows and a `G·b × p` decoder whose
    /// blocks are orthonormalised so the gate `‖x D_gᵀ‖₂` is a genuine subspace
    /// energy — the shape the block lane fits.
    fn fixture(n_rows: usize, n_blocks: usize, b: usize, p: usize) -> (Array2<f32>, Array2<f32>) {
        let rows = Array2::from_shape_fn((n_rows, p), |(i, c)| {
            (((i * 29 + c * 13) as f32) * 0.017).sin() * 0.8
        });
        let mut decoder = Array2::from_shape_fn((n_blocks * b, p), |(a, c)| {
            (((a * 11 + c * 3) as f32) * 0.009).cos()
        });
        // Orthonormalise each block's b rows so it is a real St(b, P) frame.
        for g in 0..n_blocks {
            let mut block = decoder.slice(ndarray::s![g * b..g * b + b, ..]).to_owned();
            super::super::block::gram_schmidt_rows(&mut block);
            for r in 0..b {
                for c in 0..p {
                    decoder[[g * b + r, c]] = block[[r, c]];
                }
            }
        }
        (rows, decoder)
    }

    #[test]
    fn cpu_gate_block_matches_per_row_reference() {
        // The block-form gate oracle must equal the per-row group ℓ₂ exactly.
        let (rows, decoder) = fixture(6, 10, 3, 8);
        let block = block_gate_block_cpu(rows.view(), decoder.view(), 10, 3);
        for r in 0..rows.nrows() {
            let per_row = block_gate_row_cpu(rows.row(r), decoder.view(), 10, 3);
            for g in 0..10 {
                assert_eq!(
                    block[r * 10 + g].to_bits(),
                    per_row[g].to_bits(),
                    "gate block vs per-row differ at r={r} g={g}"
                );
            }
        }
    }

    #[test]
    fn cpu_route_selects_by_gate_desc_block_asc() {
        // The CPU oracle must reproduce route_row_blocks selection semantics.
        let (rows, decoder) = fixture(4, 12, 2, 7);
        let routed = route_blocks_cpu(rows.view(), decoder.view(), 12, 2, 3);
        assert_eq!(routed.len(), 4);
        for sel in &routed {
            assert!(sel.len() <= 3 && !sel.is_empty());
            // Gates are non-increasing; ties break by ascending block index.
            for w in sel.windows(2) {
                let (ga, ba) = (w[0].1, w[0].0);
                let (gb, bb) = (w[1].1, w[1].0);
                assert!(ga > gb || (ga == gb && ba < bb), "order violated: {w:?}");
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn device_block_route_matches_cpu_selection_when_available() {
        // Selection-parity gate (SPEC 20). The m×K block MUST clear
        // DEVICE_BLOCK_GATE_MIN_ELEMS so the device path is admitted; on a CUDA
        // host we drive Required (silent CPU fallback = hard failure) and assert
        // the routed top-k BLOCK support equals the per-row CPU oracle exactly —
        // same blocks, same order — with gate values within f32 tolerance (they
        // are in fact bit-identical, as the arithmetic is shared). With no runtime
        // Required must fail closed and Auto reproduces the CPU oracle.
        let m = 512usize;
        let g = 2048usize;
        let b = 3usize;
        let k = 4usize;
        let p = 48usize;
        let krows = g * b;
        assert!(m * krows >= DEVICE_BLOCK_GATE_MIN_ELEMS);
        let (rows, decoder) = fixture(m, g, b, p);

        let cpu = route_blocks_cpu(rows.view(), decoder.view(), g, b, k);

        if !cuda_available_for_test("block-route parity") {
            return;
        }

        match route_blocks_required(
            rows.view(),
            decoder.view(),
            b,
            k,
            gam_gpu::GpuPolicy::Required,
        ) {
            Ok((routed, path, dtoh_bytes)) => {
                assert_eq!(
                    path,
                    BlockRoutePath::Device,
                    "Required succeeded but reported CPU — device did not engage"
                );
                assert_eq!(
                    dtoh_bytes,
                    m * k * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>()),
                    "device block route must download only the m×k (block, gate) shortlist"
                );
                assert_eq!(routed.len(), cpu.len());
                for (r, (dev_sel, cpu_sel)) in routed.iter().zip(&cpu).enumerate() {
                    assert_eq!(
                        dev_sel.len(),
                        cpu_sel.len(),
                        "row {r}: selection length differs"
                    );
                    for (j, ((db, dg), (cb, cg))) in dev_sel.iter().zip(cpu_sel).enumerate() {
                        assert_eq!(db, cb, "row {r} slot {j}: block differs dev={db} cpu={cb}");
                        let tol = 1e-5 * cg.abs().max(1.0);
                        assert!(
                            (dg - cg).abs() <= tol,
                            "row {r} slot {j}: gate differs dev={dg} cpu={cg} tol={tol}"
                        );
                    }
                }
            }
            Err(err) => panic!("Required block route failed after CUDA admission: {err}"),
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn device_block_route_bit_identical_gates_at_scale() {
        // Larger G to exercise multi-tile folding; asserts BIT-identical gates
        // (not just within tolerance) to prove the shared-arithmetic parity story
        // holds across tile boundaries.
        let m = 256usize;
        let g = 8192usize;
        let b = 2usize;
        let k = 6usize;
        let p = 64usize;
        let krows = g * b;
        assert!(m * krows >= DEVICE_BLOCK_GATE_MIN_ELEMS);
        let (rows, decoder) = fixture(m, g, b, p);
        let cpu = route_blocks_cpu(rows.view(), decoder.view(), g, b, k);

        if !cuda_available_for_test("block-route scale parity") {
            return;
        }

        match route_blocks_required(
            rows.view(),
            decoder.view(),
            b,
            k,
            gam_gpu::GpuPolicy::Required,
        ) {
            Ok((routed, path, _)) => {
                assert_eq!(
                    path,
                    BlockRoutePath::Device,
                    "device did not engage at scale"
                );
                for (r, (dev_sel, cpu_sel)) in routed.iter().zip(&cpu).enumerate() {
                    assert_eq!(dev_sel.len(), cpu_sel.len(), "row {r}: length differs");
                    for (j, ((db, dg), (cb, cg))) in dev_sel.iter().zip(cpu_sel).enumerate() {
                        assert_eq!(db, cb, "row {r} slot {j}: block differs");
                        assert_eq!(
                            dg.to_bits(),
                            cg.to_bits(),
                            "row {r} slot {j}: gate bits differ dev={dg} cpu={cg}"
                        );
                    }
                }
            }
            Err(err) => panic!("Required scale route failed after CUDA admission: {err}"),
        }
    }
}
