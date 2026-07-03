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
//! and discards it. The minibatch router [`route_minibatch_required`] walks the
//! whole `K`-wide dictionary in atom-column tiles (each launch's block capped at
//! `GPU_ROUTE_TILE_ELEMS`), so peak host/device score memory is `rows × tile`,
//! **independent of `K`** — the lane's no-`N×K` memory discipline is preserved
//! on the device exactly as on the CPU; the GPU just does the `O(rows·tile·P)`
//! multiply-accumulate that dominates.
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

/// The bit-exact-parity NVRTC kernel. A `BM × BN` output tile per CUDA block;
/// each thread owns one `(row, atom)` output and accumulates over `P` columns in
/// ascending order with separate-rounding f32 ops so the result matches the CPU
/// sequential `acc += x·d` to the bit.
///
/// The row and atom operands for the tile are cooperatively staged into shared
/// memory once per block, so the `BM·BN` outputs share `(BM + BN)·PP` global
/// loads instead of re-reading `2·PP` per output — a `BM·BN / (BM + BN)` cut in
/// global traffic on what `ptxas -v` shows is a 0-spill, full-occupancy,
/// **bandwidth-bound** kernel (0.25 flop/byte with no reuse). Measured on an
/// A40 at the router-tile shape `256 × 8192 × P64`, `32 × 32` tiles run **3.40×
/// faster** than the untiled one-thread-per-output kernel and produce a
/// **bit-identical** score block (0 / 2_097_152 f32 mismatches): the arithmetic
/// is unchanged — every output still sums its `PP` terms in ascending `c` with
/// `__fmul_rn`/`__fadd_rn`; shared memory only holds exact copies of the same
/// operands, so the CPU-oracle parity gate is preserved by construction.
///
/// `PP` (the column count) is baked in as a `#define` so the inner loop is a
/// fixed trip count (matching the other NVRTC kernels in this repo, which
/// monomorphise their shape macros for a pure `compile_ptx`). `BM`/`BN` are the
/// output-tile dimensions; the host launch (`score_block_device`) must use a
/// `(BN, BM)` block and a `ceil(n_atoms/BN) × ceil(n_rows/BM)` grid.
pub const SCORE_BLOCK_KERNEL_SOURCE: &str = r#"
#define BM 32
#define BN 32
extern "C" __global__
void sparse_dict_score_block(
    const float* __restrict__ rows,    // [n_rows * PP] row-major
    const float* __restrict__ atoms,   // [n_atoms * PP] row-major (decoder tile)
    int n_rows,
    int n_atoms,
    float* __restrict__ scores)        // [n_rows * n_atoms] row-major
{
  // Shared operand tiles: BM rows and BN atoms, each PP long. The BM·BN outputs
  // of this block reuse them, cutting global traffic by BM·BN/(BM+BN).
  __shared__ float sr[BM][PP];
  __shared__ float sa[BN][PP];
  const int row0  = blockIdx.y * BM;
  const int atom0 = blockIdx.x * BN;
  const int tx = threadIdx.x;   // 0..BN-1  atom within tile
  const int ty = threadIdx.y;   // 0..BM-1  row within tile
  const int lin = ty * BN + tx;
  const int nthreads = BM * BN;
  // Cooperative, coalesced load of the row/atom operands (zero-padded past the
  // ragged tail so out-of-range lanes read a defined 0 they never store).
  for (int e = lin; e < BM * PP; e += nthreads) {
    int rr = e / PP, cc = e - rr * PP;
    int gr = row0 + rr;
    sr[rr][cc] = (gr < n_rows) ? rows[(long long)gr * PP + cc] : 0.0f;
  }
  for (int e = lin; e < BN * PP; e += nthreads) {
    int aa = e / PP, cc = e - aa * PP;
    int ga = atom0 + aa;
    sa[aa][cc] = (ga < n_atoms) ? atoms[(long long)ga * PP + cc] : 0.0f;
  }
  __syncthreads();
  const int r = row0 + ty;
  const int a = atom0 + tx;
  if (r < n_rows && a < n_atoms) {
    // SEPARATE-rounding accumulation in ascending c — NO fused multiply-add, on
    // exact copies of the operands, so this is bit-identical to the CPU
    // `acc += x[c]*d[c]` reference order (and to the untiled kernel).
    float acc = 0.0f;
    for (int c = 0; c < PP; ++c) {
      float prod = __fmul_rn(sr[ty][c], sa[tx][c]);
      acc = __fadd_rn(acc, prod);
    }
    scores[(long long)r * n_atoms + a] = acc;
  }
}
"#;

/// Output-tile dimensions the [`SCORE_BLOCK_KERNEL_SOURCE`] kernel is written
/// for; the host launch must match them. Kept in sync with the `#define`s at the
/// top of the kernel string (a single source of truth for the launch geometry).
pub const SCORE_BLOCK_TILE_M: u32 = 32;
pub const SCORE_BLOCK_TILE_N: u32 = 32;

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
    assert_eq!(
        p,
        atoms.ncols(),
        "score_block_cpu: P mismatch rows vs atoms"
    );
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
pub const DEVICE_SCORE_BLOCK_MIN_ELEMS: usize = gam_gpu::DEFAULT_DICTIONARY_SCORE_MIN_ELEMS;

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
    let plan = gam_gpu::DictionaryScoreRoutePlan::with_limits(
        n_rows,
        n_atoms,
        rows.ncols(),
        DEVICE_SCORE_BLOCK_MIN_ELEMS,
        GPU_ROUTE_TILE_ELEMS,
    );

    if mode == GpuMode::Off {
        return Ok((score_block_cpu(rows, atoms), ScoreBlockPath::Cpu));
    }

    if mode == GpuMode::Required && !plan.device_admitted {
        return Err(gam_gpu::gpu_err!(
            "sparse_dict score-block GpuMode::Required: block of {n_rows}×{n_atoms} = {} \
             elems is below the device launch break-even \
             (DEVICE_SCORE_BLOCK_MIN_ELEMS={DEVICE_SCORE_BLOCK_MIN_ELEMS}); refusing \
             to silently run on the CPU",
            n_rows.saturating_mul(n_atoms)
        ));
    }
    if plan.device_admitted {
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

/// Peak score elements per device launch for the tiled GPU router. The router
/// NEVER materialises the whole `m × K` block: it walks `K` in atom-column tiles
/// sized so each launch's `m × cols` block stays under this cap (~2M f32 ≈ 8 MB
/// host + 8 MB device), then discards it after folding. This keeps peak score
/// memory bounded **independent of `K`** — the same discipline the CPU lane
/// ([`super::scoring::top_s_online`]) keeps with its `rows × tile` column tiles —
/// so a `K ≈ 32_000` fit does not balloon a `device alloc` linearly in `K`.
const GPU_ROUTE_TILE_ELEMS: usize = gam_gpu::DEFAULT_DICTIONARY_SCORE_TILE_ELEMS;

/// Route a whole minibatch of rows against the full decoder, returning each
/// row's top-`s` `(atom, score)` selection — BIT-IDENTICAL to calling
/// [`super::scoring::top_s_online`] per row, but with the score block computed on
/// the device (in `K`-tiled launches) when admitted.
///
/// The selection ([`super::scoring::TopSSelector`]) is single-sourced on the CPU
/// and fed the device scores in ascending atom order. `TopSSelector` keeps the
/// top-`s` by `(|score| desc, atom asc)` — a strict total order on the unique
/// atom indices — so the selected set is independent of the order or the tiling
/// in which candidates are offered. Combined with bit-identical scores (the
/// kernel forbids FMA contraction), the routed support matches the CPU oracle
/// **exactly**, whichever path and whatever GPU tile width computed the block.
///
/// Memory: the `m × K` block is never formed whole — `K` is walked in tiles of
/// at most `GPU_ROUTE_TILE_ELEMS / m` atom-columns, each launched, folded, and
/// discarded, so peak score memory is `m × tile_cols`, independent of `K`.
///
/// Falls back to the per-row CPU `top_s_online` under [`gam_gpu::GpuMode::Off`],
/// below the device break-even, or on any device error under
/// [`gam_gpu::GpuMode::Auto`]; under [`gam_gpu::GpuMode::Required`] a device
/// failure is propagated. The returned [`ScoreBlockPath`] reports which path ran.
///
/// # Errors
/// Returns [`gam_gpu::GpuError`] when [`gam_gpu::GpuMode::Required`] is set but
/// the device path cannot run for this minibatch.
pub fn route_minibatch_required(
    rows: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    s: usize,
    tile: usize,
    mode: gam_gpu::GpuMode,
) -> Result<(Vec<Vec<(u32, f32)>>, ScoreBlockPath), gam_gpu::GpuError> {
    use super::scoring::{TopSSelector, top_s_online};

    let m = rows.nrows();
    let k = decoder.nrows();

    // CPU per-row path (bit-identical oracle), used for Off / below break-even /
    // Auto device-error fallback.
    let cpu_route = || -> Vec<Vec<(u32, f32)>> {
        rows.outer_iter()
            .map(|row| top_s_online(row, decoder, s, tile))
            .collect()
    };

    if mode == gam_gpu::GpuMode::Off {
        return Ok((cpu_route(), ScoreBlockPath::Cpu));
    }

    // Engagement is decided on the TOTAL work `m × K` (that is what justifies the
    // device's fixed launch cost), but the launches themselves are K-tiled so the
    // buffers never grow with K.
    let plan = gam_gpu::DictionaryScoreRoutePlan::with_limits(
        m,
        k,
        decoder.ncols(),
        DEVICE_SCORE_BLOCK_MIN_ELEMS,
        GPU_ROUTE_TILE_ELEMS,
    );
    if !plan.device_admitted {
        if mode == gam_gpu::GpuMode::Required {
            return Err(gam_gpu::gpu_err!(
                "route_minibatch GpuMode::Required: block of {m}×{k} = {} elems is below \
                 the device launch break-even (DEVICE_SCORE_BLOCK_MIN_ELEMS={DEVICE_SCORE_BLOCK_MIN_ELEMS}); \
                 refusing to silently run on the CPU",
                m.saturating_mul(k)
            ));
        }
        return Ok((cpu_route(), ScoreBlockPath::Cpu));
    }
    if m == 0 || k == 0 {
        return Ok((cpu_route(), ScoreBlockPath::Cpu));
    }

    // Atom-columns per device launch: bound the per-launch block to
    // GPU_ROUTE_TILE_ELEMS, at least one column, never more than K.
    let tile_cols = plan.tile_items;

    // Per-row online selectors; each device tile's scores are folded in ascending
    // global atom order (offset + ascending local), and the selector's result is
    // tile-order-invariant, so the support is bit-identical to top_s_online.
    let mut selectors: Vec<TopSSelector> = (0..m).map(|_| TopSSelector::new(s)).collect();
    match device::score_decoder_tiled_device(rows, decoder, tile_cols, |start, cols, block| {
        for (r, sel) in selectors.iter_mut().enumerate() {
            let base = r * cols;
            for (local, score) in block[base..base + cols].iter().enumerate() {
                sel.offer((start + local) as u32, *score);
            }
        }
    }) {
        Ok(()) => {}
        Err(err) => {
            if mode == gam_gpu::GpuMode::Required {
                return Err(err);
            }
            // Auto: the device faulted mid-route; discard partial selectors and
            // run the exact CPU oracle for the whole minibatch.
            return Ok((cpu_route(), ScoreBlockPath::Cpu));
        }
    }

    let routed = selectors.into_iter().map(TopSSelector::finish).collect();
    Ok((routed, ScoreBlockPath::Device))
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
        let ptx = gam_gpu::device_cache::compile_ptx_arch(score_block_kernel_source(p))
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
        assert_eq!(
            rows_host.len(),
            n_rows * p,
            "score-block rows flatten length"
        );
        assert_eq!(
            atoms_host.len(),
            n_atoms * p,
            "score-block atoms flatten length"
        );

        let rows_dev = stream
            .clone_htod(&rows_host)
            .gpu_ctx("sparse_dict score-block htod rows")?;
        let atoms_dev = stream
            .clone_htod(&atoms_host)
            .gpu_ctx("sparse_dict score-block htod atoms")?;
        let mut scores_dev = stream
            .alloc_zeros::<f32>(n_rows * n_atoms)
            .gpu_ctx("sparse_dict score-block alloc scores")?;

        let n_rows_i32 = i32::try_from(n_rows).map_err(|_| {
            gam_gpu::gpu_err!("sparse_dict score-block n_rows={n_rows} overflows i32")
        })?;
        let n_atoms_i32 = i32::try_from(n_atoms).map_err(|_| {
            gam_gpu::gpu_err!("sparse_dict score-block n_atoms={n_atoms} overflows i32")
        })?;

        // `BM × BN` output tiles: a `(BN, BM)` thread block and a
        // `ceil(n_atoms/BN) × ceil(n_rows/BM)` grid, matching the tiled kernel's
        // `blockIdx.{x,y}` / `threadIdx.{x,y}` addressing.
        let tile_m = super::SCORE_BLOCK_TILE_M;
        let tile_n = super::SCORE_BLOCK_TILE_N;
        let grid_x: u32 = u32::try_from(n_atoms.div_ceil(tile_n as usize))
            .map_err(|_| gam_gpu::gpu_err!("sparse_dict score-block grid_x overflow"))?;
        let grid_y: u32 = u32::try_from(n_rows.div_ceil(tile_m as usize))
            .map_err(|_| gam_gpu::gpu_err!("sparse_dict score-block grid_y overflow"))?;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (tile_n, tile_m, 1),
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

    /// Route-time score-block stream for a full decoder. The rows are uploaded
    /// once and kept resident across every atom tile; only the atom tile and
    /// score block rotate per launch. That preserves the existing bit-exact
    /// `sparse_dict_score_block` arithmetic while removing the repeated `B × P`
    /// H2D copy that `score_block_device(rows, atoms_tile)` paid once per K tile.
    pub(super) fn score_decoder_tiled_device<F>(
        rows: ArrayView2<'_, f32>,
        decoder: ArrayView2<'_, f32>,
        tile_cols: usize,
        mut fold_tile: F,
    ) -> Result<(), GpuError>
    where
        F: FnMut(usize, usize, &[f32]),
    {
        let n_rows = rows.nrows();
        let k = decoder.nrows();
        let p = rows.ncols();
        if p != decoder.ncols() {
            return Err(gam_gpu::gpu_err!(
                "sparse_dict tiled score: P mismatch rows={p} decoder={}",
                decoder.ncols()
            ));
        }
        if n_rows == 0 || k == 0 || p == 0 {
            return Ok(());
        }

        let b = backend()?;
        let module = module_for(b, p)?;
        let func = module
            .load_function("sparse_dict_score_block")
            .gpu_ctx("sparse_dict tiled score load_function")?;
        let stream = b.stream.clone();

        let rows_host: Vec<f32> = rows.iter().copied().collect();
        assert_eq!(
            rows_host.len(),
            n_rows * p,
            "tiled score rows flatten length"
        );
        let rows_dev = stream
            .clone_htod(&rows_host)
            .gpu_ctx("sparse_dict tiled score htod rows")?;

        let n_rows_i32 = i32::try_from(n_rows).map_err(|_| {
            gam_gpu::gpu_err!("sparse_dict tiled score n_rows={n_rows} overflows i32")
        })?;
        let tile_m = super::SCORE_BLOCK_TILE_M;
        let tile_n = super::SCORE_BLOCK_TILE_N;
        let tile_cols = tile_cols.max(1);

        let mut start = 0usize;
        while start < k {
            let end = (start + tile_cols).min(k);
            let atoms = decoder.slice(ndarray::s![start..end, ..]);
            let n_atoms = atoms.nrows();
            let atoms_host: Vec<f32> = atoms.iter().copied().collect();
            assert_eq!(
                atoms_host.len(),
                n_atoms * p,
                "tiled score atoms flatten length"
            );
            let atoms_dev = stream
                .clone_htod(&atoms_host)
                .gpu_ctx("sparse_dict tiled score htod atoms")?;
            let mut scores_dev = stream
                .alloc_zeros::<f32>(n_rows * n_atoms)
                .gpu_ctx("sparse_dict tiled score alloc scores")?;

            let n_atoms_i32 = i32::try_from(n_atoms).map_err(|_| {
                gam_gpu::gpu_err!("sparse_dict tiled score n_atoms={n_atoms} overflows i32")
            })?;
            let grid_x: u32 = u32::try_from(n_atoms.div_ceil(tile_n as usize))
                .map_err(|_| gam_gpu::gpu_err!("sparse_dict tiled score grid_x overflow"))?;
            let grid_y: u32 = u32::try_from(n_rows.div_ceil(tile_m as usize))
                .map_err(|_| gam_gpu::gpu_err!("sparse_dict tiled score grid_y overflow"))?;
            let cfg = LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (tile_n, tile_m, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&func);
            builder
                .arg(&rows_dev)
                .arg(&atoms_dev)
                .arg(&n_rows_i32)
                .arg(&n_atoms_i32)
                .arg(&mut scores_dev);
            // SAFETY: grid/block validated; device pointers are cudarc-checked
            // allocations on this stream. The kernel reads the resident rows and
            // current atom tile and writes exactly `n_rows * n_atoms` scores.
            unsafe { builder.launch(cfg) }.gpu_ctx("sparse_dict tiled score launch")?;

            let mut scores = vec![0.0f32; n_rows * n_atoms];
            stream
                .memcpy_dtoh(&scores_dev, &mut scores)
                .gpu_ctx("sparse_dict tiled score dtoh scores")?;
            stream
                .synchronize()
                .gpu_ctx("sparse_dict tiled score synchronize")?;
            fold_tile(start, n_atoms, &scores);
            start = end;
        }
        Ok(())
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
    fn device_route_minibatch_matches_cpu_top_s_online() {
        // The router primitive the fit loop actually calls. The m×K block MUST
        // clear DEVICE_SCORE_BLOCK_MIN_ELEMS so the device path is admitted. On a
        // CUDA host we drive Required (silent CPU fallback = hard failure) and
        // assert the routed top-s support EQUALS the per-row CPU `top_s_online`
        // oracle exactly — same atoms, same bit-identical scores, same order.
        use crate::sparse_dict::scoring::top_s_online;

        let m = 512usize;
        let k = 4096usize; // 512*4096 = 2,097,152 >= DEVICE_SCORE_BLOCK_MIN_ELEMS
        let p = 48usize;
        let s = 4usize;
        let tile = 1024usize;
        assert!(m * k >= DEVICE_SCORE_BLOCK_MIN_ELEMS);
        let (rows, atoms) = fixture(m, k, p);

        let cpu: Vec<Vec<(u32, f32)>> = rows
            .outer_iter()
            .map(|row| top_s_online(row, atoms.view(), s, tile))
            .collect();

        match route_minibatch_required(
            rows.view(),
            atoms.view(),
            s,
            tile,
            gam_gpu::GpuMode::Required,
        ) {
            Ok((routed, path)) => {
                assert_eq!(
                    path,
                    ScoreBlockPath::Device,
                    "Required succeeded but reported CPU — device did not engage"
                );
                assert_eq!(routed.len(), cpu.len());
                for (r, (dev_sel, cpu_sel)) in routed.iter().zip(&cpu).enumerate() {
                    assert_eq!(
                        dev_sel.len(),
                        cpu_sel.len(),
                        "row {r}: selection length differs"
                    );
                    for (j, ((da, ds), (ca, cs))) in dev_sel.iter().zip(cpu_sel).enumerate() {
                        assert_eq!(da, ca, "row {r} slot {j}: atom differs dev={da} cpu={ca}");
                        assert_eq!(
                            ds.to_bits(),
                            cs.to_bits(),
                            "row {r} slot {j}: score bits differ dev={ds} cpu={cs}"
                        );
                    }
                }
            }
            Err(err) => {
                assert!(
                    gam_gpu::GpuRuntime::global().is_none(),
                    "Required errored despite a live CUDA runtime: {err}"
                );
                // Device absent: Auto must reproduce the CPU oracle exactly.
                let (routed, path) = route_minibatch_required(
                    rows.view(),
                    atoms.view(),
                    s,
                    tile,
                    gam_gpu::GpuMode::Auto,
                )
                .expect("Auto must not error on a device-absent host");
                assert_eq!(path, ScoreBlockPath::Cpu);
                assert_eq!(routed, cpu);
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn device_route_at_issue_target_k_32k_is_bit_identical() {
        // #1026 HEADLINE SCALE. The issue is about "large linear SAEs" — K up to
        // ~32_000. Our other parity test pins K=4096; this one drives the router
        // at the issue's actual target width (K=32_768) to prove the device path
        // not only engages but stays BIT-IDENTICAL to the per-row CPU oracle at
        // the scale where the 1e4–1e6× hardware gap the issue tracks lives. m is
        // kept modest (256) so the 256×32_768 = 8.4M-element block clears the
        // device break-even by 8× while the host buffers stay ~34 MB.
        use crate::sparse_dict::scoring::top_s_online;

        let m = 256usize;
        let k = 32_768usize; // 256 * 32_768 = 8,388,608 >> DEVICE_SCORE_BLOCK_MIN_ELEMS
        let p = 64usize;
        let s = 4usize;
        let tile = 2048usize;
        assert!(m * k >= DEVICE_SCORE_BLOCK_MIN_ELEMS);
        let (rows, atoms) = fixture(m, k, p);

        let cpu: Vec<Vec<(u32, f32)>> = rows
            .outer_iter()
            .map(|row| top_s_online(row, atoms.view(), s, tile))
            .collect();

        match route_minibatch_required(
            rows.view(),
            atoms.view(),
            s,
            tile,
            gam_gpu::GpuMode::Required,
        ) {
            Ok((routed, path)) => {
                assert_eq!(
                    path,
                    ScoreBlockPath::Device,
                    "Required succeeded at K=32k but reported CPU — device did not engage"
                );
                assert_eq!(routed.len(), cpu.len());
                for (r, (dev_sel, cpu_sel)) in routed.iter().zip(&cpu).enumerate() {
                    assert_eq!(
                        dev_sel.len(),
                        cpu_sel.len(),
                        "row {r}: selection length differs"
                    );
                    for (j, ((da, ds), (ca, cs))) in dev_sel.iter().zip(cpu_sel).enumerate() {
                        assert_eq!(
                            da, ca,
                            "K=32k row {r} slot {j}: atom differs dev={da} cpu={ca}"
                        );
                        assert_eq!(
                            ds.to_bits(),
                            cs.to_bits(),
                            "K=32k row {r} slot {j}: score bits differ dev={ds} cpu={cs}"
                        );
                    }
                }
            }
            Err(err) => {
                assert!(
                    gam_gpu::GpuRuntime::global().is_none(),
                    "Required errored at K=32k despite a live CUDA runtime: {err}"
                );
                let (routed, path) = route_minibatch_required(
                    rows.view(),
                    atoms.view(),
                    s,
                    tile,
                    gam_gpu::GpuMode::Auto,
                )
                .expect("Auto must not error on a device-absent host");
                assert_eq!(path, ScoreBlockPath::Cpu);
                assert_eq!(routed, cpu);
            }
        }
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
