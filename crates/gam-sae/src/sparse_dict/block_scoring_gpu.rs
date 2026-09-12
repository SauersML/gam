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
//! # Coding on the device (#2826)
//!
//! A pass also needs each row's codes: the support admitted from its shortlist by
//! descent in the tied loss, and the γ-free projections of the admitted blocks
//! (`super::block::code_row`). On the host that coder re-reads every shortlisted
//! atom and, in every admission round, dots the running reconstruction against
//! each remaining candidate: `O(k²·b·P)` f64 work per row, done after the device
//! has already finished its route and sits idle. [`route_and_code_blocks`] runs
//! the same coder on the device instead (`sparse_dict_block_code`, below), right
//! after the fold, reading the resident rows, decoder and shortlists. Only the
//! admitted blocks, their gates and their projections (`m × k × b`) are
//! downloaded. The kernel performs the host coder's operations in the same order
//! with separately rounded f64 arithmetic, so its codes equal the host coder's on
//! the same shortlists to the bit ([`code_block_shortlists_cpu`] is that oracle).
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

use ndarray::{Array2, Array3, ArrayView1, ArrayView2};

use super::block::{
    RowBlockCode, block_gates, block_projections_row, code_routed_rows, route_row_blocks,
};

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

/// The device block coder. One thread per row repeats `super::block::code_row` on
/// that row's resident shortlist: the orphan decision of
/// `super::block::code_routed_rows`, the γ-free candidate coordinates
/// `w_h = U_h x`, then support admission by descent in the tied loss, where round
/// one is unconditional and every later round admits the candidate with the most
/// negative `ΔL_h(S) = −(2γ−γ²) c_h + 2γ² (m · y_h)`, stopping when none lowers
/// the loss. Every accumulation runs in the host coder's order with separately
/// rounded f64 operations (the shared NVRTC options pin `--fmad=false`), so the
/// admitted blocks, their gates and their projections equal the host coder's to
/// the bit.
///
/// The kernel is appended to the score/fold source, so it shares that source's
/// `PP` and `EMPTY_TOP_ATOM` definitions. `candidates`, `coordinates` and
/// `reconstruction` are per-row scratch; `counts[row]` is the number of admitted
/// blocks, and output slots past it are left zero for the host to pad.
#[cfg(target_os = "linux")]
pub const BLOCK_CODE_KERNEL_SOURCE: &str = r#"
extern "C" __global__
void sparse_dict_block_code(
    const float* __restrict__ rows,               // [n_rows * PP] row-major
    const float* __restrict__ decoder,            // [(n_blocks*b) * PP] row-major
    const unsigned int* __restrict__ top_blocks,  // [n_rows * active] (gate desc, block asc)
    const float* __restrict__ top_gates,          // [n_rows * active]
    int n_rows,
    int active,
    int k,
    int b,
    float gamma,
    float projection_roundoff,
    int* __restrict__ candidates,                 // [n_rows * active] scratch
    double* __restrict__ coordinates,             // [n_rows * active * b] scratch
    double* __restrict__ reconstruction,          // [n_rows * PP] scratch
    int* __restrict__ counts,                     // [n_rows]
    unsigned int* __restrict__ out_blocks,        // [n_rows * active]
    float* __restrict__ out_gates,                // [n_rows * active]
    double* __restrict__ out_projections)         // [n_rows * active * b]
{
  const long long row = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
  if (row >= (long long)n_rows) return;
  const long long x0 = row * (long long)PP;
  const long long s0 = row * (long long)active;
  counts[row] = 0;

  // Orphan decision: the row norm accumulated in f64 in ascending c, rounded to
  // f32, times the host's projection roundoff. An orphan row keeps no block.
  double energy = 0.0;
  for (int c = 0; c < PP; ++c) {
    const double v = (double)rows[x0 + c];
    energy = energy + v * v;
  }
  const float floor_gate = (float)sqrt(energy) * projection_roundoff;
  const float best_gate = (top_blocks[s0] != EMPTY_TOP_ATOM) ? top_gates[s0] : 0.0f;
  if (best_gate < floor_gate) return;

  // Candidate coordinates in routed order. An empty slot or a zero gate is an
  // absent firing, not a candidate.
  int n_candidates = 0;
  for (int j = 0; j < active; ++j) {
    const unsigned int block = top_blocks[s0 + j];
    if (block == EMPTY_TOP_ATOM || top_gates[s0 + j] == 0.0f) continue;
    const long long w0 = (s0 + (long long)n_candidates) * (long long)b;
    for (int r = 0; r < b; ++r) {
      const long long a0 = ((long long)block * (long long)b + (long long)r) * (long long)PP;
      double projection = 0.0;
      for (int c = 0; c < PP; ++c) {
        projection = projection + (double)rows[x0 + c] * (double)decoder[a0 + c];
      }
      coordinates[w0 + r] = projection;
    }
    candidates[s0 + n_candidates] = j;
    ++n_candidates;
  }

  // Greedy admission against the γ-free reconstruction m of the admitted set.
  for (int c = 0; c < PP; ++c) reconstruction[x0 + c] = 0.0;
  const double g = (double)gamma;
  const double admission_scale = 2.0 * g - g * g;
  int admitted = 0;
  for (int admission_round = 0; admission_round < k; ++admission_round) {
    const int unconditional = (admission_round == 0);
    int best = -1;
    double best_gain = 0.0;
    for (int i = 0; i < n_candidates; ++i) {
      const int slot = candidates[s0 + i];
      if (slot < 0) continue;
      const unsigned int block = top_blocks[s0 + slot];
      const long long w0 = (s0 + (long long)i) * (long long)b;
      double own = 0.0;
      double overlap = 0.0;
      for (int r = 0; r < b; ++r) {
        const double coordinate = coordinates[w0 + r];
        own = own + coordinate * coordinate;
        const long long a0 = ((long long)block * (long long)b + (long long)r) * (long long)PP;
        double projected = 0.0;
        for (int c = 0; c < PP; ++c) {
          projected = projected + reconstruction[x0 + c] * (double)decoder[a0 + c];
        }
        overlap = overlap + projected * coordinate;
      }
      const double gain = (unconditional && admission_scale <= 0.0)
          ? -own
          : -admission_scale * own + 2.0 * g * g * overlap;
      if ((unconditional || gain < 0.0) && (best < 0 || gain < best_gain)) {
        best = i;
        best_gain = gain;
      }
    }
    if (best < 0) break;
    const int slot = candidates[s0 + best];
    candidates[s0 + best] = -1;
    const unsigned int block = top_blocks[s0 + slot];
    out_blocks[s0 + admitted] = block;
    out_gates[s0 + admitted] = top_gates[s0 + slot];
    const long long w0 = (s0 + (long long)best) * (long long)b;
    const long long o0 = (s0 + (long long)admitted) * (long long)b;
    for (int r = 0; r < b; ++r) {
      const double coordinate = coordinates[w0 + r];
      out_projections[o0 + r] = coordinate;
      if (coordinate != 0.0) {
        const long long a0 = ((long long)block * (long long)b + (long long)r) * (long long)PP;
        for (int c = 0; c < PP; ++c) {
          reconstruction[x0 + c] = reconstruction[x0 + c] + coordinate * (double)decoder[a0 + c];
        }
      }
    }
    ++admitted;
  }
  counts[row] = admitted;
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
/// `super::block::route_block_minibatch` up to f32 ties (that path forms `z`
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

/// CPU oracle for the block coder: code each row's given `(block, gate)`
/// shortlist exactly as a CPU-routed pass does, packed to width `k` as
/// `(blocks[m,k], gates[m,k], projections[m,k,b])`. Each row lists its admitted
/// blocks in admission order with their routed gates and γ-free projections,
/// padded with block 0 and zeros. On the device's own shortlists the device
/// coder (`route_and_code_blocks`) must reproduce it to the bit.
#[must_use]
pub fn code_block_shortlists_cpu(
    rows: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    gamma: f32,
    b: usize,
    k: usize,
    shortlists: Vec<Vec<(u32, f32)>>,
) -> (Array2<u32>, Array2<f32>, Array3<f64>) {
    assert_eq!(
        shortlists.len(),
        rows.nrows(),
        "code_block_shortlists_cpu needs one shortlist per row"
    );
    pack_row_codes(
        &code_routed_rows(rows, decoder, gamma, b, k, shortlists),
        k,
        b,
    )
}

/// Fixed-width `(blocks, gates, projections)` arrays from per-row codes, each
/// already padded to width `k`.
fn pack_row_codes(
    codes: &[RowBlockCode],
    k: usize,
    b: usize,
) -> (Array2<u32>, Array2<f32>, Array3<f64>) {
    let n = codes.len();
    let mut blocks = Array2::<u32>::zeros((n, k));
    let mut gates = Array2::<f32>::zeros((n, k));
    let mut projections = Array3::<f64>::zeros((n, k, b));
    for (i, code) in codes.iter().enumerate() {
        for j in 0..k {
            blocks[[i, j]] = code.blocks[j];
            gates[[i, j]] = code.gates[j];
            for r in 0..b {
                projections[[i, j, r]] = code.projections[j * b + r];
            }
        }
    }
    (blocks, gates, projections)
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

/// Block count `G = K / b` of a decoder with `K` rows.
#[cfg(target_os = "linux")]
fn block_count(krows: usize, b: usize) -> Result<usize, gam_gpu::GpuError> {
    if b == 0 || krows == 0 || krows % b != 0 {
        return Err(gam_gpu::gpu_err!(
            "block-gate route: decoder K={krows} rows not a positive multiple of block_size b={b}"
        ));
    }
    Ok(krows / b)
}

/// Production CPU router for one minibatch: the blocked-GEMM router — the same
/// top-`k` support as the scalar oracle (`route_blocks_cpu`, which stays as the
/// device parity reference) but ~2 orders of magnitude faster. Under the default
/// `Auto` policy every below-break-even minibatch and every CUDA-less Linux host
/// lands here, so this IS the hot CPU path (#2242: the scalar per-row oracle was
/// burning 75% of block-lane cycles).
#[cfg(target_os = "linux")]
fn cpu_block_route(
    rows: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    g: usize,
    b: usize,
    active: usize,
) -> Vec<Vec<(u32, f32)>> {
    let cpu_tile_blocks =
        (GPU_BLOCK_ROUTE_TILE_ELEMS / (rows.nrows().max(1) * b.max(1))).clamp(1, g.max(1));
    super::block::route_block_minibatch(rows, decoder, g, b, active, cpu_tile_blocks)
}

/// Where one minibatch's block route runs.
#[cfg(target_os = "linux")]
enum BlockRouteAdmission {
    Cpu,
    /// On the device, walking `G` in launches of `tile_blocks` blocks.
    Device { tile_blocks: usize },
}

/// Decide where one minibatch routes under `mode`. `Off` always takes the CPU.
/// `Auto` takes the device when the shape clears break-even and a CUDA runtime
/// resolves, and reports why when it declines. `Required` refuses every decline.
#[cfg(target_os = "linux")]
fn admit_block_route(
    rows: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    g: usize,
    b: usize,
    mode: gam_gpu::GpuPolicy,
) -> Result<BlockRouteAdmission, gam_gpu::GpuError> {
    use gam_gpu::GpuPolicy;

    if mode == GpuPolicy::Off {
        return Ok(BlockRouteAdmission::Cpu);
    }
    let m = rows.nrows();
    let krows = decoder.nrows();

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
        gam_gpu::engagement::note_route_engagement(
            "gam-sae sparse_dict block-gate router",
            "falling back to CPU",
            false,
            &format!(
                "block {m}x{krows} = {} elems below the device launch break-even \
                 (DEVICE_BLOCK_GATE_MIN_ELEMS={DEVICE_BLOCK_GATE_MIN_ELEMS})",
                m.saturating_mul(krows)
            ),
        );
        return Ok(BlockRouteAdmission::Cpu);
    }
    if m == 0 || g == 0 {
        return Ok(BlockRouteAdmission::Cpu);
    }

    let runtime = if mode == GpuPolicy::Required {
        Some(gam_gpu::GpuRuntime::require()?)
    } else {
        gam_gpu::GpuRuntime::resolve(mode)?
    };
    if runtime.is_none() {
        gam_gpu::engagement::note_route_engagement(
            "gam-sae sparse_dict block-gate router",
            "falling back to CPU",
            false,
            "Auto admission found no CUDA device",
        );
        return Ok(BlockRouteAdmission::Cpu);
    }

    // Blocks per launch: bound the per-launch `z` block `m × (tile_blocks·b)` to
    // GPU_BLOCK_ROUTE_TILE_ELEMS, at least one block, never more than G.
    let tile_blocks = (plan.tile_items / b.max(1)).clamp(1, g);
    Ok(BlockRouteAdmission::Device { tile_blocks })
}

#[cfg(target_os = "linux")]
fn note_device_route_engaged(
    rows: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    tile_blocks: usize,
    active: usize,
    coded_on_device: bool,
) {
    gam_gpu::engagement::note_route_engagement(
        "gam-sae sparse_dict block-gate router",
        "falling back to CPU",
        true,
        &format!(
            "block {}x{}, tile_blocks={tile_blocks}, active={active}, \
             coded_on_device={coded_on_device}",
            rows.nrows(),
            decoder.nrows()
        ),
    );
}

#[cfg(target_os = "linux")]
pub fn route_blocks_required(
    rows: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    b: usize,
    k: usize,
    mode: gam_gpu::GpuPolicy,
) -> Result<(Vec<Vec<(u32, f32)>>, BlockRoutePath, usize), gam_gpu::GpuError> {
    let g = block_count(decoder.nrows(), b)?;
    let active = k.max(1).min(g.max(1));
    match admit_block_route(rows, decoder, g, b, mode)? {
        BlockRouteAdmission::Cpu => Ok((
            cpu_block_route(rows, decoder, g, b, active),
            BlockRoutePath::Cpu,
            0,
        )),
        BlockRouteAdmission::Device { tile_blocks } => {
            let out = device::route_blocks_device(rows, decoder, b, g, active, tile_blocks)?;
            note_device_route_engaged(rows, decoder, tile_blocks, active, false);
            Ok((
                out.selections,
                BlockRoutePath::Device,
                out.device_dtoh_bytes,
            ))
        }
    }
}

/// Route and code one minibatch under `mode`, returning each row's codes padded
/// to width `k`. A device route codes on the device ([`BLOCK_CODE_KERNEL_SOURCE`]);
/// a CPU route codes on the host (`super::block::code_routed_rows`).
#[cfg(target_os = "linux")]
pub(super) fn route_and_code_blocks(
    rows: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    gamma: f32,
    b: usize,
    k: usize,
    mode: gam_gpu::GpuPolicy,
) -> Result<(Vec<RowBlockCode>, BlockRoutePath), gam_gpu::GpuError> {
    let g = block_count(decoder.nrows(), b)?;
    let active = k.max(1).min(g.max(1));
    match admit_block_route(rows, decoder, g, b, mode)? {
        BlockRouteAdmission::Cpu => {
            let routed = cpu_block_route(rows, decoder, g, b, active);
            Ok((
                code_routed_rows(rows, decoder, gamma, b, k, routed),
                BlockRoutePath::Cpu,
            ))
        }
        BlockRouteAdmission::Device { tile_blocks } => {
            let codes = device::route_and_code_blocks_device(
                rows,
                decoder,
                gamma,
                b,
                g,
                active,
                k,
                tile_blocks,
            )?;
            note_device_route_engaged(rows, decoder, tile_blocks, active, true);
            Ok((codes, BlockRoutePath::Device))
        }
    }
}

/// `route_and_code_blocks` packed to fixed width like
/// [`code_block_shortlists_cpu`], with the path that ran. This is the entry the
/// device parity gate drives with an explicit policy.
#[cfg(target_os = "linux")]
pub fn route_and_code_blocks_required(
    rows: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    gamma: f32,
    b: usize,
    k: usize,
    mode: gam_gpu::GpuPolicy,
) -> Result<((Array2<u32>, Array2<f32>, Array3<f64>), BlockRoutePath), gam_gpu::GpuError> {
    let (codes, path) = route_and_code_blocks(rows, decoder, gamma, b, k, mode)?;
    Ok((pack_row_codes(&codes, k, b), path))
}

#[cfg(target_os = "linux")]
mod device {
    use gam_gpu::backend_probe::CachedBackend;
    use gam_gpu::gpu_error::{GpuError, GpuResultExt};
    use ndarray::ArrayView2;
    use std::sync::Arc;

    use cudarc::driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

    use super::super::block::{RowBlockCode, code_routed_rows, orphan_projection_roundoff};
    use super::super::score_router_backend::ScoreRouterBackend as Backend;

    static BACKEND: CachedBackend<Backend> = CachedBackend::new();

    fn backend() -> Result<&'static Backend, GpuError> {
        BACKEND.get_or_probe("sparse_dict_block_gate", Backend::from_parts)
    }

    /// Combined NVRTC source: the atom lane's bit-exact score GEMM + top-`s` fold
    /// ([`super::super::scoring_gpu::score_block_kernel_source`], which bakes `PP`)
    /// plus this lane's ℓ₂-gate epilogue ([`super::BLOCK_GATE_KERNEL_SOURCE`]) and
    /// block coder ([`super::BLOCK_CODE_KERNEL_SOURCE`]).
    fn combined_kernel_source(p: usize) -> String {
        format!(
            "{}\n{}\n{}",
            super::super::scoring_gpu::score_block_kernel_source(p),
            super::BLOCK_GATE_KERNEL_SOURCE,
            super::BLOCK_CODE_KERNEL_SOURCE
        )
    }

    fn module_for(b: &Backend, p: usize) -> Result<Arc<CudaModule>, GpuError> {
        b.modules
            .get_or_compile(&b.ctx, p, "sparse_dict block-gate", combined_kernel_source)
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

    /// A routed minibatch still on the device: the resident rows and decoder and
    /// each row's folded `(block, gate)` shortlist, sorted by `(gate desc, block
    /// asc)` with empty slots at the tail.
    struct ResidentRoute {
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        rows_dev: CudaSlice<f32>,
        decoder_dev: CudaSlice<f32>,
        top_blocks_dev: CudaSlice<u32>,
        top_gates_dev: CudaSlice<f32>,
        m: usize,
        p: usize,
        active: usize,
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

    /// Route a whole minibatch's blocks on the device and leave the result there:
    /// rows and the whole decoder stay resident; per block-tile one score GEMM
    /// forms `z` (`m × tile_blocks·b`), the gate epilogue reduces it to
    /// `m × tile_blocks` gates, and the resident top-`k` fold folds those into
    /// per-row `(block, gate)` shortlists. `None` for a shape with no rows,
    /// blocks or features, where every shortlist is empty.
    fn route_resident(
        rows: ArrayView2<'_, f32>,
        decoder: ArrayView2<'_, f32>,
        b: usize,
        n_blocks: usize,
        active: usize,
        tile_blocks: usize,
    ) -> Result<Option<ResidentRoute>, GpuError> {
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
            return Ok(None);
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

        Ok(Some(ResidentRoute {
            module,
            stream,
            rows_dev,
            decoder_dev,
            top_blocks_dev,
            top_gates_dev,
            m,
            p,
            active,
        }))
    }

    /// Route a whole minibatch's blocks on the device ([`route_resident`]) and
    /// download only the final `m × k` shortlists.
    pub(super) fn route_blocks_device(
        rows: ArrayView2<'_, f32>,
        decoder: ArrayView2<'_, f32>,
        b: usize,
        n_blocks: usize,
        active: usize,
        tile_blocks: usize,
    ) -> Result<BlockRouteDeviceOutput, GpuError> {
        let Some(route) = route_resident(rows, decoder, b, n_blocks, active, tile_blocks)? else {
            return Ok(BlockRouteDeviceOutput {
                selections: vec![Vec::new(); rows.nrows()],
                device_dtoh_bytes: 0,
            });
        };
        let (m, active, stream) = (route.m, route.active, &route.stream);

        let mut top_blocks = vec![0u32; m * active];
        let mut top_gates = vec![0.0f32; m * active];
        stream
            .memcpy_dtoh(&route.top_blocks_dev, &mut top_blocks)
            .gpu_ctx("sparse_dict block-gate dtoh blocks")?;
        stream
            .memcpy_dtoh(&route.top_gates_dev, &mut top_gates)
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

    /// Route a whole minibatch on the device ([`route_resident`]), then code every
    /// row there with [`super::BLOCK_CODE_KERNEL_SOURCE`] against the resident rows,
    /// decoder and shortlists. Only each row's admitted blocks, gates and γ-free
    /// projections are downloaded; the host forms the γ-scaled codes and pads to
    /// width `k` exactly as `super::super::block::code_row` does.
    pub(super) fn route_and_code_blocks_device(
        rows: ArrayView2<'_, f32>,
        decoder: ArrayView2<'_, f32>,
        gamma: f32,
        b: usize,
        n_blocks: usize,
        active: usize,
        k: usize,
        tile_blocks: usize,
    ) -> Result<Vec<RowBlockCode>, GpuError> {
        let Some(route) = route_resident(rows, decoder, b, n_blocks, active, tile_blocks)? else {
            return Ok(code_routed_rows(
                rows,
                decoder,
                gamma,
                b,
                k,
                vec![Vec::new(); rows.nrows()],
            ));
        };
        let ResidentRoute {
            module,
            stream,
            rows_dev,
            decoder_dev,
            top_blocks_dev,
            top_gates_dev,
            m,
            p,
            active,
        } = route;

        let code_func = module
            .load_function("sparse_dict_block_code")
            .gpu_ctx("sparse_dict block-code load_function")?;
        let m_i32 =
            i32::try_from(m).map_err(|_| gam_gpu::gpu_err!("block-code m={m} overflows i32"))?;
        let active_i32 = i32::try_from(active)
            .map_err(|_| gam_gpu::gpu_err!("block-code active={active} overflows i32"))?;
        let k_i32 =
            i32::try_from(k).map_err(|_| gam_gpu::gpu_err!("block-code k={k} overflows i32"))?;
        let b_i32 =
            i32::try_from(b).map_err(|_| gam_gpu::gpu_err!("block-code b={b} overflows i32"))?;
        let roundoff = orphan_projection_roundoff(p, b);
        let slots = m * active;
        let coefficients = slots
            .checked_mul(b)
            .ok_or_else(|| gam_gpu::gpu_err!("block-code m*active*b overflows usize"))?;

        let mut candidates_dev = stream
            .alloc_zeros::<i32>(slots)
            .gpu_ctx("sparse_dict block-code alloc candidates")?;
        let mut coordinates_dev = stream
            .alloc_zeros::<f64>(coefficients)
            .gpu_ctx("sparse_dict block-code alloc coordinates")?;
        let mut reconstruction_dev = stream
            .alloc_zeros::<f64>(m * p)
            .gpu_ctx("sparse_dict block-code alloc reconstruction")?;
        let mut counts_dev = stream
            .alloc_zeros::<i32>(m)
            .gpu_ctx("sparse_dict block-code alloc counts")?;
        let mut out_blocks_dev = stream
            .alloc_zeros::<u32>(slots)
            .gpu_ctx("sparse_dict block-code alloc blocks")?;
        let mut out_gates_dev = stream
            .alloc_zeros::<f32>(slots)
            .gpu_ctx("sparse_dict block-code alloc gates")?;
        let mut out_projections_dev = stream
            .alloc_zeros::<f64>(coefficients)
            .gpu_ctx("sparse_dict block-code alloc projections")?;

        // One thread per row, launched at the gate epilogue's block width.
        let code_cfg = LaunchConfig {
            grid_dim: (
                u32::try_from(m.div_ceil(GATE_KERNEL_THREADS as usize))
                    .map_err(|_| gam_gpu::gpu_err!("block-code grid overflow"))?,
                1,
                1,
            ),
            block_dim: (GATE_KERNEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut code = stream.launch_builder(&code_func);
        code.arg(&rows_dev)
            .arg(&decoder_dev)
            .arg(&top_blocks_dev)
            .arg(&top_gates_dev)
            .arg(&m_i32)
            .arg(&active_i32)
            .arg(&k_i32)
            .arg(&b_i32)
            .arg(&gamma)
            .arg(&roundoff)
            .arg(&mut candidates_dev)
            .arg(&mut coordinates_dev)
            .arg(&mut reconstruction_dev)
            .arg(&mut counts_dev)
            .arg(&mut out_blocks_dev)
            .arg(&mut out_gates_dev)
            .arg(&mut out_projections_dev);
        // SAFETY: one thread per row within m. Each thread reads its own row, the
        // resident decoder and its own m*active shortlist slots, and writes only its
        // own candidate, coordinate, reconstruction, count and output ranges, all
        // cudarc-checked allocations of the sizes the kernel indexes.
        unsafe { code.launch(code_cfg) }.gpu_ctx("sparse_dict block-code launch")?;

        let mut counts = vec![0i32; m];
        let mut out_blocks = vec![0u32; slots];
        let mut out_gates = vec![0.0f32; slots];
        let mut out_projections = vec![0.0f64; coefficients];
        stream
            .memcpy_dtoh(&counts_dev, &mut counts)
            .gpu_ctx("sparse_dict block-code dtoh counts")?;
        stream
            .memcpy_dtoh(&out_blocks_dev, &mut out_blocks)
            .gpu_ctx("sparse_dict block-code dtoh blocks")?;
        stream
            .memcpy_dtoh(&out_gates_dev, &mut out_gates)
            .gpu_ctx("sparse_dict block-code dtoh gates")?;
        stream
            .memcpy_dtoh(&out_projections_dev, &mut out_projections)
            .gpu_ctx("sparse_dict block-code dtoh projections")?;
        stream
            .synchronize()
            .gpu_ctx("sparse_dict block-code synchronize")?;

        let gamma64 = gamma as f64;
        let admissible = active.min(k);
        let mut codes = Vec::with_capacity(m);
        for (r, &count) in counts.iter().enumerate() {
            let admitted = usize::try_from(count)
                .ok()
                .filter(|&admitted| admitted <= admissible)
                .ok_or_else(|| {
                    gam_gpu::gpu_err!(
                        "sparse_dict block-code row {r} returned admitted count {count}, outside \
                         0..={admissible}"
                    )
                })?;
            let mut blocks = Vec::with_capacity(k);
            let mut gates = Vec::with_capacity(k);
            let mut code_values = Vec::with_capacity(k * b);
            let mut projections = Vec::with_capacity(k * b);
            for slot in r * active..r * active + admitted {
                blocks.push(out_blocks[slot]);
                gates.push(out_gates[slot]);
                for &coordinate in &out_projections[slot * b..(slot + 1) * b] {
                    projections.push(coordinate);
                    code_values.push((gamma64 * coordinate) as f32);
                }
            }
            while blocks.len() < k {
                blocks.push(0);
                gates.push(0.0);
                for _ in 0..b {
                    code_values.push(0.0);
                    projections.push(0.0);
                }
            }
            codes.push(RowBlockCode {
                blocks,
                gates,
                codes: code_values,
                projections,
            });
        }
        Ok(codes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

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

}
