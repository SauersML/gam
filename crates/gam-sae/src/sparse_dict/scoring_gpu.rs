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
//! This computes one atom-tile's `rows × tile` score block on the device, then
//! folds that tile into per-row top-`s` state on the device before moving to the
//! next tile. The minibatch router [`route_minibatch_required`] walks the whole
//! `K`-wide dictionary in atom-column tiles (each launch's block capped at
//! `GPU_ROUTE_TILE_ELEMS`), so peak device score memory is `rows × tile`,
//! **independent of `K`**. The host downloads only the final `rows × s`
//! `(atom, score)` shortlists instead of every score tile. The lane's no-`N×K`
//! memory discipline is preserved exactly as on the CPU; the GPU does both the
//! `O(rows·tile·P)` multiply-accumulate and the high-K top-`s` scan.
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
//! Because the scores are identical and the device fold uses the same
//! `(|score| desc, atom asc)` ordering as [`super::scoring::TopSSelector`], the
//! routed support is IDENTICAL to the CPU oracle — parity is exact by
//! construction, not bounded by a tolerance.

#![cfg(target_os = "linux")]

use ndarray::ArrayView2;

/// The bit-exact-parity NVRTC kernel. A `BM × BN` output tile per CUDA block;
/// each thread owns one `(row, atom)` output and accumulates over `P` columns in
/// ascending order with separate-rounding f32 ops so the result matches the CPU
/// sequential `acc += x·d` to the bit.
///
/// The row and atom operands for the tile are cooperatively staged into fixed
/// `PK`-wide shared-memory chunks. This keeps per-block shared memory bounded by
/// `(BM + BN) * PK * sizeof(float)` even when the live T1 feature width is large
/// (`P=2048`), while every output still visits chunks and columns in ascending
/// `c`. The arithmetic is unchanged — each term is summed with
/// `__fmul_rn`/`__fadd_rn`; shared memory only holds exact copies of the same
/// operands, so the CPU-oracle parity gate is preserved by construction.
///
/// `PP` (the column count) is baked in as a `#define` so the inner loop is a
/// fixed trip count (matching the other NVRTC kernels in this repo, which
/// monomorphise their shape macros for a pure `compile_ptx`). `BM`/`BN` are the
/// output-tile dimensions; the host launch (`score_block_device`) must use a
/// `(BN, BM)` block and a `ceil(n_atoms/BN) × ceil(n_rows/BM)` grid.
pub const SCORE_BLOCK_KERNEL_SOURCE: &str = r#"
// Register-blocked score GEMM. A block computes a BM×BN output tile with a
// TM×TN thread block, each thread owning an RM×RN micro-tile of outputs
// (RM=BM/TM, RN=BN/TN). Every output still accumulates its own dot product in
// strictly ascending c with SEPARATE-rounding f32 ops (__fmul_rn/__fadd_rn, no
// FMA contraction), so each result is bit-identical to the CPU
// `acc += x[c]*d[c]` reference and to the earlier one-output-per-thread kernel.
// The micro-tile buys (a) RM*RN independent accumulator chains per thread to
// hide the serial __fadd_rn latency the old kernel was bound by, and (b) operand
// reuse — each staged row/atom column is consumed RN/RM times from registers
// instead of re-read from shared per output.
#define BM 64
#define BN 64
#define TM 16
#define TN 16
#define RM (BM / TM)
#define RN (BN / TN)
#define PK 32

static __device__ __forceinline__
void sparse_dict_score_block_impl(
    const float* __restrict__ rows,    // [n_rows * PP] row-major
    const float* __restrict__ atoms,   // [total_atoms * PP] row-major decoder
    int n_rows,
    int n_atoms,
    unsigned int atom_offset,          // decoder slice base (0 for the tile form)
    float* __restrict__ scores)        // [n_rows * n_atoms] row-major
{
  // Shared operand chunks: BM rows and BN atoms, each PK columns long. Chunking
  // keeps shared memory fixed-size for high-P jobs while preserving ascending-c
  // accumulation order (the acc registers persist across chunks).
  __shared__ float sr[BM][PK];
  __shared__ float sa[BN][PK];
  const int row0  = blockIdx.y * BM;
  const int atom0 = blockIdx.x * BN;
  const int tx = threadIdx.x;   // 0..TN-1
  const int ty = threadIdx.y;   // 0..TM-1
  const int lin = ty * TN + tx;
  const int nthreads = TM * TN;
  // This thread owns outputs rows [row0 + ty*RM, +RM) × atoms [atom0 + tx*RN, +RN).
  float acc[RM][RN];
  #pragma unroll
  for (int i = 0; i < RM; ++i)
    #pragma unroll
    for (int j = 0; j < RN; ++j) acc[i][j] = 0.0f;
  for (int c0 = 0; c0 < PP; c0 += PK) {
    const int chunk = (PP - c0 < PK) ? (PP - c0) : PK;
    // Cooperative, coalesced load of one P-chunk of row/atom operands
    // (zero-padded past ragged row/atom/chunk tails; +0.0 is an exact no-op in
    // the ascending-c accumulation so parity holds on padded lanes).
    for (int e = lin; e < BM * PK; e += nthreads) {
      int rr = e / PK, kc = e - rr * PK;
      int gr = row0 + rr;
      int cc = c0 + kc;
      sr[rr][kc] = (gr < n_rows && kc < chunk) ? rows[(long long)gr * PP + cc] : 0.0f;
    }
    for (int e = lin; e < BN * PK; e += nthreads) {
      int aa = e / PK, kc = e - aa * PK;
      int ga = atom0 + aa;
      int cc = c0 + kc;
      unsigned int global_atom = atom_offset + (unsigned int)ga;
      sa[aa][kc] = (ga < n_atoms && kc < chunk) ? atoms[(long long)global_atom * PP + cc] : 0.0f;
    }
    __syncthreads();
    for (int kc = 0; kc < chunk; ++kc) {
      // Stage this column's RM row-fragments and RN atom-fragments into
      // registers, then cross them: RM*RN separate-rounding MACs reusing 8 loads.
      float rf[RM];
      float af[RN];
      #pragma unroll
      for (int i = 0; i < RM; ++i) rf[i] = sr[ty * RM + i][kc];
      #pragma unroll
      for (int j = 0; j < RN; ++j) af[j] = sa[tx * RN + j][kc];
      #pragma unroll
      for (int i = 0; i < RM; ++i)
        #pragma unroll
        for (int j = 0; j < RN; ++j)
          acc[i][j] = __fadd_rn(acc[i][j], __fmul_rn(rf[i], af[j]));
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < RM; ++i) {
    int r = row0 + ty * RM + i;
    if (r >= n_rows) continue;
    #pragma unroll
    for (int j = 0; j < RN; ++j) {
      int a = atom0 + tx * RN + j;
      if (a < n_atoms) scores[(long long)r * n_atoms + a] = acc[i][j];
    }
  }
}

extern "C" __global__
void sparse_dict_score_block(
    const float* __restrict__ rows,    // [n_rows * PP] row-major
    const float* __restrict__ atoms,   // [n_atoms * PP] row-major (decoder tile)
    int n_rows,
    int n_atoms,
    float* __restrict__ scores)        // [n_rows * n_atoms] row-major
{
  sparse_dict_score_block_impl(rows, atoms, n_rows, n_atoms, 0u, scores);
}

extern "C" __global__
void sparse_dict_score_block_offset(
    const float* __restrict__ rows,    // [n_rows * PP] row-major
    const float* __restrict__ atoms,   // [total_atoms * PP] row-major decoder
    int n_rows,
    int n_atoms,
    unsigned int atom_offset,
    float* __restrict__ scores)        // [n_rows * n_atoms] row-major tile
{
  sparse_dict_score_block_impl(rows, atoms, n_rows, n_atoms, atom_offset, scores);
}

#define EMPTY_TOP_ATOM 0xffffffffu

static __device__ __forceinline__
float sparse_dict_abs_f32(float v) {
  return (v < 0.0f) ? -v : v;
}

static __device__ __forceinline__
int sparse_dict_better(float mag, unsigned int atom,
                       float ref_mag, unsigned int ref_atom) {
  return (mag > ref_mag) || (mag == ref_mag && atom < ref_atom);
}

static __device__ __forceinline__
int sparse_dict_worse(float mag, unsigned int atom,
                      float ref_mag, unsigned int ref_atom) {
  return (mag < ref_mag) || (mag == ref_mag && atom > ref_atom);
}

static __device__ __forceinline__
void sparse_dict_recompute_worst(const unsigned int* atoms,
                                 const float* mags,
                                 int count,
                                 int* worst_idx) {
  int worst = 0;
  for (int j = 1; j < count; ++j) {
    if (sparse_dict_worse(mags[j], atoms[j], mags[worst], atoms[worst])) {
      worst = j;
    }
  }
  *worst_idx = worst;
}

static __device__ __forceinline__
void sparse_dict_offer_top_s(unsigned int* atoms,
                             float* scores,
                             float* mags,
                             int active,
                             unsigned int atom,
                             float score,
                             int* count,
                             int* worst_idx) {
  if (active <= 0) {
    return;
  }
  const float mag = sparse_dict_abs_f32(score);
  if (*count < active) {
    const int slot = *count;
    atoms[slot] = atom;
    scores[slot] = score;
    mags[slot] = mag;
    *count = slot + 1;
    if (*count == active) {
      sparse_dict_recompute_worst(atoms, mags, *count, worst_idx);
    }
    return;
  }
  const int worst = *worst_idx;
  if (sparse_dict_better(mag, atom, mags[worst], atoms[worst])) {
    atoms[worst] = atom;
    scores[worst] = score;
    mags[worst] = mag;
    sparse_dict_recompute_worst(atoms, mags, active, worst_idx);
  }
}

static __device__ __forceinline__
void sparse_dict_sort_top_s(unsigned int* atoms,
                            float* scores,
                            float* mags,
                            int active,
                            int count) {
  for (int i = 1; i < count; ++i) {
    const unsigned int atom = atoms[i];
    const float score = scores[i];
    const float mag = mags[i];
    int j = i;
    while (j > 0 && sparse_dict_better(mag, atom, mags[j - 1], atoms[j - 1])) {
      atoms[j] = atoms[j - 1];
      scores[j] = scores[j - 1];
      mags[j] = mags[j - 1];
      --j;
    }
    atoms[j] = atom;
    scores[j] = score;
    mags[j] = mag;
  }
  for (int j = count; j < active; ++j) {
    atoms[j] = EMPTY_TOP_ATOM;
    scores[j] = 0.0f;
    mags[j] = -1.0f;
  }
}

extern "C" __global__
void sparse_dict_fold_top_s(
    const float* __restrict__ scores,  // [n_rows * n_atoms] current tile
    int n_rows,
    int n_atoms,
    unsigned int atom_offset,
    int active,
    unsigned int* __restrict__ top_atoms, // [n_rows * active]
    float* __restrict__ top_scores,       // [n_rows * active]
    float* __restrict__ top_mags)         // [n_rows * active]
{
  const int row = blockIdx.x;
  if (row >= n_rows || active <= 0) {
    return;
  }
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int candidate_slots = nthreads * active;

  extern __shared__ unsigned char smem[];
  unsigned int* cand_atoms = (unsigned int*)smem;
  unsigned int* best_atoms = cand_atoms + candidate_slots;
  float* cand_scores = (float*)(best_atoms + active);
  float* best_scores = cand_scores + candidate_slots;
  float* cand_mags = best_scores + active;
  float* best_mags = cand_mags + candidate_slots;

  const int local_base = tid * active;
  for (int j = 0; j < active; ++j) {
    cand_atoms[local_base + j] = EMPTY_TOP_ATOM;
    cand_scores[local_base + j] = 0.0f;
    cand_mags[local_base + j] = -1.0f;
  }
  if (tid == 0) {
    for (int j = 0; j < active; ++j) {
      best_atoms[j] = EMPTY_TOP_ATOM;
      best_scores[j] = 0.0f;
      best_mags[j] = -1.0f;
    }
  }
  __syncthreads();

  int local_count = 0;
  int local_worst = 0;
  unsigned int* local_atoms = cand_atoms + local_base;
  float* local_scores = cand_scores + local_base;
  float* local_mags = cand_mags + local_base;
  const long long row_base = (long long)row * n_atoms;
  for (int atom = tid; atom < n_atoms; atom += nthreads) {
    const float score = scores[row_base + atom];
    sparse_dict_offer_top_s(
        local_atoms,
        local_scores,
        local_mags,
        active,
        atom_offset + (unsigned int)atom,
        score,
        &local_count,
        &local_worst);
  }
  __syncthreads();

  if (tid == 0) {
    int best_count = 0;
    int best_worst = 0;
    const long long out_base = (long long)row * active;
    if (atom_offset != 0u) {
      for (int j = 0; j < active; ++j) {
        const unsigned int atom = top_atoms[out_base + j];
        if (atom != EMPTY_TOP_ATOM) {
          sparse_dict_offer_top_s(
              best_atoms,
              best_scores,
              best_mags,
              active,
              atom,
              top_scores[out_base + j],
              &best_count,
              &best_worst);
        }
      }
    }
    for (int t = 0; t < nthreads; ++t) {
      const int base = t * active;
      for (int j = 0; j < active; ++j) {
        const unsigned int atom = cand_atoms[base + j];
        if (atom != EMPTY_TOP_ATOM) {
          sparse_dict_offer_top_s(
              best_atoms,
              best_scores,
              best_mags,
              active,
              atom,
              cand_scores[base + j],
              &best_count,
              &best_worst);
        }
      }
    }
    sparse_dict_sort_top_s(best_atoms, best_scores, best_mags, active, best_count);
    for (int j = 0; j < active; ++j) {
      top_atoms[out_base + j] = best_atoms[j];
      top_scores[out_base + j] = best_scores[j];
      top_mags[out_base + j] = best_mags[j];
    }
  }
}
"#;

/// Output-tile dimensions the [`SCORE_BLOCK_KERNEL_SOURCE`] kernel is written
/// for (`BM`/`BN`); the host grid uses them to tile the `n_rows × n_atoms`
/// output. Kept in sync with the `#define`s at the top of the kernel string.
pub const SCORE_BLOCK_TILE_M: u32 = 64;
pub const SCORE_BLOCK_TILE_N: u32 = 64;

/// Thread-block dimensions (`TM`/`TN`) for the register-blocked kernel: each
/// thread owns an `(BM/TM) × (BN/TN)` micro-tile of outputs, so the launch uses
/// a `TN × TM` thread block over the `BM × BN` output tile. Kept in sync with the
/// `#define`s at the top of the kernel string.
pub const SCORE_BLOCK_THREADS_M: u32 = 16;
pub const SCORE_BLOCK_THREADS_N: u32 = 16;

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
/// device score buffer), then discards it after folding. This keeps peak score
/// memory bounded **independent of `K`** — the same discipline the CPU lane
/// ([`super::scoring::top_s_online`]) keeps with its `rows × tile` column tiles —
/// so a `K ≈ 32_000` fit does not balloon a `device alloc` linearly in `K`.
const GPU_ROUTE_TILE_ELEMS: usize = gam_gpu::DEFAULT_DICTIONARY_SCORE_TILE_ELEMS;

/// Route a whole minibatch of rows against the full decoder, returning each
/// row's top-`s` `(atom, score)` selection — BIT-IDENTICAL to calling
/// [`super::scoring::top_s_online`] per row, but with score blocks and the
/// online top-`s` fold computed on the device when admitted.
///
/// The device fold uses the same strict order as
/// [`super::scoring::TopSSelector`]: `(|score| desc, atom asc)`. Combined with
/// bit-identical score arithmetic (the score kernel forbids FMA contraction),
/// the routed support matches the CPU oracle **exactly**, while the host only
/// downloads the final `rows × active` shortlists.
///
/// Memory: the `m × K` block is never formed whole — `K` is walked in tiles of
/// at most `GPU_ROUTE_TILE_ELEMS / m` atom-columns. Each tile's score block is
/// folded into resident device top-`s` buffers and discarded, so peak score
/// memory is `m × tile_cols`, independent of `K`.
///
/// Falls back to the per-row CPU `top_s_online` under [`gam_gpu::GpuMode::Off`],
/// below the device break-even, or on any device error under
/// [`gam_gpu::GpuMode::Auto`]; under [`gam_gpu::GpuMode::Required`] a device
/// failure is propagated. The returned [`ScoreBlockPath`] reports which path ran.
///
/// # Errors
/// Returns [`gam_gpu::GpuError`] when [`gam_gpu::GpuMode::Required`] is set but
/// the device path cannot run for this minibatch.
/// One-shot engagement report for the T1 score router (#1551 class: "GPU 0%"
/// runs where the decline reason was swallowed by the Auto fallback). Routed
/// through `log::warn!` — the repo's sanctioned diagnostics path (same class as
/// the arrow_schur #1551 cleanup) — so a `log` backend, when initialised, lands
/// it in the job logs. Each category warns once per process: the route is
/// per-minibatch and a faulting device would otherwise spam thousands of
/// identical lines.
fn note_route_engagement(engaged: bool, detail: &str) {
    use std::sync::Once;
    static ENGAGED_ONCE: Once = Once::new();
    static DECLINED_ONCE: Once = Once::new();
    let once = if engaged { &ENGAGED_ONCE } else { &DECLINED_ONCE };
    once.call_once(|| {
        let verdict = if engaged {
            "device ENGAGED"
        } else {
            "device DECLINED - falling back to CPU"
        };
        log::warn!("[gam-sae sparse_dict score router] {verdict}: {detail}");
    });
}

pub fn route_minibatch_required(
    rows: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    s: usize,
    tile: usize,
    mode: gam_gpu::GpuMode,
) -> Result<(Vec<Vec<(u32, f32)>>, ScoreBlockPath, usize), gam_gpu::GpuError> {
    use super::scoring::top_s_online;

    let m = rows.nrows();
    let k = decoder.nrows();
    let active = s.max(1).min(k.max(1));

    // CPU per-row path (bit-identical oracle), used for Off / below break-even /
    // Auto device-error fallback.
    let cpu_route = || -> Vec<Vec<(u32, f32)>> {
        rows.outer_iter()
            .map(|row| top_s_online(row, decoder, s, tile))
            .collect()
    };

    if mode == gam_gpu::GpuMode::Off {
        return Ok((cpu_route(), ScoreBlockPath::Cpu, 0));
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
        note_route_engagement(
            false,
            &format!(
                "block {m}x{k} = {} elems below the device launch break-even \
                 (DEVICE_SCORE_BLOCK_MIN_ELEMS={DEVICE_SCORE_BLOCK_MIN_ELEMS})",
                m.saturating_mul(k)
            ),
        );
        return Ok((cpu_route(), ScoreBlockPath::Cpu, 0));
    }
    if m == 0 || k == 0 {
        return Ok((cpu_route(), ScoreBlockPath::Cpu, 0));
    }

    // Atom-columns per device launch: bound the per-launch block to
    // GPU_ROUTE_TILE_ELEMS, at least one column, never more than K.
    let tile_cols = plan.tile_items;

    match device::route_decoder_tiled_device(rows, decoder, active, tile_cols) {
        Ok(out) => {
            note_route_engagement(
                true,
                &format!("block {m}x{k}, tile_cols={tile_cols}, active={active}"),
            );
            return Ok((
                out.selections,
                ScoreBlockPath::Device,
                out.device_dtoh_bytes,
            ));
        }
        Err(err) => {
            note_route_engagement(false, &format!("device route fault: {err}"));
            if mode == gam_gpu::GpuMode::Required {
                return Err(err);
            }
            // Auto: the device faulted mid-route; discard partial selectors and
            // run the exact CPU oracle for the whole minibatch.
            return Ok((cpu_route(), ScoreBlockPath::Cpu, 0));
        }
    }
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
        max_shared_mem_per_block: usize,
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
                    max_shared_mem_per_block: gam_gpu::GpuRuntime::global()
                        .map(|runtime| runtime.selected_device().max_shared_mem_per_block)
                        .unwrap_or(0),
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

        // `BM × BN` output tiles with a register-blocked `TN × TM` thread block:
        // grid `ceil(n_atoms/BN) × ceil(n_rows/BM)`, each thread emitting an
        // `(BM/TM) × (BN/TN)` micro-tile via `blockIdx.{x,y}`/`threadIdx.{x,y}`.
        let tile_m = super::SCORE_BLOCK_TILE_M;
        let tile_n = super::SCORE_BLOCK_TILE_N;
        let grid_x: u32 = u32::try_from(n_atoms.div_ceil(tile_n as usize))
            .map_err(|_| gam_gpu::gpu_err!("sparse_dict score-block grid_x overflow"))?;
        let grid_y: u32 = u32::try_from(n_rows.div_ceil(tile_m as usize))
            .map_err(|_| gam_gpu::gpu_err!("sparse_dict score-block grid_y overflow"))?;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (super::SCORE_BLOCK_THREADS_N, super::SCORE_BLOCK_THREADS_M, 1),
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

    const TOP_S_FOLD_THREADS: u32 = 32;

    pub(super) struct RouteDeviceOutput {
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
            .ok_or_else(|| gam_gpu::gpu_err!("sparse_dict top-s fold shared-memory overflow"))?;
        let bytes = slots
            .checked_mul(
                std::mem::size_of::<u32>()
                    + std::mem::size_of::<f32>()
                    + std::mem::size_of::<f32>(),
            )
            .ok_or_else(|| gam_gpu::gpu_err!("sparse_dict top-s fold shared-memory overflow"))?;
        if max_shared_mem_per_block > 0 && bytes > max_shared_mem_per_block {
            return Err(gam_gpu::gpu_err!(
                "sparse_dict top-s fold requires {bytes} shared-memory bytes per row block \
                 (active={active}, threads={threads}) but the selected device reports \
                 max_shared_mem_per_block={max_shared_mem_per_block}"
            ));
        }
        u32::try_from(bytes)
            .map_err(|_| gam_gpu::gpu_err!("sparse_dict top-s fold shared-memory bytes overflow"))
    }

    /// Route-time score-block stream for a full decoder. Rows and the whole decoder
    /// stay resident for the route; one reusable score buffer rotates across K.
    /// Each score tile is folded into resident per-row top-`s` state by
    /// `sparse_dict_fold_top_s`, so the host downloads only the final `(atom,
    /// score)` shortlists instead of every `rows × tile` score.
    pub(super) fn route_decoder_tiled_device(
        rows: ArrayView2<'_, f32>,
        decoder: ArrayView2<'_, f32>,
        active: usize,
        tile_cols: usize,
    ) -> Result<RouteDeviceOutput, GpuError> {
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
            return Ok(RouteDeviceOutput {
                selections: vec![Vec::new(); n_rows],
                device_dtoh_bytes: 0,
            });
        }
        let active = active.max(1).min(k);
        if k > u32::MAX as usize {
            return Err(gam_gpu::gpu_err!(
                "sparse_dict tiled route K={k} exceeds u32 atom-index storage"
            ));
        }

        let b = backend()?;
        let module = module_for(b, p)?;
        let score_func = module
            .load_function("sparse_dict_score_block_offset")
            .gpu_ctx("sparse_dict tiled score-offset load_function")?;
        let fold_func = module
            .load_function("sparse_dict_fold_top_s")
            .gpu_ctx("sparse_dict top-s fold load_function")?;
        let stream = b.stream.clone();

        let rows_storage: Vec<f32>;
        let rows_host: &[f32] = if let Some(slice) = rows.as_slice() {
            slice
        } else {
            rows_storage = rows.iter().copied().collect();
            rows_storage.as_slice()
        };
        assert_eq!(
            rows_host.len(),
            n_rows * p,
            "tiled score rows flatten length"
        );
        let rows_dev = stream
            .clone_htod(rows_host)
            .gpu_ctx("sparse_dict tiled score htod rows")?;

        let decoder_storage: Vec<f32>;
        let decoder_host: &[f32] = if let Some(slice) = decoder.as_slice() {
            slice
        } else {
            decoder_storage = decoder.iter().copied().collect();
            decoder_storage.as_slice()
        };
        assert_eq!(
            decoder_host.len(),
            k * p,
            "tiled score decoder flatten length"
        );

        let n_rows_i32 = i32::try_from(n_rows).map_err(|_| {
            gam_gpu::gpu_err!("sparse_dict tiled score n_rows={n_rows} overflows i32")
        })?;
        let active_i32 = i32::try_from(active).map_err(|_| {
            gam_gpu::gpu_err!("sparse_dict tiled score active={active} overflows i32")
        })?;
        let tile_m = super::SCORE_BLOCK_TILE_M;
        let tile_n = super::SCORE_BLOCK_TILE_N;
        let tile_cols = tile_cols.max(1);
        let max_tile_cols = tile_cols.min(k);
        let decoder_dev = stream
            .clone_htod(decoder_host)
            .gpu_ctx("sparse_dict tiled score htod decoder")?;
        let mut scores_dev = stream
            .alloc_zeros::<f32>(n_rows * max_tile_cols)
            .gpu_ctx("sparse_dict tiled score alloc scores")?;
        let mut top_atoms_dev = stream
            .alloc_zeros::<u32>(n_rows * active)
            .gpu_ctx("sparse_dict top-s alloc atoms")?;
        let mut top_scores_dev = stream
            .alloc_zeros::<f32>(n_rows * active)
            .gpu_ctx("sparse_dict top-s alloc scores")?;
        let mut top_mags_dev = stream
            .alloc_zeros::<f32>(n_rows * active)
            .gpu_ctx("sparse_dict top-s alloc mags")?;
        let fold_shared =
            fold_shared_bytes(active, TOP_S_FOLD_THREADS, b.max_shared_mem_per_block)?;

        let mut start = 0usize;
        while start < k {
            let end = (start + tile_cols).min(k);
            let n_atoms = end - start;
            let n_atoms_i32 = i32::try_from(n_atoms).map_err(|_| {
                gam_gpu::gpu_err!("sparse_dict tiled score n_atoms={n_atoms} overflows i32")
            })?;
            let atom_offset = u32::try_from(start).map_err(|_| {
                gam_gpu::gpu_err!("sparse_dict tiled score atom offset={start} overflows u32")
            })?;
            let grid_x: u32 = u32::try_from(n_atoms.div_ceil(tile_n as usize))
                .map_err(|_| gam_gpu::gpu_err!("sparse_dict tiled score grid_x overflow"))?;
            let grid_y: u32 = u32::try_from(n_rows.div_ceil(tile_m as usize))
                .map_err(|_| gam_gpu::gpu_err!("sparse_dict tiled score grid_y overflow"))?;
            let cfg = LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (super::SCORE_BLOCK_THREADS_N, super::SCORE_BLOCK_THREADS_M, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&score_func);
            builder
                .arg(&rows_dev)
                .arg(&decoder_dev)
                .arg(&n_rows_i32)
                .arg(&n_atoms_i32)
                .arg(&atom_offset)
                .arg(&mut scores_dev);
            // SAFETY: grid/block validated; device pointers are cudarc-checked
            // allocations on this stream. The kernel reads the resident rows and
            // resident decoder slice `[atom_offset, atom_offset + n_atoms)` and
            // writes exactly `n_rows * n_atoms` scores.
            unsafe { builder.launch(cfg) }.gpu_ctx("sparse_dict tiled score launch")?;

            let fold_cfg = LaunchConfig {
                grid_dim: (
                    u32::try_from(n_rows)
                        .map_err(|_| gam_gpu::gpu_err!("sparse_dict top-s fold grid overflow"))?,
                    1,
                    1,
                ),
                block_dim: (TOP_S_FOLD_THREADS, 1, 1),
                shared_mem_bytes: fold_shared,
            };
            let mut fold = stream.launch_builder(&fold_func);
            fold.arg(&scores_dev)
                .arg(&n_rows_i32)
                .arg(&n_atoms_i32)
                .arg(&atom_offset)
                .arg(&active_i32)
                .arg(&mut top_atoms_dev)
                .arg(&mut top_scores_dev)
                .arg(&mut top_mags_dev);
            // SAFETY: the fold kernel launches one block per row, reads the
            // score tile just written by the previous launch on this stream, and
            // updates exactly `n_rows * active` shortlist slots.
            unsafe { fold.launch(fold_cfg) }.gpu_ctx("sparse_dict top-s fold launch")?;
            start = end;
        }

        let mut top_atoms = vec![0u32; n_rows * active];
        let mut top_scores = vec![0.0f32; n_rows * active];
        stream
            .memcpy_dtoh(&top_atoms_dev, &mut top_atoms)
            .gpu_ctx("sparse_dict top-s dtoh atoms")?;
        stream
            .memcpy_dtoh(&top_scores_dev, &mut top_scores)
            .gpu_ctx("sparse_dict top-s dtoh scores")?;
        stream
            .synchronize()
            .gpu_ctx("sparse_dict tiled route synchronize")?;

        let mut selections = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let mut row = Vec::with_capacity(active);
            let base = r * active;
            for j in 0..active {
                let atom = top_atoms[base + j];
                if atom != u32::MAX {
                    row.push((atom, top_scores[base + j]));
                }
            }
            selections.push(row);
        }
        Ok(RouteDeviceOutput {
            selections,
            device_dtoh_bytes: n_rows
                .saturating_mul(active)
                .saturating_mul(std::mem::size_of::<u32>() + std::mem::size_of::<f32>()),
        })
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
            Ok((routed, path, dtoh_bytes)) => {
                assert_eq!(
                    path,
                    ScoreBlockPath::Device,
                    "Required succeeded but reported CPU — device did not engage"
                );
                assert_eq!(
                    dtoh_bytes,
                    m * s * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>()),
                    "device route must download only the final top-s shortlist"
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
                let (routed, path, dtoh_bytes) = route_minibatch_required(
                    rows.view(),
                    atoms.view(),
                    s,
                    tile,
                    gam_gpu::GpuMode::Auto,
                )
                .expect("Auto must not error on a device-absent host");
                assert_eq!(path, ScoreBlockPath::Cpu);
                assert_eq!(dtoh_bytes, 0);
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
            Ok((routed, path, dtoh_bytes)) => {
                assert_eq!(
                    path,
                    ScoreBlockPath::Device,
                    "Required succeeded at K=32k but reported CPU — device did not engage"
                );
                assert_eq!(
                    dtoh_bytes,
                    m * s * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>()),
                    "K=32k device route must download only the final top-s shortlist"
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
                let (routed, path, dtoh_bytes) = route_minibatch_required(
                    rows.view(),
                    atoms.view(),
                    s,
                    tile,
                    gam_gpu::GpuMode::Auto,
                )
                .expect("Auto must not error on a device-absent host");
                assert_eq!(path, ScoreBlockPath::Cpu);
                assert_eq!(dtoh_bytes, 0);
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
