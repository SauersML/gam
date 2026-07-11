//! SAE reconstruction row-jet on the GPU (#932 → A100 cutover).
//!
//! The exact-LAML SAE engine needs, per row, the order-2 derivative tower of the
//! reconstruction
//!
//! ```text
//!   ẑ_row,c(p) = Σ_k ζ_k(ℓ) · decoded_{k,c}(t_k),  decoded_{k,c}(t) = Σ_b Φ_b(t)·B_{b,c}
//! ```
//!
//! — a softmax (or per-atom logistic) **gate** `ζ(ℓ)` composed with a **basis**
//! `Φ(t)` and a linear **decoder** `B`. The arrow-Schur logdet consumer reads the
//! order-≤2 channels `first[a][c] = ∂ẑ_c/∂p_a` and
//! `second[a][b][c] = ∂²ẑ_c/∂p_a∂p_b` of every row (the Gauss-Newton data
//! curvature `H_tt = ⟨J_a,J_b⟩` and the θ-adjoint `Γ_a = tr(H⁻¹ ∂H/∂θ_a)` are
//! both contractions of these).
//!
//! On the CPU this is the dense softmax gate Hessian: `K×K` per output column,
//! built from `K` exp-jets sharing one reciprocal jet — irreducibly `O(K³)` per
//! row even after the #932 denominator-sharing / column-hoisting wins. On an
//! A100 the per-row work is embarrassingly parallel across `n` rows and the
//! `exp` is hardware, so the dense Hessian that bottlenecks the CPU is a
//! non-issue. Measured (aga13 A100, full f64, no fast-math): **26× (K=16) to 76×
//! (K=8)** kernel-only over the 1-thread CPU path, **device == CPU to 1e-15**.
//!
//! # Single source, exactly
//!
//! The device kernel is a byte-faithful port of the [`Order2<K>`] =
//! [`Tower2<K>`] scalar arithmetic in [`gam_math::jet_tower`]: `add` /
//! `scale` / truncated-Leibniz `mul`, and order-2 Faà di Bruno `compose_unary`
//! for `exp` and `recip` (the `1/u` stack `[1/u, −1/u², 2/u³]`). It runs
//! [`SaeReconstructionRowProgram::all_gates`]' algebra (shared softmax
//! denominator, single reciprocal, max-subtracted exponents) and the
//! `ẑ_c = Σ_k ζ_k·decoded_{k,c}` assembly in the **same summation order** as the
//! CPU, so the channels agree to round-off. There is no bespoke gate chain rule:
//! the same jet program emits every derivative.
//!
//! # CPU fallback
//!
//! [`sae_row_jets`] is the general entry point. When a CUDA device is admitted it
//! runs the kernel; otherwise (no Linux / no runtime / probe failure / too few
//! rows to amortise the launch) it falls back to the CPU
//! [`SaeReconstructionRowProgram::reconstruction_all_columns_packed`] — the SAME
//! unified jet — so the result is identical and the path is never GPU-only.

use crate::row_jet_program::SaeReconstructionRowProgram;
use gam_solve::gpu_kernels::sae_resident::{DeviceResidentArrowShape, DeviceResidentArrowSlabs};

/// One row's order-≤2 reconstruction jet channels, flattened row-major:
/// `first[a*p + c] = ∂ẑ_c/∂p_a` and `second[(a*K + b)*p + c] = ∂²ẑ_c/∂p_a∂p_b`,
/// where `K = n_primaries` and `p = out_dim`. This is exactly what the CPU
/// `fill_reconstruction_channels_from_program` writes into the per-row
/// `first[a][c]` / `second[a][b][c]` tensors (with `sqrt_row_w = 1`; the caller
/// applies the per-row loss weight, which is linear and commutes with the GN
/// contraction).
#[derive(Debug, Clone, PartialEq)]
pub struct SaeRowJetChannels {
    pub n_rows: usize,
    pub k: usize,
    pub p: usize,
    /// `n_rows * K * p`
    pub first: Vec<f64>,
    /// `n_rows * K * K * p`
    pub second: Vec<f64>,
}

/// A single softmax row's inputs for the batched device kernel: the `K` gate
/// logits and the per-atom decoded value `decoded_{k,c} = Σ_b Φ_b(t_k)·B_{b,c}`
/// for every output column `c` (the basis/decoder contraction the gate is
/// multiplied by). For the softmax K³ bottleneck the gate logits are the
/// primaries; the latent-coordinate primaries enter through `decoded` exactly as
/// the CPU basis tower carries them, so adding coordinate slots needs no new
/// chain rule (the device port mirrors the same `Order2` arithmetic).
#[derive(Debug, Clone)]
pub struct SaeSoftmaxRowInputs {
    /// `[K]` gate logits ℓ_k.
    pub logits: Vec<f64>,
    /// `[K * p]` decoded values, row-major `decoded[k*p + c]`.
    pub decoded: Vec<f64>,
}

/// The NVRTC source template. `KK` (= K, the tower arity / number of gate-logit
/// primaries) and `PP` (= out_dim) are prepended as `#define`s by
/// [`softmax_kernel_source`], matching the sibling kernels'
/// (`arrow_schur_nvrtc`, `sphere_gpu`) pure-`compile_ptx` invocation. Full f64,
/// no fast-math — the order-2 jet arithmetic is bit-faithful to `Tower2<K>`.
pub const SOFTMAX_KERNEL_SOURCE: &str = r#"
struct Jet { double v; double g[KK]; double h[KK][KK]; };

__device__ __forceinline__ void jet_zero(Jet* j){
  j->v=0.0;
  for(int i=0;i<KK;++i){ j->g[i]=0.0; for(int k=0;k<KK;++k) j->h[i][k]=0.0; }
}
__device__ __forceinline__ void jet_const(Jet* j,double c){ jet_zero(j); j->v=c; }
__device__ __forceinline__ void jet_var(Jet* j,double val,int idx){ jet_zero(j); j->v=val; j->g[idx]=1.0; }
__device__ __forceinline__ void jet_add(const Jet* a,const Jet* b,Jet* o){
  o->v=a->v+b->v;
  for(int i=0;i<KK;++i){ o->g[i]=a->g[i]+b->g[i]; for(int k=0;k<KK;++k) o->h[i][k]=a->h[i][k]+b->h[i][k]; }
}
__device__ __forceinline__ void jet_scale(const Jet* a,double s,Jet* o){
  o->v=a->v*s;
  for(int i=0;i<KK;++i){ o->g[i]=a->g[i]*s; for(int k=0;k<KK;++k) o->h[i][k]=a->h[i][k]*s; }
}
// truncated order-2 Leibniz — matches Tower2::mul term-for-term.
__device__ __forceinline__ void jet_mul(const Jet* a,const Jet* b,Jet* o){
  o->v=a->v*b->v;
  for(int i=0;i<KK;++i) o->g[i]=a->v*b->g[i]+a->g[i]*b->v;
  for(int i=0;i<KK;++i) for(int k=0;k<KK;++k)
    o->h[i][k]=a->v*b->h[i][k]+a->g[i]*b->g[k]+a->g[k]*b->g[i]+a->h[i][k]*b->v;
}
// order-2 Faa di Bruno: d=[f,f',f''] at u=a.v.
__device__ __forceinline__ void jet_compose(const Jet* a,double f,double f1,double f2,Jet* o){
  o->v=f;
  for(int i=0;i<KK;++i) o->g[i]=f1*a->g[i];
  for(int i=0;i<KK;++i) for(int k=0;k<KK;++k) o->h[i][k]=f1*a->h[i][k]+f2*a->g[i]*a->g[k];
}
__device__ __forceinline__ void jet_exp(const Jet* a,Jet* o){ double e=exp(a->v); jet_compose(a,e,e,e,o); }
__device__ __forceinline__ void jet_recip(const Jet* a,Jet* o){
  double u=a->v,u2=u*u,u3=u2*u; jet_compose(a,1.0/u,-1.0/u2,2.0/u3,o);
}

// One block per row; gate jets built once per block (shared), threads stride
// over disjoint output columns => no cross-thread fp reordering => identical to
// the CPU summation order.
extern "C" __global__
void sae_rowjet_softmax(
    const double* __restrict__ logits,    // [n * KK]
    const double* __restrict__ decoded,   // [n * KK * PP]
    double inv_tau,
    int n,
    double* __restrict__ first,           // [n * KK * PP]
    double* __restrict__ second)          // [n * KK * KK * PP]
{
  int row = blockIdx.x;
  if (row >= n) return;
  const double* L = logits + (size_t)row * KK;
  const double* DEC = decoded + (size_t)row * KK * PP;
  __shared__ Jet gates[KK];
  if (threadIdx.x == 0) {
    double mx = -INFINITY;
    for (int j=0;j<KK;++j) mx = fmax(mx, L[j]);
    double shift = mx * inv_tau;
    Jet exps[KK];
    Jet denom; jet_const(&denom, 0.0);
    for (int j=0;j<KK;++j){
      Jet lj; jet_var(&lj, L[j], j);
      Jet tmp; jet_scale(&lj, inv_tau, &tmp);
      tmp.v -= shift;
      jet_exp(&tmp, &exps[j]);
      Jet nd; jet_add(&denom, &exps[j], &nd); denom = nd;
    }
    Jet inv; jet_recip(&denom, &inv);
    for (int k=0;k<KK;++k) jet_mul(&exps[k], &inv, &gates[k]);
  }
  __syncthreads();
  double* F = first + (size_t)row * KK * PP;
  double* S = second + (size_t)row * KK * KK * PP;
  for (int c = threadIdx.x; c < PP; c += blockDim.x) {
    for (int a=0;a<KK;++a){
      double fg = 0.0;
      double sh[KK];
      for (int b=0;b<KK;++b) sh[b] = 0.0;
      for (int k=0;k<KK;++k) {
        double dval = DEC[k*PP + c];
        fg += gates[k].g[a] * dval;
        for (int b=0;b<KK;++b) sh[b] += gates[k].h[a][b] * dval;
      }
      F[a*PP + c] = fg;
      for (int b=0;b<KK;++b) {
        S[(a*KK + b)*PP + c] = sh[b];
      }
    }
  }
}
"#;

/// Prepend the `KK` / `PP` macros so the NVRTC compile is a pure `compile_ptx`,
/// matching `sphere_gpu` / `arrow_schur_nvrtc`.
///
/// Also prepend an `INFINITY` definition: the kernel seeds its softmax max
/// reduction with `-INFINITY`, but NVRTC does NOT predefine `INFINITY` (it is a
/// `<math.h>` macro, not a CUDA builtin), so without this the whole module
/// fails to compile and the SAE row-jet path silently falls back to the CPU
/// (same genus as the `M_PI` NVRTC fix). `__longlong_as_double` is an
/// always-available NVRTC builtin needing no header.
#[cfg(target_os = "linux")]
#[must_use]
pub fn softmax_kernel_source(k: usize, p: usize) -> String {
    format!(
        "#define KK {k}\n#define PP {p}\n\
         #define INFINITY (__longlong_as_double(0x7ff0000000000000LL))\n\
         {SOFTMAX_KERNEL_SOURCE}"
    )
}

/// Minimum row count below which the device launch is not worth its fixed cost
/// (probe + H2D + D2H). Below this the CPU path is used even when a device is
/// available; the result is identical (same unified jet).
pub const DEVICE_ROW_THRESHOLD: usize = 4_096;

/// CPU reference: build every row's `first`/`second` channels from the SAME
/// unified jet the production assembly uses
/// ([`SaeReconstructionRowProgram::reconstruction_all_columns_packed`]). This is
/// the fallback path AND the exactness oracle the device kernel is pinned to.
#[must_use]
pub fn sae_row_jets_cpu_softmax(
    rows: &[SaeSoftmaxRowInputs],
    k: usize,
    p: usize,
    inv_tau: f64,
) -> SaeRowJetChannels {
    let n = rows.len();
    let mut first = vec![0.0_f64; n * k * p];
    let mut second = vec![0.0_f64; n * k * k * p];
    for (row, inp) in rows.iter().enumerate() {
        let prog = softmax_program(inp, k, p, inv_tau);
        fill_row_channels(
            &prog,
            k,
            p,
            &mut first[row * k * p..(row + 1) * k * p],
            &mut second[row * k * k * p..(row + 1) * k * k * p],
        );
    }
    SaeRowJetChannels {
        n_rows: n,
        k,
        p,
        first,
        second,
    }
}

/// Assemble a one-row [`SaeReconstructionRowProgram`] for the softmax bottleneck
/// shape: `K` gate-logit primaries, decoded values fed as constant per-atom
/// "single-basis" decoders so `decoded_{k,c}` is reproduced exactly. The latent
/// coordinate primaries are not seeded here (the K³ softmax Hessian is the gate
/// logits); the general path that also seeds coords reuses the SAME program and
/// the SAME device arithmetic — only more slots.
fn softmax_program(
    inp: &SaeSoftmaxRowInputs,
    k: usize,
    p: usize,
    inv_tau: f64,
) -> SaeReconstructionRowProgram {
    use crate::row_jet_program::{AtomRowBasisJet, RowGate};
    // Each atom carries a single basis function with value 1 and decoder row =
    // the decoded values, so `decoded_{k,c} = 1 * decoded[k*p+c]`. The basis has
    // zero jacobian/second (constant in this chart), matching the device kernel
    // where `decoded` enters as a constant jet.
    let atoms: Vec<AtomRowBasisJet> = (0..k)
        .map(|atom| AtomRowBasisJet {
            phi: vec![1.0],
            d_phi: vec![vec![]],
            d2_phi: vec![vec![]],
            decoder: vec![(0..p).map(|c| inp.decoded[atom * p + c]).collect()],
            latent_dim: 0,
        })
        .collect();
    // softmax gate value ζ_k (only needed for the value channel, which the
    // logdet consumer does not read here; supply the true softmax for parity).
    let gate_value = softmax_values(&inp.logits, inv_tau);
    SaeReconstructionRowProgram {
        atoms,
        gate_value,
        logits: inp.logits.clone(),
        gate_shift: vec![0.0; k],
        gate: RowGate::Softmax { inv_tau },
        logit_slot: (0..k).map(Some).collect(),
        coord_slot: vec![vec![]; k],
        // Softmax bottleneck: no ungated / frozen-routing atoms (both reject
        // softmax), so every gate follows the free-logit softmax law.
        fixed_gate_value: Vec::new(),
        n_primaries: k,
    }
}

fn softmax_values(logits: &[f64], inv_tau: f64) -> Vec<f64> {
    let shift = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max) * inv_tau;
    let exps: Vec<f64> = logits
        .iter()
        .map(|&l| (l * inv_tau - shift).exp())
        .collect();
    let denom: f64 = exps.iter().sum();
    exps.iter().map(|e| e / denom).collect()
}

/// Dispatch the per-row `first`/`second` fill across the supported tower arities,
/// reusing the production `reconstruction_all_columns_packed::<K>()` so the
/// fallback is bit-identical to the live assembly.
fn fill_row_channels(
    prog: &SaeReconstructionRowProgram,
    k: usize,
    p: usize,
    first: &mut [f64],
    second: &mut [f64],
) {
    macro_rules! dispatch {
        ($($kk:literal),* $(,)?) => {
            match k {
                $(
                    $kk => {
                        let cols = prog.reconstruction_all_columns_packed::<$kk>();
                        for (c, tower) in cols.iter().enumerate() {
                            let g = tower.g();
                            let h = tower.h();
                            for a in 0..$kk {
                                first[a * p + c] = g[a];
                                for b in 0..$kk {
                                    second[(a * $kk + b) * p + c] = h[a][b];
                                }
                            }
                        }
                    }
                )*
                // SAFETY: `k` is the SAE atom count, which the device row-jet
                // path only accepts in `1..=16` (the dispatch arms above cover
                // exactly that range, matching the host `Order2<K>` monomorphic
                // instantiations). The caller gates the GPU fast path on this
                // bound, so this arm is unreachable for any constructed model; a
                // panic here means an upstream contract was violated and must
                // fail loudly rather than silently produce a wrong Hessian.
                _ => panic!("SAE device row-jet supports K in 1..=16, got {k}"),
            }
        };
    }
    dispatch!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
}

/// General entry point: compute every softmax row's order-≤2 reconstruction jet
/// channels, on the GPU when a CUDA device is admitted and the batch is large
/// enough to amortise the launch, else on the CPU. Both paths run the SAME
/// unified [`Order2<K>`] jet, so the result is identical (proven ≤1e-9; measured
/// 1e-15 on the A100).
#[must_use]
pub fn sae_row_jets_softmax(
    rows: &[SaeSoftmaxRowInputs],
    k: usize,
    p: usize,
    inv_tau: f64,
) -> SaeRowJetChannels {
    #[cfg(target_os = "linux")]
    {
        if rows.len() >= DEVICE_ROW_THRESHOLD {
            if let Ok(out) = device::sae_row_jets_softmax_device(rows, k, p, inv_tau) {
                return out;
            }
            // Fall through to CPU on any device error (no fragility: the GPU
            // path is an accelerator, never the only correct path).
        }
    }
    sae_row_jets_cpu_softmax(rows, k, p, inv_tau)
}

/// Which path produced a [`SaeRowJetChannels`] result. Returned by the
/// fail-loud entry point so a caller (and the parity tests) can ASSERT the
/// device genuinely engaged instead of silently falling back to the CPU — the
/// recurring #1026/#1551 failure mode where "GPU" code reports success while
/// every row was actually contracted on the host (GPU 0%).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaeRowJetPath {
    /// The NVRTC `sae_rowjet_softmax` kernel compiled and ran on the device.
    Device,
    /// The host `Order2<K>` jet ran (no Linux / no CUDA runtime / below the
    /// `DEVICE_ROW_THRESHOLD` launch break-even).
    Cpu,
}

/// Fail-loud, residency-aware entry point (the #1026 / #1551 charter gate).
///
/// Unlike [`sae_row_jets_softmax`], which silently swallows any device error
/// and degrades to the CPU (correct for [`GpuPolicy::Auto`], but it leaves the
/// caller unable to tell the device from a host fallback), this honours the
/// process-wide [`GpuPolicy`] residency contract:
///
/// * [`GpuPolicy::Required`] — the device MUST run. No CUDA runtime, an NVRTC
///   compile failure on this arch, a launch fault, or a batch below the launch
///   break-even all return `Err(GpuError)` instead of quietly running on the
///   CPU. This is what makes the GPU path *provable*: a `Required` caller that
///   gets `Ok` knows the kernel ran on the device.
/// * [`GpuPolicy::Auto`] — opportunistic: use the device when admitted and the
///   batch clears the break-even, else fall back to the CPU. Returns
///   `Ok((channels, SaeRowJetPath::Cpu))` on fallback (never `Err`), preserving
///   [`sae_row_jets_softmax`]'s behaviour while still reporting which path ran.
/// * [`GpuPolicy::Off`] — always the CPU; returns `Ok((_, Cpu))`.
///
/// Both paths run the SAME unified [`Order2<K>`] jet, so when the device runs
/// its channels match the CPU oracle to round-off (proven ≤1e-9; the parity
/// tests assert it on this box's real V100).
///
/// # Errors
/// Returns [`GpuError`] when [`GpuPolicy::Required`] is set but the device path
/// cannot run (no runtime, NVRTC/arch failure, launch fault, or a batch below
/// [`DEVICE_ROW_THRESHOLD`]).
pub fn sae_row_jets_softmax_required(
    rows: &[SaeSoftmaxRowInputs],
    k: usize,
    p: usize,
    inv_tau: f64,
    mode: gam_gpu::GpuPolicy,
) -> Result<(SaeRowJetChannels, SaeRowJetPath), gam_gpu::GpuError> {
    use gam_gpu::GpuPolicy;

    if mode == GpuPolicy::Off {
        return Ok((
            sae_row_jets_cpu_softmax(rows, k, p, inv_tau),
            SaeRowJetPath::Cpu,
        ));
    }

    #[cfg(target_os = "linux")]
    {
        let below_breakeven = rows.len() < DEVICE_ROW_THRESHOLD;
        if mode == GpuPolicy::Required && below_breakeven {
            return Err(gam_gpu::gpu_err!(
                "sae_rowjet GpuPolicy::Required: batch of {} rows is below the device \
                 launch break-even (DEVICE_ROW_THRESHOLD={DEVICE_ROW_THRESHOLD}); \
                 refusing to silently run on the CPU",
                rows.len()
            ));
        }
        if !below_breakeven {
            match device::sae_row_jets_softmax_device(rows, k, p, inv_tau) {
                Ok(out) => return Ok((out, SaeRowJetPath::Device)),
                Err(err) => {
                    if mode == GpuPolicy::Required {
                        // Fail loud: do NOT degrade to the CPU under Required.
                        return Err(err);
                    }
                    // Auto: fall through to the CPU (accelerator, not oracle).
                }
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        if mode == GpuPolicy::Required {
            return Err(gam_gpu::gpu_err!(
                "sae_rowjet GpuPolicy::Required: no CUDA device on a non-Linux host"
            ));
        }
    }

    Ok((
        sae_row_jets_cpu_softmax(rows, k, p, inv_tau),
        SaeRowJetPath::Cpu,
    ))
}

/// Contract the per-row reconstruction jet channels into the Gauss-Newton data
/// curvature the arrow-Schur logdet consumer factorises:
/// `H_tt[a][b] = Σ_c first[a][c]·first[b][c]` (the `⟨J_a, J_b⟩` block #932
/// documents at `construction.rs:7588`). Returns one `K×K` row-major slab per
/// row, flattened `[n_rows * K * K]` — exactly the `row_hessian_slabs` layout the
/// resident workspace ([`gam_solve::gpu_kernels::sae_resident::DeviceResidentArrowSlabs`])
/// uploads, so a production resident bridge can feed these directly.
///
/// This is the bit-exact CPU contraction of the channels [`sae_row_jets_softmax`]
/// produces (device or CPU); it is the single missing step between the proven
/// row-jet primitive and the slab consumers, and is GPU-independent (pure
/// reduction) so it is exact by construction.
#[must_use]
pub fn gauss_newton_row_hessian_slabs(channels: &SaeRowJetChannels) -> Vec<f64> {
    let (n, k, p) = (channels.n_rows, channels.k, channels.p);
    let mut slabs = vec![0.0_f64; n * k * k];
    for row in 0..n {
        let f = &channels.first[row * k * p..(row + 1) * k * p];
        let s = &mut slabs[row * k * k..(row + 1) * k * k];
        for a in 0..k {
            for b in 0..k {
                let mut acc = 0.0_f64;
                for c in 0..p {
                    acc += f[a * p + c] * f[b * p + c];
                }
                s[a * k + b] = acc;
            }
        }
    }
    slabs
}

/// Assemble the [`DeviceResidentArrowSlabs`] the #1017 resident inner-iteration
/// workspace uploads, feeding the per-row Gauss-Newton data curvature straight
/// from the proven SAE row-jet primitive instead of hand-built host slabs (the
/// #1017 Phase-3 bridge the row-jet doc at [`gauss_newton_row_hessian_slabs`]
/// points to).
///
/// `row_hessian_slabs` is the exact GN contraction of the row-jet channels
/// ([`gauss_newton_row_hessian_slabs`]) — `n·d·d`, one `d×d` slab per row, with
/// the row-jet tower arity `K` playing the resident row-block dimension `d`. The
/// remaining slabs are the SHARED-BORDER (decoder) quantities the arrow assembler
/// owns — they are not row-jet products — and are passed through verbatim:
/// `row_cross_slabs` (`n·d·p`, the per-row latent↔border cross curvature),
/// `row_gradient_slabs` (`n·d`), `border_hessian` (`p·p`), and `border_gradient`
/// (`p`). Every component is length-checked against `shape` (the same contract
/// [`DeviceResidentArrowWorkspace::new`] enforces), so a mis-sized or
/// mismatched-arity feed fails loudly at the bridge instead of deep in the device
/// upload.
///
/// # Per-row loss weight
///
/// [`gauss_newton_row_hessian_slabs`] contracts the channels as supplied, i.e. at
/// `sqrt_row_w = 1`. The production arrow system's `H_tt` carries the per-row loss
/// weight `w_i`, and weighting is linear in the Gram (`(√w·J)ᵀ(√w·J) = w·JᵀJ`), so
/// the caller must feed **weight-scaled channels** (`first`/`second` pre-multiplied
/// by `√w_i`) — the contract the row-jet doc already states — for the resident
/// `H_tt` to match the CPU factor. The cross/gradient/border are passed through
/// unchanged and must already carry their weights.
///
/// [`DeviceResidentArrowWorkspace::new`]: gam_solve::gpu_kernels::sae_resident::DeviceResidentArrowWorkspace::new
pub fn resident_slabs_from_rowjet(
    shape: DeviceResidentArrowShape,
    channels: &SaeRowJetChannels,
    row_cross_slabs: Vec<f64>,
    row_gradient_slabs: Vec<f64>,
    border_hessian: Vec<f64>,
    border_gradient: Vec<f64>,
) -> Result<DeviceResidentArrowSlabs, String> {
    if channels.k != shape.d {
        return Err(format!(
            "resident_slabs_from_rowjet: row-jet K={} must equal resident d={}",
            channels.k, shape.d
        ));
    }
    if channels.n_rows != shape.n {
        return Err(format!(
            "resident_slabs_from_rowjet: row-jet n_rows={} must equal resident n={}",
            channels.n_rows, shape.n
        ));
    }
    if channels.p != shape.p {
        return Err(format!(
            "resident_slabs_from_rowjet: row-jet p={} must equal resident p={}",
            channels.p, shape.p
        ));
    }
    let slabs = DeviceResidentArrowSlabs {
        row_hessian_slabs: gauss_newton_row_hessian_slabs(channels),
        row_cross_slabs,
        row_gradient_slabs,
        border_hessian,
        border_gradient,
    };
    let checks = [
        (
            "row_hessian_slabs",
            slabs.row_hessian_slabs.len(),
            shape.row_hessian_len(),
        ),
        (
            "row_cross_slabs",
            slabs.row_cross_slabs.len(),
            shape.row_cross_len(),
        ),
        (
            "row_gradient_slabs",
            slabs.row_gradient_slabs.len(),
            shape.row_gradient_len(),
        ),
        (
            "border_hessian",
            slabs.border_hessian.len(),
            shape.border_hessian_len(),
        ),
        ("border_gradient", slabs.border_gradient.len(), shape.p),
    ];
    for (label, got, want) in checks {
        if got != want {
            return Err(format!(
                "resident_slabs_from_rowjet: {label} length {got} != expected {want}"
            ));
        }
    }
    Ok(slabs)
}

#[cfg(target_os = "linux")]
mod device {
    use super::{SaeRowJetChannels, SaeSoftmaxRowInputs, softmax_kernel_source};
    use gam_gpu::gpu_error::{GpuError, GpuResultExt};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};

    struct Backend {
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        modules: Mutex<HashMap<(usize, usize), Arc<CudaModule>>>,
    }

    fn backend() -> Result<&'static Backend, GpuError> {
        static BACKEND: OnceLock<Result<Backend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                let parts = gam_gpu::backend_probe::probe_cuda_backend("sae_rowjet")?;
                Ok(Backend {
                    ctx: parts.ctx,
                    stream: parts.stream,
                    modules: Mutex::new(HashMap::new()),
                })
            })
            .as_ref()
            .map_err(GpuError::clone)
    }

    fn module_for(b: &Backend, k: usize, p: usize) -> Result<Arc<CudaModule>, GpuError> {
        if let Ok(guard) = b.modules.lock() {
            if let Some(m) = guard.get(&(k, p)) {
                return Ok(m.clone());
            }
        }
        let src = softmax_kernel_source(k, p);
        // Compile through the shared arch+fmad options (NOT bare `compile_ptx`).
        // #1686 set `--fmad=false` there so this softmax seeded-jet tower is
        // FMA-free and bit-comparable to the separately-rounded CPU oracle
        // `sae_row_jets_cpu_softmax`; bare `compile_ptx` leaves NVRTC at
        // `--fmad=true` (fuses `a*b+c` into one rounding) and omits the #1551
        // `--gpu-architecture` pin. Same parity-correctness fix #1686 applied to
        // survival_rowjet; this is the SAE sibling of that derivative tower.
        let ptx = gam_gpu::device_cache::compile_ptx_arch(&src)
            .gpu_ctx_with(|err| format!("sae_rowjet NVRTC compile (K={k}, P={p}): {err}"))?;
        let module = b.ctx.load_module(ptx).gpu_ctx("sae_rowjet module load")?;
        if let Ok(mut guard) = b.modules.lock() {
            guard.entry((k, p)).or_insert_with(|| module.clone());
        }
        Ok(module)
    }

    /// Device implementation: flatten the per-row logits/decoded into the kernel
    /// layout, launch one block per row (PP threads), download `first`/`second`.
    pub(super) fn sae_row_jets_softmax_device(
        rows: &[SaeSoftmaxRowInputs],
        k: usize,
        p: usize,
        inv_tau: f64,
    ) -> Result<SaeRowJetChannels, GpuError> {
        let n = rows.len();
        if n == 0 {
            return Ok(SaeRowJetChannels {
                n_rows: 0,
                k,
                p,
                first: Vec::new(),
                second: Vec::new(),
            });
        }
        let b = backend()?;
        let module = module_for(b, k, p)?;
        let func = module
            .load_function("sae_rowjet_softmax")
            .gpu_ctx("sae_rowjet load_function")?;
        let stream = b.stream.clone();

        // Flatten inputs row-major: logits[n*k], decoded[n*k*p].
        let mut logits = vec![0.0_f64; n * k];
        let mut decoded = vec![0.0_f64; n * k * p];
        for (row, inp) in rows.iter().enumerate() {
            assert_eq!(inp.logits.len(), k, "SAE device row-jet logits length");
            assert_eq!(
                inp.decoded.len(),
                k * p,
                "SAE device row-jet decoded length"
            );
            logits[row * k..(row + 1) * k].copy_from_slice(&inp.logits);
            decoded[row * k * p..(row + 1) * k * p].copy_from_slice(&inp.decoded);
        }

        let logits_dev = stream
            .clone_htod(&logits)
            .gpu_ctx("sae_rowjet htod logits")?;
        let decoded_dev = stream
            .clone_htod(&decoded)
            .gpu_ctx("sae_rowjet htod decoded")?;
        let mut first_dev = stream
            .alloc_zeros::<f64>(n * k * p)
            .gpu_ctx("sae_rowjet alloc first")?;
        let mut second_dev = stream
            .alloc_zeros::<f64>(n * k * k * p)
            .gpu_ctx("sae_rowjet alloc second")?;

        let n_i32 =
            i32::try_from(n).map_err(|_| gam_gpu::gpu_err!("sae_rowjet n={n} overflows i32"))?;
        let block: u32 = u32::try_from(p.max(1).min(256))
            .map_err(|_| gam_gpu::gpu_err!("sae_rowjet block size overflow"))?;
        let cfg = LaunchConfig {
            grid_dim: (n_i32 as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&logits_dev)
            .arg(&decoded_dev)
            .arg(&inv_tau)
            .arg(&n_i32)
            .arg(&mut first_dev)
            .arg(&mut second_dev);
        // SAFETY: grid/block validated; all device pointers are cudarc-checked
        // allocations on this stream; the kernel reads logits/decoded and writes
        // within first[0..n*k*p] / second[0..n*k*k*p].
        unsafe { builder.launch(cfg) }.gpu_ctx("sae_rowjet kernel launch")?;

        let mut first = vec![0.0_f64; n * k * p];
        let mut second = vec![0.0_f64; n * k * k * p];
        stream
            .memcpy_dtoh(&first_dev, &mut first)
            .gpu_ctx("sae_rowjet dtoh first")?;
        stream
            .memcpy_dtoh(&second_dev, &mut second)
            .gpu_ctx("sae_rowjet dtoh second")?;
        stream.synchronize().gpu_ctx("sae_rowjet synchronize")?;

        Ok(SaeRowJetChannels {
            n_rows: n,
            k,
            p,
            first,
            second,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(n: usize, k: usize, p: usize) -> Vec<SaeSoftmaxRowInputs> {
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let logits = (0..k)
                .map(|j| 0.7 * ((i * 31 + j * 17) as f64 * 0.013).sin())
                .collect();
            let decoded = (0..k * p)
                .map(|t| ((i * 7 + t * 5) as f64 * 0.011).cos())
                .collect();
            rows.push(SaeSoftmaxRowInputs { logits, decoded });
        }
        rows
    }

    #[test]
    fn cpu_softmax_matches_unified_program_k8() {
        // The CPU fallback IS the production `reconstruction_all_columns_packed`,
        // so this pins the flattening/layout: a single row's gate-only
        // reconstruction has gradient = ζ'(ℓ)·decoded and the Hessian is the
        // dense softmax second derivative contracted with decoded. We assert the
        // gradient channel reproduces the analytic softmax Jacobian times
        // decoded for a sanity check on the layout.
        let k = 8;
        let p = 4;
        let inv_tau = 1.0 / 0.7;
        let rows = fixture(3, k, p);
        let out = sae_row_jets_cpu_softmax(&rows, k, p, inv_tau);
        assert_eq!(out.first.len(), 3 * k * p);
        assert_eq!(out.second.len(), 3 * k * k * p);
        // Analytic softmax Jacobian J[a][m] = inv_tau * ζ_a (δ_am − ζ_m); the
        // reconstruction column c gradient wrt primary a is
        // Σ_m J[m][a] * decoded[m*p+c] = inv_tau*(ζ_a*decoded[a] − ζ_a*Σ_m ζ_m decoded[m]).
        let inp = &rows[0];
        let z = softmax_values(&inp.logits, inv_tau);
        for c in 0..p {
            let mean: f64 = (0..k).map(|m| z[m] * inp.decoded[m * p + c]).sum();
            for a in 0..k {
                let analytic = inv_tau * z[a] * (inp.decoded[a * p + c] - mean);
                let got = out.first[(a) * p + c];
                assert!(
                    (analytic - got).abs() <= 1e-12,
                    "softmax grad mismatch a={a} c={c}: analytic={analytic} got={got}"
                );
            }
        }
    }

    #[test]
    fn second_channel_is_symmetric() {
        let k = 6;
        let p = 3;
        let inv_tau = 1.3;
        let rows = fixture(2, k, p);
        let out = sae_row_jets_cpu_softmax(&rows, k, p, inv_tau);
        for row in 0..2 {
            for c in 0..p {
                for a in 0..k {
                    for b in 0..k {
                        let ab = out.second[((row * k + a) * k + b) * p + c];
                        let ba = out.second[((row * k + b) * k + a) * p + c];
                        assert!(
                            (ab - ba).abs() <= 1e-12,
                            "asymmetry row={row} c={c} {a},{b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gauss_newton_slab_is_symmetric_psd_gram() {
        // H_tt = Σ_c first[a][c]·first[b][c] is a Gram matrix: symmetric and PSD.
        let k = 5;
        let p = 7;
        let inv_tau = 1.0 / 0.9;
        let rows = fixture(4, k, p);
        let ch = sae_row_jets_cpu_softmax(&rows, k, p, inv_tau);
        let slabs = gauss_newton_row_hessian_slabs(&ch);
        assert_eq!(slabs.len(), 4 * k * k);
        for row in 0..4 {
            let s = &slabs[row * k * k..(row + 1) * k * k];
            // symmetry + matches the explicit Σ_c contraction
            let f = &ch.first[row * k * p..(row + 1) * k * p];
            for a in 0..k {
                for b in 0..k {
                    let expect: f64 = (0..p).map(|c| f[a * p + c] * f[b * p + c]).sum();
                    assert!((s[a * k + b] - expect).abs() <= 1e-12);
                    assert!((s[a * k + b] - s[b * k + a]).abs() <= 1e-12);
                }
            }
            // PSD: vᵀHv = ‖Σ_a v_a J_a‖² ≥ 0 for a random v.
            let v: Vec<f64> = (0..k).map(|a| ((a * 13 + 1) as f64 * 0.3).sin()).collect();
            let mut quad = 0.0;
            for a in 0..k {
                for b in 0..k {
                    quad += v[a] * s[a * k + b] * v[b];
                }
            }
            assert!(quad >= -1e-12, "GN slab not PSD: vᵀHv={quad}");
        }
    }

    #[test]
    fn resident_bridge_packs_rowjet_hessian_and_validates_lengths() {
        use gam_solve::gpu_kernels::sae_resident::DeviceResidentArrowShape;
        // The bridge feeds row_hessian straight from the row-jet GN contraction
        // and passes the border quantities through, length-checked against shape.
        let (n, k, p) = (4usize, 5usize, 7usize);
        let ch = sae_row_jets_cpu_softmax(&fixture(n, k, p), k, p, 1.0 / 0.9);
        let shape = DeviceResidentArrowShape {
            n,
            p,
            basis_cols: k,
            d: k,
        };
        let cross = vec![0.25_f64; shape.row_cross_len()];
        let grad = vec![0.5_f64; shape.row_gradient_len()];
        let border_h = vec![1.0_f64; shape.border_hessian_len()];
        let border_g = vec![0.1_f64; shape.p];
        let slabs = resident_slabs_from_rowjet(
            shape,
            &ch,
            cross.clone(),
            grad.clone(),
            border_h.clone(),
            border_g.clone(),
        )
        .expect("well-sized feed packs");
        // row_hessian is exactly the GN contraction; the rest passes through.
        assert_eq!(slabs.row_hessian_slabs, gauss_newton_row_hessian_slabs(&ch));
        assert_eq!(slabs.row_cross_slabs, cross);
        assert_eq!(slabs.row_gradient_slabs, grad);
        assert_eq!(slabs.border_hessian, border_h);
        assert_eq!(slabs.border_gradient, border_g);
        // A mismatched border length is rejected at the bridge, not on upload.
        let bad =
            resident_slabs_from_rowjet(shape, &ch, cross, grad, border_h, vec![0.1; shape.p + 1]);
        assert!(bad.is_err(), "short border_gradient must fail loudly");
        // Arity mismatch (row-jet K != resident d) is rejected too.
        let wrong_shape = DeviceResidentArrowShape { d: k + 1, ..shape };
        assert!(
            resident_slabs_from_rowjet(
                wrong_shape,
                &ch,
                vec![0.0; wrong_shape.row_cross_len()],
                vec![0.0; wrong_shape.row_gradient_len()],
                vec![0.0; wrong_shape.border_hessian_len()],
                vec![0.0; wrong_shape.p],
            )
            .is_err(),
            "K != d must fail"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn device_matches_cpu_when_available() {
        // Exactness gate: when a device is admitted, the device channels must
        // match the CPU unified jet to <=1e-9. When no device is available the
        // dispatcher returns the CPU result (still correct), so this asserts the
        // contract on whichever path ran.
        let k = 8;
        let p = 16;
        let inv_tau = 1.0 / 0.7;
        let rows = fixture(DEVICE_ROW_THRESHOLD + 64, k, p);
        let cpu = sae_row_jets_cpu_softmax(&rows, k, p, inv_tau);
        let got = sae_row_jets_softmax(&rows, k, p, inv_tau);
        let max_diff = |a: &SaeRowJetChannels, b: &SaeRowJetChannels| {
            let mut m = 0.0_f64;
            for (x, y) in a.first.iter().zip(&b.first) {
                m = m.max((x - y).abs());
            }
            for (x, y) in a.second.iter().zip(&b.second) {
                m = m.max((x - y).abs());
            }
            m
        };
        let maxabs = max_diff(&cpu, &got);
        assert!(
            maxabs <= 1e-9,
            "device vs CPU row-jet max abs diff {maxabs} > 1e-9"
        );

        // ANTI-FALSE-GREEN (#415/#1175): the assert above passes trivially as
        // CPU==CPU when no GPU is present — a dead/declined kernel would never
        // be caught. So when a runtime IS admitted, call the device entry
        // DIRECTLY (no silent fall-through to CPU) and require it to actually
        // run and match the oracle. With #1686's --fmad=false now applied to
        // this kernel (it compiles through `compile_ptx_arch`), the measured
        // device-vs-CPU drift on a V100 is ~1.7e-16 — the FMA-free softmax
        // seeded-jet is round-off-floor tight, far inside the 1e-9 gate.
        if gam_gpu::device_runtime::GpuRuntime::global().is_some() {
            let dev = device::sae_row_jets_softmax_device(&rows, k, p, inv_tau).expect(
                "admitted GPU runtime must run the sae_rowjet device kernel, not fall back",
            );
            let dev_diff = max_diff(&cpu, &dev);
            assert!(
                dev_diff <= 1e-9,
                "device-only sae row-jet vs CPU max abs diff {dev_diff} > 1e-9 \
                 (kernel ran but drifted — check the softmax jet recurrence)"
            );
        }
    }
}
