//! Device-resident **exact per-row certified SAE encode** (#988).
//!
//! The production CPU encode is `crate::encode::EncodeAtlas::certified_encode_row`:
//! for one atom and one target row `x` at fixed amplitude `z` it
//!
//!   1. **routes** the row to the nearest certified charts by ambient
//!      reconstruction distance `‖BᵀΦ(t_c) − x‖²` (the *active-set routing*).
//!      Since #2518 both lanes consider EVERY certifiable chart: the CPU path
//!      prunes by a rigorous per-chart residual bound, and this lane — whose
//!      kernel is generated per `(d, m, p, topk, newton)` and so cannot express a
//!      data-dependent exit — simply sets `topk` to the chart count,
//!   2. **warm-starts** each candidate from that chart's distilled IFT affine
//!      predictor `t̂ = t_c + (1/z)·A₁·(x − z·m₁)`,
//!   3. runs the **per-row latent-coordinate Newton** solve inside the
//!      Kantorovich basin: at each iterate it forms the FULL, TRUE Hessian
//!      `H = JₘᵀJₘ + r·∂²m` (NO Levenberg ridge — the certificate must see the
//!      genuine field, F2), takes the Newton step `δ = −H⁻¹g`, and
//!      evaluates the certificate `h = β·η·L` (`β = 1/λ_min(H)`, `η = ‖δ‖`),
//!      first navigating into the basin (`h ≤ ½`) then refining `newton_steps`,
//!   4. **assigns** the row to the lowest-reconstruction-error CERTIFIED
//!      candidate (the *assignment/gate solve*), and
//!   5. otherwise returns the nearest chart's uncertified result — the
//!      *certificate/fallback* the exact multi-start solve owns.
//!
//! This module ships that whole pipeline as a **device kernel** for the
//! `EuclideanPatch` monomial family (the atom family whose basis
//! `Φ_α(t) = Π_axis t_axis^{α_axis}` is closed-form-evaluable on-device with
//! exact first/second jets — see [`crate::basis::EuclideanPatchEvaluator`]).
//! One CUDA block encodes one row; the per-row work is done serially by the
//! block's lead thread so the accumulation order is byte-identical to the
//! host oracle (the same `tid == 0` idiom the fused Arrow-Schur kernel in
//! `gam_solve::gpu_kernels::arrow_schur_nvrtc` uses for its Cholesky).
//!
//! # Correctness without a GPU
//!
//! Exactly the #1017 pattern of `arrow_schur_nvrtc`:
//!
//! * `emulate_certified_encode_row` is a device-free CPU emulator that mirrors
//!   the kernel's arithmetic and control flow line-for-line — the SAME monomial
//!   evaluation, the SAME cyclic-Jacobi symmetric eigensolver
//!   ([`jacobi_eigh`], the device stand-in for the host LAPACK `eigh`), the SAME
//!   basin-warmup / refine loop, the SAME routing + assignment. It is the CPU
//!   fallback AND the exactness oracle the kernel is pinned to.
//! * The parity tests assert the emulator reproduces the production
//!   `crate::encode::EncodeAtlas::certified_encode_row` on planted + random
//!   rows (support/coords/amplitude/certificate within a tight tol; the only
//!   divergence is Jacobi-vs-LAPACK eigen round-off).
//! * On Linux the CUDA source compiles to PTX through the shared
//!   `--fmad=false` NVRTC options ([`gam_gpu::device_cache::compile_ptx_arch`]),
//!   matching the sibling kernels; a device, when present, runs it and the
//!   dispatch reports [`EncodePath::Device`] honestly (the #1026/#1551 gate).
//!
//! # What still needs real hardware
//!
//! Running the PTX (a launch on a CUDA device) and confirming device==emulator
//! to round-off requires a GPU. Everything else — the kernel source, the
//! emulator, the parity against production, and (on a CUDA host) the NVRTC→PTX
//! compile + PTX audit — is verified without one.

use crate::encode::KANTOROVICH_THRESHOLD;
use gam_gpu::policy::EncodeDeploymentDecision;

/// One `EuclideanPatch` atom's frozen encode data, flattened for a device
/// launch. This is exactly what the online encode reads: the monomial exponent
/// table, the decoder `B`, and the offline-certified charts. Built from a real
/// atom + its `AtomEncodeAtlas` by `EncodeAtomDevice::from_atom_atlas` so
/// the device path consumes the identical data the CPU path does.
#[derive(Debug, Clone)]
pub struct EncodeAtomDevice {
    /// Latent dimension `d`.
    pub d: usize,
    /// Basis size `m` (number of monomials of total degree ≤ degree).
    pub m: usize,
    /// Output dimension `p`.
    pub p: usize,
    /// Number of nearest charts refined per row — the atom's FULL certifiable
    /// chart count since #2518, never the deleted `CERTIFIED_ROUTING_TOPK = 4`.
    ///
    /// The kernel is generated per `(d, m, p, topk, newton)`, so this lane needs a
    /// count fixed at generation time and cannot express the CPU path's
    /// proof-pruned early exit. Exhaustive is then the only value that keeps the
    /// two lanes returning the SAME coordinate, which the emulator-vs-production
    /// gate enforces.
    pub topk: usize,
    /// Online Newton refinement steps after a certified landing.
    pub newton_steps: usize,
    /// Monomial exponents, row-major `exponents[col*d + axis]`, length `m*d`.
    pub exponents: Vec<i32>,
    /// Decoder `B`, row-major `decoder[basis*p + out]`, length `m*p`.
    pub decoder: Vec<f64>,
    /// Charts (routing + warm-start + certificate constants).
    pub charts: Vec<EncodeChartDevice>,
}

/// One offline-certified chart, flattened.
#[derive(Debug, Clone)]
pub struct EncodeChartDevice {
    /// Chart center `t_c`, length `d`.
    pub center: Vec<f64>,
    /// In-chart radius (the Lipschitz-validity ball).
    pub radius: f64,
    /// Certified Newton radius (`> 0` ⇒ the chart is routable).
    pub certified_radius: f64,
    /// Closed-form Hessian-Lipschitz constant `L` over the chart.
    pub lipschitz: f64,
    /// Whether the chart carries a distilled IFT Jacobian `A₁` (finite β).
    pub has_jacobian: bool,
    /// `A₁`, row-major `a1[axis*p + out]`, length `d*p` (empty if `!has_jacobian`).
    pub amortized_jacobian: Vec<f64>,
    /// Amplitude-1 center reconstruction `m₁ = BᵀΦ(t_c)`, length `p`.
    pub recon_center: Vec<f64>,
}

/// A per-row Kantorovich certificate, the device/emulator mirror of
/// [`crate::encode::RowCertificate`]. `certified()` uses the SAME `h ≤ ½` gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceRowCertificate {
    pub beta: f64,
    pub eta: f64,
    pub lipschitz: f64,
    pub h: f64,
}

impl DeviceRowCertificate {
    #[inline]
    #[must_use]
    pub fn certified(&self) -> bool {
        self.h.is_finite() && self.h <= KANTOROVICH_THRESHOLD
    }
}

/// One row's encode result: the latent coordinate and its certificate.
#[derive(Debug, Clone)]
pub struct DeviceEncodeRow {
    pub coord: Vec<f64>,
    pub cert: DeviceRowCertificate,
}

// ============================================================================
// Numeric core — the byte-faithful CPU mirror of the device kernel. Every
// function here has a 1:1 CUDA counterpart in `ENCODE_KERNEL_SOURCE`; the CUDA
// comments name the mirror. These are also the CPU fallback path.
// ============================================================================

/// Cyclic Jacobi symmetric eigensolver for a `d×d` matrix (row-major, `d ≤ 8`).
/// Returns eigenvalues `vals[i]` and eigenvectors as COLUMNS
/// `vecs[col*d + row]`. This is the device stand-in for the host LAPACK `eigh`
/// used by `crate::encode::beta_eta_newton`; the Newton step is reconstructed
/// from the (eigenvector-basis-independent) spectral sum, so the result agrees
/// with LAPACK to eigen round-off. The CUDA `jacobi_eigh` mirror is identical.
pub fn jacobi_eigh(a_in: &[f64], d: usize, vals: &mut [f64], vecs: &mut [f64]) {
    // Working copy A (row-major), V = I.
    let mut a = a_in.to_vec();
    for r in 0..d {
        for c in 0..d {
            vecs[c * d + r] = if r == c { 1.0 } else { 0.0 };
        }
    }
    if d == 1 {
        vals[0] = a[0];
        return;
    }
    // Fixed, deterministic sweep count: for d ≤ 8, 30 cyclic sweeps drive the
    // off-diagonal norm to well below f64 round-off.
    for _sweep in 0..30 {
        // Off-diagonal magnitude; stop early when negligible.
        let mut off = 0.0_f64;
        for r in 0..d {
            for c in (r + 1)..d {
                off += a[r * d + c] * a[r * d + c];
            }
        }
        if off <= 1e-300 {
            break;
        }
        for pp in 0..d {
            for q in (pp + 1)..d {
                let apq = a[pp * d + q];
                if apq == 0.0 {
                    continue;
                }
                let app = a[pp * d + pp];
                let aqq = a[q * d + q];
                // Jacobi rotation angle (Golub & Van Loan 8.4.1).
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let cph = 1.0 / (1.0 + t * t).sqrt();
                let sph = t * cph;
                // Apply rotation to A (rows/cols pp,q).
                for k in 0..d {
                    let akp = a[k * d + pp];
                    let akq = a[k * d + q];
                    a[k * d + pp] = cph * akp - sph * akq;
                    a[k * d + q] = sph * akp + cph * akq;
                }
                for k in 0..d {
                    let apk = a[pp * d + k];
                    let aqk = a[q * d + k];
                    a[pp * d + k] = cph * apk - sph * aqk;
                    a[q * d + k] = sph * apk + cph * aqk;
                }
                // Accumulate eigenvectors.
                for k in 0..d {
                    let vkp = vecs[pp * d + k];
                    let vkq = vecs[q * d + k];
                    vecs[pp * d + k] = cph * vkp - sph * vkq;
                    vecs[q * d + k] = sph * vkp + cph * vkq;
                }
            }
        }
    }
    for i in 0..d {
        vals[i] = a[i * d + i];
    }
}

// ============================================================================
// Device kernel source (NVRTC). Faithful port of the numeric core above; one
// block per row, the block's lead thread runs the whole row's encode serially
// (order-identical to the emulator). Compile-time #defines D/M/P/TOPK/NEWTON.
// ============================================================================

/// The NVRTC source template. `DD`/`MM`/`PP`/`TOPK`/`NEWTON`/`RIDGE` are
/// prepended by [`encode_kernel_source`] as `#define`s, matching the sibling
/// kernels' pure `compile_ptx` invocation. Full f64, no fast-math — the encode
/// arithmetic mirrors the CPU `EncodeAtomDevice` core term-for-term.
pub const ENCODE_KERNEL_SOURCE: &str = r#"
#define KANTOROVICH 0.5

__device__ __forceinline__ double dpow(double b, int e){
  // exponentiation-by-squaring, matching llvm.powi/f64::powi and the emulator dpow.
  if (e == 0) return 1.0;
  int n = e < 0 ? -e : e;
  double r = 1.0, base = b;
  while (n > 0){ if (n & 1) r *= base; n >>= 1; if (n) base *= base; }
  return e < 0 ? 1.0 / r : r;
}

// Monomial phi/jet/hess at t (mirror of eval_basis).
__device__ void eval_basis(const int* exps, const double* t,
                           double* phi, double* jet, double* hess){
  for (int col=0; col<MM; ++col){
    double value = 1.0;
    for (int axis=0; axis<DD; ++axis){ int e=exps[col*DD+axis]; if(e!=0) value*=dpow(t[axis],e); }
    phi[col]=value;
    for (int axis=0; axis<DD; ++axis){
      int a_axis=exps[col*DD+axis]; double jval=0.0;
      if (a_axis!=0){ jval=(double)a_axis;
        for(int a=0;a<DD;++a){ int ea=(a==axis)?a_axis-1:exps[col*DD+a]; if(ea!=0) jval*=dpow(t[a],ea); } }
      jet[col*DD+axis]=jval;
    }
    for (int a=0;a<DD;++a) for(int c=0;c<DD;++c){
      double hval=0.0; int aa=exps[col*DD+a]; int ac=exps[col*DD+c];
      int adm = (aa!=0) && (a==c || ac!=0);
      if (adm){
        double lead = (a==c) ? (double)aa*(double)((aa-1)>0?(aa-1):0)
                             : (double)aa*(double)ac;
        if (lead!=0.0){ hval=lead;
          for(int axis=0;axis<DD;++axis){ int e=exps[col*DD+axis];
            if(axis==a) e=(e-1)>0?(e-1):0; if(axis==c) e=(e-1)>0?(e-1):0;
            if(e!=0) hval*=dpow(t[axis],e); } }
      }
      hess[(col*DD+a)*DD+c]=hval;
    }
  }
}

__device__ void recon_amp1(const double* dec, const double* phi, double* out){
  for(int c=0;c<PP;++c) out[c]=0.0;
  for(int b=0;b<MM;++b){ double pv=phi[b]; if(pv==0.0) continue;
    for(int c=0;c<PP;++c) out[c]+=pv*dec[b*PP+c]; }
}

// grad g[D] and full, TRUE Hessian h[D*D] (NO ridge, F2). Mirror of encode_grad_hess.
__device__ void grad_hess(const double* dec, const double* t, const double* x, double amp,
                          const double* phi, const double* jet, const double* hess,
                          double* g, double* h){
  double recon[PP]; double residual[PP]; double jm[DD*PP];
  for(int c=0;c<PP;++c) recon[c]=0.0;
  for(int b=0;b<MM;++b){ double pv=phi[b]; if(pv==0.0) continue;
    for(int c=0;c<PP;++c) recon[c]+=amp*pv*dec[b*PP+c]; }
  for(int c=0;c<PP;++c) residual[c]=recon[c]-x[c];
  for(int i=0;i<DD*PP;++i) jm[i]=0.0;
  for(int axis=0;axis<DD;++axis) for(int b=0;b<MM;++b){ double dphi=jet[b*DD+axis]; if(dphi==0.0) continue;
    for(int c=0;c<PP;++c) jm[axis*PP+c]+=amp*dphi*dec[b*PP+c]; }
  for(int a=0;a<DD;++a){
    double ga=0.0; for(int c=0;c<PP;++c) ga+=jm[a*PP+c]*residual[c]; g[a]=ga;
    for(int b=0;b<DD;++b){
      double hab=0.0; for(int c=0;c<PP;++c) hab+=jm[a*PP+c]*jm[b*PP+c];
      double curv=0.0;
      for(int basis=0;basis<MM;++basis){ double d2=hess[(basis*DD+a)*DD+b]; if(d2==0.0) continue;
        double dot=0.0; for(int c=0;c<PP;++c) dot+=residual[c]*dec[basis*PP+c];
        curv+=amp*d2*dot; }
      h[a*DD+b]=hab+curv;
    }
  }
  // NO ridge: the certificate uses the TRUE Hessian (F2).
}

// Cyclic Jacobi eigensolver (mirror of jacobi_eigh); vecs columns: vecs[col*D+row].
__device__ void jacobi_eigh(const double* a_in, double* vals, double* vecs){
  double a[DD*DD];
  for(int i=0;i<DD*DD;++i) a[i]=a_in[i];
  for(int r=0;r<DD;++r) for(int c=0;c<DD;++c) vecs[c*DD+r]=(r==c)?1.0:0.0;
  if (DD==1){ vals[0]=a[0]; return; }
  for(int sweep=0;sweep<30;++sweep){
    double off=0.0;
    for(int r=0;r<DD;++r) for(int c=r+1;c<DD;++c) off+=a[r*DD+c]*a[r*DD+c];
    if (off<=1e-300) break;
    for(int p=0;p<DD;++p) for(int q=p+1;q<DD;++q){
      double apq=a[p*DD+q]; if(apq==0.0) continue;
      double app=a[p*DD+p]; double aqq=a[q*DD+q];
      double tau=(aqq-app)/(2.0*apq);
      double t = (tau>=0.0) ? 1.0/(tau+sqrt(1.0+tau*tau)) : -1.0/(-tau+sqrt(1.0+tau*tau));
      double cph=1.0/sqrt(1.0+t*t); double sph=t*cph;
      for(int k=0;k<DD;++k){ double akp=a[k*DD+p]; double akq=a[k*DD+q];
        a[k*DD+p]=cph*akp-sph*akq; a[k*DD+q]=sph*akp+cph*akq; }
      for(int k=0;k<DD;++k){ double apk=a[p*DD+k]; double aqk=a[q*DD+k];
        a[p*DD+k]=cph*apk-sph*aqk; a[q*DD+k]=sph*apk+cph*aqk; }
      for(int k=0;k<DD;++k){ double vkp=vecs[p*DD+k]; double vkq=vecs[q*DD+k];
        vecs[p*DD+k]=cph*vkp-sph*vkq; vecs[q*DD+k]=sph*vkp+cph*vkq; }
    }
  }
  for(int i=0;i<DD;++i) vals[i]=a[i*DD+i];
}

// beta/eta/delta; returns 1 on success (lambda_min>0), 0 otherwise.
__device__ int beta_eta_newton(const double* h, const double* g,
                               double* beta, double* eta, double* delta){
  double vals[DD]; double vecs[DD*DD];
  jacobi_eigh(h, vals, vecs);
  double lmin=1.0/0.0; // +inf
  for(int i=0;i<DD;++i) if(vals[i]<lmin) lmin=vals[i];
  if (!(isfinite(lmin) && lmin>0.0)) return 0;
  *beta=1.0/lmin;
  for(int i=0;i<DD;++i) delta[i]=0.0;
  for(int col=0;col<DD;++col){ double lam=vals[col]; if(lam<=0.0) return 0;
    double vg=0.0; for(int row=0;row<DD;++row) vg+=vecs[col*DD+row]*g[row];
    double coeff=vg/lam; for(int row=0;row<DD;++row) delta[row]-=coeff*vecs[col*DD+row]; }
  double e2=0.0; for(int i=0;i<DD;++i) e2+=delta[i]*delta[i]; *eta=sqrt(e2);
  return 1;
}

// row_certificate: writes h_out (=beta*eta*L or +inf) and delta; returns certified 0/1 mask via h.
__device__ void row_certificate(const int* exps, const double* dec,
                                const double* t, const double* x, double amp, double L,
                                double* h_out, double* beta_out, double* eta_out, double* delta){
  double phi[MM]; double jet[MM*DD]; double hess[MM*DD*DD]; double g[DD]; double H[DD*DD];
  eval_basis(exps, t, phi, jet, hess);
  grad_hess(dec, t, x, amp, phi, jet, hess, g, H);
  double beta, eta;
  if (beta_eta_newton(H, g, &beta, &eta, delta)){
    *beta_out=beta; *eta_out=eta; *h_out=beta*eta*L;
  } else {
    *beta_out=1.0/0.0; *eta_out=1.0/0.0; *h_out=1.0/0.0;
    for(int i=0;i<DD;++i) delta[i]=0.0;
  }
}

__device__ int in_chart(const double* t, const double* center, double radius){
  double r2=0.0; for(int i=0;i<DD;++i){ double d=t[i]-center[i]; r2+=d*d; }
  return r2 <= radius*radius;
}

// certify_with_basin_warmup + refine. Returns 1 with coord/landing_h on success.
__device__ int certify_basin(const int* exps, const double* dec,
                             const double* t_start, const double* x, double amp,
                             const double* center, double radius, double L,
                             double* coord_out, double* landing_h){
  double t[DD]; for(int i=0;i<DD;++i) t[i]=t_start[i];
  if(!in_chart(t, center, radius)) return 0;
  double h, beta, eta; double delta[DD];
  row_certificate(exps, dec, t, x, amp, L, &h, &beta, &eta, delta);
  while(!(isfinite(h) && h<=KANTOROVICH)){
    if(!(isfinite(h) && isfinite(beta) && isfinite(eta))) return 0;
    double prev_h=h;
    double next[DD]; for(int i=0;i<DD;++i) next[i]=t[i]+delta[i];
    if(!in_chart(next, center, radius)) return 0;
    for(int i=0;i<DD;++i) t[i]=next[i];
    row_certificate(exps, dec, t, x, amp, L, &h, &beta, &eta, delta);
    if(!(isfinite(h)) || h>=prev_h) return 0;
  }
  // F5: refine, then report the certificate `h` at the REFINED landing coordinate
  // (mirror production `refine_certified_start`'s `final_cert`), NOT the pre-refine
  // basin-exit `h`. `row_certificate` mutates `h` in place at each certified refine
  // iterate, so after the loop `h` already holds the final refined certificate
  // (or the basin-exit `h` if convergence broke before any refine step) — exactly
  // production's `final_cert`.
  for(int s=0;s<NEWTON;++s){
    // convergence early-exit (mirror production refine_certified_start).
    double dnorm=0.0, tnorm=0.0;
    for(int i=0;i<DD;++i){ dnorm+=delta[i]*delta[i]; tnorm+=t[i]*t[i]; }
    if(sqrt(dnorm) <= REFINE_EPS*(1.0+sqrt(tnorm))) break;
    // in-chart soundness guard (mirror production refine_certified_start): L is
    // only valid inside the chart ball; an out-of-ball iterate would recompute h
    // with an invalid L, so refuse — exactly as the warm-up step guard above.
    double rnext[DD]; for(int i=0;i<DD;++i) rnext[i]=t[i]+delta[i];
    if(!in_chart(rnext, center, radius)) return 0;
    for(int i=0;i<DD;++i) t[i]=rnext[i];
    row_certificate(exps, dec, t, x, amp, L, &h, &beta, &eta, delta);
    if(!(isfinite(h) && h<=KANTOROVICH)) return 0;
  }
  for(int i=0;i<DD;++i) coord_out[i]=t[i];
  *landing_h=h;
  return 1;
}

// One block per row. Charts are stored flattened; the block's lead thread runs
// the full route -> warm-start -> certify -> assign pipeline serially.
extern "C" __global__ void sae_certified_encode(
    const int*    __restrict__ exps,           // MM*DD
    const double* __restrict__ dec,            // MM*PP
    const double* __restrict__ centers,        // n_charts*DD
    const double* __restrict__ radii,          // n_charts
    const double* __restrict__ cert_radii,     // n_charts
    const double* __restrict__ lips,           // n_charts
    const int*    __restrict__ has_jac,        // n_charts
    const double* __restrict__ a1,             // n_charts*DD*PP
    const double* __restrict__ recon_c,        // n_charts*PP
    int n_charts,
    const double* __restrict__ targets,        // n*PP
    const double* __restrict__ amps,           // n
    int n,
    double* __restrict__ coords_out,           // n*DD
    double* __restrict__ h_out,                // n   (certificate h; >0.5 or inf = uncertified)
    int*    __restrict__ certified_out)        // n   (1/0)
{
  int row = blockIdx.x;
  if (row >= n) return;
  if (threadIdx.x != 0) return;
  const double* x = targets + (size_t)row*PP;
  double amp = amps[row];

  // ---- routing: top-TOPK certifiable charts by the amplitude-scaled center
  //      recon distance ‖x − z·m₁(t_c)‖² (F1; z·m₁ is the reconstruction actually
  //      compared against x — an amplitude-blind score mis-routes when z != 1). ----
  int cand[TOPK]; double cand_d[TOPK]; int ncand=0;
  {
    double phi[MM]; double jet[MM*DD]; double hess[MM*DD*DD]; double recon[PP];
    for(int idx=0; idx<n_charts; ++idx){
      if (cert_radii[idx] <= 0.0) continue;
      eval_basis(exps, centers + (size_t)idx*DD, phi, jet, hess);
      recon_amp1(dec, phi, recon);
      double dist=0.0; for(int c=0;c<PP;++c){ double df=amp*recon[c]-x[c]; dist+=df*df; }
      // insert into the sorted top-TOPK by (dist, idx).
      int pos=ncand;
      while(pos>0 && (cand_d[pos-1]>dist)){ if(pos<TOPK){cand_d[pos]=cand_d[pos-1]; cand[pos]=cand[pos-1];} pos--; }
      if(pos<TOPK){ cand_d[pos]=dist; cand[pos]=idx; if(ncand<TOPK) ncand++; }
    }
  }
  // defaults: uncertified.
  for(int i=0;i<DD;++i) coords_out[(size_t)row*DD+i]=0.0;
  h_out[row]=1.0/0.0; certified_out[row]=0;
  if(ncand==0) return;

  int have_fallback=0; double fb_coord[DD]; double fb_h; int fb_cert;
  int have_best=0; double best_coord[DD]; double best_h; double best_err=1.0/0.0;

  for(int ci=0; ci<ncand; ++ci){
    int idx=cand[ci];
    const double* center = centers + (size_t)idx*DD;
    double radius=radii[idx]; double L=lips[idx];
    // amortized_warm_start.
    int ok_ws = has_jac[idx] && isfinite(amp) && (amp!=0.0);
    double t_hat[DD]; int produced=0; double coord[DD]; double landing_h; int cert=0;
    if(ok_ws){
      const double* A1 = a1 + (size_t)idx*DD*PP;
      const double* m1 = recon_c + (size_t)idx*PP;
      for(int i=0;i<DD;++i) t_hat[i]=center[i];
      for(int out=0; out<PP; ++out){ double resid=x[out]-amp*m1[out];
        for(int axis=0;axis<DD;++axis) t_hat[axis]+=A1[axis*PP+out]*resid/amp; }
      if(certify_basin(exps, dec, t_hat, x, amp, center, radius, L, coord, &landing_h)){
        produced=1; cert=(isfinite(landing_h) && landing_h<=KANTOROVICH);
      } else { produced=1; for(int i=0;i<DD;++i) coord[i]=0.0; landing_h=1.0/0.0; cert=0; }
    }
    if(!ok_ws){
      // warm start declined: fallback candidate = zeros, uncertified.
      if(!have_fallback){ have_fallback=1; for(int i=0;i<DD;++i) fb_coord[i]=0.0; fb_h=1.0/0.0; fb_cert=0; }
      continue;
    }
    if(!have_fallback){ have_fallback=1; for(int i=0;i<DD;++i) fb_coord[i]=coord[i]; fb_h=landing_h; fb_cert=cert; }
    if(cert){
      // reconstruction error at coord.
      double phi[MM]; double jet[MM*DD]; double hess[MM*DD*DD]; double recon[PP];
      eval_basis(exps, coord, phi, jet, hess); recon_amp1(dec, phi, recon);
      double e2=0.0; for(int c=0;c<PP;++c){ double r=x[c]-amp*recon[c]; e2+=r*r; }
      double err = isfinite(e2)? sqrt(e2) : 1.0/0.0;
      if(!have_best || err<best_err){ have_best=1; best_err=err; best_h=landing_h; for(int i=0;i<DD;++i) best_coord[i]=coord[i]; }
      // global-min short-circuit (mirror production certified_encode_row).
      double xnorm2=0.0; for(int c=0;c<PP;++c) xnorm2+=x[c]*x[c];
      if(best_err <= GMIN_FLOOR*(1.0+sqrt(xnorm2))) break;
    }
    (void)produced;
  }
  if(have_best){
    for(int i=0;i<DD;++i) coords_out[(size_t)row*DD+i]=best_coord[i];
    h_out[row]=best_h; certified_out[row]=1;
  } else if(have_fallback){
    for(int i=0;i<DD;++i) coords_out[(size_t)row*DD+i]=fb_coord[i];
    h_out[row]=fb_h; certified_out[row]=fb_cert;
  }
}
"#;

/// Build the full NVRTC source for one `(d, m, p, topk, newton)`
/// instantiation, prepending the `#define`s so the compile is a pure
/// `compile_ptx_arch` matching `sae_rowjet` / `arrow_schur_nvrtc`.
#[cfg(target_os = "linux")]
#[must_use]
pub fn encode_kernel_source(dev: &EncodeAtomDevice) -> String {
    format!(
        "#define DD {}\n#define MM {}\n#define PP {}\n#define TOPK {}\n#define NEWTON {}\n\
         #define GMIN_FLOOR ({:e})\n#define REFINE_EPS ({:e})\n\
         {ENCODE_KERNEL_SOURCE}",
        dev.d,
        dev.m,
        dev.p,
        dev.topk,
        dev.newton_steps,
        crate::encode::CERTIFIED_GLOBAL_MIN_RECON_FLOOR,
        crate::encode::NEWTON_REFINE_CONVERGED_EPS
    )
}

/// Which path produced the encode result — the #1026/#1551 honesty flag so a
/// caller can ASSERT the device engaged instead of silently falling back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodePath {
    /// The NVRTC `sae_certified_encode` kernel compiled and ran on the device.
    Device,
    /// The host `EncodeAtomDevice` emulator ran (no Linux / no CUDA runtime /
    /// below the launch break-even).
    Cpu,
}

/// Minimum row count below which the device launch is not worth its fixed cost.
pub const DEVICE_ROW_THRESHOLD: usize = 4_096;

/// Measured throughput of the device-resident **exact per-row certified encode**
/// (`sae_certified_encode_batch`) — the literal "batched exact per-row GPU
/// encode" of #988, timed end to end (routing + amortized warm start + basin
/// Newton + Kantorovich certificate + lowest-error assignment/fallback), NOT a
/// component solve like `gam_gpu::encode_throughput::measure_resident_solve_throughput`
/// (which times only the resident normal-equations inner cell).
///
/// The point of this struct is [`Self::decision`]: the #988 surrogate question
/// ("is the exact encode fast enough at 10⁹ rows, or must we distill a certified
/// amortized surrogate?") is answered by *this* measurement and only this one.
/// The decision is keyed on `EncodeDeploymentDecision::from_device_measurement`
/// with `engaged = (path == EncodePath::Device)`, so it inherits that type's
/// anti-green-wash contract: a CPU-emulator run (`path == Cpu`) can NEVER declare
/// the surrogate unneeded — it is honestly [`EncodeDeploymentDecision::Undetermined`]
/// (blocked on hardware), no matter how fast the CPU rate is. Only a real device
/// launch of the exact-encode kernel can move the decision to `Met`/`Unmet`.
#[derive(Debug, Clone, Copy)]
pub struct DeviceEncodeThroughput {
    /// Rows encoded in the timed batch.
    pub n_rows: usize,
    /// Wall-clock seconds for the full exact encode of the batch.
    pub encode_secs: f64,
    /// `n_rows / encode_secs` (`0.0` for a degenerate / non-positive time).
    pub rows_per_sec: f64,
    /// Which path actually ran the encode — the #1026/#1551 honesty flag.
    pub path: EncodePath,
    /// The #988 surrogate decision keyed on THIS exact-encode measurement.
    /// `Met`/`Unmet` only when `path == EncodePath::Device`; a CPU-emulator run
    /// is `Undetermined { NoDeviceEncodeKernel-adjacent }` — a fast CPU number is
    /// never a device pass.
    pub decision: EncodeDeploymentDecision,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_eigh_matches_reference_2x2() {
        // Symmetric 2x2 spectral check: reconstruct A from V diag(vals) Vᵀ.
        let a = [4.0, 1.0, 1.0, 3.0];
        let mut vals = [0.0; 2];
        let mut vecs = [0.0; 4];
        jacobi_eigh(&a, 2, &mut vals, &mut vecs);
        // A_reconstructed[r][c] = Σ_k vals[k] v_k[r] v_k[c].
        for r in 0..2 {
            for c in 0..2 {
                let mut acc = 0.0;
                for k in 0..2 {
                    acc += vals[k] * vecs[k * 2 + r] * vecs[k * 2 + c];
                }
                assert!(
                    (acc - a[r * 2 + c]).abs() < 1e-12,
                    "eig reconstruct {r},{c}"
                );
            }
        }
        // Eigenvalues of [[4,1],[1,3]] are (7±√5)/2.
        let mut vs = vals.to_vec();
        vs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((vs[0] - (7.0 - 5.0_f64.sqrt()) / 2.0).abs() < 1e-12);
        assert!((vs[1] - (7.0 + 5.0_f64.sqrt()) / 2.0).abs() < 1e-12);
    }

}
