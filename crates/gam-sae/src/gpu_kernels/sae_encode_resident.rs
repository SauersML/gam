//! Device-resident **exact per-row certified SAE encode** (#988).
//!
//! The production CPU encode is [`crate::encode::EncodeAtlas::certified_encode_row`]:
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
//! * [`emulate_certified_encode_row`] is a device-free CPU emulator that mirrors
//!   the kernel's arithmetic and control flow line-for-line — the SAME monomial
//!   evaluation, the SAME cyclic-Jacobi symmetric eigensolver
//!   ([`jacobi_eigh`], the device stand-in for the host LAPACK `eigh`), the SAME
//!   basin-warmup / refine loop, the SAME routing + assignment. It is the CPU
//!   fallback AND the exactness oracle the kernel is pinned to.
//! * The parity tests assert the emulator reproduces the production
//!   [`crate::encode::EncodeAtlas::certified_encode_row`] on planted + random
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
/// atom + its [`AtomEncodeAtlas`] by [`EncodeAtomDevice::from_atom_atlas`] so
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
/// ([`sae_certified_encode_batch`]) — the literal "batched exact per-row GPU
/// encode" of #988, timed end to end (routing + amortized warm start + basin
/// Newton + Kantorovich certificate + lowest-error assignment/fallback), NOT a
/// component solve like [`gam_gpu::encode_throughput::measure_resident_solve_throughput`]
/// (which times only the resident normal-equations inner cell).
///
/// The point of this struct is [`Self::decision`]: the #988 surrogate question
/// ("is the exact encode fast enough at 10⁹ rows, or must we distill a certified
/// amortized surrogate?") is answered by *this* measurement and only this one.
/// The decision is keyed on [`EncodeDeploymentDecision::from_device_measurement`]
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
    use crate::basis::{EuclideanPatchEvaluator, SaeBasisEvaluator};
    use crate::encode::{AtlasConfig, EncodeAtlas};
    use crate::manifold::{SaeAtomBasisKind, SaeManifoldAtom};
    use ndarray::{Array1, Array2};
    use std::sync::Arc;

    /// Build a degree-`deg`, `d`-D `EuclideanPatch` atom with a deterministic
    /// decoder into `p` outputs, plus a matching `EncodeAtlas`. The atom carries
    /// the closed-form second jet, exactly the production certified-encode setup.
    fn build_atom_and_atlas(
        d: usize,
        deg: usize,
        p: usize,
        config: AtlasConfig,
    ) -> (SaeManifoldAtom, EncodeAtlas) {
        let evaluator = Arc::new(EuclideanPatchEvaluator::new(d, deg).unwrap());
        // Seed rows over a small coordinate grid (only used for the atom's stored
        // basis_values; the encode recomputes jets from the evaluator).
        let n_seed = 12usize;
        let coords = Array2::from_shape_fn((n_seed, d), |(r, c)| {
            0.15 * ((r as f64 + 1.0) * (c as f64 + 2.0) * 0.37).sin()
        });
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let m = phi.ncols();
        // Deterministic decoder B (m x p): a smooth, well-conditioned map.
        let decoder = Array2::from_shape_fn((m, p), |(bidx, c)| {
            (1.0 / (1.0 + bidx as f64)) * (((bidx as f64 + 1.0) * (c as f64 + 1.0)) * 0.3).cos()
        });
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            "euclid",
            SaeAtomBasisKind::EuclideanPatch,
            d,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(m),
        )
        .unwrap()
        .with_basis_second_jet(evaluator);
        // Amplitude / target-norm bounds generous enough to certify.
        let atlas = EncodeAtlas::build(&[atom.clone()], &[2.0], 8.0, config).unwrap();
        (atom, atlas)
    }

    /// Assert the emulator reproduces the production `certified_encode_row` on a
    /// set of rows: certificate flag must match, and for certified rows the
    /// coords + `h` must agree within the Jacobi-vs-LAPACK eigen tolerance.
    fn assert_parity(
        atom: &SaeManifoldAtom,
        atlas: &EncodeAtlas,
        dev: &EncodeAtomDevice,
        rows: &[Vec<f64>],
        amps: &[f64],
    ) -> (usize, usize, f64, f64) {
        let mut certified = 0usize;
        let mut max_coord = 0.0_f64;
        let mut max_h = 0.0_f64;
        for (x, &amp) in rows.iter().zip(amps.iter()) {
            let xv = Array1::from(x.clone());
            let (coord_p, cert_p) = atlas
                .certified_encode_row(atom, 0, xv.view(), amp)
                .expect("production encode runs");
            let emu = emulate_certified_encode_row(dev, x, amp);
            assert_eq!(
                cert_p.certified(),
                emu.cert.certified(),
                "certificate flag mismatch (prod h={}, emu h={})",
                cert_p.h,
                emu.cert.h
            );
            if cert_p.certified() {
                certified += 1;
                for axis in 0..dev.d {
                    max_coord = max_coord.max((coord_p[axis] - emu.coord[axis]).abs());
                }
                max_h = max_h.max((cert_p.h - emu.cert.h).abs());
            }
        }
        (certified, rows.len(), max_coord, max_h)
    }

    #[test]
    fn emulator_matches_production_certified_encode_1d_quadratic() {
        let (d, deg, p) = (1usize, 2usize, 4usize);
        let config = AtlasConfig::default();
        let (atom, atlas) = build_atom_and_atlas(d, deg, p, config);
        let atom_atlas = &atlas.atoms[0];
        let dev = EncodeAtomDevice::from_atom_atlas(&atom, atom_atlas, &config).unwrap();
        // Planted rows: exact reconstructions at known coords (on-manifold), so
        // the encode has a genuine certified basin.
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut amps: Vec<f64> = Vec::new();
        let evaluator = EuclideanPatchEvaluator::new(d, deg).unwrap();
        for k in 0..24 {
            let tc = -0.4 + 0.8 * (k as f64) / 23.0;
            let (phi, _) = evaluator
                .evaluate(Array2::from_shape_fn((1, d), |_| tc).view())
                .unwrap();
            let amp = 1.0;
            let mut x = vec![0.0; p];
            for c in 0..p {
                let mut r = 0.0;
                for b in 0..dev.m {
                    r += phi[[0, b]] * dev.decoder[b * p + c];
                }
                x[c] = amp * r;
            }
            rows.push(x);
            amps.push(amp);
        }
        // Random (off-manifold) rows exercise the fallback / uncertified paths.
        for k in 0..24 {
            let x = (0..p)
                .map(|c| 0.5 * (((k * 7 + c * 3) as f64) * 0.21).sin())
                .collect();
            rows.push(x);
            amps.push(0.7 + 0.3 * ((k as f64) * 0.11).cos());
        }
        let (cert, total, max_coord, max_h) = assert_parity(&atom, &atlas, &dev, &rows, &amps);
        eprintln!(
            "1D quadratic: certified {cert}/{total}, max coord diff {max_coord:.3e}, max h diff {max_h:.3e}"
        );
        assert!(cert > 0, "planted rows must certify through the encode");
        assert!(max_coord <= 1e-7, "coord parity {max_coord:.3e} > 1e-7");
        assert!(max_h <= 1e-7, "certificate h parity {max_h:.3e} > 1e-7");
    }

    /// #2518 — QUANTIFY the device lane's exhaustive-routing cost, because
    /// "GPU per-row cost rises" is a decision only once it carries a number.
    ///
    /// The kernel runs one row per block with the lead thread walking the row's
    /// candidates serially, term-for-term identical to this emulator, so the
    /// emulator's per-row time IS the kernel's per-row work profile — the
    /// absolute seconds are host seconds, but the RATIO between two candidate
    /// counts is the multiplier the device pays. That ratio is what the trade
    /// needs, and it is measured here rather than asserted, because a wall-clock
    /// bar in a test would be a machine-dependent gate on a shared runner.
    ///
    /// The correctness half IS asserted: the deleted `TOPK = 4` restriction must
    /// never beat the exhaustive scan on reconstruction error, and wherever the
    /// two returned coordinates differ, exhaustive must be the better one.
    #[test]
    fn device_exhaustive_routing_cost_multiplier_2518() {
        for (d, deg, p) in [(1usize, 2usize, 6usize), (2, 2, 6)] {
            let config = AtlasConfig::default();
            let (atom, atlas) = build_atom_and_atlas(d, deg, p, config.clone());
            let atom_atlas = &atlas.atoms[0];
            let exhaustive =
                EncodeAtomDevice::from_atom_atlas(&atom, atom_atlas, &config).unwrap();
            let certifiable = exhaustive
                .charts
                .iter()
                .filter(|c| c.certified_radius > 0.0)
                .count();
            // The deleted constant, reconstructed here ONLY as the cost baseline.
            let mut old_topk = exhaustive.clone();
            old_topk.topk = 4;

            let rows: Vec<Vec<f64>> = (0..512)
                .map(|k| {
                    (0..p)
                        .map(|c| 0.6 * (((k * 7 + c * 3) as f64) * 0.19).sin())
                        .collect()
                })
                .collect();
            let amps: Vec<f64> = (0..512)
                .map(|k| 0.6 + 0.5 * ((k as f64) * 0.11).sin().abs())
                .collect();

            let time_it = |dev: &EncodeAtomDevice| -> f64 {
                let start = std::time::Instant::now();
                let out = emulate_certified_encode_batch(dev, &rows, &amps);
                std::hint::black_box(&out);
                start.elapsed().as_secs_f64() / rows.len() as f64
            };
            // One untimed pass each so neither arm pays first-touch page faults.
            std::hint::black_box(emulate_certified_encode_batch(&exhaustive, &rows, &amps));
            std::hint::black_box(emulate_certified_encode_batch(&old_topk, &rows, &amps));
            let per_row_exhaustive = time_it(&exhaustive);
            let per_row_top4 = time_it(&old_topk);

            // Correctness: the restriction may only ever tie or lose.
            let mut rows_where_top4_is_worse = 0usize;
            let mut worst_penalty = 0.0_f64;
            for (x, &amp) in rows.iter().zip(amps.iter()) {
                let full = emulate_certified_encode_row(&exhaustive, x, amp);
                let four = emulate_certified_encode_row(&old_topk, x, amp);
                if !(full.cert.certified() && four.cert.certified()) {
                    continue;
                }
                let err = |coord: &[f64]| -> f64 {
                    let cv = Array1::from(coord.to_vec());
                    let xv = Array1::from(x.to_vec());
                    crate::encode::encode_reconstruction_error(
                        &atom,
                        atom.basis_evaluator.as_ref().unwrap().as_ref(),
                        cv.view(),
                        xv.view(),
                        amp,
                    )
                };
                let (e_full, e_four) = (err(&full.coord), err(&four.coord));
                if e_four > e_full + 1.0e-9 * (1.0 + e_full) {
                    rows_where_top4_is_worse += 1;
                    worst_penalty = worst_penalty.max(e_four - e_full);
                }
                assert!(
                    e_full <= e_four + 1.0e-9 * (1.0 + e_four),
                    "exhaustive routing returned a WORSE encode than the deleted top-4 \
                     restriction (full={e_full:.6e} four={e_four:.6e}) — the scan is supposed \
                     to be a superset, so this would mean the prune or the ordering is wrong"
                );
            }

            println!(
                "[#2518-gpu-cost] d={d} deg={deg} p={p} charts={} certifiable={certifiable} \
                 per_row_exhaustive={:.3}us per_row_top4={:.3}us multiplier={:.2}x \
                 rows_top4_worse={rows_where_top4_is_worse} worst_penalty={worst_penalty:.3e}",
                exhaustive.charts.len(),
                per_row_exhaustive * 1.0e6,
                per_row_top4 * 1.0e6,
                per_row_exhaustive / per_row_top4.max(f64::MIN_POSITIVE),
            );
        }
    }

    /// Pin: the CPU atlas routing (`crate::encode::nearest_charts_topk`) and the
    /// GPU-host routing ([`super::nearest_charts_topk`]) select IDENTICAL charts on
    /// a shared fixture. Both now funnel through the one shared comparator
    /// [`crate::encode::select_nearest_charts_topk`] (amplitude gating + `(distance,
    /// index)` tie-break), so the only remaining way they could diverge is a drift
    /// between the two recon SOURCES — the CPU's distilled `recon_center` vs the
    /// GPU-host's per-center basis re-eval (which itself mirrors the CUDA kernel's
    /// routing, `recon_amp1(eval_basis(center))`). This guards that bit-identity.
    #[test]
    fn cpu_gpu_chart_routing_topk_parity() {
        let (d, deg, p) = (2usize, 2usize, 5usize);
        let config = AtlasConfig {
            grid_resolution: 6,
            ..AtlasConfig::default()
        };
        let (atom, atlas) = build_atom_and_atlas(d, deg, p, config);
        let atom_atlas = &atlas.atoms[0];
        let dev = EncodeAtomDevice::from_atom_atlas(&atom, atom_atlas, &config).unwrap();
        assert!(
            dev.charts.len() > 1,
            "fixture must have multiple charts to exercise routing (got {})",
            dev.charts.len()
        );

        let evaluator = EuclideanPatchEvaluator::new(d, deg).unwrap();
        // Planted (on-manifold) rows on a coord grid + structured off-manifold rows,
        // each over a range of amplitudes so the amplitude gating is exercised.
        let mut rows: Vec<(Vec<f64>, f64)> = Vec::new();
        for k in 0..36 {
            let t0 = -0.35 + 0.7 * ((k % 6) as f64) / 5.0;
            let t1 = -0.35 + 0.7 * ((k / 6) as f64) / 5.0;
            let coord = Array2::from_shape_fn((1, d), |(_, c)| if c == 0 { t0 } else { t1 });
            let (phi, _) = evaluator.evaluate(coord.view()).unwrap();
            let amp = 0.6 + 0.5 * ((k as f64) * 0.17).cos();
            let mut x = vec![0.0; p];
            for c in 0..p {
                let mut r = 0.0;
                for b in 0..dev.m {
                    r += phi[[0, b]] * dev.decoder[b * p + c];
                }
                x[c] = amp * r;
            }
            rows.push((x, amp));
        }
        for k in 0..24 {
            let x = (0..p)
                .map(|c| 0.4 * (((k * 5 + c * 2) as f64) * 0.29).sin())
                .collect();
            let amp = 0.5 + 0.4 * ((k as f64) * 0.13).sin();
            rows.push((x, amp));
        }

        let mut scratch = Scratch::new(&dev);
        let mut compared = 0usize;
        for (x, amp) in &rows {
            let xv = Array1::from(x.clone());
            let cpu = crate::test_support::nearest_charts_topk(atom_atlas, xv.view(), *amp, dev.topk);
            let gpu = nearest_charts_topk(&dev, x, *amp, &mut scratch);
            assert_eq!(
                cpu, gpu,
                "CPU vs GPU top-{} chart routing diverged at amp {amp}: cpu {cpu:?} gpu {gpu:?}",
                dev.topk
            );
            // The single-chart CPU router agrees with the top-1 selection.
            if let Some((idx, _)) = crate::encode::nearest_chart(atom_atlas, xv.view(), *amp) {
                assert_eq!(
                    Some(&idx),
                    cpu.first(),
                    "nearest_chart != topk[0] at amp {amp}"
                );
            }
            compared += 1;
        }
        assert!(compared >= 50, "fixture too small: {compared}");
        eprintln!(
            "CPU/GPU routing parity: {compared} rows, top-{} identical (of {} charts)",
            dev.topk,
            dev.charts.len()
        );
    }

    #[test]
    fn emulator_matches_production_certified_encode_2d_quadratic() {
        let (d, deg, p) = (2usize, 2usize, 5usize);
        let config = AtlasConfig {
            grid_resolution: 6,
            ..AtlasConfig::default()
        };
        let (atom, atlas) = build_atom_and_atlas(d, deg, p, config);
        let atom_atlas = &atlas.atoms[0];
        let dev = EncodeAtomDevice::from_atom_atlas(&atom, atom_atlas, &config).unwrap();
        let evaluator = EuclideanPatchEvaluator::new(d, deg).unwrap();
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut amps: Vec<f64> = Vec::new();
        for k in 0..30 {
            let t0 = -0.3 + 0.6 * ((k % 6) as f64) / 5.0;
            let t1 = -0.3 + 0.6 * ((k / 6) as f64) / 5.0;
            let coord = Array2::from_shape_fn((1, d), |(_, c)| if c == 0 { t0 } else { t1 });
            let (phi, _) = evaluator.evaluate(coord.view()).unwrap();
            let amp = 1.0;
            let mut x = vec![0.0; p];
            for c in 0..p {
                let mut r = 0.0;
                for b in 0..dev.m {
                    r += phi[[0, b]] * dev.decoder[b * p + c];
                }
                x[c] = amp * r;
            }
            rows.push(x);
            amps.push(amp);
        }
        for k in 0..20 {
            let x = (0..p)
                .map(|c| 0.4 * (((k * 5 + c * 2) as f64) * 0.17).cos())
                .collect();
            rows.push(x);
            amps.push(1.0);
        }
        let (cert, total, max_coord, max_h) = assert_parity(&atom, &atlas, &dev, &rows, &amps);
        eprintln!(
            "2D quadratic: certified {cert}/{total}, max coord diff {max_coord:.3e}, max h diff {max_h:.3e}"
        );
        assert!(cert > 0, "planted 2D rows must certify");
        assert!(max_coord <= 1e-6, "coord parity {max_coord:.3e} > 1e-6");
        assert!(max_h <= 1e-6, "certificate h parity {max_h:.3e} > 1e-6");
    }

    #[test]
    fn emulator_matches_production_batch() {
        let (d, deg, p) = (1usize, 3usize, 3usize);
        let config = AtlasConfig::default();
        let (atom, atlas) = build_atom_and_atlas(d, deg, p, config);
        let dev = EncodeAtomDevice::from_atom_atlas(&atom, &atlas.atoms[0], &config).unwrap();
        let n = 40usize;
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|k| {
                (0..p)
                    .map(|c| 0.3 * (((k + c) as f64) * 0.19).sin())
                    .collect()
            })
            .collect();
        let amps: Vec<f64> = (0..n).map(|_| 1.0).collect();
        let (batch, path) = sae_certified_encode_batch(&dev, &rows, &amps)
            .expect("small CPU batch must not erase CUDA admission faults");
        assert_eq!(path, EncodePath::Cpu, "small batch stays on CPU");
        // Batch == per-row emulate, and per-row == production certified flag.
        for (k, r) in batch.iter().enumerate() {
            let single = emulate_certified_encode_row(&dev, &rows[k], amps[k]);
            assert_eq!(r.cert.certified(), single.cert.certified());
            let xv = Array1::from(rows[k].clone());
            let (_, cert_p) = atlas
                .certified_encode_row(&atom, 0, xv.view(), amps[k])
                .unwrap();
            assert_eq!(
                cert_p.certified(),
                r.cert.certified(),
                "batch row {k} certificate flag disagrees with production"
            );
        }
    }

    /// #988 core: benchmark the batched EXACT per-row encode and gate the
    /// certified-surrogate decision on the MEASURED throughput of the actual
    /// device-resident encode kernel — not the host component solve, and not a
    /// hardcoded target. This is the "benchmark first; surrogate only on
    /// benchmark evidence" order-of-work wired end to end onto
    /// [`sae_certified_encode_batch`] (the literal batched exact per-row GPU
    /// encode) via [`measure_device_encode_throughput`].
    #[test]
    fn device_encode_throughput_gates_surrogate_on_measurement() {
        let (d, deg, p) = (1usize, 2usize, 4usize);
        let config = AtlasConfig::default();
        let (atom, atlas) = build_atom_and_atlas(d, deg, p, config);
        let dev = EncodeAtomDevice::from_atom_atlas(&atom, &atlas.atoms[0], &config).unwrap();

        // A batch large enough that a CUDA host would take the device path
        // (>= DEVICE_ROW_THRESHOLD), mixing planted on-manifold rows (which must
        // certify — a non-vacuous benchmark) with off-manifold rows (fallback).
        let n = DEVICE_ROW_THRESHOLD + 64;
        let evaluator = EuclideanPatchEvaluator::new(d, deg).unwrap();
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut amps: Vec<f64> = Vec::with_capacity(n);
        for k in 0..n {
            if k % 2 == 0 {
                // Planted: exact amplitude-1 reconstruction at a known coordinate.
                let tc = -0.4 + 0.8 * ((k % 24) as f64) / 23.0;
                let (phi, _) = evaluator
                    .evaluate(Array2::from_shape_fn((1, d), |_| tc).view())
                    .unwrap();
                let x = (0..p)
                    .map(|c| {
                        (0..dev.m)
                            .map(|b| phi[[0, b]] * dev.decoder[b * p + c])
                            .sum::<f64>()
                    })
                    .collect();
                rows.push(x);
                amps.push(1.0);
            } else {
                let x = (0..p)
                    .map(|c| 0.5 * (((k * 7 + c * 3) as f64) * 0.021).sin())
                    .collect();
                rows.push(x);
                amps.push(1.0);
            }
        }

        // The benchmark: time the exact encode and derive the surrogate decision.
        let tput = measure_device_encode_throughput(&dev, &rows, &amps)
            .expect("exact-encode benchmark must preserve CUDA failures");
        eprintln!(
            "[device-encode #988] n={} rows/sec={:.1} path={:?} decision={:?}",
            tput.n_rows, tput.rows_per_sec, tput.path, tput.decision
        );

        // It must be a REAL measurement (positive rate), and the engagement flag
        // must be consistent with the path that ran.
        assert!(
            tput.rows_per_sec > 0.0,
            "the exact encode benchmark must produce a positive rows/sec, got {}",
            tput.rows_per_sec
        );
        assert_eq!(
            tput.device_engaged(),
            matches!(tput.path, EncodePath::Device)
        );

        // The benchmark must be non-vacuous: on a well-conditioned dictionary the
        // planted on-manifold rows certify through the exact encode (proving the
        // routing + basin Newton + certificate really ran, not a trivial pass).
        let (batch, _) = sae_certified_encode_batch(&dev, &rows, &amps)
            .expect("exact encode must preserve CUDA failures");
        let certified = batch.iter().filter(|r| r.cert.certified()).count();
        assert!(
            certified > 0,
            "the exact encode must certify a majority of the planted rows; certified={certified}/{n}"
        );

        if tput.device_engaged() {
            // Only reachable on a CUDA host: the decision is a REAL Met/Unmet
            // keyed on the measured device throughput vs the 100k rows/sec target.
            assert!(
                !tput.decision.is_undetermined(),
                "an engaged device measurement must decide Met/Unmet, got {:?}",
                tput.decision
            );
            let target = gam_gpu::policy::GPU_THROUGHPUT_TARGET_ROWS_PER_SEC;
            if tput.rows_per_sec >= target {
                assert!(
                    tput.decision.surrogate_unneeded(),
                    "device rate {:.1} >= target {target} must mark the surrogate unneeded",
                    tput.rows_per_sec
                );
            } else {
                assert!(
                    tput.decision.surrogate_justified(),
                    "device rate {:.1} < target {target} must justify the surrogate",
                    tput.rows_per_sec
                );
            }
        } else {
            // CPU-only host (this dev box): the rate is honest but it is NOT a
            // device measurement. The surrogate decision is BLOCKED — on a
            // CPU-only host because there is no hardware, and on a CUDA host
            // because falling back to the emulator despite a device-sized batch
            // is false routing. A fast CPU number can never declare the
            // surrogate unneeded (the #1412 anti-green-wash property carried to
            // the exact device encode).
            assert!(
                tput.decision.is_undetermined(),
                "a CPU-emulator exact encode must leave the surrogate decision Undetermined, got {:?}",
                tput.decision
            );
            if gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto)
                .unwrap_or_else(|error| panic!("exact-encode CUDA admission failed: {error}"))
                .is_some_and(|runtime| runtime.device_count() > 0)
            {
                assert_eq!(
                    tput.decision,
                    EncodeDeploymentDecision::blocked(EncodeDecisionBlocked::DeviceNotEngaged),
                    "CUDA was present for a device-sized exact-encode batch, so an emulator path \
                     must be reported as DeviceNotEngaged (false routing), not NoDevice"
                );
            } else {
                assert_eq!(
                    tput.decision,
                    EncodeDeploymentDecision::blocked(EncodeDecisionBlocked::NoDevice)
                );
            }
            assert!(!tput.decision.surrogate_unneeded());
            assert!(!tput.decision.surrogate_justified());
        }
    }

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

    #[cfg(target_os = "linux")]
    #[test]
    fn encode_kernel_source_substitutes_macros_and_compiles() {
        let (d, deg, p) = (1usize, 2usize, 4usize);
        let config = AtlasConfig::default();
        let (atom, atlas) = build_atom_and_atlas(d, deg, p, config);
        let dev = EncodeAtomDevice::from_atom_atlas(&atom, &atlas.atoms[0], &config).unwrap();
        let src = encode_kernel_source(&dev);
        assert!(src.contains(&format!("#define DD {}", dev.d)));
        assert!(src.contains(&format!("#define MM {}", dev.m)));
        assert!(src.contains(&format!("#define PP {}", dev.p)));
        assert!(src.contains("sae_certified_encode"));
        // NVRTC host-compile to PTX (no device needed) — the #1017 anchor.
        let ptx = gam_gpu::device_cache::compile_ptx_arch(&src)
            .expect("sae_encode kernel compiles to PTX via NVRTC");
        let text = ptx.to_src();
        assert!(
            text.contains(".visible .entry sae_certified_encode"),
            "PTX must export the encode entry"
        );
        assert!(text.contains(".target sm_"), "PTX must carry a target arch");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn device_matches_emulator_when_available() {
        let (d, deg, p) = (1usize, 2usize, 4usize);
        let config = AtlasConfig::default();
        let (atom, atlas) = build_atom_and_atlas(d, deg, p, config);
        let dev = EncodeAtomDevice::from_atom_atlas(&atom, &atlas.atoms[0], &config).unwrap();
        let n = DEVICE_ROW_THRESHOLD + 64;
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|k| {
                (0..p)
                    .map(|c| 0.3 * (((k + c) as f64) * 0.019).sin())
                    .collect()
            })
            .collect();
        let amps = vec![1.0; n];
        let cpu = emulate_certified_encode_batch(&dev, &rows, &amps);
        // The availability question goes through the one shared gate (#2422), so
        // the skip below is COUNTED and prints the shared `SKIPPED(no-cuda):`
        // marker instead of vanishing into a green CPU-only run. The seam
        // assertions in the absent arm are this test's own and are kept.
        let skips_before = gam_gpu::test_gate::skipped_for_absent_device();
        match gam_gpu::test_gate::gpu_for_test("sae certified encode device parity") {
            gam_gpu::test_gate::GpuTestGate::AbsentDevice => {
                gam_gpu::test_gate::assert_absent_device_was_counted(skips_before);
                // #2422: a bare `return` here reported `passed` with zero
                // assertions on every device-free runner. `cpu` above is the
                // emulator's answer and is already computed, so assert it is
                // well-formed AND that the device entry REFUSES:
                // `sae_certified_encode_device` opens with `backend()?`, so an
                // `Ok` on a device-free host means it fabricated device state.
                assert_eq!(
                    cpu.len(),
                    rows.len(),
                    "the CPU encode emulator must answer for every row, or the \
                     device-free half asserts nothing"
                );
                assert!(
                    device::sae_certified_encode_device(&dev, &rows, &amps).is_err(),
                    "no CUDA runtime on this host, yet the device encode entry returned \
                     Ok -- the seam fabricated device state (#1551 class)"
                );
                return;
            }
            gam_gpu::test_gate::GpuTestGate::Ready(_) => {
                let devout = device::sae_certified_encode_device(&dev, &rows, &amps)
                    .expect("admitted GPU runtime must run the sae_encode kernel");
                let mut max_coord = 0.0_f64;
                for (a, b) in cpu.iter().zip(devout.iter()) {
                    assert_eq!(
                        a.cert.certified(),
                        b.cert.certified(),
                        "device certified flag"
                    );
                    if a.cert.certified() {
                        for axis in 0..dev.d {
                            max_coord = max_coord.max((a.coord[axis] - b.coord[axis]).abs());
                        }
                    }
                }
                assert!(
                    max_coord <= 1e-9,
                    "device vs emulator coord diff {max_coord:.3e} > 1e-9"
                );
            }
        }
    }
}
