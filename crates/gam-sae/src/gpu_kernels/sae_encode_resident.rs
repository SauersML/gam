//! Device-resident **exact per-row certified SAE encode** (#988).
//!
//! The production CPU encode is [`crate::encode::EncodeAtlas::certified_encode_row`]:
//! for one atom and one target row `x` at fixed amplitude `z` it
//!
//!   1. **routes** the row to the `topk` nearest certified charts by ambient
//!      reconstruction distance `‖BᵀΦ(t_c) − x‖²` (the *active-set routing*),
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

use std::time::Instant;

use crate::encode::{
    AtlasConfig, AtomEncodeAtlas, KANTOROVICH_THRESHOLD, euclidean_patch_degree,
};
use crate::manifold::SaeManifoldAtom;
use gam_gpu::policy::{EncodeDecisionBlocked, EncodeDeploymentDecision};

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
    /// Number of nearest charts refined per row (`CERTIFIED_ROUTING_TOPK`).
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

impl EncodeAtomDevice {
    /// Extract the device encode data from a real `EuclideanPatch` atom and its
    /// offline atlas. Recomputes the monomial exponent table (the atom's own
    /// basis design) so the on-device evaluation is the SAME polynomial the host
    /// `EuclideanPatchEvaluator` evaluates.
    pub fn from_atom_atlas(
        atom: &SaeManifoldAtom,
        atom_atlas: &AtomEncodeAtlas,
        config: &AtlasConfig,
    ) -> Result<Self, String> {
        let d = atom.latent_dim;
        let p = atom.output_dim();
        let m = atom.basis_size();
        let degree = euclidean_patch_degree(d, m);
        let exps = gam_terms::basis::monomial_exponents(d, degree);
        if exps.len() != m {
            return Err(format!(
                "EncodeAtomDevice::from_atom_atlas: monomial table len {} != basis_size {m} \
                 (atom is not a EuclideanPatch degree-{degree} monomial family)",
                exps.len()
            ));
        }
        let mut exponents = vec![0_i32; m * d];
        for (col, alpha) in exps.iter().enumerate() {
            for axis in 0..d {
                exponents[col * d + axis] = alpha[axis] as i32;
            }
        }
        let dec = &atom.decoder_coefficients;
        if dec.dim() != (m, p) {
            return Err(format!(
                "EncodeAtomDevice::from_atom_atlas: decoder dim {:?} != ({m}, {p})",
                dec.dim()
            ));
        }
        let mut decoder = vec![0.0_f64; m * p];
        for b in 0..m {
            for c in 0..p {
                decoder[b * p + c] = dec[[b, c]];
            }
        }
        let mut charts = Vec::with_capacity(atom_atlas.charts.len());
        for chart in &atom_atlas.charts {
            let center = chart.region.center.to_vec();
            if center.len() != d {
                return Err(format!(
                    "EncodeAtomDevice::from_atom_atlas: chart center len {} != d {d}",
                    center.len()
                ));
            }
            let (has_jacobian, amortized_jacobian) = match &chart.amortized_jacobian {
                Some(a1) => {
                    if a1.dim() != (d, p) {
                        return Err(format!(
                            "EncodeAtomDevice::from_atom_atlas: A1 dim {:?} != ({d}, {p})",
                            a1.dim()
                        ));
                    }
                    let mut flat = vec![0.0_f64; d * p];
                    for axis in 0..d {
                        for out in 0..p {
                            flat[axis * p + out] = a1[[axis, out]];
                        }
                    }
                    (true, flat)
                }
                None => (false, Vec::new()),
            };
            let recon_center = chart.recon_center.to_vec();
            charts.push(EncodeChartDevice {
                center,
                radius: chart.region.radius,
                certified_radius: chart.certified_radius,
                lipschitz: chart.lipschitz,
                has_jacobian,
                amortized_jacobian,
                recon_center,
            });
        }
        Ok(Self {
            d,
            m,
            p,
            topk: crate::encode::CERTIFIED_ROUTING_TOPK,
            newton_steps: config.newton_steps,
            exponents,
            decoder,
            charts,
        })
    }
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
    #[inline]
    fn uncertified(lipschitz: f64) -> Self {
        Self {
            beta: f64::INFINITY,
            eta: f64::INFINITY,
            lipschitz,
            h: f64::INFINITY,
        }
    }
    #[inline]
    fn uncertified_inf() -> Self {
        Self {
            beta: f64::INFINITY,
            eta: f64::INFINITY,
            lipschitz: f64::INFINITY,
            h: f64::INFINITY,
        }
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

/// `base^exp` by exponentiation-by-squaring, matching `f64::powi` (LLVM
/// `llvm.powi.f64`) so the monomial evaluation is bit-identical to the host
/// `EuclideanPatchEvaluator`. Used by the emulator; the kernel `dpow` mirror is
/// the same algorithm.
#[inline]
fn dpow(base: f64, exp: i32) -> f64 {
    // The production monomial code calls `coords.powi(exp)`; using the SAME
    // intrinsic here keeps phi/jet/hess bit-identical to production.
    base.powi(exp)
}

/// Monomial basis value/first/second jets at one coordinate `t` (length `d`).
/// Mirrors [`crate::basis::EuclideanPatchEvaluator::evaluate`] +
/// [`crate::basis::EuclideanPatchEvaluator::second_jet`] (the same falling-
/// factorial monomial derivatives), producing:
///   `phi[col]`, `jet[col*d + axis]`, `hess[(col*d + a)*d + c]`.
fn eval_basis(dev: &EncodeAtomDevice, t: &[f64], phi: &mut [f64], jet: &mut [f64], hess: &mut [f64]) {
    let (d, m) = (dev.d, dev.m);
    let exp = &dev.exponents;
    for col in 0..m {
        // value = Π_axis t_axis^{α_axis}
        let mut value = 1.0_f64;
        for axis in 0..d {
            let e = exp[col * d + axis];
            if e != 0 {
                value *= dpow(t[axis], e);
            }
        }
        phi[col] = value;
        // first jet: ∂/∂t_axis = α_axis · Π_a t_a^{(a==axis? α_a-1 : α_a)}
        for axis in 0..d {
            let a_axis = exp[col * d + axis];
            let mut jval = 0.0_f64;
            if a_axis != 0 {
                jval = a_axis as f64;
                for a in 0..d {
                    let ea = if a == axis { a_axis - 1 } else { exp[col * d + a] };
                    if ea != 0 {
                        jval *= dpow(t[a], ea);
                    }
                }
            }
            jet[col * d + axis] = jval;
        }
        // second jet: ∂²/∂t_a∂t_c (falling factorial), else 0.
        for a in 0..d {
            for c in 0..d {
                let mut hval = 0.0_f64;
                let aa = exp[col * d + a];
                let ac = exp[col * d + c];
                let admissible = aa != 0 && (a == c || ac != 0);
                if admissible {
                    let lead = if a == c {
                        (aa as f64) * ((aa - 1).max(0) as f64)
                    } else {
                        (aa as f64) * (ac as f64)
                    };
                    if lead != 0.0 {
                        hval = lead;
                        for axis in 0..d {
                            let mut e = exp[col * d + axis];
                            if axis == a {
                                e = (e - 1).max(0);
                            }
                            if axis == c {
                                e = (e - 1).max(0);
                            }
                            if e != 0 {
                                hval *= dpow(t[axis], e);
                            }
                        }
                    }
                }
                hess[(col * d + a) * d + c] = hval;
            }
        }
    }
}

/// Amplitude-1 reconstruction `m₁(t) = BᵀΦ(t)` from precomputed `phi`.
/// (Routing + reconstruction-error use this; `nearest_chart` mirror.)
fn recon_amp1(dev: &EncodeAtomDevice, phi: &[f64], out: &mut [f64]) {
    let (m, p) = (dev.m, dev.p);
    for c in 0..p {
        out[c] = 0.0;
    }
    for b in 0..m {
        let pv = phi[b];
        if pv == 0.0 {
            continue;
        }
        for c in 0..p {
            out[c] += pv * dev.decoder[b * p + c];
        }
    }
}

/// Evaluated basis buffers at a point `t`: value `Φ`, first jet `∂Φ`, and the
/// second jet `∂²Φ`. Bundled so [`encode_grad_hess`] takes them as one argument.
struct EvaluatedBasis<'a> {
    phi: &'a [f64],
    jet: &'a [f64],
    hess: &'a [f64],
}

/// Gradient `g` and FULL, TRUE Hessian `H` of the encode objective at `t`.
/// Mirror of [`crate::encode::encode_grad_hess`]:
///   `g[a] = Jₘ[a]·r`,  `H[a,b] = Jₘ[a]·Jₘ[b] + z·Σ ∂²Φ·(r·B)`,
/// with `m = z·BᵀΦ`, `r = m − x`, `Jₘ = z·BᵀJ_Φ`. NO Levenberg ridge is added:
/// the certificate must see the genuine field (F2), exactly as production's
/// `encode_grad_hess` (a ridged `H + λI` would falsely certify a singular,
/// non-isolated root). For the monomial family the
/// second jet always exists, so this never returns "no certificate".
fn encode_grad_hess(
    dev: &EncodeAtomDevice,
    x: &[f64],
    amplitude: f64,
    be: &EvaluatedBasis<'_>,
    g: &mut [f64],
    h: &mut [f64],
) {
    let (phi, jet, hess) = (be.phi, be.jet, be.hess);
    let (d, m, p) = (dev.d, dev.m, dev.p);
    // recon m(t) = z·BᵀΦ ; residual r = m − x
    let mut recon = vec![0.0_f64; p];
    for b in 0..m {
        let pv = phi[b];
        if pv == 0.0 {
            continue;
        }
        for c in 0..p {
            recon[c] += amplitude * pv * dev.decoder[b * p + c];
        }
    }
    let mut residual = vec![0.0_f64; p];
    for c in 0..p {
        residual[c] = recon[c] - x[c];
    }
    // Jₘ[axis][out] = z·Bᵀ ∂Φ/∂t_axis  (row-major jm[axis*p + out])
    let mut jm = vec![0.0_f64; d * p];
    for axis in 0..d {
        for b in 0..m {
            let dphi = jet[b * d + axis];
            if dphi == 0.0 {
                continue;
            }
            for c in 0..p {
                jm[axis * p + c] += amplitude * dphi * dev.decoder[b * p + c];
            }
        }
    }
    // g[a] = Jₘ[a]·r ; H[a,b] = Jₘ[a]·Jₘ[b] + z·Σ_b ∂²Φ·(r·B)
    for a in 0..d {
        let mut ga = 0.0;
        for c in 0..p {
            ga += jm[a * p + c] * residual[c];
        }
        g[a] = ga;
        for b in 0..d {
            let mut hab = 0.0;
            for c in 0..p {
                hab += jm[a * p + c] * jm[b * p + c];
            }
            let mut curv = 0.0;
            for basis in 0..m {
                let d2 = hess[(basis * d + a) * d + b];
                if d2 == 0.0 {
                    continue;
                }
                let mut dot = 0.0;
                for c in 0..p {
                    dot += residual[c] * dev.decoder[basis * p + c];
                }
                curv += amplitude * d2 * dot;
            }
            hab += curv;
            h[a * d + b] = hab;
        }
    }
    // NO ridge: the certificate uses the TRUE Hessian (F2). See the doc above.
}

/// Cyclic Jacobi symmetric eigensolver for a `d×d` matrix (row-major, `d ≤ 8`).
/// Returns eigenvalues `vals[i]` and eigenvectors as COLUMNS
/// `vecs[col*d + row]`. This is the device stand-in for the host LAPACK `eigh`
/// used by [`crate::encode::beta_eta_newton`]; the Newton step is reconstructed
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

/// `(β, η, δ)` from the full Hessian `H` and gradient `g`. Mirror of
/// [`crate::encode::beta_eta_newton`]: `β = 1/λ_min`, `δ = −Σ_i (vᵢᵀg/λᵢ)vᵢ`,
/// `η = ‖δ‖`; `None` when `λ_min ≤ 0` (uncertifiable start).
fn beta_eta_newton(h: &[f64], g: &[f64], d: usize) -> Option<(f64, f64, Vec<f64>)> {
    let mut vals = vec![0.0_f64; d];
    let mut vecs = vec![0.0_f64; d * d];
    jacobi_eigh(h, d, &mut vals, &mut vecs);
    let mut lambda_min = f64::INFINITY;
    for &v in &vals {
        if v < lambda_min {
            lambda_min = v;
        }
    }
    if !(lambda_min.is_finite() && lambda_min > 0.0) {
        return None;
    }
    let beta = 1.0 / lambda_min;
    let mut delta = vec![0.0_f64; d];
    for col in 0..d {
        let lam = vals[col];
        if lam <= 0.0 {
            return None;
        }
        // vᵀg
        let mut vg = 0.0;
        for row in 0..d {
            vg += vecs[col * d + row] * g[row];
        }
        let coeff = vg / lam;
        for row in 0..d {
            delta[row] -= coeff * vecs[col * d + row];
        }
    }
    let mut eta = 0.0;
    for row in 0..d {
        eta += delta[row] * delta[row];
    }
    Some((beta, eta.sqrt(), delta))
}

/// Certificate + Newton step at `t`. Mirror of [`crate::encode::row_certificate`].
fn row_certificate(
    dev: &EncodeAtomDevice,
    t: &[f64],
    x: &[f64],
    amplitude: f64,
    lipschitz: f64,
    scratch: &mut Scratch,
) -> (DeviceRowCertificate, Vec<f64>) {
    let d = dev.d;
    eval_basis(dev, t, &mut scratch.phi, &mut scratch.jet, &mut scratch.hess);
    encode_grad_hess(
        dev,
        x,
        amplitude,
        &EvaluatedBasis {
            phi: &scratch.phi,
            jet: &scratch.jet,
            hess: &scratch.hess,
        },
        &mut scratch.g,
        &mut scratch.h,
    );
    match beta_eta_newton(&scratch.h, &scratch.g, d) {
        Some((beta, eta, delta)) => (
            DeviceRowCertificate {
                beta,
                eta,
                lipschitz,
                h: beta * eta * lipschitz,
            },
            delta,
        ),
        None => (
            DeviceRowCertificate::uncertified(lipschitz),
            vec![0.0_f64; d],
        ),
    }
}

/// Per-row working buffers (register/stack arrays in the kernel).
struct Scratch {
    phi: Vec<f64>,
    jet: Vec<f64>,
    hess: Vec<f64>,
    g: Vec<f64>,
    h: Vec<f64>,
}

impl Scratch {
    fn new(dev: &EncodeAtomDevice) -> Self {
        let (d, m) = (dev.d, dev.m);
        Self {
            phi: vec![0.0; m],
            jet: vec![0.0; m * d],
            hess: vec![0.0; m * d * d],
            g: vec![0.0; d],
            h: vec![0.0; d * d],
        }
    }
}

#[inline]
fn in_chart(t: &[f64], center: &[f64], radius: f64) -> bool {
    let mut r2 = 0.0;
    for i in 0..t.len() {
        let dlt = t[i] - center[i];
        r2 += dlt * dlt;
    }
    r2 <= radius * radius
}

/// Basin-warmup + refine from `t_start`. Mirror of
/// [`crate::encode::certify_with_basin_warmup`] composed with
/// `refine_certified_start`: navigate into the `h ≤ ½` basin (staying in-chart,
/// requiring `h` to contract), then take `newton_steps` refine steps that must
/// all stay certified. Returns `(coord, final_cert)` — the certificate at the
/// REFINED landing coordinate (F5) — or `None`.
fn certify_with_basin_warmup(
    dev: &EncodeAtomDevice,
    mut t: Vec<f64>,
    x: &[f64],
    amplitude: f64,
    chart: &EncodeChartDevice,
    scratch: &mut Scratch,
) -> Option<(Vec<f64>, DeviceRowCertificate)> {
    if !in_chart(&t, &chart.center, chart.radius) {
        return None;
    }
    let (mut cert, mut delta) =
        row_certificate(dev, &t, x, amplitude, chart.lipschitz, scratch);
    while !cert.certified() {
        if !(cert.h.is_finite() && cert.beta.is_finite() && cert.eta.is_finite()) {
            return None;
        }
        let prev_h = cert.h;
        let mut next = t.clone();
        for i in 0..dev.d {
            next[i] += delta[i];
        }
        if !in_chart(&next, &chart.center, chart.radius) {
            return None;
        }
        t = next;
        let (nc, nd) = row_certificate(dev, &t, x, amplitude, chart.lipschitz, scratch);
        cert = nc;
        delta = nd;
        if !cert.h.is_finite() || cert.h >= prev_h {
            return None;
        }
    }
    // refine_certified_start: `newton_steps` further, must stay certified.
    // Mirror production's convergence early-exit: once the pending Newton step is
    // below the coordinate ULP scale the certified root is reached and further steps
    // only re-accumulate round-off (keeps device parity with the encode.rs fold).
    //
    // F5: return the certificate evaluated AT the refined landing coordinate
    // (`final_cert`), not the pre-refinement basin-exit cert. `final_cert` starts as
    // the basin-exit cert and is updated to each certified refine iterate's cert —
    // exactly production's `refine_certified_start` (the returned β/η/h describe the
    // coordinate actually returned). The old code returned the basin-exit `landing`,
    // whose Kantorovich root-radius overstates the refined point's distance to the
    // root — the source of the emulator↔production `h`-parity gap.
    let mut final_cert = cert;
    for _ in 0..dev.newton_steps {
        let dnorm = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
        let tnorm = t.iter().map(|v| v * v).sum::<f64>().sqrt();
        if dnorm <= crate::encode::NEWTON_REFINE_CONVERGED_EPS * (1.0 + tnorm) {
            break;
        }
        for i in 0..dev.d {
            t[i] += delta[i];
        }
        let (nc, nd) = row_certificate(dev, &t, x, amplitude, chart.lipschitz, scratch);
        if !nc.certified() {
            return None;
        }
        final_cert = nc;
        delta = nd;
    }
    Some((t, final_cert))
}

/// Distilled affine warm start `t̂ = t_c + (1/z)·A₁·(x − z·m₁)`. Mirror of
/// [`crate::encode::amortized_warm_start`]. `None` when the chart has no
/// Jacobian or the amplitude is not strictly positive & finite.
fn amortized_warm_start(chart: &EncodeChartDevice, x: &[f64], amplitude: f64, d: usize, p: usize) -> Option<Vec<f64>> {
    if !chart.has_jacobian {
        return None;
    }
    if !(amplitude.is_finite() && amplitude.abs() > 0.0) {
        return None;
    }
    let mut t_hat = chart.center.clone();
    for out in 0..p.min(chart.recon_center.len()) {
        let resid = x[out] - amplitude * chart.recon_center[out];
        for axis in 0..d {
            t_hat[axis] += chart.amortized_jacobian[axis * p + out] * resid / amplitude;
        }
    }
    Some(t_hat)
}

/// Reconstruction error `‖x − z·m(t)‖`. Mirror of
/// [`crate::encode::encode_reconstruction_error`].
fn recon_error(dev: &EncodeAtomDevice, t: &[f64], x: &[f64], amplitude: f64, scratch: &mut Scratch) -> f64 {
    eval_basis(dev, t, &mut scratch.phi, &mut scratch.jet, &mut scratch.hess);
    let mut err2 = 0.0;
    let p = dev.p;
    let mut recon = vec![0.0_f64; p];
    recon_amp1(dev, &scratch.phi, &mut recon);
    for c in 0..p {
        let r = x[c] - amplitude * recon[c];
        err2 += r * r;
    }
    if err2.is_finite() { err2.sqrt() } else { f64::INFINITY }
}

/// Top-`k` charts by center reconstruction distance, sorted by (distance, index)
/// — mirror of [`crate::encode::nearest_charts_topk`]. Only certifiable charts
/// (`certified_radius > 0`) are considered.
fn nearest_charts_topk(dev: &EncodeAtomDevice, x: &[f64], scratch: &mut Scratch) -> Vec<usize> {
    if dev.charts.is_empty() || dev.topk == 0 {
        return Vec::new();
    }
    let p = dev.p;
    let mut scored: Vec<(usize, f64)> = Vec::new();
    let mut recon = vec![0.0_f64; p];
    for (idx, chart) in dev.charts.iter().enumerate() {
        if chart.certified_radius <= 0.0 {
            continue;
        }
        eval_basis(dev, &chart.center, &mut scratch.phi, &mut scratch.jet, &mut scratch.hess);
        recon_amp1(dev, &scratch.phi, &mut recon);
        let mut dist = 0.0;
        for c in 0..p {
            let diff = recon[c] - x[c];
            dist += diff * diff;
        }
        scored.push((idx, dist));
    }
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored.into_iter().take(dev.topk).map(|(i, _)| i).collect()
}

/// The full exact per-row certified encode for one `EuclideanPatch` atom — the
/// device-free mirror of [`crate::encode::EncodeAtlas::certified_encode_row`].
/// This is BOTH the CPU fallback and the exactness oracle the CUDA kernel is
/// pinned to (the kernel does exactly this, one block per row).
#[must_use]
pub fn emulate_certified_encode_row(dev: &EncodeAtomDevice, x: &[f64], amplitude: f64) -> DeviceEncodeRow {
    let d = dev.d;
    let p = dev.p;
    let mut scratch = Scratch::new(dev);
    let candidates = nearest_charts_topk(dev, x, &mut scratch);
    if candidates.is_empty() {
        return DeviceEncodeRow {
            coord: vec![0.0; d],
            cert: DeviceRowCertificate::uncertified_inf(),
        };
    }
    let mut best: Option<(Vec<f64>, DeviceRowCertificate, f64)> = None;
    let mut nearest_fallback: Option<(Vec<f64>, DeviceRowCertificate)> = None;
    for chart_idx in candidates {
        let chart = &dev.charts[chart_idx];
        let Some(t_hat) = amortized_warm_start(chart, x, amplitude, d, p) else {
            if nearest_fallback.is_none() {
                nearest_fallback = Some((vec![0.0; d], DeviceRowCertificate::uncertified(chart.lipschitz)));
            }
            continue;
        };
        let (coord, cert) = match certify_with_basin_warmup(dev, t_hat, x, amplitude, chart, &mut scratch) {
            Some((c, cert)) => (c, cert),
            None => (vec![0.0; d], DeviceRowCertificate::uncertified(chart.lipschitz)),
        };
        if nearest_fallback.is_none() {
            nearest_fallback = Some((coord.clone(), cert));
        }
        if cert.certified() {
            let err = recon_error(dev, &coord, x, amplitude, &mut scratch);
            if best.as_ref().map(|(_, _, e)| err < *e).unwrap_or(true) {
                best = Some((coord, cert, err));
            }
            // Mirror production certified_encode_row's global-minimum short-circuit
            // (encode.rs): reconstruction error ≥ 0, so a certified candidate already
            // at the ambient noise floor is provably the global optimum — stop
            // refining the remaining top-K charts (keeps device parity with the fold).
            if let Some((_, _, e)) = best.as_ref() {
                let xnorm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                if *e <= crate::encode::CERTIFIED_GLOBAL_MIN_RECON_FLOOR * (1.0 + xnorm) {
                    break;
                }
            }
        }
    }
    match best {
        Some((coord, cert, _)) => DeviceEncodeRow { coord, cert },
        None => {
            let (coord, cert) = nearest_fallback
                .unwrap_or_else(|| (vec![0.0; d], DeviceRowCertificate::uncertified_inf()));
            DeviceEncodeRow { coord, cert }
        }
    }
}

/// Batched device-free encode over many rows (the CPU fallback of
/// [`sae_certified_encode_batch`]). Row-independent, so order-stable.
#[must_use]
pub fn emulate_certified_encode_batch(
    dev: &EncodeAtomDevice,
    targets: &[Vec<f64>],
    amplitudes: &[f64],
) -> Vec<DeviceEncodeRow> {
    targets
        .iter()
        .zip(amplitudes.iter())
        .map(|(x, &amp)| emulate_certified_encode_row(dev, x, amp))
        .collect()
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
    for(int i=0;i<DD;++i) t[i]+=delta[i];
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

  // ---- routing: top-TOPK certifiable charts by center recon distance. ----
  int cand[TOPK]; double cand_d[TOPK]; int ncand=0;
  {
    double phi[MM]; double jet[MM*DD]; double hess[MM*DD*DD]; double recon[PP];
    for(int idx=0; idx<n_charts; ++idx){
      if (cert_radii[idx] <= 0.0) continue;
      eval_basis(exps, centers + (size_t)idx*DD, phi, jet, hess);
      recon_amp1(dec, phi, recon);
      double dist=0.0; for(int c=0;c<PP;++c){ double df=recon[c]-x[c]; dist+=df*df; }
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

/// Batched certified encode over many rows, on the GPU when a CUDA device is
/// admitted and the batch amortises the launch, else on the CPU emulator. The
/// returned [`EncodePath`] reports which path ran honestly (`device_encode_engaged`).
/// Both paths run the SAME numeric core (Jacobi eigensolve, monomial jets), so
/// when the device runs its result matches the CPU oracle to eigen round-off.
#[must_use]
pub fn sae_certified_encode_batch(
    dev: &EncodeAtomDevice,
    targets: &[Vec<f64>],
    amplitudes: &[f64],
) -> (Vec<DeviceEncodeRow>, EncodePath) {
    #[cfg(target_os = "linux")]
    {
        if targets.len() >= DEVICE_ROW_THRESHOLD {
            if let Ok(out) = device::sae_certified_encode_device(dev, targets, amplitudes) {
                return (out, EncodePath::Device);
            }
            // Fall through to CPU on any device error (accelerator, not oracle).
        }
    }
    (
        emulate_certified_encode_batch(dev, targets, amplitudes),
        EncodePath::Cpu,
    )
}

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

impl DeviceEncodeThroughput {
    /// `true` iff the exact-encode kernel actually ran on a CUDA device — the
    /// only state in which [`Self::decision`] is a real `Met`/`Unmet`.
    #[must_use]
    pub fn device_engaged(&self) -> bool {
        matches!(self.path, EncodePath::Device)
    }
}

/// Benchmark the device-resident exact per-row encode over a batch and gate the
/// #988 certified-surrogate decision on the measured throughput.
///
/// Runs [`sae_certified_encode_batch`] once to warm allocations/compile/module
/// caches, then once more under a wall-clock timer, and reports the measured
/// rows/sec together with the honest [`EncodePath`] and the derived
/// [`EncodeDeploymentDecision`]:
///
/// * On a CUDA host with `targets.len() >= DEVICE_ROW_THRESHOLD` the batch runs
///   on the device (`path == Device`), the measurement is a genuine device rate,
///   and the decision is `Met` (≥ 100k rows/sec/GPU ⇒ ship the exact encode, no
///   surrogate) or `Unmet` (surrogate justified) by the number.
/// * On a CPU-only host (or below the launch threshold) the emulator runs
///   (`path == Cpu`); the rate is real but it is NOT a device measurement, so the
///   decision is `Undetermined` — the surrogate stays neither justified nor
///   refuted. This is the honest "needs GPU hardware" outcome.
#[must_use]
pub fn measure_device_encode_throughput(
    dev: &EncodeAtomDevice,
    targets: &[Vec<f64>],
    amplitudes: &[f64],
) -> DeviceEncodeThroughput {
    let n = targets.len();
    // Warm run (device module load / PTX cache / first-touch allocations) is not
    // timed, mirroring the resident-solve and full-path benchmarks.
    drop(sae_certified_encode_batch(dev, targets, amplitudes));
    let start = Instant::now();
    let (_out, path) = sae_certified_encode_batch(dev, targets, amplitudes);
    let elapsed = start.elapsed();
    let encode_secs = elapsed.as_secs_f64();
    let rows_per_sec = if n > 0 && encode_secs > 0.0 {
        n as f64 / encode_secs
    } else {
        0.0
    };
    let engaged = matches!(path, EncodePath::Device);
    // Key the surrogate decision on the measurement. A CPU-emulator run is an
    // honest non-measurement for the #988 device gate, but the reason matters:
    //
    //   * no CUDA runtime/device on the host => blocked on hardware;
    //   * CUDA present and the batch was large enough to require the device path,
    //     yet the kernel fell back to the emulator => false routing / device
    //     non-engagement, which must be surfaced as such rather than laundered
    //     into the same state as a CPU-only laptop.
    //
    // In either case the anti-green-wash contract is the same: an emulator rate
    // can never declare the surrogate unneeded or justified.
    let device_available = gam_gpu::device_runtime::GpuRuntime::global()
        .map(|rt| rt.device_count() > 0)
        .unwrap_or(false);
    let decision = if engaged {
        EncodeDeploymentDecision::from_device_measurement(true, rows_per_sec)
    } else if device_available && n >= DEVICE_ROW_THRESHOLD {
        EncodeDeploymentDecision::blocked(EncodeDecisionBlocked::DeviceNotEngaged)
    } else {
        EncodeDeploymentDecision::blocked(EncodeDecisionBlocked::NoDevice)
    };
    DeviceEncodeThroughput {
        n_rows: n,
        encode_secs,
        rows_per_sec,
        path,
        decision,
    }
}

#[cfg(target_os = "linux")]
mod device {
    use super::{
        DeviceEncodeRow, DeviceRowCertificate, EncodeAtomDevice, encode_kernel_source,
    };
    use gam_gpu::gpu_error::{GpuError, GpuResultExt};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};

    struct Backend {
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        modules: Mutex<HashMap<String, Arc<CudaModule>>>,
    }

    fn backend() -> Result<&'static Backend, GpuError> {
        static BACKEND: OnceLock<Result<Backend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                let parts = gam_gpu::backend_probe::probe_cuda_backend("sae_encode")?;
                Ok(Backend {
                    ctx: parts.ctx,
                    stream: parts.stream,
                    modules: Mutex::new(HashMap::new()),
                })
            })
            .as_ref()
            .map_err(GpuError::clone)
    }

    fn module_for(b: &Backend, dev: &EncodeAtomDevice) -> Result<Arc<CudaModule>, GpuError> {
        let key = format!(
            "{}-{}-{}-{}-{}",
            dev.d, dev.m, dev.p, dev.topk, dev.newton_steps
        );
        if let Ok(guard) = b.modules.lock() {
            if let Some(m) = guard.get(&key) {
                return Ok(m.clone());
            }
        }
        let src = encode_kernel_source(dev);
        let ptx = gam_gpu::device_cache::compile_ptx_arch(&src)
            .gpu_ctx_with(|err| format!("sae_encode NVRTC compile ({key}): {err}"))?;
        let module = b.ctx.load_module(ptx).gpu_ctx("sae_encode module load")?;
        if let Ok(mut guard) = b.modules.lock() {
            guard.entry(key).or_insert_with(|| module.clone());
        }
        Ok(module)
    }

    /// Device implementation: flatten the atom + charts + rows, launch one block
    /// per row, download coords/certificate.
    pub(super) fn sae_certified_encode_device(
        dev: &EncodeAtomDevice,
        targets: &[Vec<f64>],
        amplitudes: &[f64],
    ) -> Result<Vec<DeviceEncodeRow>, GpuError> {
        let n = targets.len();
        let (d, p) = (dev.d, dev.p);
        if n == 0 {
            return Ok(Vec::new());
        }
        let b = backend()?;
        let module = module_for(b, dev)?;
        let func = module
            .load_function("sae_certified_encode")
            .gpu_ctx("sae_encode load_function")?;
        let stream = b.stream.clone();
        let n_charts = dev.charts.len();

        // Flatten charts.
        let mut centers = vec![0.0_f64; n_charts * d];
        let mut radii = vec![0.0_f64; n_charts];
        let mut cert_radii = vec![0.0_f64; n_charts];
        let mut lips = vec![0.0_f64; n_charts];
        let mut has_jac = vec![0_i32; n_charts];
        let mut a1 = vec![0.0_f64; n_charts * d * p];
        let mut recon_c = vec![0.0_f64; n_charts * p];
        for (i, ch) in dev.charts.iter().enumerate() {
            centers[i * d..(i + 1) * d].copy_from_slice(&ch.center);
            radii[i] = ch.radius;
            cert_radii[i] = ch.certified_radius;
            lips[i] = ch.lipschitz;
            has_jac[i] = i32::from(ch.has_jacobian);
            if ch.has_jacobian {
                a1[i * d * p..(i + 1) * d * p].copy_from_slice(&ch.amortized_jacobian);
            }
            recon_c[i * p..(i + 1) * p].copy_from_slice(&ch.recon_center);
        }
        let mut tgt = vec![0.0_f64; n * p];
        for (i, x) in targets.iter().enumerate() {
            tgt[i * p..(i + 1) * p].copy_from_slice(x);
        }

        let exps_dev = stream.clone_htod(&dev.exponents).gpu_ctx("sae_encode htod exps")?;
        let dec_dev = stream.clone_htod(&dev.decoder).gpu_ctx("sae_encode htod dec")?;
        let centers_dev = stream.clone_htod(&centers).gpu_ctx("sae_encode htod centers")?;
        let radii_dev = stream.clone_htod(&radii).gpu_ctx("sae_encode htod radii")?;
        let cert_dev = stream.clone_htod(&cert_radii).gpu_ctx("sae_encode htod cert_radii")?;
        let lips_dev = stream.clone_htod(&lips).gpu_ctx("sae_encode htod lips")?;
        let hasj_dev = stream.clone_htod(&has_jac).gpu_ctx("sae_encode htod has_jac")?;
        let a1_dev = stream.clone_htod(&a1).gpu_ctx("sae_encode htod a1")?;
        let reconc_dev = stream.clone_htod(&recon_c).gpu_ctx("sae_encode htod recon_c")?;
        let tgt_dev = stream.clone_htod(&tgt).gpu_ctx("sae_encode htod targets")?;
        let amps_dev = stream.clone_htod(&amplitudes.to_vec()).gpu_ctx("sae_encode htod amps")?;
        let mut coords_dev = stream.alloc_zeros::<f64>(n * d).gpu_ctx("sae_encode alloc coords")?;
        let mut h_dev = stream.alloc_zeros::<f64>(n).gpu_ctx("sae_encode alloc h")?;
        let mut cert_out_dev = stream.alloc_zeros::<i32>(n).gpu_ctx("sae_encode alloc certified")?;

        let n_i32 = i32::try_from(n).map_err(|_| gam_gpu::gpu_err!("sae_encode n overflow"))?;
        let ncharts_i32 =
            i32::try_from(n_charts).map_err(|_| gam_gpu::gpu_err!("sae_encode n_charts overflow"))?;
        let cfg = LaunchConfig {
            grid_dim: (n_i32 as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&exps_dev)
            .arg(&dec_dev)
            .arg(&centers_dev)
            .arg(&radii_dev)
            .arg(&cert_dev)
            .arg(&lips_dev)
            .arg(&hasj_dev)
            .arg(&a1_dev)
            .arg(&reconc_dev)
            .arg(&ncharts_i32)
            .arg(&tgt_dev)
            .arg(&amps_dev)
            .arg(&n_i32)
            .arg(&mut coords_dev)
            .arg(&mut h_dev)
            .arg(&mut cert_out_dev);
        // SAFETY: grid/block validated; all pointers are cudarc-checked allocations
        // on this stream; the kernel reads within the flattened inputs and writes
        // coords[0..n*d], h[0..n], certified[0..n].
        unsafe { builder.launch(cfg) }.gpu_ctx("sae_encode kernel launch")?;

        let mut coords = vec![0.0_f64; n * d];
        let mut h = vec![0.0_f64; n];
        let mut cert = vec![0_i32; n];
        stream.memcpy_dtoh(&coords_dev, &mut coords).gpu_ctx("sae_encode dtoh coords")?;
        stream.memcpy_dtoh(&h_dev, &mut h).gpu_ctx("sae_encode dtoh h")?;
        stream.memcpy_dtoh(&cert_out_dev, &mut cert).gpu_ctx("sae_encode dtoh certified")?;
        stream.synchronize().gpu_ctx("sae_encode synchronize")?;

        let mut out = Vec::with_capacity(n);
        for row in 0..n {
            let coord = coords[row * d..(row + 1) * d].to_vec();
            let hv = h[row];
            out.push(DeviceEncodeRow {
                coord,
                cert: DeviceRowCertificate {
                    // beta/eta not transported; the h + certified flag is the contract.
                    beta: f64::NAN,
                    eta: f64::NAN,
                    lipschitz: f64::NAN,
                    h: hv,
                },
            });
        }
        // Reconcile the certified flag from the device (authoritative) — the h
        // value alone can be +inf for an uncertified fallback.
        for (row, o) in out.iter_mut().enumerate() {
            if cert[row] == 0 {
                o.cert.h = f64::INFINITY;
            }
        }
        Ok(out)
    }
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
        let atom = SaeManifoldAtom::new(
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
            .map(|k| (0..p).map(|c| 0.3 * (((k + c) as f64) * 0.19).sin()).collect())
            .collect();
        let amps: Vec<f64> = (0..n).map(|_| 1.0).collect();
        let (batch, path) = sae_certified_encode_batch(&dev, &rows, &amps);
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
        let tput = measure_device_encode_throughput(&dev, &rows, &amps);
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
        assert_eq!(tput.device_engaged(), matches!(tput.path, EncodePath::Device));

        // The benchmark must be non-vacuous: on a well-conditioned dictionary the
        // planted on-manifold rows certify through the exact encode (proving the
        // routing + basin Newton + certificate really ran, not a trivial pass).
        let (batch, _) = sae_certified_encode_batch(&dev, &rows, &amps);
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
            if gam_gpu::device_runtime::GpuRuntime::global()
                .map(|rt| rt.device_count() > 0)
                .unwrap_or(false)
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
                assert!((acc - a[r * 2 + c]).abs() < 1e-12, "eig reconstruct {r},{c}");
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
        assert!(text.contains(".visible .entry sae_certified_encode"),
            "PTX must export the encode entry");
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
            .map(|k| (0..p).map(|c| 0.3 * (((k + c) as f64) * 0.019).sin()).collect())
            .collect();
        let amps = vec![1.0; n];
        let cpu = emulate_certified_encode_batch(&dev, &rows, &amps);
        if gam_gpu::device_runtime::GpuRuntime::global().is_some() {
            let devout = device::sae_certified_encode_device(&dev, &rows, &amps)
                .expect("admitted GPU runtime must run the sae_encode kernel");
            let mut max_coord = 0.0_f64;
            for (a, b) in cpu.iter().zip(devout.iter()) {
                assert_eq!(a.cert.certified(), b.cert.certified(), "device certified flag");
                if a.cert.certified() {
                    for axis in 0..dev.d {
                        max_coord = max_coord.max((a.coord[axis] - b.coord[axis]).abs());
                    }
                }
            }
            assert!(max_coord <= 1e-9, "device vs emulator coord diff {max_coord:.3e} > 1e-9");
        }
    }
}
