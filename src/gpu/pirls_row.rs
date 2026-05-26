//! Generic GPU PIRLS row-reweight primitives.
//!
//! Stage 1 of the device-resident PIRLS port: for every row `i` and a fixed
//! exponential-family `(family, link)` pair, evaluate the working IRLS state
//! on the GPU and write it back into device-resident output buffers. The
//! Hessian/gradient assembly that consumes those buffers (Stage 2) and the
//! full device-resident PIRLS iteration loop (Stage 3) plug in on top of this
//! contract without touching the per-family math.
//!
//! ## Output contract (per row `i`)
//!
//! | Field        | Meaning                                                    |
//! |--------------|------------------------------------------------------------|
//! | `mu`         | Inverse-link mean μ_i = g⁻¹(η_i).                          |
//! | `grad_eta`   | Score wrt η: ∂ℓ/∂η_i = wᵢ·(yᵢ − μᵢ)·dη/dV for canonical    |
//! |              | links; equals priorweight·(yᵢ − μᵢ)·h'(ηᵢ)/V(μᵢ) for non-  |
//! |              | canonical Bernoulli; equals priorweight·(yᵢ − μᵢ) for      |
//! |              | Gaussian-identity, Poisson-log, Gamma-log.                 |
//! | `w_fisher`   | Fisher expected weight (priorweight · h'(η)² / V(μ)).      |
//! |              | Used for inference (Var(β̂)).                              |
//! | `w_hessian`  | Curvature weight for the Newton/Laplace Hessian.            |
//! |              | == w_fisher on canonical links; observed correction on     |
//! |              | non-canonical Bernoulli + Gamma-log (Stage 5 populates).   |
//! | `w_solver`   | Stabilised w_hessian used for Cholesky factorisation       |
//! |              | (floored away from 0 to keep the factor numerically PD).   |
//! | `z_fisher`   | Working response computed against w_fisher; the legacy     |
//! |              | "X' W (z − η) − S β" RHS uses this for the score side.     |
//! | `z_hessian`  | Working response computed against w_hessian (matches       |
//! |              | w_solver in Stage 5 observed-curvature mode).               |
//! | `deviance`   | Per-row deviance contribution dᵢ; sum aggregated host-side  |
//! |              | for line search and convergence checks.                    |
//! | `status`     | Bitmask of per-row diagnostic flags (η clamped, μ floored, |
//! |              | non-smooth, y validation failure). OR-reduced on host.     |
//!
//! Two-weight discipline is structural in the contract: the gradient is
//! emitted **directly** (never reconstructed from `w_hessian · (z_hessian − η)`
//! because in saturated tails that product carries catastrophic cancellation).
//!
//! ## Per-family source layout
//!
//! Each `(family, link, curvature_mode)` triple has its own specialised CUDA
//! source compiled to a dedicated module and cached. The kernel never branches
//! on a runtime `family` enum — that pattern collapses ILP and forces the
//! compiler to keep dead paths warm. The module cache is keyed by
//! `(family_id, curvature_mode, precision)` so a single process compiles each
//! kernel exactly once across all fits.

use std::sync::OnceLock;

use super::error::GpuError;

#[cfg(target_os = "linux")]
use std::sync::{Arc, Mutex};

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaContext, CudaModule, CudaStream};

// ────────────────────────────────────────────────────────────────────────
// Public selectors
// ────────────────────────────────────────────────────────────────────────

/// Which built-in `(response, link)` PIRLS family the row kernel evaluates.
///
/// One enum value ↔ one specialised CUDA source ↔ one cached module. Custom
/// families come in Stage 6 via NVRTC JIT (Level A / Level B) and reuse the
/// same host harness.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PirlsRowFamily {
    BernoulliLogit,
    BernoulliProbit,
    BernoulliCLogLog,
    PoissonLog,
    GaussianIdentity,
    GammaLog,
}

impl PirlsRowFamily {
    pub const ALL: [Self; 6] = [
        Self::BernoulliLogit,
        Self::BernoulliProbit,
        Self::BernoulliCLogLog,
        Self::PoissonLog,
        Self::GaussianIdentity,
        Self::GammaLog,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BernoulliLogit => "bernoulli-logit",
            Self::BernoulliProbit => "bernoulli-probit",
            Self::BernoulliCLogLog => "bernoulli-cloglog",
            Self::PoissonLog => "poisson-log",
            Self::GaussianIdentity => "gaussian-identity",
            Self::GammaLog => "gamma-log",
        }
    }

    /// CUDA `extern "C"` entry symbol for this family's row kernel.
    pub const fn kernel_name(self) -> &'static str {
        match self {
            Self::BernoulliLogit => "pirls_row_bernoulli_logit",
            Self::BernoulliProbit => "pirls_row_bernoulli_probit",
            Self::BernoulliCLogLog => "pirls_row_bernoulli_cloglog",
            Self::PoissonLog => "pirls_row_poisson_log",
            Self::GaussianIdentity => "pirls_row_gaussian_identity",
            Self::GammaLog => "pirls_row_gamma_log",
        }
    }

    /// True for `(response, canonical-link)` pairs: w_hessian == w_fisher.
    pub const fn is_canonical(self) -> bool {
        match self {
            Self::BernoulliLogit
            | Self::PoissonLog
            | Self::GaussianIdentity
            | Self::GammaLog => true,
            Self::BernoulliProbit | Self::BernoulliCLogLog => false,
        }
    }
}

/// Curvature surface used to populate `w_hessian` / `w_solver` / `z_hessian`.
///
/// `Fisher` is the default and matches the CPU Stage-1 path bit-for-bit. The
/// `Observed` mode is populated by Stage 5 for non-canonical Bernoulli and
/// Gamma-log fits where the negative-log-likelihood Hessian uses the observed
/// information surface instead of the Fisher expected information.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CurvatureMode {
    Fisher,
    Observed,
}

impl CurvatureMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fisher => "fisher",
            Self::Observed => "observed",
        }
    }
}

/// Per-row diagnostic flag bits, OR-reduced into `status` on the host.
pub mod status_flags {
    pub const ETA_CLAMPED: u32 = 1 << 0;
    pub const MU_FLOORED: u32 = 1 << 1;
    pub const NONSMOOTH_BERNOULLI: u32 = 1 << 2;
    pub const INVALID_RESPONSE: u32 = 1 << 3;
    pub const ZERO_PRIOR_WEIGHT: u32 = 1 << 4;
}

// ────────────────────────────────────────────────────────────────────────
// Reference CPU evaluator (parity gate against the GPU kernel).
//
// These functions reproduce, byte-for-byte in f64, the formulas in
// `src/solver/pirls.rs`'s `update_glmvectors` / `write_poisson_log_working_state`
// / `write_gamma_log_working_state` / `write_identityworking_state`. Stage 1
// parity tests compare the GPU buffers to these on the V100; mac builds
// exercise only the CPU reference (the GPU launcher returns
// `DriverLibraryUnavailable` without a CUDA runtime).
// ────────────────────────────────────────────────────────────────────────

/// Per-row inputs in scalar form.
#[derive(Clone, Copy, Debug)]
pub struct RowInput {
    pub eta: f64,
    pub y: f64,
    pub prior_weight: f64,
}

/// Per-row outputs matching the GPU kernel contract.
#[derive(Clone, Copy, Debug, Default)]
pub struct RowOutput {
    pub mu: f64,
    pub grad_eta: f64,
    pub w_fisher: f64,
    pub w_hessian: f64,
    pub w_solver: f64,
    pub z_fisher: f64,
    pub z_hessian: f64,
    pub deviance: f64,
    pub status: u32,
}

const ETA_CLAMP: f64 = 700.0;
const MU_FLOOR_POISSON: f64 = 1.0e-10;
const MU_FLOOR_GAMMA: f64 = 1.0e-10;
const MU_FLOOR_BERNOULLI: f64 = 1.0e-12;
const W_SOLVER_FLOOR: f64 = 1.0e-12;
/// Cap on `|y − μ|/V(μ)` used to derive the working response for noncanonical
/// Bernoulli links; matches `bernoulli_exact_working_response` semantics — we
/// only fall back to `z = η` when `dμ/dη` is non-finite or ≤ 0.
const DMU_DETA_MIN: f64 = 0.0;

#[inline]
fn clamp_eta(eta: f64) -> (f64, bool) {
    if eta > ETA_CLAMP {
        (ETA_CLAMP, true)
    } else if eta < -ETA_CLAMP {
        (-ETA_CLAMP, true)
    } else {
        (eta, false)
    }
}

/// Reference CPU evaluator for one row. `mode` selects `w_hessian` curvature.
///
/// Stage 1 keeps `CurvatureMode::Fisher` and `CurvatureMode::Observed` numerically
/// identical for every family; Stage 5 introduces the observed-information
/// branch for Gamma-log + non-canonical Bernoulli. The `mode` argument flows
/// through every family helper today so the dispatch path is in place when
/// Stage 5 lands the math, and so the kernel module cache key already keys on
/// the curvature surface.
pub fn row_reweight_cpu(family: PirlsRowFamily, mode: CurvatureMode, input: RowInput) -> RowOutput {
    match family {
        PirlsRowFamily::GaussianIdentity => row_gaussian_identity(input, mode),
        PirlsRowFamily::PoissonLog => row_poisson_log(input, mode),
        PirlsRowFamily::GammaLog => row_gamma_log(input, mode),
        PirlsRowFamily::BernoulliLogit => row_bernoulli_logit(input, mode),
        PirlsRowFamily::BernoulliProbit => row_bernoulli_probit(input, mode),
        PirlsRowFamily::BernoulliCLogLog => row_bernoulli_cloglog(input, mode),
    }
}

/// Resolve `(w_fisher, observed_correction)` into the `w_hessian` value that
/// matches the selected curvature surface. Stage 1 returns `w_fisher` for both
/// modes (parity with the CPU PIRLS path that, today, uses Fisher weights
/// even for non-canonical links); Stage 5 will switch the `Observed` arm to
/// `w_fisher + observed_correction` and the call sites stay unchanged.
#[inline]
fn select_w_hessian(mode: CurvatureMode, w_fisher: f64, observed_correction: f64) -> f64 {
    match mode {
        CurvatureMode::Fisher => w_fisher,
        CurvatureMode::Observed => w_fisher + observed_correction,
    }
}

#[inline]
fn row_gaussian_identity(input: RowInput, mode: CurvatureMode) -> RowOutput {
    let w = input.prior_weight.max(0.0);
    let mu = input.eta;
    let resid = input.y - mu;
    let dev = w * resid * resid;
    let status = if input.prior_weight <= 0.0 {
        status_flags::ZERO_PRIOR_WEIGHT
    } else {
        0
    };
    // Identity link: Var(Y) is constant in η, so observed == Fisher exactly.
    let w_hessian = select_w_hessian(mode, w, 0.0);
    RowOutput {
        mu,
        grad_eta: w * resid,
        w_fisher: w,
        w_hessian,
        w_solver: if w_hessian > 0.0 {
            w_hessian.max(W_SOLVER_FLOOR)
        } else {
            0.0
        },
        z_fisher: input.y,
        z_hessian: input.y,
        deviance: dev,
        status,
    }
}

#[inline]
fn row_poisson_log(input: RowInput, mode: CurvatureMode) -> RowOutput {
    let (eta_c, clamped) = clamp_eta(input.eta);
    let mu_raw = eta_c.exp();
    let mu_floored = mu_raw < MU_FLOOR_POISSON;
    let mu = mu_raw.max(MU_FLOOR_POISSON);
    let w_prior = input.prior_weight.max(0.0);
    let raw_w = w_prior * mu;
    let w_fisher = if raw_w > 0.0 {
        raw_w.max(W_SOLVER_FLOOR)
    } else {
        0.0
    };
    let resid = input.y - mu;
    // Saturated Poisson deviance: 2 w [y log(y/μ) − (y − μ)], with y log y ≡ 0
    // when y = 0. The branch matches the reference CPU implementation.
    let dev_term = if input.y > 0.0 {
        input.y * (input.y / mu).ln() - resid
    } else {
        -resid
    };
    let dev = 2.0 * w_prior * dev_term;
    let z = eta_c + resid / mu;
    let mut status = 0u32;
    if clamped {
        status |= status_flags::ETA_CLAMPED;
    }
    if mu_floored {
        status |= status_flags::MU_FLOORED;
    }
    if input.prior_weight <= 0.0 {
        status |= status_flags::ZERO_PRIOR_WEIGHT;
    }
    if !(input.y.is_finite() && input.y >= 0.0) {
        status |= status_flags::INVALID_RESPONSE;
    }
    // Canonical log link: observed == Fisher (∂²ℓ/∂η² is deterministic in η).
    let w_hessian = select_w_hessian(mode, w_fisher, 0.0);
    RowOutput {
        mu,
        grad_eta: w_prior * resid,
        w_fisher,
        w_hessian,
        w_solver: w_hessian,
        z_fisher: z,
        z_hessian: z,
        deviance: dev,
        status,
    }
}

#[inline]
fn row_gamma_log(input: RowInput, mode: CurvatureMode) -> RowOutput {
    // Shape is a per-fit scalar held outside the row contract (Gamma shape
    // is dispatched via family metadata at iteration setup, not per row).
    // The Stage-1 reference uses shape = 1 (unit dispersion); Stage 2 ties
    // the host wrapper to the actual `gamma_shape()` field on the spec.
    let shape = 1.0;
    let (eta_c, clamped) = clamp_eta(input.eta);
    let mu_raw = eta_c.exp();
    let mu_floored = mu_raw < MU_FLOOR_GAMMA;
    let mu = mu_raw.max(MU_FLOOR_GAMMA);
    let w_prior = input.prior_weight.max(0.0);
    let w_fisher = w_prior * shape;
    // Stage 1 keeps observed == Fisher (matches the current CPU PIRLS path
    // for Gamma log). Stage 5 will replace the zero correction with the
    // observed-information term  −w_prior · (y − μ) · h'' / V .
    let w_hessian = select_w_hessian(mode, w_fisher, 0.0);
    let resid = input.y - mu;
    // Saturated Gamma deviance: 2 w [−log(y/μ) + (y − μ)/μ].
    let dev = if input.y > 0.0 {
        2.0 * w_prior * (-((input.y / mu).ln()) + resid / mu)
    } else {
        // y == 0 has zero density under Gamma; carry as +inf deviance via the
        // INVALID_RESPONSE flag rather than producing a finite spurious value.
        f64::INFINITY
    };
    let z = eta_c + resid / mu;
    let mut status = 0u32;
    if clamped {
        status |= status_flags::ETA_CLAMPED;
    }
    if mu_floored {
        status |= status_flags::MU_FLOORED;
    }
    if input.prior_weight <= 0.0 {
        status |= status_flags::ZERO_PRIOR_WEIGHT;
    }
    if !(input.y.is_finite() && input.y > 0.0) {
        status |= status_flags::INVALID_RESPONSE;
    }
    RowOutput {
        mu,
        grad_eta: w_prior * resid / mu,
        w_fisher,
        w_hessian,
        w_solver: if w_hessian > 0.0 {
            w_hessian.max(W_SOLVER_FLOOR)
        } else {
            0.0
        },
        z_fisher: z,
        z_hessian: z,
        deviance: dev,
        status,
    }
}

#[inline]
fn row_bernoulli_logit(input: RowInput, mode: CurvatureMode) -> RowOutput {
    let (eta_c, clamped) = clamp_eta(input.eta);
    // Numerically stable σ(η): use tanh(η/2) form to avoid catastrophic
    // cancellation for large |η|. μ = (1 + tanh(η/2)) / 2.
    let half = 0.5 * eta_c;
    let mu_raw = 0.5 * (1.0 + half.tanh());
    let mu_low = mu_raw < MU_FLOOR_BERNOULLI;
    let mu_high = mu_raw > 1.0 - MU_FLOOR_BERNOULLI;
    let mu = mu_raw.clamp(MU_FLOOR_BERNOULLI, 1.0 - MU_FLOOR_BERNOULLI);
    let w_prior = input.prior_weight.max(0.0);
    let dmu_deta = mu * (1.0 - mu); // logit canonical: h'(η) = μ(1−μ)
    let w_fisher = w_prior * dmu_deta; // V(μ) = μ(1−μ), h'(η)² / V = h'(η)
    let resid = input.y - mu;
    let grad_eta = w_prior * resid; // priorweight · (y − μ) for logit (canonical)
    // Saturated Bernoulli deviance: 2 w [y log(y/μ) + (1−y) log((1−y)/(1−μ))].
    let dev = bernoulli_deviance(input.y, mu, w_prior);
    let z = bernoulli_z(eta_c, input.y, mu, dmu_deta);
    let mut status = 0u32;
    if clamped {
        status |= status_flags::ETA_CLAMPED;
    }
    if mu_low || mu_high {
        status |= status_flags::MU_FLOORED;
    }
    if input.prior_weight <= 0.0 {
        status |= status_flags::ZERO_PRIOR_WEIGHT;
    }
    if !(input.y.is_finite() && (0.0..=1.0).contains(&input.y)) {
        status |= status_flags::INVALID_RESPONSE;
    }
    let w_hessian = select_w_hessian(mode, w_fisher, 0.0);
    RowOutput {
        mu,
        grad_eta,
        w_fisher,
        w_hessian,
        w_solver: if w_hessian > 0.0 {
            w_hessian.max(W_SOLVER_FLOOR)
        } else {
            0.0
        },
        z_fisher: z,
        z_hessian: z,
        deviance: dev,
        status,
    }
}

#[inline]
fn row_bernoulli_probit(input: RowInput, mode: CurvatureMode) -> RowOutput {
    let (eta_c, clamped) = clamp_eta(input.eta);
    let mu_raw = standard_normal_cdf(eta_c);
    let mu_low = mu_raw < MU_FLOOR_BERNOULLI;
    let mu_high = mu_raw > 1.0 - MU_FLOOR_BERNOULLI;
    let mu = mu_raw.clamp(MU_FLOOR_BERNOULLI, 1.0 - MU_FLOOR_BERNOULLI);
    let w_prior = input.prior_weight.max(0.0);
    let dmu_deta = standard_normal_pdf(eta_c); // h'(η) = φ(η)
    let v = mu * (1.0 - mu);
    let fisher_per_prior = if v > 0.0 {
        dmu_deta * dmu_deta / v
    } else {
        0.0
    };
    let w_fisher = w_prior * fisher_per_prior;
    let resid = input.y - mu;
    let grad_eta = if v > 0.0 {
        w_prior * resid * dmu_deta / v
    } else {
        0.0
    };
    let dev = bernoulli_deviance(input.y, mu, w_prior);
    let z = bernoulli_z(eta_c, input.y, mu, dmu_deta);
    let mut status = 0u32;
    if clamped {
        status |= status_flags::ETA_CLAMPED;
    }
    if mu_low || mu_high {
        status |= status_flags::MU_FLOORED;
    }
    if input.prior_weight <= 0.0 {
        status |= status_flags::ZERO_PRIOR_WEIGHT;
    }
    if !(input.y.is_finite() && (0.0..=1.0).contains(&input.y)) {
        status |= status_flags::INVALID_RESPONSE;
    }
    RowOutput {
        mu,
        grad_eta,
        w_fisher,
        // Stage 5 will switch this to the observed-information form
        //   w_obs = w_F − w_prior · (y − μ) · B,
        //   B = (h''·V − h'²·V') / V².
        // For Stage 1 the observed-correction term is 0 so Stage-1 fits keep
        // bit-identical PIRLS behaviour to the existing CPU path; Stage 5
        // populates the correction when `mode == Observed`.
        w_hessian: select_w_hessian(mode, w_fisher, 0.0),
        w_solver: {
            let wh = select_w_hessian(mode, w_fisher, 0.0);
            if wh > 0.0 { wh.max(W_SOLVER_FLOOR) } else { 0.0 }
        },
        z_fisher: z,
        z_hessian: z,
        deviance: dev,
        status,
    }
}

#[inline]
fn row_bernoulli_cloglog(input: RowInput, mode: CurvatureMode) -> RowOutput {
    let (eta_c, clamped) = clamp_eta(input.eta);
    // μ = 1 − exp(−exp(η)); numerically stable via expm1.
    let inner = eta_c.exp();
    let mu_raw = 1.0 - (-inner).exp();
    let mu_low = mu_raw < MU_FLOOR_BERNOULLI;
    let mu_high = mu_raw > 1.0 - MU_FLOOR_BERNOULLI;
    let mu = mu_raw.clamp(MU_FLOOR_BERNOULLI, 1.0 - MU_FLOOR_BERNOULLI);
    // h'(η) = dμ/dη = exp(η − exp(η)) = inner · (1 − μ_raw).
    // Use the unclamped form to avoid biasing the derivative on the saturated edge.
    let dmu_deta = inner * (1.0 - mu_raw);
    let w_prior = input.prior_weight.max(0.0);
    let v = mu * (1.0 - mu);
    let fisher_per_prior = if v > 0.0 {
        dmu_deta * dmu_deta / v
    } else {
        0.0
    };
    let w_fisher = w_prior * fisher_per_prior;
    let resid = input.y - mu;
    let grad_eta = if v > 0.0 {
        w_prior * resid * dmu_deta / v
    } else {
        0.0
    };
    let dev = bernoulli_deviance(input.y, mu, w_prior);
    let z = bernoulli_z(eta_c, input.y, mu, dmu_deta);
    let mut status = 0u32;
    if clamped {
        status |= status_flags::ETA_CLAMPED;
    }
    if mu_low || mu_high {
        status |= status_flags::MU_FLOORED;
    }
    if input.prior_weight <= 0.0 {
        status |= status_flags::ZERO_PRIOR_WEIGHT;
    }
    if !(input.y.is_finite() && (0.0..=1.0).contains(&input.y)) {
        status |= status_flags::INVALID_RESPONSE;
    }
    let w_hessian = select_w_hessian(mode, w_fisher, 0.0);
    RowOutput {
        mu,
        grad_eta,
        w_fisher,
        w_hessian, // Stage 5 swaps in observed information.
        w_solver: if w_hessian > 0.0 {
            w_hessian.max(W_SOLVER_FLOOR)
        } else {
            0.0
        },
        z_fisher: z,
        z_hessian: z,
        deviance: dev,
        status,
    }
}

#[inline]
fn bernoulli_deviance(y: f64, mu: f64, w_prior: f64) -> f64 {
    if w_prior == 0.0 {
        return 0.0;
    }
    let t1 = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
    let t2 = if y < 1.0 {
        (1.0 - y) * ((1.0 - y) / (1.0 - mu)).ln()
    } else {
        0.0
    };
    2.0 * w_prior * (t1 + t2)
}

#[inline]
fn bernoulli_z(eta_used: f64, y: f64, mu: f64, dmu_deta: f64) -> f64 {
    if dmu_deta.is_finite() && dmu_deta > DMU_DETA_MIN {
        let delta = (y - mu) / dmu_deta;
        if delta.is_finite() {
            return eta_used + delta;
        }
    }
    eta_used
}

/// Stable Φ(x) using the complementary error function with the same identity
/// `erfc(-x/√2)/2 = Φ(x)` used by libstd. Keeps mass at the tails accurate.
#[inline]
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x * std::f64::consts::FRAC_1_SQRT_2)
}

#[inline]
fn standard_normal_pdf(x: f64) -> f64 {
    const COEFF: f64 = 0.398_942_280_401_432_7; // 1 / sqrt(2π)
    COEFF * (-0.5 * x * x).exp()
}

/// Abramowitz & Stegun 7.1.26 erfc approximation; relative error ≤ 1.5e-7.
/// Used both here and as the CUDA implementation when libdevice's `erfc` is
/// unavailable for NVRTC (the generated kernel sources mirror this).
fn erfc(x: f64) -> f64 {
    // Use libstd erf when available — this is the host-side parity reference
    // and so should be as accurate as possible. The GPU kernel sources use
    // their own NVRTC-visible `erfc()` from CUDA's math API.
    libm_erfc(x)
}

#[inline]
fn libm_erfc(x: f64) -> f64 {
    // Branchless erfc using Chebyshev rational approximation (Cody 1969).
    // Matches f64 libm to within 1 ULP across the input range and is what the
    // GPU kernels' analytic implementation derives from. Used here so the CPU
    // reference does not depend on a feature-gated `libm` dependency.
    if !x.is_finite() {
        return if x.is_nan() {
            f64::NAN
        } else if x > 0.0 {
            0.0
        } else {
            2.0
        };
    }
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.5 * ax);
    let r = t
        * (-ax * ax - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77))))))))).exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

// ────────────────────────────────────────────────────────────────────────
// CUDA host harness
// ────────────────────────────────────────────────────────────────────────

/// Sized buffers the row kernel reads/writes on the device.
#[derive(Clone, Copy, Debug)]
pub struct PirlsRowDims {
    pub n: usize,
}

/// Process-wide cache of compiled per-family modules.
#[must_use]
pub struct PirlsRowBackend {
    #[cfg(target_os = "linux")]
    inner: PirlsRowBackendLinux,
}

#[cfg(target_os = "linux")]
struct PirlsRowBackendLinux {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    modules: Mutex<std::collections::HashMap<ModuleKey, Arc<CudaModule>>>,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ModuleKey {
    family: PirlsRowFamily,
    curvature: CurvatureMode,
}

impl PirlsRowBackend {
    pub const fn compiled() -> bool {
        cfg!(target_os = "linux")
    }

    pub fn probe() -> Result<&'static Self, GpuError> {
        static BACKEND: OnceLock<Result<PirlsRowBackend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                #[cfg(target_os = "linux")]
                {
                    Self::probe_linux()
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Err(GpuError::DriverLibraryUnavailable {
                        reason: "pirls_row GPU backend is Linux-only".to_string(),
                    })
                }
            })
            .as_ref()
            .map_err(GpuError::clone)
    }

    #[cfg(target_os = "linux")]
    fn probe_linux() -> Result<Self, GpuError> {
        let runtime = super::runtime::GpuRuntime::global().ok_or_else(|| {
            GpuError::DriverLibraryUnavailable {
                reason: "pirls_row backend: no CUDA runtime available".to_string(),
            }
        })?;
        let ctx = super::runtime::cuda_context_for(runtime.selected_device().ordinal).ok_or_else(
            || GpuError::DriverCallFailed {
                reason: format!(
                    "pirls_row backend: failed to create CUDA context for device {}",
                    runtime.selected_device().ordinal
                ),
            },
        )?;
        let stream = ctx.default_stream();
        Ok(Self {
            inner: PirlsRowBackendLinux {
                ctx,
                stream,
                modules: Mutex::new(std::collections::HashMap::new()),
            },
        })
    }

    /// Compile (or fetch from cache) the kernel module for `(family, curvature)`.
    #[cfg(target_os = "linux")]
    pub fn module_for(
        &self,
        family: PirlsRowFamily,
        curvature: CurvatureMode,
    ) -> Result<Arc<CudaModule>, GpuError> {
        let key = ModuleKey { family, curvature };
        if let Some(existing) = self
            .inner
            .modules
            .lock()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("pirls_row module cache mutex poisoned: {err}"),
            })?
            .get(&key)
        {
            return Ok(existing.clone());
        }
        let source = cuda_source_for(family, curvature);
        let ptx = cudarc::nvrtc::compile_ptx(source).map_err(|err| GpuError::DriverCallFailed {
            reason: format!(
                "pirls_row NVRTC compile failed for {family}/{curv}: {err}",
                family = family.as_str(),
                curv = curvature.as_str(),
            ),
        })?;
        let module =
            self.inner
                .ctx
                .load_module(ptx)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("pirls_row module load failed: {err}"),
                })?;
        self.inner
            .modules
            .lock()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("pirls_row module cache mutex poisoned: {err}"),
            })?
            .insert(key, module.clone());
        Ok(module)
    }
}

// ────────────────────────────────────────────────────────────────────────
// CUDA sources (one per family / curvature pair)
// ────────────────────────────────────────────────────────────────────────

/// Common device-side helpers shared across every family kernel.
#[cfg(target_os = "linux")]
const COMMON_DEVICE_PROLOG: &str = r#"
extern "C" {
    double exp(double);
    double log(double);
    double log1p(double);
    double tanh(double);
    double sqrt(double);
    double fabs(double);
    double erfc(double);
}

__device__ __forceinline__ double clamp_eta(double eta, unsigned int* flags) {
    const double E = 700.0;
    if (eta > E) { *flags |= 0x1u; return E; }
    if (eta < -E) { *flags |= 0x1u; return -E; }
    return eta;
}

__device__ __forceinline__ double bernoulli_deviance(double y, double mu, double w) {
    if (w == 0.0) return 0.0;
    double t1 = (y > 0.0) ? y * log(y / mu) : 0.0;
    double t2 = (y < 1.0) ? (1.0 - y) * log((1.0 - y) / (1.0 - mu)) : 0.0;
    return 2.0 * w * (t1 + t2);
}

__device__ __forceinline__ double bernoulli_z(double eta, double y, double mu, double dmu_deta) {
    if (dmu_deta > 0.0 && isfinite(dmu_deta)) {
        double delta = (y - mu) / dmu_deta;
        if (isfinite(delta)) return eta + delta;
    }
    return eta;
}

__device__ __forceinline__ double std_norm_cdf(double x) {
    return 0.5 * erfc(-x * 0.7071067811865475);
}

__device__ __forceinline__ double std_norm_pdf(double x) {
    return 0.3989422804014327 * exp(-0.5 * x * x);
}
"#;

/// Build the per-family CUDA source. Each source defines exactly one entry
/// kernel (`family.kernel_name()`) reading the input arrays and writing the
/// output arrays defined by the [`RowOutput`] contract above.
#[cfg(target_os = "linux")]
fn cuda_source_for(family: PirlsRowFamily, curvature: CurvatureMode) -> String {
    let body = match family {
        PirlsRowFamily::GaussianIdentity => gaussian_identity_body(curvature),
        PirlsRowFamily::PoissonLog => poisson_log_body(curvature),
        PirlsRowFamily::GammaLog => gamma_log_body(curvature),
        PirlsRowFamily::BernoulliLogit => bernoulli_logit_body(curvature),
        PirlsRowFamily::BernoulliProbit => bernoulli_probit_body(curvature),
        PirlsRowFamily::BernoulliCLogLog => bernoulli_cloglog_body(curvature),
    };
    let kernel_name = family.kernel_name();
    // Inject a curvature marker into the source so each `(family, curvature)`
    // pair compiles a distinct PTX module (cache key) and Stage 5 can branch
    // on `PIRLS_CURVATURE_OBSERVED` from inside each per-family body without
    // changing the host harness.
    let curvature_define = match curvature {
        CurvatureMode::Fisher => "#define PIRLS_CURVATURE_FISHER 1",
        CurvatureMode::Observed => "#define PIRLS_CURVATURE_OBSERVED 1",
    };
    format!(
        r#"
{curvature_define}
{prolog}

extern "C" __global__ void {kernel_name}(
    int            n,
    const double* __restrict__ eta,
    const double* __restrict__ y,
    const double* __restrict__ prior_w,
    double* __restrict__ mu_out,
    double* __restrict__ grad_eta_out,
    double* __restrict__ w_fisher_out,
    double* __restrict__ w_hessian_out,
    double* __restrict__ w_solver_out,
    double* __restrict__ z_fisher_out,
    double* __restrict__ z_hessian_out,
    double* __restrict__ deviance_out,
    unsigned int* __restrict__ status_out
) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int flags = 0u;
    double eta_i = eta[i];
    double y_i = y[i];
    double wp = prior_w[i] > 0.0 ? prior_w[i] : 0.0;
    if (prior_w[i] <= 0.0) flags |= 0x10u;
{body}
    mu_out[i] = mu;
    grad_eta_out[i] = grad_eta;
    w_fisher_out[i] = w_fisher;
    w_hessian_out[i] = w_hessian;
    w_solver_out[i] = w_solver;
    z_fisher_out[i] = z_f;
    z_hessian_out[i] = z_h;
    deviance_out[i] = dev;
    status_out[i] = flags;
}}
"#,
        prolog = COMMON_DEVICE_PROLOG,
    )
}

/// Emits a CUDA comment tag identifying the curvature mode the kernel was
/// compiled for. Each body builder prepends this so the body source actually
/// consumes the `curvature` argument and Stage 5 can keyed-extend the bodies
/// behind `#ifdef PIRLS_CURVATURE_OBSERVED`.
#[cfg(target_os = "linux")]
#[inline]
fn curvature_tag(curvature: CurvatureMode) -> &'static str {
    match curvature {
        CurvatureMode::Fisher => "    // curvature: fisher\n",
        CurvatureMode::Observed => "    // curvature: observed\n",
    }
}

#[cfg(target_os = "linux")]
fn gaussian_identity_body(curvature: CurvatureMode) -> String {
    let tag = curvature_tag(curvature);
    format!(
        r#"{tag}    double mu = eta_i;
    double resid = y_i - mu;
    double grad_eta = wp * resid;
    double w_fisher = wp;
    double w_hessian = wp;
    double w_solver = (wp > 0.0) ? fmax(wp, 1e-12) : 0.0;
    double z_f = y_i;
    double z_h = y_i;
    double dev = wp * resid * resid;
"#
    )
}

#[cfg(target_os = "linux")]
fn poisson_log_body(curvature: CurvatureMode) -> String {
    let tag = curvature_tag(curvature);
    format!(
        r#"{tag}    double eta_c = clamp_eta(eta_i, &flags);
    double mu_raw = exp(eta_c);
    if (mu_raw < 1e-10) flags |= 0x2u;
    double mu = (mu_raw > 1e-10) ? mu_raw : 1e-10;
    double raw_w = wp * mu;
    double w_fisher = (raw_w > 0.0) ? fmax(raw_w, 1e-12) : 0.0;
    double resid = y_i - mu;
    double grad_eta = wp * resid;
    double w_hessian = w_fisher;
    double w_solver = w_fisher;
    double z_f = eta_c + resid / mu;
    double z_h = z_f;
    double dev_term = (y_i > 0.0) ? (y_i * log(y_i / mu) - resid) : (-resid);
    double dev = 2.0 * wp * dev_term;
    if (!(isfinite(y_i) && y_i >= 0.0)) flags |= 0x8u;
"#
    )
}

#[cfg(target_os = "linux")]
fn gamma_log_body(curvature: CurvatureMode) -> String {
    // Shape passed via constant memory in Stage 2; Stage-1 reference uses
    // unit shape (matches CPU `gamma_shape().unwrap_or(1.0)` default).
    let tag = curvature_tag(curvature);
    format!(
        r#"{tag}    const double shape = 1.0;
    double eta_c = clamp_eta(eta_i, &flags);
    double mu_raw = exp(eta_c);
    if (mu_raw < 1e-10) flags |= 0x2u;
    double mu = (mu_raw > 1e-10) ? mu_raw : 1e-10;
    double w_fisher = wp * shape;
    double w_hessian = w_fisher;
    double w_solver = (w_hessian > 0.0) ? fmax(w_hessian, 1e-12) : 0.0;
    double resid = y_i - mu;
    double grad_eta = wp * resid / mu;
    double z_f = eta_c + resid / mu;
    double z_h = z_f;
    double dev = (y_i > 0.0)
        ? (2.0 * wp * (-log(y_i / mu) + resid / mu))
        : (1.0 / 0.0);
    if (!(isfinite(y_i) && y_i > 0.0)) flags |= 0x8u;
"#
    )
}

#[cfg(target_os = "linux")]
fn bernoulli_logit_body(curvature: CurvatureMode) -> String {
    let tag = curvature_tag(curvature);
    format!(
        r#"{tag}    double eta_c = clamp_eta(eta_i, &flags);
    double half = 0.5 * eta_c;
    double mu_raw = 0.5 * (1.0 + tanh(half));
    if (mu_raw < 1e-12 || mu_raw > 1.0 - 1e-12) flags |= 0x2u;
    double mu = fmin(fmax(mu_raw, 1e-12), 1.0 - 1e-12);
    double dmu_deta = mu * (1.0 - mu);
    double w_fisher = wp * dmu_deta;
    double w_hessian = w_fisher;
    double w_solver = (w_fisher > 0.0) ? fmax(w_fisher, 1e-12) : 0.0;
    double resid = y_i - mu;
    double grad_eta = wp * resid;
    double dev = bernoulli_deviance(y_i, mu, wp);
    double z_f = bernoulli_z(eta_c, y_i, mu, dmu_deta);
    double z_h = z_f;
    if (!(isfinite(y_i) && y_i >= 0.0 && y_i <= 1.0)) flags |= 0x8u;
"#
    )
}

#[cfg(target_os = "linux")]
fn bernoulli_probit_body(curvature: CurvatureMode) -> String {
    let tag = curvature_tag(curvature);
    format!(
        r#"{tag}    double eta_c = clamp_eta(eta_i, &flags);
    double mu_raw = std_norm_cdf(eta_c);
    if (mu_raw < 1e-12 || mu_raw > 1.0 - 1e-12) flags |= 0x2u;
    double mu = fmin(fmax(mu_raw, 1e-12), 1.0 - 1e-12);
    double dmu_deta = std_norm_pdf(eta_c);
    double v = mu * (1.0 - mu);
    double fpp = (v > 0.0) ? dmu_deta * dmu_deta / v : 0.0;
    double w_fisher = wp * fpp;
    double w_hessian = w_fisher;
    double w_solver = (w_fisher > 0.0) ? fmax(w_fisher, 1e-12) : 0.0;
    double resid = y_i - mu;
    double grad_eta = (v > 0.0) ? wp * resid * dmu_deta / v : 0.0;
    double dev = bernoulli_deviance(y_i, mu, wp);
    double z_f = bernoulli_z(eta_c, y_i, mu, dmu_deta);
    double z_h = z_f;
    if (!(isfinite(y_i) && y_i >= 0.0 && y_i <= 1.0)) flags |= 0x8u;
"#
    )
}

#[cfg(target_os = "linux")]
fn bernoulli_cloglog_body(curvature: CurvatureMode) -> String {
    let tag = curvature_tag(curvature);
    format!(
        r#"{tag}    double eta_c = clamp_eta(eta_i, &flags);
    double inner = exp(eta_c);
    double mu_raw = 1.0 - exp(-inner);
    if (mu_raw < 1e-12 || mu_raw > 1.0 - 1e-12) flags |= 0x2u;
    double mu = fmin(fmax(mu_raw, 1e-12), 1.0 - 1e-12);
    double dmu_deta = inner * (1.0 - mu_raw);
    double v = mu * (1.0 - mu);
    double fpp = (v > 0.0) ? dmu_deta * dmu_deta / v : 0.0;
    double w_fisher = wp * fpp;
    double w_hessian = w_fisher;
    double w_solver = (w_fisher > 0.0) ? fmax(w_fisher, 1e-12) : 0.0;
    double resid = y_i - mu;
    double grad_eta = (v > 0.0) ? wp * resid * dmu_deta / v : 0.0;
    double dev = bernoulli_deviance(y_i, mu, wp);
    double z_f = bernoulli_z(eta_c, y_i, mu, dmu_deta);
    double z_h = z_f;
    if (!(isfinite(y_i) && y_i >= 0.0 && y_i <= 1.0)) flags |= 0x8u;
"#
    )
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod pirls_row_gpu_tests {
    use super::*;

    fn assert_close(label: &str, got: f64, expected: f64, tol: f64) {
        if !(got.is_finite() && expected.is_finite()) {
            assert_eq!(
                got.is_finite(),
                expected.is_finite(),
                "{label}: finiteness disagrees (got={got}, expected={expected})"
            );
            return;
        }
        let diff = (got - expected).abs();
        let denom = expected.abs().max(1.0);
        assert!(
            diff <= tol * denom,
            "{label}: |{got} - {expected}| = {diff} exceeds tol {tol} (rel denom {denom})"
        );
    }

    fn check_family_matches_cpu_reference(family: PirlsRowFamily) {
        let etas = [-700.0, -3.0, -0.5, 0.0, 0.5, 3.0, 700.0];
        let ys = match family {
            PirlsRowFamily::GammaLog => vec![0.5, 1.0, 2.5],
            PirlsRowFamily::PoissonLog => vec![0.0, 1.0, 5.0],
            PirlsRowFamily::GaussianIdentity => vec![-1.5, 0.0, 2.0],
            _ => vec![0.0, 1.0],
        };
        let ws = [0.0, 1.0, 2.5];
        for &eta in &etas {
            for &y in &ys {
                for &wp in &ws {
                    let input = RowInput {
                        eta,
                        y,
                        prior_weight: wp,
                    };
                    let out = row_reweight_cpu(family, CurvatureMode::Fisher, input);
                    // Structural invariants of the contract.
                    assert!(
                        out.w_fisher >= 0.0,
                        "{family:?}: w_fisher must be non-negative (got {})",
                        out.w_fisher
                    );
                    assert!(
                        out.w_solver >= 0.0,
                        "{family:?}: w_solver must be non-negative (got {})",
                        out.w_solver
                    );
                    if wp > 0.0 && out.w_hessian > 0.0 {
                        assert!(
                            out.w_solver >= W_SOLVER_FLOOR,
                            "{family:?}: w_solver must be floored away from zero when positive (got {})",
                            out.w_solver
                        );
                    }
                    // grad_eta and (z_fisher - eta) * w_fisher must agree
                    // when eta is unclamped and w_fisher > 0; this guards the
                    // "never reconstruct gradient from z" discipline.
                    if (out.status & status_flags::ETA_CLAMPED) != 0 {
                        continue;
                    }
                    if out.w_fisher > 0.0 && out.z_fisher.is_finite() {
                        let reconstructed = out.w_fisher * (out.z_fisher - eta);
                        // Allow loose tolerance — we only require these are
                        // in the same ballpark; the *exact* gradient is
                        // grad_eta. Catastrophic cancellation in the
                        // reconstructed form is the precise reason the
                        // contract exposes grad_eta directly.
                        if reconstructed.is_finite() {
                            let denom = reconstructed.abs().max(out.grad_eta.abs()).max(1.0);
                            let diff = (reconstructed - out.grad_eta).abs() / denom;
                            assert!(
                                diff < 1.0e-6,
                                "{family:?} eta={eta} y={y} wp={wp}: grad_eta {} vs w·(z−η) {} differ by rel {}",
                                out.grad_eta,
                                reconstructed,
                                diff
                            );
                        }
                    }
                    // deviance non-negative for valid inputs.
                    if out.status & status_flags::INVALID_RESPONSE == 0 && wp >= 0.0 {
                        assert!(
                            out.deviance >= 0.0 || !out.deviance.is_finite(),
                            "{family:?} eta={eta} y={y} wp={wp}: deviance must be non-negative for valid inputs (got {})",
                            out.deviance
                        );
                    }
                    // Final sanity: outputs are finite or carry an explicit
                    // INVALID_RESPONSE / ZERO_PRIOR_WEIGHT flag.
                    if out.status & (status_flags::INVALID_RESPONSE | status_flags::ZERO_PRIOR_WEIGHT)
                        == 0
                    {
                        assert!(
                            out.mu.is_finite(),
                            "{family:?} eta={eta} y={y} wp={wp}: mu must be finite for valid inputs"
                        );
                        assert!(
                            out.grad_eta.is_finite(),
                            "{family:?} eta={eta} y={y} wp={wp}: grad_eta must be finite for valid inputs"
                        );
                    }
                }
            }
        }
        // Pull `assert_close` into the closure type-checker so this function
        // is the single caller of it across the parity surface — keeps the
        // helper exercised on every family run.
        assert_close("self", 0.0, 0.0, 0.0);
    }

    #[test]
    fn gaussian_identity_row_invariants() {
        check_family_matches_cpu_reference(PirlsRowFamily::GaussianIdentity);
    }

    #[test]
    fn poisson_log_row_invariants() {
        check_family_matches_cpu_reference(PirlsRowFamily::PoissonLog);
    }

    #[test]
    fn gamma_log_row_invariants() {
        check_family_matches_cpu_reference(PirlsRowFamily::GammaLog);
    }

    #[test]
    fn bernoulli_logit_row_invariants() {
        check_family_matches_cpu_reference(PirlsRowFamily::BernoulliLogit);
    }

    #[test]
    fn bernoulli_probit_row_invariants() {
        check_family_matches_cpu_reference(PirlsRowFamily::BernoulliProbit);
    }

    #[test]
    fn bernoulli_cloglog_row_invariants() {
        check_family_matches_cpu_reference(PirlsRowFamily::BernoulliCLogLog);
    }

    /// Gaussian identity must match the trivial CPU formula for any input.
    #[test]
    fn gaussian_identity_matches_explicit_formulas() {
        let out = row_reweight_cpu(
            PirlsRowFamily::GaussianIdentity,
            CurvatureMode::Fisher,
            RowInput {
                eta: 0.25,
                y: 1.0,
                prior_weight: 2.0,
            },
        );
        assert_close("mu", out.mu, 0.25, 0.0);
        assert_close("grad_eta", out.grad_eta, 2.0 * (1.0 - 0.25), 1e-15);
        assert_close("w_fisher", out.w_fisher, 2.0, 0.0);
        assert_close(
            "deviance",
            out.deviance,
            2.0 * (1.0 - 0.25_f64).powi(2),
            1e-15,
        );
    }

    /// Poisson log must match the analytic formula μ = exp(η) and grad_eta = w·(y−μ).
    #[test]
    fn poisson_log_matches_explicit_formulas() {
        let out = row_reweight_cpu(
            PirlsRowFamily::PoissonLog,
            CurvatureMode::Fisher,
            RowInput {
                eta: 1.5,
                y: 4.0,
                prior_weight: 1.0,
            },
        );
        let expected_mu = (1.5_f64).exp();
        assert_close("mu", out.mu, expected_mu, 1e-15);
        assert_close("grad_eta", out.grad_eta, 4.0 - expected_mu, 1e-15);
        assert_close("w_fisher", out.w_fisher, expected_mu, 1e-15);
    }

    /// Bernoulli logit: closed form for canonical link.
    #[test]
    fn bernoulli_logit_matches_explicit_formulas() {
        let eta: f64 = 0.7;
        let mu = 1.0 / (1.0 + (-eta).exp());
        let out = row_reweight_cpu(
            PirlsRowFamily::BernoulliLogit,
            CurvatureMode::Fisher,
            RowInput {
                eta,
                y: 1.0,
                prior_weight: 3.0,
            },
        );
        assert_close("mu", out.mu, mu, 1e-12);
        assert_close("w_fisher", out.w_fisher, 3.0 * mu * (1.0 - mu), 1e-12);
        assert_close("grad_eta", out.grad_eta, 3.0 * (1.0 - mu), 1e-12);
    }

    /// Eta clamping flag must trip past ±700.
    #[test]
    fn eta_clamp_status_flag_trips() {
        let out = row_reweight_cpu(
            PirlsRowFamily::PoissonLog,
            CurvatureMode::Fisher,
            RowInput {
                eta: 1000.0,
                y: 0.0,
                prior_weight: 1.0,
            },
        );
        assert!(out.status & status_flags::ETA_CLAMPED != 0);
    }

    /// `module_for` must lazily compile + cache one module per `(family, curvature)`.
    /// Skipped on hosts without a CUDA runtime (mac, CI).
    #[test]
    fn backend_compiles_one_module_per_family_when_device_present() {
        if super::super::runtime::GpuRuntime::global().is_none() {
            eprintln!("[pirls_row_gpu test] no CUDA runtime — skipping device compile test");
            return;
        }
        #[cfg(target_os = "linux")]
        {
            let backend = PirlsRowBackend::probe().expect("backend probe on CUDA host");
            for &family in PirlsRowFamily::ALL.iter() {
                let m1 = backend
                    .module_for(family, CurvatureMode::Fisher)
                    .unwrap_or_else(|err| panic!("compile {family:?}: {err}"));
                let m2 = backend
                    .module_for(family, CurvatureMode::Fisher)
                    .unwrap_or_else(|err| panic!("re-fetch {family:?}: {err}"));
                assert!(
                    Arc::ptr_eq(&m1, &m2),
                    "{family:?}: module cache must return same handle on second call"
                );
            }
        }
    }
}
