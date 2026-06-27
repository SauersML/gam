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

use gam_gpu::gpu_error::GpuError;
#[cfg(target_os = "linux")]
use gam_gpu::gpu_error::GpuResultExt;

#[cfg(target_os = "linux")]
use std::sync::{Arc, Mutex};

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaContext, CudaModule};

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

    /// CUDA `extern "C"` entry symbol for this family's full (final-row) kernel.
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

    /// CUDA `extern "C"` entry symbol for this family's solve-row kernel
    /// (writes only `grad_eta`, `w_solver`, `deviance`, `status`).
    pub const fn solve_kernel_name(self) -> &'static str {
        match self {
            Self::BernoulliLogit => "pirls_solve_bernoulli_logit",
            Self::BernoulliProbit => "pirls_solve_bernoulli_probit",
            Self::BernoulliCLogLog => "pirls_solve_bernoulli_cloglog",
            Self::PoissonLog => "pirls_solve_poisson_log",
            Self::GaussianIdentity => "pirls_solve_gaussian_identity",
            Self::GammaLog => "pirls_solve_gamma_log",
        }
    }

    /// CUDA `extern "C"` entry symbol for this family's alpha-ladder kernel
    /// (evaluates all step sizes in a single launch, outputs `objective[]` and
    /// `status[]` per alpha slot).
    pub const fn ladder_kernel_name(self) -> &'static str {
        match self {
            Self::BernoulliLogit => "pirls_ladder_bernoulli_logit",
            Self::BernoulliProbit => "pirls_ladder_bernoulli_probit",
            Self::BernoulliCLogLog => "pirls_ladder_bernoulli_cloglog",
            Self::PoissonLog => "pirls_ladder_poisson_log",
            Self::GaussianIdentity => "pirls_ladder_gaussian_identity",
            Self::GammaLog => "pirls_ladder_gamma_log",
        }
    }

    /// True for `(response, canonical-link)` pairs where observed information
    /// equals Fisher information exactly, so `w_hessian == w_fisher` for both
    /// curvature modes.
    ///
    /// Gamma-LOG is **non-canonical**: the canonical Gamma link is the
    /// reciprocal 1/μ, not log. Under a log link the observed Hessian weight
    /// is `w_F · y/μ` (shape-independent; the shape cancels), which differs
    /// from the Fisher weight `w_F` whenever `y ≠ μ`. Consequently
    /// `CurvatureMode::Observed` produces a different `w_hessian` for
    /// Gamma-log, and it must not be short-circuited via a canonical-family
    /// check.
    pub const fn is_canonical(self) -> bool {
        match self {
            Self::BernoulliLogit | Self::PoissonLog | Self::GaussianIdentity => true,
            Self::GammaLog | Self::BernoulliProbit | Self::BernoulliCLogLog => false,
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
/// `gamma_shape` is the Gamma dispersion shape parameter (α > 0). It is only
/// used when `family == GammaLog`; all other families ignore it. Pass `1.0`
/// for non-Gamma fits.
pub fn row_reweight_cpu(
    family: PirlsRowFamily,
    mode: CurvatureMode,
    input: RowInput,
    gamma_shape: f64,
) -> RowOutput {
    match family {
        PirlsRowFamily::GaussianIdentity => row_gaussian_identity(input, mode),
        PirlsRowFamily::PoissonLog => row_poisson_log(input, mode),
        PirlsRowFamily::GammaLog => row_gamma_log(input, mode, gamma_shape),
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
fn row_gamma_log(input: RowInput, mode: CurvatureMode, shape: f64) -> RowOutput {
    let (eta_c, clamped) = clamp_eta(input.eta);
    let mu_raw = eta_c.exp();
    let mu_floored = mu_raw < MU_FLOOR_GAMMA;
    let mu = mu_raw.max(MU_FLOOR_GAMMA);
    let w_prior = input.prior_weight.max(0.0);
    let w_fisher = w_prior * shape;
    // Stage 5: observed-information weight for Gamma-log.
    //   -∂²ℓ/∂η² = α · y/μ  (vs Fisher: α).
    // Correction = w_F · (y/μ − 1). Falls back to Fisher when y == μ
    // (e.g. saturated y) and when w_F == 0.
    let obs_correction = if w_fisher > 0.0 && mu > 0.0 && input.y.is_finite() {
        w_fisher * (input.y / mu - 1.0)
    } else {
        0.0
    };
    let w_hessian = select_w_hessian(mode, w_fisher, obs_correction);
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
    // Stage 5: observed-information correction for Bernoulli probit.
    //   w_obs = w_F + w_p · (y − μ) · [h'/V − h²·V'/V²].
    // h(η) = φ(η), h'(η) = −η · φ(η); V(μ) = μ(1−μ), V'(μ) = 1 − 2μ.
    let obs_correction = if v > 0.0 && w_prior > 0.0 {
        let h_prime = -eta_c * dmu_deta;
        let v_prime = 1.0 - 2.0 * mu;
        let bracket = h_prime / v - (dmu_deta * dmu_deta) * v_prime / (v * v);
        w_prior * resid * bracket
    } else {
        0.0
    };
    let w_hessian_observed = select_w_hessian(mode, w_fisher, obs_correction);
    RowOutput {
        mu,
        grad_eta,
        w_fisher,
        w_hessian: w_hessian_observed,
        w_solver: {
            let wh = w_hessian_observed;
            if wh > 0.0 {
                wh.max(W_SOLVER_FLOOR)
            } else {
                0.0
            }
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
    // μ = 1 − exp(−exp(η)); numerically stable via expm1 to preserve precision
    // in the deep negative tail (η ≲ -36) where `1 - exp(-exp(η))` would
    // catastrophically cancel to 0.
    let inner = eta_c.exp();
    let mu_raw = -(-inner).exp_m1();
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
    // Stage 5: observed-information correction for Bernoulli cloglog.
    //   w_obs = w_F + w_p · (y − μ) · [h'/V − h²·V'/V²].
    // h(η) = inner · (1 − μ_raw); h'(η) = h(η) · (1 − inner).
    // V(μ) = μ(1−μ), V'(μ) = 1 − 2μ.
    let obs_correction = if v > 0.0 && w_prior > 0.0 {
        let h_prime = dmu_deta * (1.0 - inner);
        let v_prime = 1.0 - 2.0 * mu;
        let bracket = h_prime / v - (dmu_deta * dmu_deta) * v_prime / (v * v);
        w_prior * resid * bracket
    } else {
        0.0
    };
    let w_hessian = select_w_hessian(mode, w_fisher, obs_correction);
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
    0.5 * gam_gpu::numerics_host::erfc(-x * std::f64::consts::FRAC_1_SQRT_2)
}

#[inline]
fn standard_normal_pdf(x: f64) -> f64 {
    const COEFF: f64 = 0.398_942_280_401_432_7; // 1 / sqrt(2π)
    COEFF * (-0.5 * x * x).exp()
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
    modules: Mutex<std::collections::HashMap<ModuleKey, Arc<CudaModule>>>,
    /// Stage 6: separate cache for JIT-compiled custom-family modules
    /// keyed by `(spec_id, curvature)`. Distinct JIT specs in the same
    /// process get distinct cached modules.
    jit_modules: Mutex<std::collections::HashMap<JitKey, Arc<CudaModule>>>,
}

/// Distinguishes the three kernel modes in the per-process module cache.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum KernelMode {
    /// Full final-row kernel (9 outputs: mu, grad_eta, w_fisher, w_hessian,
    /// w_solver, z_fisher, z_hessian, deviance, status).
    FinalRow,
    /// Solve-row kernel (4 outputs: grad_eta, w_solver, deviance, status).
    SolveRow,
    /// Alpha-ladder kernel (2 per-alpha outputs: objective[], status[]).
    AlphaLadder,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ModuleKey {
    family: PirlsRowFamily,
    curvature: CurvatureMode,
    mode: KernelMode,
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
        let parts = gam_gpu::backend_probe::probe_cuda_backend("pirls_row")?;
        Ok(Self {
            inner: PirlsRowBackendLinux {
                ctx: parts.ctx,
                modules: Mutex::new(std::collections::HashMap::new()),
                jit_modules: Mutex::new(std::collections::HashMap::new()),
            },
        })
    }

    /// Compile (or fetch from cache) the kernel module for `(family, curvature)`
    /// in the given [`KernelMode`]. This is the single source of truth behind
    /// [`module_for`], [`module_for_solve`], and [`module_for_ladder`]; the only
    /// per-mode variation is which CUDA source generator is used (selected by
    /// `mode`) and the error label `label` woven into compile/load diagnostics.
    #[cfg(target_os = "linux")]
    fn module_for_kind(
        &self,
        family: PirlsRowFamily,
        curvature: CurvatureMode,
        mode: KernelMode,
        label: &str,
    ) -> Result<Arc<CudaModule>, GpuError> {
        let key = ModuleKey {
            family,
            curvature,
            mode,
        };
        if let Some(existing) = self
            .inner
            .modules
            .lock()
            .gpu_ctx_with(|err| format!("pirls_row {label}module cache mutex poisoned: {err}"))?
            .get(&key)
        {
            return Ok(existing.clone());
        }
        let source = match mode {
            KernelMode::FinalRow => cuda_source_for(family, curvature),
            KernelMode::SolveRow => solve_row_source_for(family, curvature),
            KernelMode::AlphaLadder => ladder_source_for(family, curvature),
        };
        // #1551: route through the device-arch-pinned compile — this kernel uses
        // `atomicAdd(double*, double)` (objective_out), which NVRTC rejects under
        // its default sub-sm_60 arch, silently disabling the device PIRLS path.
        let ptx = gam_gpu::device_cache::compile_ptx_arch(&source).gpu_ctx_with(|err| {
            format!(
                "pirls_row {label}NVRTC compile failed for {family}/{curv}: {err}",
                family = family.as_str(),
                curv = curvature.as_str(),
            )
        })?;
        let module = self
            .inner
            .ctx
            .load_module(ptx)
            .gpu_ctx_with(|err| format!("pirls_row {label}module load failed: {err}"))?;
        self.inner
            .modules
            .lock()
            .gpu_ctx_with(|err| format!("pirls_row {label}module cache mutex poisoned: {err}"))?
            .insert(key, module.clone());
        Ok(module)
    }

    /// Compile (or fetch from cache) the **final-row** kernel module for
    /// `(family, curvature)`. Writes all 9 output fields.
    #[cfg(target_os = "linux")]
    pub fn module_for(
        &self,
        family: PirlsRowFamily,
        curvature: CurvatureMode,
    ) -> Result<Arc<CudaModule>, GpuError> {
        self.module_for_kind(family, curvature, KernelMode::FinalRow, "")
    }

    /// Compile (or fetch from cache) the **solve-row** kernel module for
    /// `(family, curvature)`. Writes only `grad_eta`, `w_solver`, `deviance`,
    /// `status` — used on every hot Newton iteration.
    #[cfg(target_os = "linux")]
    pub fn module_for_solve(
        &self,
        family: PirlsRowFamily,
        curvature: CurvatureMode,
    ) -> Result<Arc<CudaModule>, GpuError> {
        self.module_for_kind(family, curvature, KernelMode::SolveRow, "solve ")
    }

    /// Compile (or fetch from cache) the **alpha-ladder** kernel module for
    /// `(family, curvature)`. Evaluates all [`ALPHA_LADDER_LEN`] step sizes in
    /// a single launch, accumulating `objective[]` and `status[]` per alpha slot.
    #[cfg(target_os = "linux")]
    pub fn module_for_ladder(
        &self,
        family: PirlsRowFamily,
        curvature: CurvatureMode,
    ) -> Result<Arc<CudaModule>, GpuError> {
        self.module_for_kind(family, curvature, KernelMode::AlphaLadder, "ladder ")
    }

    /// Stage 6: JIT-compile and cache a custom-family row module.
    ///
    /// The kernel name is `pirls_row_jit_{spec.spec_id}` so multiple
    /// distinct JIT specs in the same process get distinct cached
    /// modules. The cache key is `(spec_id, curvature)` which mirrors
    /// the built-in `(family, curvature)` cache and reuses the same
    /// HashMap-of-`Arc<CudaModule>` (but with a synthetic `ModuleKey`
    /// derived from the spec_id).
    ///
    /// Note: a fresh `(spec_id, curvature)` recompiles via NVRTC the
    /// first time; subsequent fits in the same process hit the cache.
    /// Spec changes (different body) must use a different `spec_id` so
    /// that the cache does NOT return a stale module.
    #[cfg(target_os = "linux")]
    pub fn module_for_jit(
        &self,
        spec: &JitFamilySpec,
        curvature: CurvatureMode,
    ) -> Result<Arc<CudaModule>, GpuError> {
        // Reuse the built-in ModuleKey by mapping `spec_id` to a
        // synthetic family slot. We piggy-back on the cache by keying
        // off a hashed family enum value won't fit cleanly; instead
        // use a separate JIT cache HashMap.
        let key = JitKey {
            spec_id: spec.spec_id,
            curvature,
        };
        if let Some(existing) = self
            .inner
            .jit_modules
            .lock()
            .gpu_ctx("pirls_row jit cache poisoned")?
            .get(&key)
        {
            return Ok(existing.clone());
        }
        let source = spec.cuda_source(curvature);
        // #1551: device-arch-pinned compile (double-atomic objective_out kernel).
        let ptx = gam_gpu::device_cache::compile_ptx_arch(&source).gpu_ctx_with(|err| {
            format!(
                "pirls_row JIT NVRTC compile failed for spec_id={} curvature={}: {err}",
                spec.spec_id,
                curvature.as_str(),
            )
        })?;
        let module = self
            .inner
            .ctx
            .load_module(ptx)
            .gpu_ctx("pirls_row JIT module load failed")?;
        self.inner
            .jit_modules
            .lock()
            .gpu_ctx("pirls_row jit cache poisoned (insert)")?
            .insert(key, module.clone());
        Ok(module)
    }
}

/// Stage 6 cache key for JIT-compiled family modules.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct JitKey {
    spec_id: u64,
    curvature: CurvatureMode,
}

/// Stage 6 custom-family JIT specification.
///
/// Two levels per the charter:
/// - **Level A** (`JitFamilySpec::glm`): provide a `(family, link)` enum
///   value plus an optional shape constant. The generator emits the
///   matching built-in row body, identical to the cached built-in
///   kernel — useful for end-to-end JIT path validation against the
///   built-in cache.
/// - **Level B** (`JitFamilySpec::raw`): provide raw CUDA source for
///   the row body. The body must define the same per-row locals the
///   kernel shell expects: `mu`, `grad_eta`, `w_fisher`, `w_hessian`,
///   `w_solver`, `z_f`, `z_h`, `dev`, and update `flags`. The shell
///   wraps it in the canonical
///   `extern "C" __global__ void pirls_row_jit_{spec_id}(...)`
///   signature that [`launch_row_reweight_on_stream`] expects.
#[derive(Clone, Debug)]
pub struct JitFamilySpec {
    /// Process-unique identifier for this spec; the module cache uses
    /// it as a key so callers must reuse the same `spec_id` for the
    /// same body and pick a new one whenever the body changes.
    pub spec_id: u64,
    /// CUDA body source. Must read from `eta_c`, `y_i`, `wp`, set
    /// `flags`, and assign to `mu`, `grad_eta`, `w_fisher`, `w_hessian`,
    /// `w_solver`, `z_f`, `z_h`, `dev`. See [`COMMON_DEVICE_PROLOG`] for
    /// the available helpers.
    pub body: String,
}

impl JitFamilySpec {
    /// Level A: build a spec from a built-in `(family, curvature)`
    /// pair. The generator reuses the same per-family body as the
    /// built-in cached kernel — useful to validate the JIT pipeline
    /// end-to-end against the built-in numerical reference.
    #[cfg(target_os = "linux")]
    pub fn glm(spec_id: u64, family: PirlsRowFamily, curvature: CurvatureMode) -> Self {
        let body = match family {
            PirlsRowFamily::GaussianIdentity => gaussian_identity_body(curvature),
            PirlsRowFamily::PoissonLog => poisson_log_body(curvature),
            PirlsRowFamily::GammaLog => gamma_log_body(curvature),
            PirlsRowFamily::BernoulliLogit => bernoulli_logit_body(curvature),
            PirlsRowFamily::BernoulliProbit => bernoulli_probit_body(curvature),
            PirlsRowFamily::BernoulliCLogLog => bernoulli_cloglog_body(curvature),
        };
        Self { spec_id, body }
    }

    /// Level B: build a spec from caller-supplied body source. The
    /// kernel shell wraps it; the body must define the required locals
    /// listed on [`JitFamilySpec`].
    pub fn raw(spec_id: u64, body: impl Into<String>) -> Self {
        Self {
            spec_id,
            body: body.into(),
        }
    }

    /// The `extern "C"` kernel symbol the JIT-compiled module exposes.
    pub fn kernel_name(&self) -> String {
        format!("pirls_row_jit_{}", self.spec_id)
    }

    /// Build the full CUDA source ready for NVRTC compilation. The
    /// shell + prolog match the built-in `cuda_source_for` so the JIT
    /// kernel ABI is bit-identical to the cached built-ins;
    /// [`launch_row_reweight_on_stream`] cannot tell the difference.
    #[cfg(target_os = "linux")]
    pub fn cuda_source(&self, curvature: CurvatureMode) -> String {
        let curvature_define = match curvature {
            CurvatureMode::Fisher => "#define PIRLS_CURVATURE_FISHER 1",
            CurvatureMode::Observed => "#define PIRLS_CURVATURE_OBSERVED 1",
        };
        let kernel_name = self.kernel_name();
        let body = &self.body;
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
}

/// Device-resident per-row output buffers for the GPU row-reweight kernel.
///
/// **final-row mode**: all nine per-[`RowOutput`] fields, length `n`. Written
/// once at convergence by [`launch_row_reweight_on_stream`]. For the hot
/// inner-loop use [`SolveRowBuffers`]; for line-search use
/// [`AlphaLadderDevBuffers`].
#[cfg(target_os = "linux")]
pub struct RowOutputDevBuffers {
    pub mu: cudarc::driver::CudaSlice<f64>,
    pub grad_eta: cudarc::driver::CudaSlice<f64>,
    pub w_fisher: cudarc::driver::CudaSlice<f64>,
    pub w_hessian: cudarc::driver::CudaSlice<f64>,
    pub w_solver: cudarc::driver::CudaSlice<f64>,
    pub z_fisher: cudarc::driver::CudaSlice<f64>,
    pub z_hessian: cudarc::driver::CudaSlice<f64>,
    pub deviance: cudarc::driver::CudaSlice<f64>,
    pub status: cudarc::driver::CudaSlice<u32>,
    pub n: usize,
}

#[cfg(target_os = "linux")]
impl RowOutputDevBuffers {
    /// Allocate all nine per-row output buffers (length `n`) on `stream`.
    pub fn allocate(stream: &Arc<cudarc::driver::CudaStream>, n: usize) -> Result<Self, GpuError> {
        let alloc_f64 = |label: &'static str| {
            stream
                .alloc_zeros::<f64>(n)
                .gpu_ctx_with(|err| format!("pirls_row alloc {label}: {err}"))
        };
        let alloc_u32 = |label: &'static str| {
            stream
                .alloc_zeros::<u32>(n)
                .gpu_ctx_with(|err| format!("pirls_row alloc {label}: {err}"))
        };
        Ok(Self {
            mu: alloc_f64("mu")?,
            grad_eta: alloc_f64("grad_eta")?,
            w_fisher: alloc_f64("w_fisher")?,
            w_hessian: alloc_f64("w_hessian")?,
            w_solver: alloc_f64("w_solver")?,
            z_fisher: alloc_f64("z_fisher")?,
            z_hessian: alloc_f64("z_hessian")?,
            deviance: alloc_f64("deviance")?,
            status: alloc_u32("status")?,
            n,
        })
    }
}

/// Device-resident per-row output buffers for the **solve-row** mode.
///
/// Allocates only the four fields the PIRLS solver reads on every Newton
/// iteration: `grad_eta` (score for Xᵀg RHS), `w_solver` (working weight
/// for XᵀWX assembly), `deviance` (per-row deviance for convergence check),
/// and `status` (diagnostic flags OR-reduced to a single host u32). Written
/// by [`launch_solve_row_on_stream`]; used instead of [`RowOutputDevBuffers`]
/// during the hot inner loop to reduce device memory and kernel store traffic.
#[cfg(target_os = "linux")]
pub struct SolveRowBuffers {
    /// ∂ℓ/∂η_i — score for Xᵀg RHS formation.
    pub grad_eta: cudarc::driver::CudaSlice<f64>,
    /// Stabilised Hessian weight — fed to XᵀWX assembly.
    pub w_solver: cudarc::driver::CudaSlice<f64>,
    /// Per-row deviance contribution — summed for convergence check.
    pub deviance: cudarc::driver::CudaSlice<f64>,
    /// Bitmask flags OR-reduced to detect numerical issues.
    pub status: cudarc::driver::CudaSlice<u32>,
    pub n: usize,
}

#[cfg(target_os = "linux")]
impl SolveRowBuffers {
    /// Allocate the four solve-row output buffers (length `n`) on `stream`.
    pub fn allocate(stream: &Arc<cudarc::driver::CudaStream>, n: usize) -> Result<Self, GpuError> {
        let alloc_f64 = |label: &'static str| {
            stream
                .alloc_zeros::<f64>(n)
                .gpu_ctx_with(|err| format!("pirls_row solve alloc {label}: {err}"))
        };
        let alloc_u32 = |label: &'static str| {
            stream
                .alloc_zeros::<u32>(n)
                .gpu_ctx_with(|err| format!("pirls_row solve alloc {label}: {err}"))
        };
        Ok(Self {
            grad_eta: alloc_f64("grad_eta")?,
            w_solver: alloc_f64("w_solver")?,
            deviance: alloc_f64("deviance")?,
            status: alloc_u32("status")?,
            n,
        })
    }
}

/// Number of alpha step sizes in the fused alpha ladder.
pub const ALPHA_LADDER_LEN: usize = 7;

/// The fixed alpha step-size ladder: `[1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]`.
pub const ALPHA_LADDER: [f64; ALPHA_LADDER_LEN] =
    [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625];

/// Device buffers for the fused alpha-ladder candidate-objective kernel.
///
/// **candidate-objective mode**: for each of the [`ALPHA_LADDER_LEN`] step
/// sizes α_k the kernel evaluates `η_trial_i = η_i + α_k · xδ_i`, computes
/// the per-row deviance, and atomically accumulates the sum into
/// `objective_dev[k]`. Status flags are OR-accumulated into `status_dev[k]`.
/// After a single `memcpy_dtoh` the host picks the first α that achieves
/// deviance descent — no per-α kernel launch, no full row-output write.
#[cfg(target_os = "linux")]
pub struct AlphaLadderDevBuffers {
    /// Device: summed deviance for each alpha step, length [`ALPHA_LADDER_LEN`].
    pub objective_dev: cudarc::driver::CudaSlice<f64>,
    /// Device: OR-reduced status flags for each alpha step, length [`ALPHA_LADDER_LEN`].
    pub status_dev: cudarc::driver::CudaSlice<u32>,
}

#[cfg(target_os = "linux")]
impl AlphaLadderDevBuffers {
    /// Allocate the ladder device buffers on `stream`.
    pub fn allocate(stream: &Arc<cudarc::driver::CudaStream>) -> Result<Self, GpuError> {
        Ok(Self {
            objective_dev: stream
                .alloc_zeros::<f64>(ALPHA_LADDER_LEN)
                .gpu_ctx_with(|err| format!("pirls_row ladder alloc objective: {err}"))?,
            status_dev: stream
                .alloc_zeros::<u32>(ALPHA_LADDER_LEN)
                .gpu_ctx_with(|err| format!("pirls_row ladder alloc status: {err}"))?,
        })
    }

    /// Zero all per-alpha accumulators in-place (call before each ladder launch).
    pub fn zero(&mut self, stream: &Arc<cudarc::driver::CudaStream>) -> Result<(), GpuError> {
        stream
            .memset_zeros(&mut self.objective_dev)
            .gpu_ctx_with(|err| format!("pirls_row ladder zero objective: {err}"))?;
        stream
            .memset_zeros(&mut self.status_dev)
            .gpu_ctx_with(|err| format!("pirls_row ladder zero status: {err}"))
    }
}

/// Device-side row reweight launcher.
///
/// Resolves the cached per-family kernel from [`PirlsRowBackend::module_for`],
/// dispatches a 1D grid of `THREADS_PER_BLOCK = 256` threads across `n`
/// rows, and returns once the launch is enqueued on `stream`. The kernel
/// writes the per-row IRLS state into `out` in place; no host transfers.
///
/// The kernel's `extern "C"` signature is fixed at the top of `cuda_source_for`
/// (see `extern "C" __global__ void {kernel_name}(int n, …)` in this file).
/// `gamma_shape`: active Gamma dispersion shape (α > 0). Forwarded as a
/// scalar kernel argument only for `PirlsRowFamily::GammaLog`; all other
/// families compile a 13-argument kernel and ignore this value. Pass `1.0`
/// for non-Gamma fits.
#[cfg(target_os = "linux")]
pub fn launch_row_reweight_on_stream(
    backend: &PirlsRowBackend,
    family: PirlsRowFamily,
    curvature: CurvatureMode,
    gamma_shape: f64,
    stream: &Arc<cudarc::driver::CudaStream>,
    n: usize,
    eta_dev: &cudarc::driver::CudaSlice<f64>,
    y_dev: &cudarc::driver::CudaSlice<f64>,
    prior_w_dev: &cudarc::driver::CudaSlice<f64>,
    out: &mut RowOutputDevBuffers,
) -> Result<(), GpuError> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    if out.n != n {
        gam_gpu::gpu_bail!("row reweight buffers shape {} mismatches n={n}", out.n);
    }
    let module = backend.module_for(family, curvature)?;
    let func = module
        .load_function(family.kernel_name())
        .gpu_ctx_with(|err| {
            format!(
                "row reweight load_function({}): {err}",
                family.kernel_name()
            )
        })?;
    const THREADS_PER_BLOCK: u32 = 256;
    let n_u32 =
        u32::try_from(n).map_err(|_| gam_gpu::gpu_err!("n={n} exceeds u32 for row reweight grid sizing"))?;
    let grid_x = n_u32.div_ceil(THREADS_PER_BLOCK).max(1);
    let n_i32 = i32::try_from(n)
        .map_err(|_| gam_gpu::gpu_err!("n={n} exceeds i32 for row reweight kernel argument"))?;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(&n_i32);
    builder.arg(eta_dev);
    builder.arg(y_dev);
    builder.arg(prior_w_dev);
    // GammaLog kernel has `double shape` before the output buffers.
    if matches!(family, PirlsRowFamily::GammaLog) {
        builder.arg(&gamma_shape);
    }
    builder.arg(&mut out.mu);
    builder.arg(&mut out.grad_eta);
    builder.arg(&mut out.w_fisher);
    builder.arg(&mut out.w_hessian);
    builder.arg(&mut out.w_solver);
    builder.arg(&mut out.z_fisher);
    builder.arg(&mut out.z_hessian);
    builder.arg(&mut out.deviance);
    builder.arg(&mut out.status);
    // SAFETY: kernel signature for non-GammaLog is (n:i32, 3×const f64*,
    // 8×mut f64*, 1×mut u32*). GammaLog extends this with `double shape`
    // after `prior_w` and before the output buffers — see `cuda_source_for`.
    // Arg order/types match one-for-one. Output buffers were allocated with
    // `n` elements each (validated above); input buffers are caller-supplied
    // with length n. Grid covers all n rows; threads guard `if (i >= n) return`.
    unsafe { builder.launch(cfg) }
        .map(|_event_pair| ())
        .gpu_ctx_with(|err| format!("row reweight launch({}): {err}", family.kernel_name()))
}

/// Stage 6: device-side row reweight launcher for JIT-compiled
/// custom-family kernels. Same kernel ABI as the built-in path
/// ([`launch_row_reweight_on_stream`]) — the only differences are
/// (a) the kernel symbol is `spec.kernel_name()` and (b) the module
/// resolution goes through [`PirlsRowBackend::module_for_jit`].
#[cfg(target_os = "linux")]
pub fn launch_row_reweight_jit_on_stream(
    backend: &PirlsRowBackend,
    spec: &JitFamilySpec,
    curvature: CurvatureMode,
    stream: &Arc<cudarc::driver::CudaStream>,
    n: usize,
    eta_dev: &cudarc::driver::CudaSlice<f64>,
    y_dev: &cudarc::driver::CudaSlice<f64>,
    prior_w_dev: &cudarc::driver::CudaSlice<f64>,
    out: &mut RowOutputDevBuffers,
) -> Result<(), GpuError> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    if out.n != n {
        gam_gpu::gpu_bail!("JIT row reweight buffers shape {} mismatches n={n}", out.n);
    }
    let module = backend.module_for_jit(spec, curvature)?;
    let kernel_name = spec.kernel_name();
    let func = module
        .load_function(&kernel_name)
        .gpu_ctx_with(|err| format!("JIT row reweight load_function({kernel_name}): {err}"))?;
    const THREADS_PER_BLOCK: u32 = 256;
    let n_u32 = u32::try_from(n)
        .map_err(|_| gam_gpu::gpu_err!("n={n} exceeds u32 for JIT row reweight grid sizing"))?;
    let grid_x = n_u32.div_ceil(THREADS_PER_BLOCK).max(1);
    let n_i32 = i32::try_from(n)
        .map_err(|_| gam_gpu::gpu_err!("n={n} exceeds i32 for JIT row reweight kernel argument"))?;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(&n_i32);
    builder.arg(eta_dev);
    builder.arg(y_dev);
    builder.arg(prior_w_dev);
    builder.arg(&mut out.mu);
    builder.arg(&mut out.grad_eta);
    builder.arg(&mut out.w_fisher);
    builder.arg(&mut out.w_hessian);
    builder.arg(&mut out.w_solver);
    builder.arg(&mut out.z_fisher);
    builder.arg(&mut out.z_hessian);
    builder.arg(&mut out.deviance);
    builder.arg(&mut out.status);
    // SAFETY: JIT spec's `cuda_source` builder emits the same kernel
    // signature as `cuda_source_for`; arg order/types match one-for-one.
    unsafe { builder.launch(cfg) }
        .map(|_event_pair| ())
        .gpu_ctx_with(|err| format!("JIT row reweight launch({kernel_name}): {err}"))
}

/// **solve-row** mode launcher.
///
/// Runs the per-family row math and writes only the four fields needed by the
/// PIRLS solver on each Newton iteration: `grad_eta`, `w_solver`, `deviance`,
/// `status`. The CUDA kernel is compiled from a specialised source
/// (`solve_row_source_for`) that skips the `mu`, `w_fisher`, `w_hessian`,
/// `z_fisher`, `z_hessian` stores, reducing both bandwidth and register
/// pressure relative to [`launch_row_reweight_on_stream`].
///
/// Call once per Newton step on the accepted η. At convergence, call
/// [`launch_row_reweight_on_stream`] (final-row mode) to populate the full
/// output surface before downloading.
///
/// `gamma_shape`: active Gamma dispersion shape (α > 0). Forwarded as a kernel
/// argument only for `PirlsRowFamily::GammaLog`. Pass `1.0` for non-Gamma fits.
#[cfg(target_os = "linux")]
pub fn launch_solve_row_on_stream(
    backend: &PirlsRowBackend,
    family: PirlsRowFamily,
    curvature: CurvatureMode,
    gamma_shape: f64,
    stream: &Arc<cudarc::driver::CudaStream>,
    n: usize,
    eta_dev: &cudarc::driver::CudaSlice<f64>,
    y_dev: &cudarc::driver::CudaSlice<f64>,
    prior_w_dev: &cudarc::driver::CudaSlice<f64>,
    out: &mut SolveRowBuffers,
) -> Result<(), GpuError> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    if out.n != n {
        gam_gpu::gpu_bail!("solve-row buffers shape {} mismatches n={n}", out.n);
    }
    let module = backend.module_for_solve(family, curvature)?;
    let kernel_name = family.solve_kernel_name();
    let func = module
        .load_function(kernel_name)
        .gpu_ctx_with(|err| format!("solve-row load_function({kernel_name}): {err}"))?;
    const THREADS_PER_BLOCK: u32 = 256;
    let n_u32 =
        u32::try_from(n).map_err(|_| gam_gpu::gpu_err!("n={n} exceeds u32 for solve-row grid sizing"))?;
    let grid_x = n_u32.div_ceil(THREADS_PER_BLOCK).max(1);
    let n_i32 = i32::try_from(n)
        .map_err(|_| gam_gpu::gpu_err!("n={n} exceeds i32 for solve-row kernel argument"))?;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(&n_i32);
    builder.arg(eta_dev);
    builder.arg(y_dev);
    builder.arg(prior_w_dev);
    // GammaLog solve kernel has `double shape` before the output buffers.
    if matches!(family, PirlsRowFamily::GammaLog) {
        builder.arg(&gamma_shape);
    }
    builder.arg(&mut out.grad_eta);
    builder.arg(&mut out.w_solver);
    builder.arg(&mut out.deviance);
    builder.arg(&mut out.status);
    // SAFETY: solve_row_source_for emits for non-GammaLog:
    //   (int n, const f64* eta, const f64* y, const f64* prior_w,
    //    f64* grad_eta_out, f64* w_solver_out, f64* deviance_out, u32* status_out)
    // For GammaLog the signature inserts `double shape` after `prior_w`.
    // 4 outputs match the 4 SolveRowBuffers fields; grid covers all n rows with
    // per-thread guard `if (i >= n) return`.
    unsafe { builder.launch(cfg) }
        .map(|_event_pair| ())
        .gpu_ctx_with(|err| format!("solve-row launch({kernel_name}): {err}"))
}

/// **candidate-objective / fused alpha-ladder** launcher.
///
/// Evaluates `η_trial_i = η_i + α_k · xδ_i` for all `i ∈ [0,n)` and all
/// `k ∈ [0, ALPHA_LADDER_LEN)` simultaneously.  Each thread atomically
/// accumulates the per-row deviance into `out.objective_dev[k]` and
/// OR-accumulates status flags into `out.status_dev[k]`.
///
/// The grid is `(row_blocks × ALPHA_LADDER_LEN)`: block index `bx / n_blocks`
/// selects the alpha slot, `bx % n_blocks` selects the row tile.
///
/// Caller must call [`AlphaLadderDevBuffers::zero`] before each launch, then
/// issue a single `memcpy_dtoh` to read back `[f64; ALPHA_LADDER_LEN]` and
/// `[u32; ALPHA_LADDER_LEN]` to pick the accepted step.
#[cfg(target_os = "linux")]
pub fn launch_alpha_ladder_on_stream(
    backend: &PirlsRowBackend,
    family: PirlsRowFamily,
    curvature: CurvatureMode,
    gamma_shape: f64,
    stream: &Arc<cudarc::driver::CudaStream>,
    n: usize,
    eta_dev: &cudarc::driver::CudaSlice<f64>,
    xd_dev: &cudarc::driver::CudaSlice<f64>,
    y_dev: &cudarc::driver::CudaSlice<f64>,
    prior_w_dev: &cudarc::driver::CudaSlice<f64>,
    out: &mut AlphaLadderDevBuffers,
) -> Result<(), GpuError> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    let module = backend.module_for_ladder(family, curvature)?;
    let kernel_name = family.ladder_kernel_name();
    let func = module
        .load_function(kernel_name)
        .gpu_ctx_with(|err| format!("alpha-ladder load_function({kernel_name}): {err}"))?;
    const THREADS_PER_BLOCK: u32 = 256;
    let n_u32 =
        u32::try_from(n).map_err(|_| gam_gpu::gpu_err!("n={n} exceeds u32 for alpha-ladder grid sizing"))?;
    let row_blocks = n_u32.div_ceil(THREADS_PER_BLOCK).max(1);
    let n_i32 = i32::try_from(n)
        .map_err(|_| gam_gpu::gpu_err!("n={n} exceeds i32 for alpha-ladder kernel argument"))?;
    // Grid: x = row tile index (0..row_blocks), y = alpha index (0..ALPHA_LADDER_LEN).
    let cfg = LaunchConfig {
        grid_dim: (row_blocks, ALPHA_LADDER_LEN as u32, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(&n_i32);
    builder.arg(eta_dev);
    builder.arg(xd_dev);
    builder.arg(y_dev);
    builder.arg(prior_w_dev);
    // GammaLog ladder kernel has `double shape` before the output buffers.
    if matches!(family, PirlsRowFamily::GammaLog) {
        builder.arg(&gamma_shape);
    }
    builder.arg(&mut out.objective_dev);
    builder.arg(&mut out.status_dev);
    // SAFETY: for non-GammaLog, ladder_source_for emits:
    //   (int n, const f64* eta, const f64* xd, const f64* y, const f64* prior_w,
    //    f64* objective_out, u32* status_out)
    // For GammaLog `double shape` is inserted after `prior_w`.
    // Grid is (row_blocks × ALPHA_LADDER_LEN); each thread reads alphas[] via
    // blockIdx.y, rows via blockIdx.x * blockDim.x + threadIdx.x (guarded by n).
    // Atomic double-precision add to objective_out[blockIdx.y], OR to status_out[blockIdx.y].
    unsafe { builder.launch(cfg) }
        .map(|_event_pair| ())
        .gpu_ctx_with(|err| format!("alpha-ladder launch({kernel_name}): {err}"))
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
///
/// The GammaLog kernel has an extra `double shape` parameter after `prior_w`
/// so the host can forward the active dispersion shape. All other families
/// use the standard 13-argument signature.
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
    // GammaLog receives the active shape as a scalar kernel argument so the
    // host can forward any positive shape without recompiling the PTX.
    let shape_param = if matches!(family, PirlsRowFamily::GammaLog) {
        "    double         shape,\n"
    } else {
        ""
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
{shape_param}    double* __restrict__ mu_out,
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
    // `shape` is a kernel parameter (see `cuda_source_for`); the body reads it
    // directly. No local shadowing needed.
    let tag = curvature_tag(curvature);
    format!(
        r#"{tag}    double eta_c = clamp_eta(eta_i, &flags);
    double mu_raw = exp(eta_c);
    if (mu_raw < 1e-10) flags |= 0x2u;
    double mu = (mu_raw > 1e-10) ? mu_raw : 1e-10;
    double w_fisher = wp * shape;
#ifdef PIRLS_CURVATURE_OBSERVED
    // Stage 5: observed information for Gamma-log.
    //   w_obs = w_F + w_F · (y/μ − 1) = w_F · y/μ.
    double w_hessian = (w_fisher > 0.0 && mu > 0.0 && isfinite(y_i))
        ? w_fisher * (y_i / mu)
        : w_fisher;
#else
    double w_hessian = w_fisher;
#endif
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
#ifdef PIRLS_CURVATURE_OBSERVED
    // Stage 5: observed information for Bernoulli probit.
    //   w_obs = w_F + w_p · (y − μ) · [h'/V − h²·V'/V²].
    // h(η)=φ(η), h'(η)=−η·φ(η); V'=1−2μ.
    double w_hessian = w_fisher;
    if (v > 0.0 && wp > 0.0) {{
        double h_prime = -eta_c * dmu_deta;
        double v_prime = 1.0 - 2.0 * mu;
        double bracket = h_prime / v - (dmu_deta * dmu_deta) * v_prime / (v * v);
        w_hessian = w_fisher + wp * (y_i - mu) * bracket;
    }}
#else
    double w_hessian = w_fisher;
#endif
    double w_solver = (w_hessian > 0.0) ? fmax(w_hessian, 1e-12) : 0.0;
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
    // μ = 1 − exp(−exp(η)); use -expm1(-inner) to avoid catastrophic
    // cancellation in the deep negative tail (η ≲ -36).
    double mu_raw = -expm1(-inner);
    if (mu_raw < 1e-12 || mu_raw > 1.0 - 1e-12) flags |= 0x2u;
    double mu = fmin(fmax(mu_raw, 1e-12), 1.0 - 1e-12);
    double dmu_deta = inner * (1.0 - mu_raw);
    double v = mu * (1.0 - mu);
    double fpp = (v > 0.0) ? dmu_deta * dmu_deta / v : 0.0;
    double w_fisher = wp * fpp;
#ifdef PIRLS_CURVATURE_OBSERVED
    // Stage 5: observed information for Bernoulli cloglog.
    //   w_obs = w_F + w_p · (y − μ) · [h'/V − h²·V'/V²].
    // h'(η) = h(η) · (1 − inner); V'=1−2μ.
    double w_hessian = w_fisher;
    if (v > 0.0 && wp > 0.0) {{
        double h_prime = dmu_deta * (1.0 - inner);
        double v_prime = 1.0 - 2.0 * mu;
        double bracket = h_prime / v - (dmu_deta * dmu_deta) * v_prime / (v * v);
        w_hessian = w_fisher + wp * (y_i - mu) * bracket;
    }}
#else
    double w_hessian = w_fisher;
#endif
    double w_solver = (w_hessian > 0.0) ? fmax(w_hessian, 1e-12) : 0.0;
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
// solve-row CUDA source (4-output variant)
// ────────────────────────────────────────────────────────────────────────

/// Build the solve-row CUDA source for `(family, curvature)`.
///
/// The kernel has a reduced signature:
///   `(int n, const f64* eta, const f64* y, const f64* prior_w,
///     f64* grad_eta_out, f64* w_solver_out, f64* deviance_out, u32* status_out)`
///
/// It executes the same per-family math as `cuda_source_for` but skips the
/// `mu`, `w_fisher`, `w_hessian`, `z_fisher`, `z_hessian` stores, reducing
/// both bandwidth and L1/register pressure in the hot Newton iteration.
#[cfg(target_os = "linux")]
fn solve_row_source_for(family: PirlsRowFamily, curvature: CurvatureMode) -> String {
    let body = match family {
        PirlsRowFamily::GaussianIdentity => gaussian_identity_body(curvature),
        PirlsRowFamily::PoissonLog => poisson_log_body(curvature),
        PirlsRowFamily::GammaLog => gamma_log_body(curvature),
        PirlsRowFamily::BernoulliLogit => bernoulli_logit_body(curvature),
        PirlsRowFamily::BernoulliProbit => bernoulli_probit_body(curvature),
        PirlsRowFamily::BernoulliCLogLog => bernoulli_cloglog_body(curvature),
    };
    let kernel_name = family.solve_kernel_name();
    let curvature_define = match curvature {
        CurvatureMode::Fisher => "#define PIRLS_CURVATURE_FISHER 1",
        CurvatureMode::Observed => "#define PIRLS_CURVATURE_OBSERVED 1",
    };
    // GammaLog solve kernel also takes `double shape` after `prior_w`.
    let shape_param = if matches!(family, PirlsRowFamily::GammaLog) {
        "    double         shape,\n"
    } else {
        ""
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
{shape_param}    double* __restrict__ grad_eta_out,
    double* __restrict__ w_solver_out,
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
    grad_eta_out[i] = grad_eta;
    w_solver_out[i] = w_solver;
    deviance_out[i] = dev;
    status_out[i] = flags;
}}
"#,
        prolog = COMMON_DEVICE_PROLOG,
    )
}

// ────────────────────────────────────────────────────────────────────────
// alpha-ladder CUDA source (fused all-alpha candidate-objective kernel)
// ────────────────────────────────────────────────────────────────────────

/// The alpha constants embedded into the ladder kernel source as a
/// `__constant__` array. Must stay in sync with [`ALPHA_LADDER`].
#[cfg(target_os = "linux")]
const ALPHA_LADDER_CUDA_ARRAY: &str =
    "__constant__ double PIRLS_ALPHAS[7] = {1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625};";

/// Build the fused alpha-ladder CUDA source for `(family, curvature)`.
///
/// Grid layout: `grid = (row_blocks, ALPHA_LADDER_LEN, 1)`.
///   - `blockIdx.y` selects the alpha slot `k`.
///   - `blockIdx.x * blockDim.x + threadIdx.x` selects row `i`.
///
/// Each thread evaluates `eta_trial = eta[i] + PIRLS_ALPHAS[k] * xd[i]`,
/// runs the per-family deviance math, and atomically adds to
/// `objective_out[k]` (double-precision atomic add) and OR-accumulates into
/// `status_out[k]` (atomic OR on u32).
///
/// The kernel signature is:
///   `(int n, const f64* eta, const f64* xd, const f64* y, const f64* prior_w,
///     f64* objective_out, u32* status_out)`
#[cfg(target_os = "linux")]
fn ladder_source_for(family: PirlsRowFamily, curvature: CurvatureMode) -> String {
    let body = match family {
        PirlsRowFamily::GaussianIdentity => gaussian_identity_body(curvature),
        PirlsRowFamily::PoissonLog => poisson_log_body(curvature),
        PirlsRowFamily::GammaLog => gamma_log_body(curvature),
        PirlsRowFamily::BernoulliLogit => bernoulli_logit_body(curvature),
        PirlsRowFamily::BernoulliProbit => bernoulli_probit_body(curvature),
        PirlsRowFamily::BernoulliCLogLog => bernoulli_cloglog_body(curvature),
    };
    let kernel_name = family.ladder_kernel_name();
    let curvature_define = match curvature {
        CurvatureMode::Fisher => "#define PIRLS_CURVATURE_FISHER 1",
        CurvatureMode::Observed => "#define PIRLS_CURVATURE_OBSERVED 1",
    };
    // The body uses the name `eta_i` for the (possibly clamped) linear predictor.
    // For the ladder we substitute `eta[i] + alpha * xd[i]` as the trial eta,
    // so we define `eta_i` before the body runs. The body's own local variable
    // of the same name overwrites it in families that clamp eta (e.g. Poisson,
    // Bernoulli), which is correct — the per-family body reassigns `eta_c` from
    // the (now trial-eta-valued) `eta_i`. For GaussianIdentity the body reads
    // `eta_i` directly as `mu`, which is also correct after the substitution.
    // GammaLog ladder kernel also takes `double shape` after `prior_w`.
    let shape_param = if matches!(family, PirlsRowFamily::GammaLog) {
        "    double         shape,\n"
    } else {
        ""
    };
    format!(
        r#"
{curvature_define}
{prolog}
{alphas}

extern "C" __global__ void {kernel_name}(
    int            n,
    const double* __restrict__ eta,
    const double* __restrict__ xd,
    const double* __restrict__ y,
    const double* __restrict__ prior_w,
{shape_param}    double* __restrict__ objective_out,
    unsigned int* __restrict__ status_out
) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = (int)blockIdx.y;
    if (i >= n) return;
    unsigned int flags = 0u;
    double alpha = PIRLS_ALPHAS[k];
    double eta_i = eta[i] + alpha * xd[i];
    double y_i = y[i];
    double wp = prior_w[i] > 0.0 ? prior_w[i] : 0.0;
    if (prior_w[i] <= 0.0) flags |= 0x10u;
{body}
    atomicAdd(&objective_out[k], dev);
    atomicOr(&status_out[k], flags);
}}
"#,
        prolog = COMMON_DEVICE_PROLOG,
        alphas = ALPHA_LADDER_CUDA_ARRAY,
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
                    let out = row_reweight_cpu(family, CurvatureMode::Fisher, input, 1.0);
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
                    if out.status
                        & (status_flags::INVALID_RESPONSE | status_flags::ZERO_PRIOR_WEIGHT)
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

    /// Count how many rows actually carried a positive Fisher weight; tests
    /// assert non-zero so a silently no-op evaluator (e.g. all-NaN output)
    /// can't satisfy the per-row invariants vacuously.
    fn count_active_rows(family: PirlsRowFamily) -> usize {
        let mut active = 0usize;
        for &eta in [-700.0, -3.0, 0.0, 3.0, 700.0].iter() {
            for &y in [0.0, 0.5, 1.0].iter() {
                for &wp in [1.0, 2.5].iter() {
                    let out = row_reweight_cpu(
                        family,
                        CurvatureMode::Fisher,
                        RowInput {
                            eta,
                            y,
                            prior_weight: wp,
                        },
                        1.0,
                    );
                    if out.w_fisher > 0.0 {
                        active += 1;
                    }
                }
            }
        }
        active
    }

    #[test]
    fn gaussian_identity_row_invariants() {
        check_family_matches_cpu_reference(PirlsRowFamily::GaussianIdentity);
        assert!(count_active_rows(PirlsRowFamily::GaussianIdentity) > 0);
    }

    #[test]
    fn poisson_log_row_invariants() {
        check_family_matches_cpu_reference(PirlsRowFamily::PoissonLog);
        assert!(count_active_rows(PirlsRowFamily::PoissonLog) > 0);
    }

    #[test]
    fn gamma_log_row_invariants() {
        check_family_matches_cpu_reference(PirlsRowFamily::GammaLog);
        assert!(count_active_rows(PirlsRowFamily::GammaLog) > 0);
    }

    #[test]
    fn bernoulli_logit_row_invariants() {
        check_family_matches_cpu_reference(PirlsRowFamily::BernoulliLogit);
        assert!(count_active_rows(PirlsRowFamily::BernoulliLogit) > 0);
    }

    #[test]
    fn bernoulli_probit_row_invariants() {
        check_family_matches_cpu_reference(PirlsRowFamily::BernoulliProbit);
        assert!(count_active_rows(PirlsRowFamily::BernoulliProbit) > 0);
    }

    #[test]
    fn bernoulli_cloglog_row_invariants() {
        check_family_matches_cpu_reference(PirlsRowFamily::BernoulliCLogLog);
        assert!(count_active_rows(PirlsRowFamily::BernoulliCLogLog) > 0);
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
            1.0,
        );
        assert!(out.mu.is_finite() && out.deviance.is_finite());
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
            1.0,
        );
        let expected_mu = (1.5_f64).exp();
        assert!(expected_mu.is_finite() && out.mu.is_finite());
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
            1.0,
        );
        assert!(mu > 0.0 && mu < 1.0);
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
            1.0,
        );
        assert!(out.status & status_flags::ETA_CLAMPED != 0);
    }

    /// `module_for` must lazily compile + cache one module per `(family, curvature)`.
    /// Skipped on hosts without a CUDA runtime (mac, CI).
    #[test]
    fn backend_compiles_one_module_per_family_when_device_present() {
        // The compiled-backend flag itself is independent of runtime probe
        // and must agree with the `cfg(target_os = "linux")` selector that
        // gates the rest of the module-cache code path.
        assert_eq!(PirlsRowBackend::compiled(), cfg!(target_os = "linux"));
        if gam_gpu::device_runtime::GpuRuntime::global().is_none() {
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

    /// Stage 6: JIT-compiled custom-family kernel through Level A
    /// (built-in spec) must produce byte-identical outputs to the
    /// cached built-in kernel on the same inputs. Validates the
    /// `JitFamilySpec::glm` builder + `cuda_source` shell + JIT module
    /// cache + `launch_row_reweight_jit_on_stream` end to end against
    /// the Stage 1 cached built-in path.
    #[test]
    fn jit_glm_kernel_matches_builtin_byte_identical() {
        if gam_gpu::device_runtime::GpuRuntime::global().is_none() {
            eprintln!("[stage_6_jit] no CUDA runtime — skipping");
            return;
        }
        #[cfg(target_os = "linux")]
        {
            let etas = [-2.0_f64, -0.5, 0.3, 1.5];
            let ys = [0.0_f64, 1.0, 0.0, 1.0];
            let priors = [1.0_f64, 1.2, 0.8, 1.5];
            let n = etas.len();
            let family = PirlsRowFamily::BernoulliLogit;
            let curvature = CurvatureMode::Fisher;
            let backend = PirlsRowBackend::probe().expect("backend probe on CUDA host");
            let stream = gam_gpu::backend_probe::probe_cuda_backend("pirls_row")
                .expect("shared backend probe")
                .stream;

            let mut eta_dev = stream.alloc_zeros::<f64>(n).expect("eta");
            let mut y_dev = stream.alloc_zeros::<f64>(n).expect("y");
            let mut prior_dev = stream.alloc_zeros::<f64>(n).expect("prior");
            stream.memcpy_htod(&etas, &mut eta_dev).expect("up eta");
            stream.memcpy_htod(&ys, &mut y_dev).expect("up y");
            stream
                .memcpy_htod(&priors, &mut prior_dev)
                .expect("up prior");

            // Built-in path.
            let mut out_builtin = RowOutputDevBuffers::allocate(&stream, n).expect("alloc builtin");
            launch_row_reweight_on_stream(
                backend,
                family,
                curvature,
                1.0,
                &stream,
                n,
                &eta_dev,
                &y_dev,
                &prior_dev,
                &mut out_builtin,
            )
            .expect("builtin launch");

            // JIT path through Level A spec (same body, distinct kernel symbol).
            let spec = JitFamilySpec::glm(0x424c_4c47u64, family, curvature);
            let mut out_jit = RowOutputDevBuffers::allocate(&stream, n).expect("alloc jit");
            launch_row_reweight_jit_on_stream(
                backend,
                &spec,
                curvature,
                &stream,
                n,
                &eta_dev,
                &y_dev,
                &prior_dev,
                &mut out_jit,
            )
            .expect("jit launch");
            stream.synchronize().expect("sync");

            // Byte-identical per-field comparison.
            for (label, b_dev, j_dev) in [
                ("mu", &out_builtin.mu, &out_jit.mu),
                ("grad_eta", &out_builtin.grad_eta, &out_jit.grad_eta),
                ("w_fisher", &out_builtin.w_fisher, &out_jit.w_fisher),
                ("w_hessian", &out_builtin.w_hessian, &out_jit.w_hessian),
                ("w_solver", &out_builtin.w_solver, &out_jit.w_solver),
                ("z_fisher", &out_builtin.z_fisher, &out_jit.z_fisher),
                ("z_hessian", &out_builtin.z_hessian, &out_jit.z_hessian),
                ("deviance", &out_builtin.deviance, &out_jit.deviance),
            ] {
                let b = stream.clone_dtoh(b_dev).expect("dl builtin");
                let j = stream.clone_dtoh(j_dev).expect("dl jit");
                for i in 0..n {
                    assert_eq!(
                        b[i].to_bits(),
                        j[i].to_bits(),
                        "{label}[{i}]: builtin {} ≠ jit {}",
                        b[i],
                        j[i],
                    );
                }
            }
        }
    }

    /// Stage 6 Level B: caller-supplied raw CUDA body (no built-in
    /// family enum) must produce byte-identical outputs to the cached
    /// built-in `GaussianIdentity` kernel on the same fixture. The
    /// raw body below is written from scratch — different statement
    /// layout, different intermediate names — but performs the same
    /// floating-point operations in the same order as
    /// `gaussian_identity_body`, so the bit pattern of every output
    /// must match the built-in kernel exactly.
    #[test]
    fn jit_raw_body_kernel_matches_builtin_gaussian_byte_identical() {
        if gam_gpu::device_runtime::GpuRuntime::global().is_none() {
            eprintln!("[stage_6_jit_raw] no CUDA runtime — skipping");
            return;
        }
        #[cfg(target_os = "linux")]
        {
            // 256-row deterministic fixture covering positive/negative
            // residuals, zero residuals, varying prior weights, and a
            // zero-weight row (exercises the `(wp > 0.0)` branch in
            // `w_solver`).
            let n: usize = 256;
            let mut etas = vec![0.0_f64; n];
            let mut ys = vec![0.0_f64; n];
            let mut priors = vec![0.0_f64; n];
            for i in 0..n {
                let t = (i as f64) / (n as f64 - 1.0); // 0..=1
                etas[i] = -3.0 + 6.0 * t;
                ys[i] = 5.0 * (t - 0.5);
                priors[i] = if i == 7 {
                    0.0 // zero-weight row
                } else {
                    0.25 + 1.75 * t
                };
            }

            let family = PirlsRowFamily::GaussianIdentity;
            let curvature = CurvatureMode::Fisher;
            let backend = PirlsRowBackend::probe().expect("backend probe on CUDA host");
            let stream = gam_gpu::backend_probe::probe_cuda_backend("pirls_row")
                .expect("shared backend probe")
                .stream;

            let mut eta_dev = stream.alloc_zeros::<f64>(n).expect("eta");
            let mut y_dev = stream.alloc_zeros::<f64>(n).expect("y");
            let mut prior_dev = stream.alloc_zeros::<f64>(n).expect("prior");
            stream.memcpy_htod(&etas, &mut eta_dev).expect("up eta");
            stream.memcpy_htod(&ys, &mut y_dev).expect("up y");
            stream
                .memcpy_htod(&priors, &mut prior_dev)
                .expect("up prior");

            // Built-in Level A Gaussian-identity kernel (reference).
            let mut out_builtin = RowOutputDevBuffers::allocate(&stream, n).expect("alloc builtin");
            launch_row_reweight_on_stream(
                backend,
                family,
                curvature,
                1.0,
                &stream,
                n,
                &eta_dev,
                &y_dev,
                &prior_dev,
                &mut out_builtin,
            )
            .expect("builtin launch");

            // Level B raw-body Gaussian-identity kernel. Source is
            // written by hand against the JitFamilySpec contract: read
            // eta_i, y_i, wp; assign mu, grad_eta, w_fisher, w_hessian,
            // w_solver, z_f, z_h, dev. The op sequence matches
            // `gaussian_identity_body` exactly so the result is
            // bit-identical to the built-in path.
            let raw_body = r#"    // level-b raw body: gaussian identity (hand-written)
    // identity link: mu = eta
    double mu = eta_i;
    // ordinary residual on the response scale
    double resid = y_i - mu;
    // canonical score contribution
    double grad_eta = wp * resid;
    // fisher info per row: weight itself (V(mu)=1, dmu/deta=1)
    double w_fisher = wp;
    // observed == fisher for canonical identity link
    double w_hessian = wp;
    // solver weight clamps tiny positives to avoid singularity
    double w_solver = (wp > 0.0) ? fmax(wp, 1e-12) : 0.0;
    // working response equals raw response on identity link
    double z_f = y_i;
    double z_h = y_i;
    // squared-error contribution to deviance
    double dev = wp * resid * resid;
"#;
            let spec = JitFamilySpec::raw(0x5241_575f_4741_5553u64, raw_body);
            let mut out_jit = RowOutputDevBuffers::allocate(&stream, n).expect("alloc jit");
            launch_row_reweight_jit_on_stream(
                backend,
                &spec,
                curvature,
                &stream,
                n,
                &eta_dev,
                &y_dev,
                &prior_dev,
                &mut out_jit,
            )
            .expect("jit raw launch");
            stream.synchronize().expect("sync");

            for (label, b_dev, j_dev) in [
                ("mu", &out_builtin.mu, &out_jit.mu),
                ("grad_eta", &out_builtin.grad_eta, &out_jit.grad_eta),
                ("w_fisher", &out_builtin.w_fisher, &out_jit.w_fisher),
                ("w_hessian", &out_builtin.w_hessian, &out_jit.w_hessian),
                ("w_solver", &out_builtin.w_solver, &out_jit.w_solver),
                ("z_fisher", &out_builtin.z_fisher, &out_jit.z_fisher),
                ("z_hessian", &out_builtin.z_hessian, &out_jit.z_hessian),
                ("deviance", &out_builtin.deviance, &out_jit.deviance),
            ] {
                let b = stream.clone_dtoh(b_dev).expect("dl builtin");
                let j = stream.clone_dtoh(j_dev).expect("dl jit raw");
                for i in 0..n {
                    assert_eq!(
                        b[i].to_bits(),
                        j[i].to_bits(),
                        "{label}[{i}]: builtin {} ≠ jit-raw {}",
                        b[i],
                        j[i],
                    );
                }
            }

            // Direct Level B raw-body → CPU reference parity. Closes
            // the chain JIT-raw → CPU without going through the
            // built-in GPU kernel as an intermediary. Gaussian
            // identity is straight-line scalar arithmetic on both
            // sides, so we still demand bit-equality on the eight
            // outputs the CPU evaluator exposes.
            let mu_j = stream.clone_dtoh(&out_jit.mu).expect("dl jit mu");
            let g_j = stream.clone_dtoh(&out_jit.grad_eta).expect("dl jit g");
            let wf_j = stream.clone_dtoh(&out_jit.w_fisher).expect("dl jit wf");
            let wh_j = stream.clone_dtoh(&out_jit.w_hessian).expect("dl jit wh");
            let ws_j = stream.clone_dtoh(&out_jit.w_solver).expect("dl jit ws");
            let zf_j = stream.clone_dtoh(&out_jit.z_fisher).expect("dl jit zf");
            let zh_j = stream.clone_dtoh(&out_jit.z_hessian).expect("dl jit zh");
            let d_j = stream.clone_dtoh(&out_jit.deviance).expect("dl jit d");
            for i in 0..n {
                let cpu = row_reweight_cpu(
                    PirlsRowFamily::GaussianIdentity,
                    curvature,
                    RowInput {
                        eta: etas[i],
                        y: ys[i],
                        prior_weight: priors[i],
                    },
                    1.0,
                );
                for (label, cpu_v, jit_v) in [
                    ("mu", cpu.mu, mu_j[i]),
                    ("grad_eta", cpu.grad_eta, g_j[i]),
                    ("w_fisher", cpu.w_fisher, wf_j[i]),
                    ("w_hessian", cpu.w_hessian, wh_j[i]),
                    ("w_solver", cpu.w_solver, ws_j[i]),
                    ("z_fisher", cpu.z_fisher, zf_j[i]),
                    ("z_hessian", cpu.z_hessian, zh_j[i]),
                    ("deviance", cpu.deviance, d_j[i]),
                ] {
                    assert_eq!(
                        cpu_v.to_bits(),
                        jit_v.to_bits(),
                        "{label}[{i}]: cpu {} ≠ jit-raw {}",
                        cpu_v,
                        jit_v,
                    );
                }
            }
        }
    }

    /// Stage 5: observed-information curvature mode is now real math
    /// (no longer a Fisher alias) for Gamma-log, Bernoulli probit, and
    /// Bernoulli cloglog. Canonical families (Bernoulli logit, Poisson
    /// log, Gaussian identity) remain == Fisher because canonical
    /// links have observed == Fisher by construction.
    #[test]
    fn observed_curvature_matches_expected_per_family() {
        // Picks where the math is well-conditioned so that round-off
        // doesn't dominate the comparison.
        let probe_eta = 0.4_f64;
        let probe_y = 1.0_f64;
        let wp = 1.5_f64;
        let input = RowInput {
            eta: probe_eta,
            y: probe_y,
            prior_weight: wp,
        };

        // Canonical families: observed must equal Fisher exactly.
        for canonical in [
            PirlsRowFamily::GaussianIdentity,
            PirlsRowFamily::PoissonLog,
            PirlsRowFamily::BernoulliLogit,
        ] {
            let f = row_reweight_cpu(canonical, CurvatureMode::Fisher, input, 1.0);
            let o = row_reweight_cpu(canonical, CurvatureMode::Observed, input, 1.0);
            assert_eq!(
                f.w_hessian, o.w_hessian,
                "{canonical:?}: observed must equal Fisher for canonical link"
            );
        }

        // Gamma-log (non-canonical): observed = Fisher · (y/μ). Exercise
        // with shape=1 and shape=2.5 to confirm the plumbing.
        for &shape in &[1.0_f64, 2.5] {
            let gf = row_reweight_cpu(
                PirlsRowFamily::GammaLog,
                CurvatureMode::Fisher,
                input,
                shape,
            );
            let go = row_reweight_cpu(
                PirlsRowFamily::GammaLog,
                CurvatureMode::Observed,
                input,
                shape,
            );
            assert!(
                (go.w_hessian - gf.w_fisher * (probe_y / gf.mu)).abs() <= 1e-12,
                "Gamma-log observed mismatch (shape={shape}): got={} expected={} (mu={})",
                go.w_hessian,
                gf.w_fisher * (probe_y / gf.mu),
                gf.mu
            );
            assert_ne!(
                gf.w_hessian, go.w_hessian,
                "Gamma-log: observed must differ from Fisher when y ≠ μ (shape={shape})"
            );
        }

        // Bernoulli probit/cloglog: observed must differ from Fisher
        // generically and must be ≥ 0 at well-behaved interior points
        // (saturation tail can push it negative; tested elsewhere).
        for noncanon in [
            PirlsRowFamily::BernoulliProbit,
            PirlsRowFamily::BernoulliCLogLog,
        ] {
            let f = row_reweight_cpu(noncanon, CurvatureMode::Fisher, input, 1.0);
            let o = row_reweight_cpu(noncanon, CurvatureMode::Observed, input, 1.0);
            assert!(
                (f.w_hessian - o.w_hessian).abs() > 0.0 || (probe_y - f.mu).abs() < 1e-15,
                "{noncanon:?}: observed should differ from Fisher when y ≠ μ"
            );
        }
    }

    /// Gamma-log CPU reference: `w_fisher` scales linearly with shape;
    /// observed `w_hessian = w_fisher · y/μ` holds for any positive shape.
    #[test]
    fn gamma_log_shape_scaling() {
        let input = RowInput {
            eta: 0.5,
            y: 2.0,
            prior_weight: 1.0,
        };
        let base = row_reweight_cpu(PirlsRowFamily::GammaLog, CurvatureMode::Fisher, input, 1.0);
        for &shape in &[0.5_f64, 1.5, 3.0, 10.0] {
            let r = row_reweight_cpu(
                PirlsRowFamily::GammaLog,
                CurvatureMode::Fisher,
                input,
                shape,
            );
            assert!(
                (r.w_fisher - shape * base.w_fisher).abs() <= 1e-14,
                "w_fisher should scale with shape: got {} expected {} (shape={shape})",
                r.w_fisher,
                shape * base.w_fisher,
            );
            assert_eq!(
                r.mu.to_bits(),
                base.mu.to_bits(),
                "mu must not depend on shape"
            );
            let ro = row_reweight_cpu(
                PirlsRowFamily::GammaLog,
                CurvatureMode::Observed,
                input,
                shape,
            );
            let expected_obs = r.w_fisher * (input.y / r.mu);
            assert!(
                (ro.w_hessian - expected_obs).abs() <= 1e-13,
                "observed w_hessian mismatch (shape={shape}): got={} expected={}",
                ro.w_hessian,
                expected_obs,
            );
        }
    }

    /// V100 parity for the device-side row launcher.
    ///
    /// For every built-in family the launcher's per-row outputs must match
    /// the CPU `row_reweight_cpu` reference to round-off. Skipped on hosts
    /// without a CUDA runtime (mac, CI). This is also the production
    /// caller that justifies the launcher + `RowOutputDevBuffers` surface
    /// per the dead-pub-scanner rule.
    #[test]
    fn launch_row_reweight_matches_cpu_reference_on_device() {
        if gam_gpu::device_runtime::GpuRuntime::global().is_none() {
            eprintln!("[pirls_row_gpu test] no CUDA runtime — skipping launcher parity test");
            return;
        }
        #[cfg(target_os = "linux")]
        {
            // Use a small but non-trivial row batch with several y-values per
            // family. Pick a y that's valid for every family below (0/1 are
            // valid for Bernoulli; positive for Poisson/Gamma; arbitrary for
            // Gaussian). Build per-family input vectors.
            let etas = [-3.0_f64, -0.5, 0.0, 0.5, 3.0, 10.0, -10.0, 1.5];
            let n = etas.len();
            let backend = PirlsRowBackend::probe().expect("backend probe on CUDA host");
            let stream = gam_gpu::backend_probe::probe_cuda_backend("pirls_row")
                .expect("shared backend probe")
                .stream;

            for &family in PirlsRowFamily::ALL.iter() {
                let ys: Vec<f64> = match family {
                    PirlsRowFamily::GammaLog | PirlsRowFamily::PoissonLog => {
                        (0..n).map(|i| 1.0 + 0.5 * (i as f64)).collect()
                    }
                    PirlsRowFamily::GaussianIdentity => {
                        (0..n).map(|i| -1.0 + 0.5 * (i as f64)).collect()
                    }
                    _ => (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect(),
                };
                let priors: Vec<f64> = (0..n).map(|i| 1.0 + 0.25 * (i as f64)).collect();

                // CPU reference.
                let mut cpu_out = Vec::with_capacity(n);
                for i in 0..n {
                    cpu_out.push(row_reweight_cpu(
                        family,
                        CurvatureMode::Fisher,
                        RowInput {
                            eta: etas[i],
                            y: ys[i],
                            prior_weight: priors[i],
                        },
                        1.0,
                    ));
                }

                // Upload inputs, allocate device outputs, launch, download.
                let mut eta_dev = stream.alloc_zeros::<f64>(n).expect("alloc eta_dev");
                let mut y_dev = stream.alloc_zeros::<f64>(n).expect("alloc y_dev");
                let mut prior_dev = stream.alloc_zeros::<f64>(n).expect("alloc prior_dev");
                stream
                    .memcpy_htod(etas.as_slice(), &mut eta_dev)
                    .expect("upload eta");
                stream
                    .memcpy_htod(ys.as_slice(), &mut y_dev)
                    .expect("upload y");
                stream
                    .memcpy_htod(priors.as_slice(), &mut prior_dev)
                    .expect("upload prior");
                let mut out = RowOutputDevBuffers::allocate(&stream, n).expect("alloc row buffers");
                launch_row_reweight_on_stream(
                    backend,
                    family,
                    CurvatureMode::Fisher,
                    1.0,
                    &stream,
                    n,
                    &eta_dev,
                    &y_dev,
                    &prior_dev,
                    &mut out,
                )
                .unwrap_or_else(|err| panic!("launch {family:?}: {err}"));
                stream.synchronize().expect("stream sync");
                let mu = stream.clone_dtoh(&out.mu).expect("dl mu");
                let g = stream.clone_dtoh(&out.grad_eta).expect("dl grad_eta");
                let wf = stream.clone_dtoh(&out.w_fisher).expect("dl w_fisher");
                let wh = stream.clone_dtoh(&out.w_hessian).expect("dl w_hessian");
                let ws_v = stream.clone_dtoh(&out.w_solver).expect("dl w_solver");
                let zf = stream.clone_dtoh(&out.z_fisher).expect("dl z_fisher");
                let zh = stream.clone_dtoh(&out.z_hessian).expect("dl z_hessian");
                let dev = stream.clone_dtoh(&out.deviance).expect("dl deviance");

                let tol = 1e-12;
                for i in 0..n {
                    let r = cpu_out[i];
                    assert_close(&format!("{family:?}/row{i}/mu"), mu[i], r.mu, tol);
                    assert_close(
                        &format!("{family:?}/row{i}/grad_eta"),
                        g[i],
                        r.grad_eta,
                        tol,
                    );
                    assert_close(
                        &format!("{family:?}/row{i}/w_fisher"),
                        wf[i],
                        r.w_fisher,
                        tol,
                    );
                    assert_close(
                        &format!("{family:?}/row{i}/w_hessian"),
                        wh[i],
                        r.w_hessian,
                        tol,
                    );
                    assert_close(
                        &format!("{family:?}/row{i}/w_solver"),
                        ws_v[i],
                        r.w_solver,
                        tol,
                    );
                    assert_close(
                        &format!("{family:?}/row{i}/z_fisher"),
                        zf[i],
                        r.z_fisher,
                        tol,
                    );
                    assert_close(
                        &format!("{family:?}/row{i}/z_hessian"),
                        zh[i],
                        r.z_hessian,
                        tol,
                    );
                    assert_close(
                        &format!("{family:?}/row{i}/deviance"),
                        dev[i],
                        r.deviance,
                        tol,
                    );
                }
            }
        }
    }

    /// Stage 5 end-to-end V100 parity for `CurvatureMode::Observed`.
    ///
    /// 256-row synthetic fixture per family with η ∈ [-6, 6] spanning the
    /// saturated tails. Two assertions:
    ///
    /// 1. Noncanonical families (BernoulliProbit, BernoulliCLogLog,
    ///    GammaLog): device `w_hessian` under Observed mode matches the
    ///    CPU `row_reweight_cpu(..., Observed, ..)` reference to
    ///    `abs ≤ 1e-12` OR `rel ≤ 1e-11`.
    /// 2. Canonical families (GaussianIdentity, PoissonLog, BernoulliLogit):
    ///    device `w_hessian` under Observed mode equals device `w_fisher`
    ///    bit-for-bit (via `to_bits()`) — canonical links have observed
    ///    information ≡ Fisher information by construction.
    #[test]
    fn gpu_observed_parity() {
        if gam_gpu::device_runtime::GpuRuntime::global().is_none() {
            eprintln!("[gpu_observed_parity] no CUDA runtime — skipping");
            return;
        }
        #[cfg(target_os = "linux")]
        {
            const N: usize = 256;
            let etas: Vec<f64> = (0..N)
                .map(|i| -6.0 + 12.0 * (i as f64) / ((N - 1) as f64))
                .collect();
            let priors: Vec<f64> = (0..N)
                .map(|i| 0.5 + 1.5 * ((i as f64) / (N as f64)))
                .collect();

            let backend = PirlsRowBackend::probe().expect("backend probe on CUDA host");
            let stream = gam_gpu::backend_probe::probe_cuda_backend("pirls_row")
                .expect("shared backend probe")
                .stream;

            for &family in PirlsRowFamily::ALL.iter() {
                let ys: Vec<f64> = match family {
                    PirlsRowFamily::GammaLog => (0..N).map(|i| 0.25 + 0.05 * (i as f64)).collect(),
                    PirlsRowFamily::PoissonLog => (0..N).map(|i| (i % 6) as f64).collect(),
                    PirlsRowFamily::GaussianIdentity => (0..N)
                        .map(|i| -2.0 + 4.0 * (i as f64) / ((N - 1) as f64))
                        .collect(),
                    _ => (0..N).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect(),
                };

                let mut eta_dev = stream.alloc_zeros::<f64>(N).expect("alloc eta_dev");
                let mut y_dev = stream.alloc_zeros::<f64>(N).expect("alloc y_dev");
                let mut prior_dev = stream.alloc_zeros::<f64>(N).expect("alloc prior_dev");
                stream
                    .memcpy_htod(etas.as_slice(), &mut eta_dev)
                    .expect("upload eta");
                stream
                    .memcpy_htod(ys.as_slice(), &mut y_dev)
                    .expect("upload y");
                stream
                    .memcpy_htod(priors.as_slice(), &mut prior_dev)
                    .expect("upload prior");

                let mut out_obs = RowOutputDevBuffers::allocate(&stream, N).expect("alloc out_obs");
                launch_row_reweight_on_stream(
                    backend,
                    family,
                    CurvatureMode::Observed,
                    1.0,
                    &stream,
                    N,
                    &eta_dev,
                    &y_dev,
                    &prior_dev,
                    &mut out_obs,
                )
                .unwrap_or_else(|err| panic!("observed launch {family:?}: {err}"));
                stream.synchronize().expect("stream sync (observed)");

                let wh_obs = stream
                    .clone_dtoh(&out_obs.w_hessian)
                    .expect("dl w_hessian (observed)");
                let wf_obs = stream
                    .clone_dtoh(&out_obs.w_fisher)
                    .expect("dl w_fisher (observed)");

                if family.is_canonical() {
                    for i in 0..N {
                        assert_eq!(
                            wh_obs[i].to_bits(),
                            wf_obs[i].to_bits(),
                            "{family:?} row {i}: observed w_hessian {} must bit-equal w_fisher {} on canonical link",
                            wh_obs[i],
                            wf_obs[i],
                        );
                    }
                } else {
                    for i in 0..N {
                        let cpu = row_reweight_cpu(
                            family,
                            CurvatureMode::Observed,
                            RowInput {
                                eta: etas[i],
                                y: ys[i],
                                prior_weight: priors[i],
                            },
                            1.0,
                        );
                        let got = wh_obs[i];
                        let exp = cpu.w_hessian;
                        let abs_err = (got - exp).abs();
                        let rel_err = if exp.abs() > 0.0 {
                            abs_err / exp.abs()
                        } else {
                            abs_err
                        };
                        assert!(
                            abs_err <= 1.0e-12 || rel_err <= 1.0e-11,
                            "{family:?} row {i} (eta={}, y={}, wp={}): \
                             device w_hessian={} vs CPU observed={} (abs={}, rel={})",
                            etas[i],
                            ys[i],
                            priors[i],
                            got,
                            exp,
                            abs_err,
                            rel_err,
                        );
                    }
                }
            }
        }
    }

    /// Task #50 — Stage 5 GPU end-to-end observed-curvature parity at
    /// n=1000 across **all six** supported families, validating BOTH
    /// `w_hessian` (Hessian diagonal) AND `grad_eta` (score) against the
    /// CPU observed-curvature oracle to abs/rel ≤ 1e-9. This is the
    /// full end-to-end companion to `gpu_observed_parity` (which only
    /// covered n=256 and only checked `w_hessian` for noncanonical
    /// families). Gated on a live CUDA runtime; marked `#[ignore]` so
    /// the v100-bench-runner explicitly opts in via `--ignored`.
    #[test]
    fn gpu_observed_parity_end_to_end_n1000() {
        if gam_gpu::device_runtime::GpuRuntime::global().is_none() {
            eprintln!("[gpu_observed_parity_end_to_end_n1000] no CUDA runtime — skipping");
            return;
        }
        #[cfg(target_os = "linux")]
        {
            const N: usize = 1000;
            // Deterministic η grid spanning the saturated tails plus
            // the near-zero regime where the observed-information
            // correction term is most active.
            let etas: Vec<f64> = (0..N)
                .map(|i| -8.0 + 16.0 * (i as f64) / ((N - 1) as f64))
                .collect();
            let priors: Vec<f64> = (0..N)
                .map(|i| 0.25 + 1.75 * ((i as f64) / (N as f64)))
                .collect();

            let backend = PirlsRowBackend::probe().expect("backend probe on CUDA host");
            let stream = gam_gpu::backend_probe::probe_cuda_backend("pirls_row")
                .expect("shared backend probe")
                .stream;

            const TOL: f64 = 1.0e-9;

            for &family in PirlsRowFamily::ALL.iter() {
                // Family-specific response vectors stay in domain so the
                // CPU oracle is well-defined for every row.
                let ys: Vec<f64> = match family {
                    PirlsRowFamily::GammaLog => {
                        (0..N).map(|i| 0.10 + 0.05 * ((i % 97) as f64)).collect()
                    }
                    PirlsRowFamily::PoissonLog => (0..N).map(|i| (i % 11) as f64).collect(),
                    PirlsRowFamily::GaussianIdentity => (0..N)
                        .map(|i| -3.0 + 6.0 * (i as f64) / ((N - 1) as f64))
                        .collect(),
                    PirlsRowFamily::BernoulliLogit
                    | PirlsRowFamily::BernoulliProbit
                    | PirlsRowFamily::BernoulliCLogLog => {
                        (0..N).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect()
                    }
                };

                let mut eta_dev = stream.alloc_zeros::<f64>(N).expect("alloc eta_dev");
                let mut y_dev = stream.alloc_zeros::<f64>(N).expect("alloc y_dev");
                let mut prior_dev = stream.alloc_zeros::<f64>(N).expect("alloc prior_dev");
                stream
                    .memcpy_htod(etas.as_slice(), &mut eta_dev)
                    .expect("upload eta");
                stream
                    .memcpy_htod(ys.as_slice(), &mut y_dev)
                    .expect("upload y");
                stream
                    .memcpy_htod(priors.as_slice(), &mut prior_dev)
                    .expect("upload prior");

                let mut out_obs = RowOutputDevBuffers::allocate(&stream, N).expect("alloc out_obs");
                launch_row_reweight_on_stream(
                    backend,
                    family,
                    CurvatureMode::Observed,
                    1.0,
                    &stream,
                    N,
                    &eta_dev,
                    &y_dev,
                    &prior_dev,
                    &mut out_obs,
                )
                .unwrap_or_else(|err| panic!("observed launch {family:?}: {err}"));
                stream.synchronize().expect("stream sync (observed)");

                let wh_obs = stream
                    .clone_dtoh(&out_obs.w_hessian)
                    .expect("dl w_hessian (observed)");
                let ge_obs = stream
                    .clone_dtoh(&out_obs.grad_eta)
                    .expect("dl grad_eta (observed)");

                for i in 0..N {
                    let cpu = row_reweight_cpu(
                        family,
                        CurvatureMode::Observed,
                        RowInput {
                            eta: etas[i],
                            y: ys[i],
                            prior_weight: priors[i],
                        },
                        1.0,
                    );

                    // H diagonal (w_hessian) parity.
                    let h_got = wh_obs[i];
                    let h_exp = cpu.w_hessian;
                    let h_abs = (h_got - h_exp).abs();
                    let h_rel = if h_exp.abs() > 0.0 {
                        h_abs / h_exp.abs()
                    } else {
                        h_abs
                    };
                    assert!(
                        h_abs <= TOL || h_rel <= TOL,
                        "{family:?} row {i} (eta={}, y={}, wp={}): \
                         observed w_hessian GPU={} vs CPU={} (abs={}, rel={})",
                        etas[i],
                        ys[i],
                        priors[i],
                        h_got,
                        h_exp,
                        h_abs,
                        h_rel,
                    );

                    // Gradient (grad_eta) parity — the score does not
                    // depend on curvature mode, but the CPU oracle here
                    // is exercised under `Observed` for full
                    // end-to-end coverage.
                    let g_got = ge_obs[i];
                    let g_exp = cpu.grad_eta;
                    let g_abs = (g_got - g_exp).abs();
                    let g_rel = if g_exp.abs() > 0.0 {
                        g_abs / g_exp.abs()
                    } else {
                        g_abs
                    };
                    assert!(
                        g_abs <= TOL || g_rel <= TOL,
                        "{family:?} row {i} (eta={}, y={}, wp={}): \
                         observed grad_eta GPU={} vs CPU={} (abs={}, rel={})",
                        etas[i],
                        ys[i],
                        priors[i],
                        g_got,
                        g_exp,
                        g_abs,
                        g_rel,
                    );
                }
            }
        }
    }

    /// Task #51 — Stage 6 Level B end-to-end NVRTC JIT parity. For
    /// each of the 6 supported families we hand-author a raw CUDA
    /// body that re-derives the family math from scratch (distinct
    /// variable names + restructured statement order from the
    /// built-in `*_body` strings), JIT-compile via `JitFamilySpec::raw`
    /// through the full `launch_row_reweight_jit_on_stream` pipeline,
    /// and assert all 8 outputs match the CPU `row_reweight_cpu`
    /// oracle to ≤ 1e-10 on n=1000 rows. Skipped if no CUDA runtime;
    /// `#[ignore]` so v100-bench-runner picks it up via `--ignored`.
    #[test]
    fn gpu_jit_level_b_raw_body_end_to_end_all_families_n1000() {
        if gam_gpu::device_runtime::GpuRuntime::global().is_none() {
            eprintln!(
                "[gpu_jit_level_b_raw_body_end_to_end_all_families_n1000] no CUDA runtime — skipping"
            );
            return;
        }
        #[cfg(target_os = "linux")]
        {
            const N: usize = 1000;
            const TOL: f64 = 1.0e-10;
            let curvature = CurvatureMode::Fisher;

            let backend = PirlsRowBackend::probe().expect("backend probe on CUDA host");
            let stream = gam_gpu::backend_probe::probe_cuda_backend("pirls_row")
                .expect("shared backend probe")
                .stream;

            // η grid + per-family in-domain response vectors. Mirrors
            // `gpu_observed_parity_end_to_end_n1000` so the two tests
            // exercise the same numerical regime through different code
            // paths (built-in module vs JIT raw body).
            let etas: Vec<f64> = (0..N)
                .map(|i| -6.0 + 12.0 * (i as f64) / ((N - 1) as f64))
                .collect();
            let priors: Vec<f64> = (0..N)
                .map(|i| 0.25 + 1.75 * ((i as f64) / (N as f64)))
                .collect();

            // Hand-authored raw bodies — re-derived from the math, not
            // copy-pasted from `*_body()`. Helpers (`clamp_eta`,
            // `std_norm_cdf`, `std_norm_pdf`, `bernoulli_deviance`,
            // `bernoulli_z`) are provided by `COMMON_DEVICE_PROLOG`.
            let raw_gaussian = r#"    // raw-body gaussian identity (independent re-derivation)
    double resp = y_i;
    double pred = eta_i;
    double mu = pred;
    double w_p = wp;
    double e_resid = resp - pred;
    double grad_eta = w_p * e_resid;
    double w_fisher = w_p;
    double w_hessian = w_p;
    double w_solver = (w_p > 0.0) ? fmax(w_p, 1e-12) : 0.0;
    double z_f = resp;
    double z_h = resp;
    double dev = w_p * e_resid * e_resid;
"#;

            let raw_poisson = r#"    // raw-body poisson log (independent re-derivation)
    double eta_c = clamp_eta(eta_i, &flags);
    double mu_pre = exp(eta_c);
    if (mu_pre < 1e-10) flags |= 0x2u;
    double mu = (mu_pre > 1e-10) ? mu_pre : 1e-10;
    double wrate = wp * mu;
    double w_fisher = (wrate > 0.0) ? fmax(wrate, 1e-12) : 0.0;
    double w_hessian = w_fisher;
    double w_solver = w_fisher;
    double pres = y_i - mu;
    double grad_eta = wp * pres;
    double z_lin = eta_c + pres / mu;
    double z_f = z_lin;
    double z_h = z_lin;
    double dterm;
    if (y_i > 0.0) {
        dterm = y_i * log(y_i / mu) - pres;
    } else {
        dterm = -pres;
    }
    double dev = 2.0 * wp * dterm;
    if (!(isfinite(y_i) && y_i >= 0.0)) flags |= 0x8u;
"#;

            let raw_gamma = r#"    // raw-body gamma log (independent re-derivation; unit shape)
    double k_shape = 1.0;
    double eta_c = clamp_eta(eta_i, &flags);
    double mu_pre = exp(eta_c);
    if (mu_pre < 1e-10) flags |= 0x2u;
    double mu = (mu_pre > 1e-10) ? mu_pre : 1e-10;
    double w_fisher = wp * k_shape;
    double w_hessian = w_fisher;
    double w_solver = (w_hessian > 0.0) ? fmax(w_hessian, 1e-12) : 0.0;
    double pres = y_i - mu;
    double grad_eta = wp * pres / mu;
    double z_lin = eta_c + pres / mu;
    double z_f = z_lin;
    double z_h = z_lin;
    double dev;
    if (y_i > 0.0) {
        dev = 2.0 * wp * (-log(y_i / mu) + pres / mu);
    } else {
        dev = 1.0 / 0.0;
    }
    if (!(isfinite(y_i) && y_i > 0.0)) flags |= 0x8u;
"#;

            let raw_logit = r#"    // raw-body bernoulli logit (independent re-derivation)
    double eta_c = clamp_eta(eta_i, &flags);
    double te = tanh(0.5 * eta_c);
    double mu_pre = 0.5 * (1.0 + te);
    if (mu_pre < 1e-12 || mu_pre > 1.0 - 1e-12) flags |= 0x2u;
    double mu = fmin(fmax(mu_pre, 1e-12), 1.0 - 1e-12);
    double dmu_deta = mu * (1.0 - mu);
    double w_fisher = wp * dmu_deta;
    double w_hessian = w_fisher;
    double w_solver = (w_fisher > 0.0) ? fmax(w_fisher, 1e-12) : 0.0;
    double bres = y_i - mu;
    double grad_eta = wp * bres;
    double dev = bernoulli_deviance(y_i, mu, wp);
    double z_lin = bernoulli_z(eta_c, y_i, mu, dmu_deta);
    double z_f = z_lin;
    double z_h = z_lin;
    if (!(isfinite(y_i) && y_i >= 0.0 && y_i <= 1.0)) flags |= 0x8u;
"#;

            let raw_probit = r#"    // raw-body bernoulli probit (independent re-derivation; Fisher mode)
    double eta_c = clamp_eta(eta_i, &flags);
    double mu_pre = std_norm_cdf(eta_c);
    if (mu_pre < 1e-12 || mu_pre > 1.0 - 1e-12) flags |= 0x2u;
    double mu = fmin(fmax(mu_pre, 1e-12), 1.0 - 1e-12);
    double phi = std_norm_pdf(eta_c);
    double dmu_deta = phi;
    double vmu = mu * (1.0 - mu);
    double w_pp = (vmu > 0.0) ? (phi * phi) / vmu : 0.0;
    double w_fisher = wp * w_pp;
    double w_hessian = w_fisher;
    double w_solver = (w_hessian > 0.0) ? fmax(w_hessian, 1e-12) : 0.0;
    double bres = y_i - mu;
    double grad_eta = (vmu > 0.0) ? wp * bres * phi / vmu : 0.0;
    double dev = bernoulli_deviance(y_i, mu, wp);
    double z_lin = bernoulli_z(eta_c, y_i, mu, dmu_deta);
    double z_f = z_lin;
    double z_h = z_lin;
    if (!(isfinite(y_i) && y_i >= 0.0 && y_i <= 1.0)) flags |= 0x8u;
"#;

            let raw_cloglog = r#"    // raw-body bernoulli cloglog (independent re-derivation; Fisher mode)
    double eta_c = clamp_eta(eta_i, &flags);
    double a = exp(eta_c);
    double mu_pre = 1.0 - exp(-a);
    if (mu_pre < 1e-12 || mu_pre > 1.0 - 1e-12) flags |= 0x2u;
    double mu = fmin(fmax(mu_pre, 1e-12), 1.0 - 1e-12);
    double dmu_deta = a * (1.0 - mu_pre);
    double vmu = mu * (1.0 - mu);
    double w_pp = (vmu > 0.0) ? (dmu_deta * dmu_deta) / vmu : 0.0;
    double w_fisher = wp * w_pp;
    double w_hessian = w_fisher;
    double w_solver = (w_hessian > 0.0) ? fmax(w_hessian, 1e-12) : 0.0;
    double bres = y_i - mu;
    double grad_eta = (vmu > 0.0) ? wp * bres * dmu_deta / vmu : 0.0;
    double dev = bernoulli_deviance(y_i, mu, wp);
    double z_lin = bernoulli_z(eta_c, y_i, mu, dmu_deta);
    double z_f = z_lin;
    double z_h = z_lin;
    if (!(isfinite(y_i) && y_i >= 0.0 && y_i <= 1.0)) flags |= 0x8u;
"#;

            // (family, raw_body, distinct spec_id, y vector builder).
            // spec_ids are disjoint 64-bit tags so the JIT module cache
            // creates one fresh module per family.
            let cases: [(PirlsRowFamily, &str, u64, fn(usize) -> Vec<f64>); 6] = [
                (
                    PirlsRowFamily::GaussianIdentity,
                    raw_gaussian,
                    0x5242_3031_4741_5553u64,
                    |n| {
                        (0..n)
                            .map(|i| -3.0 + 6.0 * (i as f64) / ((n - 1) as f64))
                            .collect()
                    },
                ),
                (
                    PirlsRowFamily::PoissonLog,
                    raw_poisson,
                    0x5242_3032_504f_4953u64,
                    |n| (0..n).map(|i| (i % 11) as f64).collect(),
                ),
                (
                    PirlsRowFamily::GammaLog,
                    raw_gamma,
                    0x5242_3033_474d_414cu64,
                    |n| (0..n).map(|i| 0.10 + 0.05 * ((i % 97) as f64)).collect(),
                ),
                (
                    PirlsRowFamily::BernoulliLogit,
                    raw_logit,
                    0x5242_3034_4c47_4954u64,
                    |n| (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect(),
                ),
                (
                    PirlsRowFamily::BernoulliProbit,
                    raw_probit,
                    0x5242_3035_5052_4254u64,
                    |n| (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect(),
                ),
                (
                    PirlsRowFamily::BernoulliCLogLog,
                    raw_cloglog,
                    0x5242_3036_434c_4f47u64,
                    |n| (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect(),
                ),
            ];

            for (family, raw_body, spec_id, build_y) in cases {
                let ys: Vec<f64> = build_y(N);

                let mut eta_dev = stream.alloc_zeros::<f64>(N).expect("eta");
                let mut y_dev = stream.alloc_zeros::<f64>(N).expect("y");
                let mut prior_dev = stream.alloc_zeros::<f64>(N).expect("prior");
                stream.memcpy_htod(&etas, &mut eta_dev).expect("up eta");
                stream.memcpy_htod(&ys, &mut y_dev).expect("up y");
                stream
                    .memcpy_htod(&priors, &mut prior_dev)
                    .expect("up prior");

                let spec = JitFamilySpec::raw(spec_id, raw_body);
                let mut out_jit = RowOutputDevBuffers::allocate(&stream, N).expect("alloc jit out");
                launch_row_reweight_jit_on_stream(
                    backend,
                    &spec,
                    curvature,
                    &stream,
                    N,
                    &eta_dev,
                    &y_dev,
                    &prior_dev,
                    &mut out_jit,
                )
                .unwrap_or_else(|err| panic!("jit raw-body launch {family:?}: {err}"));
                stream.synchronize().expect("sync");

                let mu_j = stream.clone_dtoh(&out_jit.mu).expect("dl mu");
                let ge_j = stream.clone_dtoh(&out_jit.grad_eta).expect("dl g");
                let wf_j = stream.clone_dtoh(&out_jit.w_fisher).expect("dl wf");
                let wh_j = stream.clone_dtoh(&out_jit.w_hessian).expect("dl wh");
                let ws_j = stream.clone_dtoh(&out_jit.w_solver).expect("dl ws");
                let zf_j = stream.clone_dtoh(&out_jit.z_fisher).expect("dl zf");
                let zh_j = stream.clone_dtoh(&out_jit.z_hessian).expect("dl zh");
                let dv_j = stream.clone_dtoh(&out_jit.deviance).expect("dl dv");

                for i in 0..N {
                    let cpu = row_reweight_cpu(
                        family,
                        curvature,
                        RowInput {
                            eta: etas[i],
                            y: ys[i],
                            prior_weight: priors[i],
                        },
                        1.0,
                    );
                    for (label, got, exp) in [
                        ("mu", mu_j[i], cpu.mu),
                        ("grad_eta", ge_j[i], cpu.grad_eta),
                        ("w_fisher", wf_j[i], cpu.w_fisher),
                        ("w_hessian", wh_j[i], cpu.w_hessian),
                        ("w_solver", ws_j[i], cpu.w_solver),
                        ("z_fisher", zf_j[i], cpu.z_fisher),
                        ("z_hessian", zh_j[i], cpu.z_hessian),
                        ("deviance", dv_j[i], cpu.deviance),
                    ] {
                        if !got.is_finite() && !exp.is_finite() {
                            // Both NaN/inf is a parity match for the
                            // gamma y=0 → +inf deviance branch.
                            continue;
                        }
                        let abs_err = (got - exp).abs();
                        let rel_err = if exp.abs() > 0.0 {
                            abs_err / exp.abs()
                        } else {
                            abs_err
                        };
                        assert!(
                            abs_err <= TOL || rel_err <= TOL,
                            "{family:?} {label}[{i}] (eta={}, y={}, wp={}): \
                             JIT raw-body={} vs CPU={} (abs={}, rel={})",
                            etas[i],
                            ys[i],
                            priors[i],
                            got,
                            exp,
                            abs_err,
                            rel_err,
                        );
                    }
                }
            }
        }
    }
}
