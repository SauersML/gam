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
//! |              | Gaussian/Poisson and priorweight·shape·(yᵢ/μᵢ−1) for Gamma.|
//! | `w_fisher`   | Fisher expected weight (priorweight · h'(η)² / V(μ)).      |
//! |              | Used for inference (Var(β̂)).                              |
//! | `w_hessian`  | Curvature weight for the Newton/Laplace Hessian.            |
//! |              | == w_fisher on canonical links; observed correction on     |
//! |              | non-canonical Bernoulli + Gamma-log (Stage 5 populates).   |
//! | `w_solver`   | Exact `w_hessian` consumed by matrix assembly. Statistical  |
//! |              | rows are never floored or sign-projected; conditioning is  |
//! |              | exclusively an assembled-matrix ridge operation.           |
//! | `deviance`   | Per-row deviance contribution dᵢ; sum aggregated host-side  |
//! |              | for line search and convergence checks.                    |
//! | `status`     | Zero on success; otherwise a refusal code. A failed thread  |
//! |              | writes no numerical outputs.                               |
//!
//! Two-weight discipline is structural in the contract: the gradient is
//! emitted **directly** and never reconstructed from an artificial working
//! response, because that product is both unnecessary and unstable in tails.
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
use gam_problem::EstimationError;

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
}

/// Curvature surface used to populate `w_hessian` / `w_solver`.
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

/// Device refusal codes.  Zero is the only success value; a non-zero code
/// means that the row kernel wrote no numerical outputs for that row.  The
/// host deterministically selects the smallest failing row and replays the
/// CPU oracle to recover the full typed [`EstimationError`] (including the
/// exact offending value).
pub mod status_codes {
    pub const OK: u32 = 0;
    pub const ETA_DOMAIN: u32 = 1;
    pub const PRIOR_WEIGHT: u32 = 2;
    pub const RESPONSE: u32 = 3;
    pub const GAMMA_SHAPE: u32 = 4;
    pub const INVERSE_LINK: u32 = 5;
    pub const FISHER_WEIGHT: u32 = 6;
    pub const OBSERVED_WEIGHT: u32 = 7;
    pub const GRADIENT: u32 = 8;
    pub const DEVIANCE: u32 = 9;
    pub const FINAL_OUTPUT: u32 = 10;

    pub const fn quantity(code: u32) -> &'static str {
        match code {
            ETA_DOMAIN => "inverse-link eta domain",
            PRIOR_WEIGHT => "prior weight",
            RESPONSE => "response",
            GAMMA_SHAPE => "Gamma shape",
            INVERSE_LINK => "inverse-link jet",
            FISHER_WEIGHT => "Fisher weight",
            OBSERVED_WEIGHT => "observed Hessian weight",
            GRADIENT => "eta gradient",
            DEVIANCE => "deviance contribution",
            FINAL_OUTPUT => "final row output",
            _ => "unknown GPU PIRLS refusal",
        }
    }
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
    pub deviance: f64,
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
) -> Result<RowOutput, EstimationError> {
    row_reweight_cpu_at(0, family, mode, input, gamma_shape)
}

/// Indexed form of [`row_reweight_cpu`], used to reproduce a device refusal
/// with the correct row in its typed error.
pub fn row_reweight_cpu_at(
    row: usize,
    family: PirlsRowFamily,
    mode: CurvatureMode,
    input: RowInput,
    gamma_shape: f64,
) -> Result<RowOutput, EstimationError> {
    match family {
        PirlsRowFamily::GaussianIdentity => row_gaussian_identity(row, input, mode),
        PirlsRowFamily::PoissonLog => row_poisson_log(row, input, mode),
        PirlsRowFamily::GammaLog => row_gamma_log(row, input, mode, gamma_shape),
        PirlsRowFamily::BernoulliLogit => row_bernoulli_logit(row, input, mode),
        PirlsRowFamily::BernoulliProbit => row_bernoulli_probit(row, input, mode),
        PirlsRowFamily::BernoulliCLogLog => row_bernoulli_cloglog(row, input, mode),
    }
}

/// Recover the typed error represented by a device status vector.  Device
/// threads write one code per row, so scanning in index order makes concurrent
/// failures deterministic.  The scalar CPU replay supplies the exact
/// quantity/value payload without expanding the hot GPU ABI.
pub fn replay_first_refusal(
    family: PirlsRowFamily,
    mode: CurvatureMode,
    gamma_shape: f64,
    eta: &[f64],
    y: &[f64],
    prior_weight: &[f64],
    status: &[u32],
) -> Result<(), EstimationError> {
    let n = eta.len();
    if y.len() != n || prior_weight.len() != n || status.len() != n {
        return Err(EstimationError::InvalidInput(format!(
            "GPU PIRLS refusal replay length mismatch: eta={n}, y={}, prior_weight={}, status={}",
            y.len(),
            prior_weight.len(),
            status.len(),
        )));
    }
    let Some((row, &code)) = status
        .iter()
        .enumerate()
        .find(|(_, code)| **code != status_codes::OK)
    else {
        return Ok(());
    };
    let input = RowInput {
        eta: eta[row],
        y: y[row],
        prior_weight: prior_weight[row],
    };
    match row_reweight_cpu_at(row, family, mode, input, gamma_shape) {
        Err(error) => Err(error),
        Ok(_) => Err(row_error(
            row,
            status_codes::quantity(code),
            input.eta,
            f64::from(code),
        )),
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
fn row_error(row: usize, quantity: &'static str, eta: f64, value: f64) -> EstimationError {
    EstimationError::PirlsRowGeometryUnrepresentable {
        row,
        quantity,
        eta,
        value,
    }
}

#[inline]
fn finite_eta(link: &'static str, eta: f64) -> Result<(), EstimationError> {
    if eta.is_finite() {
        Ok(())
    } else {
        Err(EstimationError::InverseLinkDomainViolation {
            link,
            eta,
            lower: -f64::MAX,
            upper: f64::MAX,
        })
    }
}

#[inline]
fn prior_weight(row: usize, input: RowInput) -> Result<f64, EstimationError> {
    if input.prior_weight.is_finite() && input.prior_weight >= 0.0 {
        Ok(input.prior_weight)
    } else {
        Err(row_error(
            row,
            "prior weight",
            input.eta,
            input.prior_weight,
        ))
    }
}

#[inline]
fn certify_output(row: usize, eta: f64, output: RowOutput) -> Result<RowOutput, EstimationError> {
    for (quantity, value) in [
        ("mean", output.mu),
        ("eta gradient", output.grad_eta),
        ("Fisher weight", output.w_fisher),
        ("observed Hessian weight", output.w_hessian),
        ("solver Hessian weight", output.w_solver),
        ("deviance contribution", output.deviance),
    ] {
        if !value.is_finite() {
            return Err(row_error(row, quantity, eta, value));
        }
    }
    Ok(output)
}

/// Evaluate `a*b/c` while avoiding a false overflow/underflow caused solely by
/// operation order.  At least one of the product-first or quotient-first forms
/// is normally representable whenever the final positive f64 is; all three are
/// tried in a fixed order so CPU/device refusal and rounding stay deterministic.
#[inline]
fn positive_mul_div(a: f64, b: f64, c: f64) -> f64 {
    let product = a * b;
    if product.is_finite() && product > 0.0 {
        let value = product / c;
        if value.is_finite() && value > 0.0 {
            return value;
        }
    }
    let quotient_a = a / c;
    if quotient_a.is_finite() && quotient_a > 0.0 {
        let value = quotient_a * b;
        if value.is_finite() && value > 0.0 {
            return value;
        }
    }
    let quotient_b = b / c;
    if quotient_b.is_finite() && quotient_b > 0.0 {
        let value = quotient_b * a;
        if value.is_finite() && value > 0.0 {
            return value;
        }
    }
    product / c
}

/// `u - log1p(u)` without cancellation around zero.
#[inline]
fn gamma_unit_deviance_near_one(u: f64) -> f64 {
    if u.abs() > 0.125 {
        return u - u.ln_1p();
    }
    let mut power = u * u;
    let mut sum = 0.5 * power;
    for degree in 3..=32 {
        power *= u;
        let term = power / f64::from(degree);
        let next = if degree % 2 == 0 {
            sum + term
        } else {
            sum - term
        };
        if next == sum {
            break;
        }
        sum = next;
    }
    sum
}

/// `(1+u)log1p(u)-u` without cancellation around zero.
#[inline]
fn poisson_unit_deviance_near_one(u: f64) -> f64 {
    if u.abs() > 0.125 {
        return (1.0 + u) * u.ln_1p() - u;
    }
    let mut power = u * u;
    let mut sum = 0.5 * power;
    for degree in 3..=32 {
        power *= u;
        let coefficient =
            if degree % 2 == 0 { 1.0 } else { -1.0 } / (f64::from(degree) * f64::from(degree - 1));
        let next = sum + coefficient * power;
        if next == sum {
            break;
        }
        sum = next;
    }
    sum
}

#[inline]
fn row_gaussian_identity(
    row: usize,
    input: RowInput,
    mode: CurvatureMode,
) -> Result<RowOutput, EstimationError> {
    finite_eta("standard identity inverse link", input.eta)?;
    let w = prior_weight(row, input)?;
    let mu = input.eta;
    if w > 0.0 && !input.y.is_finite() {
        return Err(row_error(row, "Gaussian response", input.eta, input.y));
    }
    let resid = input.y - mu;
    let (grad_eta, dev) = if w == 0.0 {
        (0.0, 0.0)
    } else {
        (w * resid, w * resid * resid)
    };
    let w_hessian = select_w_hessian(mode, w, 0.0);
    certify_output(
        row,
        input.eta,
        RowOutput {
            mu,
            grad_eta,
            w_fisher: w,
            w_hessian,
            w_solver: w_hessian,
            deviance: dev,
        },
    )
}

#[inline]
fn row_poisson_log(
    row: usize,
    input: RowInput,
    mode: CurvatureMode,
) -> Result<RowOutput, EstimationError> {
    let mu = crate::mixture_link::log_link_solver_exp(input.eta)?;
    let w_prior = prior_weight(row, input)?;
    if w_prior > 0.0 && !(input.y.is_finite() && input.y >= 0.0) {
        return Err(row_error(row, "Poisson response", input.eta, input.y));
    }
    if w_prior == 0.0 {
        return certify_output(
            row,
            input.eta,
            RowOutput {
                mu,
                ..RowOutput::default()
            },
        );
    }
    let w_fisher = w_prior * mu;
    if !(w_fisher.is_finite() && w_fisher > 0.0) {
        return Err(row_error(row, "Poisson Fisher weight", input.eta, w_fisher));
    }
    let grad_eta = w_prior * (input.y - mu);
    let u = (input.y - mu) / mu;
    let dev_base = if input.y == 0.0 {
        w_fisher
    } else {
        // Accurate around saturation. In either far tail this dimensionless
        // ratio can become non-finite before multiplication by a tiny weight;
        // only then switch to the algebraically identical absolute-coordinate
        // expression, whose products are balanced independently.
        let scaled_unit = w_fisher * poisson_unit_deviance_near_one(u);
        if scaled_unit.is_finite() && scaled_unit >= 0.0 {
            scaled_unit
        } else {
            let weighted_y = positive_mul_div(w_fisher, input.y, mu);
            weighted_y * (input.y.ln() - input.eta - 1.0) + w_fisher
        }
    };
    let dev = 2.0 * dev_base;
    let w_hessian = select_w_hessian(mode, w_fisher, 0.0);
    certify_output(
        row,
        input.eta,
        RowOutput {
            mu,
            grad_eta,
            w_fisher,
            w_hessian,
            w_solver: w_hessian,
            deviance: dev,
        },
    )
}

#[inline]
fn row_gamma_log(
    row: usize,
    input: RowInput,
    mode: CurvatureMode,
    shape: f64,
) -> Result<RowOutput, EstimationError> {
    let mu = crate::mixture_link::log_link_solver_exp(input.eta)?;
    if !(shape.is_finite() && shape > 0.0) {
        return Err(row_error(row, "Gamma shape", input.eta, shape));
    }
    let w_prior = prior_weight(row, input)?;
    if w_prior > 0.0 && !(input.y.is_finite() && input.y > 0.0) {
        return Err(row_error(row, "Gamma response", input.eta, input.y));
    }
    if w_prior == 0.0 {
        return certify_output(
            row,
            input.eta,
            RowOutput {
                mu,
                ..RowOutput::default()
            },
        );
    }
    let w_fisher = w_prior * shape;
    if !(w_fisher.is_finite() && w_fisher > 0.0) {
        return Err(row_error(row, "Gamma Fisher weight", input.eta, w_fisher));
    }
    let observed_ratio = match mode {
        CurvatureMode::Fisher => None,
        CurvatureMode::Observed => {
            let weighted_ratio = positive_mul_div(w_fisher, input.y, mu);
            if !(weighted_ratio.is_finite() && weighted_ratio > 0.0) {
                return Err(row_error(
                    row,
                    "Gamma observed Hessian weight",
                    input.eta,
                    weighted_ratio,
                ));
            }
            Some(weighted_ratio)
        }
    };
    let w_hessian = observed_ratio.unwrap_or(w_fisher);
    if !w_hessian.is_finite() {
        return Err(row_error(
            row,
            "Gamma observed Hessian weight",
            input.eta,
            w_hessian,
        ));
    }
    let u = (input.y - mu) / mu;
    // `u` rounds to exactly -1 when y/mu is a representable but very small
    // ratio, and the dimensionless expression can overflow for the opposite
    // tail. Preserve the local-series path whenever it succeeds, then use the
    // weighted absolute-coordinate identity for those two tail cases.
    let scaled_unit = w_fisher * gamma_unit_deviance_near_one(u);
    let need_weighted_ratio = !u.is_finite() || !(scaled_unit.is_finite() && scaled_unit >= 0.0);
    let weighted_ratio = if need_weighted_ratio {
        observed_ratio.unwrap_or_else(|| positive_mul_div(w_fisher, input.y, mu))
    } else {
        0.0
    };
    let grad_eta = if u.is_finite() {
        w_fisher * u
    } else {
        weighted_ratio - w_fisher
    };
    let dev_base = if scaled_unit.is_finite() && scaled_unit >= 0.0 {
        scaled_unit
    } else {
        weighted_ratio - w_fisher * (1.0 + input.y.ln() - input.eta)
    };
    let dev = 2.0 * dev_base;
    certify_output(
        row,
        input.eta,
        RowOutput {
            mu,
            grad_eta,
            w_fisher,
            w_hessian,
            w_solver: w_hessian,
            deviance: dev,
        },
    )
}

#[inline]
fn bernoulli_response(row: usize, input: RowInput, w: f64) -> Result<(), EstimationError> {
    if w == 0.0 || (input.y.is_finite() && (0.0..=1.0).contains(&input.y)) {
        Ok(())
    } else {
        Err(row_error(row, "binomial response", input.eta, input.y))
    }
}

#[inline]
fn row_bernoulli_logit(
    row: usize,
    input: RowInput,
    mode: CurvatureMode,
) -> Result<RowOutput, EstimationError> {
    finite_eta("standard logit inverse link", input.eta)?;
    let w_prior = prior_weight(row, input)?;
    bernoulli_response(row, input, w_prior)?;
    let tail = (-input.eta.abs()).exp();
    let denom = 1.0 + tail;
    let (mu, residual) = if input.eta >= 0.0 {
        let one_minus_mu = tail / denom;
        let residual = if input.y == 1.0 {
            one_minus_mu
        } else {
            (input.y - 1.0) + one_minus_mu
        };
        (1.0 / denom, residual)
    } else {
        let mu = tail / denom;
        (mu, input.y - mu)
    };
    let dmu_deta = tail / (denom * denom);
    if !(dmu_deta.is_finite() && dmu_deta > 0.0) {
        return Err(row_error(
            row,
            "canonical-logit inverse-link jet",
            input.eta,
            dmu_deta,
        ));
    }
    if w_prior == 0.0 {
        return certify_output(
            row,
            input.eta,
            RowOutput {
                mu,
                ..RowOutput::default()
            },
        );
    }
    let w_fisher = w_prior * dmu_deta;
    if !(w_fisher.is_finite() && w_fisher > 0.0) {
        return Err(row_error(row, "logit Fisher weight", input.eta, w_fisher));
    }
    let grad_eta = w_prior * residual;
    let dev = bernoulli_logit_deviance(input.y, input.eta, w_prior);
    let w_hessian = select_w_hessian(mode, w_fisher, 0.0);
    certify_output(
        row,
        input.eta,
        RowOutput {
            mu,
            grad_eta,
            w_fisher,
            w_hessian,
            w_solver: w_hessian,
            deviance: dev,
        },
    )
}

#[inline]
fn row_bernoulli_probit(
    row: usize,
    input: RowInput,
    mode: CurvatureMode,
) -> Result<RowOutput, EstimationError> {
    finite_eta("standard probit inverse link", input.eta)?;
    let d1 = standard_normal_pdf(input.eta);
    row_bernoulli_noncanonical(
        row,
        input,
        mode,
        standard_normal_cdf(input.eta),
        d1,
        -input.eta * d1,
    )
}

#[inline]
fn row_bernoulli_cloglog(
    row: usize,
    input: RowInput,
    mode: CurvatureMode,
) -> Result<RowOutput, EstimationError> {
    finite_eta("standard complementary-log-log inverse link", input.eta)?;
    let inner = input.eta.exp();
    let mu = -(-inner).exp_m1();
    let complement = (-inner).exp();
    let d1 = inner * complement;
    row_bernoulli_noncanonical(row, input, mode, mu, d1, d1 * (1.0 - inner))
}

#[inline]
fn row_bernoulli_noncanonical(
    row: usize,
    input: RowInput,
    mode: CurvatureMode,
    mu: f64,
    d1: f64,
    d2: f64,
) -> Result<RowOutput, EstimationError> {
    let w_prior = prior_weight(row, input)?;
    bernoulli_response(row, input, w_prior)?;
    if !(mu.is_finite() && mu > 0.0 && mu < 1.0 && d1.is_finite() && d1 > 0.0 && d2.is_finite()) {
        return Err(row_error(row, "inverse-link jet", input.eta, mu));
    }
    if w_prior == 0.0 {
        return certify_output(
            row,
            input.eta,
            RowOutput {
                mu,
                ..RowOutput::default()
            },
        );
    }
    let v = mu * (1.0 - mu);
    let fisher_per_prior = d1 * d1 / v;
    let w_fisher = w_prior * fisher_per_prior;
    if !(v.is_finite()
        && v > 0.0
        && fisher_per_prior.is_finite()
        && fisher_per_prior > 0.0
        && w_fisher.is_finite()
        && w_fisher > 0.0)
    {
        return Err(row_error(
            row,
            "Bernoulli Fisher weight",
            input.eta,
            w_fisher,
        ));
    }
    let resid = input.y - mu;
    let grad_eta = w_prior * resid * d1 / v;
    let bracket = d2 / v - d1 * d1 * (1.0 - 2.0 * mu) / (v * v);
    // -d²ℓ/dη² = W_F - prior·(y-μ)·d(h'/V)/dη.
    let observed_correction = -w_prior * resid * bracket;
    let w_hessian = select_w_hessian(mode, w_fisher, observed_correction);
    if !w_hessian.is_finite() {
        return Err(row_error(
            row,
            "Bernoulli observed Hessian weight",
            input.eta,
            w_hessian,
        ));
    }
    let dev = bernoulli_deviance(input.y, mu, w_prior);
    certify_output(
        row,
        input.eta,
        RowOutput {
            mu,
            grad_eta,
            w_fisher,
            w_hessian,
            w_solver: w_hessian,
            deviance: dev,
        },
    )
}

#[inline]
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

#[inline]
fn expm1_minus_x(x: f64) -> f64 {
    if x.abs() > 0.5 {
        return x.exp_m1() - x;
    }
    let mut term = 0.5 * x * x;
    let mut sum = term;
    let mut degree = 2.0;
    loop {
        degree += 1.0;
        term *= x / degree;
        let next = sum + term;
        if next == sum {
            return next;
        }
        sum = next;
    }
}

#[inline]
fn log1p_minus_x(x: f64) -> f64 {
    if x.abs() > 0.5 {
        return x.ln_1p() - x;
    }
    let mut power = x * x;
    let mut sign = -1.0;
    let mut degree = 2.0;
    let mut sum = sign * power / degree;
    loop {
        power *= x;
        sign = -sign;
        degree += 1.0;
        let next = sum + sign * power / degree;
        if next == sum {
            return next;
        }
        sum = next;
    }
}

#[inline]
fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `KL(sigmoid(a) || sigmoid(b))` without subtracting entropy from cross
/// entropy. The local branch evaluates only second-order remainders.
#[inline]
fn bernoulli_kl_from_logits(a: f64, b: f64) -> f64 {
    if a == b {
        return 0.0;
    }
    let h = b - a;
    if h.abs() <= 0.5 {
        let (p, local_h) = if a <= 0.0 {
            (logistic(a), h)
        } else {
            (logistic(-a), -h)
        };
        let em1 = local_h.exp_m1();
        let x = p * em1;
        return log1p_minus_x(x) + p * expm1_minus_x(local_h);
    }
    if a <= 0.0 {
        let p = logistic(a);
        p * (a - b) + softplus(b) - softplus(a)
    } else {
        let q = logistic(-a);
        q * (b - a) + softplus(-b) - softplus(-a)
    }
}

/// Accurate Poisson deviance cell `x log(x/m) + m - x`. The local series is
/// used only when its contraction factor is small, so the loop is short and
/// deterministic on both CPU and GPU.
#[inline]
fn bd0(x: f64, m: f64) -> f64 {
    if x == 0.0 {
        return m;
    }
    if x == m {
        return 0.0;
    }
    let hi = x.max(m);
    let lo = x.min(m);
    if (x - m).abs() / hi < 0.2 {
        let v = ((x - m) / hi) / (1.0 + lo / hi);
        let mut sum = (x - m) * v;
        let mut term = 2.0 * x * v;
        let v2 = v * v;
        let mut denominator = 3.0;
        loop {
            term *= v2;
            let next = sum + term / denominator;
            if next == sum {
                return next;
            }
            sum = next;
            denominator += 2.0;
        }
    }
    x * (x.ln() - m.ln()) + (m - x)
}

#[inline]
fn bernoulli_logit_deviance(y: f64, eta: f64, w: f64) -> f64 {
    let unit = if y == 0.0 {
        softplus(eta)
    } else if y == 1.0 {
        softplus(-eta)
    } else {
        let response_logit = y.ln() - (-y).ln_1p();
        bernoulli_kl_from_logits(response_logit, eta)
    };
    2.0 * w * unit
}

#[inline]
fn bernoulli_deviance(y: f64, mu: f64, w: f64) -> f64 {
    2.0 * w * (bd0(y, mu) + bd0(1.0 - y, 1.0 - mu))
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
    /// Full final-row kernel (mu, grad_eta, w_hessian, w_solver, deviance,
    /// status).
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
    /// `(family, curvature)`. Writes the full production row surface.
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
///   value plus the Gamma shape constant. The generator emits the
///   matching built-in row body, identical to the cached built-in
///   kernel — useful for end-to-end JIT path validation against the
///   built-in cache.
/// - **Level B** (`JitFamilySpec::raw`): provide raw CUDA source for
///   the row body. The body must define the same per-row locals the
///   kernel shell expects: `mu`, `grad_eta`, `w_hessian`, `w_solver`,
///   `dev`, and update `status`. The shell
///   wraps it in the canonical
///   `extern "C" __global__ void pirls_row_jit_{spec_id}(...)`
///   signature that [`launch_row_reweight_on_stream`] expects.
#[derive(Clone, Debug)]
pub struct JitFamilySpec {
    /// Process-unique identifier for this spec; the module cache uses
    /// it as a key so callers must reuse the same `spec_id` for the
    /// same body and pick a new one whenever the body changes.
    pub spec_id: u64,
    /// CUDA body source. Must read from `eta_i`, `y_i`, `wp`, set
    /// `status`, and assign to `mu`, `grad_eta`, `w_hessian`, `w_solver`,
    /// and `dev`. See [`common_device_prolog`] for
    /// the available helpers.
    pub body: String,
}

impl JitFamilySpec {
    /// Level A: build a spec from a built-in `(family, curvature)`
    /// pair. The generator reuses the same per-family body as the
    /// built-in cached kernel — useful to validate the JIT pipeline
    /// end-to-end against the built-in numerical reference.
    #[cfg(target_os = "linux")]
    pub fn glm(
        spec_id: u64,
        family: PirlsRowFamily,
        curvature: CurvatureMode,
        gamma_shape: f64,
    ) -> Self {
        let mut body = match family {
            PirlsRowFamily::GaussianIdentity => gaussian_identity_body(curvature),
            PirlsRowFamily::PoissonLog => poisson_log_body(curvature),
            PirlsRowFamily::GammaLog => gamma_log_body(curvature),
            PirlsRowFamily::BernoulliLogit => bernoulli_logit_body(curvature),
            PirlsRowFamily::BernoulliProbit => bernoulli_probit_body(curvature),
            PirlsRowFamily::BernoulliCLogLog => bernoulli_cloglog_body(curvature),
        };
        if matches!(family, PirlsRowFamily::GammaLog) {
            body.insert_str(0, &format!("    const double shape = {gamma_shape:?};\n"));
        }
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
    double* __restrict__ w_hessian_out,
    double* __restrict__ w_solver_out,
    double* __restrict__ deviance_out,
    unsigned int* __restrict__ status_out
) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int status = PIRLS_OK;
    double eta_i = eta[i];
    double y_i = y[i];
    double wp = prior_w[i];
{body}
    if (status == PIRLS_OK) {{
        mu_out[i] = mu;
        grad_eta_out[i] = grad_eta;
        w_hessian_out[i] = w_hessian;
        w_solver_out[i] = w_solver;
        deviance_out[i] = dev;
    }}
    status_out[i] = status;
}}
"#,
            prolog = common_device_prolog(),
        )
    }
}

/// Device-resident per-row output buffers for the GPU row-reweight kernel.
///
/// **final-row mode**: the five production numerical fields plus status,
/// length `n`. Written
/// once at convergence by [`launch_row_reweight_on_stream`]. For the hot
/// inner-loop use [`SolveRowBuffers`]; for line-search use
/// [`AlphaLadderDevBuffers`].
#[cfg(target_os = "linux")]
pub struct RowOutputDevBuffers {
    pub mu: cudarc::driver::CudaSlice<f64>,
    pub grad_eta: cudarc::driver::CudaSlice<f64>,
    pub w_hessian: cudarc::driver::CudaSlice<f64>,
    pub w_solver: cudarc::driver::CudaSlice<f64>,
    pub deviance: cudarc::driver::CudaSlice<f64>,
    pub status: cudarc::driver::CudaSlice<u32>,
    pub n: usize,
}

#[cfg(target_os = "linux")]
impl RowOutputDevBuffers {
    /// Allocate all production per-row output buffers (length `n`) on `stream`.
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
            w_hessian: alloc_f64("w_hessian")?,
            w_solver: alloc_f64("w_solver")?,
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
/// and `status` (one exact refusal code per row). Written
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
    /// Zero on success, otherwise the row's deterministic refusal code.
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
/// `objective_dev[k]`. Each row writes its refusal code to
/// `status_dev[k * n + i]`, preserving deterministic row identity.
/// After a single `memcpy_dtoh` the host picks the first α that achieves
/// deviance descent — no per-α kernel launch, no full row-output write.
#[cfg(target_os = "linux")]
pub struct AlphaLadderDevBuffers {
    /// Device: summed deviance for each alpha step, length [`ALPHA_LADDER_LEN`].
    pub objective_dev: cudarc::driver::CudaSlice<f64>,
    /// Device: row refusal codes in alpha-major order, length
    /// `ALPHA_LADDER_LEN * n`.
    pub status_dev: cudarc::driver::CudaSlice<u32>,
    pub n: usize,
}

#[cfg(target_os = "linux")]
impl AlphaLadderDevBuffers {
    /// Allocate the ladder device buffers on `stream`.
    pub fn allocate(stream: &Arc<cudarc::driver::CudaStream>, n: usize) -> Result<Self, GpuError> {
        let status_len = ALPHA_LADDER_LEN.checked_mul(n).ok_or_else(|| {
            gam_gpu::gpu_err!("pirls_row ladder status length overflows: {ALPHA_LADDER_LEN} * {n}")
        })?;
        Ok(Self {
            objective_dev: stream
                .alloc_zeros::<f64>(ALPHA_LADDER_LEN)
                .gpu_ctx_with(|err| format!("pirls_row ladder alloc objective: {err}"))?,
            status_dev: stream
                .alloc_zeros::<u32>(status_len)
                .gpu_ctx_with(|err| format!("pirls_row ladder alloc status: {err}"))?,
            n,
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
/// families compile a ten-argument kernel and ignore this value. Pass `1.0`
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
    let n_u32 = u32::try_from(n)
        .map_err(|_| gam_gpu::gpu_err!("n={n} exceeds u32 for row reweight grid sizing"))?;
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
    builder.arg(&mut out.w_hessian);
    builder.arg(&mut out.w_solver);
    builder.arg(&mut out.deviance);
    builder.arg(&mut out.status);
    // SAFETY: kernel signature for non-GammaLog is (n:i32, 3×const f64*,
    // 5×mut f64*, 1×mut u32*). GammaLog extends this with `double shape`
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
    builder.arg(&mut out.w_hessian);
    builder.arg(&mut out.w_solver);
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
/// (`solve_row_source_for`) that skips the `mu` and `w_hessian` stores,
/// reducing both bandwidth and register
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
    let n_u32 = u32::try_from(n)
        .map_err(|_| gam_gpu::gpu_err!("n={n} exceeds u32 for solve-row grid sizing"))?;
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
/// writes each refusal code to `out.status_dev[k * n + i]`.
///
/// The grid is `(row_blocks × ALPHA_LADDER_LEN)`: block index `bx / n_blocks`
/// selects the alpha slot, `bx % n_blocks` selects the row tile.
///
/// Caller must call [`AlphaLadderDevBuffers::zero`] before each launch, then
/// reduce the alpha-major status matrix with the deterministic smallest-row
/// reducer, then read the seven objectives plus seven row/code summaries.
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
    if out.n != n {
        gam_gpu::gpu_bail!("alpha-ladder buffers shape {} mismatches n={n}", out.n);
    }
    let module = backend.module_for_ladder(family, curvature)?;
    let kernel_name = family.ladder_kernel_name();
    let func = module
        .load_function(kernel_name)
        .gpu_ctx_with(|err| format!("alpha-ladder load_function({kernel_name}): {err}"))?;
    const THREADS_PER_BLOCK: u32 = 256;
    let n_u32 = u32::try_from(n)
        .map_err(|_| gam_gpu::gpu_err!("n={n} exceeds u32 for alpha-ladder grid sizing"))?;
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
    // Atomic double-precision add to objective_out[blockIdx.y]; each thread
    // owns status_out[blockIdx.y * n + i].
    unsafe { builder.launch(cfg) }
        .map(|_event_pair| ())
        .gpu_ctx_with(|err| format!("alpha-ladder launch({kernel_name}): {err}"))
}

// ────────────────────────────────────────────────────────────────────────
// CUDA sources (one per family / curvature pair)
// ────────────────────────────────────────────────────────────────────────

/// Common device-side helpers shared across every family kernel.  The log-link
/// interval is injected from the same Rust constants used by the CPU oracle,
/// so the generated CUDA source cannot drift to a second domain policy.
#[cfg(target_os = "linux")]
fn common_device_prolog() -> String {
    // Brace-heavy CUDA source: keep it a plain raw string literal so every C
    // brace stays literal (no `format!` grammar to escape), and substitute the
    // two host solver bounds by unique sentinel token.
    r#"
extern "C" {
    double exp(double);
    double log(double);
    double log1p(double);
    double expm1(double);
    double fabs(double);
    double erfc(double);
}

static constexpr double PIRLS_LOG_ETA_MIN = __PIRLS_LOG_ETA_MIN__;
static constexpr double PIRLS_LOG_ETA_MAX = __PIRLS_LOG_ETA_MAX__;

static constexpr unsigned int PIRLS_OK = 0u;
static constexpr unsigned int PIRLS_ETA_DOMAIN = 1u;
static constexpr unsigned int PIRLS_PRIOR_WEIGHT = 2u;
static constexpr unsigned int PIRLS_RESPONSE = 3u;
static constexpr unsigned int PIRLS_GAMMA_SHAPE = 4u;
static constexpr unsigned int PIRLS_INVERSE_LINK = 5u;
static constexpr unsigned int PIRLS_FISHER_WEIGHT = 6u;
static constexpr unsigned int PIRLS_OBSERVED_WEIGHT = 7u;
static constexpr unsigned int PIRLS_GRADIENT = 8u;
static constexpr unsigned int PIRLS_DEVIANCE = 9u;
static constexpr unsigned int PIRLS_FINAL_OUTPUT = 10u;

__device__ __forceinline__ void pirls_refuse(unsigned int* status, unsigned int code) {
    if (*status == PIRLS_OK) *status = code;
}

__device__ __forceinline__ bool pirls_log_eta_valid(double eta) {
    return eta >= PIRLS_LOG_ETA_MIN && eta <= PIRLS_LOG_ETA_MAX;
}

__device__ __forceinline__ double softplus(double x) {
    return (x > 0.0 ? x : 0.0) + log1p(exp(-fabs(x)));
}

__device__ __forceinline__ double expm1_minus_x(double x) {
    if (fabs(x) > 0.5) return expm1(x) - x;
    double term = 0.5 * x * x;
    double sum = term;
    double degree = 2.0;
    for (;;) {
        degree += 1.0;
        term *= x / degree;
        double next = sum + term;
        if (next == sum) return next;
        sum = next;
    }
}

__device__ __forceinline__ double log1p_minus_x(double x) {
    if (fabs(x) > 0.5) return log1p(x) - x;
    double power = x * x;
    double sign = -1.0;
    double degree = 2.0;
    double sum = sign * power / degree;
    for (;;) {
        power *= x;
        sign = -sign;
        degree += 1.0;
        double next = sum + sign * power / degree;
        if (next == sum) return next;
        sum = next;
    }
}

__device__ __forceinline__ double logistic(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + exp(-x));
    double e = exp(x);
    return e / (1.0 + e);
}

__device__ __forceinline__ double bernoulli_kl_from_logits(double a, double b) {
    if (a == b) return 0.0;
    double h = b - a;
    if (fabs(h) <= 0.5) {
        double p = a <= 0.0 ? logistic(a) : logistic(-a);
        double local_h = a <= 0.0 ? h : -h;
        double em1 = expm1(local_h);
        double x = p * em1;
        return log1p_minus_x(x) + p * expm1_minus_x(local_h);
    }
    if (a <= 0.0) {
        double p = logistic(a);
        return p * (a - b) + softplus(b) - softplus(a);
    }
    double q = logistic(-a);
    return q * (b - a) + softplus(-b) - softplus(-a);
}

__device__ __forceinline__ double bd0(double x, double m) {
    if (x == 0.0) return m;
    if (x == m) return 0.0;
    double hi = x > m ? x : m;
    double lo = x < m ? x : m;
    if (fabs(x - m) / hi < 0.2) {
        double v = ((x - m) / hi) / (1.0 + lo / hi);
        double sum = (x - m) * v;
        double term = 2.0 * x * v;
        double v2 = v * v;
        double denominator = 3.0;
        for (;;) {
            term *= v2;
            double next = sum + term / denominator;
            if (next == sum) return next;
            sum = next;
            denominator += 2.0;
        }
    }
    return x * (log(x) - log(m)) + (m - x);
}

__device__ __forceinline__ double bernoulli_deviance(double y, double mu, double w) {
    return 2.0 * w * (bd0(y, mu) + bd0(1.0 - y, 1.0 - mu));
}

__device__ __forceinline__ double logit_deviance(double y, double eta, double w) {
    double unit;
    if (y == 0.0) unit = softplus(eta);
    else if (y == 1.0) unit = softplus(-eta);
    else {
        double response_logit = log(y) - log1p(-y);
        unit = bernoulli_kl_from_logits(response_logit, eta);
    }
    return 2.0 * w * unit;
}

__device__ __forceinline__ double std_norm_cdf(double x) {
    return 0.5 * erfc(-x * 0.7071067811865475);
}

__device__ __forceinline__ double std_norm_pdf(double x) {
    return 0.3989422804014327 * exp(-0.5 * x * x);
}

__device__ __forceinline__ double positive_mul_div(double a, double b, double c) {
    double product = a * b;
    if (isfinite(product) && product > 0.0) {
        double value = product / c;
        if (isfinite(value) && value > 0.0) return value;
    }
    double quotient_a = a / c;
    if (isfinite(quotient_a) && quotient_a > 0.0) {
        double value = quotient_a * b;
        if (isfinite(value) && value > 0.0) return value;
    }
    double quotient_b = b / c;
    if (isfinite(quotient_b) && quotient_b > 0.0) {
        double value = quotient_b * a;
        if (isfinite(value) && value > 0.0) return value;
    }
    return product / c;
}

__device__ __forceinline__ double gamma_unit_deviance_near_one(double u) {
    if (fabs(u) > 0.125) return u - log1p(u);
    double power = u * u;
    double sum = 0.5 * power;
    for (int degree = 3; degree <= 32; ++degree) {
        power *= u;
        double term = power / (double)degree;
        double next = sum + ((degree & 1) ? -term : term);
        if (next == sum) break;
        sum = next;
    }
    return sum;
}

__device__ __forceinline__ double poisson_unit_deviance_near_one(double u) {
    if (fabs(u) > 0.125) return (1.0 + u) * log1p(u) - u;
    double power = u * u;
    double sum = 0.5 * power;
    for (int degree = 3; degree <= 32; ++degree) {
        power *= u;
        double coefficient = ((degree & 1) ? -1.0 : 1.0)
            / ((double)degree * (double)(degree - 1));
        double next = sum + coefficient * power;
        if (next == sum) break;
        sum = next;
    }
    return sum;
}

__device__ __forceinline__ bool pirls_outputs_finite(
    double mu, double grad_eta, double w_fisher, double w_hessian,
    double w_solver, double dev
) {
    return isfinite(mu) && isfinite(grad_eta) && isfinite(w_fisher)
        && isfinite(w_hessian) && isfinite(w_solver) && isfinite(dev);
}
"#
    .replace(
        "__PIRLS_LOG_ETA_MIN__",
        &format!("{:?}", crate::mixture_link::LOG_LINK_SOLVER_ETA_MIN),
    )
    .replace(
        "__PIRLS_LOG_ETA_MAX__",
        &format!("{:?}", crate::mixture_link::LOG_LINK_SOLVER_ETA_MAX),
    )
}

/// Build the per-family CUDA source. Each source defines exactly one entry
/// kernel (`family.kernel_name()`) reading the input arrays and writing the
/// output arrays defined by the [`RowOutput`] contract above.
///
/// The GammaLog kernel has an extra `double shape` parameter after `prior_w`
/// so the host can forward the active dispersion shape. All other families
/// use the standard ten-argument signature.
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
    double* __restrict__ w_hessian_out,
    double* __restrict__ w_solver_out,
    double* __restrict__ deviance_out,
    unsigned int* __restrict__ status_out
) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int status = PIRLS_OK;
    double eta_i = eta[i];
    double y_i = y[i];
    double wp = prior_w[i];
{body}
    if (status == PIRLS_OK) {{
        mu_out[i] = mu;
        grad_eta_out[i] = grad_eta;
        w_hessian_out[i] = w_hessian;
        w_solver_out[i] = w_solver;
        deviance_out[i] = dev;
    }}
    status_out[i] = status;
}}
"#,
        prolog = common_device_prolog(),
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
        r#"{tag}    double mu = 0.0, grad_eta = 0.0, w_fisher = 0.0;
    double w_hessian = 0.0, w_solver = 0.0, dev = 0.0;
    if (!isfinite(eta_i)) pirls_refuse(&status, PIRLS_ETA_DOMAIN);
    if (status == PIRLS_OK && !(isfinite(wp) && wp >= 0.0))
        pirls_refuse(&status, PIRLS_PRIOR_WEIGHT);
    if (status == PIRLS_OK && wp > 0.0 && !isfinite(y_i))
        pirls_refuse(&status, PIRLS_RESPONSE);
    if (status == PIRLS_OK) {{
        mu = eta_i;
        w_fisher = wp;
        w_hessian = wp;
        w_solver = w_hessian;
        if (wp > 0.0) {{
            double resid = y_i - mu;
            grad_eta = wp * resid;
            dev = wp * resid * resid;
        }}
    }}
    if (status == PIRLS_OK && !pirls_outputs_finite(
            mu, grad_eta, w_fisher, w_hessian, w_solver, dev))
        pirls_refuse(&status, PIRLS_FINAL_OUTPUT);
"#
    )
}

#[cfg(target_os = "linux")]
fn poisson_log_body(curvature: CurvatureMode) -> String {
    let tag = curvature_tag(curvature);
    format!(
        r#"{tag}    double mu = 0.0, grad_eta = 0.0, w_fisher = 0.0;
    double w_hessian = 0.0, w_solver = 0.0, dev = 0.0;
    if (!pirls_log_eta_valid(eta_i)) pirls_refuse(&status, PIRLS_ETA_DOMAIN);
    if (status == PIRLS_OK && !(isfinite(wp) && wp >= 0.0))
        pirls_refuse(&status, PIRLS_PRIOR_WEIGHT);
    if (status == PIRLS_OK && wp > 0.0 && !(isfinite(y_i) && y_i >= 0.0))
        pirls_refuse(&status, PIRLS_RESPONSE);
    if (status == PIRLS_OK) {{
        mu = exp(eta_i);
        if (!(isfinite(mu) && mu > 0.0)) pirls_refuse(&status, PIRLS_INVERSE_LINK);
    }}
    if (status == PIRLS_OK && wp > 0.0) {{
        w_fisher = wp * mu;
        if (!(isfinite(w_fisher) && w_fisher > 0.0))
            pirls_refuse(&status, PIRLS_FISHER_WEIGHT);
        if (status == PIRLS_OK) {{
            w_hessian = w_fisher;
            w_solver = w_hessian;
            grad_eta = wp * (y_i - mu);
            double u = (y_i - mu) / mu;
            double dev_base;
            if (y_i == 0.0) {{
                dev_base = w_fisher;
            }} else {{
                double scaled_unit = w_fisher * poisson_unit_deviance_near_one(u);
                if (isfinite(scaled_unit) && scaled_unit >= 0.0) {{
                    dev_base = scaled_unit;
                }} else {{
                    double weighted_y = positive_mul_div(w_fisher, y_i, mu);
                    dev_base = weighted_y * (log(y_i) - eta_i - 1.0) + w_fisher;
                }}
            }}
            if (!isfinite(grad_eta)) pirls_refuse(&status, PIRLS_GRADIENT);
            dev = 2.0 * dev_base;
            if (!isfinite(dev)) pirls_refuse(&status, PIRLS_DEVIANCE);
        }}
    }}
    if (status == PIRLS_OK && !pirls_outputs_finite(
            mu, grad_eta, w_fisher, w_hessian, w_solver, dev))
        pirls_refuse(&status, PIRLS_FINAL_OUTPUT);
"#
    )
}

#[cfg(target_os = "linux")]
fn gamma_log_body(curvature: CurvatureMode) -> String {
    // `shape` is a kernel parameter (see `cuda_source_for`); the body reads it
    // directly. No local shadowing needed.
    let tag = curvature_tag(curvature);
    format!(
        r#"{tag}    double mu = 0.0, grad_eta = 0.0, w_fisher = 0.0;
    double w_hessian = 0.0, w_solver = 0.0, dev = 0.0;
    if (!pirls_log_eta_valid(eta_i)) pirls_refuse(&status, PIRLS_ETA_DOMAIN);
    if (status == PIRLS_OK && !(isfinite(shape) && shape > 0.0))
        pirls_refuse(&status, PIRLS_GAMMA_SHAPE);
    if (status == PIRLS_OK && !(isfinite(wp) && wp >= 0.0))
        pirls_refuse(&status, PIRLS_PRIOR_WEIGHT);
    if (status == PIRLS_OK && wp > 0.0 && !(isfinite(y_i) && y_i > 0.0))
        pirls_refuse(&status, PIRLS_RESPONSE);
    if (status == PIRLS_OK) {{
        mu = exp(eta_i);
        if (!(isfinite(mu) && mu > 0.0)) pirls_refuse(&status, PIRLS_INVERSE_LINK);
    }}
    if (status == PIRLS_OK && wp > 0.0) {{
        w_fisher = wp * shape;
        if (!(isfinite(w_fisher) && w_fisher > 0.0))
            pirls_refuse(&status, PIRLS_FISHER_WEIGHT);
#ifdef PIRLS_CURVATURE_OBSERVED
        double weighted_ratio_observed = positive_mul_div(w_fisher, y_i, mu);
        if (!(isfinite(weighted_ratio_observed) && weighted_ratio_observed > 0.0))
            pirls_refuse(&status, PIRLS_OBSERVED_WEIGHT);
        w_hessian = weighted_ratio_observed;
#else
        w_hessian = w_fisher;
#endif
        if (!isfinite(w_hessian)) pirls_refuse(&status, PIRLS_OBSERVED_WEIGHT);
        w_solver = w_hessian;
        double u = (y_i - mu) / mu;
        double scaled_unit = w_fisher * gamma_unit_deviance_near_one(u);
        bool need_weighted_ratio = !isfinite(u)
            || !(isfinite(scaled_unit) && scaled_unit >= 0.0);
        double weighted_ratio = 0.0;
#ifdef PIRLS_CURVATURE_OBSERVED
        weighted_ratio = weighted_ratio_observed;
#else
        if (need_weighted_ratio)
            weighted_ratio = positive_mul_div(w_fisher, y_i, mu);
#endif
        grad_eta = isfinite(u) ? w_fisher * u : weighted_ratio - w_fisher;
        double dev_base;
        if (isfinite(scaled_unit) && scaled_unit >= 0.0) {{
            dev_base = scaled_unit;
        }} else {{
            dev_base = weighted_ratio - w_fisher * (1.0 + log(y_i) - eta_i);
        }}
        if (!isfinite(grad_eta)) pirls_refuse(&status, PIRLS_GRADIENT);
        dev = 2.0 * dev_base;
        if (!isfinite(dev)) pirls_refuse(&status, PIRLS_DEVIANCE);
    }}
    if (status == PIRLS_OK && !pirls_outputs_finite(
            mu, grad_eta, w_fisher, w_hessian, w_solver, dev))
        pirls_refuse(&status, PIRLS_FINAL_OUTPUT);
"#
    )
}

#[cfg(target_os = "linux")]
fn bernoulli_logit_body(curvature: CurvatureMode) -> String {
    let tag = curvature_tag(curvature);
    format!(
        r#"{tag}    double mu = 0.0, grad_eta = 0.0, w_fisher = 0.0;
    double w_hessian = 0.0, w_solver = 0.0, dev = 0.0;
    if (!isfinite(eta_i)) pirls_refuse(&status, PIRLS_ETA_DOMAIN);
    if (status == PIRLS_OK && !(isfinite(wp) && wp >= 0.0))
        pirls_refuse(&status, PIRLS_PRIOR_WEIGHT);
    if (status == PIRLS_OK && wp > 0.0
            && !(isfinite(y_i) && y_i >= 0.0 && y_i <= 1.0))
        pirls_refuse(&status, PIRLS_RESPONSE);
    double tail = exp(-fabs(eta_i));
    double denom = 1.0 + tail;
    double dmu_deta = tail / (denom * denom);
    if (status == PIRLS_OK) {{
        mu = eta_i >= 0.0 ? 1.0 / denom : tail / denom;
        if (!(isfinite(mu) && mu >= 0.0 && mu <= 1.0
                && isfinite(dmu_deta) && dmu_deta > 0.0))
            pirls_refuse(&status, PIRLS_INVERSE_LINK);
    }}
    if (status == PIRLS_OK && wp > 0.0) {{
        double residual;
        if (eta_i >= 0.0) {{
            double one_minus_mu = tail / denom;
            residual = y_i == 1.0 ? one_minus_mu : (y_i - 1.0) + one_minus_mu;
        }} else {{
            residual = y_i - mu;
        }}
        w_fisher = wp * dmu_deta;
        if (!(isfinite(w_fisher) && w_fisher > 0.0))
            pirls_refuse(&status, PIRLS_FISHER_WEIGHT);
        w_hessian = w_fisher;
        w_solver = w_hessian;
        grad_eta = wp * residual;
        if (!isfinite(grad_eta)) pirls_refuse(&status, PIRLS_GRADIENT);
        dev = logit_deviance(y_i, eta_i, wp);
        if (!isfinite(dev)) pirls_refuse(&status, PIRLS_DEVIANCE);
    }}
    if (status == PIRLS_OK && !pirls_outputs_finite(
            mu, grad_eta, w_fisher, w_hessian, w_solver, dev))
        pirls_refuse(&status, PIRLS_FINAL_OUTPUT);
"#
    )
}

#[cfg(target_os = "linux")]
fn bernoulli_probit_body(curvature: CurvatureMode) -> String {
    let tag = curvature_tag(curvature);
    format!(
        r#"{tag}    double mu = 0.0, grad_eta = 0.0, w_fisher = 0.0;
    double w_hessian = 0.0, w_solver = 0.0, dev = 0.0;
    if (!isfinite(eta_i)) pirls_refuse(&status, PIRLS_ETA_DOMAIN);
    if (status == PIRLS_OK && !(isfinite(wp) && wp >= 0.0))
        pirls_refuse(&status, PIRLS_PRIOR_WEIGHT);
    if (status == PIRLS_OK && wp > 0.0
            && !(isfinite(y_i) && y_i >= 0.0 && y_i <= 1.0))
        pirls_refuse(&status, PIRLS_RESPONSE);
    double dmu_deta = 0.0, d2mu_deta2 = 0.0, v = 0.0;
    if (status == PIRLS_OK) {{
        mu = std_norm_cdf(eta_i);
        dmu_deta = std_norm_pdf(eta_i);
        d2mu_deta2 = -eta_i * dmu_deta;
        if (!(isfinite(mu) && mu > 0.0 && mu < 1.0
                && isfinite(dmu_deta) && dmu_deta > 0.0
                && isfinite(d2mu_deta2)))
            pirls_refuse(&status, PIRLS_INVERSE_LINK);
    }}
    if (status == PIRLS_OK && wp > 0.0) {{
        v = mu * (1.0 - mu);
        double fisher_per_prior = dmu_deta * dmu_deta / v;
        w_fisher = wp * fisher_per_prior;
        if (!(isfinite(v) && v > 0.0 && isfinite(fisher_per_prior)
                && fisher_per_prior > 0.0 && isfinite(w_fisher) && w_fisher > 0.0))
            pirls_refuse(&status, PIRLS_FISHER_WEIGHT);
        double resid = y_i - mu;
#ifdef PIRLS_CURVATURE_OBSERVED
        double bracket = d2mu_deta2 / v
            - (dmu_deta * dmu_deta) * (1.0 - 2.0 * mu) / (v * v);
        w_hessian = w_fisher - wp * resid * bracket;
#else
        w_hessian = w_fisher;
#endif
        if (!isfinite(w_hessian)) pirls_refuse(&status, PIRLS_OBSERVED_WEIGHT);
        w_solver = w_hessian;
        grad_eta = wp * resid * dmu_deta / v;
        if (!isfinite(grad_eta)) pirls_refuse(&status, PIRLS_GRADIENT);
        dev = bernoulli_deviance(y_i, mu, wp);
        if (!isfinite(dev)) pirls_refuse(&status, PIRLS_DEVIANCE);
    }}
    if (status == PIRLS_OK && !pirls_outputs_finite(
            mu, grad_eta, w_fisher, w_hessian, w_solver, dev))
        pirls_refuse(&status, PIRLS_FINAL_OUTPUT);
"#
    )
}

#[cfg(target_os = "linux")]
fn bernoulli_cloglog_body(curvature: CurvatureMode) -> String {
    let tag = curvature_tag(curvature);
    format!(
        r#"{tag}    double mu = 0.0, grad_eta = 0.0, w_fisher = 0.0;
    double w_hessian = 0.0, w_solver = 0.0, dev = 0.0;
    if (!isfinite(eta_i)) pirls_refuse(&status, PIRLS_ETA_DOMAIN);
    if (status == PIRLS_OK && !(isfinite(wp) && wp >= 0.0))
        pirls_refuse(&status, PIRLS_PRIOR_WEIGHT);
    if (status == PIRLS_OK && wp > 0.0
            && !(isfinite(y_i) && y_i >= 0.0 && y_i <= 1.0))
        pirls_refuse(&status, PIRLS_RESPONSE);
    double inner = 0.0, dmu_deta = 0.0, d2mu_deta2 = 0.0, v = 0.0;
    if (status == PIRLS_OK) {{
        inner = exp(eta_i);
        double complement = exp(-inner);
        mu = -expm1(-inner);
        dmu_deta = inner * complement;
        d2mu_deta2 = dmu_deta * (1.0 - inner);
        if (!(isfinite(mu) && mu > 0.0 && mu < 1.0
                && isfinite(dmu_deta) && dmu_deta > 0.0
                && isfinite(d2mu_deta2)))
            pirls_refuse(&status, PIRLS_INVERSE_LINK);
    }}
    if (status == PIRLS_OK && wp > 0.0) {{
        v = mu * (1.0 - mu);
        double fisher_per_prior = dmu_deta * dmu_deta / v;
        w_fisher = wp * fisher_per_prior;
        if (!(isfinite(v) && v > 0.0 && isfinite(fisher_per_prior)
                && fisher_per_prior > 0.0 && isfinite(w_fisher) && w_fisher > 0.0))
            pirls_refuse(&status, PIRLS_FISHER_WEIGHT);
        double resid = y_i - mu;
#ifdef PIRLS_CURVATURE_OBSERVED
        double bracket = d2mu_deta2 / v
            - (dmu_deta * dmu_deta) * (1.0 - 2.0 * mu) / (v * v);
        w_hessian = w_fisher - wp * resid * bracket;
#else
        w_hessian = w_fisher;
#endif
        if (!isfinite(w_hessian)) pirls_refuse(&status, PIRLS_OBSERVED_WEIGHT);
        w_solver = w_hessian;
        grad_eta = wp * resid * dmu_deta / v;
        if (!isfinite(grad_eta)) pirls_refuse(&status, PIRLS_GRADIENT);
        dev = bernoulli_deviance(y_i, mu, wp);
        if (!isfinite(dev)) pirls_refuse(&status, PIRLS_DEVIANCE);
    }}
    if (status == PIRLS_OK && !pirls_outputs_finite(
            mu, grad_eta, w_fisher, w_hessian, w_solver, dev))
        pirls_refuse(&status, PIRLS_FINAL_OUTPUT);
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
/// `mu` and `w_hessian` stores, reducing
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
    unsigned int status = PIRLS_OK;
    double eta_i = eta[i];
    double y_i = y[i];
    double wp = prior_w[i];
{body}
    if (status == PIRLS_OK) {{
        grad_eta_out[i] = grad_eta;
        w_solver_out[i] = w_solver;
        deviance_out[i] = dev;
    }}
    status_out[i] = status;
}}
"#,
        prolog = common_device_prolog(),
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
/// `objective_out[k]` (double-precision atomic add) and writes the exact row
/// code to `status_out[k*n+i]`.
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
    // The body uses the name `eta_i` for the exact trial linear predictor.
    // For the ladder we substitute `eta[i] + alpha * xd[i]` as the trial eta,
    // so we define `eta_i` before the body runs. The body's own local variable
    // For GaussianIdentity the body reads `eta_i` directly as `mu`; log-link
    // bodies certify the shared solver interval and never modify the value.
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
    unsigned int status = PIRLS_OK;
    double alpha = PIRLS_ALPHAS[k];
    double eta_i = eta[i] + alpha * xd[i];
    double y_i = y[i];
    double wp = prior_w[i];
{body}
    if (status == PIRLS_OK) atomicAdd(&objective_out[k], dev);
    status_out[k * n + i] = status;
}}
"#,
        prolog = common_device_prolog(),
        alphas = ALPHA_LADDER_CUDA_ARRAY,
    )
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "pirls_row_tests.rs"]
mod pirls_row_tests;
