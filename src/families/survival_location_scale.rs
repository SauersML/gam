use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, BlockwiseFitResult, CustomFamily, FamilyEvaluation,
    ParameterBlockSpec, ParameterBlockState, fit_custom_family,
};
use crate::faer_ndarray::{FaerSvd, fast_xt_diag_x};
use crate::families::sigma_link::{
    bounded_sigma_derivs_up_to_third, bounded_sigma_derivs_up_to_third_scalar,
};
use crate::matrix::{DesignMatrix, SymmetricMatrix, xt_diag_x_symmetric};
use crate::mixture_link::{
    inverse_link_jet_for_inverse_link, inverse_link_pdf_third_derivative_for_inverse_link,
};
use crate::pirls::LinearInequalityConstraints;
use crate::probability::{normal_cdf, normal_pdf};
use crate::types::{InverseLink, LinkFunction};
use ndarray::{Array1, Array2, s};

const MIN_PROB: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualDistribution {
    Gaussian,
    Gumbel,
    Logistic,
}

pub trait ResidualDistributionOps {
    fn cdf(&self, z: f64) -> f64;
    fn pdf(&self, z: f64) -> f64;
    fn pdf_derivative(&self, z: f64) -> f64;
    fn pdf_second_derivative(&self, z: f64) -> f64;
    fn pdf_third_derivative(&self, z: f64) -> f64;
}

impl ResidualDistributionOps for ResidualDistribution {
    fn cdf(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => normal_cdf(z),
            ResidualDistribution::Gumbel => {
                // F(z)=1-exp(-exp(z))
                if z == f64::INFINITY {
                    return 1.0;
                }
                if z == f64::NEG_INFINITY {
                    return 0.0;
                }
                let ez = z.exp();
                1.0 - (-ez).exp()
            }
            ResidualDistribution::Logistic => {
                if z == f64::INFINITY {
                    1.0
                } else if z == f64::NEG_INFINITY {
                    0.0
                } else if z >= 0.0 {
                    let e = (-z).exp();
                    1.0 / (1.0 + e)
                } else {
                    let e = z.exp();
                    e / (1.0 + e)
                }
            }
        }
    }

    fn pdf(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => normal_pdf(z),
            ResidualDistribution::Gumbel => {
                if z == f64::INFINITY || z == f64::NEG_INFINITY {
                    return 0.0;
                }
                let ez = z.exp();
                ez * (-ez).exp()
            }
            ResidualDistribution::Logistic => {
                let s = self.cdf(z);
                s * (1.0 - s)
            }
        }
    }

    fn pdf_derivative(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => -z * normal_pdf(z),
            ResidualDistribution::Gumbel => {
                if z == f64::INFINITY || z == f64::NEG_INFINITY {
                    return 0.0;
                }
                let ez = z.exp();
                let f = ez * (-ez).exp();
                f * (1.0 - ez)
            }
            ResidualDistribution::Logistic => {
                let s = self.cdf(z);
                let f = s * (1.0 - s);
                f * (1.0 - 2.0 * s)
            }
        }
    }

    fn pdf_second_derivative(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => {
                let f = normal_pdf(z);
                (z * z - 1.0) * f
            }
            ResidualDistribution::Gumbel => {
                if z == f64::INFINITY || z == f64::NEG_INFINITY {
                    return 0.0;
                }
                let ez = z.exp();
                let f = ez * (-ez).exp();
                f * (1.0 - 3.0 * ez + ez * ez)
            }
            ResidualDistribution::Logistic => {
                let s = self.cdf(z);
                let f = s * (1.0 - s);
                f * (1.0 - 6.0 * s + 6.0 * s * s)
            }
        }
    }

    fn pdf_third_derivative(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => {
                let f = normal_pdf(z);
                -(z * z * z - 3.0 * z) * f
            }
            ResidualDistribution::Gumbel => {
                if z == f64::INFINITY || z == f64::NEG_INFINITY {
                    return 0.0;
                }
                let ez = z.exp();
                let f = ez * (-ez).exp();
                f * (1.0 - 7.0 * ez + 6.0 * ez * ez - ez * ez * ez)
            }
            ResidualDistribution::Logistic => {
                let s = self.cdf(z);
                let f = s * (1.0 - s);
                f * (1.0 - 14.0 * s + 36.0 * s * s - 24.0 * s * s * s)
            }
        }
    }
}

#[inline]
fn residual_distribution_link(distribution: ResidualDistribution) -> LinkFunction {
    match distribution {
        ResidualDistribution::Gaussian => LinkFunction::Probit,
        ResidualDistribution::Gumbel => LinkFunction::CLogLog,
        ResidualDistribution::Logistic => LinkFunction::Logit,
    }
}

#[inline]
pub fn residual_distribution_inverse_link(distribution: ResidualDistribution) -> InverseLink {
    InverseLink::Standard(residual_distribution_link(distribution))
}

#[derive(Clone)]
pub struct TimeBlockInput {
    pub design_entry: Array2<f64>,
    pub design_exit: Array2<f64>,
    pub design_derivative_exit: Array2<f64>,
    pub offset_entry: Array1<f64>,
    pub offset_exit: Array1<f64>,
    pub derivative_offset_exit: Array1<f64>,
    pub penalties: Vec<Array2<f64>>,
    pub initial_log_lambdas: Option<Array1<f64>>,
    pub initial_beta: Option<Array1<f64>>,
}

#[derive(Clone)]
pub struct CovariateBlockInput {
    pub design: DesignMatrix,
    pub offset: Array1<f64>,
    pub penalties: Vec<Array2<f64>>,
    pub initial_log_lambdas: Option<Array1<f64>>,
    pub initial_beta: Option<Array1<f64>>,
}

#[derive(Clone)]
pub struct LinkWiggleBlockInput {
    pub design: DesignMatrix,
    pub penalties: Vec<Array2<f64>>,
    pub initial_log_lambdas: Option<Array1<f64>>,
    pub initial_beta: Option<Array1<f64>>,
}

#[derive(Clone)]
pub struct SurvivalLocationScaleSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<f64>,
    pub weights: Array1<f64>,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub inverse_link: InverseLink,
    pub derivative_guard: f64,
    pub derivative_softness: f64,
    /// Optional anchor time for identifiability of h(t).
    ///
    /// If `None`, the model anchors at the earliest observed entry time.
    /// If `Some(t_anchor)`, the nearest observed entry-time row is used.
    pub time_anchor: Option<f64>,
    pub max_iter: usize,
    pub tol: f64,
    pub time_block: TimeBlockInput,
    pub threshold_block: CovariateBlockInput,
    pub log_sigma_block: CovariateBlockInput,
    pub link_wiggle_block: Option<LinkWiggleBlockInput>,
}

#[derive(Clone)]
pub struct SurvivalLocationScaleFitResult {
    pub beta_time: Array1<f64>,
    pub beta_threshold: Array1<f64>,
    pub beta_log_sigma: Array1<f64>,
    pub beta_link_wiggle: Option<Array1<f64>>,
    pub lambdas_time: Array1<f64>,
    pub lambdas_threshold: Array1<f64>,
    pub lambdas_log_sigma: Array1<f64>,
    pub lambdas_link_wiggle: Option<Array1<f64>>,
    pub log_likelihood: f64,
    pub penalized_objective: f64,
    pub iterations: usize,
    pub final_grad_norm: f64,
    pub converged: bool,
    pub covariance_conditional: Option<Array2<f64>>,
}

#[derive(Clone)]
pub struct SurvivalLocationScalePredictInput {
    pub x_time_exit: Array2<f64>,
    pub eta_time_offset_exit: Array1<f64>,
    pub x_threshold: DesignMatrix,
    pub eta_threshold_offset: Array1<f64>,
    pub x_log_sigma: DesignMatrix,
    pub eta_log_sigma_offset: Array1<f64>,
    pub x_link_wiggle: Option<DesignMatrix>,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub inverse_link: InverseLink,
}

#[derive(Clone, Debug)]
pub struct SurvivalLocationScalePredictResult {
    pub eta: Array1<f64>,
    pub survival_prob: Array1<f64>,
}

#[derive(Clone)]
pub struct SurvivalLocationScalePredictUncertaintyResult {
    pub eta: Array1<f64>,
    pub survival_prob: Array1<f64>,
    pub eta_standard_error: Array1<f64>,
    pub response_standard_error: Option<Array1<f64>>,
}

struct SurvivalLocationScaleFamily {
    n: usize,
    y: Array1<f64>,
    w: Array1<f64>,
    sigma_min: f64,
    sigma_max: f64,
    inverse_link: InverseLink,
    derivative_guard: f64,
    derivative_softness: f64,
    x_time_entry: Array2<f64>,
    x_time_exit: Array2<f64>,
    x_time_deriv: Array2<f64>,
    offset_time_deriv: Array1<f64>,
    x_threshold: DesignMatrix,
    x_log_sigma: DesignMatrix,
    x_link_wiggle: Option<DesignMatrix>,
}

#[derive(Clone, Copy)]
struct SurvivalPredictorState {
    h0: f64,
    h1: f64,
    d_raw: f64,
    q: f64,
}

#[derive(Clone, Copy)]
struct SurvivalRowDerivatives {
    ll: f64,
    d1_q: f64,
    d2_q: f64,
    d3_q: f64,
    grad_time_eta_h0: f64,
    grad_time_eta_h1: f64,
    grad_time_eta_d: f64,
    h_time_h0: f64,
    h_time_h1: f64,
    h_time_d: f64,
    d_h_h0: f64,
    d_h_h1: f64,
    d_h_d: f64,
}

impl SurvivalLocationScaleFamily {
    const BLOCK_TIME: usize = 0;
    const BLOCK_THRESHOLD: usize = 1;
    const BLOCK_LOG_SIGMA: usize = 2;
    const BLOCK_LINK_WIGGLE: usize = 3;

    #[inline]
    fn expected_blocks(&self) -> usize {
        if self.x_link_wiggle.is_some() { 4 } else { 3 }
    }

    /// Hazard-like survival ratio and its first derivative.
    ///
    /// Let `F` be the CDF, `f = F'` the PDF, and `S = 1 - F` the survival
    /// function so `S' = -f`.
    ///
    /// Define `r = f / S`. By quotient rule:
    /// `r' = (f' S - f S') / S^2`.
    /// Since `S' = -f`, this becomes:
    /// `r' = f'/S + f^2/S^2 = f'/S + r^2`.
    ///
    /// Sign note: the `f'/S` term is strictly additive. A minus here is wrong.
    fn survival_ratio_first_derivative(f: f64, fp: f64, s: f64) -> (f64, f64) {
        let r = f / s;
        let dr = (r * r) + fp / s;
        (r, dr)
    }

    /// Second derivative of the survival ratio `r = f/S`.
    ///
    /// Starting from `r' = f'/S + r^2`:
    /// `r'' = d/du[f'/S] + 2 r r'`.
    /// With `S' = -f`, we get:
    /// `d/du[f'/S] = f''/S + f' f / S^2`.
    /// Therefore:
    /// `r'' = 2 r r' + f''/S + f' f / S^2`.
    ///
    /// Equivalent expanded form:
    /// `r'' = f''/S + 3 f f' / S^2 + 2 f^3 / S^3`.
    fn survival_ratio_second_derivative(r: f64, dr: f64, f: f64, fp: f64, fpp: f64, s: f64) -> f64 {
        (2.0 * r * dr) + (fpp / s + fp * f / (s * s))
    }

    /// Clamp-aware log-pdf and its first/second/third derivatives.
    ///
    /// Let `L(u) = log f(u)` on the unclamped branch. The exact derivatives are:
    ///
    /// `L'   = f'/f`
    ///
    /// `L''  = d/du(f'/f)
    ///       = (f'' f - (f')²) / f²
    ///       = f''/f - (f'/f)²`
    ///
    /// For the third derivative, differentiate `L'' = f''/f - (f'/f)^2`:
    ///
    /// `d/du[f''/f]   = f'''/f - f'f''/f²`
    ///
    /// `d/du[(f'/f)²] = 2(f'/f)(f''/f - (f'/f)²)
    ///                = 2f'f''/f² - 2(f')³/f³`
    ///
    /// so
    ///
    /// `L''' = f'''/f - 3 f'f''/f² + 2(f')³/f³`.
    ///
    /// This is the exact `d³(log f)/du³` term used in the survival exact-Newton
    /// Hessian directional derivative. If it is dropped, the event contribution
    /// to `d³ℓ/dq³` is wrong, which then corrupts the block Hessian drift
    /// `D H[u]`.
    ///
    /// The objective actually uses `log(max(f, MIN_PROB))`, not `log f`, so once
    /// `f <= MIN_PROB` the active branch is constant and all derivatives must be
    /// zero. That is why the clamped branch returns `(log MIN_PROB, 0, 0, 0)`.
    fn clamped_log_pdf_with_derivatives(
        f: f64,
        fp: f64,
        fpp: f64,
        fppp: f64,
    ) -> (f64, f64, f64, f64) {
        if f <= MIN_PROB {
            (MIN_PROB.ln(), 0.0, 0.0, 0.0)
        } else {
            let d1 = fp / f;
            let d2 = fpp / f - d1 * d1;
            let d3 = fppp / f - 3.0 * fp * fpp / (f * f) + 2.0 * fp * fp * fp / (f * f * f);
            (f.ln(), d1, d2, d3)
        }
    }

    /// Clamp-aware survival value and derivatives of `-log(clamp(S, MIN_PROB, 1))`.
    ///
    /// Returns `(S_clamped, r, dr, ddr)` where:
    /// - `r   = d/du[-log S_clamped]`
    /// - `dr  = d²/du²[-log S_clamped]`
    /// - `ddr = d³/du³[-log S_clamped]`
    ///
    /// If `S` is clamped at either bound (`S <= MIN_PROB` or `S >= 1`), these
    /// derivatives are all zero because the clamped log term is locally constant.
    fn clamped_survival_neglog_derivatives(
        raw_s: f64,
        f: f64,
        fp: f64,
        fpp: f64,
    ) -> (f64, f64, f64, f64) {
        let s = raw_s.clamp(MIN_PROB, 1.0);
        if raw_s <= MIN_PROB || raw_s >= 1.0 {
            (s, 0.0, 0.0, 0.0)
        } else {
            let (r, dr) = Self::survival_ratio_first_derivative(f, fp, s);
            let ddr = Self::survival_ratio_second_derivative(r, dr, f, fp, fpp, s);
            (s, r, dr, ddr)
        }
    }

    /// Clamp-aware `log(max(x, floor))` value and first three derivatives.
    ///
    /// Once `x <= floor`, the active branch is constant so every derivative is
    /// zero. This keeps the exact-Newton derivatives aligned with the objective.
    fn clamped_log_with_derivatives(raw_x: f64, floor: f64) -> (f64, f64, f64, f64) {
        let x = raw_x.max(floor);
        if raw_x <= floor {
            (x.ln(), 0.0, 0.0, 0.0)
        } else {
            let inv = 1.0 / x;
            (x.ln(), inv, -inv * inv, 2.0 * inv * inv * inv)
        }
    }

    fn row_predictor_state(
        &self,
        h0: f64,
        h1: f64,
        d_raw: f64,
        eta_t: f64,
        sigma: f64,
        eta_w: Option<f64>,
    ) -> SurvivalPredictorState {
        let base = -eta_t / sigma.max(1e-12);
        SurvivalPredictorState {
            h0,
            h1,
            d_raw,
            q: base + eta_w.unwrap_or(0.0),
        }
    }

    fn row_derivatives(
        &self,
        row: usize,
        state: SurvivalPredictorState,
    ) -> Result<Option<SurvivalRowDerivatives>, String> {
        let w = self.w[row];
        if w <= 0.0 {
            return Ok(None);
        }
        let d = self.y[row].clamp(0.0, 1.0);
        let u0 = -state.h0 + state.q;
        let u1 = -state.h1 + state.q;
        let j0 = inverse_link_jet_for_inverse_link(&self.inverse_link, u0)
            .map_err(|e| format!("inverse link evaluation failed at row {row} entry: {e}"))?;
        let j1 = inverse_link_jet_for_inverse_link(&self.inverse_link, u1)
            .map_err(|e| format!("inverse link evaluation failed at row {row} exit: {e}"))?;
        let (s0, r0, dr0, ddr0) =
            Self::clamped_survival_neglog_derivatives(1.0 - j0.mu, j0.d1, j0.d2, j0.d3);
        let (s1, r1, dr1, ddr1) =
            Self::clamped_survival_neglog_derivatives(1.0 - j1.mu, j1.d1, j1.d2, j1.d3);
        let fppp1 = inverse_link_pdf_third_derivative_for_inverse_link(&self.inverse_link, u1)
            .map_err(|e| {
                format!("inverse link third-derivative evaluation failed at row {row} exit: {e}")
            })?;
        let (log_phi1, dlogphi1, d2logphi1, d3logphi1) =
            Self::clamped_log_pdf_with_derivatives(j1.d1, j1.d2, j1.d3, fppp1);

        let guard = self.derivative_guard;
        let soft = self.derivative_softness.max(0.0);
        let g = state.d_raw;
        let (log_g_safe, d_log_g, d2_log_g, d3_log_g) =
            Self::clamped_log_with_derivatives(g + soft, 1e-12);
        if !g.is_finite() {
            return Err(format!(
                "survival probit-location-scale non-finite d_eta/dt at row {row}: {g}"
            ));
        }
        if guard > 0.0 && g <= guard {
            return Err(format!(
                "survival probit-location-scale monotonicity violated at row {row}: d_eta/dt={g:.3e} <= guard={:.3e}",
                guard
            ));
        }

        let d1_q = w * (r0 + d * dlogphi1 + (1.0 - d) * (-r1));
        let d2_q = w * (dr0 + d * d2logphi1 + (1.0 - d) * (-dr1));
        let d3_q = w * (ddr0 + d * d3logphi1 + (1.0 - d) * (-ddr1));

        Ok(Some(SurvivalRowDerivatives {
            ll: w * (d * (log_phi1 + log_g_safe) + (1.0 - d) * s1.ln() - s0.ln()),
            d1_q,
            d2_q,
            d3_q,
            grad_time_eta_h0: -w * r0,
            grad_time_eta_h1: -w * (d * dlogphi1 + (1.0 - d) * (-r1)),
            grad_time_eta_d: w * d * d_log_g,
            h_time_h0: -w * dr0,
            h_time_h1: -w * (d * d2logphi1 + (1.0 - d) * (-dr1)),
            h_time_d: -w * d * d2_log_g,
            d_h_h0: w * ddr0,
            d_h_h1: w * d * d3logphi1 - w * (1.0 - d) * ddr1,
            d_h_d: -w * d * d3_log_g,
        }))
    }
}

fn validate_cov_block(name: &str, n: usize, b: &CovariateBlockInput) -> Result<(), String> {
    if b.design.nrows() != n {
        return Err(format!(
            "{name} design row mismatch: got {}, expected {n}",
            b.design.nrows()
        ));
    }
    if b.offset.len() != n {
        return Err(format!(
            "{name} offset length mismatch: got {}, expected {n}",
            b.offset.len()
        ));
    }
    let p = b.design.ncols();
    if let Some(beta0) = &b.initial_beta
        && beta0.len() != p
    {
        return Err(format!(
            "{name} initial_beta length mismatch: got {}, expected {p}",
            beta0.len()
        ));
    }
    let k = b.penalties.len();
    if let Some(rho0) = &b.initial_log_lambdas
        && rho0.len() != k
    {
        return Err(format!(
            "{name} initial_log_lambdas length mismatch: got {}, expected {k}",
            rho0.len()
        ));
    }
    for (idx, s) in b.penalties.iter().enumerate() {
        let (r, c) = s.dim();
        if r != p || c != p {
            return Err(format!("{name} penalty {idx} must be {p}x{p}, got {r}x{c}"));
        }
    }
    Ok(())
}

fn validate_wiggle_block(n: usize, b: &LinkWiggleBlockInput) -> Result<(), String> {
    if b.design.nrows() != n {
        return Err(format!(
            "link_wiggle_block design row mismatch: got {}, expected {n}",
            b.design.nrows()
        ));
    }
    let p = b.design.ncols();
    if let Some(beta0) = &b.initial_beta
        && beta0.len() != p
    {
        return Err(format!(
            "link_wiggle_block initial_beta length mismatch: got {}, expected {p}",
            beta0.len()
        ));
    }
    let k = b.penalties.len();
    if let Some(rho0) = &b.initial_log_lambdas
        && rho0.len() != k
    {
        return Err(format!(
            "link_wiggle_block initial_log_lambdas length mismatch: got {}, expected {k}",
            rho0.len()
        ));
    }
    for (idx, s) in b.penalties.iter().enumerate() {
        let (r, c) = s.dim();
        if r != p || c != p {
            return Err(format!(
                "link_wiggle_block penalty {idx} must be {p}x{p}, got {r}x{c}"
            ));
        }
    }
    Ok(())
}

fn validate_time_block(n: usize, b: &TimeBlockInput) -> Result<(), String> {
    if b.design_entry.nrows() != n
        || b.design_exit.nrows() != n
        || b.design_derivative_exit.nrows() != n
        || b.offset_entry.len() != n
        || b.offset_exit.len() != n
        || b.derivative_offset_exit.len() != n
    {
        return Err("time_block input size mismatch".to_string());
    }
    let p = b.design_exit.ncols();
    if b.design_entry.ncols() != p || b.design_derivative_exit.ncols() != p {
        return Err("time_block design column mismatch across entry/exit/derivative".to_string());
    }
    if let Some(beta0) = &b.initial_beta
        && beta0.len() != p
    {
        return Err(format!(
            "time_block initial_beta length mismatch: got {}, expected {p}",
            beta0.len()
        ));
    }
    let k = b.penalties.len();
    if let Some(rho0) = &b.initial_log_lambdas
        && rho0.len() != k
    {
        return Err(format!(
            "time_block initial_log_lambdas length mismatch: got {}, expected {k}",
            rho0.len()
        ));
    }
    for (idx, s) in b.penalties.iter().enumerate() {
        let (r, c) = s.dim();
        if r != p || c != p {
            return Err(format!(
                "time_block penalty {idx} must be {p}x{p}, got {r}x{c}"
            ));
        }
    }
    Ok(())
}

fn stack_time_design(input: &TimeBlockInput) -> Array2<f64> {
    let n = input.design_exit.nrows();
    let p = input.design_exit.ncols();
    let mut out = Array2::<f64>::zeros((3 * n, p));
    out.slice_mut(s![0..n, ..]).assign(&input.design_entry);
    out.slice_mut(s![n..2 * n, ..]).assign(&input.design_exit);
    out.slice_mut(s![2 * n..3 * n, ..])
        .assign(&input.design_derivative_exit);
    out
}

fn stack_time_offset(input: &TimeBlockInput) -> Array1<f64> {
    let n = input.offset_exit.len();
    let mut out = Array1::<f64>::zeros(3 * n);
    out.slice_mut(s![0..n]).assign(&input.offset_entry);
    out.slice_mut(s![n..2 * n]).assign(&input.offset_exit);
    out.slice_mut(s![2 * n..3 * n])
        .assign(&input.derivative_offset_exit);
    out
}

#[derive(Clone)]
struct TimeIdentifiabilityTransform {
    z: Array2<f64>,
}

#[derive(Clone)]
struct TimeBlockPrepared {
    design_entry: Array2<f64>,
    design_exit: Array2<f64>,
    design_derivative_exit: Array2<f64>,
    penalties: Vec<Array2<f64>>,
    initial_beta: Option<Array1<f64>>,
    transform: TimeIdentifiabilityTransform,
}

fn prepare_identified_time_block(
    input: &TimeBlockInput,
    anchor_row: usize,
) -> Result<TimeBlockPrepared, String> {
    let p = input.design_exit.ncols();
    if p < 2 {
        return Err(format!(
            "time_block needs at least 2 columns for identifiability, got {p}"
        ));
    }
    if anchor_row >= input.design_exit.nrows() {
        return Err(format!(
            "time_block anchor row out of bounds: got {anchor_row}, nrows={}",
            input.design_exit.nrows()
        ));
    }

    // Identifiability: enforce h(t_anchor)=0 by constraining c^T beta = 0,
    // where c is the time basis row at anchor time. Reparameterize beta = Z theta
    // with columns of Z spanning null(c^T), so the constraint is exact for all theta.
    let c = input.design_entry.row(anchor_row).to_owned();
    let mut c_mat = Array2::<f64>::zeros((p, 1));
    c_mat.column_mut(0).assign(&c);
    let (u_opt, singular_values, _) = c_mat
        .svd(true, false)
        .map_err(|e| format!("time_block identifiability SVD failed: {e}"))?;
    let u = u_opt.ok_or_else(|| "time_block identifiability SVD returned no U".to_string())?;
    let max_sigma = singular_values
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let tol = (p.max(1) as f64) * f64::EPSILON * max_sigma.max(1.0);
    let rank = singular_values
        .iter()
        .filter(|&&sigma| sigma.abs() > tol)
        .count();
    if rank >= p {
        return Err(
            "time_block identifiability constraint removed all columns; add richer time basis"
                .to_string(),
        );
    }
    let z = if rank == 0 {
        Array2::<f64>::eye(p)
    } else {
        u.slice(s![.., rank..]).to_owned()
    };
    let design_entry = input.design_entry.dot(&z);
    let design_exit = input.design_exit.dot(&z);
    let design_derivative_exit = input.design_derivative_exit.dot(&z);
    let penalties = input
        .penalties
        .iter()
        .map(|s| z.t().dot(s).dot(&z))
        .collect::<Vec<_>>();
    let initial_beta = input.initial_beta.as_ref().map(|b| z.t().dot(b));

    Ok(TimeBlockPrepared {
        design_entry,
        design_exit,
        design_derivative_exit,
        penalties,
        initial_beta,
        transform: TimeIdentifiabilityTransform { z },
    })
}

fn initial_log_lambdas(
    penalties: &[Array2<f64>],
    rho0: Option<Array1<f64>>,
) -> Result<Array1<f64>, String> {
    let k = penalties.len();
    let rho = rho0.unwrap_or_else(|| Array1::zeros(k));
    if rho.len() != k {
        return Err(format!(
            "initial_log_lambdas mismatch: got {}, expected {k}",
            rho.len()
        ));
    }
    Ok(rho)
}

fn select_anchor_row(age_entry: &Array1<f64>, time_anchor: Option<f64>) -> Result<usize, String> {
    if age_entry.is_empty() {
        return Err("select_anchor_row: empty age_entry".to_string());
    }
    match time_anchor {
        None => age_entry
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .ok_or_else(|| "select_anchor_row: failed to select earliest entry".to_string()),
        Some(t_anchor) => {
            if !t_anchor.is_finite() {
                return Err(format!(
                    "fit_survival_location_scale: non-finite time_anchor {t_anchor}"
                ));
            }
            age_entry
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    let da = (a.1 - t_anchor).abs();
                    let db = (b.1 - t_anchor).abs();
                    da.total_cmp(&db)
                })
                .map(|(i, _)| i)
                .ok_or_else(|| "select_anchor_row: failed to select nearest entry".to_string())
        }
    }
}

fn design_times_dense(design: &DesignMatrix, rhs: &Array2<f64>) -> Result<Array2<f64>, String> {
    if design.ncols() != rhs.nrows() {
        return Err(format!(
            "design_times_dense shape mismatch: design is {}x{}, rhs is {}x{}",
            design.nrows(),
            design.ncols(),
            rhs.nrows(),
            rhs.ncols()
        ));
    }
    match design {
        DesignMatrix::Dense(x) => Ok(x.dot(rhs)),
        DesignMatrix::Sparse(_) => {
            let n = design.nrows();
            let q = rhs.ncols();
            let mut out = Array2::<f64>::zeros((n, q));
            for j in 0..q {
                let col = rhs.column(j).to_owned();
                out.column_mut(j)
                    .assign(&design.matrix_vector_multiply(&col));
            }
            Ok(out)
        }
    }
}

fn rowwise_dot_design_with_dense(
    design: &DesignMatrix,
    row_values: &Array2<f64>,
) -> Result<Array1<f64>, String> {
    if design.nrows() != row_values.nrows() || design.ncols() != row_values.ncols() {
        return Err(format!(
            "rowwise_dot_design_with_dense shape mismatch: design is {}x{}, row_values is {}x{}",
            design.nrows(),
            design.ncols(),
            row_values.nrows(),
            row_values.ncols()
        ));
    }
    match design {
        DesignMatrix::Dense(x) => Ok(Array1::from_iter(
            (0..x.nrows()).map(|i| x.row(i).dot(&row_values.row(i))),
        )),
        DesignMatrix::Sparse(xs) => {
            let csr = xs.to_csr_arc().ok_or_else(|| {
                "rowwise_dot_design_with_dense: failed to obtain CSR view".to_string()
            })?;
            let sym = csr.symbolic();
            let row_ptr = sym.row_ptr();
            let col_idx = sym.col_idx();
            let vals = csr.val();
            let mut out = Array1::<f64>::zeros(xs.nrows());
            for i in 0..xs.nrows() {
                let start = row_ptr[i];
                let end = row_ptr[i + 1];
                let mut acc = 0.0_f64;
                for ptr in start..end {
                    let j = col_idx[ptr];
                    acc += vals[ptr] * row_values[[i, j]];
                }
                out[i] = acc;
            }
            Ok(out)
        }
    }
}

fn rowwise_cross_quadratic_dense_design(
    left_dense: &Array2<f64>,
    middle: &Array2<f64>,
    right: &DesignMatrix,
) -> Result<Array1<f64>, String> {
    if left_dense.ncols() != middle.nrows() {
        return Err(format!(
            "rowwise_cross_quadratic_dense_design shape mismatch: left is {}x{}, middle is {}x{}",
            left_dense.nrows(),
            left_dense.ncols(),
            middle.nrows(),
            middle.ncols()
        ));
    }
    let left_middle = left_dense.dot(middle);
    rowwise_dot_design_with_dense(right, &left_middle)
}

fn rowwise_cross_quadratic_design(
    left: &DesignMatrix,
    middle: &Array2<f64>,
    right: &DesignMatrix,
) -> Result<Array1<f64>, String> {
    let left_middle = design_times_dense(left, middle)?;
    rowwise_dot_design_with_dense(right, &left_middle)
}

fn validate_predict_inverse_link(inverse_link: &InverseLink) -> Result<(), String> {
    match inverse_link {
        InverseLink::Standard(LinkFunction::Sas) => Err(
            "prediction requires explicit SasLinkState; state-less Standard(Sas) is unsupported"
                .to_string(),
        ),
        InverseLink::Standard(LinkFunction::BetaLogistic) => Err(
            "prediction requires explicit Beta-Logistic link state; state-less Standard(BetaLogistic) is unsupported"
                .to_string(),
        ),
        _ => Ok(()),
    }
}

fn inverse_link_failure_prob_checked(inverse_link: &InverseLink, eta: f64) -> Result<f64, String> {
    inverse_link_jet_for_inverse_link(inverse_link, eta)
        .map(|j| j.mu.clamp(0.0, 1.0))
        .map_err(|e| format!("inverse link prediction failed at eta={eta}: {e}"))
}

fn inverse_link_survival_prob_checked(inverse_link: &InverseLink, eta: f64) -> Result<f64, String> {
    inverse_link_failure_prob_checked(inverse_link, eta).map(|f| (1.0 - f).clamp(0.0, 1.0))
}

fn inverse_link_survival_prob_value(inverse_link: &InverseLink, eta: f64) -> f64 {
    match inverse_link {
        InverseLink::Standard(LinkFunction::Probit) => (1.0 - normal_cdf(eta)).clamp(0.0, 1.0),
        InverseLink::Standard(LinkFunction::Logit) => {
            let e = eta.clamp(-700.0, 700.0);
            (1.0 / (1.0 + e.exp())).clamp(0.0, 1.0)
        }
        InverseLink::Standard(LinkFunction::CLogLog) => {
            let e = eta.clamp(-30.0, 30.0);
            (-e.exp()).exp().clamp(0.0, 1.0)
        }
        InverseLink::Standard(LinkFunction::Identity) => (1.0 - eta).clamp(0.0, 1.0),
        InverseLink::Sas(_) | InverseLink::BetaLogistic(_) | InverseLink::Mixture(_) => {
            inverse_link_survival_prob_checked(inverse_link, eta)
                .expect("validated inverse link should evaluate during prediction")
        }
        InverseLink::Standard(LinkFunction::Sas)
        | InverseLink::Standard(LinkFunction::BetaLogistic) => {
            panic!("state-less SAS/Beta-Logistic inverse link is invalid for prediction")
        }
    }
}

struct PredictionLinearPredictors {
    h: Array1<f64>,
    eta_t: Array1<f64>,
    eta_ls: Array1<f64>,
    eta_w: Option<Array1<f64>>,
}

fn prediction_linear_predictors(
    input: &SurvivalLocationScalePredictInput,
    fit: &SurvivalLocationScaleFitResult,
) -> Result<PredictionLinearPredictors, String> {
    validate_predict_inverse_link(&input.inverse_link)?;
    let n = input.x_time_exit.nrows();
    if input.x_time_exit.ncols() != fit.beta_time.len() {
        return Err(format!(
            "predict_survival_location_scale: time design/beta mismatch: {} vs {}",
            input.x_time_exit.ncols(),
            fit.beta_time.len()
        ));
    }
    if input.eta_time_offset_exit.len() != n
        || input.x_threshold.nrows() != n
        || input.eta_threshold_offset.len() != n
        || input.x_log_sigma.nrows() != n
        || input.eta_log_sigma_offset.len() != n
    {
        return Err("predict_survival_location_scale: row mismatch across inputs".to_string());
    }
    if let (Some(xw), Some(beta_w)) = (&input.x_link_wiggle, &fit.beta_link_wiggle) {
        if xw.nrows() != n {
            return Err(format!(
                "predict_survival_location_scale: link-wiggle row mismatch: got {}, expected {n}",
                xw.nrows()
            ));
        }
        if xw.ncols() != beta_w.len() {
            return Err(format!(
                "predict_survival_location_scale: link-wiggle design/beta mismatch: {} vs {}",
                xw.ncols(),
                beta_w.len()
            ));
        }
    } else if input.x_link_wiggle.is_some() || fit.beta_link_wiggle.is_some() {
        return Err(
            "predict_survival_location_scale: link-wiggle metadata is partial; both design and beta must be provided"
                .to_string(),
        );
    }
    Ok(PredictionLinearPredictors {
        h: input.x_time_exit.dot(&fit.beta_time) + &input.eta_time_offset_exit,
        eta_t: input
            .x_threshold
            .matrix_vector_multiply(&fit.beta_threshold)
            + &input.eta_threshold_offset,
        eta_ls: input
            .x_log_sigma
            .matrix_vector_multiply(&fit.beta_log_sigma)
            + &input.eta_log_sigma_offset,
        eta_w: if let (Some(xw), Some(beta_w)) = (&input.x_link_wiggle, &fit.beta_link_wiggle) {
            Some(xw.matrix_vector_multiply(beta_w))
        } else {
            None
        },
    })
}

fn lift_conditional_covariance(
    cov_reduced: &Array2<f64>,
    z: &Array2<f64>,
    p_threshold: usize,
    p_log_sigma: usize,
    p_link_wiggle: usize,
) -> Array2<f64> {
    let p_time_reduced = z.ncols();
    let p_time_full = z.nrows();
    let p_reduced = p_time_reduced + p_threshold + p_log_sigma + p_link_wiggle;
    let p_full = p_time_full + p_threshold + p_log_sigma + p_link_wiggle;
    if cov_reduced.nrows() != p_reduced || cov_reduced.ncols() != p_reduced {
        return cov_reduced.clone();
    }

    let mut t_map = Array2::<f64>::zeros((p_full, p_reduced));
    t_map
        .slice_mut(s![0..p_time_full, 0..p_time_reduced])
        .assign(z);
    for j in 0..p_threshold {
        t_map[[p_time_full + j, p_time_reduced + j]] = 1.0;
    }
    for j in 0..p_log_sigma {
        t_map[[
            p_time_full + p_threshold + j,
            p_time_reduced + p_threshold + j,
        ]] = 1.0;
    }
    for j in 0..p_link_wiggle {
        t_map[[
            p_time_full + p_threshold + p_log_sigma + j,
            p_time_reduced + p_threshold + p_log_sigma + j,
        ]] = 1.0;
    }
    t_map.dot(cov_reduced).dot(&t_map.t())
}

impl CustomFamily for SurvivalLocationScaleFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != self.expected_blocks() {
            return Err(format!(
                "SurvivalLocationScaleFamily expects {} blocks, got {}",
                self.expected_blocks(),
                block_states.len()
            ));
        }
        let n = self.n;
        let eta_time = &block_states[Self::BLOCK_TIME].eta;
        let eta_t = &block_states[Self::BLOCK_THRESHOLD].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let eta_w = self
            .x_link_wiggle
            .as_ref()
            .map(|_| &block_states[Self::BLOCK_LINK_WIGGLE].eta);
        if eta_time.len() != 3 * n || eta_t.len() != n || eta_ls.len() != n {
            return Err("survival probit-location-scale eta dimension mismatch".to_string());
        }
        if let Some(w) = eta_w
            && w.len() != n
        {
            return Err("survival probit-location-scale wiggle eta dimension mismatch".to_string());
        }

        let h0 = eta_time.slice(s![0..n]);
        let h1 = eta_time.slice(s![n..2 * n]);
        let d_raw = eta_time.slice(s![2 * n..3 * n]);

        let (sigma, ds, d2s, _d3s) =
            bounded_sigma_derivs_up_to_third(eta_ls.view(), self.sigma_min, self.sigma_max);
        let mut ll = 0.0;

        let mut grad_time_eta_h0 = Array1::<f64>::zeros(n);
        let mut grad_time_eta_h1 = Array1::<f64>::zeros(n);
        let mut grad_time_eta_d = Array1::<f64>::zeros(n);
        let mut h_time_h0 = Array1::<f64>::zeros(n);
        let mut h_time_h1 = Array1::<f64>::zeros(n);
        let mut h_time_d = Array1::<f64>::zeros(n);

        let mut d1_q = Array1::<f64>::zeros(n);
        let mut d2_q = Array1::<f64>::zeros(n);

        for i in 0..n {
            let state = self.row_predictor_state(
                h0[i],
                h1[i],
                d_raw[i],
                eta_t[i],
                sigma[i],
                eta_w.map(|w| w[i]),
            );
            let Some(row) = self.row_derivatives(i, state)? else {
                continue;
            };
            ll += row.ll;
            d1_q[i] = row.d1_q;
            d2_q[i] = row.d2_q;
            grad_time_eta_h0[i] = row.grad_time_eta_h0;
            grad_time_eta_h1[i] = row.grad_time_eta_h1;
            grad_time_eta_d[i] = row.grad_time_eta_d;
            h_time_h0[i] = row.h_time_h0;
            h_time_h1[i] = row.h_time_h1;
            h_time_d[i] = row.h_time_d;
        }

        // Block 0: exact beta-space gradient/Hessian
        let grad_time = self.x_time_entry.t().dot(&grad_time_eta_h0)
            + self.x_time_exit.t().dot(&grad_time_eta_h1)
            + self.x_time_deriv.t().dot(&grad_time_eta_d);
        let hess_time = fast_xt_diag_x(&self.x_time_entry, &h_time_h0)
            + fast_xt_diag_x(&self.x_time_exit, &h_time_h1)
            + fast_xt_diag_x(&self.x_time_deriv, &h_time_d);

        // Block 1: threshold eta_t enters q linearly with dq/deta_t = -1/sigma.
        let dq_t = sigma.mapv(|s| -1.0 / s.max(1e-12));
        let grad_eta_t = &d1_q * &dq_t;
        let h_eta_t = -(&d2_q * &dq_t.mapv(|v| v * v));
        let grad_t = self.x_threshold.transpose_vector_multiply(&grad_eta_t);
        let hess_t = xt_diag_x_symmetric(&self.x_threshold, &h_eta_t)?;

        // Block 2: eta_ls enters q via bounded sigma map.
        let dq_ls = Array1::from_iter(
            eta_t
                .iter()
                .zip(sigma.iter())
                .zip(ds.iter())
                .map(|((&t, &s), &d1)| t * d1 / (s * s).max(1e-12)),
        );
        let d2q_ls = Array1::from_iter(
            eta_t
                .iter()
                .zip(sigma.iter())
                .zip(ds.iter())
                .zip(d2s.iter())
                .map(|(((&t, &s), &d1), &d2)| {
                    t * (d2 / (s * s).max(1e-12) - 2.0 * d1 * d1 / (s * s * s).max(1e-12))
                }),
        );
        let grad_eta_ls = &d1_q * &dq_ls;
        let h_eta_ls = -(&d2_q * &dq_ls.mapv(|v| v * v) + &(&d1_q * &d2q_ls));
        let grad_ls = self.x_log_sigma.transpose_vector_multiply(&grad_eta_ls);
        let hess_ls = xt_diag_x_symmetric(&self.x_log_sigma, &h_eta_ls)?;

        let mut block_working_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: grad_time,
                hessian: SymmetricMatrix::Dense(hess_time),
            },
            BlockWorkingSet::ExactNewton {
                gradient: grad_t,
                hessian: hess_t,
            },
            BlockWorkingSet::ExactNewton {
                gradient: grad_ls,
                hessian: hess_ls,
            },
        ];
        if let Some(x_w) = self.x_link_wiggle.as_ref() {
            let grad_w = x_w.transpose_vector_multiply(&d1_q);
            let hess_w = xt_diag_x_symmetric(x_w, &(-&d2_q))?;
            block_working_sets.push(BlockWorkingSet::ExactNewton {
                gradient: grad_w,
                hessian: hess_w,
            });
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            block_working_sets,
        })
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != self.expected_blocks() {
            return Err(format!(
                "SurvivalLocationScaleFamily expects {} blocks, got {}",
                self.expected_blocks(),
                block_states.len()
            ));
        }
        let n = self.n;
        let eta_time = &block_states[Self::BLOCK_TIME].eta;
        let eta_t = &block_states[Self::BLOCK_THRESHOLD].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let eta_w = self
            .x_link_wiggle
            .as_ref()
            .map(|_| &block_states[Self::BLOCK_LINK_WIGGLE].eta);
        if eta_time.len() != 3 * n || eta_t.len() != n || eta_ls.len() != n {
            return Err("survival probit-location-scale eta dimension mismatch".to_string());
        }
        if let Some(w) = eta_w
            && w.len() != n
        {
            return Err("survival probit-location-scale wiggle eta dimension mismatch".to_string());
        }

        let h0 = eta_time.slice(s![0..n]);
        let h1 = eta_time.slice(s![n..2 * n]);
        let d_raw = eta_time.slice(s![2 * n..3 * n]);

        let (sigma, ds, d2s, d3s) =
            bounded_sigma_derivs_up_to_third(eta_ls.view(), self.sigma_min, self.sigma_max);
        let mut d1_q = Array1::<f64>::zeros(n);
        let mut d2_q = Array1::<f64>::zeros(n);
        let mut d3_q = Array1::<f64>::zeros(n);
        let mut d_h_h0 = Array1::<f64>::zeros(n);
        let mut d_h_h1 = Array1::<f64>::zeros(n);
        let mut d_h_d = Array1::<f64>::zeros(n);

        for i in 0..n {
            let state = self.row_predictor_state(
                h0[i],
                h1[i],
                d_raw[i],
                eta_t[i],
                sigma[i],
                eta_w.map(|w| w[i]),
            );
            let Some(row) = self.row_derivatives(i, state)? else {
                continue;
            };
            d1_q[i] = row.d1_q;
            d2_q[i] = row.d2_q;
            d3_q[i] = row.d3_q;
            d_h_h0[i] = row.d_h_h0;
            d_h_h1[i] = row.d_h_h1;
            d_h_d[i] = row.d_h_d;
        }

        match block_idx {
            Self::BLOCK_TIME => {
                if d_beta.len() != self.x_time_entry.ncols() {
                    return Err(format!(
                        "time block d_beta length mismatch: got {}, expected {}",
                        d_beta.len(),
                        self.x_time_entry.ncols()
                    ));
                }
                let d_eta_h0 = self.x_time_entry.dot(d_beta);
                let d_eta_h1 = self.x_time_exit.dot(d_beta);
                let d_eta_d = self.x_time_deriv.dot(d_beta);
                let w0 = &d_h_h0 * &d_eta_h0;
                let w1 = &d_h_h1 * &d_eta_h1;
                let wd = &d_h_d * &d_eta_d;
                let d_h = fast_xt_diag_x(&self.x_time_entry, &w0)
                    + fast_xt_diag_x(&self.x_time_exit, &w1)
                    + fast_xt_diag_x(&self.x_time_deriv, &wd);
                Ok(Some(d_h))
            }
            Self::BLOCK_THRESHOLD => {
                if d_beta.len() != self.x_threshold.ncols() {
                    return Err(format!(
                        "threshold block d_beta length mismatch: got {}, expected {}",
                        d_beta.len(),
                        self.x_threshold.ncols()
                    ));
                }
                let dq_t = sigma.mapv(|s| -1.0 / s.max(1e-12));
                let d_eta_t = self.x_threshold.matrix_vector_multiply(d_beta);
                // Since q(eta_t) is linear in eta_t, only dq_t is nonzero:
                // H = -d²ℓ/deta², so dH[u] picks up the leading minus too.
                let d_h_eta = -(&d3_q * &dq_t.mapv(|v| v * v * v) * &d_eta_t);
                let d_h = xt_diag_x_symmetric(&self.x_threshold, &d_h_eta)?;
                Ok(Some(d_h.to_dense()))
            }
            Self::BLOCK_LOG_SIGMA => {
                if d_beta.len() != self.x_log_sigma.ncols() {
                    return Err(format!(
                        "log-sigma block d_beta length mismatch: got {}, expected {}",
                        d_beta.len(),
                        self.x_log_sigma.ncols()
                    ));
                }
                let dq = Array1::from_iter(
                    eta_t
                        .iter()
                        .zip(sigma.iter())
                        .zip(ds.iter())
                        .map(|((&t, &s), &d1)| t * d1 / (s * s).max(1e-12)),
                );
                let d2q = Array1::from_iter(
                    eta_t
                        .iter()
                        .zip(sigma.iter())
                        .zip(ds.iter())
                        .zip(d2s.iter())
                        .map(|(((&t, &s), &d1), &d2)| {
                            t * (d2 / (s * s).max(1e-12) - 2.0 * d1 * d1 / (s * s * s).max(1e-12))
                        }),
                );
                let d3q = Array1::from_iter(
                    eta_t
                        .iter()
                        .zip(sigma.iter())
                        .zip(ds.iter())
                        .zip(d2s.iter())
                        .zip(d3s.iter())
                        .map(|((((&t, &s), &d1), &d2), &d3)| {
                            t * (d3 / (s * s).max(1e-12) - 6.0 * d1 * d2 / (s * s * s).max(1e-12)
                                + 6.0 * d1 * d1 * d1 / (s * s * s * s).max(1e-12))
                        }),
                );
                let d_eta_ls = self.x_log_sigma.matrix_vector_multiply(d_beta);
                // Full third-order chain rule for H = -d²ℓ/deta²:
                // dH/deta = -[ d³ℓ/dq³ (dq)^3 + 3 d²ℓ/dq² dq d²q + dℓ/dq d³q ].
                let d_h_eta = (&d3_q * &dq.mapv(|v| v * v * v)
                    + &(&d2_q * &(3.0 * &dq * &d2q))
                    + &(&d1_q * &d3q))
                    * &d_eta_ls;
                let d_h_eta = -d_h_eta;
                let d_h = xt_diag_x_symmetric(&self.x_log_sigma, &d_h_eta)?;
                Ok(Some(d_h.to_dense()))
            }
            Self::BLOCK_LINK_WIGGLE => {
                let x_w = self.x_link_wiggle.as_ref().ok_or_else(|| {
                    "link wiggle directional derivative requested but wiggle block is disabled"
                        .to_string()
                })?;
                if d_beta.len() != x_w.ncols() {
                    return Err(format!(
                        "link-wiggle block d_beta length mismatch: got {}, expected {}",
                        d_beta.len(),
                        x_w.ncols()
                    ));
                }
                let d_eta_w = x_w.matrix_vector_multiply(d_beta);
                let d_h_eta = -(&d3_q * &d_eta_w);
                let d_h = xt_diag_x_symmetric(x_w, &d_h_eta)?;
                Ok(Some(d_h.to_dense()))
            }
            _ => Ok(None),
        }
    }

    fn block_linear_constraints(
        &self,
        _block_states: &[ParameterBlockState],
        block_idx: usize,
        _spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx != Self::BLOCK_TIME {
            return Ok(None);
        }
        let n = self.x_time_deriv.nrows();
        let p = self.x_time_deriv.ncols();
        if self.offset_time_deriv.len() != n {
            return Err(format!(
                "time derivative offset length mismatch: got {}, expected {n}",
                self.offset_time_deriv.len()
            ));
        }
        if n == 0 || p == 0 || self.derivative_guard <= 0.0 {
            return Ok(None);
        }
        let mut a = Array2::<f64>::zeros((n, p));
        a.assign(&self.x_time_deriv);
        let mut b = Array1::<f64>::zeros(n);
        let guard = self.derivative_guard;
        for i in 0..n {
            b[i] = guard - self.offset_time_deriv[i];
        }
        Ok(Some(LinearInequalityConstraints { a, b }))
    }
}

pub fn fit_survival_location_scale(
    spec: SurvivalLocationScaleSpec,
) -> Result<SurvivalLocationScaleFitResult, String> {
    let n = spec.event_target.len();
    if n == 0 {
        return Err("fit_survival_location_scale: empty dataset".to_string());
    }
    if spec.age_entry.len() != n || spec.age_exit.len() != n || spec.weights.len() != n {
        return Err("fit_survival_location_scale: top-level input size mismatch".to_string());
    }
    if !spec.sigma_min.is_finite()
        || !spec.sigma_max.is_finite()
        || spec.sigma_min <= 0.0
        || spec.sigma_max <= 0.0
        || spec.sigma_min >= spec.sigma_max
    {
        return Err(format!(
            "fit_survival_location_scale: invalid sigma bounds (min={}, max={})",
            spec.sigma_min, spec.sigma_max
        ));
    }
    if !(spec.tol.is_finite() && spec.tol > 0.0) {
        return Err(format!(
            "fit_survival_location_scale: invalid tol {}",
            spec.tol
        ));
    }
    if spec.max_iter == 0 {
        return Err("fit_survival_location_scale: max_iter must be > 0".to_string());
    }
    validate_time_block(n, &spec.time_block)?;
    validate_cov_block("threshold_block", n, &spec.threshold_block)?;
    validate_cov_block("log_sigma_block", n, &spec.log_sigma_block)?;
    if let Some(w) = spec.link_wiggle_block.as_ref() {
        validate_wiggle_block(n, w)?;
    }

    for i in 0..n {
        if !spec.age_entry[i].is_finite()
            || !spec.age_exit[i].is_finite()
            || spec.age_exit[i] < spec.age_entry[i]
        {
            return Err(format!(
                "fit_survival_location_scale: invalid interval at row {} (entry={}, exit={})",
                i + 1,
                spec.age_entry[i],
                spec.age_exit[i]
            ));
        }
        if !spec.weights[i].is_finite() || spec.weights[i] < 0.0 {
            return Err(format!(
                "fit_survival_location_scale: invalid weight at row {} ({})",
                i + 1,
                spec.weights[i]
            ));
        }
        if !spec.event_target[i].is_finite() || !(0.0..=1.0).contains(&spec.event_target[i]) {
            return Err(format!(
                "fit_survival_location_scale: event_target must be in [0,1], found {} at row {}",
                spec.event_target[i],
                i + 1
            ));
        }
    }

    let anchor_row = select_anchor_row(&spec.age_entry, spec.time_anchor)?;
    let time_prepared = prepare_identified_time_block(&spec.time_block, anchor_row)?;
    let time_stacked_design = stack_time_design(&TimeBlockInput {
        design_entry: time_prepared.design_entry.clone(),
        design_exit: time_prepared.design_exit.clone(),
        design_derivative_exit: time_prepared.design_derivative_exit.clone(),
        offset_entry: spec.time_block.offset_entry.clone(),
        offset_exit: spec.time_block.offset_exit.clone(),
        derivative_offset_exit: spec.time_block.derivative_offset_exit.clone(),
        penalties: time_prepared.penalties.clone(),
        initial_log_lambdas: spec.time_block.initial_log_lambdas.clone(),
        initial_beta: time_prepared.initial_beta.clone(),
    });
    let time_stacked_offset = stack_time_offset(&spec.time_block);

    let time_spec = ParameterBlockSpec {
        name: "time_transform".to_string(),
        design: DesignMatrix::Dense(time_stacked_design),
        offset: time_stacked_offset,
        penalties: time_prepared.penalties.clone(),
        initial_log_lambdas: initial_log_lambdas(
            &time_prepared.penalties,
            spec.time_block.initial_log_lambdas.clone(),
        )?,
        initial_beta: time_prepared.initial_beta.clone(),
    };
    let threshold_spec = ParameterBlockSpec {
        name: "threshold".to_string(),
        design: spec.threshold_block.design.clone(),
        offset: spec.threshold_block.offset.clone(),
        penalties: spec.threshold_block.penalties.clone(),
        initial_log_lambdas: initial_log_lambdas(
            &spec.threshold_block.penalties,
            spec.threshold_block.initial_log_lambdas.clone(),
        )?,
        initial_beta: spec.threshold_block.initial_beta.clone(),
    };
    let log_sigma_spec = ParameterBlockSpec {
        name: "log_sigma".to_string(),
        design: spec.log_sigma_block.design.clone(),
        offset: spec.log_sigma_block.offset.clone(),
        penalties: spec.log_sigma_block.penalties.clone(),
        initial_log_lambdas: initial_log_lambdas(
            &spec.log_sigma_block.penalties,
            spec.log_sigma_block.initial_log_lambdas.clone(),
        )?,
        initial_beta: spec.log_sigma_block.initial_beta.clone(),
    };
    let wiggle_spec = if let Some(w) = spec.link_wiggle_block.as_ref() {
        Some(ParameterBlockSpec {
            name: "link_wiggle".to_string(),
            design: w.design.clone(),
            offset: Array1::zeros(n),
            penalties: w.penalties.clone(),
            initial_log_lambdas: initial_log_lambdas(&w.penalties, w.initial_log_lambdas.clone())?,
            initial_beta: w.initial_beta.clone(),
        })
    } else {
        None
    };

    let family = SurvivalLocationScaleFamily {
        n,
        y: spec.event_target,
        w: spec.weights,
        sigma_min: spec.sigma_min,
        sigma_max: spec.sigma_max,
        inverse_link: spec.inverse_link,
        derivative_guard: spec.derivative_guard,
        derivative_softness: spec.derivative_softness,
        x_time_entry: time_prepared.design_entry,
        x_time_exit: time_prepared.design_exit,
        x_time_deriv: time_prepared.design_derivative_exit,
        offset_time_deriv: spec.time_block.derivative_offset_exit.clone(),
        x_threshold: spec.threshold_block.design.clone(),
        x_log_sigma: spec.log_sigma_block.design.clone(),
        x_link_wiggle: wiggle_spec.as_ref().map(|s| s.design.clone()),
    };

    let options = BlockwiseFitOptions {
        inner_max_cycles: spec.max_iter,
        inner_tol: spec.tol,
        outer_max_iter: 60,
        outer_tol: 1e-5,
        ..BlockwiseFitOptions::default()
    };
    let block_specs = if let Some(w) = wiggle_spec {
        vec![time_spec, threshold_spec, log_sigma_spec, w]
    } else {
        vec![time_spec, threshold_spec, log_sigma_spec]
    };
    let fit: BlockwiseFitResult = fit_custom_family(&family, &block_specs, &options)?;

    let k_time = spec.time_block.penalties.len();
    let k_t = spec.threshold_block.penalties.len();
    let k_ls = spec.log_sigma_block.penalties.len();
    let lambdas = fit.log_lambdas.mapv(f64::exp);
    let lambdas_time = lambdas.slice(s![0..k_time]).to_owned();
    let lambdas_threshold = lambdas.slice(s![k_time..k_time + k_t]).to_owned();
    let lambdas_log_sigma = lambdas
        .slice(s![k_time + k_t..k_time + k_t + k_ls])
        .to_owned();
    let k_w = spec
        .link_wiggle_block
        .as_ref()
        .map(|w| w.penalties.len())
        .unwrap_or(0);
    let lambdas_link_wiggle = if k_w > 0 {
        Some(
            lambdas
                .slice(s![k_time + k_t + k_ls..k_time + k_t + k_ls + k_w])
                .to_owned(),
        )
    } else {
        None
    };

    let beta_time_reduced = fit.block_states[SurvivalLocationScaleFamily::BLOCK_TIME]
        .beta
        .clone();
    let beta_time = time_prepared.transform.z.dot(&beta_time_reduced);
    let beta_threshold = fit.block_states[SurvivalLocationScaleFamily::BLOCK_THRESHOLD]
        .beta
        .clone();
    let beta_log_sigma = fit.block_states[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA]
        .beta
        .clone();
    let beta_link_wiggle = if family.x_link_wiggle.is_some() {
        Some(
            fit.block_states[SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE]
                .beta
                .clone(),
        )
    } else {
        None
    };

    let covariance_conditional = fit.covariance_conditional.as_ref().map(|cov_reduced| {
        let z = &time_prepared.transform.z;
        let p_t = beta_threshold.len();
        let p_ls = beta_log_sigma.len();
        let p_w = beta_link_wiggle.as_ref().map_or(0, |b| b.len());
        lift_conditional_covariance(cov_reduced, z, p_t, p_ls, p_w)
    });

    Ok(SurvivalLocationScaleFitResult {
        beta_time,
        beta_threshold,
        beta_log_sigma,
        beta_link_wiggle,
        lambdas_time,
        lambdas_threshold,
        lambdas_log_sigma,
        lambdas_link_wiggle,
        log_likelihood: fit.log_likelihood,
        penalized_objective: fit.penalized_objective,
        iterations: fit.inner_cycles,
        final_grad_norm: fit.outer_final_gradient_norm,
        converged: fit.converged,
        covariance_conditional,
    })
}

pub fn predict_survival_location_scale(
    input: &SurvivalLocationScalePredictInput,
    fit: &SurvivalLocationScaleFitResult,
) -> Result<SurvivalLocationScalePredictResult, String> {
    let predictors = prediction_linear_predictors(input, fit)?;
    let n = input.x_time_exit.nrows();
    let (sigma, _, _, _) = bounded_sigma_derivs_up_to_third(
        predictors.eta_ls.view(),
        input.sigma_min,
        input.sigma_max,
    );
    let eta = Array1::from_iter(
        predictors
            .h
            .iter()
            .zip(predictors.eta_t.iter())
            .zip(sigma.iter())
            .enumerate()
            .map(|(i, ((&hh, &tt), &ss))| {
                let mut q = -hh - tt / ss.max(1e-12);
                if let Some(w) = predictors.eta_w.as_ref() {
                    q += w[i];
                }
                q
            }),
    );
    let mut survival_prob = Array1::<f64>::zeros(n);
    for (i, &v) in eta.iter().enumerate() {
        survival_prob[i] = inverse_link_survival_prob_checked(&input.inverse_link, v)?;
    }
    Ok(SurvivalLocationScalePredictResult { eta, survival_prob })
}

pub fn predict_survival_location_scale_posterior_mean(
    input: &SurvivalLocationScalePredictInput,
    fit: &SurvivalLocationScaleFitResult,
    covariance: &Array2<f64>,
) -> Result<SurvivalLocationScalePredictResult, String> {
    // Uncertainty-aware survival posterior mean with conditional Gaussian
    // reduction.
    //
    // The deterministic survival predictor already computes the latent pieces
    //
    //   h  = time block linear predictor
    //   t  = threshold block linear predictor
    //   ls = log-sigma block linear predictor
    //   w  = optional link-wiggle predictor.
    //
    // Under the Gaussian coefficient approximation, the rowwise latent vector
    //
    //   (h, t, ls, w)
    //
    // is jointly Gaussian, with w identically zero when no wiggle block is
    // present. The posterior-mean target is
    //
    //   E[g(eta)],
    //   eta = -h - t / sigma(ls) + w.
    //
    // A direct evaluation is a 3D expectation without wiggle and a 4D
    // expectation with wiggle. We do not need to integrate over all latent
    // dimensions directly. Conditioning on ls is enough:
    //
    //   (h, t, w) | ls  is Gaussian,
    //   eta | ls        is affine in (h, t, w),
    //
    // so eta | ls is itself Gaussian with exact conditional mean and variance.
    // That reduces the full posterior mean to
    //
    //   E[g(eta)] = E_ls[ E[g(eta) | ls] ],
    //
    // i.e. a 1D outer Gaussian expectation over ls, where the inner object is
    // the existing Gaussian-uncertain scalar inverse-link expectation.
    //
    // If the conditioning algebra becomes numerically unsafe, this routine
    // falls back to direct adaptive Gaussian expectation rather than forcing
    // the reduction.
    let pred = predict_survival_location_scale(input, fit)?;
    let n = input.x_time_exit.nrows();
    let predictors = prediction_linear_predictors(input, fit)?;
    let p_time = fit.beta_time.len();
    let p_t = fit.beta_threshold.len();
    let p_ls = fit.beta_log_sigma.len();
    let p_w = fit.beta_link_wiggle.as_ref().map_or(0, |b| b.len());
    let p_total = p_time + p_t + p_ls + p_w;
    if covariance.nrows() != p_total || covariance.ncols() != p_total {
        return Err(format!(
            "predict_survival_location_scale_posterior_mean: covariance shape mismatch: got {}x{}, expected {}x{}",
            covariance.nrows(),
            covariance.ncols(),
            p_total,
            p_total
        ));
    }

    if input.x_threshold.nrows() != n || input.x_log_sigma.nrows() != n {
        return Err(
            "predict_survival_location_scale_posterior_mean: row mismatch across design views"
                .to_string(),
        );
    }

    let cov_hh = covariance.slice(s![0..p_time, 0..p_time]).to_owned();
    let cov_tt = covariance
        .slice(s![p_time..p_time + p_t, p_time..p_time + p_t])
        .to_owned();
    let cov_ll = covariance
        .slice(s![
            p_time + p_t..p_time + p_t + p_ls,
            p_time + p_t..p_time + p_t + p_ls
        ])
        .to_owned();
    let cov_ht = covariance
        .slice(s![0..p_time, p_time..p_time + p_t])
        .to_owned();
    let cov_hl = covariance
        .slice(s![0..p_time, p_time + p_t..p_time + p_t + p_ls])
        .to_owned();
    let cov_tl = covariance
        .slice(s![p_time..p_time + p_t, p_time + p_t..p_time + p_t + p_ls])
        .to_owned();
    let (var_w_rows, cov_hw_rows, cov_tw_rows, cov_lw_rows) =
        if let Some(xw) = input.x_link_wiggle.as_ref() {
            let cov_ww = covariance
                .slice(s![
                    p_time + p_t + p_ls..p_total,
                    p_time + p_t + p_ls..p_total
                ])
                .to_owned();
            let cov_hw = covariance
                .slice(s![0..p_time, p_time + p_t + p_ls..p_total])
                .to_owned();
            let cov_tw = covariance
                .slice(s![p_time..p_time + p_t, p_time + p_t + p_ls..p_total])
                .to_owned();
            let cov_lw = covariance
                .slice(s![
                    p_time + p_t..p_time + p_t + p_ls,
                    p_time + p_t + p_ls..p_total
                ])
                .to_owned();
            (
                Some(xw.quadratic_form_diag(&cov_ww)?),
                Some(rowwise_cross_quadratic_dense_design(
                    &input.x_time_exit,
                    &cov_hw,
                    xw,
                )?),
                Some(rowwise_cross_quadratic_design(
                    &input.x_threshold,
                    &cov_tw,
                    xw,
                )?),
                Some(rowwise_cross_quadratic_design(
                    &input.x_log_sigma,
                    &cov_lw,
                    xw,
                )?),
            )
        } else {
            (None, None, None, None)
        };

    let xh_hh = input.x_time_exit.dot(&cov_hh);
    let var_t_rows = input.x_threshold.quadratic_form_diag(&cov_tt)?;
    let var_ls_rows = input.x_log_sigma.quadratic_form_diag(&cov_ll)?;
    let cov_ht_rows =
        rowwise_cross_quadratic_dense_design(&input.x_time_exit, &cov_ht, &input.x_threshold)?;
    let cov_hl_rows =
        rowwise_cross_quadratic_dense_design(&input.x_time_exit, &cov_hl, &input.x_log_sigma)?;
    let cov_tl_rows =
        rowwise_cross_quadratic_design(&input.x_threshold, &cov_tl, &input.x_log_sigma)?;

    let mu_h = predictors.h;
    let mu_t = predictors.eta_t;
    let mu_ls = predictors.eta_ls;
    let mu_w = predictors.eta_w;
    let link = input.inverse_link.link_function();
    let mixture_state = input.inverse_link.mixture_state();
    let sas_state = input.inverse_link.sas_state();
    let quad_ctx = crate::quadrature::QuadratureContext::new();
    const VAR_L_DEGENERATE_TOL: f64 = 1e-12;
    const CROSS_L_DEGENERATE_TOL: f64 = 1e-10;
    let gaussian_survival_mean = |mu_loc: f64, var_loc: f64| {
        let var_loc = var_loc.max(0.0);
        if matches!(input.inverse_link, InverseLink::Mixture(_)) {
            return crate::quadrature::normal_expectation_1d_adaptive(
                &quad_ctx,
                mu_loc,
                var_loc.sqrt(),
                |z| inverse_link_survival_prob_value(&input.inverse_link, z),
            )
            .clamp(0.0, 1.0);
        }
        crate::quadrature::integrated_inverse_link_jet_with_state(
            &quad_ctx,
            link,
            mu_loc,
            var_loc.sqrt(),
            mixture_state,
            sas_state,
        )
        .map(|jet| (1.0 - jet.mean).clamp(0.0, 1.0))
        .unwrap_or_else(|_| {
            if link == LinkFunction::Probit {
                let denom = (1.0 + var_loc).sqrt().max(1e-12);
                (1.0 - normal_cdf(mu_loc / denom)).clamp(0.0, 1.0)
            } else {
                crate::quadrature::normal_expectation_1d_adaptive(
                    &quad_ctx,
                    mu_loc,
                    var_loc.sqrt(),
                    |z| inverse_link_survival_prob_value(&input.inverse_link, z),
                )
                .clamp(0.0, 1.0)
            }
        })
    };

    let fallback_row = |i: usize| {
        if let Some(mu_w) = mu_w.as_ref() {
            crate::quadrature::normal_expectation_nd_adaptive::<4, _>(
                &quad_ctx,
                [mu_h[i], mu_t[i], mu_ls[i], mu_w[i]],
                [
                    [
                        var_h_row(i, input, &xh_hh),
                        cov_ht_rows[i],
                        cov_hl_rows[i],
                        cov_hw_rows.as_ref().expect("wiggle cov_hw rows")[i],
                    ],
                    [
                        cov_ht_rows[i],
                        var_t_rows[i],
                        cov_tl_rows[i],
                        cov_tw_rows.as_ref().expect("wiggle cov_tw rows")[i],
                    ],
                    [
                        cov_hl_rows[i],
                        cov_tl_rows[i],
                        var_ls_rows[i],
                        cov_lw_rows.as_ref().expect("wiggle cov_lw rows")[i],
                    ],
                    [
                        cov_hw_rows.as_ref().expect("wiggle cov_hw rows")[i],
                        cov_tw_rows.as_ref().expect("wiggle cov_tw rows")[i],
                        cov_lw_rows.as_ref().expect("wiggle cov_lw rows")[i],
                        var_w_rows.as_ref().expect("wiggle var rows")[i],
                    ],
                ],
                11,
                |x| {
                    let sigma = bounded_sigma_derivs_up_to_third_scalar(
                        x[2],
                        input.sigma_min,
                        input.sigma_max,
                    )
                    .0
                    .max(1e-12);
                    inverse_link_survival_prob_value(
                        &input.inverse_link,
                        -x[0] - x[1] / sigma + x[3],
                    )
                },
            )
            .clamp(0.0, 1.0)
        } else {
            crate::quadrature::normal_expectation_3d_adaptive(
                &quad_ctx,
                [mu_h[i], mu_t[i], mu_ls[i]],
                [
                    [var_h_row(i, input, &xh_hh), cov_ht_rows[i], cov_hl_rows[i]],
                    [cov_ht_rows[i], var_t_rows[i], cov_tl_rows[i]],
                    [cov_hl_rows[i], cov_tl_rows[i], var_ls_rows[i]],
                ],
                |h, t, ls| {
                    let sigma = bounded_sigma_derivs_up_to_third_scalar(
                        ls,
                        input.sigma_min,
                        input.sigma_max,
                    )
                    .0
                    .max(1e-12);
                    inverse_link_survival_prob_value(&input.inverse_link, -h - t / sigma)
                },
            )
            .clamp(0.0, 1.0)
        }
    };

    let survival_prob = Array1::from_iter((0..n).map(|i| {
        let var_h = var_h_row(i, input, &xh_hh);
        let var_t = var_t_rows[i];
        let var_ls = var_ls_rows[i];
        let cov_ht_i = cov_ht_rows[i];
        let cov_hl_i = cov_hl_rows[i];
        let cov_tl_i = cov_tl_rows[i];
        let mu_w_i = mu_w.as_ref().map_or(0.0, |vals| vals[i]);
        let var_w_i = var_w_rows.as_ref().map_or(0.0, |vals| vals[i]);
        let cov_hw_i = cov_hw_rows.as_ref().map_or(0.0, |vals| vals[i]);
        let cov_tw_i = cov_tw_rows.as_ref().map_or(0.0, |vals| vals[i]);
        let cov_lw_i = cov_lw_rows.as_ref().map_or(0.0, |vals| vals[i]);
        let mu_l_i = mu_ls[i];

        if !(var_h.is_finite()
            && var_t.is_finite()
            && var_ls.is_finite()
            && var_w_i.is_finite()
            && cov_ht_i.is_finite()
            && cov_hl_i.is_finite()
            && cov_tl_i.is_finite()
            && cov_hw_i.is_finite()
            && cov_tw_i.is_finite()
            && cov_lw_i.is_finite())
        {
            return fallback_row(i);
        }

        // Exact degenerate limit: if Var(L)=0, PSD implies the cross-covariances
        // with L vanish, so the outer expectation collapses to the point mass
        // L = mu_l.
        if var_ls <= VAR_L_DEGENERATE_TOL {
            if cov_hl_i.abs() > CROSS_L_DEGENERATE_TOL
                || cov_tl_i.abs() > CROSS_L_DEGENERATE_TOL
                || cov_lw_i.abs() > CROSS_L_DEGENERATE_TOL
            {
                return fallback_row(i);
            }
            let sigma =
                bounded_sigma_derivs_up_to_third_scalar(mu_l_i, input.sigma_min, input.sigma_max)
                    .0
                    .max(1e-12);
            let q_l = 1.0 / sigma;
            let mu_base = -mu_h[i] - q_l * mu_t[i] + mu_w_i;
            let var_base = var_h + q_l * q_l * var_t + var_w_i + 2.0 * q_l * cov_ht_i
                - 2.0 * cov_hw_i
                - 2.0 * q_l * cov_tw_i;
            let mu_loc = mu_base;
            let var_loc = var_base.max(0.0);
            return gaussian_survival_mean(mu_loc, var_loc);
        }

        crate::quadrature::normal_expectation_1d_adaptive(&quad_ctx, mu_l_i, var_ls.sqrt(), |ls| {
            let sigma =
                bounded_sigma_derivs_up_to_third_scalar(ls, input.sigma_min, input.sigma_max)
                    .0
                    .max(1e-12);
            let q_l = 1.0 / sigma;
            let delta_l = ls - mu_l_i;
            let mu_base = -mu_h[i] - q_l * mu_t[i] + mu_w_i;
            let cov_eta_l = cov_hl_i + q_l * cov_tl_i - cov_lw_i;
            let mu_loc = mu_base - (cov_eta_l / var_ls) * delta_l;
            let var_base = var_h + q_l * q_l * var_t + var_w_i + 2.0 * q_l * cov_ht_i
                - 2.0 * cov_hw_i
                - 2.0 * q_l * cov_tw_i;
            let var_loc = (var_base - cov_eta_l * cov_eta_l / var_ls).max(0.0);
            if !mu_loc.is_finite() || !var_loc.is_finite() {
                return fallback_row(i);
            }
            // This is the payoff of the conditional Gaussian reduction above:
            // for fixed ls, eta = -h - t / sigma(ls) + w is Gaussian, so we
            // hand its conditional mean and standard deviation straight to
            // the shared integrated-expectation dispatcher instead of
            // integrating over h, t, and w explicitly.
            gaussian_survival_mean(mu_loc, var_loc)
        })
        .clamp(0.0, 1.0)
    }));

    Ok(SurvivalLocationScalePredictResult {
        eta: pred.eta,
        survival_prob,
    })
}

pub fn predict_survival_location_scale_with_uncertainty(
    input: &SurvivalLocationScalePredictInput,
    fit: &SurvivalLocationScaleFitResult,
    covariance: &Array2<f64>,
    posterior_mean: bool,
    include_response_sd: bool,
) -> Result<SurvivalLocationScalePredictUncertaintyResult, String> {
    let base = predict_survival_location_scale(input, fit)?;
    let n = input.x_time_exit.nrows();
    let p_time = fit.beta_time.len();
    let p_t = fit.beta_threshold.len();
    let p_ls = fit.beta_log_sigma.len();
    let p_w = fit.beta_link_wiggle.as_ref().map_or(0, |b| b.len());
    let p_total = p_time + p_t + p_ls + p_w;
    if covariance.nrows() != p_total || covariance.ncols() != p_total {
        return Err(format!(
            "predict_survival_location_scale_with_uncertainty: covariance shape mismatch: got {}x{}, expected {}x{}",
            covariance.nrows(),
            covariance.ncols(),
            p_total,
            p_total
        ));
    }
    if p_w > 0 && (fit.beta_link_wiggle.is_none() || input.x_link_wiggle.is_none()) {
        return Err(
            "predict_survival_location_scale_with_uncertainty: wiggle covariance provided but wiggle design/beta is partial"
                .to_string(),
        );
    }

    let predictors = prediction_linear_predictors(input, fit)?;

    if input.x_threshold.nrows() != n || input.x_log_sigma.nrows() != n {
        return Err(
            "predict_survival_location_scale_with_uncertainty: row mismatch across design views"
                .to_string(),
        );
    }
    if let Some(xw) = input.x_link_wiggle.as_ref() {
        if xw.nrows() != n {
            return Err(
                "predict_survival_location_scale_with_uncertainty: x_link_wiggle row mismatch"
                    .to_string(),
            );
        }
    }

    let (sigma, ds, _, _) = bounded_sigma_derivs_up_to_third(
        predictors.eta_ls.view(),
        input.sigma_min,
        input.sigma_max,
    );

    let cov_hh = covariance.slice(s![0..p_time, 0..p_time]).to_owned();
    let cov_tt = covariance
        .slice(s![p_time..p_time + p_t, p_time..p_time + p_t])
        .to_owned();
    let cov_ll = covariance
        .slice(s![
            p_time + p_t..p_time + p_t + p_ls,
            p_time + p_t..p_time + p_t + p_ls
        ])
        .to_owned();
    let cov_ht = covariance
        .slice(s![0..p_time, p_time..p_time + p_t])
        .to_owned();
    let cov_hl = covariance
        .slice(s![0..p_time, p_time + p_t..p_time + p_t + p_ls])
        .to_owned();
    let cov_tl = covariance
        .slice(s![p_time..p_time + p_t, p_time + p_t..p_time + p_t + p_ls])
        .to_owned();
    let xh_hh = input.x_time_exit.dot(&cov_hh);
    let var_h_rows =
        Array1::from_iter((0..n).map(|i| input.x_time_exit.row(i).dot(&xh_hh.row(i)).max(0.0)));
    let var_t_rows = input.x_threshold.quadratic_form_diag(&cov_tt)?;
    let var_ls_rows = input.x_log_sigma.quadratic_form_diag(&cov_ll)?;
    let cov_ht_rows =
        rowwise_cross_quadratic_dense_design(&input.x_time_exit, &cov_ht, &input.x_threshold)?;
    let cov_hl_rows =
        rowwise_cross_quadratic_dense_design(&input.x_time_exit, &cov_hl, &input.x_log_sigma)?;
    let cov_tl_rows =
        rowwise_cross_quadratic_design(&input.x_threshold, &cov_tl, &input.x_log_sigma)?;
    let (var_w_rows, cov_hw_rows, cov_tw_rows, cov_lw_rows) =
        if let Some(xw) = input.x_link_wiggle.as_ref() {
            let cov_ww = covariance
                .slice(s![
                    p_time + p_t + p_ls..p_total,
                    p_time + p_t + p_ls..p_total
                ])
                .to_owned();
            let cov_hw = covariance
                .slice(s![0..p_time, p_time + p_t + p_ls..p_total])
                .to_owned();
            let cov_tw = covariance
                .slice(s![p_time..p_time + p_t, p_time + p_t + p_ls..p_total])
                .to_owned();
            let cov_lw = covariance
                .slice(s![
                    p_time + p_t..p_time + p_t + p_ls,
                    p_time + p_t + p_ls..p_total
                ])
                .to_owned();
            (
                Some(xw.quadratic_form_diag(&cov_ww)?),
                Some(rowwise_cross_quadratic_dense_design(
                    &input.x_time_exit,
                    &cov_hw,
                    xw,
                )?),
                Some(rowwise_cross_quadratic_design(
                    &input.x_threshold,
                    &cov_tw,
                    xw,
                )?),
                Some(rowwise_cross_quadratic_design(
                    &input.x_log_sigma,
                    &cov_lw,
                    xw,
                )?),
            )
        } else {
            (None, None, None, None)
        };

    let mut eta_var = Array1::<f64>::zeros(n);
    for i in 0..n {
        let inv_sigma = 1.0 / sigma[i].max(1e-12);
        let coeff_ls = predictors.eta_t[i] * ds[i] / sigma[i].powi(2).max(1e-12);
        let mut acc = var_h_rows[i]
            + inv_sigma * inv_sigma * var_t_rows[i]
            + coeff_ls * coeff_ls * var_ls_rows[i]
            + 2.0 * inv_sigma * cov_ht_rows[i]
            - 2.0 * coeff_ls * cov_hl_rows[i]
            - 2.0 * inv_sigma * coeff_ls * cov_tl_rows[i];
        if let Some(var_w) = var_w_rows.as_ref() {
            acc += var_w[i];
        }
        if let Some(cov_hw) = cov_hw_rows.as_ref() {
            acc -= 2.0 * cov_hw[i];
        }
        if let Some(cov_tw) = cov_tw_rows.as_ref() {
            acc -= 2.0 * inv_sigma * cov_tw[i];
        }
        if let Some(cov_lw) = cov_lw_rows.as_ref() {
            acc += 2.0 * coeff_ls * cov_lw[i];
        }
        eta_var[i] = acc.max(0.0);
    }
    let eta_se = eta_var.mapv(f64::sqrt);

    let survival_prob = if posterior_mean {
        predict_survival_location_scale_posterior_mean(input, fit, covariance)?.survival_prob
    } else {
        base.survival_prob.clone()
    };

    let response_standard_error = if include_response_sd {
        let quad_ctx = crate::quadrature::QuadratureContext::new();
        Some(Array1::from_iter((0..n).map(|i| {
            let m2 = crate::quadrature::normal_expectation_1d_adaptive(
                &quad_ctx,
                base.eta[i],
                eta_se[i],
                |x| {
                    let p = inverse_link_survival_prob_value(&input.inverse_link, x);
                    p * p
                },
            );
            (m2 - survival_prob[i] * survival_prob[i]).max(0.0).sqrt()
        })))
    } else {
        None
    };

    Ok(SurvivalLocationScalePredictUncertaintyResult {
        eta: base.eta,
        survival_prob,
        eta_standard_error: eta_se,
        response_standard_error,
    })
}

#[inline]
fn var_h_row(i: usize, input: &SurvivalLocationScalePredictInput, xh_hh: &Array2<f64>) -> f64 {
    input.x_time_exit.row(i).dot(&xh_hh.row(i)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_family::evaluate_custom_family_joint_hyper;
    use crate::mixture_link::{
        state_from_beta_logistic_spec, state_from_sas_spec, state_from_spec,
    };
    use crate::types::{LinkComponent, MixtureLinkSpec, SasLinkSpec};
    use faer::sparse::{SparseColMat, Triplet};
    use ndarray::{Array1, array};

    fn sparse_design_from_dense(dense: &Array2<f64>) -> DesignMatrix {
        let mut triplets = Vec::new();
        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                let value = dense[[i, j]];
                if value != 0.0 {
                    triplets.push(Triplet::new(i, j, value));
                }
            }
        }
        DesignMatrix::from(
            SparseColMat::try_new_from_triplets(dense.nrows(), dense.ncols(), &triplets)
                .expect("build sparse design"),
        )
    }

    fn survival_exact_newton_test_family() -> SurvivalLocationScaleFamily {
        SurvivalLocationScaleFamily {
            n: 3,
            y: array![1.0, 0.0, 1.0],
            w: array![1.0, 0.8, 1.2],
            sigma_min: 0.3,
            sigma_max: 2.5,
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
            derivative_guard: 1e-8,
            derivative_softness: 1e-6,
            x_time_entry: array![[1.0], [1.0], [1.0]],
            x_time_exit: array![[1.2], [0.9], [1.4]],
            x_time_deriv: array![[1.0], [1.0], [1.0]],
            offset_time_deriv: array![0.5, 0.7, 0.6],
            x_threshold: DesignMatrix::Dense(array![[1.0], [0.4], [-0.6]]),
            x_log_sigma: DesignMatrix::Dense(array![[1.0], [-0.3], [0.5]]),
            x_link_wiggle: None,
        }
    }

    fn survival_exact_newton_test_family_with_inverse_link(
        inverse_link: InverseLink,
    ) -> SurvivalLocationScaleFamily {
        SurvivalLocationScaleFamily {
            inverse_link,
            ..survival_exact_newton_test_family()
        }
    }

    fn sparse_survival_exact_newton_test_family() -> SurvivalLocationScaleFamily {
        let mut family = survival_exact_newton_test_family();
        family.x_threshold = sparse_design_from_dense(&array![[1.0], [0.4], [-0.6]]);
        family.x_log_sigma = sparse_design_from_dense(&array![[1.0], [-0.3], [0.5]]);
        family
    }

    fn survival_exact_newton_test_states(beta_t: f64) -> Vec<ParameterBlockState> {
        vec![
            ParameterBlockState {
                beta: array![0.2],
                eta: array![0.1, 0.35, -0.2, 0.25, 0.6, 0.15, 0.5, 0.7, 0.6],
            },
            ParameterBlockState {
                beta: array![beta_t],
                eta: array![beta_t, 0.4 * beta_t, -0.6 * beta_t],
            },
            ParameterBlockState {
                beta: array![-0.15],
                eta: array![-0.15, 0.045, -0.075],
            },
        ]
    }

    fn survival_exact_newton_rebuild_states(
        beta_time: &Array1<f64>,
        beta_threshold: &Array1<f64>,
        beta_log_sigma: &Array1<f64>,
    ) -> Vec<ParameterBlockState> {
        vec![
            ParameterBlockState {
                beta: beta_time.clone(),
                eta: array![
                    beta_time[0],
                    beta_time[0],
                    beta_time[0],
                    1.2 * beta_time[0],
                    0.9 * beta_time[0],
                    1.4 * beta_time[0],
                    beta_time[0] + 0.5,
                    beta_time[0] + 0.7,
                    beta_time[0] + 0.6
                ],
            },
            ParameterBlockState {
                beta: beta_threshold.clone(),
                eta: array![
                    beta_threshold[0],
                    0.4 * beta_threshold[0],
                    -0.6 * beta_threshold[0]
                ],
            },
            ParameterBlockState {
                beta: beta_log_sigma.clone(),
                eta: array![
                    beta_log_sigma[0],
                    -0.3 * beta_log_sigma[0],
                    0.5 * beta_log_sigma[0]
                ],
            },
        ]
    }

    fn survival_outer_gradient_test_specs() -> Vec<ParameterBlockSpec> {
        vec![
            ParameterBlockSpec {
                name: "time_transform".to_string(),
                design: DesignMatrix::Dense(array![
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.2],
                    [0.9],
                    [1.4],
                    [1.0],
                    [1.0],
                    [1.0]
                ]),
                offset: Array1::zeros(9),
                penalties: vec![Array2::eye(1)],
                initial_log_lambdas: array![0.0],
                initial_beta: Some(array![0.2]),
            },
            ParameterBlockSpec {
                name: "threshold".to_string(),
                design: DesignMatrix::Dense(array![[1.0], [0.4], [-0.6]]),
                offset: Array1::zeros(3),
                penalties: Vec::new(),
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: Some(array![0.35]),
            },
            ParameterBlockSpec {
                name: "log_sigma".to_string(),
                design: DesignMatrix::Dense(array![[1.0], [-0.3], [0.5]]),
                offset: Array1::zeros(3),
                penalties: Vec::new(),
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: Some(array![-0.15]),
            },
        ]
    }

    fn survival_non_probit_test_links() -> Vec<(&'static str, InverseLink)> {
        vec![
            (
                "logistic",
                residual_distribution_inverse_link(ResidualDistribution::Logistic),
            ),
            (
                "cloglog",
                residual_distribution_inverse_link(ResidualDistribution::Gumbel),
            ),
            (
                "sas",
                InverseLink::Sas(
                    state_from_sas_spec(SasLinkSpec {
                        initial_epsilon: 0.1,
                        initial_log_delta: -0.2,
                    })
                    .expect("sas state"),
                ),
            ),
            (
                "beta-logistic",
                InverseLink::BetaLogistic(
                    state_from_beta_logistic_spec(SasLinkSpec {
                        initial_epsilon: 0.05,
                        initial_log_delta: 0.1,
                    })
                    .expect("beta-logistic state"),
                ),
            ),
        ]
    }

    #[test]
    fn identified_time_block_zeroes_anchor_row() {
        let design_entry = array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]];
        let design_exit = array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]];
        let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
        let time_block = TimeBlockInput {
            design_entry,
            design_exit,
            design_derivative_exit,
            offset_entry: Array1::zeros(3),
            offset_exit: Array1::zeros(3),
            derivative_offset_exit: Array1::zeros(3),
            penalties: vec![Array2::eye(3)],
            initial_log_lambdas: None,
            initial_beta: None,
        };
        let prepared = prepare_identified_time_block(&time_block, 0).expect("prepare time block");
        let entry_anchor_row = prepared.design_entry.row(0);
        let entry_max_abs = entry_anchor_row
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max);
        assert!(
            entry_max_abs <= 1e-10,
            "entry anchor row not zero after identifiability transform: max_abs={entry_max_abs}"
        );

        let exit_anchor_row = prepared.design_exit.row(0);
        let exit_max_abs = exit_anchor_row
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max);
        assert!(
            exit_max_abs > 1e-6,
            "test setup should keep exit anchor row distinct from zero to detect anchor mixups"
        );
    }

    #[test]
    fn select_anchor_row_defaults_to_earliest_entry() {
        let age_entry = array![5.0, 1.0, 3.0];
        let idx = select_anchor_row(&age_entry, None).expect("select default anchor");
        assert_eq!(idx, 1);
    }

    #[test]
    fn survival_ratio_derivatives_prefer_correct_signs() {
        let dists = [
            ResidualDistribution::Gaussian,
            ResidualDistribution::Gumbel,
            ResidualDistribution::Logistic,
        ];
        let zs = [-1.2, -0.5, 0.4, 0.6, 1.1];
        let h = 1e-6_f64;
        let tie_tol = 1e-12_f64;
        let nondeg_tol = 1e-12_f64;
        let mut saw_strict_dr = false;
        let mut saw_strict_ddr = false;

        for &dist in &dists {
            for &z in &zs {
                let r = |u: f64| {
                    let f = dist.pdf(u);
                    let s = 1.0 - dist.cdf(u);
                    f / s
                };
                let dr_plus = |u: f64| {
                    let f = dist.pdf(u);
                    let s = 1.0 - dist.cdf(u);
                    let fp = dist.pdf_derivative(u);
                    let ratio = f / s;
                    (ratio * ratio) + fp / s
                };
                let dr_minus = |u: f64| {
                    let f = dist.pdf(u);
                    let s = 1.0 - dist.cdf(u);
                    let fp = dist.pdf_derivative(u);
                    let ratio = f / s;
                    (ratio * ratio) - fp / s
                };
                let ddr_plus = |u: f64| {
                    let f = dist.pdf(u);
                    let s = 1.0 - dist.cdf(u);
                    let fp = dist.pdf_derivative(u);
                    let fpp = dist.pdf_second_derivative(u);
                    let ratio = f / s;
                    let dr = (ratio * ratio) + fp / s;
                    (2.0 * ratio * dr) + (fpp / s + fp * f / (s * s))
                };
                let ddr_minus = |u: f64| {
                    let f = dist.pdf(u);
                    let s = 1.0 - dist.cdf(u);
                    let fp = dist.pdf_derivative(u);
                    let fpp = dist.pdf_second_derivative(u);
                    let ratio = f / s;
                    let dr = (ratio * ratio) - fp / s;
                    (2.0 * ratio * dr) - (fpp / s + fp * f / (s * s))
                };

                let dr_fd = (r(z + h) - r(z - h)) / (2.0 * h);
                let ddr_fd = (dr_plus(z + h) - dr_plus(z - h)) / (2.0 * h);
                let dr_plus_err = (dr_plus(z) - dr_fd).abs();
                let dr_minus_err = (dr_minus(z) - dr_fd).abs();
                let ddr_plus_err = (ddr_plus(z) - ddr_fd).abs();
                let ddr_minus_err = (ddr_minus(z) - ddr_fd).abs();
                let f = dist.pdf(z);
                let s = 1.0 - dist.cdf(z);
                let fp = dist.pdf_derivative(z);
                let fpp = dist.pdf_second_derivative(z);
                let dr_signal = (fp / s).abs();
                let ddr_signal = (fpp / s + fp * f / (s * s)).abs();

                if dr_signal > nondeg_tol {
                    saw_strict_dr = true;
                    assert!(
                        dr_plus_err + tie_tol < dr_minus_err,
                        "dr sign check failed for {:?} at z={}: plus_err={}, minus_err={}, signal={}",
                        dist,
                        z,
                        dr_plus_err,
                        dr_minus_err,
                        dr_signal
                    );
                } else {
                    // At stationary points (fp≈0), plus/minus formulas coincide to first order.
                    assert!(
                        (dr_plus_err - dr_minus_err).abs() <= tie_tol,
                        "dr tie check failed for {:?} at z={}: plus_err={}, minus_err={}, signal={}",
                        dist,
                        z,
                        dr_plus_err,
                        dr_minus_err,
                        dr_signal
                    );
                }

                if ddr_signal > nondeg_tol {
                    saw_strict_ddr = true;
                    assert!(
                        ddr_plus_err + tie_tol < ddr_minus_err,
                        "ddr sign check failed for {:?} at z={}: plus_err={}, minus_err={}, signal={}",
                        dist,
                        z,
                        ddr_plus_err,
                        ddr_minus_err,
                        ddr_signal
                    );
                } else {
                    assert!(
                        (ddr_plus_err - ddr_minus_err).abs() <= tie_tol,
                        "ddr tie check failed for {:?} at z={}: plus_err={}, minus_err={}, signal={}",
                        dist,
                        z,
                        ddr_plus_err,
                        ddr_minus_err,
                        ddr_signal
                    );
                }
            }
        }

        assert!(
            saw_strict_dr,
            "expected at least one non-degenerate dr check"
        );
        assert!(
            saw_strict_ddr,
            "expected at least one non-degenerate ddr check"
        );
    }

    #[test]
    fn survival_ratio_helper_matches_closed_form_identities() {
        let dists = [
            ResidualDistribution::Gaussian,
            ResidualDistribution::Gumbel,
            ResidualDistribution::Logistic,
        ];
        let zs = [-1.4, -0.7, -0.1, 0.3, 0.9, 1.4];

        for &dist in &dists {
            for &z in &zs {
                let f = dist.pdf(z);
                let s = (1.0 - dist.cdf(z)).clamp(MIN_PROB, 1.0);
                let fp = dist.pdf_derivative(z);
                let fpp = dist.pdf_second_derivative(z);

                let (r, dr) =
                    SurvivalLocationScaleFamily::survival_ratio_first_derivative(f, fp, s);
                let ddr = SurvivalLocationScaleFamily::survival_ratio_second_derivative(
                    r, dr, f, fp, fpp, s,
                );

                let r_expected = f / s;
                let dr_expected = (r_expected * r_expected) + fp / s;
                let ddr_expected = (2.0 * r_expected * dr_expected) + (fpp / s + fp * f / (s * s));

                assert!(
                    (r - r_expected).abs() <= 1e-14,
                    "r mismatch for {:?} at z={}: got {}, expected {}",
                    dist,
                    z,
                    r,
                    r_expected
                );
                assert!(
                    (dr - dr_expected).abs() <= 1e-12,
                    "dr mismatch for {:?} at z={}: got {}, expected {}",
                    dist,
                    z,
                    dr,
                    dr_expected
                );
                assert!(
                    (ddr - ddr_expected).abs() <= 1e-10,
                    "ddr mismatch for {:?} at z={}: got {}, expected {}",
                    dist,
                    z,
                    ddr,
                    ddr_expected
                );
            }
        }
    }

    #[test]
    fn residual_pdf_third_derivative_matches_second_derivative_fd() {
        let dists = [
            ResidualDistribution::Gaussian,
            ResidualDistribution::Gumbel,
            ResidualDistribution::Logistic,
        ];
        let zs = [-1.1, -0.4, 0.2, 0.9];
        let h = 1e-6_f64;

        for &dist in &dists {
            for &z in &zs {
                let fd = (dist.pdf_second_derivative(z + h) - dist.pdf_second_derivative(z - h))
                    / (2.0 * h);
                let analytic = dist.pdf_third_derivative(z);
                assert_eq!(
                    analytic.signum(),
                    fd.signum(),
                    "pdf''' sign mismatch for {:?} at z={}: analytic={} fd={}",
                    dist,
                    z,
                    analytic,
                    fd
                );
                assert!(
                    (analytic - fd).abs() < 5e-5,
                    "pdf''' mismatch for {:?} at z={}: analytic={} fd={}",
                    dist,
                    z,
                    analytic,
                    fd
                );
            }
        }
    }

    #[test]
    fn clamped_log_pdf_derivatives_are_zero_in_saturated_region() {
        let f = MIN_PROB * 0.1;
        let fp = 3.0;
        let fpp = -7.0;
        let fppp = 11.0;
        let (logf, d1, d2, d3) =
            SurvivalLocationScaleFamily::clamped_log_pdf_with_derivatives(f, fp, fpp, fppp);
        assert!((logf - MIN_PROB.ln()).abs() <= 1e-15);
        assert_eq!(d1, 0.0);
        assert_eq!(d2, 0.0);
        assert_eq!(d3, 0.0);
    }

    #[test]
    fn clamped_survival_neglog_derivatives_are_zero_on_clamp_bounds() {
        // Lower clamp active.
        let (s_low, r_low, dr_low, ddr_low) =
            SurvivalLocationScaleFamily::clamped_survival_neglog_derivatives(
                MIN_PROB * 0.1,
                0.2,
                -0.3,
                0.4,
            );
        assert_eq!(s_low, MIN_PROB);
        assert_eq!(r_low, 0.0);
        assert_eq!(dr_low, 0.0);
        assert_eq!(ddr_low, 0.0);

        // Upper clamp active.
        let (s_high, r_high, dr_high, ddr_high) =
            SurvivalLocationScaleFamily::clamped_survival_neglog_derivatives(1.1, 0.2, -0.3, 0.4);
        assert_eq!(s_high, 1.0);
        assert_eq!(r_high, 0.0);
        assert_eq!(dr_high, 0.0);
        assert_eq!(ddr_high, 0.0);
    }

    #[test]
    fn clamped_log_with_derivatives_is_flat_below_floor() {
        let (log_x, d1, d2, d3) =
            SurvivalLocationScaleFamily::clamped_log_with_derivatives(-0.25, 1e-12);
        assert!((log_x - 1e-12_f64.ln()).abs() <= 1e-15);
        assert_eq!(d1, 0.0);
        assert_eq!(d2, 0.0);
        assert_eq!(d3, 0.0);
    }

    #[test]
    fn inverse_link_survival_prob_complements_failure_prob() {
        let eta = 0.37;
        let failure = inverse_link_failure_prob_checked(
            &residual_distribution_inverse_link(ResidualDistribution::Gaussian),
            eta,
        )
        .expect("failure probability");
        let survival = inverse_link_survival_prob_checked(
            &residual_distribution_inverse_link(ResidualDistribution::Gaussian),
            eta,
        )
        .expect("survival probability");
        assert!((survival - (1.0 - failure)).abs() <= 1e-14);
    }

    #[test]
    fn lift_conditional_covariance_preserves_wiggle_block() {
        let z = array![[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]];
        let cov_reduced = array![
            [2.0, 0.1, 0.2, 0.3, 0.4],
            [0.1, 3.0, 0.5, 0.6, 0.7],
            [0.2, 0.5, 4.0, 0.8, 0.9],
            [0.3, 0.6, 0.8, 5.0, 1.1],
            [0.4, 0.7, 0.9, 1.1, 6.0],
        ];
        let lifted = lift_conditional_covariance(&cov_reduced, &z, 1, 1, 1);
        assert_eq!(lifted.dim(), (6, 6));
        assert!((lifted[[5, 5]] - 6.0).abs() <= 1e-12);
        assert!((lifted[[0, 5]] - 0.4).abs() <= 1e-12);
        assert!((lifted[[3, 5]] - 0.9).abs() <= 1e-12);
        assert!((lifted[[4, 5]] - 1.1).abs() <= 1e-12);
    }

    #[test]
    fn threshold_exact_newton_hessian_matches_negative_gradient_jacobian() {
        let family = survival_exact_newton_test_family();
        let beta_t = 0.35;
        let states = survival_exact_newton_test_states(beta_t);
        let eval = family.evaluate(&states).expect("evaluate at center");
        let BlockWorkingSet::ExactNewton { gradient, hessian } =
            &eval.block_working_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD]
        else {
            panic!("threshold block should use exact newton");
        };
        let hessian = hessian.to_dense();

        let eps = 1e-6;
        let eval_plus = family
            .evaluate(&survival_exact_newton_test_states(beta_t + eps))
            .expect("evaluate at beta + eps");
        let eval_minus = family
            .evaluate(&survival_exact_newton_test_states(beta_t - eps))
            .expect("evaluate at beta - eps");
        let grad_plus =
            match &eval_plus.block_working_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("threshold block should use exact newton"),
            };
        let grad_minus =
            match &eval_minus.block_working_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("threshold block should use exact newton"),
            };
        let fd_neg_grad_jac = -(grad_plus - grad_minus) / (2.0 * eps);

        assert!(
            (gradient[0]).is_finite() && hessian[[0, 0]].is_finite(),
            "non-finite threshold exact-newton quantities: grad={} hess={}",
            gradient[0],
            hessian[[0, 0]]
        );
        assert_eq!(
            hessian[[0, 0]].signum(),
            fd_neg_grad_jac.signum(),
            "threshold Hessian sign mismatch: analytic={} fd={}",
            hessian[[0, 0]],
            fd_neg_grad_jac
        );
        assert!(
            (hessian[[0, 0]] - fd_neg_grad_jac).abs() <= 1e-5,
            "threshold Hessian mismatch: analytic={} fd={}",
            hessian[[0, 0]],
            fd_neg_grad_jac
        );
    }

    #[test]
    fn exact_newton_block_directional_derivatives_match_fd_for_non_probit_links() {
        let extract_hessian = |eval: FamilyEvaluation, block_idx: usize| -> Array2<f64> {
            match &eval.block_working_sets[block_idx] {
                BlockWorkingSet::ExactNewton { hessian, .. } => hessian.to_dense(),
                BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton block"),
            }
        };

        let beta_time = array![0.2];
        let beta_threshold = array![0.35];
        let beta_log_sigma = array![-0.15];
        let eps = 1e-6;

        for (label, inverse_link) in survival_non_probit_test_links() {
            let family = survival_exact_newton_test_family_with_inverse_link(inverse_link);
            let states =
                survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
            let base_eval = family.evaluate(&states).expect("base eval");

            for (block_idx, direction) in [
                (SurvivalLocationScaleFamily::BLOCK_TIME, array![1.0]),
                (SurvivalLocationScaleFamily::BLOCK_THRESHOLD, array![1.0]),
                (SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA, array![1.0]),
            ] {
                let analytic = family
                    .exact_newton_hessian_directional_derivative(&states, block_idx, &direction)
                    .expect("analytic dH")
                    .expect("expected exact dH");

                let mut beta_time_plus = beta_time.clone();
                let mut beta_threshold_plus = beta_threshold.clone();
                let mut beta_log_sigma_plus = beta_log_sigma.clone();
                match block_idx {
                    SurvivalLocationScaleFamily::BLOCK_TIME => {
                        beta_time_plus += &(eps * &direction);
                    }
                    SurvivalLocationScaleFamily::BLOCK_THRESHOLD => {
                        beta_threshold_plus += &(eps * &direction);
                    }
                    SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA => {
                        beta_log_sigma_plus += &(eps * &direction);
                    }
                    _ => panic!("unexpected block"),
                }

                let plus_states = survival_exact_newton_rebuild_states(
                    &beta_time_plus,
                    &beta_threshold_plus,
                    &beta_log_sigma_plus,
                );
                let h_plus =
                    extract_hessian(family.evaluate(&plus_states).expect("plus eval"), block_idx);
                let h_base = extract_hessian(base_eval.clone(), block_idx);
                let fd = (h_plus - h_base) / eps;
                crate::testing::assert_matrix_derivative_fd(
                    &fd,
                    &analytic,
                    5e-4,
                    &format!("survival {label} block {} dH", block_idx),
                );
            }
        }
    }

    #[test]
    fn outer_laml_gradient_matches_fd_for_non_probit_survival_links() {
        let specs = survival_outer_gradient_test_specs();
        let options = BlockwiseFitOptions {
            use_reml_objective: true,
            ridge_floor: 1e-10,
            outer_max_iter: 1,
            ..BlockwiseFitOptions::default()
        };
        let rho = array![0.12];
        let derivative_blocks = vec![Vec::new(), Vec::new(), Vec::new()];
        let h = 1e-5;

        for (label, inverse_link) in survival_non_probit_test_links() {
            let family = survival_exact_newton_test_family_with_inverse_link(inverse_link);
            let center = evaluate_custom_family_joint_hyper(
                &family,
                &specs,
                &options,
                &rho,
                &derivative_blocks,
                None,
                false,
            )
            .expect("center outer objective/gradient");
            assert!(center.objective.is_finite());
            assert_eq!(center.gradient.len(), rho.len());

            for k in 0..rho.len() {
                let mut rho_p = rho.clone();
                let mut rho_m = rho.clone();
                rho_p[k] += h;
                rho_m[k] -= h;
                let fp = evaluate_custom_family_joint_hyper(
                    &family,
                    &specs,
                    &options,
                    &rho_p,
                    &derivative_blocks,
                    Some(&center.warm_start),
                    false,
                )
                .expect("objective+")
                .objective;
                let fm = evaluate_custom_family_joint_hyper(
                    &family,
                    &specs,
                    &options,
                    &rho_m,
                    &derivative_blocks,
                    Some(&center.warm_start),
                    false,
                )
                .expect("objective-")
                .objective;
                let g_fd = (fp - fm) / (2.0 * h);
                let abs_err = (center.gradient[k] - g_fd).abs();
                let rel = abs_err / g_fd.abs().max(1e-8);
                assert_eq!(
                    center.gradient[k].signum(),
                    g_fd.signum(),
                    "outer survival LAML gradient sign mismatch for {} at {}: analytic={} fd={}",
                    label,
                    k,
                    center.gradient[k],
                    g_fd
                );
                assert!(
                    abs_err < 2e-4 || rel < 2e-2,
                    "outer survival LAML gradient mismatch for {} at {}: analytic={} fd={} abs={} rel={}",
                    label,
                    k,
                    center.gradient[k],
                    g_fd,
                    abs_err,
                    rel
                );
            }
        }
    }

    #[test]
    fn posterior_mean_prediction_matches_deterministic_when_covariance_is_zero() {
        let input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.2],
            x_threshold: DesignMatrix::Dense(array![[1.0, -0.2]]),
            eta_threshold_offset: array![0.0],
            x_log_sigma: DesignMatrix::Dense(array![[1.0, 0.3]]),
            eta_log_sigma_offset: array![0.0],
            x_link_wiggle: None,
            sigma_min: 0.1,
            sigma_max: 2.0,
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        };
        let fit = SurvivalLocationScaleFitResult {
            beta_time: array![0.4, -0.1],
            beta_threshold: array![0.2, 0.3],
            beta_log_sigma: array![-0.5, 0.1],
            beta_link_wiggle: None,
            lambdas_time: Array1::zeros(0),
            lambdas_threshold: Array1::zeros(0),
            lambdas_log_sigma: Array1::zeros(0),
            lambdas_link_wiggle: None,
            log_likelihood: 0.0,
            penalized_objective: 0.0,
            iterations: 0,
            final_grad_norm: 0.0,
            converged: true,
            covariance_conditional: None,
        };
        let deterministic = predict_survival_location_scale(&input, &fit).expect("predict");
        let expected =
            inverse_link_survival_prob_checked(&input.inverse_link, deterministic.eta[0])
                .expect("expected survival");
        assert!((deterministic.survival_prob[0] - expected).abs() <= 1e-12);
        let posterior =
            predict_survival_location_scale_posterior_mean(&input, &fit, &Array2::zeros((6, 6)))
                .expect("posterior mean");
        assert!((deterministic.survival_prob[0] - posterior.survival_prob[0]).abs() <= 1e-10);
    }

    #[test]
    fn sparse_exact_newton_matches_dense_working_sets() {
        let dense_family = survival_exact_newton_test_family();
        let sparse_family = sparse_survival_exact_newton_test_family();
        let states = survival_exact_newton_test_states(0.35);

        let dense_eval = dense_family.evaluate(&states).expect("dense evaluate");
        let sparse_eval = sparse_family.evaluate(&states).expect("sparse evaluate");
        assert!((dense_eval.log_likelihood - sparse_eval.log_likelihood).abs() <= 1e-12);
        assert_eq!(
            dense_eval.block_working_sets.len(),
            sparse_eval.block_working_sets.len()
        );
        for (dense_block, sparse_block) in dense_eval
            .block_working_sets
            .iter()
            .zip(sparse_eval.block_working_sets.iter())
        {
            match (dense_block, sparse_block) {
                (
                    BlockWorkingSet::ExactNewton {
                        gradient: dense_g,
                        hessian: dense_h,
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: sparse_g,
                        hessian: sparse_h,
                    },
                ) => {
                    let dense_h = dense_h.to_dense();
                    let sparse_h = sparse_h.to_dense();
                    assert_eq!(dense_g.len(), sparse_g.len());
                    assert_eq!(dense_h.dim(), sparse_h.dim());
                    for i in 0..dense_g.len() {
                        assert!((dense_g[i] - sparse_g[i]).abs() <= 1e-12);
                    }
                    for i in 0..dense_h.nrows() {
                        for j in 0..dense_h.ncols() {
                            assert!((dense_h[[i, j]] - sparse_h[[i, j]]).abs() <= 1e-12);
                        }
                    }
                }
                _ => panic!("expected exact-newton blocks"),
            }
        }

        let direction = array![0.2];
        let dense_dh = dense_family
            .exact_newton_hessian_directional_derivative(&states, 1, &direction)
            .expect("dense directional derivative")
            .expect("dense threshold directional derivative");
        let sparse_dh = sparse_family
            .exact_newton_hessian_directional_derivative(&states, 1, &direction)
            .expect("sparse directional derivative")
            .expect("sparse threshold directional derivative");
        assert_eq!(dense_dh.dim(), sparse_dh.dim());
        for i in 0..dense_dh.nrows() {
            for j in 0..dense_dh.ncols() {
                assert!((dense_dh[[i, j]] - sparse_dh[[i, j]]).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn prediction_applies_threshold_and_log_sigma_offsets() {
        let fit = SurvivalLocationScaleFitResult {
            beta_time: array![0.4, -0.1],
            beta_threshold: array![0.2, 0.3],
            beta_log_sigma: array![-0.5, 0.1],
            beta_link_wiggle: None,
            lambdas_time: Array1::zeros(0),
            lambdas_threshold: Array1::zeros(0),
            lambdas_log_sigma: Array1::zeros(0),
            lambdas_link_wiggle: None,
            log_likelihood: 0.0,
            penalized_objective: 0.0,
            iterations: 0,
            final_grad_norm: 0.0,
            converged: true,
            covariance_conditional: None,
        };
        let input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.2],
            x_threshold: DesignMatrix::Dense(array![[1.0, -0.2]]),
            eta_threshold_offset: array![0.7],
            x_log_sigma: DesignMatrix::Dense(array![[1.0, 0.3]]),
            eta_log_sigma_offset: array![0.4],
            x_link_wiggle: None,
            sigma_min: 0.1,
            sigma_max: 2.0,
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        };
        let pred = predict_survival_location_scale(&input, &fit).expect("predict");

        let eta_t = array![1.0, -0.2].dot(&fit.beta_threshold) + input.eta_threshold_offset[0];
        let eta_ls = array![1.0, 0.3].dot(&fit.beta_log_sigma) + input.eta_log_sigma_offset[0];
        let sigma =
            bounded_sigma_derivs_up_to_third_scalar(eta_ls, input.sigma_min, input.sigma_max).0;
        let h = array![1.0, 0.5].dot(&fit.beta_time) + input.eta_time_offset_exit[0];
        let expected_eta = -h - eta_t / sigma.max(1e-12);
        let expected_survival =
            inverse_link_survival_prob_checked(&input.inverse_link, expected_eta)
                .expect("expected survival");

        assert!((pred.eta[0] - expected_eta).abs() <= 1e-12);
        assert!((pred.survival_prob[0] - expected_survival).abs() <= 1e-12);
    }

    #[test]
    fn sparse_prediction_and_uncertainty_match_dense() {
        let fit = SurvivalLocationScaleFitResult {
            beta_time: array![0.4, -0.1],
            beta_threshold: array![0.2, 0.3],
            beta_log_sigma: array![-0.5, 0.1],
            beta_link_wiggle: Some(array![0.05, -0.02]),
            lambdas_time: Array1::zeros(0),
            lambdas_threshold: Array1::zeros(0),
            lambdas_log_sigma: Array1::zeros(0),
            lambdas_link_wiggle: Some(Array1::zeros(0)),
            log_likelihood: 0.0,
            penalized_objective: 0.0,
            iterations: 0,
            final_grad_norm: 0.0,
            converged: true,
            covariance_conditional: None,
        };
        let x_threshold_dense = array![[1.0, -0.2], [0.0, 0.6]];
        let x_log_sigma_dense = array![[1.0, 0.3], [0.0, -0.4]];
        let x_wiggle_dense = array![[1.0, 0.1], [0.0, -0.2]];
        let dense_input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5], [1.0, -0.3]],
            eta_time_offset_exit: array![0.2, -0.1],
            x_threshold: DesignMatrix::Dense(x_threshold_dense.clone()),
            eta_threshold_offset: array![0.7, -0.2],
            x_log_sigma: DesignMatrix::Dense(x_log_sigma_dense.clone()),
            eta_log_sigma_offset: array![0.4, 0.1],
            x_link_wiggle: Some(DesignMatrix::Dense(x_wiggle_dense.clone())),
            sigma_min: 0.1,
            sigma_max: 2.0,
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        };
        let sparse_input = SurvivalLocationScalePredictInput {
            x_threshold: sparse_design_from_dense(&x_threshold_dense),
            x_log_sigma: sparse_design_from_dense(&x_log_sigma_dense),
            x_link_wiggle: Some(sparse_design_from_dense(&x_wiggle_dense)),
            ..dense_input.clone()
        };
        let covariance = array![
            [0.03, 0.01, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
            [0.01, 0.02, 0.0, 0.0, 0.0, 0.0, -0.005, 0.0],
            [0.0, 0.0, 0.04, 0.01, 0.0, 0.0, 0.006, 0.001],
            [0.0, 0.0, 0.01, 0.03, 0.0, 0.0, -0.004, 0.002],
            [0.0, 0.0, 0.0, 0.0, 0.02, 0.005, 0.003, 0.001],
            [0.0, 0.0, 0.0, 0.0, 0.005, 0.02, -0.002, 0.004],
            [0.01, -0.005, 0.006, -0.004, 0.003, -0.002, 0.025, 0.006],
            [0.0, 0.0, 0.001, 0.002, 0.001, 0.004, 0.006, 0.018],
        ];

        let dense_pred =
            predict_survival_location_scale(&dense_input, &fit).expect("dense predict");
        let sparse_pred =
            predict_survival_location_scale(&sparse_input, &fit).expect("sparse predict");
        assert_eq!(dense_pred.eta.len(), sparse_pred.eta.len());
        for i in 0..dense_pred.eta.len() {
            assert!((dense_pred.eta[i] - sparse_pred.eta[i]).abs() <= 1e-12);
            assert!((dense_pred.survival_prob[i] - sparse_pred.survival_prob[i]).abs() <= 1e-12);
        }

        let dense_unc = predict_survival_location_scale_with_uncertainty(
            &dense_input,
            &fit,
            &covariance,
            false,
            true,
        )
        .expect("dense uncertainty");
        let sparse_unc = predict_survival_location_scale_with_uncertainty(
            &sparse_input,
            &fit,
            &covariance,
            false,
            true,
        )
        .expect("sparse uncertainty");
        for i in 0..dense_unc.eta.len() {
            assert!((dense_unc.eta[i] - sparse_unc.eta[i]).abs() <= 1e-12);
            assert!((dense_unc.survival_prob[i] - sparse_unc.survival_prob[i]).abs() <= 1e-12);
            assert!(
                (dense_unc.eta_standard_error[i] - sparse_unc.eta_standard_error[i]).abs() <= 1e-12
            );
            let dense_sd = dense_unc
                .response_standard_error
                .as_ref()
                .expect("dense response sd")[i];
            let sparse_sd = sparse_unc
                .response_standard_error
                .as_ref()
                .expect("sparse response sd")[i];
            assert!((dense_sd - sparse_sd).abs() <= 1e-12);
        }

        let dense_pm =
            predict_survival_location_scale_posterior_mean(&dense_input, &fit, &covariance)
                .expect("dense wiggle posterior mean");
        let sparse_pm =
            predict_survival_location_scale_posterior_mean(&sparse_input, &fit, &covariance)
                .expect("sparse wiggle posterior mean");
        for i in 0..dense_pm.eta.len() {
            assert!((dense_pm.eta[i] - sparse_pm.eta[i]).abs() <= 1e-12);
            assert!((dense_pm.survival_prob[i] - sparse_pm.survival_prob[i]).abs() <= 1e-10);
        }
    }

    #[test]
    fn gaussian_posterior_mean_reduction_matches_3d_ghq_small_case() {
        let input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.1],
            x_threshold: DesignMatrix::Dense(array![[1.0, 0.25]]),
            eta_threshold_offset: array![0.0],
            x_log_sigma: DesignMatrix::Dense(array![[1.0, -0.15]]),
            eta_log_sigma_offset: array![0.0],
            x_link_wiggle: None,
            sigma_min: 0.2,
            sigma_max: 1.8,
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        };
        let fit = SurvivalLocationScaleFitResult {
            beta_time: array![0.3, -0.2],
            beta_threshold: array![0.1, 0.2],
            beta_log_sigma: array![-0.4, 0.15],
            beta_link_wiggle: None,
            lambdas_time: Array1::zeros(0),
            lambdas_threshold: Array1::zeros(0),
            lambdas_log_sigma: Array1::zeros(0),
            lambdas_link_wiggle: None,
            log_likelihood: 0.0,
            penalized_objective: 0.0,
            iterations: 0,
            final_grad_norm: 0.0,
            converged: true,
            covariance_conditional: None,
        };
        let covariance = array![
            [0.03, 0.01, 0.0, 0.0, 0.0, 0.0],
            [0.01, 0.02, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.04, 0.01, 0.0, 0.0],
            [0.0, 0.0, 0.01, 0.03, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.02, 0.005],
            [0.0, 0.0, 0.0, 0.0, 0.005, 0.02],
        ];
        let reduced = predict_survival_location_scale_posterior_mean(&input, &fit, &covariance)
            .expect("reduced posterior mean");

        let mu_h = input.x_time_exit.row(0).dot(&fit.beta_time) + input.eta_time_offset_exit[0];
        let x_t = input.x_threshold.to_dense_arc();
        let x_ls = input.x_log_sigma.to_dense_arc();
        let mu_t = x_t.row(0).dot(&fit.beta_threshold);
        let mu_ls = x_ls.row(0).dot(&fit.beta_log_sigma);
        let cov_hh = covariance.slice(s![0..2, 0..2]).to_owned();
        let cov_tt = covariance.slice(s![2..4, 2..4]).to_owned();
        let cov_ll = covariance.slice(s![4..6, 4..6]).to_owned();
        let cov_ht = covariance.slice(s![0..2, 2..4]).to_owned();
        let cov_hl = covariance.slice(s![0..2, 4..6]).to_owned();
        let cov_tl = covariance.slice(s![2..4, 4..6]).to_owned();
        let var_h = input
            .x_time_exit
            .row(0)
            .dot(&cov_hh.dot(&input.x_time_exit.row(0).to_owned()));
        let var_t = x_t.row(0).dot(&cov_tt.dot(&x_t.row(0).to_owned()));
        let var_ls = x_ls.row(0).dot(&cov_ll.dot(&x_ls.row(0).to_owned()));
        let cov_ht_i = input
            .x_time_exit
            .row(0)
            .dot(&cov_ht.dot(&x_t.row(0).to_owned()));
        let cov_hl_i = input
            .x_time_exit
            .row(0)
            .dot(&cov_hl.dot(&x_ls.row(0).to_owned()));
        let cov_tl_i = x_t.row(0).dot(&cov_tl.dot(&x_ls.row(0).to_owned()));
        let quad_ctx = crate::quadrature::QuadratureContext::new();
        let ghq = crate::quadrature::normal_expectation_3d_adaptive(
            &quad_ctx,
            [mu_h, mu_t, mu_ls],
            [
                [var_h, cov_ht_i, cov_hl_i],
                [cov_ht_i, var_t, cov_tl_i],
                [cov_hl_i, cov_tl_i, var_ls],
            ],
            |h, t, ls| {
                let sigma =
                    bounded_sigma_derivs_up_to_third_scalar(ls, input.sigma_min, input.sigma_max).0;
                (1.0 - normal_cdf(-h - t / sigma.max(1e-12))).clamp(0.0, 1.0)
            },
        );
        assert!((reduced.survival_prob[0] - ghq).abs() <= 2e-4);
    }

    #[test]
    fn sparse_posterior_mean_matches_dense() {
        let x_threshold_dense = array![[1.0, 0.25], [0.0, -0.1]];
        let x_log_sigma_dense = array![[1.0, -0.15], [0.0, 0.2]];
        let dense_input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5], [1.0, -0.4]],
            eta_time_offset_exit: array![0.1, -0.2],
            x_threshold: DesignMatrix::Dense(x_threshold_dense.clone()),
            eta_threshold_offset: array![0.0, 0.05],
            x_log_sigma: DesignMatrix::Dense(x_log_sigma_dense.clone()),
            eta_log_sigma_offset: array![0.0, -0.03],
            x_link_wiggle: None,
            sigma_min: 0.2,
            sigma_max: 1.8,
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        };
        let sparse_input = SurvivalLocationScalePredictInput {
            x_threshold: sparse_design_from_dense(&x_threshold_dense),
            x_log_sigma: sparse_design_from_dense(&x_log_sigma_dense),
            ..dense_input.clone()
        };
        let fit = SurvivalLocationScaleFitResult {
            beta_time: array![0.3, -0.2],
            beta_threshold: array![0.1, 0.2],
            beta_log_sigma: array![-0.4, 0.15],
            beta_link_wiggle: None,
            lambdas_time: Array1::zeros(0),
            lambdas_threshold: Array1::zeros(0),
            lambdas_log_sigma: Array1::zeros(0),
            lambdas_link_wiggle: None,
            log_likelihood: 0.0,
            penalized_objective: 0.0,
            iterations: 0,
            final_grad_norm: 0.0,
            converged: true,
            covariance_conditional: None,
        };
        let covariance = array![
            [0.03, 0.01, 0.0, 0.0, 0.0, 0.0],
            [0.01, 0.02, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.04, 0.01, 0.0, 0.0],
            [0.0, 0.0, 0.01, 0.03, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.02, 0.005],
            [0.0, 0.0, 0.0, 0.0, 0.005, 0.02],
        ];

        let dense_pm =
            predict_survival_location_scale_posterior_mean(&dense_input, &fit, &covariance)
                .expect("dense posterior mean");
        let sparse_pm =
            predict_survival_location_scale_posterior_mean(&sparse_input, &fit, &covariance)
                .expect("sparse posterior mean");
        for i in 0..dense_pm.eta.len() {
            assert!((dense_pm.eta[i] - sparse_pm.eta[i]).abs() <= 1e-12);
            assert!((dense_pm.survival_prob[i] - sparse_pm.survival_prob[i]).abs() <= 1e-10);
        }
    }

    #[test]
    fn wiggle_posterior_mean_reduction_matches_4d_ghq_small_case() {
        let input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.2],
            x_threshold: DesignMatrix::Dense(array![[1.0, -0.2]]),
            eta_threshold_offset: array![0.0],
            x_log_sigma: DesignMatrix::Dense(array![[1.0, 0.3]]),
            eta_log_sigma_offset: array![0.0],
            x_link_wiggle: Some(DesignMatrix::Dense(array![[1.0, 0.1]])),
            sigma_min: 0.1,
            sigma_max: 2.0,
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        };
        let fit = SurvivalLocationScaleFitResult {
            beta_time: array![0.4, -0.1],
            beta_threshold: array![0.2, 0.3],
            beta_log_sigma: array![-0.5, 0.1],
            beta_link_wiggle: Some(array![0.05, -0.02]),
            lambdas_time: Array1::zeros(0),
            lambdas_threshold: Array1::zeros(0),
            lambdas_log_sigma: Array1::zeros(0),
            lambdas_link_wiggle: Some(Array1::zeros(0)),
            log_likelihood: 0.0,
            penalized_objective: 0.0,
            iterations: 0,
            final_grad_norm: 0.0,
            converged: true,
            covariance_conditional: None,
        };
        let covariance = array![
            [0.03, 0.01, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
            [0.01, 0.02, 0.0, 0.0, 0.0, 0.0, -0.005, 0.0],
            [0.0, 0.0, 0.04, 0.01, 0.0, 0.0, 0.006, 0.001],
            [0.0, 0.0, 0.01, 0.03, 0.0, 0.0, -0.004, 0.002],
            [0.0, 0.0, 0.0, 0.0, 0.02, 0.005, 0.003, 0.001],
            [0.0, 0.0, 0.0, 0.0, 0.005, 0.02, -0.002, 0.004],
            [0.01, -0.005, 0.006, -0.004, 0.003, -0.002, 0.025, 0.006],
            [0.0, 0.0, 0.001, 0.002, 0.001, 0.004, 0.006, 0.018],
        ];
        let reduced = predict_survival_location_scale_posterior_mean(&input, &fit, &covariance)
            .expect("wiggle posterior mean");

        let x_t = input.x_threshold.to_dense_arc();
        let x_ls = input.x_log_sigma.to_dense_arc();
        let x_w = input
            .x_link_wiggle
            .as_ref()
            .expect("wiggle design")
            .to_dense_arc();
        let mu_h = input.x_time_exit.row(0).dot(&fit.beta_time) + input.eta_time_offset_exit[0];
        let mu_t = x_t.row(0).dot(&fit.beta_threshold);
        let mu_ls = x_ls.row(0).dot(&fit.beta_log_sigma);
        let mu_w = x_w
            .row(0)
            .dot(fit.beta_link_wiggle.as_ref().expect("wiggle beta"));
        let cov_hh = covariance.slice(s![0..2, 0..2]).to_owned();
        let cov_tt = covariance.slice(s![2..4, 2..4]).to_owned();
        let cov_ll = covariance.slice(s![4..6, 4..6]).to_owned();
        let cov_ww = covariance.slice(s![6..8, 6..8]).to_owned();
        let cov_ht = covariance.slice(s![0..2, 2..4]).to_owned();
        let cov_hl = covariance.slice(s![0..2, 4..6]).to_owned();
        let cov_hw = covariance.slice(s![0..2, 6..8]).to_owned();
        let cov_tl = covariance.slice(s![2..4, 4..6]).to_owned();
        let cov_tw = covariance.slice(s![2..4, 6..8]).to_owned();
        let cov_lw = covariance.slice(s![4..6, 6..8]).to_owned();
        let var_h = input
            .x_time_exit
            .row(0)
            .dot(&cov_hh.dot(&input.x_time_exit.row(0).to_owned()));
        let var_t = x_t.row(0).dot(&cov_tt.dot(&x_t.row(0).to_owned()));
        let var_ls = x_ls.row(0).dot(&cov_ll.dot(&x_ls.row(0).to_owned()));
        let var_w = x_w.row(0).dot(&cov_ww.dot(&x_w.row(0).to_owned()));
        let cov_ht_i = input
            .x_time_exit
            .row(0)
            .dot(&cov_ht.dot(&x_t.row(0).to_owned()));
        let cov_hl_i = input
            .x_time_exit
            .row(0)
            .dot(&cov_hl.dot(&x_ls.row(0).to_owned()));
        let cov_hw_i = input
            .x_time_exit
            .row(0)
            .dot(&cov_hw.dot(&x_w.row(0).to_owned()));
        let cov_tl_i = x_t.row(0).dot(&cov_tl.dot(&x_ls.row(0).to_owned()));
        let cov_tw_i = x_t.row(0).dot(&cov_tw.dot(&x_w.row(0).to_owned()));
        let cov_lw_i = x_ls.row(0).dot(&cov_lw.dot(&x_w.row(0).to_owned()));
        let quad_ctx = crate::quadrature::QuadratureContext::new();
        let ghq = crate::quadrature::normal_expectation_nd_adaptive::<4, _>(
            &quad_ctx,
            [mu_h, mu_t, mu_ls, mu_w],
            [
                [var_h, cov_ht_i, cov_hl_i, cov_hw_i],
                [cov_ht_i, var_t, cov_tl_i, cov_tw_i],
                [cov_hl_i, cov_tl_i, var_ls, cov_lw_i],
                [cov_hw_i, cov_tw_i, cov_lw_i, var_w],
            ],
            11,
            |x| {
                let sigma =
                    bounded_sigma_derivs_up_to_third_scalar(x[2], input.sigma_min, input.sigma_max)
                        .0
                        .max(1e-12);
                (1.0 - normal_cdf(-x[0] - x[1] / sigma + x[3])).clamp(0.0, 1.0)
            },
        );
        assert!((reduced.survival_prob[0] - ghq).abs() <= 2e-4);
    }

    #[test]
    fn predict_rejects_stateless_beta_logistic_inverse_link() {
        let fit = SurvivalLocationScaleFitResult {
            beta_time: array![0.4, -0.1],
            beta_threshold: array![0.2, 0.3],
            beta_log_sigma: array![-0.5, 0.1],
            beta_link_wiggle: None,
            lambdas_time: Array1::zeros(0),
            lambdas_threshold: Array1::zeros(0),
            lambdas_log_sigma: Array1::zeros(0),
            lambdas_link_wiggle: None,
            log_likelihood: 0.0,
            penalized_objective: 0.0,
            iterations: 0,
            final_grad_norm: 0.0,
            converged: true,
            covariance_conditional: None,
        };
        let input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.2],
            x_threshold: DesignMatrix::Dense(array![[1.0, -0.2]]),
            eta_threshold_offset: array![0.0],
            x_log_sigma: DesignMatrix::Dense(array![[1.0, 0.3]]),
            eta_log_sigma_offset: array![0.0],
            x_link_wiggle: None,
            sigma_min: 0.1,
            sigma_max: 2.0,
            inverse_link: InverseLink::Standard(LinkFunction::BetaLogistic),
        };

        let err = predict_survival_location_scale(&input, &fit)
            .err()
            .expect("should reject");
        assert!(err.contains("state-less Standard(BetaLogistic)"));
    }

    #[test]
    fn predict_supports_sas_beta_logistic_and_mixture_links() {
        let fit = SurvivalLocationScaleFitResult {
            beta_time: array![0.4, -0.1],
            beta_threshold: array![0.2, 0.3],
            beta_log_sigma: array![-0.5, 0.1],
            beta_link_wiggle: None,
            lambdas_time: Array1::zeros(0),
            lambdas_threshold: Array1::zeros(0),
            lambdas_log_sigma: Array1::zeros(0),
            lambdas_link_wiggle: None,
            log_likelihood: 0.0,
            penalized_objective: 0.0,
            iterations: 0,
            final_grad_norm: 0.0,
            converged: true,
            covariance_conditional: None,
        };
        let base = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.2],
            x_threshold: DesignMatrix::Dense(array![[1.0, -0.2]]),
            eta_threshold_offset: array![0.0],
            x_log_sigma: DesignMatrix::Dense(array![[1.0, 0.3]]),
            eta_log_sigma_offset: array![0.0],
            x_link_wiggle: None,
            sigma_min: 0.1,
            sigma_max: 2.0,
            inverse_link: InverseLink::Standard(LinkFunction::Probit),
        };

        let sas = InverseLink::Sas(
            state_from_sas_spec(SasLinkSpec {
                initial_epsilon: 0.1,
                initial_log_delta: -0.2,
            })
            .expect("sas state"),
        );
        let beta_logistic = InverseLink::BetaLogistic(
            state_from_beta_logistic_spec(SasLinkSpec {
                initial_epsilon: 0.05,
                initial_log_delta: 0.1,
            })
            .expect("beta-logistic state"),
        );
        let mixture = InverseLink::Mixture(
            state_from_spec(&MixtureLinkSpec {
                components: vec![LinkComponent::Probit, LinkComponent::Logit],
                initial_rho: array![0.2],
            })
            .expect("mixture state"),
        );

        for link in [sas, beta_logistic, mixture] {
            let mut input = base.clone();
            input.inverse_link = link;
            let pred = predict_survival_location_scale(&input, &fit).expect("predict");
            assert!(pred.survival_prob[0].is_finite());
            assert!(pred.survival_prob[0] > 0.0 && pred.survival_prob[0] < 1.0);
            let cov = Array2::eye(6) * 1e-3;
            let pm = predict_survival_location_scale_posterior_mean(&input, &fit, &cov)
                .expect("posterior mean");
            assert!(pm.survival_prob[0].is_finite());
            assert!(pm.survival_prob[0] > 0.0 && pm.survival_prob[0] < 1.0);
        }
    }
}
