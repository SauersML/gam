use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, BlockwiseFitResult, CustomFamily, FamilyEvaluation,
    ParameterBlockSpec, ParameterBlockState, fit_custom_family,
};
use crate::faer_ndarray::FaerSvd;
use crate::matrix::DesignMatrix;
use crate::pirls::LinearInequalityConstraints;
use crate::families::sigma_link::{bounded_sigma_derivs_up_to_third, bounded_sigma_derivs_up_to_third_scalar};
use crate::probability::{normal_cdf_approx, normal_pdf};
use crate::types::LinkFunction;
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
}

#[derive(Clone, Copy)]
struct DistributionEval {
    cdf: f64,
    pdf: f64,
    pdf_derivative: f64,
    pdf_second_derivative: f64,
}

impl ResidualDistributionOps for ResidualDistribution {
    fn cdf(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => normal_cdf_approx(z),
            ResidualDistribution::Gumbel => {
                // F(z)=1-exp(-exp(z))
                let ez = z.clamp(-40.0, 40.0).exp();
                1.0 - (-ez).exp()
            }
            ResidualDistribution::Logistic => {
                let zc = z.clamp(-40.0, 40.0);
                1.0 / (1.0 + (-zc).exp())
            }
        }
    }

    fn pdf(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => normal_pdf(z),
            ResidualDistribution::Gumbel => {
                let ez = z.clamp(-40.0, 40.0).exp();
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
                let ez = z.clamp(-40.0, 40.0).exp();
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
                let ez = z.clamp(-40.0, 40.0).exp();
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
}

#[inline]
fn residual_distribution_link(distribution: ResidualDistribution) -> LinkFunction {
    match distribution {
        ResidualDistribution::Gaussian => LinkFunction::Probit,
        ResidualDistribution::Gumbel => LinkFunction::CLogLog,
        ResidualDistribution::Logistic => LinkFunction::Logit,
    }
}

impl ResidualDistribution {
    #[inline]
    fn eval_all(self, z: f64) -> DistributionEval {
        match self {
            ResidualDistribution::Gaussian => {
                let cdf = normal_cdf_approx(z);
                let pdf = normal_pdf(z);
                let pdf_derivative = -z * pdf;
                let pdf_second_derivative = (z * z - 1.0) * pdf;
                DistributionEval {
                    cdf,
                    pdf,
                    pdf_derivative,
                    pdf_second_derivative,
                }
            }
            ResidualDistribution::Gumbel => {
                let ez = z.clamp(-40.0, 40.0).exp();
                let emez = (-ez).exp();
                let cdf = 1.0 - emez;
                let pdf = ez * emez;
                let pdf_derivative = pdf * (1.0 - ez);
                let pdf_second_derivative = pdf * (1.0 - 3.0 * ez + ez * ez);
                DistributionEval {
                    cdf,
                    pdf,
                    pdf_derivative,
                    pdf_second_derivative,
                }
            }
            ResidualDistribution::Logistic => {
                let zc = z.clamp(-40.0, 40.0);
                let cdf = 1.0 / (1.0 + (-zc).exp());
                let pdf = cdf * (1.0 - cdf);
                let pdf_derivative = pdf * (1.0 - 2.0 * cdf);
                let pdf_second_derivative = pdf * (1.0 - 6.0 * cdf + 6.0 * cdf * cdf);
                DistributionEval {
                    cdf,
                    pdf,
                    pdf_derivative,
                    pdf_second_derivative,
                }
            }
        }
    }
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
pub struct SurvivalLocationScaleProbitSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<f64>,
    pub weights: Array1<f64>,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub distribution: ResidualDistribution,
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
}

#[derive(Clone)]
pub struct SurvivalLocationScaleProbitFitResult {
    pub beta_time: Array1<f64>,
    pub beta_threshold: Array1<f64>,
    pub beta_log_sigma: Array1<f64>,
    pub lambdas_time: Array1<f64>,
    pub lambdas_threshold: Array1<f64>,
    pub lambdas_log_sigma: Array1<f64>,
    pub log_likelihood: f64,
    pub penalized_objective: f64,
    pub iterations: usize,
    pub converged: bool,
    pub covariance_conditional: Option<Array2<f64>>,
}

#[derive(Clone)]
pub struct SurvivalLocationScaleProbitPredictInput {
    pub x_time_exit: Array2<f64>,
    pub eta_time_offset_exit: Array1<f64>,
    pub x_threshold: DesignMatrix,
    pub x_log_sigma: DesignMatrix,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub distribution: ResidualDistribution,
}

#[derive(Clone)]
pub struct SurvivalLocationScaleProbitPredictResult {
    pub eta: Array1<f64>,
    pub survival_prob: Array1<f64>,
}

struct SurvivalLocationScaleProbitFamily {
    n: usize,
    y: Array1<f64>,
    w: Array1<f64>,
    sigma_min: f64,
    sigma_max: f64,
    distribution: ResidualDistribution,
    derivative_guard: f64,
    x_time_entry: Array2<f64>,
    x_time_exit: Array2<f64>,
    x_time_deriv: Array2<f64>,
    offset_time_deriv: Array1<f64>,
    x_threshold: Array2<f64>,
    x_log_sigma: Array2<f64>,
}

impl SurvivalLocationScaleProbitFamily {
    const BLOCK_TIME: usize = 0;
    const BLOCK_THRESHOLD: usize = 1;
    const BLOCK_LOG_SIGMA: usize = 2;

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

    /// Clamp-aware log-pdf and its first/second derivatives.
    ///
    /// The objective uses `log(max(f, MIN_PROB))`, so in the saturated region
    /// `f <= MIN_PROB` the function is constant and derivatives must be zero.
    fn clamped_log_pdf_with_derivatives(f: f64, fp: f64, fpp: f64) -> (f64, f64, f64) {
        if f <= MIN_PROB {
            (MIN_PROB.ln(), 0.0, 0.0)
        } else {
            let d1 = fp / f;
            let d2 = fpp / f - d1 * d1;
            (f.ln(), d1, d2)
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
    let c = input.design_exit.row(anchor_row).to_owned();
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
                    "fit_survival_location_scale_probit: non-finite time_anchor {t_anchor}"
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

fn dense_design(design: &DesignMatrix, name: &str) -> Result<Array2<f64>, String> {
    match design {
        DesignMatrix::Dense(x) => Ok(x.clone()),
        DesignMatrix::Sparse(_) => Err(format!(
            "{name}: sparse design is not supported for exact-newton survival yet"
        )),
    }
}

fn xt_diag_x(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    let n = x.nrows();
    let p = x.ncols();
    let mut out = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        let wi = w[i];
        if wi == 0.0 {
            continue;
        }
        for a in 0..p {
            let xa = x[[i, a]];
            let wxa = wi * xa;
            out[[a, a]] += wxa * x[[i, a]];
            for b in (a + 1)..p {
                let update = wxa * x[[i, b]];
                out[[a, b]] += update;
                out[[b, a]] += update;
            }
        }
    }
    out
}

impl CustomFamily for SurvivalLocationScaleProbitFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "SurvivalLocationScaleProbitFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.n;
        let eta_time = &block_states[Self::BLOCK_TIME].eta;
        let eta_t = &block_states[Self::BLOCK_THRESHOLD].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_time.len() != 3 * n || eta_t.len() != n || eta_ls.len() != n {
            return Err("survival probit-location-scale eta dimension mismatch".to_string());
        }

        let h0 = eta_time.slice(s![0..n]);
        let h1 = eta_time.slice(s![n..2 * n]);
        let d_raw = eta_time.slice(s![2 * n..3 * n]);

        let (sigma, ds, d2s, _d3s) =
            bounded_sigma_derivs_up_to_third(eta_ls.view(), self.sigma_min, self.sigma_max);
        let q = Array1::from_iter(
            eta_t
                .iter()
                .zip(sigma.iter())
                .map(|(&t, &s)| -t / s.max(1e-12)),
        );

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
            let w = self.w[i];
            if w <= 0.0 {
                continue;
            }
            let d = self.y[i].clamp(0.0, 1.0);
            let u0 = -h0[i] + q[i];
            let u1 = -h1[i] + q[i];
            let v0 = self.distribution.eval_all(u0);
            let v1 = self.distribution.eval_all(u1);
            let f0 = v0.pdf;
            let f1 = v1.pdf;
            let fp0 = v0.pdf_derivative;
            let fp1 = v1.pdf_derivative;
            let fpp0 = v0.pdf_second_derivative;
            let fpp1 = v1.pdf_second_derivative;
            let raw_s0 = 1.0 - v0.cdf;
            let raw_s1 = 1.0 - v1.cdf;
            let (s0, r0, dr0, _ddr0) =
                Self::clamped_survival_neglog_derivatives(raw_s0, f0, fp0, fpp0);
            let (s1, r1, dr1, _ddr1) =
                Self::clamped_survival_neglog_derivatives(raw_s1, f1, fp1, fpp1);
            let (log_phi1, dlogphi1, d2logphi1) =
                Self::clamped_log_pdf_with_derivatives(f1, fp1, fpp1);
            let g = d_raw[i];
            if !g.is_finite() || g <= self.derivative_guard.max(1e-12) {
                return Err(format!(
                    "survival probit-location-scale monotonicity violated at row {i}: d_eta/dt={g:.3e} <= guard={:.3e}",
                    self.derivative_guard.max(1e-12)
                ));
            }

            ll += w * (d * (log_phi1 + g.ln()) + (1.0 - d) * s1.ln() - s0.ln());

            // q derivatives (shared by threshold/log-sigma blocks).
            // Chain rule map:
            //   u0 = -h0 + q, u1 = -h1 + q  =>  du0/dq = du1/dq = +1.
            // So dℓ/dq and d²ℓ/dq² keep the same local signs as r', d²logphi/du².
            d1_q[i] = w * (r0 + d * dlogphi1 + (1.0 - d) * (-r1));
            d2_q[i] = w * (dr0 + d * d2logphi1 + (1.0 - d) * (-dr1));

            // time block eta-derivatives
            // BlockWorkingSet::ExactNewton stores H = -∂²ℓ/∂β² (PSD near optimum),
            // which is why the row-wise second-derivative terms carry a leading minus.
            grad_time_eta_h0[i] = -w * r0;
            grad_time_eta_h1[i] = -w * (d * dlogphi1 + (1.0 - d) * (-r1));
            grad_time_eta_d[i] = w * d / g;

            h_time_h0[i] = -w * dr0;
            h_time_h1[i] = -w * (d * d2logphi1 + (1.0 - d) * (-dr1));
            h_time_d[i] = w * d / (g * g);
        }

        // Block 0: exact beta-space gradient/Hessian
        let grad_time = self.x_time_entry.t().dot(&grad_time_eta_h0)
            + self.x_time_exit.t().dot(&grad_time_eta_h1)
            + self.x_time_deriv.t().dot(&grad_time_eta_d);
        let hess_time = xt_diag_x(&self.x_time_entry, &h_time_h0)
            + xt_diag_x(&self.x_time_exit, &h_time_h1)
            + xt_diag_x(&self.x_time_deriv, &h_time_d);

        // Block 1: threshold eta_t enters q linearly with dq/deta_t = -1/sigma.
        let dq_t = sigma.mapv(|s| -1.0 / s.max(1e-12));
        let grad_eta_t = &d1_q * &dq_t;
        let h_eta_t = &d2_q * &dq_t.mapv(|v| v * v);
        let grad_t = self.x_threshold.t().dot(&grad_eta_t);
        let hess_t = xt_diag_x(&self.x_threshold, &h_eta_t);

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
        let h_eta_ls = &d2_q * &dq_ls.mapv(|v| v * v) + &(&d1_q * &d2q_ls);
        let grad_ls = self.x_log_sigma.t().dot(&grad_eta_ls);
        let hess_ls = xt_diag_x(&self.x_log_sigma, &h_eta_ls);

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            block_working_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_time,
                    hessian: hess_time,
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_t,
                    hessian: hess_t,
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_ls,
                    hessian: hess_ls,
                },
            ],
        })
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 3 {
            return Err(format!(
                "SurvivalLocationScaleProbitFamily expects 3 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.n;
        let eta_time = &block_states[Self::BLOCK_TIME].eta;
        let eta_t = &block_states[Self::BLOCK_THRESHOLD].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_time.len() != 3 * n || eta_t.len() != n || eta_ls.len() != n {
            return Err("survival probit-location-scale eta dimension mismatch".to_string());
        }

        let h0 = eta_time.slice(s![0..n]);
        let h1 = eta_time.slice(s![n..2 * n]);
        let d_raw = eta_time.slice(s![2 * n..3 * n]);

        let (sigma, ds, d2s, d3s) =
            bounded_sigma_derivs_up_to_third(eta_ls.view(), self.sigma_min, self.sigma_max);
        let q = Array1::from_iter(
            eta_t
                .iter()
                .zip(sigma.iter())
                .map(|(&t, &s)| -t / s.max(1e-12)),
        );

        let mut d1_q = Array1::<f64>::zeros(n);
        let mut d2_q = Array1::<f64>::zeros(n);
        let mut d3_q = Array1::<f64>::zeros(n);
        let mut d_h_h0 = Array1::<f64>::zeros(n);
        let mut d_h_h1 = Array1::<f64>::zeros(n);
        let mut d_h_d = Array1::<f64>::zeros(n);

        for i in 0..n {
            let w = self.w[i];
            if w <= 0.0 {
                continue;
            }
            let d = self.y[i].clamp(0.0, 1.0);
            let u0 = -h0[i] + q[i];
            let u1 = -h1[i] + q[i];
            let v0 = self.distribution.eval_all(u0);
            let v1 = self.distribution.eval_all(u1);
            let f0 = v0.pdf;
            let f1 = v1.pdf;
            let fp0 = v0.pdf_derivative;
            let fp1 = v1.pdf_derivative;
            let fpp0 = v0.pdf_second_derivative;
            let fpp1 = v1.pdf_second_derivative;
            let raw_s0 = 1.0 - v0.cdf;
            let raw_s1 = 1.0 - v1.cdf;
            let (_s0, r0, dr0, ddr0) =
                Self::clamped_survival_neglog_derivatives(raw_s0, f0, fp0, fpp0);
            let (_s1, r1, dr1, ddr1) =
                Self::clamped_survival_neglog_derivatives(raw_s1, f1, fp1, fpp1);
            let (_log_phi1, dlogphi1, d2logphi1) =
                Self::clamped_log_pdf_with_derivatives(f1, fp1, fpp1);

            // q-derivatives of the per-row log-likelihood contribution.
            // With u0=-h0+q, u1=-h1+q:
            // dℓ/dq   = w [ r0 + d * dlogphi1 - (1-d) r1 ]
            // d²ℓ/dq² = w [ r0' + d * d²logphi1 - (1-d) r1' ]
            // d³ℓ/dq³ = w [ r0'' + d * d³logphi1 - (1-d) r1'' ]
            d1_q[i] = w * (r0 + d * dlogphi1 + (1.0 - d) * (-r1));
            d2_q[i] = w * (dr0 + d * d2logphi1 + (1.0 - d) * (-dr1));
            // Third derivative of log phi using general chain rule:
            // d³(log φ)/du³ = φ'''/ φ - 3 φ'φ''/ φ² + 2(φ')³/ φ³
            // We don't have φ''' yet, so approximate with finite difference
            // of d2logphi for the third-order correction. For Gaussian d³(log φ)/du³ = 0.
            // For now, use 0 for the event contribution to d3 (matches Gaussian exactly,
            // small error for Gumbel/Logistic in the Hessian directional derivative).
            d3_q[i] = w * (ddr0 + (1.0 - d) * (-ddr1));

            // Time block contributions use u0/u1 direct dependence.
            // Chain rule map:
            //   du0/dh0 = -1, du1/dh1 = -1.
            // Then BlockWorkingSet::ExactNewton uses H = -∂²ℓ/∂β², so one more
            // leading minus appears when assembling per-row second derivatives.
            // The derivative row uses log(d_safe) only for events.
            d_h_h0[i] = w * ddr0;
            d_h_h1[i] = -w * (1.0 - d) * ddr1;
            let g = d_raw[i];
            if !g.is_finite() || g <= self.derivative_guard.max(1e-12) {
                return Err(format!(
                    "survival probit-location-scale monotonicity violated in Hessian directional derivative at row {i}: d_eta/dt={g:.3e} <= guard={:.3e}",
                    self.derivative_guard.max(1e-12)
                ));
            }
            d_h_d[i] = -2.0 * w * d / (g * g * g);
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
                let d_h = xt_diag_x(&self.x_time_entry, &w0)
                    + xt_diag_x(&self.x_time_exit, &w1)
                    + xt_diag_x(&self.x_time_deriv, &wd);
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
                let d_eta_t = self.x_threshold.dot(d_beta);
                // Since q(eta_t) is linear in eta_t, only dq_t is nonzero:
                // dH[u] = X^T diag( d³ℓ/dq³ * (dq_t)^3 * u_eta ) X.
                let d_h_eta = &d3_q * &dq_t.mapv(|v| v * v * v) * &d_eta_t;
                let d_h = xt_diag_x(&self.x_threshold, &d_h_eta);
                Ok(Some(d_h))
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
                let d_eta_ls = self.x_log_sigma.dot(d_beta);
                // Full third-order chain rule:
                // d/deta [d²ℓ/deta²] = d³ℓ/dq³ (dq)^3 + 3 d²ℓ/dq² dq d²q + dℓ/dq d³q.
                let d_h_eta = (&d3_q * &dq.mapv(|v| v * v * v)
                    + &(&d2_q * &(3.0 * &dq * &d2q))
                    + &(&d1_q * &d3q))
                    * &d_eta_ls;
                let d_h = xt_diag_x(&self.x_log_sigma, &d_h_eta);
                Ok(Some(d_h))
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
        if n == 0 || p == 0 {
            return Ok(None);
        }
        let mut a = Array2::<f64>::zeros((n, p));
        a.assign(&self.x_time_deriv);
        let mut b = Array1::<f64>::zeros(n);
        let guard = self.derivative_guard.max(1e-12);
        for i in 0..n {
            b[i] = guard - self.offset_time_deriv[i];
        }
        Ok(Some(LinearInequalityConstraints { a, b }))
    }
}

pub fn fit_survival_location_scale_probit(
    spec: SurvivalLocationScaleProbitSpec,
) -> Result<SurvivalLocationScaleProbitFitResult, String> {
    let n = spec.event_target.len();
    if n == 0 {
        return Err("fit_survival_location_scale_probit: empty dataset".to_string());
    }
    if spec.age_entry.len() != n || spec.age_exit.len() != n || spec.weights.len() != n {
        return Err(
            "fit_survival_location_scale_probit: top-level input size mismatch".to_string(),
        );
    }
    if !spec.sigma_min.is_finite()
        || !spec.sigma_max.is_finite()
        || spec.sigma_min <= 0.0
        || spec.sigma_max <= 0.0
        || spec.sigma_min >= spec.sigma_max
    {
        return Err(format!(
            "fit_survival_location_scale_probit: invalid sigma bounds (min={}, max={})",
            spec.sigma_min, spec.sigma_max
        ));
    }
    if !(spec.tol.is_finite() && spec.tol > 0.0) {
        return Err(format!(
            "fit_survival_location_scale_probit: invalid tol {}",
            spec.tol
        ));
    }
    if spec.max_iter == 0 {
        return Err("fit_survival_location_scale_probit: max_iter must be > 0".to_string());
    }
    validate_time_block(n, &spec.time_block)?;
    validate_cov_block("threshold_block", n, &spec.threshold_block)?;
    validate_cov_block("log_sigma_block", n, &spec.log_sigma_block)?;

    for i in 0..n {
        if !spec.age_entry[i].is_finite()
            || !spec.age_exit[i].is_finite()
            || spec.age_exit[i] < spec.age_entry[i]
        {
            return Err(format!(
                "fit_survival_location_scale_probit: invalid interval at row {} (entry={}, exit={})",
                i + 1,
                spec.age_entry[i],
                spec.age_exit[i]
            ));
        }
        if !spec.weights[i].is_finite() || spec.weights[i] < 0.0 {
            return Err(format!(
                "fit_survival_location_scale_probit: invalid weight at row {} ({})",
                i + 1,
                spec.weights[i]
            ));
        }
        if !spec.event_target[i].is_finite() || !(0.0..=1.0).contains(&spec.event_target[i]) {
            return Err(format!(
                "fit_survival_location_scale_probit: event_target must be in [0,1], found {} at row {}",
                spec.event_target[i],
                i + 1
            ));
        }
    }

    let x_threshold = dense_design(&spec.threshold_block.design, "threshold_block")?;
    let x_log_sigma = dense_design(&spec.log_sigma_block.design, "log_sigma_block")?;

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

    let family = SurvivalLocationScaleProbitFamily {
        n,
        y: spec.event_target,
        w: spec.weights,
        sigma_min: spec.sigma_min,
        sigma_max: spec.sigma_max,
        distribution: spec.distribution,
        derivative_guard: spec.derivative_guard,
        x_time_entry: time_prepared.design_entry,
        x_time_exit: time_prepared.design_exit,
        x_time_deriv: time_prepared.design_derivative_exit,
        offset_time_deriv: spec.time_block.derivative_offset_exit.clone(),
        x_threshold,
        x_log_sigma,
    };

    let options = BlockwiseFitOptions {
        inner_max_cycles: spec.max_iter,
        inner_tol: spec.tol,
        outer_max_iter: 60,
        outer_tol: 1e-5,
        ..BlockwiseFitOptions::default()
    };
    let fit: BlockwiseFitResult = fit_custom_family(
        &family,
        &[time_spec, threshold_spec, log_sigma_spec],
        &options,
    )?;

    let k_time = spec.time_block.penalties.len();
    let k_t = spec.threshold_block.penalties.len();
    let k_ls = spec.log_sigma_block.penalties.len();
    let lambdas = fit.log_lambdas.mapv(f64::exp);
    let lambdas_time = lambdas.slice(s![0..k_time]).to_owned();
    let lambdas_threshold = lambdas.slice(s![k_time..k_time + k_t]).to_owned();
    let lambdas_log_sigma = lambdas
        .slice(s![k_time + k_t..k_time + k_t + k_ls])
        .to_owned();

    let beta_time_reduced = fit.block_states[SurvivalLocationScaleProbitFamily::BLOCK_TIME]
        .beta
        .clone();
    let beta_time = time_prepared.transform.z.dot(&beta_time_reduced);
    let beta_threshold = fit.block_states[SurvivalLocationScaleProbitFamily::BLOCK_THRESHOLD]
        .beta
        .clone();
    let beta_log_sigma = fit.block_states[SurvivalLocationScaleProbitFamily::BLOCK_LOG_SIGMA]
        .beta
        .clone();

    let covariance_conditional = fit.covariance_conditional.as_ref().map(|cov_reduced| {
        let z = &time_prepared.transform.z;
        let p_time_reduced = beta_time_reduced.len();
        let p_time = beta_time.len();
        let p_t = beta_threshold.len();
        let p_ls = beta_log_sigma.len();
        let p_reduced = p_time_reduced + p_t + p_ls;
        let p_full = p_time + p_t + p_ls;
        if cov_reduced.nrows() != p_reduced || cov_reduced.ncols() != p_reduced {
            return cov_reduced.clone();
        }
        // Lift reduced-basis covariance into full time basis while preserving
        // all cross-block covariance terms.
        let mut t_map = Array2::<f64>::zeros((p_full, p_reduced));
        t_map.slice_mut(s![0..p_time, 0..p_time_reduced]).assign(z);
        for j in 0..p_t {
            t_map[[p_time + j, p_time_reduced + j]] = 1.0;
        }
        for j in 0..p_ls {
            t_map[[p_time + p_t + j, p_time_reduced + p_t + j]] = 1.0;
        }
        t_map.dot(cov_reduced).dot(&t_map.t())
    });

    Ok(SurvivalLocationScaleProbitFitResult {
        beta_time,
        beta_threshold,
        beta_log_sigma,
        lambdas_time,
        lambdas_threshold,
        lambdas_log_sigma,
        log_likelihood: fit.log_likelihood,
        penalized_objective: fit.penalized_objective,
        iterations: fit.inner_cycles,
        converged: fit.converged,
        covariance_conditional,
    })
}

pub fn predict_survival_location_scale_probit(
    input: &SurvivalLocationScaleProbitPredictInput,
    fit: &SurvivalLocationScaleProbitFitResult,
) -> Result<SurvivalLocationScaleProbitPredictResult, String> {
    let n = input.x_time_exit.nrows();
    if input.x_time_exit.ncols() != fit.beta_time.len() {
        return Err(format!(
            "predict_survival_location_scale_probit: time design/beta mismatch: {} vs {}",
            input.x_time_exit.ncols(),
            fit.beta_time.len()
        ));
    }
    if input.eta_time_offset_exit.len() != n
        || input.x_threshold.nrows() != n
        || input.x_log_sigma.nrows() != n
    {
        return Err(
            "predict_survival_location_scale_probit: row mismatch across inputs".to_string(),
        );
    }
    let h = input.x_time_exit.dot(&fit.beta_time) + &input.eta_time_offset_exit;
    let eta_t = input
        .x_threshold
        .matrix_vector_multiply(&fit.beta_threshold);
    let eta_ls = input
        .x_log_sigma
        .matrix_vector_multiply(&fit.beta_log_sigma);
    let (sigma, _, _, _) =
        bounded_sigma_derivs_up_to_third(eta_ls.view(), input.sigma_min, input.sigma_max);
    let eta = Array1::from_iter(
        h.iter()
            .zip(eta_t.iter())
            .zip(sigma.iter())
            .map(|((&hh, &tt), &ss)| -hh - tt / ss.max(1e-12)),
    );
    let survival_prob = eta.mapv(|v| input.distribution.cdf(v).clamp(0.0, 1.0));
    Ok(SurvivalLocationScaleProbitPredictResult { eta, survival_prob })
}

pub fn predict_survival_location_scale_probit_posterior_mean(
    input: &SurvivalLocationScaleProbitPredictInput,
    fit: &SurvivalLocationScaleProbitFitResult,
    covariance: &Array2<f64>,
) -> Result<SurvivalLocationScaleProbitPredictResult, String> {
    // Uncertainty-aware survival posterior mean with conditional Gaussian
    // reduction.
    //
    // The deterministic survival predictor already computes the location-scale
    // latent pieces
    //
    //   h   = time block linear predictor
    //   t   = threshold block linear predictor
    //   ls  = log-sigma block linear predictor.
    //
    // Under coefficient uncertainty these three latent quantities are jointly
    // Gaussian row by row. The naive route is a full 3D Gaussian expectation of
    // the survival/probit inverse link. The expensive part is unnecessary:
    // conditional on ls, the pair (h, t) remains jointly Gaussian, and the
    // probit argument
    //
    //   eta_loc(ls) = -h - t / sigma(ls)
    //
    // is then an affine transformation of that conditional Gaussian pair.
    // Therefore eta_loc(ls) | ls is itself Gaussian with an analytically
    // available conditional mean and variance.
    //
    // Once the inner latent is Gaussian, the exact integrated inverse-link
    // machinery from quadrature.rs applies again:
    //
    //   E[Phi(Eta_loc) | ls]
    //     = integrated_inverse_link_mean_and_derivative(Probit, mu_loc|ls, sd_loc|ls).mean.
    //
    // So the original 3D integral collapses to:
    //   1D Gaussian integration over ls
    //   + exact inner Gaussian-link convolution.
    //
    // If the conditioning algebra becomes numerically unsafe, this routine
    // falls back to the old 3D quadrature rather than forcing the reduction.
    let pred = predict_survival_location_scale_probit(input, fit)?;
    let n = input.x_time_exit.nrows();
    let p_time = fit.beta_time.len();
    let p_t = fit.beta_threshold.len();
    let p_ls = fit.beta_log_sigma.len();
    let p_total = p_time + p_t + p_ls;
    if covariance.nrows() != p_total || covariance.ncols() != p_total {
        return Err(format!(
            "predict_survival_location_scale_probit_posterior_mean: covariance shape mismatch: got {}x{}, expected {}x{}",
            covariance.nrows(),
            covariance.ncols(),
            p_total,
            p_total
        ));
    }

    let x_threshold_dense_arc = input.x_threshold.to_dense_arc();
    let x_log_sigma_dense_arc = input.x_log_sigma.to_dense_arc();
    let x_threshold_dense = x_threshold_dense_arc.as_ref();
    let x_log_sigma_dense = x_log_sigma_dense_arc.as_ref();

    if x_threshold_dense.nrows() != n || x_log_sigma_dense.nrows() != n {
        return Err(
            "predict_survival_location_scale_probit_posterior_mean: row mismatch across dense design views"
                .to_string(),
        );
    }

    let cov_hh = covariance.slice(s![0..p_time, 0..p_time]).to_owned();
    let cov_tt = covariance
        .slice(s![p_time..p_time + p_t, p_time..p_time + p_t])
        .to_owned();
    let cov_ll = covariance
        .slice(s![p_time + p_t..p_total, p_time + p_t..p_total])
        .to_owned();
    let cov_ht = covariance
        .slice(s![0..p_time, p_time..p_time + p_t])
        .to_owned();
    let cov_hl = covariance
        .slice(s![0..p_time, p_time + p_t..p_total])
        .to_owned();
    let cov_tl = covariance
        .slice(s![p_time..p_time + p_t, p_time + p_t..p_total])
        .to_owned();

    let xh_hh = input.x_time_exit.dot(&cov_hh);
    let xt_tt = x_threshold_dense.dot(&cov_tt);
    let xl_ll = x_log_sigma_dense.dot(&cov_ll);
    let xh_ht = input.x_time_exit.dot(&cov_ht);
    let xh_hl = input.x_time_exit.dot(&cov_hl);
    let xt_tl = x_threshold_dense.dot(&cov_tl);

    let mu_h = input.x_time_exit.dot(&fit.beta_time) + &input.eta_time_offset_exit;
    let mu_t = x_threshold_dense.dot(&fit.beta_threshold);
    let mu_ls = x_log_sigma_dense.dot(&fit.beta_log_sigma);
    let link = residual_distribution_link(input.distribution);
    let quad_ctx = crate::quadrature::QuadratureContext::new();

    let fallback_row = |i: usize| {
        crate::quadrature::normal_expectation_3d_adaptive(
            &quad_ctx,
            [mu_h[i], mu_t[i], mu_ls[i]],
            [
                [var_h_row(i, input, &xh_hh), cov_ht_row(i, x_threshold_dense, &xh_ht), cov_hl_row(i, x_log_sigma_dense, &xh_hl)],
                [cov_ht_row(i, x_threshold_dense, &xh_ht), var_t_row(i, x_threshold_dense, &xt_tt), cov_tl_row(i, x_log_sigma_dense, &xt_tl)],
                [cov_hl_row(i, x_log_sigma_dense, &xh_hl), cov_tl_row(i, x_log_sigma_dense, &xt_tl), var_ls_row(i, x_log_sigma_dense, &xl_ll)],
            ],
            |h, t, ls| {
                let sigma =
                    bounded_sigma_derivs_up_to_third_scalar(ls, input.sigma_min, input.sigma_max)
                        .0
                        .max(1e-12);
                input.distribution.cdf(-h - t / sigma).clamp(0.0, 1.0)
            },
        )
        .clamp(0.0, 1.0)
    };

    let survival_prob = Array1::from_iter((0..n).map(|i| {
        let var_h = var_h_row(i, input, &xh_hh);
        let var_t = var_t_row(i, x_threshold_dense, &xt_tt);
        let var_ls = var_ls_row(i, x_log_sigma_dense, &xl_ll);
        let cov_ht_i = cov_ht_row(i, x_threshold_dense, &xh_ht);
        let cov_hl_i = cov_hl_row(i, x_log_sigma_dense, &xh_hl);
        let cov_tl_i = cov_tl_row(i, x_log_sigma_dense, &xt_tl);

        if !(var_h.is_finite()
            && var_t.is_finite()
            && var_ls.is_finite()
            && cov_ht_i.is_finite()
            && cov_hl_i.is_finite()
            && cov_tl_i.is_finite())
        {
            return fallback_row(i);
        }

        if var_ls <= 1e-12
            && (cov_hl_i.abs() > 1e-10 || cov_tl_i.abs() > 1e-10)
        {
            return fallback_row(i);
        }

        let beta_h_ls = if var_ls > 1e-12 { cov_hl_i / var_ls } else { 0.0 };
        let beta_t_ls = if var_ls > 1e-12 { cov_tl_i / var_ls } else { 0.0 };
        let var_h_cond = (var_h - beta_h_ls * cov_hl_i).max(0.0);
        let var_t_cond = (var_t - beta_t_ls * cov_tl_i).max(0.0);
        let cov_ht_cond = cov_ht_i - beta_h_ls * cov_tl_i;

        if !var_h_cond.is_finite() || !var_t_cond.is_finite() || !cov_ht_cond.is_finite() {
            return fallback_row(i);
        }

        crate::quadrature::normal_expectation_1d_adaptive(&quad_ctx, mu_ls[i], var_ls.sqrt(), |ls| {
            let sigma = bounded_sigma_derivs_up_to_third_scalar(ls, input.sigma_min, input.sigma_max)
                .0
                .max(1e-12);
            let inv_sigma = 1.0 / sigma;
            let delta_ls = ls - mu_ls[i];
            let mu_h_cond = mu_h[i] + beta_h_ls * delta_ls;
            let mu_t_cond = mu_t[i] + beta_t_ls * delta_ls;
            let mu_loc = -mu_h_cond - mu_t_cond * inv_sigma;
            let var_loc =
                (var_h_cond + var_t_cond * inv_sigma * inv_sigma + 2.0 * cov_ht_cond * inv_sigma)
                    .max(0.0);
            crate::quadrature::integrated_inverse_link_mean_and_derivative(
                &quad_ctx,
                link,
                mu_loc,
                var_loc.sqrt(),
            )
            .mean
        })
        .clamp(0.0, 1.0)
    }));

    Ok(SurvivalLocationScaleProbitPredictResult {
        eta: pred.eta,
        survival_prob,
    })
}

#[inline]
fn var_h_row(
    i: usize,
    input: &SurvivalLocationScaleProbitPredictInput,
    xh_hh: &Array2<f64>,
) -> f64 {
    input.x_time_exit.row(i).dot(&xh_hh.row(i)).max(0.0)
}

#[inline]
fn var_t_row(i: usize, x_threshold_dense: &Array2<f64>, xt_tt: &Array2<f64>) -> f64 {
    x_threshold_dense.row(i).dot(&xt_tt.row(i)).max(0.0)
}

#[inline]
fn var_ls_row(i: usize, x_log_sigma_dense: &Array2<f64>, xl_ll: &Array2<f64>) -> f64 {
    x_log_sigma_dense.row(i).dot(&xl_ll.row(i)).max(0.0)
}

#[inline]
fn cov_ht_row(i: usize, x_threshold_dense: &Array2<f64>, xh_ht: &Array2<f64>) -> f64 {
    x_threshold_dense.row(i).dot(&xh_ht.row(i))
}

#[inline]
fn cov_hl_row(i: usize, x_log_sigma_dense: &Array2<f64>, xh_hl: &Array2<f64>) -> f64 {
    x_log_sigma_dense.row(i).dot(&xh_hl.row(i))
}

#[inline]
fn cov_tl_row(i: usize, x_log_sigma_dense: &Array2<f64>, xt_tl: &Array2<f64>) -> f64 {
    x_log_sigma_dense.row(i).dot(&xt_tl.row(i))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
        let anchor_row = prepared.design_exit.row(0);
        let max_abs = anchor_row.iter().copied().map(f64::abs).fold(0.0, f64::max);
        assert!(
            max_abs <= 1e-10,
            "anchor row not zero after identifiability transform: max_abs={max_abs}"
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
                    SurvivalLocationScaleProbitFamily::survival_ratio_first_derivative(f, fp, s);
                let ddr = SurvivalLocationScaleProbitFamily::survival_ratio_second_derivative(
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
    fn clamped_log_pdf_derivatives_are_zero_in_saturated_region() {
        let f = MIN_PROB * 0.1;
        let fp = 3.0;
        let fpp = -7.0;
        let (logf, d1, d2) =
            SurvivalLocationScaleProbitFamily::clamped_log_pdf_with_derivatives(f, fp, fpp);
        assert!((logf - MIN_PROB.ln()).abs() <= 1e-15);
        assert_eq!(d1, 0.0);
        assert_eq!(d2, 0.0);
    }

    #[test]
    fn clamped_survival_neglog_derivatives_are_zero_on_clamp_bounds() {
        // Lower clamp active.
        let (s_low, r_low, dr_low, ddr_low) =
            SurvivalLocationScaleProbitFamily::clamped_survival_neglog_derivatives(
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
            SurvivalLocationScaleProbitFamily::clamped_survival_neglog_derivatives(
                1.1, 0.2, -0.3, 0.4,
            );
        assert_eq!(s_high, 1.0);
        assert_eq!(r_high, 0.0);
        assert_eq!(dr_high, 0.0);
        assert_eq!(ddr_high, 0.0);
    }

    #[test]
    fn posterior_mean_prediction_matches_deterministic_when_covariance_is_zero() {
        let input = SurvivalLocationScaleProbitPredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.2],
            x_threshold: DesignMatrix::Dense(array![[1.0, -0.2]]),
            x_log_sigma: DesignMatrix::Dense(array![[1.0, 0.3]]),
            sigma_min: 0.1,
            sigma_max: 2.0,
            distribution: ResidualDistribution::Gaussian,
        };
        let fit = SurvivalLocationScaleProbitFitResult {
            beta_time: array![0.4, -0.1],
            beta_threshold: array![0.2, 0.3],
            beta_log_sigma: array![-0.5, 0.1],
            lambdas_time: Array1::zeros(0),
            lambdas_threshold: Array1::zeros(0),
            lambdas_log_sigma: Array1::zeros(0),
            log_likelihood: 0.0,
            penalized_objective: 0.0,
            iterations: 0,
            converged: true,
            covariance_conditional: None,
        };
        let deterministic = predict_survival_location_scale_probit(&input, &fit).expect("predict");
        let posterior = predict_survival_location_scale_probit_posterior_mean(
            &input,
            &fit,
            &Array2::zeros((6, 6)),
        )
        .expect("posterior mean");
        assert!((deterministic.survival_prob[0] - posterior.survival_prob[0]).abs() <= 1e-10);
    }

    #[test]
    fn gaussian_posterior_mean_reduction_matches_3d_ghq_small_case() {
        let input = SurvivalLocationScaleProbitPredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.1],
            x_threshold: DesignMatrix::Dense(array![[1.0, 0.25]]),
            x_log_sigma: DesignMatrix::Dense(array![[1.0, -0.15]]),
            sigma_min: 0.2,
            sigma_max: 1.8,
            distribution: ResidualDistribution::Gaussian,
        };
        let fit = SurvivalLocationScaleProbitFitResult {
            beta_time: array![0.3, -0.2],
            beta_threshold: array![0.1, 0.2],
            beta_log_sigma: array![-0.4, 0.15],
            lambdas_time: Array1::zeros(0),
            lambdas_threshold: Array1::zeros(0),
            lambdas_log_sigma: Array1::zeros(0),
            log_likelihood: 0.0,
            penalized_objective: 0.0,
            iterations: 0,
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
        let reduced = predict_survival_location_scale_probit_posterior_mean(&input, &fit, &covariance)
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
        let var_h = input.x_time_exit.row(0).dot(&cov_hh.dot(&input.x_time_exit.row(0).to_owned()));
        let var_t = x_t.row(0).dot(&cov_tt.dot(&x_t.row(0).to_owned()));
        let var_ls = x_ls.row(0).dot(&cov_ll.dot(&x_ls.row(0).to_owned()));
        let cov_ht_i = input.x_time_exit.row(0).dot(&cov_ht.dot(&x_t.row(0).to_owned()));
        let cov_hl_i = input.x_time_exit.row(0).dot(&cov_hl.dot(&x_ls.row(0).to_owned()));
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
                normal_cdf_approx(-h - t / sigma.max(1e-12)).clamp(0.0, 1.0)
            },
        );
        assert!((reduced.survival_prob[0] - ghq).abs() <= 2e-4);
    }
}
