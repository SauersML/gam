use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, BlockwiseFitResult, CustomFamily, FamilyEvaluation,
    ParameterBlockSpec, ParameterBlockState, fit_custom_family,
};
use crate::faer_ndarray::FaerSvd;
use crate::matrix::DesignMatrix;
use crate::probability::{normal_cdf_approx, normal_pdf};
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

#[derive(Clone)]
struct GuardedDerivative {
    safe: Array1<f64>,
    d1: Array1<f64>,
    d2: Array1<f64>,
    d3: Array1<f64>,
}

#[derive(Clone)]
struct SurvivalLocationScaleProbitFamily {
    n: usize,
    y: Array1<f64>,
    w: Array1<f64>,
    sigma_min: f64,
    sigma_max: f64,
    distribution: ResidualDistribution,
    derivative_guard: f64,
    derivative_softness: f64,
    x_time_entry: Array2<f64>,
    x_time_exit: Array2<f64>,
    x_time_deriv: Array2<f64>,
    x_threshold: Array2<f64>,
    x_log_sigma: Array2<f64>,
}

impl SurvivalLocationScaleProbitFamily {
    const BLOCK_TIME: usize = 0;
    const BLOCK_THRESHOLD: usize = 1;
    const BLOCK_LOG_SIGMA: usize = 2;
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

fn safe_sigma_from_eta(
    eta: &Array1<f64>,
    sigma_min: f64,
    sigma_max: f64,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let span = (sigma_max - sigma_min).max(1e-12);
    let mut sigma = Array1::<f64>::zeros(eta.len());
    let mut ds = Array1::<f64>::zeros(eta.len());
    let mut d2s = Array1::<f64>::zeros(eta.len());
    for i in 0..eta.len() {
        let z = eta[i].clamp(-40.0, 40.0);
        let p = 1.0 / (1.0 + (-z).exp());
        let d1 = p * (1.0 - p);
        sigma[i] = sigma_min + span * p;
        ds[i] = span * d1;
        d2s[i] = span * d1 * (1.0 - 2.0 * p);
    }
    (sigma, ds, d2s)
}

fn guarded_derivative(raw: &Array1<f64>, guard: f64, tau: f64) -> GuardedDerivative {
    let g = guard.max(1e-12);
    let t = tau.max(1e-9);
    let mut safe = Array1::<f64>::zeros(raw.len());
    let mut d1 = Array1::<f64>::zeros(raw.len());
    let mut d2 = Array1::<f64>::zeros(raw.len());
    let mut d3 = Array1::<f64>::zeros(raw.len());
    for i in 0..raw.len() {
        let z = (raw[i] - g) / t;
        if z >= 30.0 {
            safe[i] = raw[i];
            d1[i] = 1.0;
            d2[i] = 0.0;
            d3[i] = 0.0;
            continue;
        }
        if z <= -30.0 {
            let ez = z.exp();
            safe[i] = g + t * ez;
            d1[i] = ez;
            d2[i] = ez / t;
            d3[i] = ez / (t * t);
            continue;
        }
        let softplus = if z > 0.0 {
            z + (-z).exp().ln_1p()
        } else {
            z.exp().ln_1p()
        };
        let sig = 1.0 / (1.0 + (-z).exp());
        safe[i] = g + t * softplus;
        d1[i] = sig;
        d2[i] = sig * (1.0 - sig) / t;
        d3[i] = sig * (1.0 - sig) * (1.0 - 2.0 * sig) / (t * t);
    }
    GuardedDerivative { safe, d1, d2, d3 }
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
            for b in 0..p {
                out[[a, b]] += wi * xa * x[[i, b]];
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

        let h0 = eta_time.slice(s![0..n]).to_owned();
        let h1 = eta_time.slice(s![n..2 * n]).to_owned();
        let d_raw = eta_time.slice(s![2 * n..3 * n]).to_owned();
        let d_guard = guarded_derivative(&d_raw, self.derivative_guard, self.derivative_softness);

        let (sigma, ds, d2s) = safe_sigma_from_eta(eta_ls, self.sigma_min, self.sigma_max);
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
            let phi0 = self.distribution.pdf(u0).max(MIN_PROB);
            let phi1 = self.distribution.pdf(u1).max(MIN_PROB);
            let s0 = (1.0 - self.distribution.cdf(u0)).clamp(MIN_PROB, 1.0);
            let s1 = (1.0 - self.distribution.cdf(u1)).clamp(MIN_PROB, 1.0);
            let g = d_guard.safe[i].max(MIN_PROB);

            ll += w * (d * (phi1.ln() + g.ln()) + (1.0 - d) * s1.ln() - s0.ln());

            let r0 = phi0 / s0;
            let r1 = phi1 / s1;
            let dr0 = (r0 * r0) - self.distribution.pdf_derivative(u0) / s0;
            let dr1 = (r1 * r1) - self.distribution.pdf_derivative(u1) / s1;

            // q derivatives (shared by threshold/log-sigma blocks)
            d1_q[i] = w * (r0 + d * (-u1) + (1.0 - d) * (-r1));
            d2_q[i] = w * (dr0 + d * (-1.0) + (1.0 - d) * (-dr1));

            // time block eta-derivatives
            grad_time_eta_h0[i] = -w * r0;
            grad_time_eta_h1[i] = -w * (d * (-u1) + (1.0 - d) * (-r1));
            grad_time_eta_d[i] = w * d * d_guard.d1[i] / g;

            h_time_h0[i] = -w * dr0;
            h_time_h1[i] = -w * (d * (-1.0) + (1.0 - d) * (-dr1));
            h_time_d[i] =
                -w * d * ((d_guard.d2[i] / g) - (d_guard.d1[i] * d_guard.d1[i]) / (g * g));
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

        let h0 = eta_time.slice(s![0..n]).to_owned();
        let h1 = eta_time.slice(s![n..2 * n]).to_owned();
        let d_raw = eta_time.slice(s![2 * n..3 * n]).to_owned();
        let d_guard = guarded_derivative(&d_raw, self.derivative_guard, self.derivative_softness);

        let (sigma, ds, d2s) = safe_sigma_from_eta(eta_ls, self.sigma_min, self.sigma_max);
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
            let phi0 = self.distribution.pdf(u0).max(MIN_PROB);
            let phi1 = self.distribution.pdf(u1).max(MIN_PROB);
            let s0 = (1.0 - self.distribution.cdf(u0)).clamp(MIN_PROB, 1.0);
            let s1 = (1.0 - self.distribution.cdf(u1)).clamp(MIN_PROB, 1.0);
            let fp0 = self.distribution.pdf_derivative(u0);
            let fp1 = self.distribution.pdf_derivative(u1);
            let fpp0 = self.distribution.pdf_second_derivative(u0);
            let fpp1 = self.distribution.pdf_second_derivative(u1);

            let r0 = phi0 / s0;
            let r1 = phi1 / s1;
            let dr0 = (r0 * r0) - fp0 / s0;
            let dr1 = (r1 * r1) - fp1 / s1;
            let ddr0 = 2.0 * r0 * dr0 - (fpp0 / s0 + fp0 * phi0 / (s0 * s0));
            let ddr1 = 2.0 * r1 * dr1 - (fpp1 / s1 + fp1 * phi1 / (s1 * s1));

            // q-derivatives of the per-row log-likelihood contribution.
            // With u0=-h0+q, u1=-h1+q:
            // dℓ/dq   = w [ r0 - d*u1 - (1-d) r1 ]
            // d²ℓ/dq² = w [ r0' - d - (1-d) r1' ]
            // d³ℓ/dq³ = w [ r0'' - (1-d) r1'' ]
            d1_q[i] = w * (r0 + d * (-u1) + (1.0 - d) * (-r1));
            d2_q[i] = w * (dr0 + d * (-1.0) + (1.0 - d) * (-dr1));
            d3_q[i] = w * (ddr0 + (1.0 - d) * (-ddr1));

            // Time block contributions use u0/u1 direct dependence,
            // while derivative row uses log(d_safe) term only for events.
            d_h_h0[i] = w * ddr0;
            d_h_h1[i] = -w * (1.0 - d) * ddr1;
            let g = d_guard.safe[i].max(MIN_PROB);
            let a = d_guard.d1[i];
            let b = d_guard.d2[i];
            let c = d_guard.d3[i];
            let de = c / g - 3.0 * a * b / (g * g) + 2.0 * a * a * a / (g * g * g);
            d_h_d[i] = -w * d * de;
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
                let span = (self.sigma_max - self.sigma_min).max(1e-12);
                // sigma(eta) = sigma_min + span * p, p = logistic(eta)
                // ds   = span * a,                 a = p(1-p)
                // d2s  = span * a(1-2p)
                // d3s  = span * (a(1-2p)^2 - 2a^2)
                let d3s = Array1::from_iter(eta_ls.iter().map(|&z_raw| {
                    let z = z_raw.clamp(-40.0, 40.0);
                    let p = 1.0 / (1.0 + (-z).exp());
                    let a = p * (1.0 - p);
                    span * (a * (1.0 - 2.0 * p) * (1.0 - 2.0 * p) - 2.0 * a * a)
                }));

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
        derivative_softness: spec.derivative_softness,
        x_time_entry: time_prepared.design_entry,
        x_time_exit: time_prepared.design_exit,
        x_time_deriv: time_prepared.design_derivative_exit,
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

    Ok(SurvivalLocationScaleProbitFitResult {
        beta_time,
        beta_threshold: fit.block_states[SurvivalLocationScaleProbitFamily::BLOCK_THRESHOLD]
            .beta
            .clone(),
        beta_log_sigma: fit.block_states[SurvivalLocationScaleProbitFamily::BLOCK_LOG_SIGMA]
            .beta
            .clone(),
        lambdas_time,
        lambdas_threshold,
        lambdas_log_sigma,
        log_likelihood: fit.log_likelihood,
        penalized_objective: fit.penalized_objective,
        iterations: fit.inner_cycles,
        converged: fit.converged,
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
    let (sigma, _, _) = safe_sigma_from_eta(&eta_ls, input.sigma_min, input.sigma_max);
    let eta = Array1::from_iter(
        h.iter()
            .zip(eta_t.iter())
            .zip(sigma.iter())
            .map(|((&hh, &tt), &ss)| -hh - tt / ss.max(1e-12)),
    );
    let survival_prob = eta.mapv(|v| input.distribution.cdf(v).clamp(0.0, 1.0));
    Ok(SurvivalLocationScaleProbitPredictResult { eta, survival_prob })
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
}
