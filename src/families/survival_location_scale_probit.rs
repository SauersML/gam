use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, BlockwiseFitResult, CustomFamily, FamilyEvaluation,
    ParameterBlockSpec, ParameterBlockState, fit_custom_family,
};
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
    for i in 0..raw.len() {
        let z = (raw[i] - g) / t;
        if z >= 30.0 {
            safe[i] = raw[i];
            d1[i] = 1.0;
            d2[i] = 0.0;
            continue;
        }
        if z <= -30.0 {
            let ez = z.exp();
            safe[i] = g + t * ez;
            d1[i] = ez;
            d2[i] = ez / t;
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
    }
    GuardedDerivative { safe, d1, d2 }
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

    let time_stacked_design = stack_time_design(&spec.time_block);
    let time_stacked_offset = stack_time_offset(&spec.time_block);

    let time_spec = ParameterBlockSpec {
        name: "time_transform".to_string(),
        design: DesignMatrix::Dense(time_stacked_design),
        offset: time_stacked_offset,
        penalties: spec.time_block.penalties.clone(),
        initial_log_lambdas: initial_log_lambdas(
            &spec.time_block.penalties,
            spec.time_block.initial_log_lambdas.clone(),
        )?,
        initial_beta: spec.time_block.initial_beta.clone(),
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
        x_time_entry: spec.time_block.design_entry,
        x_time_exit: spec.time_block.design_exit,
        x_time_deriv: spec.time_block.design_derivative_exit,
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

    Ok(SurvivalLocationScaleProbitFitResult {
        beta_time: fit.block_states[SurvivalLocationScaleProbitFamily::BLOCK_TIME]
            .beta
            .clone(),
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
