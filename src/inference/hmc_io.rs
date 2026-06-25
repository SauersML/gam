use crate::faer_ndarray::{FaerCholesky, FaerEigh};
use crate::matrix::DesignMatrix;
use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NutsConfig {
    pub n_samples: usize,
    pub nwarmup: usize,
    pub n_chains: usize,
    pub target_accept: f64,
    #[serde(default = "default_nuts_seed")]
    pub seed: u64,
}

fn default_nuts_seed() -> u64 {
    42
}

impl Default for NutsConfig {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            nwarmup: 500,
            n_chains: 4,
            target_accept: 0.9,
            seed: 42,
        }
    }
}

impl NutsConfig {
    pub fn for_dimension(n_params: usize) -> Self {
        let effective_autocorr = (n_params as f64).sqrt().max(1.0);
        let target_ess = 100 * n_params;
        let raw_samples = (target_ess as f64 * (1.0 + 2.0 * effective_autocorr) * 1.5) as usize;
        let n_samples = raw_samples.clamp(500, 10_000);
        let n_chains = if n_params > 50 { 4 } else { 2 };
        Self {
            n_samples,
            nwarmup: n_samples,
            n_chains,
            target_accept: 0.9,
            seed: 42,
        }
    }
}

fn validate_nuts_config(config: &NutsConfig) -> Result<(), String> {
    if !(config.target_accept.is_finite()
        && config.target_accept > 0.0
        && config.target_accept < 1.0)
    {
        return Err(format!(
            "NUTS target_accept must be finite and lie in (0, 1), got {}",
            config.target_accept
        ));
    }
    if config.n_chains == 0 {
        return Err(
            "NUTS n_chains must be >= 1; with zero chains the sampler has no initial positions to run"
                .to_string(),
        );
    }
    if config.n_samples < 4 {
        return Err(format!(
            "NUTS n_samples must be >= 4 so split-R-hat / ESS diagnostics are defined, got {}",
            config.n_samples
        ));
    }
    Ok(())
}

fn solve_upper_triangular_transpose(l: &Array2<f64>, dim: usize) -> Array2<f64> {
    let mut result = Array2::<f64>::zeros((dim, dim));
    if dim == 0 {
        return result;
    }
    let l_owned;
    let l_rows: &[f64] = if let Some(s) = l.as_slice() {
        s
    } else {
        l_owned = l.to_owned();
        l_owned
            .as_slice()
            .expect("owned standard-layout Array2 has contiguous storage")
    };
    let mut y = vec![0.0_f64; dim];
    for col in 0..dim {
        let d_col = l_rows[col * dim + col];
        y[col] = if d_col.abs() > 1e-15 {
            1.0 / d_col
        } else {
            0.0
        };
        for i in (col + 1)..dim {
            let row_off = i * dim;
            let mut sum = 0.0_f64;
            for k in col..i {
                sum += l_rows[row_off + k] * y[k];
            }
            let d = l_rows[row_off + i];
            y[i] = if d.abs() > 1e-15 { -sum / d } else { 0.0 };
        }
        let res_row_start = col * dim + col;
        let res_row = &mut result.as_slice_mut().expect("owned Array2 contiguous")
            [res_row_start..res_row_start + (dim - col)];
        for (k, slot) in res_row.iter_mut().enumerate() {
            *slot = y[col + k];
        }
        for slot in &mut y[col..dim] {
            *slot = 0.0;
        }
    }
    result
}

fn hessian_whitening_chol(hessian: ArrayView2<'_, f64>, dim: usize) -> Result<Array2<f64>, String> {
    let factor = hessian
        .to_owned()
        .cholesky(Side::Lower)
        .map_err(|err| format!("Gaussian-posterior fallback Cholesky failed: {err:?}"))?;
    Ok(solve_upper_triangular_transpose(
        &factor.lower_triangular(),
        dim,
    ))
}

#[inline]
fn sample_standard_normal<R: rand::Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1 = rng.random::<f64>().max(1e-16);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

pub struct GaussianModePosterior {
    pub samples: Array2<f64>,
    pub posterior_mean: Array1<f64>,
    pub posterior_std: Array1<f64>,
    pub rhat: f64,
    pub ess: f64,
}

pub fn sample_gaussian_mode_posterior(
    mode: ArrayView1<'_, f64>,
    hessian: ArrayView2<'_, f64>,
    config: &NutsConfig,
) -> Result<GaussianModePosterior, String> {
    validate_nuts_config(config)?;
    let dim = mode.len();
    if hessian.nrows() != dim || hessian.ncols() != dim {
        return Err(format!(
            "Gaussian-posterior fallback: hessian shape {:?} does not match mode dim {dim}",
            hessian.dim()
        ));
    }
    if dim == 0 {
        return Err("Gaussian-posterior fallback: zero-dimensional posterior".to_string());
    }
    let mut h = hessian.to_owned();
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (h[[i, j]] + h[[j, i]]);
            h[[i, j]] = avg;
            h[[j, i]] = avg;
        }
    }
    let diag_scale = (0..dim).map(|i| h[[i, i]].abs()).fold(0.0_f64, f64::max);
    let jitter = (diag_scale * 1e-10).max(1e-12);
    for i in 0..dim {
        h[[i, i]] += jitter;
    }
    let chol = hessian_whitening_chol(h.view(), dim)?;
    let total = config.n_samples * config.n_chains;
    let mut samples = Array2::<f64>::zeros((total, dim));
    for chain in 0..config.n_chains {
        let stream = config.seed
            ^ 0x51A6_2C73_90E4_1DBF
            ^ ((chain as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
        let mut rng = StdRng::seed_from_u64(crate::linalg::utils::splitmix64_hash(stream));
        for draw in 0..config.n_samples {
            let row = chain * config.n_samples + draw;
            let mut z = Array1::<f64>::zeros(dim);
            for value in z.iter_mut() {
                *value = sample_standard_normal(&mut rng);
            }
            let beta = mode.to_owned() + chol.dot(&z);
            samples.row_mut(row).assign(&beta);
        }
    }
    let posterior_mean = samples
        .mean_axis(ndarray::Axis(0))
        .unwrap_or_else(|| Array1::zeros(dim));
    let posterior_std = samples.std_axis(ndarray::Axis(0), 0.0);
    Ok(GaussianModePosterior {
        samples,
        posterior_mean,
        posterior_std,
        rhat: 1.0,
        ess: total as f64,
    })
}

pub fn laplace_directional_cubic_diagnostic(
    hessian: &Array2<f64>,
    design: &DesignMatrix,
    c_weights: &Array1<f64>,
    _refine_supremum: bool,
) -> Result<(f64, Array1<f64>), String> {
    let p = hessian.nrows();
    if p == 0 || hessian.ncols() != p {
        return Ok((0.0, Array1::zeros(0)));
    }
    let sym_h = (hessian + &hessian.t()) * 0.5;
    let (evals, evecs) = sym_h
        .eigh(Side::Lower)
        .map_err(|err| format!("directional cubic diagnostic eigendecomposition failed: {err}"))?;
    let n = design.nrows();
    if c_weights.len() != n {
        return Err(format!(
            "directional cubic diagnostic weight length mismatch: got {}, expected {n}",
            c_weights.len()
        ));
    }
    let mut directional = Array1::<f64>::zeros(p);
    let max_eval = evals.iter().fold(0.0_f64, |acc, &ev| acc.max(ev.abs()));
    let tol = (max_eval * 1.0e-12).max(1.0e-14);
    for r in 0..p {
        let lambda = evals[r];
        if lambda <= tol {
            continue;
        }
        let v = evecs.column(r).to_owned();
        let mut cubic = 0.0_f64;
        let chunk = design.try_row_chunk(0..n).map_err(|err| {
            format!("directional cubic diagnostic materialization failed: {err:?}")
        })?;
        for i in 0..n {
            let xv = chunk.row(i).dot(&v);
            cubic += c_weights[i] * xv * xv * xv;
        }
        directional[r] = cubic / lambda.powf(1.5);
    }
    let max_abs = directional.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    Ok((max_abs, directional))
}

#[derive(Clone, Debug)]
pub struct LaplaceTrustworthiness {
    pub directional_skewness: Array1<f64>,
    pub untrustworthy_directions: Vec<usize>,
    pub threshold: f64,
    pub max_abs_skewness: f64,
}

impl LaplaceTrustworthiness {
    pub fn fallback_required(&self) -> bool {
        !self.untrustworthy_directions.is_empty()
    }
}

pub fn laplace_skewness_threshold(n_eff: f64) -> f64 {
    if !(n_eff > 0.0) {
        return f64::INFINITY;
    }
    ((24.0 / 5.0) / n_eff).sqrt()
}

pub fn laplace_trustworthiness_from_skewness(
    directional_skewness: &Array1<f64>,
    n_eff: f64,
) -> LaplaceTrustworthiness {
    let threshold = laplace_skewness_threshold(n_eff);
    let mut untrustworthy_directions = Vec::new();
    let mut max_abs_skewness = 0.0_f64;
    for (r, &gamma) in directional_skewness.iter().enumerate() {
        let abs_gamma = if gamma.is_finite() { gamma.abs() } else { 0.0 };
        max_abs_skewness = max_abs_skewness.max(abs_gamma);
        if abs_gamma > threshold {
            untrustworthy_directions.push(r);
        }
    }
    LaplaceTrustworthiness {
        directional_skewness: directional_skewness.clone(),
        untrustworthy_directions,
        threshold,
        max_abs_skewness,
    }
}

pub trait BlockExcessTarget {
    fn block_dim(&self) -> usize;
    fn rho_dim(&self) -> usize;
    fn block_curvatures(&self) -> &Array1<f64>;
    fn excess(&self, t: &Array1<f64>) -> f64;
    fn excess_rho_gradient(&self, t: &Array1<f64>) -> Array1<f64>;
    fn displaced_neg_score(&self, t: &Array1<f64>) -> Array1<f64>;
    fn base_neg_score(&self) -> Array1<f64>;

    fn excess_with_displaced_neg_score(&self, t: &Array1<f64>) -> (f64, Option<Array1<f64>>) {
        let excess = self.excess(t);
        if excess.is_finite() {
            (excess, Some(self.displaced_neg_score(t)))
        } else {
            (excess, None)
        }
    }

    fn excess_with_displaced_neg_score_batch(
        &self,
        draws: &Array2<f64>,
    ) -> Vec<(f64, Option<Array1<f64>>)> {
        let mut out = Vec::with_capacity(draws.ncols());
        let mut t = Array1::<f64>::zeros(draws.nrows());
        for s in 0..draws.ncols() {
            t.assign(&draws.column(s));
            out.push(self.excess_with_displaced_neg_score(&t));
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct BlockSampledMoments {
    pub e_t: Array1<f64>,
    pub e_tt: Array2<f64>,
    pub e_neg_score: Array1<f64>,
    pub e_t_neg_score: Array2<f64>,
}

#[derive(Clone, Debug)]
pub struct BlockSampledMarginal {
    pub value: f64,
    pub rho_gradient: Array1<f64>,
    pub importance_ess: f64,
    pub n_draws: usize,
    pub moments: Option<BlockSampledMoments>,
}

fn block_sampling_draws(block_dim: usize) -> usize {
    const BASE: usize = 256;
    const PER_DIM: usize = 256;
    const CAP: usize = 4096;
    (BASE + PER_DIM * block_dim).min(CAP)
}

pub fn block_sampled_marginal_correction<T: BlockExcessTarget>(
    target: &T,
) -> Result<BlockSampledMarginal, String> {
    let m = target.block_dim();
    let k = target.rho_dim();
    if m == 0 {
        return Ok(BlockSampledMarginal {
            value: 0.0,
            rho_gradient: Array1::zeros(k),
            importance_ess: 0.0,
            n_draws: 0,
            moments: None,
        });
    }
    let lambdas = target.block_curvatures();
    if lambdas.len() != m {
        return Err(format!(
            "block_sampled_marginal_correction: block_curvatures len {} != block_dim {m}",
            lambdas.len()
        ));
    }
    let inv_sqrt_lambda = lambdas.mapv(|l| if l > 0.0 { 1.0 / l.sqrt() } else { f64::NAN });
    if inv_sqrt_lambda.iter().any(|v| !v.is_finite()) {
        return Err(
            "block_sampled_marginal_correction: non-positive block curvature (mode is not a strict local minimum in a sampled direction)"
                .to_string(),
        );
    }
    let n_draws = block_sampling_draws(m);
    let mut seed_bits: u64 = 0x9E37_79B9_7F4A_7C15;
    seed_bits ^= (m as u64).rotate_left(17);
    seed_bits = seed_bits.wrapping_mul(0x1000_0000_01B3);
    seed_bits ^= (k as u64).rotate_left(31);
    seed_bits = seed_bits.wrapping_mul(0x1000_0000_01B3);
    let mut rng = StdRng::seed_from_u64(seed_bits);
    let n_obs = target.base_neg_score().len();
    let mut draws = Array2::<f64>::zeros((m, n_draws));
    for s in 0..n_draws {
        for r in 0..m {
            draws[[r, s]] = sample_standard_normal(&mut rng) * inv_sqrt_lambda[r];
        }
    }
    let mut max_lw = f64::NEG_INFINITY;
    let mut sum_w = 0.0_f64;
    let mut sum_w2 = 0.0_f64;
    let mut grad_acc = Array1::<f64>::zeros(k);
    let mut e_t_acc = Array1::<f64>::zeros(m);
    let mut e_tt_acc = Array2::<f64>::zeros((m, m));
    let mut e_ngs_acc = Array1::<f64>::zeros(n_obs);
    let mut e_t_ngs_acc = Array2::<f64>::zeros((n_obs, m));
    let mut t = Array1::<f64>::zeros(m);
    for (sidx, (excess, displaced_ngs)) in target
        .excess_with_displaced_neg_score_batch(&draws)
        .into_iter()
        .enumerate()
    {
        t.assign(&draws.column(sidx));
        if !excess.is_finite() {
            continue;
        }
        let Some(ngs) = displaced_ngs else {
            continue;
        };
        if ngs.len() != n_obs {
            return Err(format!(
                "block_sampled_marginal_correction: displaced_neg_score len {} != {n_obs}",
                ngs.len()
            ));
        }
        let lw = -excess;
        if lw > max_lw {
            let rescale = (max_lw - lw).exp();
            sum_w *= rescale;
            sum_w2 *= rescale * rescale;
            grad_acc *= rescale;
            e_t_acc *= rescale;
            e_tt_acc *= rescale;
            e_ngs_acc *= rescale;
            e_t_ngs_acc *= rescale;
            max_lw = lw;
        }
        let w = (lw - max_lw).exp();
        sum_w += w;
        sum_w2 += w * w;
        grad_acc.scaled_add(-w, &target.excess_rho_gradient(&t));
        e_t_acc.scaled_add(w, &t);
        e_ngs_acc.scaled_add(w, &ngs);
        for r in 0..m {
            let wt_r = w * t[r];
            for q in 0..m {
                e_tt_acc[(q, r)] += wt_r * t[q];
            }
            e_t_ngs_acc.column_mut(r).scaled_add(wt_r, &ngs);
        }
    }
    if !max_lw.is_finite() {
        return Err(
            "block_sampled_marginal_correction: all importance draws were infeasible".to_string(),
        );
    }
    let value = max_lw + (sum_w / n_draws as f64).ln();
    let (rho_gradient, moments) = if sum_w > 0.0 {
        (
            grad_acc / sum_w,
            Some(BlockSampledMoments {
                e_t: e_t_acc / sum_w,
                e_tt: e_tt_acc / sum_w,
                e_neg_score: e_ngs_acc / sum_w,
                e_t_neg_score: e_t_ngs_acc / sum_w,
            }),
        )
    } else {
        (Array1::zeros(k), None)
    };
    let importance_ess = if sum_w2 > 0.0 {
        (sum_w * sum_w) / sum_w2
    } else {
        0.0
    };
    if !value.is_finite() || rho_gradient.iter().any(|v| !v.is_finite()) {
        return Err(
            "block_sampled_marginal_correction: produced a non-finite correction or gradient"
                .to_string(),
        );
    }
    Ok(BlockSampledMarginal {
        value,
        rho_gradient,
        importance_ess,
        n_draws,
        moments,
    })
}
