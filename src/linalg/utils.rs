use crate::construction::calculate_condition_number;
use crate::faer_ndarray::FaerEigh;
use crate::faer_ndarray::{
    FaerArrayView, FaerCholesky, FaerLinalgError, array2_to_matmut,
    factorize_symmetricwith_fallback,
};
use faer::Side;
use ndarray::{Array1, Array2};
use rand::{RngExt, SeedableRng, rngs::StdRng};

const HESSIAN_CONDITION_TARGET: f64 = 1e10;
const MAX_FACTORIZATION_ATTEMPTS: usize = 4;
const MAX_SOLVE_RETRIES: usize = 8;

#[derive(Default, Clone, Copy)]
pub(crate) struct KahanSum {
    sum: f64,
    c: f64,
}

impl KahanSum {
    pub(crate) fn add(&mut self, value: f64) {
        let y = value - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    pub(crate) fn sum(self) -> f64 {
        self.sum
    }
}

pub(crate) fn matrix_inversewith_regularization(
    matrix: &Array2<f64>,
    label: &str,
) -> Option<Array2<f64>> {
    StableSolver::new(label).inversewith_regularization(matrix)
}

pub(crate) struct StableSolver<'a> {
    label: &'a str,
}

impl<'a> StableSolver<'a> {
    pub(crate) fn new(label: &'a str) -> Self {
        Self { label }
    }

    pub(crate) fn factorize(
        &self,
        matrix: &Array2<f64>,
    ) -> Result<crate::faer_ndarray::FaerSymmetricFactor, FaerLinalgError> {
        let view = FaerArrayView::new(matrix);
        factorize_symmetricwith_fallback(view.as_ref(), Side::Lower)
    }

    pub(crate) fn inversewith_regularization(&self, matrix: &Array2<f64>) -> Option<Array2<f64>> {
        let p = matrix.nrows();
        if p == 0 || matrix.ncols() != p {
            return None;
        }

        let mut planner = RidgePlanner::new(matrix);
        let (factor, _, regularized) = self.factorize_with_ridge_plan(matrix, &mut planner)?;
        let mut inv = Array2::<f64>::eye(p);
        let mut invview = array2_to_matmut(&mut inv);
        factor.solve_in_place(invview.as_mut());

        if !inv.iter().all(|v| v.is_finite()) {
            log::warn!("Non-finite inverse produced for {}", self.label);
            return None;
        }

        // Numerical solves can leave tiny asymmetry; enforce symmetry explicitly.
        for i in 0..p {
            for j in (i + 1)..p {
                let avg = 0.5 * (inv[[i, j]] + inv[[j, i]]);
                inv[[i, j]] = avg;
                inv[[j, i]] = avg;
            }
        }
        debug_assert_eq!(regularized.nrows(), p);
        Some(inv)
    }

    pub(crate) fn solvevectorwithridge_retries(
        &self,
        matrix: &Array2<f64>,
        rhs: &Array1<f64>,
        baseridge: f64,
    ) -> Option<Array1<f64>> {
        let p = matrix.nrows();
        if matrix.ncols() != p || rhs.len() != p {
            return None;
        }

        for retry in 0..MAX_SOLVE_RETRIES {
            let ridge = if baseridge > 0.0 {
                baseridge * 10f64.powi(retry as i32)
            } else {
                0.0
            };
            let h = addridge(matrix, ridge);
            let factor = match self.factorize(&h) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let mut out = rhs.clone();
            let mut out_mat = crate::faer_ndarray::array1_to_col_matmut(&mut out);
            factor.solve_in_place(out_mat.as_mut());
            if out.iter().all(|v| v.is_finite()) {
                return Some(out);
            }
        }
        None
    }

    pub(crate) fn inversewithridge_retries(
        &self,
        matrix: &Array2<f64>,
        baseridge: f64,
        max_retry: usize,
    ) -> Option<Array2<f64>> {
        let p = matrix.nrows();
        if p == 0 || matrix.ncols() != p {
            return None;
        }
        for retry in 0..max_retry {
            let ridge = if baseridge > 0.0 {
                baseridge * 10f64.powi(retry as i32)
            } else {
                0.0
            };
            let h = addridge(matrix, ridge);
            let factor = match self.factorize(&h) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let mut inv = Array2::<f64>::eye(p);
            let mut invview = array2_to_matmut(&mut inv);
            factor.solve_in_place(invview.as_mut());
            if inv.iter().all(|v| v.is_finite()) {
                for i in 0..p {
                    for j in (i + 1)..p {
                        let avg = 0.5 * (inv[[i, j]] + inv[[j, i]]);
                        inv[[i, j]] = avg;
                        inv[[j, i]] = avg;
                    }
                }
                return Some(inv);
            }
        }
        None
    }

    fn factorize_with_ridge_plan(
        &self,
        matrix: &Array2<f64>,
        planner: &mut RidgePlanner,
    ) -> Option<(crate::faer_ndarray::FaerSymmetricFactor, f64, Array2<f64>)> {
        loop {
            let ridge = planner.ridge();
            let h_eff = addridge(matrix, ridge);
            if let Ok(factor) = self.factorize(&h_eff) {
                return Some((factor, ridge, h_eff));
            }
            if planner.attempts() >= MAX_FACTORIZATION_ATTEMPTS {
                log::warn!(
                    "Failed to factorize {} after ridge {:.3e}",
                    self.label,
                    ridge
                );
                return None;
            }
            planner.bumpwith_matrix(matrix);
        }
    }
}

pub(crate) fn max_abs_diag(matrix: &Array2<f64>) -> f64 {
    matrix
        .diag()
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max)
        .max(1.0)
}

pub(crate) fn addridge(matrix: &Array2<f64>, ridge: f64) -> Array2<f64> {
    if ridge <= 0.0 {
        return matrix.clone();
    }
    let mut regularized = matrix.clone();
    let n = regularized.nrows();
    for i in 0..n {
        regularized[[i, i]] += ridge;
    }
    regularized
}

pub(crate) fn boundary_hit_step_fraction(
    slack: f64,
    directional_slack_change: f64,
    current_step_limit: f64,
) -> Option<f64> {
    if !slack.is_finite()
        || !directional_slack_change.is_finite()
        || !current_step_limit.is_finite()
        || current_step_limit <= 0.0
    {
        return None;
    }

    let scale = slack
        .abs()
        .max(directional_slack_change.abs())
        .max(current_step_limit.abs())
        .max(1.0);
    let directional_tol = (64.0 * f64::EPSILON * scale).max(1e-14);
    if directional_slack_change >= -directional_tol {
        return None;
    }

    let step = (slack / -directional_slack_change).max(0.0);
    if step.is_finite() && step < current_step_limit {
        return Some(step);
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgSolveInfo {
    pub iterations: usize,
    pub converged: bool,
    pub relative_residual_norm: f64,
}

pub fn solve_spd_pcg_with_info<F>(
    apply: F,
    rhs: &Array1<f64>,
    preconditioner_diag: &Array1<f64>,
    rel_tol: f64,
    max_iter: usize,
) -> Option<(Array1<f64>, PcgSolveInfo)>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let p = rhs.len();
    if p == 0 || preconditioner_diag.len() != p {
        return None;
    }
    let rhs_norm = rhs.dot(rhs).sqrt();
    if !rhs_norm.is_finite() {
        return None;
    }
    if rhs_norm == 0.0 {
        return Some((
            Array1::<f64>::zeros(p),
            PcgSolveInfo {
                iterations: 0,
                converged: true,
                relative_residual_norm: 0.0,
            },
        ));
    }

    let tol = rel_tol.max(1e-12) * rhs_norm.max(1.0);
    let mut x = Array1::<f64>::zeros(p);
    let mut r = rhs.clone();
    let mut z = Array1::<f64>::zeros(p);
    for i in 0..p {
        z[i] = r[i] / preconditioner_diag[i].abs().max(1e-12);
    }
    let mut p_dir = z.clone();
    let mut rz_old = r.dot(&z);
    if !rz_old.is_finite() || rz_old <= 0.0 {
        return None;
    }

    for iter in 0..max_iter.max(1) {
        let ap = apply(&p_dir);
        let denom = p_dir.dot(&ap);
        if !denom.is_finite() || denom <= 0.0 {
            return None;
        }
        let alpha = rz_old / denom;
        if !alpha.is_finite() {
            return None;
        }
        x.scaled_add(alpha, &p_dir);
        r.scaled_add(-alpha, &ap);
        if (iter + 1) % 32 == 0 {
            let ax = apply(&x);
            r = rhs - &ax;
        }
        let r_norm = r.dot(&r).sqrt();
        if r_norm.is_finite() && r_norm <= tol {
            return x.iter().all(|v| v.is_finite()).then_some((
                x,
                PcgSolveInfo {
                    iterations: iter + 1,
                    converged: true,
                    relative_residual_norm: r_norm / rhs_norm.max(1.0),
                },
            ));
        }
        for i in 0..p {
            z[i] = r[i] / preconditioner_diag[i].abs().max(1e-12);
        }
        let rz_new = r.dot(&z);
        if !rz_new.is_finite() || rz_new <= 0.0 {
            return None;
        }
        let beta = rz_new / rz_old;
        if !beta.is_finite() {
            return None;
        }
        for i in 0..p {
            p_dir[i] = z[i] + beta * p_dir[i];
        }
        rz_old = rz_new;
    }
    None
}

pub fn solve_spd_pcg<F>(
    apply: F,
    rhs: &Array1<f64>,
    preconditioner_diag: &Array1<f64>,
    rel_tol: f64,
    max_iter: usize,
) -> Option<Array1<f64>>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    solve_spd_pcg_with_info(apply, rhs, preconditioner_diag, rel_tol, max_iter)
        .map(|(solution, _)| solution)
}

pub fn stochastic_lanczos_logdet_spd(
    matrix: &Array2<f64>,
    num_probes: usize,
    lanczos_steps: usize,
    seed: u64,
) -> Result<f64, String> {
    let p = matrix.nrows();
    if matrix.ncols() != p {
        return Err("SLQ requires a square matrix".to_string());
    }
    if p <= 2048 {
        let chol = matrix
            .clone()
            .cholesky(Side::Lower)
            .map_err(|_| "SLQ exact dense-system Cholesky failed".to_string())?;
        return Ok(2.0 * chol.diag().mapv(f64::ln).sum());
    }
    stochastic_lanczos_logdet_spd_operator(p, |v| matrix.dot(v), num_probes, lanczos_steps, seed)
}

pub fn stochastic_lanczos_logdet_spd_operator<F>(
    dim: usize,
    apply: F,
    num_probes: usize,
    lanczos_steps: usize,
    seed: u64,
) -> Result<f64, String>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    if dim == 0 {
        return Ok(0.0);
    }
    if dim <= 128 {
        let matrix = materialize_symmetric_operator(dim, &apply)?;
        let chol = matrix
            .cholesky(Side::Lower)
            .map_err(|_| "SLQ exact tiny-system Cholesky failed".to_string())?;
        return Ok(2.0 * chol.diag().mapv(f64::ln).sum());
    }
    let probes = num_probes.max(1);
    let steps = lanczos_steps.clamp(2, dim.max(2));
    let mut rng = StdRng::seed_from_u64(seed);
    let probe_vectors = orthogonal_rademacher_probes(dim, probes, &mut rng);
    let mut estimate = KahanSum::default();
    for z in &probe_vectors {
        estimate.add(lanczos_logdet_probe_operator(dim, &apply, z, steps)?);
    }
    Ok(estimate.sum() / probes as f64)
}

fn orthogonal_rademacher_probes(dim: usize, probes: usize, rng: &mut StdRng) -> Vec<Array1<f64>> {
    let block = probes.min(dim.max(1));
    let mut out = Vec::with_capacity(probes);
    while out.len() < probes {
        let remaining = probes - out.len();
        let take = remaining.min(block);
        let mut local = Vec::<Array1<f64>>::with_capacity(take);
        for _ in 0..take {
            let mut z = Array1::<f64>::zeros(dim);
            for i in 0..dim {
                z[i] = if rng.random::<bool>() { 1.0 } else { -1.0 };
            }
            for prev in &local {
                let proj = prev.dot(&z) / prev.dot(prev).max(1e-12);
                if proj != 0.0 {
                    z.scaled_add(-proj, prev);
                }
            }
            let norm = z.dot(&z).sqrt();
            if norm > 1e-12 {
                z.mapv_inplace(|v| v * (dim as f64).sqrt() / norm);
                local.push(z);
            }
        }
        if local.is_empty() {
            let mut fallback = Array1::<f64>::zeros(dim);
            for i in 0..dim {
                fallback[i] = if rng.random::<bool>() { 1.0 } else { -1.0 };
            }
            out.push(fallback);
        } else {
            out.extend(local);
        }
    }
    out
}

fn materialize_symmetric_operator<F>(dim: usize, apply: &F) -> Result<Array2<f64>, String>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let mut matrix = Array2::<f64>::zeros((dim, dim));
    for j in 0..dim {
        let mut basis = Array1::<f64>::zeros(dim);
        basis[j] = 1.0;
        let col = apply(&basis);
        if col.len() != dim {
            return Err("operator returned wrong output dimension".to_string());
        }
        for i in 0..dim {
            matrix[[i, j]] = col[i];
        }
    }
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (matrix[[i, j]] + matrix[[j, i]]);
            matrix[[i, j]] = avg;
            matrix[[j, i]] = avg;
        }
    }
    Ok(matrix)
}

fn lanczos_logdet_probe_operator<F>(
    dim: usize,
    apply: &F,
    probe: &Array1<f64>,
    max_steps: usize,
) -> Result<f64, String>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let p = dim;
    let probe_norm_sq = probe.dot(probe);
    if !probe_norm_sq.is_finite() || probe_norm_sq <= 0.0 {
        return Err("SLQ probe has invalid norm".to_string());
    }
    let mut q_prev = Array1::<f64>::zeros(p);
    let mut q = probe.mapv(|v| v / probe_norm_sq.sqrt());
    let mut q_basis = Vec::<Array1<f64>>::with_capacity(max_steps);
    let mut alphas = Vec::<f64>::with_capacity(max_steps);
    let mut betas = Vec::<f64>::with_capacity(max_steps.saturating_sub(1));
    let mut beta_prev = 0.0_f64;
    let tol = 1e-12;

    for _ in 0..max_steps {
        let mut v = apply(&q);
        if beta_prev > 0.0 {
            v.scaled_add(-beta_prev, &q_prev);
        }
        let alpha = q.dot(&v);
        if !alpha.is_finite() {
            return Err("SLQ Lanczos produced non-finite alpha".to_string());
        }
        v.scaled_add(-alpha, &q);
        // Reorthogonalize against the accumulated basis to limit ghost Ritz
        // values and reduce variance on moderate-dimension SPD systems.
        for basis_vec in &q_basis {
            let proj = basis_vec.dot(&v);
            if proj != 0.0 {
                v.scaled_add(-proj, basis_vec);
            }
        }
        let q_self_proj = q.dot(&v);
        if q_self_proj != 0.0 {
            v.scaled_add(-q_self_proj, &q);
        }
        let beta = v.dot(&v).sqrt();
        alphas.push(alpha);
        if !beta.is_finite() {
            return Err("SLQ Lanczos produced non-finite beta".to_string());
        }
        q_basis.push(q.clone());
        if beta <= tol {
            break;
        }
        betas.push(beta);
        q_prev = q;
        q = v.mapv(|x| x / beta);
        beta_prev = beta;
    }

    let m = alphas.len();
    if m == 0 {
        return Err("SLQ Lanczos did not produce any steps".to_string());
    }
    let mut tri = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        tri[[i, i]] = alphas[i];
        if i + 1 < m {
            tri[[i, i + 1]] = betas[i];
            tri[[i + 1, i]] = betas[i];
        }
    }
    let (evals, evecs) = FaerEigh::eigh(&tri, Side::Lower)
        .map_err(|e| format!("SLQ tridiagonal eig failed: {e}"))?;
    let mut quad = 0.0_f64;
    for k in 0..m {
        if evals[k] <= 0.0 {
            return Err("SLQ encountered non-positive Ritz value on SPD surface".to_string());
        }
        let weight = evecs[(0, k)] * evecs[(0, k)];
        quad += weight * evals[k].ln();
    }
    Ok(probe_norm_sq * quad)
}

pub fn default_slq_parameters(dim: usize) -> (usize, usize) {
    let probes = if dim <= 32 {
        256
    } else if dim <= 64 {
        512
    } else if dim < 128 {
        512
    } else if dim < 512 {
        48
    } else {
        32
    };
    let steps = if dim <= 128 {
        dim.max(8)
    } else {
        dim.clamp(32, 96)
    };
    (probes, steps)
}

#[derive(Clone)]
pub(crate) struct RidgePlanner {
    cond_estimate: Option<f64>,
    ridge: f64,
    attempts: usize,
    scale: f64,
}

impl RidgePlanner {
    pub(crate) fn new(matrix: &Array2<f64>) -> Self {
        let scale = max_abs_diag(matrix);
        let min_step = scale * 1e-10;
        // Most Hessians factorize on the first attempt. Avoid an eager exact
        // condition-number decomposition here and only pay for spectral
        // diagnostics after an actual factorization failure.
        Self {
            cond_estimate: None,
            ridge: min_step,
            attempts: 0,
            scale,
        }
    }

    pub(crate) fn ridge(&self) -> f64 {
        self.ridge
    }

    pub(crate) fn cond_estimate(&self) -> Option<f64> {
        self.cond_estimate
    }

    #[inline]
    fn estimate_conditionwithridge(&self, matrix: &Array2<f64>, ridge: f64) -> Option<f64> {
        let regularized = if ridge > 0.0 {
            addridge(matrix, ridge)
        } else {
            matrix.clone()
        };
        calculate_condition_number(&regularized)
            .ok()
            .filter(|c| c.is_finite() && *c > 0.0)
    }

    pub(crate) fn bumpwith_matrix(&mut self, matrix: &Array2<f64>) {
        self.attempts += 1;
        let min_step = self.scale * 1e-10;
        let base = self.ridge.max(min_step);

        // Estimate conditioning at the current ridge level.
        let cond_now = self.estimate_conditionwithridge(matrix, base);
        self.cond_estimate = cond_now;

        self.ridge = if let Some(cond) = cond_now {
            let ratio = cond / HESSIAN_CONDITION_TARGET;
            // Primary update from condition feedback.
            // sqrt-ratio avoids wild overshoot while still scaling with severity.
            let mut multiplier = if ratio > 1.0 {
                ratio.sqrt().clamp(1.5, 10.0)
            } else {
                // Factorization failed despite "acceptable" condition number.
                // This usually indicates indefiniteness/numerical fragility, so
                // use a stronger fallback than ×2, increasing with attempts.
                (2.0 + self.attempts as f64).clamp(3.0, 10.0)
            };

            let mut proposal = base * multiplier;
            // Verify whether the proposal actually improves condition enough.
            // If not, escalate once more before returning.
            if let Some(cond_next) = self.estimate_conditionwithridge(matrix, proposal)
                && cond_next > cond * 0.9
                && ratio > 1.0
            {
                multiplier = (multiplier * 1.8).clamp(2.0, 10.0);
                proposal = base * multiplier;
            }
            proposal.max(min_step)
        } else if self.ridge <= 0.0 {
            min_step
        } else {
            // Condition estimate unavailable: geometric fallback.
            (base * 10.0).max(min_step)
        };

        if !self.ridge.is_finite() || self.ridge <= 0.0 {
            self.ridge = self.scale;
        }
    }

    pub(crate) fn attempts(&self) -> usize {
        self.attempts
    }
}

#[cfg(test)]
mod tests {
    use super::{
        boundary_hit_step_fraction, default_slq_parameters, solve_spd_pcg,
        stochastic_lanczos_logdet_spd, stochastic_lanczos_logdet_spd_operator,
    };
    use crate::faer_ndarray::FaerCholesky;
    use faer::Side;
    use ndarray::{Array1, array};

    #[test]
    fn boundary_hit_step_fraction_ignores_near_tangential_direction() {
        let step = boundary_hit_step_fraction(1.0, -1e-16, 1.0);
        assert_eq!(step, None);
    }

    #[test]
    fn boundary_hit_step_fraction_returns_first_finite_hit() {
        let step = boundary_hit_step_fraction(0.25, -0.5, 1.0);
        assert_eq!(step, Some(0.5));
    }

    #[test]
    fn boundary_hit_step_fraction_rejects_non_finite_candidate() {
        let step = boundary_hit_step_fraction(1.0, f64::NEG_INFINITY, 1.0);
        assert_eq!(step, None);
    }

    #[test]
    fn solve_spd_pcg_matches_reference_solution() {
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let m = Array1::from_vec(vec![4.0, 3.0]);
        let x = solve_spd_pcg(|v| h.dot(v), &b, &m, 1e-10, 20).expect("pcg solve");
        assert!((x[0] - 0.0909090909).abs() < 1e-8);
        assert!((x[1] - 0.6363636363).abs() < 1e-8);
    }

    #[test]
    fn stochastic_lanczos_logdet_tracks_exact_spd_logdet() {
        let h = array![[5.0, 1.0, 0.2], [1.0, 4.0, 0.3], [0.2, 0.3, 3.5]];
        let exact = h.clone().cholesky(Side::Lower).expect("chol");
        let exact_logdet = 2.0 * exact.diag().mapv(f64::ln).sum();
        let approx = stochastic_lanczos_logdet_spd(&h, 16, 12, 7).expect("slq");
        assert!((approx - exact_logdet).abs() < 5e-2);
    }

    #[test]
    fn operator_slq_matches_dense_slq() {
        let h = array![
            [3.5, 0.2, 0.1, 0.0],
            [0.2, 4.0, 0.3, 0.1],
            [0.1, 0.3, 5.0, 0.4],
            [0.0, 0.1, 0.4, 3.8]
        ];
        let (probes, steps) = default_slq_parameters(h.nrows());
        let dense = stochastic_lanczos_logdet_spd(&h, probes, steps, 13).expect("dense slq");
        let op = stochastic_lanczos_logdet_spd_operator(h.nrows(), |v| h.dot(v), probes, steps, 13)
            .expect("operator slq");
        assert!((dense - op).abs() < 1e-10);
    }
}
