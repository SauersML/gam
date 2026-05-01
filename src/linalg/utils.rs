use crate::construction::calculate_condition_number;
use crate::faer_ndarray::FaerEigh;
use crate::faer_ndarray::{
    FaerArrayView, FaerCholesky, FaerLinalgError, array2_to_matmut,
    factorize_symmetricwith_fallback,
};
use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, Zip};
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

pub(crate) fn row_mismatch_message(
    y_len: usize,
    w_len: usize,
    x_rows: usize,
    offset_len: usize,
) -> Option<String> {
    if y_len == w_len && y_len == x_rows && y_len == offset_len {
        None
    } else {
        Some(format!(
            "Row mismatch: y={}, w={}, X.rows={}, offset={}",
            y_len, w_len, x_rows, offset_len
        ))
    }
}

pub(crate) fn predict_gam_dimension_mismatch_message(
    x_rows: usize,
    x_cols: usize,
    beta_len: usize,
    offset_len: usize,
) -> Option<String> {
    if x_cols != beta_len {
        return Some(format!(
            "predict_gam dimension mismatch: X has {} columns but beta has length {}",
            x_cols, beta_len
        ));
    }
    if x_rows != offset_len {
        return Some(format!(
            "predict_gam dimension mismatch: X has {} rows but offset has length {}",
            x_rows, offset_len
        ));
    }
    None
}

pub(crate) fn add_relative_diag_ridge(matrix: &mut Array2<f64>, scale: f64, floor: f64) -> f64 {
    let ridge = scale
        * matrix
            .diag()
            .iter()
            .map(|&value| value.abs())
            .fold(0.0, f64::max)
            .max(floor);
    for idx in 0..matrix.nrows() {
        matrix[[idx, idx]] += ridge;
    }
    ridge
}

pub(crate) fn boundary_hit_indices(
    values: ArrayView1<'_, f64>,
    bound: f64,
    tolerance: f64,
) -> (Vec<usize>, Vec<usize>) {
    let at_lower = values
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value <= -bound + tolerance).then_some(idx))
        .collect();
    let at_upper = values
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value >= bound - tolerance).then_some(idx))
        .collect();
    (at_lower, at_upper)
}

pub(crate) fn symmetric_spectrum_condition_number(matrix: &Array2<f64>) -> f64 {
    matrix
        .eigh(Side::Lower)
        .ok()
        .map(|(evals, _)| {
            let min = evals
                .iter()
                .fold(f64::INFINITY, |acc, &value| acc.min(value));
            let max = evals
                .iter()
                .fold(f64::NEG_INFINITY, |acc, &value| acc.max(value));
            max / min.max(1e-12)
        })
        .unwrap_or(f64::NAN)
}

/// Enforce exact symmetry on a square matrix by averaging off-diagonal pairs.
pub(crate) fn enforce_symmetry(matrix: &mut Array2<f64>) {
    let n = matrix.nrows();
    debug_assert_eq!(n, matrix.ncols());
    for i in 0..n {
        for j in i + 1..n {
            let avg = 0.5 * (matrix[[i, j]] + matrix[[j, i]]);
            matrix[[i, j]] = avg;
            matrix[[j, i]] = avg;
        }
    }
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
    if p == 0 || preconditioner_diag.len() != p || max_iter == 0 {
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

    // Precompute reciprocal preconditioner once. Each PCG iteration applies
    // M^{-1} via a single elementwise multiply (z = inv_m * r), avoiding the
    // per-element `.abs().max(1e-12)` and division on every iteration. The
    // floor mirrors the previous guard against zero/negative diagonals.
    let mut inv_m = Array1::<f64>::zeros(p);
    Zip::from(&mut inv_m)
        .and(preconditioner_diag)
        .par_for_each(|inv, &m| {
            *inv = 1.0 / m.abs().max(1e-12);
        });

    let mut z = Array1::<f64>::zeros(p);
    Zip::from(&mut z)
        .and(&r)
        .and(&inv_m)
        .par_for_each(|zi, &ri, &im| {
            *zi = ri * im;
        });
    let mut p_dir = z.clone();
    let mut rz_old = r.dot(&z);
    if !rz_old.is_finite() || rz_old <= 0.0 {
        return None;
    }

    for iter in 0..max_iter {
        let ap = apply(&p_dir);
        if ap.len() != p {
            return None;
        }
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
            // Periodic residual refresh: r <- rhs - A x. Done in-place via
            // assign + scaled_add to avoid the prior fresh-allocation pattern
            // (`r = rhs - &ax;`) inside the hot loop.
            let ax = apply(&x);
            if ax.len() != p {
                return None;
            }
            r.assign(rhs);
            r.scaled_add(-1.0, &ax);
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
        Zip::from(&mut z)
            .and(&r)
            .and(&inv_m)
            .par_for_each(|zi, &ri, &im| {
                *zi = ri * im;
            });
        let rz_new = r.dot(&z);
        if !rz_new.is_finite() || rz_new <= 0.0 {
            return None;
        }
        let beta = rz_new / rz_old;
        if !beta.is_finite() {
            return None;
        }
        // p <- z + beta * p (fused, SIMD-friendly via ndarray::Zip; parallel
        // over coefficient dimension at biobank-scale p).
        Zip::from(&mut p_dir).and(&z).par_for_each(|pi, &zi| {
            *pi = zi + beta * *pi;
        });
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
        let mut matrix = materialize_symmetric_operator(dim, &apply)?;
        // Symmetrize defensively: matrix-free operator residuals can introduce
        // small asymmetry that trips Cholesky on otherwise-PSD systems.
        enforce_symmetry(&mut matrix);
        if let Ok(chol) = matrix.clone().cholesky(Side::Lower) {
            return Ok(2.0 * chol.diag().mapv(f64::ln).sum());
        }
        // Cholesky failed (operator-induced indefiniteness from PCG residuals
        // or SLQ trace correction). Fall back to symmetric eigendecomposition
        // with a small positive floor so logdet stays finite. This preserves
        // the trace-correction contract: we still return Σ ln(λ_i) on the
        // symmetrized matrix, just with eigenvalues clipped to a tiny floor
        // when they wander non-positive due to numerical noise.
        match FaerEigh::eigh(&matrix, Side::Lower) {
            Ok((evals, _)) => {
                // Floor eigenvalues at a scale-relative tiny positive value.
                let max_abs = evals
                    .iter()
                    .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
                    .max(1.0);
                let floor = max_abs * 1e-14;
                let mut logdet = 0.0_f64;
                for &lam in evals.iter() {
                    let clipped = if lam.is_finite() && lam > floor {
                        lam
                    } else {
                        floor
                    };
                    logdet += clipped.ln();
                }
                log::warn!(
                    "[STAGE] SLQ tiny-system Cholesky failed (dim={dim}); recovered \
                     logdet via symmetric eigendecomposition with positive-eigenvalue \
                     floor {floor:.3e} (max|λ|={max_abs:.3e})"
                );
                return Ok(logdet);
            }
            Err(_) => {
                // Eigendecomposition also failed. Last-resort dense logdet
                // via LU on |det| with sign tracking would not help here
                // (we need an SPD-style logdet); surface a descriptive error
                // so the outer driver can fall back.
                return Err("SLQ exact tiny-system Cholesky failed and symmetric \
                     eigendecomposition fallback also failed"
                    .to_string());
            }
        }
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
        // Hoist the target-norm scalar out of the inner-row hot path.
        let target_norm = (dim as f64).sqrt();
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
                // Single multiply by precomputed scale replaces per-element
                // `v * sqrt(dim) / norm` (one division+sqrt was being JITted
                // per element by the optimizer at best).
                let scale = target_norm / norm;
                z.mapv_inplace(|v| v * scale);
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
    let mut basis = Array1::<f64>::zeros(dim);
    for j in 0..dim {
        basis[j] = 1.0;
        let col = apply(&basis);
        if col.len() != dim {
            return Err("operator returned wrong output dimension".to_string());
        }
        // Single contiguous column write replaces a scalar `[[i, j]]`-indexed
        // copy loop; ndarray emits a tight memcpy/strided copy.
        matrix.column_mut(j).assign(&col);
        basis[j] = 0.0;
    }
    // Symmetrize the upper/lower triangles in one symmetric pass.
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
    // Store the accumulated orthonormal Lanczos basis as columns of a (p × max_steps)
    // contiguous matrix so reorthogonalization can be performed via two BLAS2 calls
    // (`Q^T v` and `v -= Q (Q^T v)`) instead of `k` separate dot/axpy pairs at step k.
    // This converts an O(m^2 p) sequence of BLAS1 ops into matmul-friendly traffic
    // with substantially better cache locality and SIMD throughput.
    let mut q_basis = Array2::<f64>::zeros((p, max_steps));
    let mut q_prev = Array1::<f64>::zeros(p);
    // Hoist the per-element division into a single multiply by a precomputed
    // reciprocal norm, avoiding `dim` calls to `sqrt` from the previous
    // `mapv(|v| v / probe_norm_sq.sqrt())` form.
    let inv_probe_norm = 1.0 / probe_norm_sq.sqrt();
    let mut q = probe.mapv(|v| v * inv_probe_norm);
    let mut alphas = Vec::<f64>::with_capacity(max_steps);
    let mut betas = Vec::<f64>::with_capacity(max_steps.saturating_sub(1));
    let mut beta_prev = 0.0_f64;
    let tol = 1e-12;
    let mut basis_count = 0usize;
    let mut proj_buf = Array1::<f64>::zeros(max_steps);

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
        // Batched classical Gram-Schmidt against the accumulated basis Q[:, ..k]:
        //   proj = Q^T v  (length k)
        //   v   -= Q proj
        // Equivalent (in exact arithmetic) to the previous per-column MGS loop.
        if basis_count > 0 {
            let q_view = q_basis.slice(ndarray::s![.., ..basis_count]);
            {
                let mut proj = proj_buf.slice_mut(ndarray::s![..basis_count]);
                ndarray::linalg::general_mat_vec_mul(1.0, &q_view.t(), &v, 0.0, &mut proj);
            }
            let proj_ro = proj_buf.slice(ndarray::s![..basis_count]);
            ndarray::linalg::general_mat_vec_mul(-1.0, &q_view, &proj_ro, 1.0, &mut v);
        }
        // One additional pass against the current q (mirrors the original explicit
        // self-projection step; serves as the "twice-is-enough" CGS reinforcement).
        let q_self_proj = q.dot(&v);
        if q_self_proj != 0.0 {
            v.scaled_add(-q_self_proj, &q);
        }
        let beta = v.dot(&v).sqrt();
        alphas.push(alpha);
        if !beta.is_finite() {
            return Err("SLQ Lanczos produced non-finite beta".to_string());
        }
        q_basis.column_mut(basis_count).assign(&q);
        basis_count += 1;
        if beta <= tol {
            break;
        }
        betas.push(beta);
        q_prev = q;
        // Precomputed reciprocal beta replaces a per-element division.
        let inv_beta = 1.0 / beta;
        q = v.mapv(|x| x * inv_beta);
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

// ═══════════════════════════════════════════════════════════════════════════
//  Implicit Hessian operator helpers
//
//  These primitives accept a closure that applies the Hessian operator
//  H v = (X^T W X) v + Σ λ_k S_k v  (+ ridge v), without ever materializing
//  H as a `p × p` dense matrix. They are the symmetric counterparts of
//  `solve_newton_direction_implicit` in `solver/pirls.rs`: same operator,
//  different consumers. Used by the REML log-det / EDF path when the penalty
//  set contains operator-form penalties at threshold sizes.
// ═══════════════════════════════════════════════════════════════════════════

/// Stochastic Lanczos quadrature estimate of `log det(H)` where `H` is
/// supplied as an SPD matvec closure. Thin wrapper around
/// `stochastic_lanczos_logdet_spd_operator` with a name that reflects the
/// REML use case. The closure must apply an *already-regularized* H (i.e. the
/// caller bakes any ridge into `apply_h`); this function does not add a
/// regularization shift.
pub fn log_det_implicit_h<F>(
    apply_h: F,
    dim: usize,
    num_probes: usize,
    lanczos_steps: usize,
    seed: u64,
) -> Result<f64, String>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    stochastic_lanczos_logdet_spd_operator(dim, apply_h, num_probes, lanczos_steps, seed)
}

/// Hutchinson trace estimator: `tr(A) ≈ (1/m) Σ z_i^T A z_i` over `m`
/// Rademacher probes drawn from a deterministic RNG. Generic — `apply_a`
/// must compute `A v` and `dim` is the side length of `A`.
///
/// At biobank-scale `dim`, this is the only feasible way to estimate
/// `tr(A)` when `A` itself cannot be materialized (e.g. `A = H^{-1} B`).
/// For a true matrix-free implementation choose `num_probes` ≥ 32 to keep
/// the Monte-Carlo standard error below ~1% of `‖A‖_F / √dim`.
pub fn hutchinson_trace_estimator<F>(
    apply_a: F,
    dim: usize,
    num_probes: usize,
    seed: u64,
) -> Result<f64, String>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    if dim == 0 {
        return Ok(0.0);
    }
    if num_probes == 0 {
        return Err("hutchinson_trace_estimator: num_probes must be > 0".to_string());
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let probes = orthogonal_rademacher_probes(dim, num_probes, &mut rng);
    let mut acc = KahanSum::default();
    for z in probes.iter() {
        let az = apply_a(z);
        if az.len() != dim {
            return Err(format!(
                "hutchinson_trace_estimator: apply_a returned length {} (expected {})",
                az.len(),
                dim,
            ));
        }
        let mut local = KahanSum::default();
        for i in 0..dim {
            local.add(z[i] * az[i]);
        }
        acc.add(local.sum());
    }
    Ok(acc.sum() / num_probes as f64)
}

/// Hutchinson estimate of `tr(H^{-1} B)` for SPD `H` and arbitrary `B`,
/// both supplied as matvec closures. Each probe vector `z` is mapped to
/// `B z` and then `H^{-1} (B z)` is solved by PCG using `pcg_diag` as a
/// Jacobi preconditioner. Returns `(1/m) Σ z^T H^{-1} B z`.
///
/// This is the workhorse for REML EDF computation when `H` is implicit:
/// EDF = `tr(H^{-1} (X^T W X))`. With operator-form penalties wired through
/// `apply_h`, this avoids materializing the `p × p` Hessian or its inverse.
///
/// Caller controls `pcg_rel_tol` and `pcg_max_iter`; failure of any single
/// inner PCG solve aborts the estimator with `Err`. For routine use,
/// `pcg_rel_tol = 1e-8` and `pcg_max_iter = max(p, 1000)` are reasonable.
pub fn hutchinson_trace_inv_h_b<FH, FB>(
    apply_h: FH,
    apply_b: FB,
    pcg_diag: &Array1<f64>,
    dim: usize,
    num_probes: usize,
    pcg_rel_tol: f64,
    pcg_max_iter: usize,
    seed: u64,
) -> Result<f64, String>
where
    FH: Fn(&Array1<f64>) -> Array1<f64>,
    FB: Fn(&Array1<f64>) -> Array1<f64>,
{
    if dim == 0 {
        return Ok(0.0);
    }
    if num_probes == 0 {
        return Err("hutchinson_trace_inv_h_b: num_probes must be > 0".to_string());
    }
    if pcg_diag.len() != dim {
        return Err(format!(
            "hutchinson_trace_inv_h_b: pcg_diag length {} != dim {}",
            pcg_diag.len(),
            dim,
        ));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let probes = orthogonal_rademacher_probes(dim, num_probes, &mut rng);
    let mut acc = KahanSum::default();
    for z in probes.iter() {
        let bz = apply_b(z);
        let h_inv_bz = solve_spd_pcg(&apply_h, &bz, pcg_diag, pcg_rel_tol, pcg_max_iter)
            .ok_or_else(|| "hutchinson_trace_inv_h_b: inner PCG failed to converge".to_string())?;
        let mut local = KahanSum::default();
        for i in 0..dim {
            local.add(z[i] * h_inv_bz[i]);
        }
        acc.add(local.sum());
    }
    Ok(acc.sum() / num_probes as f64)
}

#[cfg(test)]
mod tests {
    use super::{
        boundary_hit_step_fraction, default_slq_parameters, solve_spd_pcg, solve_spd_pcg_with_info,
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
    fn solve_spd_pcg_rejects_zero_iteration_budget() {
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let m = Array1::from_vec(vec![4.0, 3.0]);
        assert!(solve_spd_pcg_with_info(|v| h.dot(v), &b, &m, 1e-10, 0).is_none());
        assert!(solve_spd_pcg(|v| h.dot(v), &b, &m, 1e-10, 0).is_none());
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
