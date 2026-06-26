use super::*;

#[derive(Debug, Clone)]
pub struct SaeArrowVector {
    pub t: Array1<f64>,
    pub beta: Array1<f64>,
}

pub(crate) struct DeflatedArrowSolver<'a> {
    pub(crate) cache: &'a ArrowFactorCache,
    pub(crate) gauge_basis: Vec<Array1<f64>>,
    pub(crate) gauge_response_physical: Vec<Array1<f64>>,
    pub(crate) woodbury_factor: Option<FaerCholeskyFactor>,
    pub(crate) gauge_stiffness_recip: f64,
}

impl<'a> DeflatedArrowSolver<'a> {
    pub(crate) fn plain(cache: &'a ArrowFactorCache) -> Self {
        Self {
            cache,
            gauge_basis: Vec::new(),
            gauge_response_physical: Vec::new(),
            woodbury_factor: None,
            gauge_stiffness_recip: 0.0,
        }
    }

    pub(crate) fn from_orthonormal_gauges(
        cache: &'a ArrowFactorCache,
        gauge_basis: Vec<Array1<f64>>,
        stiffness: f64,
    ) -> Result<Self, String> {
        if gauge_basis.is_empty() {
            return Ok(Self::plain(cache));
        }
        if !(stiffness.is_finite() && stiffness > 0.0) {
            return Err(format!(
                "DeflatedArrowSolver: gauge stiffness must be finite and positive; got {stiffness}"
            ));
        }
        let full_len = cache.delta_t_len() + cache.k;
        let mut gauge_responses = Vec::with_capacity(gauge_basis.len());
        for gauge in &gauge_basis {
            if gauge.len() != full_len {
                return Err(format!(
                    "DeflatedArrowSolver: gauge length {} != cache full length {full_len}",
                    gauge.len()
                ));
            }
            let (sol_t, sol_beta) = cache
                .full_inverse_apply(
                    gauge.slice(s![..cache.delta_t_len()]),
                    gauge.slice(s![cache.delta_t_len()..]),
                )
                .map_err(|err| format!("DeflatedArrowSolver: gauge back-solve: {err}"))?;
            gauge_responses.push(flatten_arrow_parts(sol_t.view(), sol_beta.view()));
        }

        let rank = gauge_basis.len();
        let stiffness_recip = stiffness.recip();
        let mut gauge_metric = Array2::<f64>::zeros((rank, rank));
        let mut woodbury = Array2::<f64>::eye(rank);
        for i in 0..rank {
            woodbury[[i, i]] *= stiffness_recip;
            for j in 0..rank {
                let value = gauge_basis[i].dot(&gauge_responses[j]);
                gauge_metric[[i, j]] = value;
                woodbury[[i, j]] += value;
            }
        }
        let woodbury_factor = woodbury
            .cholesky(Side::Lower)
            .map_err(|err| format!("DeflatedArrowSolver: gauge Woodbury factor failed: {err}"))?;
        let mut gauge_response_physical = gauge_responses;
        for j in 0..rank {
            for i in 0..rank {
                let coeff = gauge_metric[[i, j]];
                for row in 0..full_len {
                    gauge_response_physical[j][row] -= coeff * gauge_basis[i][row];
                }
            }
        }
        Ok(Self {
            cache,
            gauge_basis,
            gauge_response_physical,
            woodbury_factor: Some(woodbury_factor),
            gauge_stiffness_recip: stiffness_recip,
        })
    }

    pub(crate) fn solve(
        &self,
        rhs_t: ArrayView1<'_, f64>,
        rhs_beta: ArrayView1<'_, f64>,
    ) -> Result<SaeArrowVector, String> {
        let (sol_t, sol_beta) = self
            .cache
            .full_inverse_apply(rhs_t, rhs_beta)
            .map_err(|err| format!("DeflatedArrowSolver: full inverse: {err}"))?;
        let Some(factor) = self.woodbury_factor.as_ref() else {
            return Ok(SaeArrowVector {
                t: sol_t,
                beta: sol_beta,
            });
        };

        let full_len = self.cache.delta_t_len() + self.cache.k;
        let mut flat = flatten_arrow_parts(sol_t.view(), sol_beta.view());
        if flat.len() != full_len {
            return Err(format!(
                "DeflatedArrowSolver: solution length {} != cache full length {full_len}",
                flat.len()
            ));
        }
        let mut gauge_coeffs = Array1::<f64>::zeros(self.gauge_basis.len());
        for (idx, gauge) in self.gauge_basis.iter().enumerate() {
            gauge_coeffs[idx] = gauge.dot(&flat);
        }
        let weights = factor.solvevec(&gauge_coeffs);
        for (gauge, &coeff) in self.gauge_basis.iter().zip(gauge_coeffs.iter()) {
            for i in 0..flat.len() {
                flat[i] -= gauge[i] * coeff;
            }
        }
        for (response, &weight) in self.gauge_response_physical.iter().zip(weights.iter()) {
            for i in 0..flat.len() {
                flat[i] -= response[i] * weight;
            }
        }
        for (gauge, &weight) in self.gauge_basis.iter().zip(weights.iter()) {
            let coeff = self.gauge_stiffness_recip * weight;
            for i in 0..flat.len() {
                flat[i] += gauge[i] * coeff;
            }
        }
        Ok(SaeArrowVector {
            t: flat.slice(s![..self.cache.delta_t_len()]).to_owned(),
            beta: flat.slice(s![self.cache.delta_t_len()..]).to_owned(),
        })
    }

    pub(crate) fn latent_inverse_diagonal(&self) -> Result<Array1<f64>, String> {
        if self.woodbury_factor.is_none() {
            return self
                .cache
                .latent_block_inverse_diagonal()
                .map_err(|err| format!("DeflatedArrowSolver: latent inverse diagonal: {err}"));
        }
        let total_t = self.cache.delta_t_len();
        let mut out = Array1::<f64>::zeros(total_t);
        let rhs_beta = Array1::<f64>::zeros(self.cache.k);
        for idx in 0..total_t {
            let mut rhs_t = Array1::<f64>::zeros(total_t);
            rhs_t[idx] = 1.0;
            let solved = self.solve(rhs_t.view(), rhs_beta.view())?;
            out[idx] = solved.t[idx];
        }
        Ok(out)
    }
}

pub(crate) fn flatten_arrow_parts(
    t: ArrayView1<'_, f64>,
    beta: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(t.len() + beta.len());
    for i in 0..t.len() {
        out[i] = t[i];
    }
    for i in 0..beta.len() {
        out[t.len() + i] = beta[i];
    }
    out
}

pub(crate) fn apply_cached_arrow_hessian(
    cache: &ArrowFactorCache,
    v_t: ArrayView1<'_, f64>,
    v_beta: ArrayView1<'_, f64>,
) -> Result<SaeArrowVector, String> {
    let total_t = cache.delta_t_len();
    if v_t.len() != total_t || v_beta.len() != cache.k {
        return Err(format!(
            "apply_cached_arrow_hessian: vector shapes (t={}, beta={}) != cache shapes \
             (t={total_t}, beta={})",
            v_t.len(),
            v_beta.len(),
            cache.k
        ));
    }

    let mut out_t = Array1::<f64>::zeros(total_t);
    let mut out_beta = Array1::<f64>::zeros(cache.k);
    for row in 0..cache.n_rows() {
        let di = cache.row_dims[row];
        let base = cache.row_offsets[row];
        let row_v = v_t.slice(s![base..base + di]);
        let factor = cache.undamped_factor(row);
        let av = cholesky_factor_apply(factor, row_v);
        for j in 0..di {
            out_t[base + j] += av[j];
        }
        if cache.k > 0 {
            let mut b_vbeta = Array1::<f64>::zeros(di);
            if !cache.apply_htbeta_row(row, v_beta, &mut b_vbeta) {
                return Err(format!(
                    "apply_cached_arrow_hessian: H_tβ^({row}) apply failed"
                ));
            }
            for j in 0..di {
                out_t[base + j] += b_vbeta[j];
            }
            if !cache.apply_htbeta_row_transpose(row, row_v, &mut out_beta, None) {
                return Err(format!(
                    "apply_cached_arrow_hessian: H_βt^({row}) apply failed"
                ));
            }
        }
    }

    if cache.k > 0 {
        let Some(schur_factor) = cache.schur_factor.as_ref() else {
            return Err(
                "apply_cached_arrow_hessian: dense Schur factor is required for gauge probing"
                    .to_string(),
            );
        };
        let schur_v = cholesky_factor_apply(schur_factor.view(), v_beta);
        for i in 0..cache.k {
            out_beta[i] += schur_v[i];
        }
        for row in 0..cache.n_rows() {
            let di = cache.row_dims[row];
            let mut b_vbeta = Array1::<f64>::zeros(di);
            if !cache.apply_htbeta_row(row, v_beta, &mut b_vbeta) {
                return Err(format!(
                    "apply_cached_arrow_hessian: H_tβ^({row}) Schur correction apply failed"
                ));
            }
            let a_inv_b_vbeta = cholesky_solve_vector(cache.undamped_factor(row), b_vbeta.view());
            if !cache.apply_htbeta_row_transpose(row, a_inv_b_vbeta.view(), &mut out_beta, None) {
                return Err(format!(
                    "apply_cached_arrow_hessian: H_βt^({row}) Schur correction apply failed"
                ));
            }
        }
    }

    Ok(SaeArrowVector {
        t: out_t,
        beta: out_beta,
    })
}

pub(crate) fn cholesky_factor_apply(
    factor: ArrayView2<'_, f64>,
    vector: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let n = factor.nrows();
    let mut lt_v = Array1::<f64>::zeros(n);
    for row in 0..n {
        let mut acc = 0.0_f64;
        for col in row..n {
            acc += factor[[col, row]] * vector[col];
        }
        lt_v[row] = acc;
    }
    let mut out = Array1::<f64>::zeros(n);
    for row in 0..n {
        let mut acc = 0.0_f64;
        for col in 0..=row {
            acc += factor[[row, col]] * lt_v[col];
        }
        out[row] = acc;
    }
    out
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum SaeLocalRowVar {
    Logit { atom: usize },
    Coord { atom: usize, axis: usize },
}

#[derive(Debug, Clone)]
pub(crate) struct SaeBorderChannel {
    pub(crate) atom: usize,
    pub(crate) basis_col: usize,
    pub(crate) index: usize,
    pub(crate) output: Vec<f64>,
}

#[derive(Debug, Clone)]
pub(crate) struct SaeRowJets {
    pub(crate) vars: Vec<SaeLocalRowVar>,
    pub(crate) first: Vec<Vec<f64>>,
    pub(crate) second: Vec<Vec<Vec<f64>>>,
    pub(crate) beta: Vec<Vec<f64>>,
    pub(crate) beta_deriv: Vec<Vec<Vec<f64>>>,
    pub(crate) beta_l_deriv: Vec<Vec<Vec<f64>>>,
}

pub(crate) fn sae_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Euclidean inner product `⟨a, b⟩` over the concatenated `(t, β)` blocks of two
/// arrow vectors. Used by the #1418 `B`-preconditioned CG inner solve.
pub(crate) fn sae_inner(a: &SaeArrowVector, b: &SaeArrowVector) -> f64 {
    sae_dot(a.t.as_slice().unwrap_or(&[]), b.t.as_slice().unwrap_or(&[]))
        + sae_dot(
            a.beta.as_slice().unwrap_or(&[]),
            b.beta.as_slice().unwrap_or(&[]),
        )
}

/// Euclidean norm `‖a‖` over the concatenated `(t, β)` blocks of an arrow vector.
pub(crate) fn sae_norm(a: &SaeArrowVector) -> f64 {
    sae_inner(a, a).max(0.0).sqrt()
}

/// #1418: solve `A x = rhs` by **`B`-preconditioned conjugate gradients**, where
/// `apply_a(v) = A v` is the exact stationarity-Jacobian matvec and the
/// `solver` (the assembled `B` factorization) supplies the SPD preconditioner
/// `B⁻¹`. The IFT step `θ̂_ρ = −A⁻¹ g_ρ` must invert the EXACT `A`, not the
/// surrogate `B`; the earlier truncated Neumann series `Σ_m (−B⁻¹ΔC)^m B⁻¹ rhs`
/// equals `A⁻¹ rhs` only when `ρ(B⁻¹ΔC) < 1`, and DIVERGED for large
/// `ΔC = ⟨r, ∂²f⟩`. PCG converges for any spectral radius in ≤ `dim` steps — one
/// `A` matvec and one `B⁻¹` solve per step, no second factorization. On
/// non-positive curvature `pᵀ A p ≤ 0` (the high-residual `A` can be indefinite
/// away from a strict minimum) it stops at the last finite iterate.
pub(crate) fn solve_b_preconditioned_cg<F>(
    solver: &DeflatedArrowSolver<'_>,
    rhs: &SaeArrowVector,
    apply_a: F,
) -> Result<SaeArrowVector, String>
where
    F: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
{
    // x_0 = B⁻¹ rhs (the surrogate step; CG corrects it onto A⁻¹ rhs).
    let mut x = solver
        .solve(rhs.t.view(), rhs.beta.view())
        .map_err(|err| format!("solve_b_preconditioned_cg: B inverse: {err}"))?;
    // r_0 = rhs − A x_0; z_0 = B⁻¹ r_0; p_0 = z_0.
    let ax = apply_a(&x)?;
    let mut r = SaeArrowVector {
        t: &rhs.t - &ax.t,
        beta: &rhs.beta - &ax.beta,
    };
    let mut z = solver
        .solve(r.t.view(), r.beta.view())
        .map_err(|err| format!("solve_b_preconditioned_cg: B preconditioner: {err}"))?;
    let mut p = z.clone();
    let mut rz = sae_inner(&r, &z);

    let rhs_norm = sae_norm(rhs).max(1.0);
    let max_iters = (x.t.len() + x.beta.len()).clamp(8, 256);
    let rel_tol = 1.0e-10;
    for _ in 0..max_iters {
        if !rz.is_finite() || rz <= 0.0 {
            break; // preconditioned residual exhausted / degenerate.
        }
        let ap = apply_a(&p)?;
        let p_ap = sae_inner(&p, &ap);
        if !p_ap.is_finite() || p_ap <= 0.0 {
            break; // non-positive curvature: keep the finite iterate.
        }
        let alpha = rz / p_ap;
        for idx in 0..x.t.len() {
            x.t[idx] += alpha * p.t[idx];
            r.t[idx] -= alpha * ap.t[idx];
        }
        for idx in 0..x.beta.len() {
            x.beta[idx] += alpha * p.beta[idx];
            r.beta[idx] -= alpha * ap.beta[idx];
        }
        if sae_norm(&r) <= rel_tol * rhs_norm {
            break;
        }
        z = solver
            .solve(r.t.view(), r.beta.view())
            .map_err(|err| format!("solve_b_preconditioned_cg: B preconditioner: {err}"))?;
        let rz_next = sae_inner(&r, &z);
        let beta = rz_next / rz;
        for idx in 0..p.t.len() {
            p.t[idx] = z.t[idx] + beta * p.t[idx];
        }
        for idx in 0..p.beta.len() {
            p.beta[idx] = z.beta[idx] + beta * p.beta[idx];
        }
        rz = rz_next;
    }
    Ok(x)
}
