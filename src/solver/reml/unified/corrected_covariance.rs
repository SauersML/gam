use super::*;

// ═══════════════════════════════════════════════════════════════════════════
//  Corrected coefficient covariance (smoothing-parameter uncertainty)
// ═══════════════════════════════════════════════════════════════════════════

/// Diagnostic returned when the (free-subspace) outer Hessian is indefinite.
///
/// An indefinite outer Hessian at the reported optimum means one of:
///  - the outer optimization has not converged to a stationary point,
///  - the reported point is a saddle, not a minimum,
///  - the active-bound set on θ is wrong (the unconstrained Hessian is
///    being inspected on directions that the constraint set actually pins),
///  - the objective being inspected is a surrogate / smoothed version of
///    the true REML/LAML criterion.
///
/// The previous implementation silently clamped the negative eigenvalues to
/// zero, which under-reports the uncertainty in those directions (it pretends
/// the directions don't exist). That is not "conservative" — it is wrong.
/// We refuse to fabricate a covariance and instead return this diagnostic.
#[derive(Debug, Clone)]
pub struct OuterHessianIndefinite {
    /// Most-negative eigenvalue of the projected (free-subspace) outer Hessian.
    pub min_eigenvalue: f64,
    /// Indices of θ-coordinates that were detected as active on a bound.
    pub active_constraints: Vec<usize>,
    /// θ at the reported optimum (if available; empty otherwise).
    pub theta: Vec<f64>,
    /// L2 norm of the outer gradient at θ (NaN if unavailable).
    pub gradient_norm: f64,
    /// Frobenius norm of the outer Hessian.
    pub hessian_norm: f64,
    /// Suggested next action for the caller / user.
    pub suggested_action: &'static str,
}

/// Errors that can arise while building the corrected covariance.
#[derive(Debug, Clone)]
pub enum CorrectedCovarianceError {
    /// Argument shapes do not agree at a named corrected-covariance stage.
    ShapeMismatch {
        context: &'static str,
        reason: String,
    },
    /// The eigendecomposition failed numerically at a named matrix stage.
    EigendecompositionFailed {
        context: &'static str,
        reason: String,
    },
    /// The projected outer Hessian is indefinite — see diagnostic.
    Indefinite(OuterHessianIndefinite),
}

impl core::fmt::Display for CorrectedCovarianceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShapeMismatch { context, reason } => {
                write!(f, "shape mismatch in {context}: {reason}")
            }
            Self::EigendecompositionFailed { context, reason } => {
                write!(f, "eigendecomposition failed in {context}: {reason}")
            }
            Self::Indefinite(d) => write!(
                f,
                "outer Hessian indefinite on free subspace (min eigenvalue = {:.3e}, \
                 ||H||_F = {:.3e}, ||g||_2 = {:.3e}, active = {:?}, theta = {:?}); {}",
                d.min_eigenvalue,
                d.hessian_norm,
                d.gradient_norm,
                d.active_constraints,
                d.theta,
                d.suggested_action,
            ),
        }
    }
}

impl std::error::Error for CorrectedCovarianceError {}

/// Result describing the corrected covariance plus structural diagnostics.
#[derive(Debug, Clone)]
pub struct CorrectedCovariance {
    /// The p×p corrected covariance V*_α.
    pub matrix: Array2<f64>,
    /// θ-indices that were treated as active on a bound and excluded from V_θ.
    pub active_constraints: Vec<usize>,
    /// θ-indices in the free subspace whose curvature was so close to zero
    /// that they were treated as structurally rank-deficient (pseudoinverse).
    pub rank_deficient_directions: Vec<usize>,
}

impl CorrectedCovariance {
    pub(crate) fn has_structural_diagnostics(&self) -> bool {
        !self.active_constraints.is_empty() || !self.rank_deficient_directions.is_empty()
    }
}

/// Suggested action text returned with `OuterHessianIndefinite`.
pub(crate) const INDEFINITE_SUGGESTED_ACTION: &str = "refit with a tighter outer tolerance, verify the inspected objective is the true \
     REML/LAML cost rather than a surrogate, and audit recent active-set transitions";

/// Detect θ-coordinates that are sitting on the [-RHO_BOUND, RHO_BOUND] bound.
///
/// We use the same `tolerance = 1e-8` as the rest of the outer code path so the
/// active-set view here agrees with the optimizer's view at the reported optimum.
pub(crate) fn detect_active_theta_bounds(theta: Option<&[f64]>, q: usize) -> Vec<usize> {
    let Some(theta) = theta else {
        return Vec::new();
    };
    if theta.len() != q {
        return Vec::new();
    }
    let bound = crate::solver::estimate::RHO_BOUND;
    // Same active-bound tolerance the outer optimizer uses, so this active-set
    // view agrees with the optimizer's at the reported optimum.
    const ACTIVE_THETA_BOUND_TOL: f64 = 1e-8;
    theta
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (v.abs() >= bound - ACTIVE_THETA_BOUND_TOL).then_some(i))
        .collect()
}

/// Decide which θ-coordinates are bounded (ρ-style) vs unbounded (ψ-style).
///
/// We treat the LAST `ext_len` coordinates as ψ (unbounded extended
/// hyperparameters) and the FIRST `rho_len` as ρ (bounded by ±RHO_BOUND).
/// This matches the layout used everywhere else in this file: J_α has ρ
/// columns first, then ext columns.
pub(crate) fn active_bound_indices_for_theta(
    theta: Option<&[f64]>,
    rho_len: usize,
    ext_len: usize,
) -> Vec<usize> {
    let q = rho_len + ext_len;
    let mut active = detect_active_theta_bounds(theta, q);
    // Drop ψ-coordinates: they are unbounded by construction.
    active.retain(|&i| i < rho_len);
    active
}

/// Inertia gate + projected-inverse on the free subspace of θ.
///
/// Returns `(V_θ_full, rank_deficient_free_indices)` where `V_θ_full` is q×q
/// with rows/columns of active coordinates set to zero. If the projected
/// Hessian is indefinite beyond tolerance, returns the diagnostic instead.
pub(crate) fn projected_inverse_with_inertia_gate(
    outer_hessian: &Array2<f64>,
    active: &[usize],
    theta_for_diag: Option<&[f64]>,
    gradient_norm: f64,
) -> Result<(Array2<f64>, Vec<usize>), CorrectedCovarianceError> {
    let q = outer_hessian.nrows();
    let mut is_active = vec![false; q];
    for &i in active {
        if i < q {
            is_active[i] = true;
        }
    }
    let free: Vec<usize> = (0..q).filter(|i| !is_active[*i]).collect();
    let qf = free.len();

    let h_norm = outer_hessian.iter().map(|v| v * v).sum::<f64>().sqrt();

    let mut v_theta_full = Array2::<f64>::zeros((q, q));
    if qf == 0 {
        return Ok((v_theta_full, Vec::new()));
    }

    let mut h_ff = Array2::<f64>::zeros((qf, qf));
    for (a, &ia) in free.iter().enumerate() {
        for (b, &ib) in free.iter().enumerate() {
            h_ff[[a, b]] = outer_hessian[[ia, ib]];
        }
    }

    let (evals, evecs) = h_ff.eigh(faer::Side::Lower).map_err(|e| {
        CorrectedCovarianceError::EigendecompositionFailed {
            context: "projected outer Hessian",
            reason: e.to_string(),
        }
    })?;

    let eps = f64::EPSILON;
    let neg_tol = 8.0 * eps * (q.max(1) as f64) * h_norm.max(1.0);
    let min_eig = evals.iter().copied().fold(f64::INFINITY, f64::min);
    if min_eig < -neg_tol {
        let diagnostic = OuterHessianIndefinite {
            min_eigenvalue: min_eig,
            active_constraints: active.to_vec(),
            theta: theta_for_diag.map(|t| t.to_vec()).unwrap_or_default(),
            gradient_norm,
            hessian_norm: h_norm,
            suggested_action: INDEFINITE_SUGGESTED_ACTION,
        };
        return Err(CorrectedCovarianceError::Indefinite(diagnostic));
    }

    let pos_tol = 8.0 * eps * (q.max(1) as f64) * h_norm.max(1.0);
    let mut v_theta_ff = Array2::<f64>::zeros((qf, qf));
    let mut rank_deficient_free: Vec<usize> = Vec::new();
    for j in 0..qf {
        let sigma = evals[j];
        if sigma.abs() <= pos_tol {
            rank_deficient_free.push(j);
            continue;
        }
        let inv_sigma = 1.0 / sigma;
        let u = evecs.column(j);
        for a in 0..qf {
            let ua = inv_sigma * u[a];
            for b in a..qf {
                let val = ua * u[b];
                v_theta_ff[[a, b]] += val;
                if a != b {
                    v_theta_ff[[b, a]] += val;
                }
            }
        }
    }

    for (a, &ia) in free.iter().enumerate() {
        for (b, &ib) in free.iter().enumerate() {
            v_theta_full[[ia, ib]] = v_theta_ff[[a, b]];
        }
    }

    let rank_deficient_directions: Vec<usize> =
        rank_deficient_free.into_iter().map(|j| free[j]).collect();

    Ok((v_theta_full, rank_deficient_directions))
}

/// Corrected covariance of the coefficient vector, accounting for uncertainty
/// in the smoothing/hyperparameters θ = (ρ, ψ).
///
/// The standard conditional covariance H^{-1} ignores uncertainty in θ.
/// The corrected covariance adds the propagation term:
///
/// ```text
///   V*_α = H^{-1} + J_α V_θ J_α^T
/// ```
///
/// where:
/// - H^{-1} is obtained via `hop.solve` on identity columns,
/// - J_α = [-v_1, …, -v_k, -ext_v_1, …, -ext_v_m] is the p×q matrix of
///   negated mode responses (implicit-function sensitivities ∂β̂/∂θ),
/// - V_θ is the inverse of the outer Hessian RESTRICTED to the free subspace
///   of θ (coordinates that are not pinned to a bound) and inertia-gated.
///
/// # Active-bound handling and inertia gate
///
/// If `theta_at_optimum` is supplied, ρ-coordinates sitting on ±`RHO_BOUND`
/// are treated as active and excluded from V_θ. The remaining free Hessian
/// block H_FF is eigen-decomposed:
///   - if min(σ) < -8·ε·q·‖H‖_F → return [`CorrectedCovarianceError::Indefinite`]
///     with a diagnostic (the previous behavior of clamping negatives to zero
///     under-reports uncertainty and is therefore refused);
///   - if |σ| ≤ 8·ε·q·‖H‖_F → that direction is treated as structurally
///     rank-deficient (Moore-Penrose drop) and listed in
///     `rank_deficient_directions` for the caller to surface;
///   - otherwise H_FF is inverted exactly via the spectral expansion.
pub fn compute_corrected_covariance(
    v_ks: &[Array1<f64>],
    ext_v: &[Array1<f64>],
    outer_hessian: &Array2<f64>,
    hop: &dyn HessianOperator,
) -> Result<Array2<f64>, CorrectedCovarianceError> {
    compute_corrected_covariance_with_constraints(v_ks, ext_v, outer_hessian, hop, None, f64::NAN)
        .map(|cov| {
            if cov.has_structural_diagnostics() {
                log::debug!(
                    "corrected covariance diagnostics: active_constraints={:?} rank_deficient_directions={:?}",
                    cov.active_constraints,
                    cov.rank_deficient_directions
                );
            }
            cov.matrix
        })
}

/// Constraint- and inertia-aware version of [`compute_corrected_covariance`].
///
/// Prefer this entry point when θ at the optimum and the outer-gradient norm
/// are available — it auto-derives the active-bound set on ρ and emits the
/// rank-deficient diagnostic alongside the matrix.
pub fn compute_corrected_covariance_with_constraints(
    v_ks: &[Array1<f64>],
    ext_v: &[Array1<f64>],
    outer_hessian: &Array2<f64>,
    hop: &dyn HessianOperator,
    theta_at_optimum: Option<&[f64]>,
    gradient_norm: f64,
) -> Result<CorrectedCovariance, CorrectedCovarianceError> {
    let p = hop.dim();
    let q = v_ks.len() + ext_v.len();

    if q == 0 {
        let eye = Array2::eye(p);
        return Ok(CorrectedCovariance {
            matrix: hop.solve_multi(&eye),
            active_constraints: Vec::new(),
            rank_deficient_directions: Vec::new(),
        });
    }

    if outer_hessian.nrows() != q || outer_hessian.ncols() != q {
        return Err(CorrectedCovarianceError::ShapeMismatch {
            context: "compute_corrected_covariance outer Hessian",
            reason: format!(
                "dimension ({}, {}) does not match total hyperparameter count q = {} (rho: {}, ext: {})",
                outer_hessian.nrows(),
                outer_hessian.ncols(),
                q,
                v_ks.len(),
                ext_v.len(),
            ),
        });
    }

    let mut j_alpha = Array2::zeros((p, q));
    for (col, v) in v_ks.iter().enumerate() {
        for row in 0..p {
            j_alpha[[row, col]] = -v[row];
        }
    }
    for (i, v) in ext_v.iter().enumerate() {
        let col = v_ks.len() + i;
        for row in 0..p {
            j_alpha[[row, col]] = -v[row];
        }
    }

    let active = active_bound_indices_for_theta(theta_at_optimum, v_ks.len(), ext_v.len());

    let (v_theta, rank_deficient_directions) = projected_inverse_with_inertia_gate(
        outer_hessian,
        &active,
        theta_at_optimum,
        gradient_norm,
    )?;

    let j_v_theta = j_alpha.dot(&v_theta);
    let correction = j_v_theta.dot(&j_alpha.t());

    let eye = Array2::eye(p);
    let mut h_inv = hop.solve_multi(&eye);
    h_inv += &correction;

    enforce_symmetry_inplace(&mut h_inv);

    Ok(CorrectedCovariance {
        matrix: h_inv,
        active_constraints: active,
        rank_deficient_directions,
    })
}

/// Compute only the diagonal of the corrected covariance V*_alpha.
///
/// This is much cheaper than the full p x p matrix: O(p q) instead of O(p^2 q).
///
/// ```text
///   diag(V*_alpha) = diag(H^{-1}) + row_norms(J_alpha L_theta)^2
/// ```
///
/// where L_theta is the Cholesky-like square root of V_theta. When V_theta
/// is obtained via positive-projected eigendecomposition, L_theta = U sqrt(D+)
/// where D+ contains the positive-part eigenvalues.
///
/// # Arguments
/// - `v_ks`: mode responses for rho coordinates
/// - `ext_v`: mode responses for extended (psi) coordinates
/// - `outer_hessian`: the q x q outer Hessian
/// - `hop`: the HessianOperator providing H^{-1}
///
/// # Returns
/// A p-vector of corrected marginal variances.
pub fn compute_corrected_covariance_diagonal(
    v_ks: &[Array1<f64>],
    ext_v: &[Array1<f64>],
    outer_hessian: &Array2<f64>,
    hop: &dyn HessianOperator,
) -> Result<Array1<f64>, CorrectedCovarianceError> {
    compute_corrected_covariance_diagonal_with_constraints(
        v_ks,
        ext_v,
        outer_hessian,
        hop,
        None,
        f64::NAN,
    )
    .map(|d| {
        if d.has_structural_diagnostics() {
            log::debug!(
                "corrected covariance diagonal diagnostics: active_constraints={:?} rank_deficient_directions={:?}",
                d.active_constraints,
                d.rank_deficient_directions
            );
        }
        d.diagonal
    })
}

/// Diagonal of the corrected covariance plus active-set / rank-deficiency
/// diagnostics. See [`compute_corrected_covariance_with_constraints`] for the
/// full version (the inertia gate logic is identical).
#[derive(Debug, Clone)]
pub struct CorrectedCovarianceDiagonal {
    pub diagonal: Array1<f64>,
    pub active_constraints: Vec<usize>,
    pub rank_deficient_directions: Vec<usize>,
}

impl CorrectedCovarianceDiagonal {
    pub(crate) fn has_structural_diagnostics(&self) -> bool {
        !self.active_constraints.is_empty() || !self.rank_deficient_directions.is_empty()
    }
}

pub fn compute_corrected_covariance_diagonal_with_constraints(
    v_ks: &[Array1<f64>],
    ext_v: &[Array1<f64>],
    outer_hessian: &Array2<f64>,
    hop: &dyn HessianOperator,
    theta_at_optimum: Option<&[f64]>,
    gradient_norm: f64,
) -> Result<CorrectedCovarianceDiagonal, CorrectedCovarianceError> {
    let p = hop.dim();
    let q = v_ks.len() + ext_v.len();

    let mut diag = Array1::zeros(p);
    for i in 0..p {
        let mut e_i = Array1::zeros(p);
        e_i[i] = 1.0;
        let h_inv_ei = hop.solve(&e_i);
        diag[i] = h_inv_ei[i];
    }

    if q == 0 {
        return Ok(CorrectedCovarianceDiagonal {
            diagonal: diag,
            active_constraints: Vec::new(),
            rank_deficient_directions: Vec::new(),
        });
    }

    if outer_hessian.nrows() != q || outer_hessian.ncols() != q {
        return Err(CorrectedCovarianceError::ShapeMismatch {
            context: "compute_corrected_covariance_diagonal outer Hessian",
            reason: format!(
                "dimension ({}, {}) does not match q = {}",
                outer_hessian.nrows(),
                outer_hessian.ncols(),
                q,
            ),
        });
    }

    let active = active_bound_indices_for_theta(theta_at_optimum, v_ks.len(), ext_v.len());
    let (v_theta_full, rank_deficient_directions) = projected_inverse_with_inertia_gate(
        outer_hessian,
        &active,
        theta_at_optimum,
        gradient_norm,
    )?;

    // Symmetric square root of V_θ via eigendecomposition (PSD by construction).
    let (sym_evals, sym_evecs) = v_theta_full.eigh(faer::Side::Lower).map_err(|e| {
        CorrectedCovarianceError::EigendecompositionFailed {
            context: "corrected covariance hyperparameter covariance",
            reason: e.to_string(),
        }
    })?;
    let mut v_theta_sqrt = Array2::<f64>::zeros((q, q));
    for j in 0..q {
        let s = sym_evals[j];
        if s <= 0.0 {
            continue;
        }
        let scale = s.sqrt();
        for row in 0..q {
            v_theta_sqrt[[row, j]] = sym_evecs[[row, j]] * scale;
        }
    }

    let mut j_alpha = Array2::zeros((p, q));
    for (col, v) in v_ks.iter().enumerate() {
        for row in 0..p {
            j_alpha[[row, col]] = -v[row];
        }
    }
    for (i, v) in ext_v.iter().enumerate() {
        let col = v_ks.len() + i;
        for row in 0..p {
            j_alpha[[row, col]] = -v[row];
        }
    }

    let m = j_alpha.dot(&v_theta_sqrt);
    for i in 0..p {
        let mut row_norm_sq = 0.0;
        for j in 0..m.ncols() {
            row_norm_sq += m[[i, j]] * m[[i, j]];
        }
        diag[i] += row_norm_sq;
    }

    Ok(CorrectedCovarianceDiagonal {
        diagonal: diag,
        active_constraints: active,
        rank_deficient_directions,
    })
}

/// Enforce exact symmetry on a square matrix by averaging off-diagonal pairs.
pub(crate) fn enforce_symmetry_inplace(m: &mut Array2<f64>) {
    let n = m.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (m[[i, j]] + m[[j, i]]);
            m[[i, j]] = avg;
            m[[j, i]] = avg;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Smooth spectral regularization
// ═══════════════════════════════════════════════════════════════════════════
//
// For indefinite or near-singular Hessians, hard eigenvalue clamping
// `max(σ, ε)` is non-smooth and creates inconsistency between log|H| and
// H⁻¹ at the threshold boundary. We use instead the C∞ regularizer:
//
//   r_ε(σ) = ½(σ + √(σ² + 4ε²))
//
// Properties:
//   - C∞ and strictly positive for all σ ∈ ℝ
//   - r_ε(σ) → σ  as σ → +∞  (transparent for well-conditioned eigenvalues)
//   - r_ε(σ) → ε  as σ → 0   (smooth floor)
//   - r_ε(σ) → ε²/|σ| as σ → -∞  (damps negative eigenvalues)
//
// Its derivative is:
//
//   r'_ε(σ) = ½(1 + σ/√(σ² + 4ε²))
//
// Using the SAME r_ε for both log-determinant and inverse ensures the
// gradient is the exact derivative of a single scalar objective — no
// inconsistency from mixing different regularizations.

/// Smooth spectral regularizer: `r_ε(σ) = ½(σ + √(σ² + 4ε²))`.
///
/// Returns a strictly positive value for any real `sigma`. For large positive
/// `sigma` this is approximately `sigma`; near zero it smoothly floors at `epsilon`.
#[inline]
pub(crate) fn spectral_regularize(sigma: f64, epsilon: f64) -> f64 {
    let disc = sigma.hypot(2.0 * epsilon);
    if sigma >= 0.0 {
        0.5 * sigma + 0.5 * disc
    } else {
        // Avoid catastrophic cancellation in 0.5 * (σ + disc) when σ is
        // large and negative: r_ε(σ) = 2 ε² / (disc - σ).
        (2.0 * epsilon * epsilon) / (disc - sigma)
    }
}

/// Compute the spectral regularization scale for a set of eigenvalues.
///
/// `ε = √(machine_eps) · p` where `p` is the matrix dimension — a
/// ρ-INDEPENDENT numerical-stability floor on the smooth regularization
/// `r_ε(σ)`.  The previous formulation `ε = √(machine_eps) · max(|σ_max|, 1)`
/// coupled ε to the largest eigenvalue of H(ρ), which made ε a function of
/// ρ whenever the Hessian spectrum moved with ρ.  That coupling leaked a
/// spurious ∂log|H|_reg/∂ρ contribution through the near-zero eigenvalues:
/// for σ_j ≪ ε we have `log r_ε(σ_j) ≈ log ε`, so `d log r_ε(σ_j)/dρ`
/// picks up `(1/ε) · dε/dρ` whenever max|σ_j| moved.  That created a
/// first-order derivative mismatch in outer REML gradients (up to ~1.5% of
/// the dominant `d log|H|/dρ` term on problems with one near-singular
/// direction, e.g. multi-block GAMLSS wiggle models where the intercept/wiggle
/// direction is effectively in the null space of the likelihood curvature).
///
/// The analytic gradient formula `tr(G_ε(H) · dH/dρ_k)` assumes ε is
/// fixed; removing the ρ-coupling restores that assumption.  Scaling ε
/// by the matrix dimension `p` (a ρ-independent integer, set by the
/// problem geometry) gives numerical stability for larger systems without
/// reintroducing ρ leakage.  The absolute floor stays below any physically
/// meaningful eigenvalue (for p ≤ 10⁶, ε ≤ 1.5e-2; well-conditioned
/// problems have min σ ≫ ε and are unaffected).
#[inline]
pub(crate) fn spectral_epsilon(eigenvalues: &[f64]) -> f64 {
    f64::EPSILON.sqrt() * (eigenvalues.len() as f64).max(1.0)
}
