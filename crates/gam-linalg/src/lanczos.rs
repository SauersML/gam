use faer::Side;
use ndarray::{Array1, Array2, Axis};

use crate::faer_ndarray::{FaerEigh, strict_symmetric_eigh};

fn tridiagonal_eigenpairs(
    diagonal: &[f64],
    off_diagonal: &[f64],
) -> Result<(Array1<f64>, Array2<f64>), String> {
    let dimension = diagonal.len();
    if dimension == 0 || off_diagonal.len() + 1 != dimension {
        return Err(format!(
            "tridiagonal eigensystem shape mismatch: diagonal={}, off_diagonal={}",
            dimension,
            off_diagonal.len()
        ));
    }
    let matrix = tridiagonal_from_coefficients(diagonal, off_diagonal);
    strict_symmetric_eigh(&matrix, Side::Lower)
        .map_err(|error| format!("tridiagonal eigendecomposition failed: {error:?}"))
}

/// Partition a descending symmetric spectrum exactly as mgcv's
/// `Rlanczos(..., lm=-1)` contract does: merge the positive and negative ends
/// by absolute magnitude, then return all selected upper eigenpairs followed
/// by the selected lower eigenpairs.  Keeping this column chart matters after
/// the polynomial side-condition QR; an arbitrary magnitude sort represents
/// the same exact subspace but takes a different finite-precision QR path.
fn mgcv_largest_magnitude_indices(values_ascending: &Array1<f64>, rank: usize) -> Vec<usize> {
    let n = values_ascending.len();
    let mut upper = 0usize;
    let mut lower = 0usize;
    while upper + lower < rank {
        let upper_index = n - 1 - upper;
        let lower_index = lower;
        if values_ascending[upper_index].abs() >= values_ascending[lower_index].abs() {
            upper += 1;
        } else {
            lower += 1;
        }
    }
    let mut indices = Vec::with_capacity(rank);
    indices.extend((n - upper..n).rev());
    indices.extend((0..lower).rev());
    indices
}

#[derive(Debug, Clone, Copy)]
pub struct SymmetricLanczosOptions {
    pub max_steps: usize,
    pub residual_tol: f64,
    /// Local reorthogonalization: re-project the new Lanczos vector against the
    /// CURRENT and PREVIOUS vectors only (cheap; controls the dominant
    /// three-term-recurrence drift). Sufficient for SLQ log-det quadrature where
    /// only the first row of the Ritz vectors is read.
    pub local_reorthogonalize: bool,
    /// Full reorthogonalization: classical Gram–Schmidt against the ENTIRE
    /// accumulated Krylov basis, applied twice for numerical robustness. This
    /// keeps `Q_k` orthonormal to machine precision, so the factorization
    /// `H Q_k = Q_k T_k + β_k q_{k+1} e_kᵀ` holds exactly and the per-Ritz-pair
    /// residual `β_k·|e_kᵀ y_i|` is a SHARP eigenvalue bound (no ghost
    /// eigenvalues). Required by callers that certify extreme-eigenvalue bounds
    /// from the returned `residual_norm`. When set it supersedes
    /// `local_reorthogonalize`.
    pub full_reorthogonalize: bool,
}

#[derive(Debug, Clone)]
pub struct SymmetricLanczosEigenpairs {
    pub eigenvalues: Array1<f64>,
    pub eigenvectors: Array2<f64>,
    /// Ritz vectors lifted from the tridiagonal Krylov coordinates back into
    /// the original operator coordinates.  Populated only by
    /// [`symmetric_lanczos_eigenpairs_with_original_vectors`]; the ordinary
    /// eigenpair routine leaves it absent so callers that need only quadrature
    /// nodes do not retain an extra `dimension × steps` block.
    pub original_eigenvectors: Option<Array2<f64>>,
    /// `β_k`: the Euclidean norm of the (unnormalized) next Lanczos vector after
    /// the final accepted step — i.e. the off-diagonal that WOULD extend `T_k`.
    /// This is the residual norm in `H Q_k = Q_k T_k + β_k q_{k+1} e_kᵀ`; with
    /// full reorthogonalization it yields the sharp per-Ritz-pair residual
    /// `β_k·|e_kᵀ y_i|`. Zero on a lucky breakdown (Krylov space exhausted, so
    /// the Ritz spectrum is exact).
    pub residual_norm: f64,
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn tridiagonal_from_coefficients(alphas: &[f64], betas: &[f64]) -> Array2<f64> {
    let k = alphas.len();
    let mut tri = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        tri[[i, i]] = alphas[i];
        if i + 1 < k {
            tri[[i, i + 1]] = betas[i];
            tri[[i + 1, i]] = betas[i];
        }
    }
    tri
}

pub fn symmetric_lanczos_eigenpairs(
    dim: usize,
    start: &[f64],
    options: SymmetricLanczosOptions,
    apply: impl FnMut(&[f64], &mut [f64]) -> Result<(), String>,
) -> Result<SymmetricLanczosEigenpairs, String> {
    symmetric_lanczos_eigenpairs_impl(dim, start, options, false, apply)
}

/// [`symmetric_lanczos_eigenpairs`] with the tridiagonal Ritz vectors lifted
/// back through the retained Krylov basis.
///
/// This is opt-in because the lifted block is `dimension × steps`.  Exact-A
/// matrix-free evidence uses it only while classifying Ritz directions in the
/// majorizer metric and drops it immediately after reducing those directions to
/// scalar quadratic forms (#2515).
pub fn symmetric_lanczos_eigenpairs_with_original_vectors(
    dim: usize,
    start: &[f64],
    options: SymmetricLanczosOptions,
    apply: impl FnMut(&[f64], &mut [f64]) -> Result<(), String>,
) -> Result<SymmetricLanczosEigenpairs, String> {
    symmetric_lanczos_eigenpairs_impl(dim, start, options, true, apply)
}

fn symmetric_lanczos_eigenpairs_impl(
    dim: usize,
    start: &[f64],
    options: SymmetricLanczosOptions,
    lift_original_vectors: bool,
    mut apply: impl FnMut(&[f64], &mut [f64]) -> Result<(), String>,
) -> Result<SymmetricLanczosEigenpairs, String> {
    if dim == 0 {
        return Err("symmetric Lanczos requires positive dimension".to_string());
    }
    if start.len() != dim {
        return Err(format!(
            "symmetric Lanczos start-vector dimension mismatch: got {}, expected {dim}",
            start.len()
        ));
    }
    if options.max_steps == 0 {
        return Err("symmetric Lanczos requires max_steps > 0".to_string());
    }
    if !options.residual_tol.is_finite() || options.residual_tol < 0.0 {
        return Err(format!(
            "symmetric Lanczos requires finite non-negative residual_tol, got {}",
            options.residual_tol
        ));
    }

    let mut q_prev = vec![0.0_f64; dim];
    let mut q = start.to_vec();
    if q.iter().any(|v| !v.is_finite()) {
        return Err("symmetric Lanczos start vector contains non-finite entries".to_string());
    }
    let q_norm = norm2(&q);
    if !q_norm.is_finite() || q_norm <= 0.0 {
        return Err("symmetric Lanczos start vector must have positive finite norm".to_string());
    }
    for v in &mut q {
        *v /= q_norm;
    }

    let steps = options.max_steps.min(dim).max(1);
    let mut alphas = Vec::<f64>::with_capacity(steps);
    let mut betas = Vec::<f64>::with_capacity(steps.saturating_sub(1));
    let mut beta_prev = 0.0_f64;
    let mut w = vec![0.0_f64; dim];
    // Full-reorthogonalization basis (only retained when requested; classical
    // Gram–Schmidt below sweeps it twice). Kept as `q_j` BEFORE the matvec so it
    // mirrors the three-term recurrence order.
    let retain_basis = options.full_reorthogonalize || lift_original_vectors;
    let mut basis: Vec<Vec<f64>> = if retain_basis {
        Vec::with_capacity(steps)
    } else {
        Vec::new()
    };
    // β_k carried out of the loop: the norm of the unnormalized next Lanczos
    // vector after the final accepted α. Zero on a lucky breakdown.
    let mut residual_norm = 0.0_f64;

    for step in 0..steps {
        if retain_basis {
            basis.push(q.clone());
        }
        w.fill(0.0);
        apply(&q, &mut w)?;
        if w.len() != dim || w.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "symmetric Lanczos matvec expected finite vector of length {dim}, got {}",
                w.len()
            ));
        }
        if step > 0 {
            for i in 0..dim {
                w[i] -= beta_prev * q_prev[i];
            }
        }

        let alpha = dot(&q, &w);
        if !alpha.is_finite() {
            return Err("symmetric Lanczos produced non-finite alpha".to_string());
        }
        for i in 0..dim {
            w[i] -= alpha * q[i];
        }
        if options.full_reorthogonalize {
            // Classical Gram–Schmidt against the whole basis, swept twice for
            // robustness at small scale (Q_k orthonormal ⇒ sharp residual bound).
            for _pass in 0..2 {
                for qi in basis.iter() {
                    let proj = dot(qi, &w);
                    for i in 0..dim {
                        w[i] -= proj * qi[i];
                    }
                }
            }
        } else if options.local_reorthogonalize {
            let proj_q = dot(&q, &w);
            for i in 0..dim {
                w[i] -= proj_q * q[i];
            }
            if step > 0 {
                let proj_prev = dot(&q_prev, &w);
                for i in 0..dim {
                    w[i] -= proj_prev * q_prev[i];
                }
            }
        }

        let beta = norm2(&w);
        alphas.push(alpha);
        if !beta.is_finite() {
            return Err("symmetric Lanczos produced non-finite beta".to_string());
        }
        residual_norm = beta;
        if step + 1 == steps || beta <= options.residual_tol {
            // Lucky breakdown / exhausted Krylov space: the Ritz spectrum is
            // exact, so report a zero residual rather than the tolerance floor.
            if beta <= options.residual_tol {
                residual_norm = 0.0;
            }
            break;
        }
        betas.push(beta);
        q_prev.clone_from(&q);
        for i in 0..dim {
            q[i] = w[i] / beta;
        }
        beta_prev = beta;
    }

    let tri = tridiagonal_from_coefficients(&alphas, &betas);
    let (eigenvalues, eigenvectors) = tri.eigh(Side::Lower).map_err(|err| {
        format!("symmetric Lanczos tridiagonal eigendecomposition failed: {err:?}")
    })?;
    let original_eigenvectors = lift_original_vectors.then(|| {
        Array2::from_shape_fn((dim, eigenvectors.ncols()), |(row, ritz)| {
            basis
                .iter()
                .enumerate()
                .map(|(krylov, direction)| direction[row] * eigenvectors[[krylov, ritz]])
                .sum()
        })
    });
    Ok(SymmetricLanczosEigenpairs {
        eigenvalues,
        eigenvectors,
        original_eigenvectors,
        residual_norm,
    })
}

/// Configuration for an adaptive, certified extreme-eigenpair solve.
#[derive(Debug, Clone, Copy)]
pub struct SymmetricExtremeLanczosOptions {
    /// Number of eigenpairs of largest absolute eigenvalue to return.
    pub target_rank: usize,
    /// Hard work bound. Failure to certify by this step is an error.
    pub max_steps: usize,
    /// Recompute the tiny tridiagonal eigensystem at this cadence.
    pub check_every: usize,
    /// Required `||A v - λ v||₂ / max(||Λ_selected||∞, 1)` for every returned
    /// pair. Scaling every Ritz residual by the retained operator norm gives
    /// one invariant certificate for the requested eigenspace, including
    /// clustered eigenvalues.
    pub relative_residual_tol: f64,
    /// Norm below which the Krylov recurrence has exactly exhausted its space.
    pub breakdown_tol: f64,
}

#[derive(Debug, Clone)]
pub struct SymmetricExtremeLanczosEigenpairs {
    pub eigenvalues: Array1<f64>,
    /// Original-coordinate Ritz vectors, one per column.
    pub eigenvectors: Array2<f64>,
    /// Sharp residual bounds `β_k |e_kᵀ y_i|`, one per returned pair.
    pub residual_bounds: Array1<f64>,
}

/// Compute extreme-magnitude eigenpairs and stop at the first certified
/// Lanczos checkpoint.
///
/// Full double reorthogonalization keeps the Krylov basis orthonormal, making
/// `β_k |e_kᵀ y_i|` a sharp original-coordinate Ritz residual. The old
/// fixed-step pattern paid for every guessed step before checking convergence;
/// this solver checks the inexpensive tridiagonal problem periodically and
/// performs no matrix-vector products after the requested eigenspace is
/// certified.
pub fn symmetric_extreme_lanczos_eigenpairs(
    dim: usize,
    start: &[f64],
    options: SymmetricExtremeLanczosOptions,
    mut apply: impl FnMut(&[f64], &mut [f64]) -> Result<(), String>,
) -> Result<SymmetricExtremeLanczosEigenpairs, String> {
    if dim == 0 {
        return Err("extreme symmetric Lanczos requires positive dimension".to_string());
    }
    if start.len() != dim {
        return Err(format!(
            "extreme symmetric Lanczos start-vector dimension mismatch: got {}, expected {dim}",
            start.len()
        ));
    }
    if options.target_rank == 0 || options.target_rank > dim {
        return Err(format!(
            "extreme symmetric Lanczos target rank {} must lie in 1..={dim}",
            options.target_rank
        ));
    }
    if options.max_steps < options.target_rank {
        return Err(format!(
            "extreme symmetric Lanczos max_steps {} is smaller than target rank {}",
            options.max_steps, options.target_rank
        ));
    }
    if options.check_every == 0 {
        return Err("extreme symmetric Lanczos requires check_every > 0".to_string());
    }
    if !options.relative_residual_tol.is_finite() || options.relative_residual_tol <= 0.0 {
        return Err(format!(
            "extreme symmetric Lanczos requires a positive finite relative residual tolerance, \
             got {}",
            options.relative_residual_tol
        ));
    }
    if !options.breakdown_tol.is_finite() || options.breakdown_tol < 0.0 {
        return Err(format!(
            "extreme symmetric Lanczos requires a finite non-negative breakdown tolerance, got {}",
            options.breakdown_tol
        ));
    }
    if start.iter().any(|value| !value.is_finite()) {
        return Err("extreme symmetric Lanczos start contains non-finite entries".to_string());
    }

    let mut q = start.to_vec();
    let q_norm = norm2(&q);
    if !q_norm.is_finite() || q_norm <= 0.0 {
        return Err(
            "extreme symmetric Lanczos start vector must have positive finite norm".to_string(),
        );
    }
    for value in &mut q {
        *value /= q_norm;
    }

    let steps = options.max_steps.min(dim);
    let mut q_prev = vec![0.0_f64; dim];
    let mut beta_prev = 0.0_f64;
    let mut w = vec![0.0_f64; dim];
    let mut basis = Vec::<Vec<f64>>::with_capacity(steps);
    let mut alphas = Vec::<f64>::with_capacity(steps);
    let mut betas = Vec::<f64>::with_capacity(steps.saturating_sub(1));
    let mut last_worst_relative_residual = f64::INFINITY;

    for step in 0..steps {
        basis.push(q.clone());
        w.fill(0.0);
        apply(&q, &mut w)?;
        if w.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "extreme symmetric Lanczos operator produced non-finite values at step {}",
                step + 1
            ));
        }
        // Rlanczos deliberately computes alpha in a scalar left-to-right loop
        // even though its reorthogonalization projections use BLAS DDOT.
        // Substituting DDOT here changes the Krylov chart on clustered spectra.
        let alpha = dot(&q, &w);
        if !alpha.is_finite() {
            return Err("extreme symmetric Lanczos produced non-finite alpha".to_string());
        }
        if step == 0 {
            for i in 0..dim {
                w[i] -= alpha * q[i];
            }
        } else {
            for i in 0..dim {
                w[i] -= alpha * q[i] + beta_prev * q_prev[i];
            }
        }
        if step > 0 {
            for _ in 0..2 {
                for qi in &basis {
                    let projection = -dot(&w, qi);
                    for (output, &basis_value) in w.iter_mut().zip(qi) {
                        *output = projection.mul_add(basis_value, *output);
                    }
                }
            }
        }
        let beta = norm2(&w);
        if !beta.is_finite() {
            return Err("extreme symmetric Lanczos produced non-finite beta".to_string());
        }
        alphas.push(alpha);

        let k = alphas.len();
        let exhausted = beta <= options.breakdown_tol;
        let completed_index = k - 1;
        let checkpoint = k >= options.target_rank
            && (exhausted
                || k == steps
                || (completed_index >= options.target_rank
                    && completed_index.is_multiple_of(options.check_every)));
        if checkpoint {
            let (values, vectors) = tridiagonal_eigenpairs(&alphas, &betas)?;
            let selected_indices =
                mgcv_largest_magnitude_indices(&values, options.target_rank);
            let residual_scale = if exhausted { 0.0 } else { beta };
            let selected_operator_scale = selected_indices
                .iter()
                .map(|&index| values[index].abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let mut residual_bounds = Array1::<f64>::zeros(options.target_rank);
            last_worst_relative_residual = 0.0;
            for (out, &j) in selected_indices.iter().enumerate() {
                let residual = residual_scale * vectors[[k - 1, j]].abs();
                residual_bounds[out] = residual;
                last_worst_relative_residual =
                    last_worst_relative_residual.max(residual / selected_operator_scale);
            }
            if last_worst_relative_residual <= options.relative_residual_tol {
                let mut selected_vectors = Array2::<f64>::zeros((dim, options.target_rank));
                for (output_column, &small_column) in selected_indices.iter().enumerate() {
                    for krylov_column in 0..k {
                        let scale = vectors[[krylov_column, small_column]];
                        for row in 0..dim {
                            selected_vectors[[row, output_column]] +=
                                basis[krylov_column][row] * scale;
                        }
                    }
                }
                return Ok(SymmetricExtremeLanczosEigenpairs {
                    eigenvalues: values.select(Axis(0), &selected_indices),
                    eigenvectors: selected_vectors,
                    residual_bounds,
                });
            }
            if exhausted {
                return Err(format!(
                    "extreme symmetric Lanczos exhausted its Krylov space after {k} steps before \
                     certification (worst relative residual {last_worst_relative_residual:.3e})"
                ));
            }
        }
        if step + 1 == steps {
            break;
        }
        betas.push(beta);
        q_prev.clone_from(&q);
        for i in 0..dim {
            q[i] = w[i] / beta;
        }
        beta_prev = beta;
    }

    Err(format!(
        "extreme symmetric Lanczos failed to certify rank {} after {} steps \
         (worst relative residual {last_worst_relative_residual:.3e}, tolerance {:.3e})",
        options.target_rank, steps, options.relative_residual_tol
    ))
}

pub fn symmetric_lanczos_log_quadrature(
    eigenpairs: &SymmetricLanczosEigenpairs,
    spd_context: &str,
) -> Result<f64, String> {
    let k = eigenpairs.eigenvalues.len();
    if eigenpairs.eigenvectors.nrows() == 0 || eigenpairs.eigenvectors.ncols() != k {
        return Err(format!(
            "{spd_context}: Lanczos eigenvector shape mismatch: got ({}, {}), expected first row and {k} columns",
            eigenpairs.eigenvectors.nrows(),
            eigenpairs.eigenvectors.ncols(),
        ));
    }
    let mut quad = 0.0_f64;
    for j in 0..k {
        let theta = eigenpairs.eigenvalues[j];
        if !theta.is_finite() || theta <= 0.0 {
            return Err(format!(
                "{spd_context}: expected positive finite Ritz value {j}, got {theta:.3e}"
            ));
        }
        let weight = eigenpairs.eigenvectors[[0, j]] * eigenpairs.eigenvectors[[0, j]];
        quad += weight * theta.ln();
    }
    Ok(quad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn no_reortho() -> SymmetricLanczosOptions {
        SymmetricLanczosOptions {
            max_steps: 10,
            residual_tol: 1e-12,
            local_reorthogonalize: false,
            full_reorthogonalize: false,
        }
    }

    // ── symmetric_lanczos_log_quadrature ─────────────────────────────────────

    #[test]
    fn log_quadrature_empty_eigenvectors_is_error() {
        let ep = SymmetricLanczosEigenpairs {
            eigenvalues: array![1.0],
            eigenvectors: ndarray::Array2::zeros((0, 1)),
            original_eigenvectors: None,
            residual_norm: 0.0,
        };
        assert!(symmetric_lanczos_log_quadrature(&ep, "ctx").is_err());
    }

    #[test]
    fn log_quadrature_non_positive_eigenvalue_is_error() {
        let ep = SymmetricLanczosEigenpairs {
            eigenvalues: array![0.0],
            eigenvectors: array![[1.0]],
            original_eigenvectors: None,
            residual_norm: 0.0,
        };
        let err = symmetric_lanczos_log_quadrature(&ep, "myctx").unwrap_err();
        assert!(err.contains("myctx"), "error should mention context: {err}");
    }

    #[test]
    fn log_quadrature_single_eigenvalue_at_e_gives_one() {
        let ep = SymmetricLanczosEigenpairs {
            eigenvalues: array![std::f64::consts::E],
            eigenvectors: array![[1.0]],
            original_eigenvectors: None,
            residual_norm: 0.0,
        };
        let result = symmetric_lanczos_log_quadrature(&ep, "ctx").unwrap();
        assert!((result - 1.0).abs() < 1e-14);
    }

    #[test]
    fn log_quadrature_two_eigenvalues_weighted_sum() {
        // weights 0.5² each, eigenvalues 2 and 8 → 0.25*(ln2 + ln8) = 0.25*4*ln2 = ln2
        let ep = SymmetricLanczosEigenpairs {
            eigenvalues: array![2.0, 8.0],
            eigenvectors: array![[0.5, 0.5]],
            original_eigenvectors: None,
            residual_norm: 0.0,
        };
        let result = symmetric_lanczos_log_quadrature(&ep, "ctx").unwrap();
        let expected = 0.25 * (2.0_f64.ln() + 8.0_f64.ln());
        assert!((result - expected).abs() < 1e-14);
    }

    // ── symmetric_lanczos_eigenpairs — validation ─────────────────────────────

    #[test]
    fn eigenpairs_zero_dim_is_error() {
        let r = symmetric_lanczos_eigenpairs(0, &[], no_reortho(), |_, _| Ok(()));
        assert!(r.is_err());
    }

    #[test]
    fn eigenpairs_start_dim_mismatch_is_error() {
        let r = symmetric_lanczos_eigenpairs(3, &[1.0, 0.0], no_reortho(), |_, _| Ok(()));
        assert!(r.is_err());
    }

    #[test]
    fn eigenpairs_zero_max_steps_is_error() {
        let opts = SymmetricLanczosOptions {
            max_steps: 0,
            ..no_reortho()
        };
        let r = symmetric_lanczos_eigenpairs(1, &[1.0], opts, |_, _| Ok(()));
        assert!(r.is_err());
    }

    #[test]
    fn eigenpairs_infinite_residual_tol_is_error() {
        let opts = SymmetricLanczosOptions {
            residual_tol: f64::INFINITY,
            ..no_reortho()
        };
        let r = symmetric_lanczos_eigenpairs(1, &[1.0], opts, |_, _| Ok(()));
        assert!(r.is_err());
    }

    #[test]
    fn eigenpairs_non_finite_start_is_error() {
        let r = symmetric_lanczos_eigenpairs(1, &[f64::NAN], no_reortho(), |_, _| Ok(()));
        assert!(r.is_err());
    }

    // ── symmetric_lanczos_eigenpairs — correctness ────────────────────────────

    #[test]
    fn eigenpairs_1x1_diagonal_recovers_exact_eigenvalue() {
        let ep = symmetric_lanczos_eigenpairs(1, &[1.0], no_reortho(), |q, w| {
            w[0] = 3.0 * q[0];
            Ok(())
        })
        .unwrap();
        assert_eq!(ep.eigenvalues.len(), 1);
        assert!((ep.eigenvalues[0] - 3.0).abs() < 1e-12);
        assert_eq!(ep.residual_norm, 0.0);
    }

    #[test]
    fn eigenpairs_2x2_diagonal_recovers_both_eigenvalues() {
        let sq2_inv = 1.0_f64 / 2.0_f64.sqrt();
        let ep = symmetric_lanczos_eigenpairs(2, &[sq2_inv, sq2_inv], no_reortho(), |q, w| {
            w[0] = 1.0 * q[0];
            w[1] = 4.0 * q[1];
            Ok(())
        })
        .unwrap();
        assert_eq!(ep.eigenvalues.len(), 2);
        let mut evs = ep.eigenvalues.to_vec();
        evs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((evs[0] - 1.0).abs() < 1e-10, "smallest: {}", evs[0]);
        assert!((evs[1] - 4.0).abs() < 1e-10, "largest: {}", evs[1]);
        assert_eq!(ep.residual_norm, 0.0);
    }

    #[test]
    fn adaptive_extreme_lanczos_returns_certified_original_coordinate_eigenpairs() {
        let diagonal = [9.0_f64, -7.0, 4.0, 2.0, 1.0, 0.5];
        let start = [1.0_f64, -0.7, 0.5, 1.3, -0.2, 0.9];
        let pairs = symmetric_extreme_lanczos_eigenpairs(
            6,
            &start,
            SymmetricExtremeLanczosOptions {
                target_rank: 2,
                max_steps: 6,
                check_every: 1,
                relative_residual_tol: 1e-10,
                breakdown_tol: 1e-14,
            },
            |q, out| {
                for i in 0..6 {
                    out[i] = diagonal[i] * q[i];
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(pairs.eigenvectors.dim(), (6, 2));
        assert!((pairs.eigenvalues[0] - 9.0).abs() < 1e-10);
        assert!((pairs.eigenvalues[1] + 7.0).abs() < 1e-10);
        for j in 0..2 {
            let lambda = pairs.eigenvalues[j];
            let mut residual_squared = 0.0;
            for i in 0..6 {
                let residual =
                    diagonal[i] * pairs.eigenvectors[[i, j]] - lambda * pairs.eigenvectors[[i, j]];
                residual_squared += residual * residual;
            }
            assert!(residual_squared.sqrt() < 1e-10);
            assert!(pairs.residual_bounds[j] < 1e-10);
        }
    }
}
