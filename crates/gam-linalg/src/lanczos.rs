use faer::Side;
use ndarray::{Array1, Array2};

use crate::faer_ndarray::FaerEigh;

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
    /// `β_k`: the Euclidean norm of the (unnormalized) next Lanczos vector after
    /// the final accepted step — i.e. the off-diagonal that WOULD extend `T_k`.
    /// This is the residual norm in `H Q_k = Q_k T_k + β_k q_{k+1} e_kᵀ`; with
    /// full reorthogonalization it yields the sharp per-Ritz-pair residual
    /// `β_k·|e_kᵀ y_i|`. Zero on a lucky breakdown (Krylov space exhausted, so
    /// the Ritz spectrum is exact).
    pub residual_norm: f64,
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
    let mut basis: Vec<Vec<f64>> = if options.full_reorthogonalize {
        Vec::with_capacity(steps)
    } else {
        Vec::new()
    };
    // β_k carried out of the loop: the norm of the unnormalized next Lanczos
    // vector after the final accepted α. Zero on a lucky breakdown.
    let mut residual_norm = 0.0_f64;

    for step in 0..steps {
        if options.full_reorthogonalize {
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
    Ok(SymmetricLanczosEigenpairs {
        eigenvalues,
        eigenvectors,
        residual_norm,
    })
}
