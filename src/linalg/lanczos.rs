use faer::Side;
use ndarray::Array2;

use crate::linalg::faer_ndarray::FaerEigh;

#[derive(Clone, Debug)]
pub struct LanczosEigenDecomposition {
    pub eigenvalues: ndarray::Array1<f64>,
    pub eigenvectors: Array2<f64>,
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0_f64;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline]
fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

pub fn symmetric_lanczos_eigh<F>(
    dim: usize,
    mut q: Vec<f64>,
    max_steps: usize,
    breakdown_tol: f64,
    context: &str,
    mut matvec_into: F,
) -> Result<LanczosEigenDecomposition, String>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<(), String>,
{
    if dim == 0 {
        return Err(format!("{context} Lanczos requires dim > 0"));
    }
    if q.len() != dim {
        return Err(format!(
            "{context} Lanczos start vector length {} != dim {dim}",
            q.len()
        ));
    }
    if max_steps == 0 {
        return Err(format!("{context} Lanczos requires max_steps > 0"));
    }
    if !(breakdown_tol.is_finite() && breakdown_tol >= 0.0) {
        return Err(format!(
            "{context} Lanczos requires finite non-negative breakdown_tol, got {breakdown_tol}"
        ));
    }

    let q_norm = norm2(&q);
    if !(q_norm.is_finite() && q_norm > 0.0) {
        return Err(format!(
            "{context} Lanczos start vector must have finite positive norm, got {q_norm}"
        ));
    }
    for value in &mut q {
        *value /= q_norm;
    }

    let steps = max_steps.min(dim).max(1);
    let mut q_prev = vec![0.0_f64; dim];
    let mut alphas = Vec::<f64>::with_capacity(steps);
    let mut betas = Vec::<f64>::with_capacity(steps.saturating_sub(1));
    let mut beta_prev = 0.0_f64;

    for step in 0..steps {
        let mut w = vec![0.0_f64; dim];
        matvec_into(&q, &mut w)?;
        if w.len() != dim || w.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "{context} Lanczos matvec produced a non-finite vector"
            ));
        }

        if step > 0 {
            for i in 0..dim {
                w[i] -= beta_prev * q_prev[i];
            }
        }
        let alpha = dot(&q, &w);
        if !alpha.is_finite() {
            return Err(format!("{context} Lanczos produced non-finite alpha"));
        }
        for i in 0..dim {
            w[i] -= alpha * q[i];
        }

        // One local reorthogonalization pass against the two live basis vectors
        // keeps the small tridiagonal symmetric without storing the full basis.
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

        let beta = norm2(&w);
        alphas.push(alpha);
        if step + 1 == steps || beta <= breakdown_tol {
            break;
        }
        if !beta.is_finite() {
            return Err(format!("{context} Lanczos produced non-finite beta"));
        }
        betas.push(beta);
        q_prev = q;
        q = w;
        for value in &mut q {
            *value /= beta;
        }
        beta_prev = beta;
    }

    let k = alphas.len();
    let mut tri = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        tri[[i, i]] = alphas[i];
        if i + 1 < k {
            tri[[i, i + 1]] = betas[i];
            tri[[i + 1, i]] = betas[i];
        }
    }
    let (eigenvalues, eigenvectors) = tri
        .eigh(Side::Lower)
        .map_err(|e| format!("{context} Lanczos tridiagonal eigendecomposition failed: {e}"))?;
    Ok(LanczosEigenDecomposition {
        eigenvalues,
        eigenvectors,
    })
}

pub fn lanczos_log_quadrature(
    decomp: &LanczosEigenDecomposition,
    spd_context: &str,
) -> Result<f64, String> {
    let mut quad = 0.0_f64;
    for j in 0..decomp.eigenvalues.len() {
        let theta = decomp.eigenvalues[j];
        if !theta.is_finite() || theta <= 0.0 {
            return Err(format!(
                "{spd_context} expected SPD operator, Lanczos Ritz value {j} is {theta:.3e}"
            ));
        }
        let weight = decomp.eigenvectors[[0, j]] * decomp.eigenvectors[[0, j]];
        quad += weight * theta.ln();
    }
    Ok(quad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_lanczos_eigh_recovers_diagonal_ritz_values() {
        let diag = [1.0_f64, 2.0, 4.0];
        let q0 = vec![1.0, 1.0, 1.0];
        let decomp = symmetric_lanczos_eigh(
            diag.len(),
            q0,
            diag.len(),
            1e-14,
            "test diagonal",
            |q, out| {
                for i in 0..diag.len() {
                    out[i] = diag[i] * q[i];
                }
                Ok(())
            },
        )
        .expect("lanczos diagonal");

        for expected in diag {
            assert!(
                decomp
                    .eigenvalues
                    .iter()
                    .any(|value| (*value - expected).abs() < 1e-10),
                "missing diagonal Ritz value {expected}; got {:?}",
                decomp.eigenvalues
            );
        }
        let quad = lanczos_log_quadrature(&decomp, "test diagonal").expect("log quadrature");
        assert!(quad.is_finite());
    }
}
