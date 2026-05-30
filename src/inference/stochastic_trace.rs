//! Variance-reduced stochastic trace estimation for large SPD operators.
//!
//! REML / LAML smoothing-parameter gradients need traces like
//! `tr(H⁻¹ S_k)` and log-determinant derivatives of operators that are too
//! large to factor densely. The classical Hutchinson estimator
//! `tr(A) ≈ (1/k) Σ_j z_jᵀ A z_j` (with `E[z zᵀ] = I`, e.g. Rademacher probes)
//! is unbiased but high-variance. A deterministic-plus-stochastic *control
//! variate* removes the dominant spectrum exactly and only estimates the
//! residual stochastically (Hutch++ / stochastic-Lanczos-quadrature style):
//!
//! ```text
//! tr(A) = tr(QᵀA Q) + tr((I − QQᵀ) A),
//! ```
//!
//! where `Q` (`n×r`) is an orthonormal basis for an approximate dominant
//! subspace (e.g. Ritz vectors carried across outer iterations). The first term
//! is exact; the second is estimated by Hutchinson on probes deflated against
//! `Q`. When `span(Q)` captures the large eigenvalues the residual is small and
//! its estimate has far lower variance than plain Hutchinson — fewer
//! matrix-vector products for the same accuracy, and better CPU/GPU parity.
//!
//! The operators are supplied through their already-computed actions
//! (`A·probes`, `A·Q`) so the estimators are matrix-free and decoupled from how
//! `A` is applied (dense, sparse Cholesky solve, GPU kernel, …).

use ndarray::ArrayView2;

/// Hutchinson trace estimate `(1/k) Σ_j z_jᵀ A z_j` from probes `Z` (`n×k`) and
/// their images `A·Z` (`n×k`). Unbiased for `tr(A)` when `(1/k) Σ_j z_j z_jᵀ = I`
/// (Rademacher probes in expectation; a full scaled-Hadamard probe set makes it
/// exact for any `A`).
pub fn hutchinson_trace(probes: ArrayView2<'_, f64>, a_probes: ArrayView2<'_, f64>) -> f64 {
    let k = probes.ncols();
    if k == 0 {
        return 0.0;
    }
    let mut acc = 0.0;
    for j in 0..k {
        acc += probes.column(j).dot(&a_probes.column(j));
    }
    acc / k as f64
}

/// Control-variate trace: exact low-rank part `tr(QᵀA Q)` plus a Hutchinson
/// estimate of the residual `tr((I − QQᵀ) A)`.
///
/// `q` is an orthonormal basis (`n×r`, `QᵀQ = I`), `a_q = A·Q` (`n×r`),
/// `probes` (`n×k`) and `a_probes = A·probes` (`n×k`). The decomposition
/// `tr(A) = tr(QᵀAQ) + tr((I−QQᵀ)A)` is exact, so this is unbiased for any `Q`
/// and reduces to [`hutchinson_trace`] when `Q` is empty. With `k = 0` it
/// returns just the exact low-rank part (useful when `Q` already spans the
/// operator).
pub fn controlled_trace(
    q: ArrayView2<'_, f64>,
    a_q: ArrayView2<'_, f64>,
    probes: ArrayView2<'_, f64>,
    a_probes: ArrayView2<'_, f64>,
) -> f64 {
    let r = q.ncols();
    let mut exact = 0.0;
    for i in 0..r {
        exact += q.column(i).dot(&a_q.column(i));
    }
    let k = probes.ncols();
    if k == 0 {
        return exact;
    }
    // residual term: (1/k) Σ_j ((I − QQᵀ) z_j)ᵀ (A z_j), using the symmetry of
    // (I − QQᵀ) so only the probe is deflated, not its image.
    let mut residual = 0.0;
    for j in 0..k {
        let z = probes.column(j);
        let qtz = q.t().dot(&z);
        let z_def = &z.to_owned() - &q.dot(&qtz);
        residual += z_def.dot(&a_probes.column(j));
    }
    exact + residual / k as f64
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, ArrayView2};

    use super::*;

    // Order-4 Hadamard matrix; columns are orthogonal with norm 2 and
    // (1/4) H Hᵀ = I, so a full Hadamard probe set makes Hutchinson exact.
    fn hadamard4() -> Array2<f64> {
        Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 1.0, 1.0, 1.0, //
                1.0, -1.0, 1.0, -1.0, //
                1.0, 1.0, -1.0, -1.0, //
                1.0, -1.0, -1.0, 1.0,
            ],
        )
        .unwrap()
    }

    fn spd4() -> Array2<f64> {
        // A = M Mᵀ + diag, symmetric positive definite, known trace.
        let m = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.2, -0.3, 0.1, //
                0.0, 1.1, 0.4, -0.2, //
                0.5, -0.1, 0.9, 0.3, //
                -0.2, 0.3, 0.0, 1.2,
            ],
        )
        .unwrap();
        let mut a = m.dot(&m.t());
        for i in 0..4 {
            a[[i, i]] += 0.5;
        }
        a
    }

    fn trace(a: ArrayView2<'_, f64>) -> f64 {
        (0..a.nrows()).map(|i| a[[i, i]]).sum()
    }

    #[test]
    fn hutchinson_with_full_hadamard_probes_is_exact() {
        let a = spd4();
        let probes = hadamard4();
        let a_probes = a.dot(&probes);
        let est = hutchinson_trace(probes.view(), a_probes.view());
        assert!((est - trace(a.view())).abs() < 1e-10, "{est} vs {}", trace(a.view()));
    }

    #[test]
    fn controlled_trace_with_partial_basis_and_hadamard_probes_is_exact() {
        let a = spd4();
        // Q = first two normalized Hadamard columns (orthonormal).
        let h = hadamard4();
        let mut q = Array2::<f64>::zeros((4, 2));
        for i in 0..4 {
            q[[i, 0]] = h[[i, 0]] / 2.0;
            q[[i, 1]] = h[[i, 1]] / 2.0;
        }
        let a_q = a.dot(&q);
        let probes = hadamard4();
        let a_probes = a.dot(&probes);
        let est = controlled_trace(q.view(), a_q.view(), probes.view(), a_probes.view());
        assert!((est - trace(a.view())).abs() < 1e-10, "{est} vs {}", trace(a.view()));
    }

    #[test]
    fn full_rank_basis_makes_residual_vanish() {
        let a = spd4();
        // Q = full normalized Hadamard basis ⇒ (I − QQᵀ) = 0.
        let h = hadamard4();
        let q = h.mapv(|v| v / 2.0);
        let a_q = a.dot(&q);
        // even with zero probes, exact part already equals the trace.
        let empty = Array2::<f64>::zeros((4, 0));
        let est = controlled_trace(q.view(), a_q.view(), empty.view(), empty.view());
        assert!((est - trace(a.view())).abs() < 1e-10);
    }

    #[test]
    fn controlled_reduces_to_hutchinson_with_empty_basis() {
        let a = spd4();
        let probes = hadamard4();
        let a_probes = a.dot(&probes);
        let empty_q = Array2::<f64>::zeros((4, 0));
        let empty_aq = Array2::<f64>::zeros((4, 0));
        let controlled = controlled_trace(
            empty_q.view(),
            empty_aq.view(),
            probes.view(),
            a_probes.view(),
        );
        let hutch = hutchinson_trace(probes.view(), a_probes.view());
        assert!((controlled - hutch).abs() < 1e-12);
    }
}
