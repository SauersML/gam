//! Operator form of the closed-form Duchon penalty (currently a lazy
//! materializer).
//!
//! ## Status
//!
//! This module exposes the closed-form anisotropic Duchon penalty as an
//! *operator* with `matvec`, `diag`, `trace`, and `log_det_plus_lambda_i`
//! methods — the same surface a true matrix-free implementation would have.
//! Today, however, every `matvec` ultimately falls back to a fully
//! materialized `K×K` dense Gram (see "Cancellation budget" below). A
//! `OnceLock<Array2<f64>>` cache amortizes that build over the lifetime of
//! the operator, so repeated matvecs cost `O(K²)` flops without rebuild —
//! but the `O(K²)` *memory* is unavoidable in the current implementation.
//!
//! For our K range (200–2000) this is acceptable. At K = 2000 the cache is
//! one 32 MB allocation per operator; a typical run keeps a handful of
//! operators alive (one per derivative order × axis), so total penalty
//! memory stays under ~100 MB.
//!
//! ## Cancellation budget
//!
//! A genuinely matrix-free `matvec` would chain
//!     `S' w = T^T diag(Z, I_poly)^T G_raw diag(Z, I_poly) T w`
//! by applying each factor to a vector. Doing so produces heavy floating-
//! point cancellation under typical `Z` (e.g. `Z` projecting out a near-
//! constant mode of `G_raw` whose row-sums largely cancel against off-
//! diagonal contributions). Vector-chain associativity disagrees with the
//! materialized chain at a magnitude proportional to that cancellation,
//! not at FP roundoff. The materialized path uses `fast_atb`/`fast_ab`
//! which preserve the dense reference; we mirror it by building the same
//! `K×K` matrix and computing `out = M · w`.
//!
//! ## Future work for a true matrix-free path
//!
//! The integration analysis on 2026-04-30 (memory:
//! `matrix_free_penalty_integration_assessment.md`) concluded that wiring
//! the operator surface through `PenaltyCandidate` / `CanonicalPenaltyBlock`
//! / PIRLS / REML is a multi-week refactor that yields **zero** speedup
//! until two things change:
//!   1. `matvec` becomes truly matrix-free (no `K×K` storage), which
//!      requires a numerically stable formulation of the constrained chain
//!      that matches `dense_form` to FP precision under cancellation.
//!   2. The downstream Hessian assembly `H = X^T W X + Σ λ_k S_k` learns
//!      to consume operator-form `S_k` (e.g. PCG-against-implicit-H), since
//!      the destination Hessian is currently dense `p × p` and adding an
//!      operator to a dense destination materializes it column-by-column.
//!
//! Until those land, this module is best treated as a memory-aware lazy
//! materializer: same numbers as the dense path, single `dense_form` build
//! per operator instance.

use std::sync::OnceLock;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};

use crate::faer_ndarray::{fast_ab, fast_atb};
use crate::linalg::utils::stochastic_lanczos_logdet_spd_operator;
use crate::terms::basis::closed_form_anisotropic_pair_block;

/// Matrix-free closed-form anisotropic Duchon penalty operator.
///
/// Stores the parameters of the closed-form pair-block (`q, m, s, κ, η`, knot
/// centers, and optional constraint factors), plus a `OnceLock` cache of the
/// fully assembled dense `dim × dim` operator. The first `matvec` (or any
/// other call that needs the dense form) builds the cache; subsequent calls
/// reuse it. See module docs for the cancellation-budget reason the cache
/// holds a dense matrix instead of the operator parameters alone.
pub struct ClosedFormPenaltyOperator {
    /// Derivative order (0 = mass, 1 = tension, 2 = stiffness).
    q: usize,
    /// Outer kernel order parameter.
    m: usize,
    /// Inner Matérn order parameter.
    s: usize,
    /// Inverse length scale (κ ≥ 0).
    kappa: f64,
    /// Knot centers in the un-anisotropized coordinate, shape (K, d).
    centers: Array2<f64>,
    /// Per-axis centered anisotropy log-scales (length d). The pair-block
    /// builder consumes these directly and applies the `J = exp(Σ η_k)`
    /// Jacobian internally.
    eta_centered: Vec<f64>,
    /// Optional kernel-nullspace transform Z (K × kernel_cols).
    kernel_nullspace: Option<Array2<f64>>,
    /// Number of polynomial-block columns padded after the kernel block.
    polynomial_block_cols: usize,
    /// Optional outer spatial identifiability transform T (total_pre × total).
    outer_identifiability: Option<Array2<f64>>,
    /// Lazily-populated dense form. Populated on first call to `dense_form`,
    /// `matvec`, `diag`, `trace`, or `log_det_plus_lambda_i`.
    cached_dense: OnceLock<Array2<f64>>,
}

// `OnceLock<T>` is not `Clone`; cloning the operator resets its cache so the
// new instance rebuilds on first use. This matches the legacy `derive(Clone)`
// behavior (which also produced a fresh dense build per matvec — the cache is
// strictly an addition).
impl Clone for ClosedFormPenaltyOperator {
    fn clone(&self) -> Self {
        Self {
            q: self.q,
            m: self.m,
            s: self.s,
            kappa: self.kappa,
            centers: self.centers.clone(),
            eta_centered: self.eta_centered.clone(),
            kernel_nullspace: self.kernel_nullspace.clone(),
            polynomial_block_cols: self.polynomial_block_cols,
            outer_identifiability: self.outer_identifiability.clone(),
            cached_dense: OnceLock::new(),
        }
    }
}

impl ClosedFormPenaltyOperator {
    /// Build an operator with the same closed-form parameters that
    /// `basis::closed_form_operator_penalty_in_total_basis` consumes.
    pub fn new(
        centers: ArrayView2<'_, f64>,
        q: usize,
        m: usize,
        s: usize,
        kappa: f64,
        aniso_log_scales: Option<&[f64]>,
        kernel_nullspace: Option<&Array2<f64>>,
        polynomial_block_cols: usize,
        outer_identifiability: Option<&Array2<f64>>,
    ) -> Self {
        let d = centers.ncols();
        let eta_centered: Vec<f64> = if let Some(eta) = aniso_log_scales {
            let mean = if eta.len() <= 1 {
                0.0
            } else {
                eta.iter().sum::<f64>() / eta.len() as f64
            };
            eta.iter().map(|&v| (v - mean).clamp(-50.0, 50.0)).collect()
        } else {
            vec![0.0_f64; d]
        };
        Self {
            q,
            m,
            s,
            kappa,
            centers: centers.to_owned(),
            eta_centered,
            kernel_nullspace: kernel_nullspace.cloned(),
            polynomial_block_cols,
            outer_identifiability: outer_identifiability.cloned(),
            cached_dense: OnceLock::new(),
        }
    }

    /// Return the cached dense form, building it on first call.
    fn ensure_dense(&self) -> &Array2<f64> {
        self.cached_dense.get_or_init(|| self.build_dense())
    }

    /// Number of raw kernel rows (K).
    pub fn raw_dim(&self) -> usize {
        self.centers.nrows()
    }

    /// Number of columns *after* applying constraint composition: the
    /// dimension that callers see when invoking matvec/dense_form.
    pub fn dim(&self) -> usize {
        let kernel_cols = self
            .kernel_nullspace
            .as_ref()
            .map(|z| z.ncols())
            .unwrap_or_else(|| self.centers.nrows());
        let total_pre = kernel_cols + self.polynomial_block_cols;
        match &self.outer_identifiability {
            Some(t) => t.ncols(),
            None => total_pre,
        }
    }

    /// Evaluate `(S w)` writing the result into `out`.
    ///
    /// With constraints composed, `S' = T^T diag(Z, I_poly)^T S_raw diag(Z, I_poly) T`.
    /// We apply the chain right-to-left:
    ///   1. `u = T w`            (dim → total_pre)
    ///   2. `u_kernel = u[..kernel_cols]; u_poly = u[kernel_cols..]`
    ///   3. `v = Z u_kernel`     (kernel_cols → K)
    ///   4. `y = S_raw v`        (K → K), via on-the-fly pair-block evaluation
    ///   5. `y_kernel = Z^T y`   (K → kernel_cols)
    ///   6. compose with zero polynomial block, then `out = T^T [y_kernel; 0]`
    pub fn matvec(&self, w: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        assert_eq!(
            w.len(),
            self.dim(),
            "ClosedFormPenaltyOperator::matvec: input dim mismatch"
        );
        assert_eq!(
            out.len(),
            self.dim(),
            "ClosedFormPenaltyOperator::matvec: output dim mismatch"
        );

        // The constraint chain `T^T diag(Z, I_poly)^T G_raw diag(Z, I_poly) T`
        // produces heavy floating-point cancellation under typical Z (e.g. Z
        // projecting out a near-constant mode of G_raw whose row-sums largely
        // cancel against off-diagonal contributions). Any matvec scheme that
        // associates the chain differently from `dense_form` — symmetry-
        // halved accumulation, `Z (Gv)` vs `(Z^T G Z) v`, BLAS-vs-manual
        // reductions — disagrees with the dense path at a magnitude
        // proportional to the cancellation budget, not at FP roundoff.
        //
        // To stay bit-compatible with `dense_form` we use the same
        // materialized constrained operator and compute `out = M w` against
        // it. The `OnceLock` cache amortizes the `K×K` build over the
        // lifetime of the operator; the first call pays the build cost,
        // subsequent calls are just a dense gemv. See module docs for why
        // a vector-chain matvec disagrees with `dense_form` numerically.
        let m = self.ensure_dense();
        out.assign(&m.dot(&w));
    }

    /// Diagonal `S[i,i]` for i in 0..dim. With constraint composition the
    /// diagonal is *not* the K-space diagonal; we extract it via
    /// `e_i^T S' e_i = matvec(e_i)[i]`. That is `O(dim)` matvecs and is meant
    /// for diagnostics (e.g., Jacobi preconditioning), not hot paths.
    pub fn diag(&self) -> Array1<f64> {
        let n = self.dim();
        let mut out = Array1::<f64>::zeros(n);
        let mut e = Array1::<f64>::zeros(n);
        let mut col = Array1::<f64>::zeros(n);
        for i in 0..n {
            e.fill(0.0);
            e[i] = 1.0;
            self.matvec(e.view(), col.view_mut());
            out[i] = col[i];
        }
        out
    }

    /// Trace `tr(S')`. Same `O(dim)` matvecs as `diag`; stored separately so
    /// callers that only need the trace don't allocate the full diagonal.
    pub fn trace(&self) -> f64 {
        self.diag().sum()
    }

    /// Stochastic Lanczos quadrature estimate of `log det(S' + λI)`.
    /// `S'` is rank-deficient under typical constraints (kernel/polynomial
    /// nullspace), so the regularization `λ > 0` is mandatory for SLQ.
    pub fn log_det_plus_lambda_i(
        &self,
        lambda: f64,
        num_probes: usize,
        lanczos_steps: usize,
        seed: u64,
    ) -> Result<f64, String> {
        assert!(lambda > 0.0, "log_det_plus_lambda_i requires λ > 0");
        let n = self.dim();
        let me = self.clone();
        stochastic_lanczos_logdet_spd_operator(
            n,
            move |v: &Array1<f64>| {
                let mut out = Array1::<f64>::zeros(n);
                me.matvec(v.view(), out.view_mut());
                out.zip_mut_with(v, |o, &vi| *o += lambda * vi);
                out
            },
            num_probes,
            lanczos_steps,
            seed,
        )
    }

    /// Materialize the full constrained operator as a dense `Array2`. Useful
    /// as a fallback for callers that still expect dense penalties or for
    /// validation against the existing `closed_form_operator_penalty_in_total_basis`
    /// pipeline. Uses the internal cache: the first call builds, subsequent
    /// calls clone from the cache.
    pub fn dense_form(&self) -> Array2<f64> {
        self.ensure_dense().clone()
    }

    /// Internal builder. Do not call directly — go through `ensure_dense` so
    /// the result is cached.
    fn build_dense(&self) -> Array2<f64> {
        // Build raw K×K kernel block via the existing dense path so we share
        // its cancellation-detector logic for small κ.
        let g_raw = closed_form_anisotropic_pair_block(
            self.centers.view(),
            self.q,
            self.m,
            self.s,
            self.kappa,
            if self.eta_centered.iter().all(|&e| e == 0.0) {
                None
            } else {
                Some(self.eta_centered.as_slice())
            },
        );
        let kernel_cols = self
            .kernel_nullspace
            .as_ref()
            .map(|z| z.ncols())
            .unwrap_or_else(|| self.centers.nrows());
        let g_kernel = match &self.kernel_nullspace {
            Some(z) => {
                let zt_g = fast_atb(z, &g_raw);
                fast_ab(&zt_g, z)
            }
            None => g_raw,
        };
        let total_pre = kernel_cols + self.polynomial_block_cols;
        let g_padded = if self.polynomial_block_cols == 0 {
            g_kernel
        } else {
            let mut padded = Array2::<f64>::zeros((total_pre, total_pre));
            padded
                .slice_mut(ndarray::s![0..kernel_cols, 0..kernel_cols])
                .assign(&g_kernel);
            padded
        };
        match &self.outer_identifiability {
            Some(t) => {
                let tt_g = fast_atb(t, &g_padded);
                fast_ab(&tt_g, t)
            }
            None => g_padded,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    fn small_centers() -> Array2<f64> {
        Array::from_shape_vec(
            (5, 2),
            vec![
                0.10, 0.20, //
                0.40, 0.15, //
                0.55, 0.65, //
                0.80, 0.30, //
                0.25, 0.85, //
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_operator_dense_agrees_unconstrained() {
        let centers = small_centers();
        let op = ClosedFormPenaltyOperator::new(
            centers.view(),
            1, // tension
            2,
            1,
            1.0,
            None,
            None,
            0,
            None,
        );
        let dense = op.dense_form();
        let n = op.dim();
        let mut e = Array1::<f64>::zeros(n);
        let mut col = Array1::<f64>::zeros(n);
        for i in 0..n {
            e.fill(0.0);
            e[i] = 1.0;
            op.matvec(e.view(), col.view_mut());
            for j in 0..n {
                let scale = dense[[j, i]].abs().max(1.0);
                assert_abs_diff_eq!(col[j], dense[[j, i]], epsilon = 1e-9 * scale);
            }
        }
    }

    #[test]
    fn test_operator_diag_agrees() {
        let centers = small_centers();
        let op = ClosedFormPenaltyOperator::new(
            centers.view(),
            2, // stiffness
            2,
            1,
            0.5,
            Some(&[0.10, -0.10]),
            None,
            0,
            None,
        );
        let dense = op.dense_form();
        let diag_op = op.diag();
        for i in 0..op.dim() {
            assert_abs_diff_eq!(diag_op[i], dense[[i, i]], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_operator_matvec_random_vector() {
        let centers = small_centers();
        let op = ClosedFormPenaltyOperator::new(
            centers.view(),
            0, // mass
            2,
            1,
            1.5,
            None,
            None,
            0,
            None,
        );
        let dense = op.dense_form();
        let n = op.dim();
        // Pseudo-random vector via deterministic LCG so the test is reproducible
        // and free of an rng dependency.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut v = Array1::<f64>::zeros(n);
        for vi in v.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *vi = ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
        }
        let mut got = Array1::<f64>::zeros(n);
        op.matvec(v.view(), got.view_mut());
        let want = dense.dot(&v);
        for i in 0..n {
            assert_abs_diff_eq!(got[i], want[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_operator_with_kernel_nullspace_constraint() {
        let centers = small_centers();
        let k = centers.nrows();
        // Synthetic Z: project out the constant direction in K-space.
        let mut z = Array2::<f64>::zeros((k, k - 1));
        let inv_sqrt_k = 1.0 / (k as f64).sqrt();
        let constant: Vec<f64> = (0..k).map(|_| inv_sqrt_k).collect();
        // Gram-Schmidt against the constant direction starting from canonical e_1..e_{k-1}.
        for c in 0..(k - 1) {
            let mut col = vec![0.0; k];
            col[c + 1] = 1.0;
            let inner: f64 = col.iter().zip(constant.iter()).map(|(a, b)| a * b).sum();
            for i in 0..k {
                col[i] -= inner * constant[i];
            }
            let norm = col.iter().map(|v| v * v).sum::<f64>().sqrt();
            for i in 0..k {
                z[[i, c]] = col[i] / norm;
            }
        }

        let op = ClosedFormPenaltyOperator::new(
            centers.view(),
            1,
            2,
            1,
            1.0,
            Some(&[0.05, -0.05]),
            Some(&z),
            0,
            None,
        );
        let dense = op.dense_form();
        let n = op.dim();
        assert_eq!(n, k - 1);
        let mut e = Array1::<f64>::zeros(n);
        let mut col = Array1::<f64>::zeros(n);
        for i in 0..n {
            e.fill(0.0);
            e[i] = 1.0;
            op.matvec(e.view(), col.view_mut());
            for j in 0..n {
                let scale = dense[[j, i]].abs().max(1.0);
                assert_abs_diff_eq!(col[j], dense[[j, i]], epsilon = 1e-9 * scale);
            }
        }
    }

    #[test]
    fn test_log_det_plus_lambda_matches_dense() {
        let centers = small_centers();
        let op = ClosedFormPenaltyOperator::new(
            centers.view(),
            1,
            2,
            1,
            1.0,
            None,
            None,
            0,
            None,
        );
        let dense = op.dense_form();
        let n = op.dim();
        let lambda = 0.25_f64;
        // Dense reference: log det(S + λI) via symmetric eigendecomposition.
        // The closed-form tension Gram is not in general PSD before
        // constraint composition (it is a difference of inner products in the
        // Schoenberg expansion), so `S + λI` can have small negative
        // eigenvalues for moderate λ at this K and Cholesky on the dense
        // reference fails. SLQ already falls back to a floored
        // eigendecomposition for tiny systems, so we match that contract on
        // the reference path: compute log det via eigenvalues of the
        // symmetrized operator with the same positive-eigenvalue floor SLQ
        // applies (`max|λ| · 1e-14`).
        let mut reg = dense.clone();
        for i in 0..n {
            reg[[i, i]] += lambda;
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (reg[[i, j]] + reg[[j, i]]);
                reg[[i, j]] = avg;
                reg[[j, i]] = avg;
            }
        }
        let est = op
            .log_det_plus_lambda_i(lambda, 8, 12, 7)
            .expect("slq logdet");
        use faer::Side;
        use crate::faer_ndarray::FaerEigh;
        let (evals, _) = FaerEigh::eigh(&reg, Side::Lower).expect("eigh");
        let max_abs = evals
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
            .max(1.0);
        let floor = max_abs * 1e-14;
        let mut reference = 0.0_f64;
        for &lam in evals.iter() {
            let clipped = if lam.is_finite() && lam > floor {
                lam
            } else {
                floor
            };
            reference += clipped.ln();
        }
        assert_abs_diff_eq!(est, reference, epsilon = 1e-6);
    }
}
