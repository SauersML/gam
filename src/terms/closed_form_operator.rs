//! Operator form of the closed-form Duchon penalty.
//!
//! ## Status
//!
//! `matvec` is analytic and streaming: it applies the constraint transforms,
//! evaluates the raw pair kernel row-by-row with Kahan summation, and never
//! allocates the raw `K×K` Gram. `dense_form()` is still available for
//! callers that explicitly need a materialized matrix, and caches only that
//! opt-in dense build.


use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use rayon::prelude::*;
use smallvec::SmallVec;

use crate::faer_ndarray::{fast_ab, fast_atb};
use crate::terms::basis::{
    closed_form_anisotropic_pair_block, closed_form_anisotropic_pair_value_with_powers,
    closed_form_penalty, pure_duchon_diagonal_epsilon,
};

/// Matrix-free closed-form anisotropic Duchon penalty operator.
///
/// Stores the parameters of the closed-form pair-block (`q, m, s, κ, η`, knot
/// centers, and optional constraint factors). The hot `matvec` path stays
/// matrix-free; `cached_dense` is populated only by `dense_form()`.
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
    /// Per-axis raw anisotropy log-scales (length d). The pair-block builder
    /// consumes these directly and applies the `J = exp(Σ η_k)` Jacobian
    /// internally.
    eta_raw: Vec<f64>,
    /// Cached powers of `B = diag(exp(-2η))` for the analytic pair kernel.
    eta_metric_powers: closed_form_penalty::AnisoMetricPowers,
    /// Optional kernel-nullspace transform Z (K × kernel_cols).
    kernel_nullspace: Option<Array2<f64>>,
    /// Number of polynomial-block columns padded after the kernel block.
    polynomial_block_cols: usize,
    /// Optional outer spatial identifiability transform T (total_pre × total).
    outer_identifiability: Option<Array2<f64>>,
    /// Diagonal epsilon convention for regimes without an exact analytic
    /// self-pair. In the convergent closed-form regimes this is zero and is
    /// never read by pair evaluation.
    diagonal_epsilon: f64,
    /// Lazily-populated dense form. Populated only by `dense_form`; the
    /// matvec/diag/trace/log-det paths stay matrix-free.
    ///
    /// `RayonSafeOnce` (not `OnceLock`) because `build_dense` runs faer
    /// GEMMs (`fast_ab`, `fast_atb`) which dispatch nested rayon work — a
    /// plain `OnceLock` here would deadlock if `dense_form` is first hit
    /// concurrently from inside an outer par_iter. See
    /// `feedback_oncelock_rayon_deadlock`.
    cached_dense: crate::resource::RayonSafeOnce<Array2<f64>>,
}

// Cloning the operator resets its cache so the new instance rebuilds on first
// use. This matches the legacy `derive(Clone)` behavior (which also produced a
// fresh dense build per matvec — the cache is strictly an addition).
impl Clone for ClosedFormPenaltyOperator {
    fn clone(&self) -> Self {
        Self {
            q: self.q,
            m: self.m,
            s: self.s,
            kappa: self.kappa,
            centers: self.centers.clone(),
            eta_raw: self.eta_raw.clone(),
            eta_metric_powers: self.eta_metric_powers.clone(),
            kernel_nullspace: self.kernel_nullspace.clone(),
            polynomial_block_cols: self.polynomial_block_cols,
            outer_identifiability: self.outer_identifiability.clone(),
            diagonal_epsilon: self.diagonal_epsilon,
            cached_dense: crate::resource::RayonSafeOnce::new(),
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
        let eta_raw: Vec<f64> = if let Some(eta) = aniso_log_scales {
            assert_eq!(
                eta.len(),
                d,
                "ClosedFormPenaltyOperator::new: eta dimension mismatch"
            );
            eta.to_vec()
        } else {
            vec![0.0_f64; d]
        };
        let diagonal_epsilon =
            if closed_form_penalty::analytic_self_pair_bundle(q, m, s, kappa, &eta_raw).is_some() {
                0.0
            } else {
                pure_duchon_diagonal_epsilon(centers, &eta_raw)
            };
        Self {
            q,
            m,
            s,
            kappa,
            centers: centers.to_owned(),
            eta_metric_powers: closed_form_penalty::AnisoMetricPowers::new(&eta_raw),
            eta_raw,
            kernel_nullspace: kernel_nullspace.cloned(),
            polynomial_block_cols,
            outer_identifiability: outer_identifiability.cloned(),
            diagonal_epsilon,
            cached_dense: crate::resource::RayonSafeOnce::new(),
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

    #[inline]
    fn is_raw_layout(&self) -> bool {
        self.kernel_nullspace.is_none()
            && self.polynomial_block_cols == 0
            && self.outer_identifiability.is_none()
    }

    fn raw_diagonal_value(&self) -> f64 {
        let mut r0: SmallVec<[f64; 16]> = SmallVec::with_capacity(self.centers.ncols());
        r0.resize(self.centers.ncols(), 0.0);
        closed_form_anisotropic_pair_value_with_powers(
            self.q,
            self.m,
            self.s,
            self.kappa,
            &self.eta_raw,
            &self.eta_metric_powers,
            r0.as_slice(),
            self.diagonal_epsilon,
        )
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

        let pre = match &self.outer_identifiability {
            Some(t) => t.dot(&w),
            None => w.to_owned(),
        };
        let kernel_cols = self
            .kernel_nullspace
            .as_ref()
            .map(|z| z.ncols())
            .unwrap_or_else(|| self.centers.nrows());
        let pre_kernel = pre.slice(ndarray::s![0..kernel_cols]);
        let raw_input = match &self.kernel_nullspace {
            Some(z) => z.dot(&pre_kernel),
            None => pre_kernel.to_owned(),
        };
        let raw_output = self.raw_pair_matvec(raw_input.view());
        let kernel_output = match &self.kernel_nullspace {
            Some(z) => z.t().dot(&raw_output),
            None => raw_output,
        };
        let total_pre = kernel_cols + self.polynomial_block_cols;
        let mut projected = Array1::<f64>::zeros(total_pre);
        projected
            .slice_mut(ndarray::s![0..kernel_cols])
            .assign(&kernel_output);
        let final_output = match &self.outer_identifiability {
            Some(t) => t.t().dot(&projected),
            None => projected,
        };
        out.assign(&final_output);
    }

    /// Diagonal `S[i,i]` for i in 0..dim. In the raw layout this is the
    /// analytic self-pair repeated K times. With constraint composition the
    /// diagonal is *not* the K-space diagonal; we extract it via
    /// `e_i^T S' e_i = matvec(e_i)[i]`.
    pub fn diag(&self) -> Array1<f64> {
        let n = self.dim();
        if self.is_raw_layout() {
            return Array1::from_elem(n, self.raw_diagonal_value());
        }
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

    /// Trace `tr(S')`. In raw layout this is K times the analytic self-pair;
    /// otherwise it uses the composed-basis diagonal.
    pub fn trace(&self) -> f64 {
        if self.is_raw_layout() {
            return self.raw_diagonal_value() * self.dim() as f64;
        }
        self.diag().sum()
    }

    /// Exact `log det(S' + λI)`.
    /// `S'` is rank-deficient under typical constraints (kernel/polynomial
    /// nullspace), so the regularization `λ > 0` is mandatory.
    pub fn log_det_plus_lambda_i(&self, lambda: f64) -> Result<f64, String> {
        assert!(lambda > 0.0, "log_det_plus_lambda_i requires λ > 0");
        let n = self.dim();
        let mut dense = self.dense_form();
        for i in 0..n {
            dense[[i, i]] += lambda;
        }
        let (evals, _) =
            crate::faer_ndarray::FaerEigh::eigh(&dense, faer::Side::Lower).map_err(|e| {
                format!("ClosedFormPenaltyOperator logdet eigendecomposition failed: {e}")
            })?;
        let mut logdet = 0.0;
        for (idx, &ev) in evals.iter().enumerate() {
            if !ev.is_finite() || ev <= 0.0 {
                return Err(format!(
                    "ClosedFormPenaltyOperator expected SPD S+λI, eigenvalue {idx} is {ev:.3e}"
                ));
            }
            logdet += ev.ln();
        }
        Ok(logdet)
    }

    /// Materialize the full constrained operator as a dense `Array2` for
    /// callers that explicitly request a matrix or for validation against
    /// `closed_form_operator_penalty_in_total_basis`. Uses the internal
    /// cache: the first call builds, subsequent calls clone from the cache.
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
            if self.eta_raw.iter().all(|&e| e == 0.0) {
                None
            } else {
                Some(self.eta_raw.as_slice())
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

    fn raw_pair_matvec(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(
            v.len(),
            self.centers.nrows(),
            "ClosedFormPenaltyOperator::raw_pair_matvec: input dim mismatch"
        );
        let k = self.centers.nrows();
        let d = self.centers.ncols();
        let rows: Vec<f64> = (0..k)
            .into_par_iter()
            .map(|i| {
                let mut r: SmallVec<[f64; 16]> = SmallVec::with_capacity(d);
                r.resize(d, 0.0);
                let mut sum = 0.0_f64;
                let mut correction = 0.0_f64;
                for j in 0..k {
                    for axis in 0..d {
                        r[axis] = self.centers[[i, axis]] - self.centers[[j, axis]];
                    }
                    let gij = closed_form_anisotropic_pair_value_with_powers(
                        self.q,
                        self.m,
                        self.s,
                        self.kappa,
                        &self.eta_raw,
                        &self.eta_metric_powers,
                        r.as_slice(),
                        self.diagonal_epsilon,
                    );
                    let y = gij * v[j] - correction;
                    let next = sum + y;
                    correction = (next - sum) - y;
                    sum = next;
                }
                sum
            })
            .collect();
        Array1::from_vec(rows)
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
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
    fn test_operator_matvec_stays_matrix_free_until_dense_requested() {
        let centers = small_centers();
        let op = ClosedFormPenaltyOperator::new(
            centers.view(),
            1,
            2,
            1,
            1.0,
            Some(&[0.35, 0.10]),
            None,
            0,
            None,
        );
        let v = Array1::from_vec(vec![0.2, -0.1, 0.4, -0.3, 0.7]);
        let mut out = Array1::<f64>::zeros(op.dim());
        op.matvec(v.view(), out.view_mut());
        assert!(
            op.cached_dense.get().is_none(),
            "matvec must not populate the dense KxK cache"
        );
        let dense = op.dense_form();
        assert!(
            op.cached_dense.get().is_some(),
            "dense_form should be the only path that populates the dense cache"
        );
        let expected = dense.dot(&v);
        for i in 0..op.dim() {
            assert_abs_diff_eq!(out[i], expected[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_operator_preserves_raw_anisotropy_coordinates() {
        let centers = small_centers();
        let eta = [0.35, 0.10];
        let op =
            ClosedFormPenaltyOperator::new(centers.view(), 1, 2, 1, 1.0, Some(&eta), None, 0, None);
        let dense = op.dense_form();
        let reference = crate::terms::basis::closed_form_operator_penalty_in_total_basis(
            centers.view(),
            1,
            2,
            1,
            1.0,
            Some(&eta),
            None,
            0,
            None,
        );
        for i in 0..op.dim() {
            for j in 0..op.dim() {
                let scale = reference[[i, j]].abs().max(1.0);
                assert_abs_diff_eq!(dense[[i, j]], reference[[i, j]], epsilon = 1e-12 * scale);
            }
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
        let op = ClosedFormPenaltyOperator::new(centers.view(), 1, 2, 1, 1.0, None, None, 0, None);
        let dense = op.dense_form();
        let n = op.dim();
        let lambda = 10.0_f64;
        // Dense reference: exact log det(S + λI) via symmetric
        // eigendecomposition.  λ is intentionally large enough that the
        // regularized matrix is strictly SPD; if it is not, the operator method
        // should error rather than flooring non-positive eigenvalues.
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
        let est = op.log_det_plus_lambda_i(lambda).expect("exact logdet");
        use crate::faer_ndarray::FaerEigh;
        use faer::Side;
        let (evals, _) = FaerEigh::eigh(&reg, Side::Lower).expect("eigh");
        let mut reference = 0.0_f64;
        for (idx, &lam) in evals.iter().enumerate() {
            assert!(lam > 0.0, "reference eigenvalue {idx} is {lam:.3e}");
            reference += lam.ln();
        }
        assert_abs_diff_eq!(est, reference, epsilon = 1e-10);
    }
}
