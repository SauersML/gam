//! Matrix-free operator form of the closed-form Duchon penalty.
//!
//! At biobank scale `K` (number of knots) reaches ~2000, so a dense `K×K`
//! penalty matrix per derivative order per axis becomes a meaningful chunk of
//! memory (32 MB at K=2000, f64). For three derivative orders × multiple axes
//! this dominates the closed-form pipeline.
//!
//! This module exposes the same closed-form anisotropic Duchon penalty as
//! `basis::closed_form_anisotropic_pair_block`, but as an *operator* — it
//! evaluates `(S v)[i] = J · Σ_j g_q(t_i − t_j; m, s, κ, η) · v[j]` on the fly
//! without forming the full `K×K` Gram. Constraint composition is handled
//! lazily: the operator stores optional kernel-nullspace `Z` and outer
//! identifiability `T` factors and applies them via `S' v = T^T S T v`.
//!
//! Performance reference (untuned):
//! - K = 200, d = 2: matvec ≈ 4 ms (40K g_q evaluations, ~100 ns each).
//! - K = 2000, d = 2: matvec ≈ 400 ms (4M g_q evaluations).
//!
//! `dense_form` is provided as a fallback for callers that still need a dense
//! `Array2`. Once Task #6 (analytic radial-derivative g_q) lands, matvec speed
//! improves automatically because `anisotropic_duchon_penalty` is the inner
//! kernel.
//!
//! Integration points for follow-up wiring:
//! - `closed_form_operator_penalty_in_total_basis` in `basis.rs` produces a
//!   dense `Array2`; PIRLS/REML penalty handling currently expects dense
//!   `Array2<f64>` inside `PenaltyCandidate { matrix, .. }`. Replacing that
//!   with `enum PenaltyForm { Dense(Array2), Operator(Arc<dyn Operator>) }`
//!   is the wire-in point. The operator log-det path uses
//!   `stochastic_lanczos_logdet_spd_operator` which already accepts a closure.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};

use crate::faer_ndarray::{fast_ab, fast_atb};
use crate::linalg::utils::stochastic_lanczos_logdet_spd_operator;
use crate::terms::basis::{
    closed_form_anisotropic_pair_block, closed_form_penalty::anisotropic_duchon_penalty,
};

/// Matrix-free closed-form anisotropic Duchon penalty operator.
///
/// Stores only the parameters of the closed-form pair-block (`q, m, s, κ, η`,
/// knot centers, and optional constraint factors). Memory footprint is
/// `O(K·d)` plus the constraint factors, vs `O(K²)` for the dense Gram.
#[derive(Clone)]
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
    /// Per-axis centered anisotropy log-scales (length d) and global Jacobian
    /// `J = exp(Σ η_k)`. The pair-block is `J · g_q(r; m, s, κ, η)`.
    eta_centered: Vec<f64>,
    j_prefactor: f64,
    /// Optional kernel-nullspace transform Z (K × kernel_cols).
    kernel_nullspace: Option<Array2<f64>>,
    /// Number of polynomial-block columns padded after the kernel block.
    polynomial_block_cols: usize,
    /// Optional outer spatial identifiability transform T (total_pre × total).
    outer_identifiability: Option<Array2<f64>>,
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
        let j_prefactor = eta_centered.iter().sum::<f64>().exp();
        Self {
            q,
            m,
            s,
            kappa,
            centers: centers.to_owned(),
            eta_centered,
            j_prefactor,
            kernel_nullspace: kernel_nullspace.cloned(),
            polynomial_block_cols,
            outer_identifiability: outer_identifiability.cloned(),
        }
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

        let kernel_cols = self
            .kernel_nullspace
            .as_ref()
            .map(|z| z.ncols())
            .unwrap_or_else(|| self.centers.nrows());
        let total_pre = kernel_cols + self.polynomial_block_cols;

        // Step 1: u = T w (or u = w if no T).
        let u_owned: Array1<f64> = match &self.outer_identifiability {
            Some(t) => {
                // t is (total_pre × dim); u = t · w.
                let mut u = Array1::<f64>::zeros(total_pre);
                for i in 0..total_pre {
                    let mut acc = 0.0;
                    for j in 0..w.len() {
                        acc += t[[i, j]] * w[j];
                    }
                    u[i] = acc;
                }
                u
            }
            None => w.to_owned(),
        };

        // Step 2/3: lift kernel block to K-space via Z.
        let u_kernel = u_owned.slice(ndarray::s![0..kernel_cols]);
        let v_k: Array1<f64> = match &self.kernel_nullspace {
            Some(z) => {
                let k = self.centers.nrows();
                let mut v = Array1::<f64>::zeros(k);
                for i in 0..k {
                    let mut acc = 0.0;
                    for c in 0..kernel_cols {
                        acc += z[[i, c]] * u_kernel[c];
                    }
                    v[i] = acc;
                }
                v
            }
            None => u_kernel.to_owned(),
        };

        // Step 4: y = S_raw v with on-the-fly pair-block evaluation. Symmetry
        // halves the work: g(r) = g(-r). Evaluate the lower triangle and
        // accumulate symmetrically.
        let k = self.centers.nrows();
        let d = self.centers.ncols();
        let mut y = Array1::<f64>::zeros(k);
        let mut r_buf = vec![0.0_f64; d];
        for i in 0..k {
            // Diagonal: r = 0.
            for axis in 0..d {
                r_buf[axis] = 0.0;
            }
            let g_diag = self.j_prefactor
                * anisotropic_duchon_penalty(
                    self.q,
                    self.m,
                    self.s,
                    self.kappa,
                    &self.eta_centered,
                    &r_buf,
                );
            y[i] += g_diag * v_k[i];
            for j in 0..i {
                for axis in 0..d {
                    r_buf[axis] = self.centers[[i, axis]] - self.centers[[j, axis]];
                }
                let g_ij = self.j_prefactor
                    * anisotropic_duchon_penalty(
                        self.q,
                        self.m,
                        self.s,
                        self.kappa,
                        &self.eta_centered,
                        &r_buf,
                    );
                y[i] += g_ij * v_k[j];
                y[j] += g_ij * v_k[i];
            }
        }

        // Step 5: y_kernel = Z^T y.
        let y_kernel: Array1<f64> = match &self.kernel_nullspace {
            Some(z) => {
                let mut yk = Array1::<f64>::zeros(kernel_cols);
                for c in 0..kernel_cols {
                    let mut acc = 0.0;
                    for i in 0..k {
                        acc += z[[i, c]] * y[i];
                    }
                    yk[c] = acc;
                }
                yk
            }
            None => y,
        };

        // Step 6: pad polynomial block with zeros and apply T^T.
        let mut padded = Array1::<f64>::zeros(total_pre);
        padded
            .slice_mut(ndarray::s![0..kernel_cols])
            .assign(&y_kernel);
        match &self.outer_identifiability {
            Some(t) => {
                // out = t^T · padded.
                for j in 0..out.len() {
                    let mut acc = 0.0;
                    for i in 0..total_pre {
                        acc += t[[i, j]] * padded[i];
                    }
                    out[j] = acc;
                }
            }
            None => {
                out.assign(&padded);
            }
        }
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
    /// pipeline.
    pub fn dense_form(&self) -> Array2<f64> {
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
        let mut reg = dense.clone();
        for i in 0..n {
            reg[[i, i]] += lambda;
        }
        // Use SLQ with deterministic seed; for n = K = 5 this falls into the
        // exact dense Cholesky path inside `stochastic_lanczos_logdet_spd_operator`.
        let est = op
            .log_det_plus_lambda_i(lambda, 8, 12, 7)
            .expect("slq logdet");
        // Reference via direct Cholesky.
        use faer::Side;
        use crate::faer_ndarray::FaerCholesky;
        let chol = reg.cholesky(Side::Lower).expect("cholesky");
        let reference = 2.0 * chol.diag().mapv(f64::ln).sum();
        assert_abs_diff_eq!(est, reference, epsilon = 1e-6);
    }
}
