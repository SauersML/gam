use super::*;

/// Configuration for stochastic trace estimation.
#[derive(Clone, Debug)]
pub struct StochasticTraceConfig {
    /// Minimum number of probe vectors (default: 10).
    pub n_probes_min: usize,
    /// Maximum number of probe vectors (default: 200).
    pub n_probes_max: usize,
    /// Target relative accuracy ε for the adaptive stopping criterion (default: 0.01).
    pub relative_tol: f64,
    /// Protection threshold τ_rel for near-zero traces (default: 1e-8).
    pub tau_rel: f64,
    /// Relative tolerance for iterative solves inside stochastic trace probes.
    pub solve_rel_tol: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Hutch++ low-rank sketch dimension. `None` = plain Hutchinson.
    /// `Some(m_s)` runs the Meyer–Musco Hutch++ split: m_s sketch matvecs
    /// build an orthonormal range basis Q via randomized range finder, the
    /// projected trace tr(QᵀM Q) is computed exactly (m_s additional
    /// matvecs), and the residual tr((I-QQᵀ)M(I-QQᵀ)) is estimated by
    /// Hutchinson with the remaining probe budget. Achieves O(1/ε)
    /// matvecs for ε relative error vs O(1/ε²) for plain Hutchinson;
    /// the gain is largest when M has rapidly decaying singular values.
    pub hutchpp_sketch_dim: Option<usize>,
}

impl Default for StochasticTraceConfig {
    fn default() -> Self {
        Self {
            n_probes_min: 10,
            n_probes_max: 200,
            relative_tol: 0.01,
            tau_rel: 1e-8,
            solve_rel_tol: 1e-8,
            seed: 0xCAFE_BABE,
            hutchpp_sketch_dim: None,
        }
    }
}

impl StochasticTraceConfig {
    /// Fast, scale-aware estimator for second-order outer-Hessian traces.
    ///
    /// These traces shape the ARC/Newton model; they are not the REML
    /// objective itself. The default 200-probe estimator is too strict for
    /// high-dimensional marginal-slope jobs because near-zero off-diagonal
    /// cross traces never satisfy a pure relative-error test. A bounded probe
    /// budget with a scale-relative zero floor preserves the large curvature
    /// entries and lets ARC's trust-region logic absorb residual noise.
    pub(crate) fn outer_hessian(dim: usize, n_coords: usize) -> Self {
        let large_problem = dim >= 512 || n_coords >= 4;
        Self {
            n_probes_min: if large_problem { 4 } else { 6 },
            n_probes_max: if large_problem { 8 } else { 24 },
            relative_tol: if large_problem { 0.12 } else { 0.05 },
            tau_rel: 1e-3,
            solve_rel_tol: if large_problem { 1e-4 } else { 1e-5 },
            seed: 0xC0A5_7ACE,
            hutchpp_sketch_dim: None,
        }
    }
}

/// Stochastic trace estimator using Rademacher probes with adaptive stopping.
///
/// Estimates `tr(H⁻¹ A_k)` for multiple matrices `A_k` simultaneously,
/// sharing a single `H⁻¹` solve per probe across all coordinates.
///
/// # Adaptive stopping
///
/// After each probe (once `n_probes_min` is reached), the estimator checks:
///
/// ```text
/// max_k  s_{M,k} / (√M · max(|q̄_{M,k}|, τ_rel))  ≤  ε
/// ```
///
/// where `s_{M,k}` is the sample standard deviation of the per-probe
/// estimates for coordinate k, and `q̄_{M,k}` is the running mean.
///
/// # Bias from approximate solves
///
/// If `H⁻¹` is computed approximately (e.g., via PCG with tolerance δ_PCG),
/// the bias satisfies `|bias| ≤ (δ_PCG · p / λ_min(H)) · ‖Ḣ_k‖₂`.
/// Set δ_PCG small enough that this is below the Monte Carlo tolerance.
pub struct StochasticTraceEstimator {
    pub(crate) config: StochasticTraceConfig,
    pub(crate) trace_state: Arc<Mutex<StochasticTraceState>>,
}

pub(crate) enum StochasticTraceTargets<'a> {
    Dense(&'a [&'a Array2<f64>]),
    Mixed {
        dense_matrices: &'a [&'a Array2<f64>],
        operators: &'a [&'a dyn HyperOperator],
    },
    Structural {
        dense_matrices: &'a [&'a Array2<f64>],
        implicit_ops: &'a [&'a ImplicitHyperOperator],
    },
}

impl StochasticTraceTargets<'_> {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Dense(matrices) => matrices.len(),
            Self::Mixed {
                dense_matrices,
                operators,
            } => dense_matrices.len() + operators.len(),
            Self::Structural {
                dense_matrices,
                implicit_ops,
            } => dense_matrices.len() + implicit_ops.len(),
        }
    }
}

impl StochasticTraceEstimator {
    /// Create a new estimator with the given configuration.
    pub fn new(config: StochasticTraceConfig) -> Self {
        Self {
            config,
            trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        }
    }

    /// Create a new estimator sharing fit-level stochastic trace state.
    pub(crate) fn with_shared_trace_state(
        mut config: StochasticTraceConfig,
        trace_state: Arc<Mutex<StochasticTraceState>>,
    ) -> Self {
        let override_tol = match trace_state.lock() {
            Ok(guard) => guard.solve_rel_tol_override,
            Err(poisoned) => poisoned.into_inner().solve_rel_tol_override,
        };
        if let Some(rel_tol) = override_tol.filter(|v| v.is_finite() && *v > 0.0) {
            config.solve_rel_tol = rel_tol;
        }
        Self {
            config,
            trace_state,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(StochasticTraceConfig::default())
    }

    pub(crate) fn for_outer_hessian(dim: usize, n_coords: usize) -> Self {
        Self::new(StochasticTraceConfig::outer_hessian(dim, n_coords))
    }

    pub(crate) fn for_outer_hessian_with_trace_state(
        dim: usize,
        n_coords: usize,
        trace_state: Arc<Mutex<StochasticTraceState>>,
    ) -> Self {
        Self::with_shared_trace_state(
            StochasticTraceConfig::outer_hessian(dim, n_coords),
            trace_state,
        )
    }

    pub(crate) fn effective_probe_min(&self) -> usize {
        let floor = match self.trace_state.lock() {
            Ok(guard) => guard.monotone_probe_floor,
            Err(poisoned) => poisoned.into_inner().monotone_probe_floor,
        };
        self.config
            .n_probes_min
            .max(floor)
            .min(self.config.n_probes_max)
    }

    pub(crate) fn raise_probe_floor(&self, k_drawn: usize) {
        let mut state = match self.trace_state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if k_drawn > state.monotone_probe_floor {
            let old = state.monotone_probe_floor;
            state.monotone_probe_floor = k_drawn;
            log::info!("[CRN-PIN] probe_floor raised {old}->{k_drawn} (k_drawn={k_drawn})");
        }
    }

    pub(crate) fn estimate_from_probe_batch<F>(
        &self,
        hop: &dyn HessianFactorization,
        n_coords: usize,
        mut evaluate_probe: F,
    ) -> Vec<f64>
    where
        F: FnMut(&Array1<f64>, &Array1<f64>, &mut [f64]),
    {
        if n_coords == 0 {
            return Vec::new();
        }

        let p = hop.dim();
        if p == 0 {
            return vec![0.0; n_coords];
        }

        let mut means = vec![0.0_f64; n_coords];
        let mut m2s = vec![0.0_f64; n_coords];
        let mut probe_values = vec![0.0_f64; n_coords];
        let mut rng_state = Xoshiro256SS::from_seed(self.config.seed);
        let check_interval = 4;
        let effective_n_probes_min = self.effective_probe_min();

        let mut z = Array1::<f64>::zeros(p);
        let mut n_drawn = 0usize;
        for m in 0..self.config.n_probes_max {
            rademacher_probe_into(z.view_mut(), &mut rng_state);
            let probe_id = stochastic_trace_probe_id(self.config.seed, m);
            let w = hop.stochastic_trace_solve_for_probe(
                &z,
                self.config.solve_rel_tol,
                probe_id,
                Some(&self.trace_state),
            );
            evaluate_probe(&z, &w, &mut probe_values);

            for k in 0..n_coords {
                let q_k = probe_values[k];
                let count = (m + 1) as f64;
                let delta = q_k - means[k];
                means[k] += delta / count;
                let delta2 = q_k - means[k];
                m2s[k] += delta * delta2;
            }

            let n_done = m + 1;
            n_drawn = n_done;
            if n_done >= effective_n_probes_min
                && n_done % check_interval == 0
                && self.check_convergence(n_done, &means, &m2s)
            {
                break;
            }
        }

        self.record_probe_batch(Self::max_probe_variance(&m2s, n_drawn), n_drawn);
        self.raise_probe_floor(n_drawn);
        means
    }

    pub(crate) fn estimate_matrix_from_probe_batch<F>(
        &self,
        hop: &dyn HessianFactorization,
        n_coords: usize,
        mut evaluate_probe: F,
    ) -> Array2<f64>
    where
        F: FnMut(u64, &Array1<f64>, &mut Array2<f64>),
    {
        if n_coords == 0 {
            return Array2::zeros((0, 0));
        }
        let p = hop.dim();
        if p == 0 {
            return Array2::zeros((n_coords, n_coords));
        }

        let mut means = Array2::<f64>::zeros((n_coords, n_coords));
        let mut m2s = Array2::<f64>::zeros((n_coords, n_coords));
        let mut probe_values = Array2::<f64>::zeros((n_coords, n_coords));
        let mut rng_state = Xoshiro256SS::from_seed(self.config.seed);
        let check_interval = 4;
        let effective_n_probes_min = self.effective_probe_min();
        let mut z = Array1::<f64>::zeros(p);
        let mut n_drawn = 0usize;

        for m in 0..self.config.n_probes_max {
            rademacher_probe_into(z.view_mut(), &mut rng_state);
            let probe_id = stochastic_trace_probe_id(self.config.seed, m);
            probe_values.fill(0.0);
            evaluate_probe(probe_id, &z, &mut probe_values);

            let count = (m + 1) as f64;
            for d in 0..n_coords {
                for e in 0..n_coords {
                    let q = probe_values[[d, e]];
                    let delta = q - means[[d, e]];
                    means[[d, e]] += delta / count;
                    let delta2 = q - means[[d, e]];
                    m2s[[d, e]] += delta * delta2;
                }
            }

            let n_done = m + 1;
            n_drawn = n_done;
            if n_done >= effective_n_probes_min
                && n_done % check_interval == 0
                && self.check_matrix_convergence(n_done, &means, &m2s)
            {
                break;
            }
        }

        self.record_probe_batch(
            Self::max_probe_variance(m2s.as_slice().unwrap(), n_drawn),
            n_drawn,
        );
        self.raise_probe_floor(n_drawn);
        for d in 0..n_coords {
            for e in (d + 1)..n_coords {
                let avg = 0.5 * (means[[d, e]] + means[[e, d]]);
                means[[d, e]] = avg;
                means[[e, d]] = avg;
            }
        }
        means
    }

    pub(crate) fn max_probe_variance(m2s: &[f64], n_drawn: usize) -> f64 {
        if n_drawn <= 1 {
            return 0.0;
        }
        let denom = (n_drawn - 1) as f64;
        m2s.iter()
            .map(|m2| (*m2 / denom).max(0.0))
            .fold(0.0_f64, f64::max)
    }

    pub(crate) fn record_probe_batch(&self, sigma_sq: f64, n_drawn: usize) {
        let mut state = match self.trace_state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.last_probe_sigma_sq = Some(state.last_probe_sigma_sq.unwrap_or(0.0).max(sigma_sq));
        state.last_probe_count = state.last_probe_count.max(n_drawn);
    }

    pub(crate) fn estimate_hinv_traces(
        &self,
        hop: &dyn HessianFactorization,
        targets: StochasticTraceTargets<'_>,
    ) -> Vec<f64> {
        let n_coords = targets.len();
        if n_coords == 0 {
            return Vec::new();
        }

        match targets {
            StochasticTraceTargets::Dense(matrices) => {
                let mut a_w = Array1::<f64>::zeros(hop.dim());
                self.estimate_from_probe_batch(hop, n_coords, |z, w, probe_values| {
                    for k in 0..matrices.len() {
                        dense::matvec_into(matrices[k], w.view(), a_w.view_mut());
                        probe_values[k] = z.dot(&a_w);
                    }
                })
            }
            StochasticTraceTargets::Mixed {
                dense_matrices,
                operators,
            } => {
                let mut a_w = Array1::<f64>::zeros(hop.dim());
                self.estimate_from_probe_batch(hop, n_coords, |z, w, probe_values| {
                    for k in 0..dense_matrices.len() {
                        dense::matvec_into(dense_matrices[k], w.view(), a_w.view_mut());
                        probe_values[k] = z.dot(&a_w);
                    }

                    let dense_count = dense_matrices.len();
                    for (oi, op) in operators.iter().enumerate() {
                        let k = dense_count + oi;
                        if op.has_fast_bilinear_view() {
                            probe_values[k] = op.bilinear_view(w.view(), z.view());
                        } else {
                            op.mul_vec_into(w.view(), a_w.view_mut());
                            probe_values[k] = z.dot(&a_w);
                        }
                    }
                })
            }
            StochasticTraceTargets::Structural {
                dense_matrices,
                implicit_ops,
            } => {
                if implicit_ops.is_empty() {
                    let no_ops: [&dyn HyperOperator; 0] = [];
                    return self.estimate_hinv_traces(
                        hop,
                        StochasticTraceTargets::Mixed {
                            dense_matrices,
                            operators: &no_ops,
                        },
                    );
                }

                let x_design = implicit_ops[0].x_design.clone();
                let mut x_vec = Array1::<f64>::zeros(x_design.nrows());
                let mut y_vec = Array1::<f64>::zeros(x_design.nrows());
                let mut a_w = Array1::<f64>::zeros(hop.dim());
                self.estimate_from_probe_batch(hop, n_coords, |z, w, probe_values| {
                    x_design.apply_view_into(z.view(), x_vec.view_mut());
                    x_design.apply_view_into(w.view(), y_vec.view_mut());

                    for k in 0..dense_matrices.len() {
                        dense::matvec_into(dense_matrices[k], w.view(), a_w.view_mut());
                        probe_values[k] = z.dot(&a_w);
                    }

                    let dense_count = dense_matrices.len();
                    for (oi, op) in implicit_ops.iter().enumerate() {
                        let k = dense_count + oi;
                        probe_values[k] = op.bilinear_with_shared_x(&x_vec, &y_vec, z, w);
                    }
                })
            }
        }
    }

    /// Estimate a single trace `tr(H⁻¹ A)` using the same batched Hutchinson
    /// core as the multi-coordinate path.
    pub fn estimate_single_trace(
        &self,
        hop: &dyn HessianFactorization,
        matrix: &Array2<f64>,
    ) -> f64 {
        let matrices = [matrix];
        self.estimate_hinv_traces(hop, StochasticTraceTargets::Dense(&matrices))[0]
    }

    /// Estimate `tr(H⁻¹ A_k)` for multiple matrices `A_k` simultaneously.
    ///
    /// Uses Rademacher probes and adaptive stopping. Each probe requires
    /// exactly ONE `H⁻¹` solve (shared across all k), plus one `A_k`
    /// matrix–vector product per coordinate k.
    ///
    /// # Arguments
    /// - `hop`: the Hessian operator providing `solve(rhs)`.
    /// - `matrices`: the `A_k` matrices for which to estimate `tr(H⁻¹ A_k)`.
    ///
    /// # Returns
    /// A vector of estimated traces, one per input matrix.
    pub fn estimate_traces(
        &self,
        hop: &dyn HessianFactorization,
        matrices: &[&Array2<f64>],
    ) -> Vec<f64> {
        self.estimate_hinv_traces(hop, StochasticTraceTargets::Dense(matrices))
    }

    /// Estimate `tr(H⁻¹ A_k)` for a mix of dense matrices and implicit operators.
    ///
    /// This extends [`estimate_traces`] to support implicit `HyperOperator` trait
    /// objects alongside dense matrices. The dense matrices are passed first,
    /// followed by the operators. Each probe requires ONE `H⁻¹` solve (shared),
    /// plus one matvec per coordinate.
    ///
    /// # Arguments
    /// - `hop`: the Hessian operator providing `solve(rhs)`.
    /// - `dense_matrices`: dense `A_k` matrices for which to estimate `tr(H⁻¹ A_k)`.
    /// - `operators`: implicit `HyperOperator` trait objects.
    ///
    /// # Returns
    /// A vector of estimated traces: first for dense matrices, then for operators.
    pub fn estimate_traces_with_operators(
        &self,
        hop: &dyn HessianFactorization,
        dense_matrices: &[&Array2<f64>],
        operators: &[&dyn HyperOperator],
    ) -> Vec<f64> {
        self.estimate_hinv_traces(
            hop,
            StochasticTraceTargets::Mixed {
                dense_matrices,
                operators,
            },
        )
    }

    /// Estimate first-order traces `tr(H⁻¹ A_d)` for implicit operators using the
    /// weighted-Gram structure, sharing one H⁻¹ solve and two X multiplies per probe.
    ///
    /// For each implicit operator d, the bilinear form `u^T A_d z` is computed using
    /// shared `x_vec = X z` and `y_vec = X u`, plus per-axis `forward_mul` calls.
    /// This avoids the X^T multiply per axis that the standard `mul_vec` requires.
    ///
    /// Dense matrices are handled alongside implicit operators in a single pass.
    ///
    /// # Arguments
    /// - `hop`: the Hessian operator providing `solve(rhs)`.
    /// - `dense_matrices`: dense A_k matrices.
    /// - `implicit_ops`: implicit `ImplicitHyperOperator` trait objects.
    ///
    /// # Returns
    /// Estimated traces: first for dense matrices, then for implicit operators.
    pub fn estimate_traces_structural(
        &self,
        hop: &dyn HessianFactorization,
        dense_matrices: &[&Array2<f64>],
        implicit_ops: &[&ImplicitHyperOperator],
    ) -> Vec<f64> {
        self.estimate_hinv_traces(
            hop,
            StochasticTraceTargets::Structural {
                dense_matrices,
                implicit_ops,
            },
        )
    }

    /// Estimate the full D×D matrix of second-order traces `tr(H⁻¹ A_d H⁻¹ A_e)`
    /// for implicit operators, using the CORRECT estimator.
    ///
    /// The correct Girard-Hutchinson estimator for `tr(H⁻¹ A_d H⁻¹ A_e)` is:
    ///
    /// ```text
    /// u = H⁻¹ z
    /// q_e = A_e z        for each axis e
    /// r_e = H⁻¹ q_e      for each axis e  (block solve, D RHS)
    /// estimate = u^T A_d r_e
    /// ```
    ///
    /// This gives tr(H⁻¹ A_d H⁻¹ A_e) correctly, NOT tr(A_d H⁻² A_e).
    ///
    /// Dense matrices are included alongside implicit operators. The output
    /// is a (total × total) matrix of cross-traces, symmetrized.
    ///
    /// # Arguments
    /// - `hop`: the Hessian operator providing `solve` and `solve_multi`.
    /// - `dense_matrices`: dense A_k matrices.
    /// - `implicit_ops`: implicit `ImplicitHyperOperator` trait objects.
    ///
    /// # Returns
    /// Estimated D×D matrix of `tr(H⁻¹ A_d H⁻¹ A_e)` values, symmetrized.
    pub fn estimate_second_order_traces(
        &self,
        hop: &dyn HessianFactorization,
        dense_matrices: &[&Array2<f64>],
        implicit_ops: &[&ImplicitHyperOperator],
    ) -> Array2<f64> {
        let n_dense = dense_matrices.len();
        let n_ops = implicit_ops.len();
        let total = n_dense + n_ops;
        if total == 0 {
            return Array2::zeros((0, 0));
        }

        let p = hop.dim();
        if p == 0 {
            return Array2::zeros((total, total));
        }

        if total == 1 {
            let value = if n_dense == 1 {
                self.estimate_second_order_single_dense(hop, dense_matrices[0])
            } else {
                self.estimate_second_order_single_implicit(hop, implicit_ops[0])
            };
            return Array2::from_elem((1, 1), value);
        }

        // Get the shared X reference from the first implicit operator.
        let x_design = if n_ops > 0 {
            Some(implicit_ops[0].x_design.clone())
        } else {
            None
        };

        let mut q_columns = Array2::zeros((p, total));
        let mut dense_a_u: Vec<Array1<f64>> = (0..n_dense).map(|_| Array1::zeros(p)).collect();
        let n_obs = implicit_ops.first().map(|op| op.w_diag.len()).unwrap_or(0);
        let mut x_vec = Array1::<f64>::zeros(n_obs);
        let mut y_vec = Array1::<f64>::zeros(n_obs);
        let mut x_r: Vec<Array1<f64>> = (0..total).map(|_| Array1::zeros(n_obs)).collect();

        struct ImplicitSecondOrderScratch {
            pub(crate) w_dx_u: Array1<f64>,
            pub(crate) w_y: Array1<f64>,
            pub(crate) u_s: Array1<f64>,
        }

        self.estimate_matrix_from_probe_batch(hop, total, |probe_id, z, probe_values| {
            // Step 1: u = H⁻¹ z (shared solve)
            let u = hop.stochastic_trace_solve_for_probe(
                z,
                self.config.solve_rel_tol,
                probe_id,
                Some(&self.trace_state),
            );

            if let Some(ref x) = x_design {
                x.apply_view_into(z.view(), x_vec.view_mut());
            }

            // Step 2: Form q_e = A_e z for all axes e. Each operator column is
            // independent, so fill the destination columns in parallel while
            // keeping only per-worker implicit matvec scratch.
            {
                use ndarray::Axis;
                use ndarray::parallel::prelude::*;

                q_columns
                    .axis_iter_mut(Axis(1))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(e, q_col)| {
                        if e < n_dense {
                            dense::matvec_into(dense_matrices[e], z.view(), q_col);
                        } else {
                            let op = implicit_ops[e - n_dense];
                            let mut n_work = Array1::<f64>::zeros(n_obs);
                            let mut p_work = Array1::<f64>::zeros(p);
                            op.matvec_with_shared_xz_into(
                                x_vec.view(),
                                z.view(),
                                q_col,
                                n_work.view_mut(),
                                p_work.view_mut(),
                            );
                        }
                    });
            }

            // Step 3: R = H⁻¹ [q_1, ..., q_D] (block solve, total RHS)
            let r = hop.stochastic_trace_solve_multi(&q_columns, self.config.solve_rel_tol);

            // Step 4: Compute T[d, e] = u^T A_d r_e for all (d, e) pairs.
            // For dense A_d: T[d, e] = (A_d^T u)^T r_e = (A_d u)^T r_e (A_d symmetric)
            // For implicit A_d: use shared X multiplies and bounded per-pair scratch.

            // Precompute X u and X r_e for implicit operators.
            if let Some(ref x) = x_design {
                x.apply_view_into(u.view(), y_vec.view_mut());
            }

            // For dense operators, precompute A_d u once.
            for d in 0..n_dense {
                dense::matvec_into(dense_matrices[d], u.view(), dense_a_u[d].view_mut());
            }

            // Precompute X r_e for all axes e (for implicit operators). These
            // columns are independent and reused by every implicit row.
            if let Some(ref x) = x_design {
                use rayon::prelude::*;
                x_r.par_iter_mut().enumerate().for_each(|(e, x_r_e)| {
                    x.apply_view_into(r.column(e), x_r_e.view_mut());
                });
            }

            // Precompute row-wise implicit quantities that are reused across all
            // columns. Deliberately do not materialize (∂X/∂ψ_d) r_e for every
            // d×e pair; those n_obs-sized vectors are built inside the pair task
            // below, which bounds scratch by the number of active rayon workers
            // rather than n_ops * total.
            let implicit_scratch: Vec<ImplicitSecondOrderScratch> = {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                (0..n_ops)
                    .into_par_iter()
                    .map(|idx| {
                        let op = implicit_ops[idx];
                        let dx_u = op
                            .implicit_deriv
                            .forward_mul(op.axis, &u.view())
                            .expect(
                                "radial scalar evaluation failed during implicit derivative forward_mul",
                            );
                        let w = &*op.w_diag;
                        let mut w_dx_u = Array1::<f64>::zeros(n_obs);
                        let mut w_y = Array1::<f64>::zeros(n_obs);
                        for i in 0..w.len() {
                            w_dx_u[i] = w[i] * dx_u[i];
                            w_y[i] = w[i] * y_vec[i];
                        }
                        let mut u_s = Array1::<f64>::zeros(p);
                        dense::transpose_matvec_into(&op.s_psi, u.view(), u_s.view_mut());
                        ImplicitSecondOrderScratch { w_dx_u, w_y, u_s }
                    })
                    .collect()
            };

            let pair_count = total * total;
            let pair_values: Vec<(usize, usize, f64)> = {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                (0..pair_count)
                    .into_par_iter()
                    .map(|pair_idx| {
                        let d = pair_idx / total;
                        let e = pair_idx % total;
                        let r_e = r.column(e);
                        let val = if d < n_dense {
                            // Dense A_d: u^T A_d r_e = (A_d u)^T r_e
                            dense_a_u[d].dot(&r_e)
                        } else {
                            // Implicit A_d: compute u^T A_d r_e using shared X multiplies.
                            // u^T A_d r_e = ((∂X/∂ψ_d)u)^T (W X r_e)
                            //             + (Xu)^T (W (∂X/∂ψ_d) r_e)
                            //             + u^T S_psi r_e
                            let oi = d - n_dense;
                            let op = implicit_ops[oi];
                            let scratch = &implicit_scratch[oi];
                            let x_re = &x_r[e];
                            let dx_re = op
                                .implicit_deriv
                                .forward_mul(op.axis, &r_e)
                                .expect(
                                    "radial scalar evaluation failed during implicit derivative forward_mul",
                                );

                            let mut design_val = 0.0f64;
                            for i in 0..scratch.w_dx_u.len() {
                                design_val += scratch.w_dx_u[i] * x_re[i];
                                design_val += scratch.w_y[i] * dx_re[i];
                            }

                            // Non-Gaussian fixed-β third-derivative correction:
                            //   uᵀ Xᵀ diag(c ⊙ X_{ψ_d} β̂) X r_e
                            //   = Σ_i y_vec[i] · c_x_psi_beta_i · x_re[i]
                            if let Some(c_x_psi_beta) = op.c_x_psi_beta.as_ref() {
                                let c = c_x_psi_beta.as_ref();
                                for i in 0..scratch.w_dx_u.len() {
                                    design_val += y_vec[i] * c[i] * x_re[i];
                                }
                            }

                            // Penalty: u^T S_psi r_e = (S_psi^T u)^T r_e
                            let penalty_val = scratch.u_s.dot(&r_e);
                            design_val + penalty_val
                        };
                        (d, e, val)
                    })
                    .collect()
            };

            for (d, e, val) in pair_values {
                probe_values[[d, e]] = val;
            }
        })
    }

    /// Estimate the full D×D matrix of second-order traces `tr(H⁻¹ A_d H⁻¹ A_e)`
    /// for a mix of dense matrices and generic hyperoperators.
    pub fn estimate_second_order_traces_with_operators(
        &self,
        hop: &dyn HessianFactorization,
        dense_matrices: &[&Array2<f64>],
        operators: &[&dyn HyperOperator],
    ) -> Array2<f64> {
        let n_dense = dense_matrices.len();
        let n_ops = operators.len();
        let total = n_dense + n_ops;
        if total == 0 {
            return Array2::zeros((0, 0));
        }

        let p = hop.dim();
        if p == 0 {
            return Array2::zeros((total, total));
        }

        if total == 1 {
            let value = if n_dense == 1 {
                self.estimate_second_order_single_dense(hop, dense_matrices[0])
            } else {
                self.estimate_second_order_single_operator(hop, operators[0])
            };
            return Array2::from_elem((1, 1), value);
        }

        let mut q_columns = Array2::zeros((p, total));
        let mut a_u_columns = Array2::zeros((p, total));

        self.estimate_matrix_from_probe_batch(hop, total, |probe_id, z, probe_values| {
            let u = hop.stochastic_trace_solve_for_probe(
                z,
                self.config.solve_rel_tol,
                probe_id,
                Some(&self.trace_state),
            );

            for e in 0..n_dense {
                dense::matvec_into(dense_matrices[e], z.view(), q_columns.column_mut(e));
                dense::matvec_into(dense_matrices[e], u.view(), a_u_columns.column_mut(e));
            }
            for (oi, op) in operators.iter().enumerate() {
                let e = n_dense + oi;
                op.mul_vec_into(z.view(), q_columns.column_mut(e));
                op.mul_vec_into(u.view(), a_u_columns.column_mut(e));
            }

            let r = hop.stochastic_trace_solve_multi(&q_columns, self.config.solve_rel_tol);

            for d in 0..total {
                let a_d_u = a_u_columns.column(d);
                for e in d..total {
                    let r_e = r.column(e);
                    let val = a_d_u.dot(&r_e);
                    probe_values[[d, e]] = val;
                    if d != e {
                        let r_d = r.column(d);
                        let val_sym = a_u_columns.column(e).dot(&r_d);
                        probe_values[[e, d]] = val_sym;
                    }
                }
            }
        })
    }

    pub(crate) fn estimate_second_order_single_dense(
        &self,
        hop: &dyn HessianFactorization,
        matrix: &Array2<f64>,
    ) -> f64 {
        let p = hop.dim();
        if p == 0 {
            return 0.0;
        }

        if self.config.hutchpp_sketch_dim.is_some() {
            let op = DenseMatrixHyperOperator {
                matrix: matrix.clone(),
            };
            return hutchpp_estimate_trace_hinv_op_squared(hop, &op, &self.config);
        }

        let mut q = Array1::<f64>::zeros(p);
        self.estimate_matrix_from_probe_batch(hop, 1, |probe_id, z, probe_values| {
            let u = hop.stochastic_trace_solve_for_probe(
                z,
                self.config.solve_rel_tol,
                probe_id,
                Some(&self.trace_state),
            );
            dense::matvec_into(matrix, z.view(), q.view_mut());
            let r = hop.stochastic_trace_solve(&q, self.config.solve_rel_tol);
            probe_values[[0, 0]] = dense::bilinear(matrix, u.view(), r.view());
        })[[0, 0]]
    }

    pub(crate) fn estimate_second_order_single_implicit(
        &self,
        hop: &dyn HessianFactorization,
        op: &ImplicitHyperOperator,
    ) -> f64 {
        let p = hop.dim();
        if p == 0 {
            return 0.0;
        }

        if self.config.hutchpp_sketch_dim.is_some() {
            return hutchpp_estimate_trace_hinv_op_squared(hop, op, &self.config);
        }

        let n_obs = op.w_diag.len();
        let mut x_z = Array1::<f64>::zeros(n_obs);
        let mut x_u = Array1::<f64>::zeros(n_obs);
        let mut x_r = Array1::<f64>::zeros(n_obs);
        let mut n_work = Array1::<f64>::zeros(n_obs);
        let mut p_work = Array1::<f64>::zeros(p);
        let mut q = Array1::<f64>::zeros(p);
        self.estimate_matrix_from_probe_batch(hop, 1, |probe_id, z, probe_values| {
            let u = hop.stochastic_trace_solve_for_probe(
                z,
                self.config.solve_rel_tol,
                probe_id,
                Some(&self.trace_state),
            );
            op.x_design.apply_view_into(z.view(), x_z.view_mut());
            op.matvec_with_shared_xz_into(
                x_z.view(),
                z.view(),
                q.view_mut(),
                n_work.view_mut(),
                p_work.view_mut(),
            );
            let r = hop.stochastic_trace_solve(&q, self.config.solve_rel_tol);

            op.x_design.apply_view_into(u.view(), x_u.view_mut());
            op.x_design.apply_view_into(r.view(), x_r.view_mut());
            let dx_u = op
                .implicit_deriv
                .forward_mul(op.axis, &u.view())
                .expect("radial scalar evaluation failed during implicit derivative forward_mul");
            let dx_r = op
                .implicit_deriv
                .forward_mul(op.axis, &r.view())
                .expect("radial scalar evaluation failed during implicit derivative forward_mul");

            let w = &*op.w_diag;
            let mut value = 0.0;
            for i in 0..w.len() {
                let wi = w[i];
                value += dx_u[i] * wi * x_r[i];
                value += x_u[i] * wi * dx_r[i];
            }
            // Non-Gaussian fixed-β third-derivative correction:
            //   uᵀ Xᵀ diag(c ⊙ X_{ψ_d} β̂) X r = Σ_i (X u)_i · c_x_psi_beta_i · (X r)_i
            if let Some(c_x_psi_beta) = op.c_x_psi_beta.as_ref() {
                let c = c_x_psi_beta.as_ref();
                for i in 0..w.len() {
                    value += x_u[i] * c[i] * x_r[i];
                }
            }
            value += dense::bilinear(&op.s_psi, r.view(), u.view());

            probe_values[[0, 0]] = value;
        })[[0, 0]]
    }

    pub(crate) fn estimate_second_order_single_operator(
        &self,
        hop: &dyn HessianFactorization,
        op: &dyn HyperOperator,
    ) -> f64 {
        let p = hop.dim();
        if p == 0 {
            return 0.0;
        }

        let mut q = Array1::<f64>::zeros(p);
        let mut a_u = Array1::<f64>::zeros(p);
        self.estimate_matrix_from_probe_batch(hop, 1, |probe_id, z, probe_values| {
            let u = hop.stochastic_trace_solve_for_probe(
                z,
                self.config.solve_rel_tol,
                probe_id,
                Some(&self.trace_state),
            );
            op.mul_vec_into(z.view(), q.view_mut());
            op.mul_vec_into(u.view(), a_u.view_mut());
            let r = hop.stochastic_trace_solve(&q, self.config.solve_rel_tol);
            probe_values[[0, 0]] = a_u.dot(&r);
        })[[0, 0]]
    }

    /// Check the adaptive stopping criterion.
    ///
    /// Returns `true` if all coordinates have converged:
    /// ```text
    /// max_k  s_{M,k} / (√M · max(|q̄_{M,k}|, τ_rel))  ≤  ε
    /// ```
    pub(crate) fn check_convergence(&self, n: usize, means: &[f64], m2s: &[f64]) -> bool {
        if n < 2 {
            return false;
        }
        let sqrt_n = (n as f64).sqrt();
        let n_f = n as f64;

        for k in 0..means.len() {
            let variance = m2s[k] / (n_f - 1.0);
            let std_dev = variance.max(0.0).sqrt();
            let denom = sqrt_n * means[k].abs().max(self.config.tau_rel);
            let rel_err = std_dev / denom;
            if rel_err > self.config.relative_tol {
                return false;
            }
        }
        true
    }

    pub(crate) fn check_matrix_convergence(
        &self,
        n: usize,
        means: &Array2<f64>,
        m2s: &Array2<f64>,
    ) -> bool {
        if n < 2 {
            return false;
        }
        let sqrt_n = (n as f64).sqrt();
        let n_f = n as f64;
        let scale_floor = means
            .iter()
            .fold(0.0_f64, |acc, &value| acc.max(value.abs()))
            .max(1.0)
            * self.config.tau_rel;
        for ((d, e), &mean) in means.indexed_iter() {
            let variance = m2s[[d, e]] / (n_f - 1.0);
            let std_dev = variance.max(0.0).sqrt();
            let denom = sqrt_n * mean.abs().max(scale_floor);
            let rel_err = std_dev / denom;
            if rel_err > self.config.relative_tol {
                return false;
            }
        }
        true
    }
}

pub(crate) fn stochastic_trace_hinv_products_with_floor(
    hop: &dyn HessianFactorization,
    targets: StochasticTraceTargets<'_>,
    trace_state: Option<Arc<Mutex<StochasticTraceState>>>,
) -> Vec<f64> {
    let estimator = match trace_state {
        Some(state) => StochasticTraceEstimator::with_shared_trace_state(
            StochasticTraceConfig::default(),
            state,
        ),
        None => StochasticTraceEstimator::with_defaults(),
    };
    match targets {
        StochasticTraceTargets::Dense(matrices) if matrices.len() == 1 => {
            vec![estimator.estimate_single_trace(hop, matrices[0])]
        }
        StochasticTraceTargets::Dense(matrices) => estimator.estimate_traces(hop, matrices),
        StochasticTraceTargets::Mixed {
            dense_matrices,
            operators,
        } => estimator.estimate_traces_with_operators(hop, dense_matrices, operators),
        StochasticTraceTargets::Structural {
            dense_matrices,
            implicit_ops,
        } => estimator.estimate_traces_structural(hop, dense_matrices, implicit_ops),
    }
}

pub(crate) fn stochastic_trace_hinv_crosses<'a>(
    hop: &dyn HessianFactorization,
    dense_matrices: &'a [Array2<f64>],
    coord_has_operator: &[bool],
    generic_ops: &[&'a dyn HyperOperator],
    implicit_ops: &[&'a ImplicitHyperOperator],
) -> Array2<f64> {
    // The `_with_floor` variant takes a slice of references; adapt the owned
    // slice without copying the matrices.
    let dense_refs: Vec<&'a Array2<f64>> = dense_matrices.iter().collect();
    stochastic_trace_hinv_crosses_with_floor(
        hop,
        &dense_refs,
        coord_has_operator,
        generic_ops,
        implicit_ops,
        None,
    )
}

pub(crate) fn stochastic_trace_hinv_crosses_with_floor<'a>(
    hop: &dyn HessianFactorization,
    dense_matrices: &[&'a Array2<f64>],
    coord_has_operator: &[bool],
    generic_ops: &[&'a dyn HyperOperator],
    implicit_ops: &[&'a ImplicitHyperOperator],
    trace_state: Option<Arc<Mutex<StochasticTraceState>>>,
) -> Array2<f64> {
    let estimator = match trace_state {
        Some(state) => StochasticTraceEstimator::for_outer_hessian_with_trace_state(
            hop.dim(),
            coord_has_operator.len(),
            state,
        ),
        None => StochasticTraceEstimator::for_outer_hessian(hop.dim(), coord_has_operator.len()),
    };
    let raw_cross = if generic_ops.len() == implicit_ops.len() {
        estimator.estimate_second_order_traces(hop, dense_matrices, implicit_ops)
    } else {
        estimator.estimate_second_order_traces_with_operators(hop, dense_matrices, generic_ops)
    };

    let total_coords = coord_has_operator.len();
    let n_dense_total = coord_has_operator.iter().filter(|&&b| !b).count();
    let mut original_to_raw = Vec::with_capacity(total_coords);
    let mut dense_cursor = 0usize;
    let mut operator_cursor = n_dense_total;
    for &has_operator in coord_has_operator {
        if has_operator {
            original_to_raw.push(operator_cursor);
            operator_cursor += 1;
        } else {
            original_to_raw.push(dense_cursor);
            dense_cursor += 1;
        }
    }

    let mut mapped = Array2::zeros((total_coords, total_coords));
    for i in 0..total_coords {
        for j in 0..total_coords {
            mapped[[i, j]] = raw_cross[[original_to_raw[i], original_to_raw[j]]];
        }
    }
    mapped
}

// Lightweight xoshiro256ss RNG
//
// We use a self-contained xoshiro256ss implementation so that the stochastic
// trace estimator does not impose any new dependency requirements. The
// codebase already uses `rand` (0.10), but a minimal inline RNG avoids
// pulling in the full `rand` trait machinery for what is just a stream of
// random bits for ±1 generation.

/// Minimal xoshiro256** PRNG (period 2^256 − 1).
///
/// This is used exclusively for Rademacher probe generation. The state is
/// seeded deterministically from a u64 via splitmix64.
pub(crate) struct Xoshiro256SS {
    pub(crate) s: [u64; 4],
}

impl Xoshiro256SS {
    /// Seed from a single u64 via splitmix64 expansion.
    pub(crate) fn from_seed(seed: u64) -> Self {
        let mut sm = seed;
        let s0 = splitmix64(&mut sm);
        let s1 = splitmix64(&mut sm);
        let s2 = splitmix64(&mut sm);
        let s3 = splitmix64(&mut sm);
        // Guard against the all-zero state (astronomically unlikely but
        // formally required for xoshiro correctness).
        let s = if s0 | s1 | s2 | s3 == 0 {
            [1, 0, 0, 0]
        } else {
            [s0, s1, s2, s3]
        };
        Self { s }
    }

    /// Generate the next u64.
    #[inline]
    pub(crate) fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }
}

/// Splitmix64: deterministic expansion of a single u64 seed into a sequence.
#[inline]
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    gam_linalg::utils::splitmix64(state)
}

#[inline]
pub(crate) fn stochastic_trace_probe_id(seed: u64, probe_index: usize) -> u64 {
    let mut state = seed ^ (probe_index as u64).wrapping_mul(0xD1B54A32D192ED03);
    splitmix64(&mut state)
}

pub(crate) fn rademacher_probe_into(mut z: ArrayViewMut1<'_, f64>, rng: &mut Xoshiro256SS) {
    let mut bits: u64 = 0;
    let mut remaining_bits = 0u32;

    for i in 0..z.len() {
        if remaining_bits == 0 {
            bits = rng.next_u64();
            remaining_bits = 64;
        }
        z[i] = if bits & 1 == 0 { 1.0 } else { -1.0 };
        bits >>= 1;
        remaining_bits -= 1;
    }
}

/// Modified Gram–Schmidt orthonormalization of the columns of `y`,
/// writing the orthonormal basis into `q` and returning the retained
/// rank.
///
/// `y` and `q` must have the same shape `(p, m)`. Columns whose
/// reduction norm falls below `1e-12` of the largest input column
/// norm are dropped (numerical-rank cutoff). After this call,
/// `q.column(0..rank)` is column-orthonormal and approximates
/// `range(y)`; later columns of `q` are zeroed.
pub(crate) fn modified_gram_schmidt(y: &Array2<f64>, q: &mut Array2<f64>) -> usize {
    let p = y.nrows();
    let m = y.ncols();
    assert_eq!(q.dim(), (p, m));
    q.fill(0.0);
    if p == 0 || m == 0 {
        return 0;
    }
    let mut max_norm: f64 = 0.0;
    for j in 0..m {
        let n = y.column(j).dot(&y.column(j)).sqrt();
        if n > max_norm {
            max_norm = n;
        }
    }
    let drop_tol = (max_norm * 1.0e-12).max(f64::MIN_POSITIVE);
    let mut rank = 0usize;
    for j in 0..m {
        let mut v = y.column(j).to_owned();
        for k in 0..rank {
            let qk = q.column(k);
            let proj = qk.dot(&v);
            if proj != 0.0 {
                v.scaled_add(-proj, &qk);
            }
        }
        let norm = v.dot(&v).sqrt();
        if !norm.is_finite() || norm <= drop_tol {
            continue;
        }
        let inv = 1.0 / norm;
        v.iter_mut().for_each(|x| *x *= inv);
        q.column_mut(rank).assign(&v);
        rank += 1;
    }
    rank
}

/// Shared Hutch++ stochastic-trace scaffold (Meyer–Musco 2021, SOSA).
///
/// Estimates `tr(B)` for a linear map `B: x ↦ apply(hop, x, &mut tmp)`,
/// where `apply` is the per-probe action of `B` on a vector (using `tmp`
/// as scratch and returning a fresh `Array1<f64>`). The three public
/// estimators below differ *only* in this closure:
///
/// * `tr(H⁻¹ M)`        — `apply` = `M`-apply then one solve;
/// * `tr((H⁻¹ A)²)`     — `apply` = apply/solve/apply/solve;
/// * `tr(H⁻¹ A_L H⁻¹ A_R)` — `apply` = `A_R`/solve/`A_L`/solve.
///
/// Everything else (sketch dim, RNG seeding, randomized range finder +
/// modified Gram–Schmidt, exact low-rank trace `tr(Qᵀ B Q)`, residual
/// Hutchinson on `(I - Q Qᵀ) B (I - Q Qᵀ)` with the Welford-style
/// adaptive relative-error stop) is identical, so it lives here once.
/// `B` need not be self-adjoint: on Rademacher probes `E[zᵀ B z] = tr(B)`
/// regardless, and the projected `tr(Qᵀ B Q)` is exact on `range(Q)`.
pub(crate) fn hutchpp_estimate_trace_with_apply<F>(
    p: usize,
    config: &StochasticTraceConfig,
    apply: F,
) -> f64
where
    F: Fn(ArrayView1<'_, f64>, &mut Array1<f64>) -> Array1<f64>,
{
    if p == 0 {
        return 0.0;
    }
    let sketch_dim = config.hutchpp_sketch_dim.unwrap_or(0).min(p);
    let mut rng_state = Xoshiro256SS::from_seed(config.seed);

    // Phase 1: build orthonormal Q ∈ R^{p × sketch_dim} approximating
    // range(B) via a randomized range finder.
    let mut q = Array2::<f64>::zeros((p, sketch_dim));
    let mut q_rank = 0usize;
    if sketch_dim > 0 {
        let mut y = Array2::<f64>::zeros((p, sketch_dim));
        let mut z = Array1::<f64>::zeros(p);
        let mut tmp = Array1::<f64>::zeros(p);
        for j in 0..sketch_dim {
            rademacher_probe_into(z.view_mut(), &mut rng_state);
            let w = apply(z.view(), &mut tmp);
            y.column_mut(j).assign(&w);
        }
        q_rank = modified_gram_schmidt(&y, &mut q);
    }

    // Phase 2: T_low = tr(Qᵀ B Q), exact on range(Q).
    let mut t_low = 0.0;
    if q_rank > 0 {
        let mut tmp = Array1::<f64>::zeros(p);
        for j in 0..q_rank {
            let qcol = q.column(j).to_owned();
            let w = apply(qcol.view(), &mut tmp);
            t_low += qcol.dot(&w);
        }
    }

    // Phase 3: residual Hutchinson on (I - Q Qᵀ) B (I - Q Qᵀ).
    // Budget = remaining matvecs from n_probes_max minus the 2*q_rank
    // we already spent (sketch + Q-trace), but never below n_probes_min.
    let used = 2 * q_rank;
    let residual_budget_max = config.n_probes_max.saturating_sub(used);
    let residual_min = config.n_probes_min.min(residual_budget_max);
    let residual_budget = residual_budget_max.max(residual_min);
    if residual_budget == 0 {
        return t_low;
    }

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    let mut z = Array1::<f64>::zeros(p);
    let mut z_tilde = Array1::<f64>::zeros(p);
    let mut tmp = Array1::<f64>::zeros(p);
    let check_interval = 4usize;
    for _ in 0..residual_budget {
        rademacher_probe_into(z.view_mut(), &mut rng_state);
        // z_tilde = (I - Q Qᵀ) z = z - Q (Qᵀ z)
        z_tilde.assign(&z);
        if q_rank > 0 {
            for j in 0..q_rank {
                let qcol = q.column(j);
                let proj = qcol.dot(&z);
                if proj != 0.0 {
                    z_tilde.scaled_add(-proj, &qcol);
                }
            }
        }
        let w = apply(z_tilde.view(), &mut tmp);
        let q_val = z_tilde.dot(&w);
        sum += q_val;
        sum_sq += q_val * q_val;
        count += 1;

        // Adaptive stopping: same Welford-style relative-error check
        // as `estimate_from_probe_batch`, applied to the residual mean.
        if count >= residual_min && count.is_multiple_of(check_interval) && count >= 2 {
            let n = count as f64;
            let mean = sum / n;
            let var = (sum_sq - n * mean * mean) / (n - 1.0).max(1.0);
            if var.is_finite() && var >= 0.0 {
                let stderr = (var / n).sqrt();
                let denom = (mean.abs()).max(config.tau_rel);
                if stderr / denom <= config.relative_tol {
                    break;
                }
            }
        }
    }
    let mean_residual = if count > 0 { sum / count as f64 } else { 0.0 };
    t_low + mean_residual
}

/// Hutch++ estimate of `tr(H⁻¹ M)` where `M` is accessed through its
/// matrix-vector product (operator-only, dim p).
///
/// Total cost: `2 m_s + m_h` H⁻¹ solves and `M·v` matvecs, where
/// `m_s = config.hutchpp_sketch_dim.unwrap_or(0)` and `m_h` is the
/// number of residual Hutchinson probes drawn (between
/// `config.n_probes_min` and `config.n_probes_max - 2 m_s`).
///
/// When `hutchpp_sketch_dim` is `None`, this falls back to plain
/// Hutchinson on the full probe budget — the result is deterministic
/// for a given seed because the probe RNG is seeded from
/// `config.seed`.
///
/// # Algorithm (Meyer–Musco 2021, SOSA)
///
/// 1. Sketch: draw `Z_s ∈ {±1}^{p × m_s}` Rademacher, compute
///    `Y = H⁻¹ M Z_s`, orthonormalize columns: `Y = Q R`.
/// 2. Low-rank trace: `T_low = tr(Qᵀ H⁻¹ M Q)` exactly via `m_s`
///    additional matvecs into `W = H⁻¹ M Q` and accumulating
///    `Σ_j Q[:,j] · W[:,j]`.
/// 3. Residual Hutchinson on the orthogonal complement: for each
///    residual probe `z`, set `z̃ = (I - Q Qᵀ) z`, compute
///    `w̃ = H⁻¹ M z̃`, and accumulate `z̃ · w̃` (which equals
///    `z̃ᵀ (H⁻¹ M) z̃` because `z̃` is in the complement).
/// 4. Output: `T_low + (1/m_h) Σ residual estimates`.
///
/// # When this wins over plain Hutchinson
///
/// Hutch++ converges in `O(1/ε)` matvecs vs `O(1/ε²)` for Hutchinson.
/// The gain is largest when `H⁻¹ M` has rapid singular-value decay —
/// the sketch captures the dominant subspace exactly and Hutchinson
/// only handles the small residual. For roughly-flat spectra both
/// methods perform similarly per-matvec.
pub(crate) fn hutchpp_estimate_trace_hinv_operator<H, O>(
    hop: &H,
    op: &O,
    config: &StochasticTraceConfig,
) -> f64
where
    H: HessianFactorization + ?Sized,
    O: HyperOperator + ?Sized,
{
    let p = hop.dim();
    assert_eq!(op.dim(), p, "Hutch++: operator dim mismatch");
    // B x = H⁻¹ M x: apply M then a single solve.
    hutchpp_estimate_trace_with_apply(p, config, |x, tmp| {
        op.mul_vec_into(x, tmp.view_mut());
        hop.stochastic_trace_solve(tmp, config.solve_rel_tol)
    })
}

/// Hutch++ estimate of `tr((H⁻¹ A)²) = tr(H⁻¹ A H⁻¹ A)` for a symmetric
/// HVP-only operator `A`. Cost per applied "matvec" is 2 H⁻¹ solves and
/// 2 A applies; total cost is `2 m_s + m_h` such matvecs.
///
/// Although `B = H⁻¹ A` is not symmetric in the standard inner product,
/// `B²` is similar to `(H^{-1/2} A H^{-1/2})²` (PSD), so its trace is
/// nonnegative and Hutch++ on the linear map `x ↦ B² x` produces an
/// unbiased estimate of `tr(B²)` on standard probes.
pub(crate) fn hutchpp_estimate_trace_hinv_op_squared<H, O>(
    hop: &H,
    op: &O,
    config: &StochasticTraceConfig,
) -> f64
where
    H: HessianFactorization + ?Sized,
    O: HyperOperator + ?Sized,
{
    let p = hop.dim();
    assert_eq!(op.dim(), p, "Hutch++ squared: operator dim mismatch");
    // B x = (H⁻¹ A)² x = H⁻¹ A H⁻¹ A x via two solve+apply legs.
    hutchpp_estimate_trace_with_apply(p, config, |x, tmp| {
        op.mul_vec_into(x, tmp.view_mut());
        let mid = hop.stochastic_trace_solve(tmp, config.solve_rel_tol);
        op.mul_vec_into(mid.view(), tmp.view_mut());
        hop.stochastic_trace_solve(tmp, config.solve_rel_tol)
    })
}

/// Hutch++-style estimate of `tr(H⁻¹ A_left H⁻¹ A_right)` for two
/// (possibly distinct) symmetric HVP-only operators. Uses a shared
/// sketch built from `M = M_L M_R` where `M_L = H⁻¹ A_left` and
/// `M_R = H⁻¹ A_right`; per matvec is 2 H⁻¹ solves + 2 A applies.
///
/// On standard Rademacher probes `E[zᵀ M z] = tr(M)` regardless of
/// symmetry, so the residual Hutchinson average is unbiased even when
/// `M` is not self-adjoint in the standard inner product.
///
/// A leave-one-out XTrace estimator (Epperly & Tropp 2024, arxiv
/// 2301.07825) would reduce variance further by exchanging each probe
/// between sketch and residual roles, at O(m²) bookkeeping cost.
pub(crate) fn hutchpp_estimate_trace_hinv_operator_cross<H, L, R>(
    hop: &H,
    left: &L,
    right: &R,
    config: &StochasticTraceConfig,
) -> f64
where
    H: HessianFactorization + ?Sized,
    L: HyperOperator + ?Sized,
    R: HyperOperator + ?Sized,
{
    let p = hop.dim();
    assert_eq!(left.dim(), p, "cross trace: left operator dim mismatch");
    assert_eq!(right.dim(), p, "cross trace: right operator dim mismatch");
    // M x = H⁻¹ A_L H⁻¹ A_R x.
    hutchpp_estimate_trace_with_apply(p, config, |x, tmp| {
        right.mul_vec_into(x, tmp.view_mut());
        let mid = hop.stochastic_trace_solve(tmp, config.solve_rel_tol);
        left.mul_vec_into(mid.view(), tmp.view_mut());
        hop.stochastic_trace_solve(tmp, config.solve_rel_tol)
    })
}
