    use super::*;
    use faer::Side;

    enum FaerFactor {
        Llt(FaerLlt<f64>),
        Lblt(FaerLblt<f64>),
        Ldlt(FaerLdlt<f64>),
    }

    impl FaerFactor {
        fn solve(&self, rhs: faer::MatRef<'_, f64>) -> FaerMat<f64> {
            match self {
                FaerFactor::Llt(f) => f.solve(rhs),
                FaerFactor::Lblt(f) => f.solve(rhs),
                FaerFactor::Ldlt(f) => f.solve(rhs),
            }
        }

        fn solve_in_place(&self, rhs: faer::MatMut<'_, f64>) {
            match self {
                FaerFactor::Llt(f) => f.solve_in_place(rhs),
                FaerFactor::Lblt(f) => f.solve_in_place(rhs),
                FaerFactor::Ldlt(f) => f.solve_in_place(rhs),
            }
        }
    }

    /// Holds the state for the outer REML optimization and supplies cost and
    /// gradient evaluations to the `wolfe_bfgs` optimizer.
    ///
    /// The `cache` field uses `RefCell` to enable interior mutability. This is a crucial
    /// performance optimization. The `cost_and_grad` closure required by the BFGS
    /// optimizer takes an immutable reference `&self`. However, we want to cache the
    /// results of the expensive P-IRLS computation to avoid re-calculating the fit
    /// for the same `rho` vector, which can happen during the line search.
    /// `RefCell` allows us to mutate the cache through a `&self` reference,
    /// making this optimization possible while adhering to the optimizer's API.

    #[derive(Clone)]
    struct EvalShared {
        key: Option<Vec<u64>>,
        pirls_result: Arc<PirlsResult>,
        h_eff: Arc<Array2<f64>>,
        ridge_passport: RidgePassport,
        /// The exact H_total matrix used for LAML cost computation.
        /// For Firth: h_eff - h_phi. For non-Firth: h_eff.
        h_total: Arc<Array2<f64>>,

        // ══════════════════════════════════════════════════════════════════════
        // WHY TWO INVERSES? (Hybrid Approach for Indefinite Hessians)
        // ══════════════════════════════════════════════════════════════════════
        //
        // The LAML gradient has two terms requiring DIFFERENT matrix inverses:
        //
        // 1. TRACE TERM (∂/∂ρ log|H|): Uses PSEUDOINVERSE H₊†
        //    - Cost defines log|H| = Σᵢ log(λᵢ) for λᵢ > ε only (truncated)
        //    - Derivative: ∂J/∂ρ = ½ tr(H₊† ∂H/∂ρ)
        //    - H₊† = Σᵢ (1/λᵢ) uᵢuᵢᵀ for positive λᵢ only
        //    - Negative eigenvalues contribute 0 to cost, so their derivative contribution is 0
        //
        // 2. IMPLICIT TERM (dβ/dρ): Uses RIDGED FACTOR (H + δI)⁻¹
        //    - PIRLS stabilizes indefinite H by adding ridge: solves (H + δI)β = ...
        //    - Stationarity condition: G(β,ρ) = ∇L + δβ = 0
        //    - By Implicit Function Theorem: dβ/dρ = (H + δI)⁻¹ (λₖ Sₖ β)
        //    - Must use ridged inverse because β moves on the RIDGED surface
        //
        // EXAMPLE: H = -5 (indefinite), ridge δ = 10
        //   Trace term: Pseudoinverse → 0 (correct: truncated eigenvalue)
        //               Ridged inverse → 0.2 (WRONG: gradient of non-existent curve)
        //   Implicit term: Ridged inverse → 1/5 (correct: solver sees stiffness +5)
        //                  Pseudoinverse → 0 or ∞ (WRONG: ignores ridge physics)
        //
        // ══════════════════════════════════════════════════════════════════════
        /// Positive-spectrum factor W = U_+ diag(1/sqrt(lambda_+)).
        /// This avoids materializing H₊† = W Wᵀ in hot paths.
        ///
        /// We use identities:
        ///   H₊† v = W (Wᵀ v)
        ///   tr(H₊† S_k) = ||R_k W||_F², where S_k = R_kᵀ R_k.
        h_pos_factor_w: Arc<Array2<f64>>,

        /// Log determinant via truncation: Σᵢ log(λᵢ) for λᵢ > ε only.
        h_total_log_det: f64,
    }

    impl EvalShared {
        fn matches(&self, key: &Option<Vec<u64>>) -> bool {
            match (&self.key, key) {
                (None, None) => true,
                (Some(a), Some(b)) => a == b,
                _ => false,
            }
        }
    }

    struct RemlWorkspace {
        rho_plus: Array1<f64>,
        rho_minus: Array1<f64>,
        lambda_values: Array1<f64>,
        grad_primary: Array1<f64>,
        grad_secondary: Array1<f64>,
        cost_gradient: Array1<f64>,
        prior_gradient: Array1<f64>,
    }

    impl RemlWorkspace {
        fn new(max_penalties: usize) -> Self {
            RemlWorkspace {
                rho_plus: Array1::zeros(max_penalties),
                rho_minus: Array1::zeros(max_penalties),
                lambda_values: Array1::zeros(max_penalties),
                grad_primary: Array1::zeros(max_penalties),
                grad_secondary: Array1::zeros(max_penalties),
                cost_gradient: Array1::zeros(max_penalties),
                prior_gradient: Array1::zeros(max_penalties),
            }
        }

        fn reset_for_eval(&mut self, penalties: usize) {
            if penalties == 0 {
                return;
            }
            self.grad_primary.slice_mut(s![..penalties]).fill(0.0);
            self.grad_secondary.slice_mut(s![..penalties]).fill(0.0);
            self.cost_gradient.slice_mut(s![..penalties]).fill(0.0);
            self.prior_gradient.slice_mut(s![..penalties]).fill(0.0);
        }

        fn set_lambda_values(&mut self, rho: &Array1<f64>) {
            let len = rho.len();
            if len == 0 {
                return;
            }
            let mut view = self.lambda_values.slice_mut(s![..len]);
            for (dst, &src) in view.iter_mut().zip(rho.iter()) {
                *dst = src.exp();
            }
        }

        fn lambda_view(&self, len: usize) -> ArrayView1<'_, f64> {
            self.lambda_values.slice(s![..len])
        }

        fn cost_gradient_view(&mut self, len: usize) -> ArrayViewMut1<'_, f64> {
            self.cost_gradient.slice_mut(s![..len])
        }

        fn zero_cost_gradient(&mut self, len: usize) {
            self.cost_gradient.slice_mut(s![..len]).fill(0.0);
        }

        fn cost_gradient_view_const(&self, len: usize) -> ArrayView1<'_, f64> {
            self.cost_gradient.slice(s![..len])
        }

        fn soft_prior_cost_and_grad<'a>(
            &'a mut self,
            rho: &Array1<f64>,
        ) -> (f64, ArrayView1<'a, f64>) {
            let len = rho.len();
            let mut grad_view = self.prior_gradient.slice_mut(s![..len]);
            grad_view.fill(0.0);

            if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
                return (0.0, self.prior_gradient.slice(s![..len]));
            }

            let inv_bound = 1.0 / RHO_BOUND;
            let sharp = RHO_SOFT_PRIOR_SHARPNESS;
            let mut cost = 0.0;
            for (grad, &ri) in grad_view.iter_mut().zip(rho.iter()) {
                let scaled = sharp * ri * inv_bound;
                cost += scaled.cosh().ln();
                *grad = sharp * inv_bound * scaled.tanh();
            }

            if RHO_SOFT_PRIOR_WEIGHT != 1.0 {
                for grad in grad_view.iter_mut() {
                    *grad *= RHO_SOFT_PRIOR_WEIGHT;
                }
                cost *= RHO_SOFT_PRIOR_WEIGHT;
            }

            (cost, self.prior_gradient.slice(s![..len]))
        }
    }

    struct PirlsLruCache {
        map: HashMap<Vec<u64>, Arc<PirlsResult>>,
        order: VecDeque<Vec<u64>>,
        capacity: usize,
    }

    impl PirlsLruCache {
        fn new(capacity: usize) -> Self {
            Self {
                map: HashMap::new(),
                order: VecDeque::new(),
                capacity: capacity.max(1),
            }
        }

        fn touch(&mut self, key: &Vec<u64>) {
            if let Some(pos) = self.order.iter().position(|k| k == key) {
                self.order.remove(pos);
            }
            self.order.push_back(key.clone());
        }

        fn get(&mut self, key: &Vec<u64>) -> Option<Arc<PirlsResult>> {
            let value = self.map.get(key).cloned();
            if value.is_some() {
                self.touch(key);
            }
            value
        }

        fn insert(&mut self, key: Vec<u64>, value: Arc<PirlsResult>) {
            if self.map.contains_key(&key) {
                self.map.insert(key.clone(), value);
                self.touch(&key);
                return;
            }

            while self.map.len() >= self.capacity {
                if let Some(evict_key) = self.order.pop_front() {
                    self.map.remove(&evict_key);
                } else {
                    break;
                }
            }

            self.order.push_back(key.clone());
            self.map.insert(key, value);
        }
    }

    pub(crate) struct RemlState<'a> {
        y: ArrayView1<'a, f64>,
        x: DesignMatrix,
        weights: ArrayView1<'a, f64>,
        offset: Array1<f64>,
        // Original penalty matrices S_k (p × p), ρ-independent basis
        s_full_list: Vec<Array2<f64>>,
        pub(super) rs_list: Vec<Array2<f64>>, // Pre-computed penalty square roots
        balanced_penalty_root: Array2<f64>,
        reparam_invariant: ReparamInvariant,
        p: usize,
        config: &'a RemlConfig,
        nullspace_dims: Vec<usize>,
        coefficient_lower_bounds: Option<Array1<f64>>,
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,

        cache: RwLock<PirlsLruCache>,
        faer_factor_cache: RwLock<HashMap<Vec<u64>, Arc<FaerFactor>>>,
        pirls_cache_enabled: AtomicBool,
        current_eval_bundle: RwLock<Option<EvalShared>>,
        cost_last: RwLock<Option<CostAgg>>,
        cost_repeat: RwLock<u64>,
        cost_last_emit: RwLock<u64>,
        cost_eval_count: RwLock<u64>,
        raw_cond_snapshot: RwLock<f64>,
        gaussian_cond_snapshot: RwLock<f64>,
        workspace: Mutex<RemlWorkspace>,
        pub(super) warm_start_beta: RwLock<Option<Coefficients>>,
        warm_start_enabled: AtomicBool,
    }

    #[derive(Clone)]
    struct CostKey {
        compact: String,
    }

    #[derive(Clone)]
    struct CostAgg {
        key: CostKey,
        count: u64,
        stab_cond_min: f64,
        stab_cond_max: f64,
        stab_cond_last: f64,
        raw_cond_min: f64,
        raw_cond_max: f64,
        raw_cond_last: f64,
        laml_min: f64,
        laml_max: f64,
        laml_last: f64,
        edf_min: f64,
        edf_max: f64,
        edf_last: f64,
        trace_min: f64,
        trace_max: f64,
        trace_last: f64,
    }

    impl CostKey {
        fn new(rho: &[f64], smooth: &[f64], stab_cond: f64, raw_cond: f64) -> Self {
            let rho_compact = format_compact_series(rho, |v| format!("{:.3}", v));
            let smooth_compact = format_compact_series(smooth, |v| format!("{:.2e}", v));
            let compact = format!(
                "rho={} | smooth={} | κ(stable/raw)={:.3e}/{:.3e}",
                rho_compact, smooth_compact, stab_cond, raw_cond
            );
            let compact = compact.replace("-0.000", "0.000");
            Self { compact }
        }

        fn approx_eq(&self, other: &Self) -> bool {
            self.compact == other.compact
        }

        fn format_compact(&self) -> String {
            self.compact.clone()
        }
    }

    impl CostAgg {
        fn new(
            key: CostKey,
            laml: f64,
            edf: f64,
            trace: f64,
            stab_cond: f64,
            raw_cond: f64,
        ) -> Self {
            Self {
                key,
                count: 1,
                stab_cond_min: stab_cond,
                stab_cond_max: stab_cond,
                stab_cond_last: stab_cond,
                raw_cond_min: raw_cond,
                raw_cond_max: raw_cond,
                raw_cond_last: raw_cond,
                laml_min: laml,
                laml_max: laml,
                laml_last: laml,
                edf_min: edf,
                edf_max: edf,
                edf_last: edf,
                trace_min: trace,
                trace_max: trace,
                trace_last: trace,
            }
        }

        fn update(&mut self, laml: f64, edf: f64, trace: f64, stab_cond: f64, raw_cond: f64) {
            self.count += 1;
            self.laml_last = laml;
            self.edf_last = edf;
            self.trace_last = trace;
            self.stab_cond_last = stab_cond;
            self.raw_cond_last = raw_cond;
            if stab_cond < self.stab_cond_min {
                self.stab_cond_min = stab_cond;
            }
            if stab_cond > self.stab_cond_max {
                self.stab_cond_max = stab_cond;
            }
            if raw_cond < self.raw_cond_min {
                self.raw_cond_min = raw_cond;
            }
            if raw_cond > self.raw_cond_max {
                self.raw_cond_max = raw_cond;
            }
            if laml < self.laml_min {
                self.laml_min = laml;
            }
            if laml > self.laml_max {
                self.laml_max = laml;
            }
            if edf < self.edf_min {
                self.edf_min = edf;
            }
            if edf > self.edf_max {
                self.edf_max = edf;
            }
            if trace < self.trace_min {
                self.trace_min = trace;
            }
            if trace > self.trace_max {
                self.trace_max = trace;
            }
        }

        fn format_summary(&self) -> String {
            let key = self.key.format_compact();
            let metric =
                |label: &str, min: f64, max: f64, last: f64, fmt: &dyn Fn(f64) -> String| {
                    if approx_f64(min, max, 1e-6, 1e-9) && approx_f64(min, last, 1e-6, 1e-9) {
                        format!("{label}={}", fmt(min))
                    } else {
                        let range = format_range(min, max, |v| fmt(v));
                        format!("{label}={range} last={}", fmt(last))
                    }
                };
            let kappa = if approx_f64(self.stab_cond_min, self.stab_cond_max, 1e-6, 1e-9)
                && approx_f64(self.raw_cond_min, self.raw_cond_max, 1e-6, 1e-9)
                && approx_f64(self.stab_cond_min, self.stab_cond_last, 1e-6, 1e-9)
                && approx_f64(self.raw_cond_min, self.raw_cond_last, 1e-6, 1e-9)
            {
                format!(
                    "κ(stable/raw)={}/{}",
                    format_cond(self.stab_cond_min),
                    format_cond(self.raw_cond_min)
                )
            } else {
                let stable = format_range(self.stab_cond_min, self.stab_cond_max, format_cond);
                let raw = format_range(self.raw_cond_min, self.raw_cond_max, format_cond);
                format!(
                    "κ(stable/raw)={stable}/{raw} last={}/{}",
                    format_cond(self.stab_cond_last),
                    format_cond(self.raw_cond_last)
                )
            };
            let laml = metric("LAML", self.laml_min, self.laml_max, self.laml_last, &|v| {
                format!("{:.6e}", v)
            });
            let edf = metric("EDF", self.edf_min, self.edf_max, self.edf_last, &|v| {
                format!("{:.6}", v)
            });
            let trace = metric(
                "tr(H^-1 Sλ)",
                self.trace_min,
                self.trace_max,
                self.trace_last,
                &|v| format!("{:.6}", v),
            );
            let count = if self.count > 1 {
                format!(" | count={}", self.count)
            } else {
                String::new()
            };
            format!("{key}{count} | {kappa} | {laml} | {edf} | {trace}",)
        }
    }

    // Formatting utilities moved to crate::diagnostics
    impl<'a> RemlState<'a> {
        #[inline]
        fn should_compute_hot_diagnostics(&self, eval_idx: u64) -> bool {
            // Keep expensive diagnostics out of the hot path unless they can
            // be surfaced. This has zero effect on optimization math.
            (log::log_enabled!(log::Level::Info) || log::log_enabled!(log::Level::Warn))
                && (eval_idx == 1 || eval_idx % 200 == 0)
        }

        fn log_gam_cost(
            &self,
            rho: &Array1<f64>,
            lambdas: &[f64],
            laml: f64,
            stab_cond: f64,
            raw_cond: f64,
            edf: f64,
            trace_h_inv_s_lambda: f64,
        ) {
            const GAM_REPEAT_EMIT: u64 = 50;
            const GAM_MIN_EMIT_GAP: u64 = 200;
            let rho_q = quantize_vec(rho.as_slice().unwrap_or_default(), 5e-3, 1e-6);
            let smooth_q = quantize_vec(lambdas, 5e-3, 1e-6);
            let stab_q = quantize_value(stab_cond, 5e-3, 1e-6);
            let raw_q = quantize_value(raw_cond, 5e-3, 1e-6);
            let key = CostKey::new(&rho_q, &smooth_q, stab_q, raw_q);

            let mut last_opt = self.cost_last.write().unwrap();
            let mut repeat = self.cost_repeat.write().unwrap();
            let mut last_emit = self.cost_last_emit.write().unwrap();
            let eval_idx = *self.cost_eval_count.read().unwrap();

            if let Some(last) = last_opt.as_mut() {
                if last.key.approx_eq(&key) {
                    last.update(laml, edf, trace_h_inv_s_lambda, stab_q, raw_q);
                    *repeat += 1;
                    if *repeat >= GAM_REPEAT_EMIT
                        && eval_idx.saturating_sub(*last_emit) >= GAM_MIN_EMIT_GAP
                    {
                        println!("[GAM COST] {}", last.format_summary());
                        *repeat = 0;
                        *last_emit = eval_idx;
                    }
                    return;
                }

                let emit_prev =
                    last.count > 1 && eval_idx.saturating_sub(*last_emit) >= GAM_MIN_EMIT_GAP;
                if emit_prev {
                    println!("[GAM COST] {}", last.format_summary());
                    *last_emit = eval_idx;
                }
            }

            let new_agg = CostAgg::new(key, laml, edf, trace_h_inv_s_lambda, stab_q, raw_q);
            if eval_idx.saturating_sub(*last_emit) >= GAM_MIN_EMIT_GAP {
                println!("[GAM COST] {}", new_agg.format_summary());
                *last_emit = eval_idx;
            }
            *last_opt = Some(new_agg);
            *repeat = 0;
        }

        #[allow(dead_code)]
        pub fn reset_optimizer_tracking(&self) {
            self.current_eval_bundle.write().unwrap().take();
            self.cost_last.write().unwrap().take();
            *self.cost_repeat.write().unwrap() = 0;
            *self.cost_last_emit.write().unwrap() = 0;
            *self.cost_eval_count.write().unwrap() = 0;
            *self.raw_cond_snapshot.write().unwrap() = f64::NAN;
            *self.gaussian_cond_snapshot.write().unwrap() = f64::NAN;
        }

        /// Compute soft prior cost without needing workspace
        fn compute_soft_prior_cost(&self, rho: &Array1<f64>) -> f64 {
            let len = rho.len();
            if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
                return 0.0;
            }

            let inv_bound = 1.0 / RHO_BOUND;
            let sharp = RHO_SOFT_PRIOR_SHARPNESS;
            let mut cost = 0.0;
            for &ri in rho.iter() {
                let scaled = sharp * ri * inv_bound;
                cost += scaled.cosh().ln();
            }

            cost * RHO_SOFT_PRIOR_WEIGHT
        }

        /// Compute soft prior gradient without workspace mutation.
        fn compute_soft_prior_grad(&self, rho: &Array1<f64>) -> Array1<f64> {
            let len = rho.len();
            let mut grad = Array1::<f64>::zeros(len);
            if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
                return grad;
            }
            let inv_bound = 1.0 / RHO_BOUND;
            let sharp = RHO_SOFT_PRIOR_SHARPNESS;
            for (g, &ri) in grad.iter_mut().zip(rho.iter()) {
                let scaled = sharp * ri * inv_bound;
                *g = sharp * inv_bound * scaled.tanh() * RHO_SOFT_PRIOR_WEIGHT;
            }
            grad
        }

        /// Add the exact Hessian of the soft rho prior in place.
        ///
        /// Prior definition per coordinate:
        ///   C_i(rho_i) = w * log(cosh(a * rho_i)),
        ///   a = RHO_SOFT_PRIOR_SHARPNESS / RHO_BOUND,
        ///   w = RHO_SOFT_PRIOR_WEIGHT.
        ///
        /// Then:
        ///   dC_i/drho_i   = w * a * tanh(a * rho_i),
        ///   d²C_i/drho_i² = w * a² * sech²(a * rho_i)
        ///                = w * a² * (1 - tanh²(a * rho_i)).
        ///
        /// The prior is separable across coordinates, so off-diagonals are zero.
        fn add_soft_prior_hessian_in_place(&self, rho: &Array1<f64>, hess: &mut Array2<f64>) {
            let len = rho.len();
            if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
                return;
            }
            let a = RHO_SOFT_PRIOR_SHARPNESS / RHO_BOUND;
            let prefactor = RHO_SOFT_PRIOR_WEIGHT * a * a;
            for i in 0..len {
                let t = (a * rho[i]).tanh();
                hess[[i, i]] += prefactor * (1.0 - t * t);
            }
        }

        /// Returns the effective Hessian and the ridge value used (if any).
        /// Uses the same Hessian matrix in both cost and gradient calculations.
        ///
        /// PIRLS folds any stabilization ridge directly into the penalized objective:
        ///   l_p(β; ρ) = l(β) - 0.5 * βᵀ (S_λ + ridge I) β.
        /// Therefore the curvature used in LAML is
        ///   H_eff = X'WX + S_λ + ridge I,
        /// and adding another ridge here places the Laplace expansion on a different surface.
        fn effective_hessian(
            &self,
            pr: &PirlsResult,
        ) -> Result<(Array2<f64>, RidgePassport), EstimationError> {
            let base = pr.stabilized_hessian_transformed.clone();

            if base.cholesky(Side::Lower).is_ok() {
                return Ok((base, pr.ridge_passport));
            }

            Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            })
        }

        #[allow(dead_code)]
        pub(super) fn new<X>(
            y: ArrayView1<'a, f64>,
            x: X,
            weights: ArrayView1<'a, f64>,
            s_list: Vec<Array2<f64>>,
            p: usize,
            config: &'a RemlConfig,
            nullspace_dims: Option<Vec<usize>>,
            coefficient_lower_bounds: Option<Array1<f64>>,
            linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        ) -> Result<Self, EstimationError>
        where
            X: Into<DesignMatrix>,
        {
            let zero_offset = Array1::<f64>::zeros(y.len());
            Self::new_with_offset(
                y,
                x,
                weights,
                zero_offset.view(),
                s_list,
                p,
                config,
                nullspace_dims,
                coefficient_lower_bounds,
                linear_constraints,
            )
        }

        pub(super) fn new_with_offset<X>(
            y: ArrayView1<'a, f64>,
            x: X,
            weights: ArrayView1<'a, f64>,
            offset: ArrayView1<'_, f64>,
            s_list: Vec<Array2<f64>>,
            p: usize,
            config: &'a RemlConfig,
            nullspace_dims: Option<Vec<usize>>,
            coefficient_lower_bounds: Option<Array1<f64>>,
            linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        ) -> Result<Self, EstimationError>
        where
            X: Into<DesignMatrix>,
        {
            // Pre-compute penalty square roots once
            let rs_list = compute_penalty_square_roots(&s_list)?;
            let x = x.into();

            let expected_len = s_list.len();
            let nullspace_dims = match nullspace_dims {
                Some(dims) => {
                    if dims.len() != expected_len {
                        return Err(EstimationError::InvalidInput(format!(
                            "nullspace_dims length {} does not match penalties {}",
                            dims.len(),
                            expected_len
                        )));
                    }
                    dims
                }
                None => vec![0; expected_len],
            };

            let penalty_count = rs_list.len();
            let workspace = RemlWorkspace::new(penalty_count);

            let balanced_penalty_root = create_balanced_penalty_root(&s_list, p)?;
            let reparam_invariant = precompute_reparam_invariant(&rs_list, p)?;

            Ok(Self {
                y,
                x,
                weights,
                offset: offset.to_owned(),
                s_full_list: s_list,
                rs_list,
                balanced_penalty_root,
                reparam_invariant,
                p,
                config,
                nullspace_dims,
                coefficient_lower_bounds,
                linear_constraints,
                cache: RwLock::new(PirlsLruCache::new(MAX_PIRLS_CACHE_ENTRIES)),
                faer_factor_cache: RwLock::new(HashMap::new()),
                pirls_cache_enabled: AtomicBool::new(true),
                current_eval_bundle: RwLock::new(None),
                cost_last: RwLock::new(None),
                cost_repeat: RwLock::new(0),
                cost_last_emit: RwLock::new(0),
                cost_eval_count: RwLock::new(0),
                raw_cond_snapshot: RwLock::new(f64::NAN),
                gaussian_cond_snapshot: RwLock::new(f64::NAN),
                workspace: Mutex::new(workspace),
                warm_start_beta: RwLock::new(None),
                warm_start_enabled: AtomicBool::new(true),
            })
        }

        /// Creates a sanitized cache key from rho values.
        /// Returns None if any component is NaN, in which case caching is skipped.
        /// Maps -0.0 to 0.0 to ensure consistency in caching.
        fn rho_key_sanitized(&self, rho: &Array1<f64>) -> Option<Vec<u64>> {
            let mut key = Vec::with_capacity(rho.len());
            for &v in rho.iter() {
                if v.is_nan() {
                    return None; // Don't cache NaN values
                }
                if v == 0.0 {
                    // This handles both +0.0 and -0.0
                    key.push(0.0f64.to_bits());
                } else {
                    key.push(v.to_bits());
                }
            }
            Some(key)
        }

        fn prepare_eval_bundle_with_key(
            &self,
            rho: &Array1<f64>,
            key: Option<Vec<u64>>,
        ) -> Result<EvalShared, EstimationError> {
            let pirls_result = self.execute_pirls_if_needed(rho)?;
            let (h_eff, ridge_passport) = self.effective_hessian(pirls_result.as_ref())?;

            // Spectral consistency threshold for eigenvalue truncation.
            //
            // Root-cause fix:
            // An absolute cutoff is scale-dependent and can misclassify near-null
            // modes when ||H|| varies by orders of magnitude. Use a relative rule
            // anchored to the dominant eigenvalue so pseudoinverse support and
            // log|H|_+ are stable across problem scales.
            const EIG_REL_THRESHOLD: f64 = 1e-10;
            const EIG_ABS_FLOOR: f64 = 1e-14;

            let dim = h_eff.nrows();

            // Compute spectral quantities from the same curvature used by inner PIRLS.
            // This path stays on H_eff for cost/gradient consistency.
            let h_total = h_eff.clone();
            let (eigvals, eigvecs) = h_total
                .eigh(Side::Lower)
                .map_err(|e| EstimationError::EigendecompositionFailed(e))?;
            let max_eig = eigvals.iter().copied().fold(0.0_f64, f64::max);
            // Non-Gaussian outer gradients consume trace(H^{-1} H_k) terms that are
            // sensitive to dropped low modes. Since PIRLS already stabilizes H with a
            // structural ridge, keep the full stabilized spectrum for these families.
            let eig_threshold = if self.config.link_function() == LinkFunction::Identity {
                (max_eig * EIG_REL_THRESHOLD).max(EIG_ABS_FLOOR)
            } else {
                EIG_ABS_FLOOR
            };

            // Positive-part Hessian log-determinant convention:
            //   log|H|_+ = Σ_{λ_i(H) > τ} log λ_i(H),
            // where τ is a relative+absolute cutoff tied to spectrum scale.
            //
            // This avoids unstable rank flips from tiny signed eigenvalues and keeps
            // logdet/traces/pseudoinverse operations on the same effective subspace.
            let h_total_log_det: f64 = eigvals
                .iter()
                .filter(|&&v| v > eig_threshold)
                .map(|&v| v.ln())
                .sum();

            if !h_total_log_det.is_finite() {
                return Err(EstimationError::ModelIsIllConditioned {
                    condition_number: f64::INFINITY,
                });
            }

            // Build factor W for the Moore-Penrose pseudoinverse on the kept subspace:
            //   H_+^† = U_+ diag(1/λ_+) U_+ᵀ = W Wᵀ,
            //   W := U_+ diag(1/sqrt(λ_+)).
            //
            // Later trace terms use this representation directly, e.g.
            //   tr(H_+^† S_k) = ||R_k W||_F^2
            // without materializing H_+^† as a dense matrix.
            let valid_indices: Vec<usize> = eigvals
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| if v > eig_threshold { Some(i) } else { None })
                .collect();

            let valid_count = valid_indices.len();
            let mut w = Array2::<f64>::zeros((dim, valid_count));

            for (w_col_idx, &eig_idx) in valid_indices.iter().enumerate() {
                let val = eigvals[eig_idx];
                let scale = 1.0 / val.sqrt();
                let u_col = eigvecs.column(eig_idx);

                let mut w_col = w.column_mut(w_col_idx);
                Zip::from(&mut w_col)
                    .and(&u_col)
                    .for_each(|w_elem, &u_elem| {
                        *w_elem = u_elem * scale;
                    });
            }

            Ok(EvalShared {
                key,
                pirls_result,
                h_eff: Arc::new(h_eff),
                ridge_passport,
                h_total: Arc::new(h_total),
                h_pos_factor_w: Arc::new(w),
                h_total_log_det,
            })
        }

        fn obtain_eval_bundle(&self, rho: &Array1<f64>) -> Result<EvalShared, EstimationError> {
            let key = self.rho_key_sanitized(rho);
            if let Some(existing) = self.current_eval_bundle.read().unwrap().as_ref()
                && existing.matches(&key)
            {
                return Ok(existing.clone());
            }
            let bundle = self.prepare_eval_bundle_with_key(rho, key)?;
            *self.current_eval_bundle.write().unwrap() = Some(bundle.clone());
            Ok(bundle)
        }

        pub(super) fn last_ridge_used(&self) -> Option<f64> {
            self.current_eval_bundle
                .read()
                .unwrap()
                .as_ref()
                .map(|bundle| bundle.ridge_passport.delta)
        }

        /// Calculate effective degrees of freedom (EDF) using a consistent approach
        /// for both cost and gradient calculations, ensuring identical values.
        ///
        /// # Arguments
        /// * `pr` - PIRLS result containing the penalty matrices
        /// * `lambdas` - Smoothing parameters (lambda values)
        /// * `h_eff` - Effective Hessian matrix
        ///
        /// # Returns
        /// * Effective degrees of freedom value
        fn edf_from_h_and_e(
            &self,
            e_transformed: &Array2<f64>, // rank x p_eff
            lambdas: ArrayView1<'_, f64>,
            h_eff: &Array2<f64>, // p_eff x p_eff
        ) -> Result<f64, EstimationError> {
            // Why caching by ρ is sound:
            // The effective degrees of freedom (EDF) calculation is one of only two places where
            // we ask for a Faer factorization through `get_faer_factor`.  The cache inside that
            // helper uses only the vector of log smoothing parameters (ρ) as the key.  At first
            // glance that can look risky—two different Hessians with the same ρ might appear to be
            // conflated.  The surrounding call graph prevents that situation:
            //   • Identity / Gaussian models call `edf_from_h_and_rk` with the stabilized Hessian
            //     `pirls_result.stabilized_hessian_transformed`.
            //   • Non-Gaussian (logit / LAML) models call it with the effective / ridged Hessian
            //     returned by `effective_hessian(pr)`.
            // Within a given `RemlState` we never switch between those two flavours—the state is
            // constructed for a single link function, so the cost/gradient pathways stay aligned.
            // Because of that design, a given ρ vector corresponds to exactly one Hessian type in
            // practice, and the cache cannot hand back a factorization of an unintended matrix.

            // Prefer an un-ridged factorization when the stabilized Hessian is already PD.
            // Only fall back to the RidgePlanner path if direct factorization fails.
            let rho_like = lambdas.mapv(|lam| lam.ln());
            let factor = {
                let h_view = FaerArrayView::new(h_eff);
                if let Ok(f) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
                    Arc::new(FaerFactor::Llt(f))
                } else if let Ok(f) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
                    Arc::new(FaerFactor::Ldlt(f))
                } else {
                    self.get_faer_factor(&rho_like, h_eff)
                }
            };

            // Use the single λ-weighted penalty root E for S_λ = Eᵀ E to compute
            // trace(H⁻¹ S_λ) = ⟨H⁻¹ Eᵀ, Eᵀ⟩_F directly.
            let e_t = e_transformed.t().to_owned(); // (p_eff × rank_total)
            let e_view = FaerArrayView::new(&e_t);
            let x = factor.solve(e_view.as_ref());
            let trace_h_inv_s_lambda = faer_frob_inner(x.as_ref(), e_view.as_ref());

            // Calculate EDF as p - trace, clamped to the penalty nullspace dimension
            let p = h_eff.ncols() as f64;
            let rank_s = e_transformed.nrows() as f64;
            let mp = (p - rank_s).max(0.0);
            let edf = (p - trace_h_inv_s_lambda).clamp(mp, p);

            Ok(edf)
        }

        fn active_constraint_free_basis(&self, pr: &PirlsResult) -> Option<Array2<f64>> {
            let lin = pr.linear_constraints_transformed.as_ref()?;
            let active_tol = 1e-8;
            let beta_t = pr.beta_transformed.as_ref();
            let mut active_rows: Vec<Array1<f64>> = Vec::new();
            for i in 0..lin.a.nrows() {
                let slack = lin.a.row(i).dot(beta_t) - lin.b[i];
                if slack <= active_tol {
                    active_rows.push(lin.a.row(i).to_owned());
                }
            }
            if active_rows.is_empty() {
                return None;
            }

            let p_t = lin.a.ncols();
            let mut a_t = Array2::<f64>::zeros((p_t, active_rows.len()));
            for (j, row) in active_rows.iter().enumerate() {
                for k in 0..p_t {
                    a_t[[k, j]] = row[k];
                }
            }

            let q_row = Self::orthonormalize_columns(&a_t, 1e-10); // basis for active row-space^T
            let rank = q_row.ncols();
            if rank == 0 {
                return None;
            }
            if rank >= p_t {
                return Some(Array2::<f64>::zeros((p_t, 0)));
            }

            // Build orthonormal basis for null(A_active) as complement of row-space.
            let mut z = Array2::<f64>::zeros((p_t, p_t - rank));
            let mut kept = 0usize;
            for j in 0..p_t {
                let mut v = Array1::<f64>::zeros(p_t);
                v[j] = 1.0;
                for t in 0..rank {
                    let qt = q_row.column(t);
                    let proj = qt.dot(&v);
                    v -= &qt.mapv(|x| x * proj);
                }
                for t in 0..kept {
                    let zt = z.column(t);
                    let proj = zt.dot(&v);
                    v -= &zt.mapv(|x| x * proj);
                }
                let nrm = v.dot(&v).sqrt();
                if nrm > 1e-10 {
                    z.column_mut(kept).assign(&v.mapv(|x| x / nrm));
                    kept += 1;
                    if kept == p_t - rank {
                        break;
                    }
                }
            }
            Some(z.slice(ndarray::s![.., 0..kept]).to_owned())
        }

        fn enforce_constraint_kkt(&self, pr: &PirlsResult) -> Result<(), EstimationError> {
            let Some(kkt) = pr.constraint_kkt.as_ref() else {
                return Ok(());
            };
            let tol_primal = 1e-7;
            let tol_dual = 1e-7;
            let tol_comp = 1e-7;
            let tol_stat = 5e-6;
            if kkt.primal_feasibility > tol_primal
                || kkt.dual_feasibility > tol_dual
                || kkt.complementarity > tol_comp
                || kkt.stationarity > tol_stat
            {
                let mut worst_row_msg = String::new();
                if let Some(lin) = pr.linear_constraints_transformed.as_ref() {
                    let mut worst = 0.0_f64;
                    let mut worst_row = 0usize;
                    for i in 0..lin.a.nrows() {
                        let slack = lin.a.row(i).dot(&pr.beta_transformed.0) - lin.b[i];
                        let viol = (-slack).max(0.0);
                        if viol > worst {
                            worst = viol;
                            worst_row = i;
                        }
                    }
                    if worst > 0.0 {
                        worst_row_msg =
                            format!("; worst_row={} worst_violation={:.3e}", worst_row, worst);
                    }
                }
                return Err(EstimationError::ParameterConstraintViolation(format!(
                    "KKT residuals exceed tolerance: primal={:.3e}, dual={:.3e}, comp={:.3e}, stat={:.3e}; active={}/{}{}",
                    kkt.primal_feasibility,
                    kkt.dual_feasibility,
                    kkt.complementarity,
                    kkt.stationarity,
                    kkt.n_active,
                    kkt.n_constraints,
                    worst_row_msg
                )));
            }
            Ok(())
        }

        fn project_with_basis(matrix: &Array2<f64>, z: &Array2<f64>) -> Array2<f64> {
            let zt_m = z.t().dot(matrix);
            zt_m.dot(z)
        }

        fn fixed_subspace_penalty_rank_and_logdet(
            &self,
            e_transformed: &Array2<f64>,
            ridge_passport: RidgePassport,
        ) -> Result<(usize, f64), EstimationError> {
            let structural_rank = e_transformed.nrows().min(e_transformed.ncols());
            if structural_rank == 0 {
                return Ok((0, 0.0));
            }

            // Keep objective rank fixed to the structural penalty rank to avoid
            // rho-dependent rank flips from tiny eigenvalue jitter.
            let mut s_lambda = e_transformed.t().dot(e_transformed);
            let ridge = ridge_passport.penalty_logdet_ridge();
            if ridge > 0.0 {
                for i in 0..s_lambda.nrows() {
                    s_lambda[[i, i]] += ridge;
                }
            }
            let (evals, _) = s_lambda
                .eigh(Side::Lower)
                .map_err(EstimationError::EigendecompositionFailed)?;
            let mut order: Vec<usize> = (0..evals.len()).collect();
            order.sort_by(|&a, &b| {
                evals[b]
                    .partial_cmp(&evals[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });

            let max_ev = order
                .first()
                .map(|&idx| evals[idx].abs())
                .unwrap_or(1.0)
                .max(1.0);
            let floor = (1e-12 * max_ev).max(1e-12);
            let log_det = order
                .iter()
                .take(structural_rank)
                .map(|&idx| evals[idx].max(floor).ln())
                .sum();
            Ok((structural_rank, log_det))
        }

        fn update_warm_start_from(&self, pr: &PirlsResult) {
            if !self.warm_start_enabled.load(Ordering::Relaxed) {
                return;
            }
            match pr.status {
                pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                    let beta_original = pr.reparam_result.qs.dot(pr.beta_transformed.as_ref());
                    self.warm_start_beta
                        .write()
                        .unwrap()
                        .replace(Coefficients::new(beta_original));
                }
                _ => {
                    self.warm_start_beta.write().unwrap().take();
                }
            }
        }

        /// Clear warm-start state. Used in tests to ensure consistent starting points
        /// when comparing different gradient computation paths.
        #[cfg(test)]
        #[allow(dead_code)]
        pub fn clear_warm_start(&self) {
            self.warm_start_beta.write().unwrap().take();
            self.current_eval_bundle.write().unwrap().take();
        }

        /// Returns the per-penalty square-root matrices in the transformed coefficient basis
        /// without any λ weighting. Each returned R_k satisfies S_k = R_kᵀ R_k in that basis.
        /// Using these avoids accidental double counting of λ when forming derivatives.
        ///
        /// # Arguments
        /// * `pr` - The PIRLS result with the transformation matrix Qs
        ///
        /// # Returns
        fn factorize_faer(&self, h: &Array2<f64>) -> FaerFactor {
            let mut planner = RidgePlanner::new(h);
            loop {
                let ridge = planner.ridge();
                if ridge > 0.0 {
                    let regularized = add_ridge(h, ridge);
                    let view = FaerArrayView::new(&regularized);
                    if let Ok(f) = FaerLlt::new(view.as_ref(), Side::Lower) {
                        return FaerFactor::Llt(f);
                    }
                    if let Ok(f) = FaerLdlt::new(view.as_ref(), Side::Lower) {
                        return FaerFactor::Ldlt(f);
                    }
                    if planner.attempts() >= MAX_FACTORIZATION_ATTEMPTS {
                        let f = FaerLblt::new(view.as_ref(), Side::Lower);
                        return FaerFactor::Lblt(f);
                    }
                } else {
                    let h_view = FaerArrayView::new(h);
                    if let Ok(f) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
                        return FaerFactor::Llt(f);
                    }
                    if let Ok(f) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
                        return FaerFactor::Ldlt(f);
                    }
                }
                planner.bump_with_matrix(h);
            }
        }

        fn get_faer_factor(&self, rho: &Array1<f64>, h: &Array2<f64>) -> Arc<FaerFactor> {
            // Cache strategy: ρ alone is the key.
            // The cache deliberately ignores which Hessian matrix we are factoring.  Today this is
            // sound because every caller obeys a single rule:
            //   • Identity/Gaussian REML cost & gradient only ever request factors of the
            //     stabilized Hessian.
            //   • Non-Gaussian (logit/LAML) cost and gradient request factors of the effective/ridged Hessian.
            // Consequently each ρ corresponds to exactly one matrix within the lifetime of a
            // `RemlState`, so returning the cached factorization is correct.
            // This design is still brittle: adding a new code path that calls `get_faer_factor`
            // with a different H for the same ρ would silently reuse the wrong factor.  If such a
            // path ever appears, extend the key (for example by tagging the Hessian variant) or
            // split the cache.  The current key maximizes cache
            // hits across repeated EDF/gradient evaluations for the same smoothing parameters.
            let key_opt = self.rho_key_sanitized(rho);
            if let Some(key) = &key_opt
                && let Some(f) = self.faer_factor_cache.read().unwrap().get(key)
            {
                return Arc::clone(f);
            }
            let fact = Arc::new(self.factorize_faer(h));

            if let Some(key) = key_opt {
                let mut cache = self.faer_factor_cache.write().unwrap();
                if cache.len() > 64 {
                    cache.clear();
                }
                cache.insert(key, Arc::clone(&fact));
            }
            fact
        }

        /// Numerical gradient of the penalized log-likelihood part w.r.t. rho via central differences.
        /// Returns g_pll where g_pll[k] = - d/d rho_k penalised_ll(rho), suitable for COST gradient assembly.
        #[cfg(test)]
        #[allow(dead_code)]
        fn numeric_penalised_ll_grad(
            &self,
            rho: &Array1<f64>,
        ) -> Result<Array1<f64>, EstimationError> {
            let mut workspace = self.workspace.lock().unwrap();
            self.numeric_penalised_ll_grad_with_workspace(rho, &mut workspace)
        }

        fn numeric_penalised_ll_grad_with_workspace(
            &self,
            rho: &Array1<f64>,
            workspace: &mut RemlWorkspace,
        ) -> Result<Array1<f64>, EstimationError> {
            let len = rho.len();
            if len == 0 {
                return Ok(Array1::zeros(0));
            }

            let x = &self.x;
            let offset_view = self.offset.view();
            let y = self.y;
            let weights = self.weights;
            let rs_list = &self.rs_list;
            let p_dim = self.p;
            let config = self.config;
            let firth_bias = config.firth_bias_reduction;
            let link_is_logit = matches!(config.link_function(), LinkFunction::Logit);
            let balanced_root = &self.balanced_penalty_root;
            let reparam_invariant = &self.reparam_invariant;

            // Capture the current best beta to warm-start the gradient probes.
            // This is crucial for stability: if we start from zero, P-IRLS might converge
            // to a different local optimum (or stall differently) than the main cost evaluation,
            // creating huge phantom gradients that violate the envelope theorem.
            let warm_start_initial = if self.warm_start_enabled.load(Ordering::Relaxed) {
                self.warm_start_beta.read().unwrap().clone()
            } else {
                None
            };

            // Run a fresh PIRLS solve for each perturbed smoothing vector.  We avoid the
            // `execute_pirls_if_needed` cache here because these evaluations happen in parallel
            // and never reuse the same ρ, so the cache would not help and would require
            // synchronization across threads.
            let evaluate_penalised_ll = |rho_vec: &Array1<f64>| -> Result<f64, EstimationError> {
                let (pirls_result, _) = pirls::fit_model_for_fixed_rho_matrix(
                    LogSmoothingParamsView::new(rho_vec.view()),
                    x,
                    offset_view,
                    y,
                    weights,
                    rs_list,
                    Some(balanced_root),
                    Some(reparam_invariant),
                    p_dim,
                    &config.as_pirls_config(),
                    warm_start_initial.as_ref(),
                    self.coefficient_lower_bounds.as_ref(),
                    self.linear_constraints.as_ref(),
                    None, // No SE for base model
                )?;
                self.enforce_constraint_kkt(&pirls_result)?;

                match pirls_result.status {
                    pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                        let penalty = pirls_result.stable_penalty_term;
                        let mut penalised = -0.5 * pirls_result.deviance - 0.5 * penalty;
                        // Include Firth log-det term in LAML for consistency with inner PIRLS
                        if firth_bias && link_is_logit {
                            if let Some(firth_log_det) = pirls_result.firth_log_det {
                                penalised += firth_log_det; // Jeffreys prior contribution
                            }
                        }
                        Ok(penalised)
                    }
                    pirls::PirlsStatus::Unstable => {
                        Err(EstimationError::PerfectSeparationDetected {
                            iteration: pirls_result.iteration,
                            max_abs_eta: pirls_result.max_abs_eta,
                        })
                    }
                    pirls::PirlsStatus::MaxIterationsReached => {
                        if pirls_result.last_gradient_norm > 1.0 {
                            Err(EstimationError::PirlsDidNotConverge {
                                max_iterations: pirls_result.iteration,
                                last_change: pirls_result.last_gradient_norm,
                            })
                        } else {
                            let penalty = pirls_result.stable_penalty_term;
                            let mut penalised = -0.5 * pirls_result.deviance - 0.5 * penalty;
                            // Include Firth log-det term in LAML for consistency with inner PIRLS
                            if firth_bias && link_is_logit {
                                if let Some(firth_log_det) = pirls_result.firth_log_det {
                                    penalised += firth_log_det; // Jeffreys prior contribution
                                }
                            }
                            Ok(penalised)
                        }
                    }
                }
            };

            let grad_values = (0..len)
                .into_par_iter()
                .map(|k| -> Result<f64, EstimationError> {
                    let h_rel = 1e-4_f64 * (1.0 + rho[k].abs());
                    let h_abs = 1e-5_f64;
                    let h = h_rel.max(h_abs);

                    let mut rho_plus = rho.clone();
                    rho_plus[k] += 0.5 * h;
                    let mut rho_minus = rho.clone();
                    rho_minus[k] -= 0.5 * h;

                    let fp = evaluate_penalised_ll(&rho_plus)?;
                    let fm = evaluate_penalised_ll(&rho_minus)?;
                    Ok(-(fp - fm) / h)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let grad_array = Array1::from_vec(grad_values);
            let mut g_view = workspace.grad_secondary.slice_mut(s![..len]);
            g_view.assign(&grad_array);

            Ok(grad_array)
        }

        /// Compute 0.5 * log|H_eff(rho)| using the SAME stabilized Hessian and logdet path as compute_cost.
        fn half_logh_at(&self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
            let pr = self.execute_pirls_if_needed(rho)?;
            let (h_eff, _) = self.effective_hessian(&pr)?;
            let chol = h_eff.clone().cholesky(Side::Lower).map_err(|_| {
                let min_eig = h_eff
                    .clone()
                    .eigh(Side::Lower)
                    .ok()
                    .and_then(|(eigs, _)| eigs.iter().cloned().reduce(f64::min))
                    .unwrap_or(f64::NAN);
                EstimationError::HessianNotPositiveDefinite {
                    min_eigenvalue: min_eig,
                }
            })?;
            let log_det_h = 2.0 * chol.diag().mapv(f64::ln).sum();
            Ok(0.5 * log_det_h)
        }

        /// Numerical gradient of 0.5 * log|H_eff(rho)| with respect to rho via central differences.
        fn numeric_half_logh_grad_with_workspace(
            &self,
            rho: &Array1<f64>,
            workspace: &mut RemlWorkspace,
        ) -> Result<Array1<f64>, EstimationError> {
            let len = rho.len();
            if len == 0 {
                return Ok(Array1::zeros(0));
            }

            let mut g_view = workspace.grad_primary.slice_mut(s![..len]);
            g_view.fill(0.0);

            for k in 0..len {
                let h_rel = 1e-4_f64 * (1.0 + rho[k].abs());
                let h_abs = 1e-5_f64;
                let h = h_rel.max(h_abs);

                workspace.rho_plus.assign(rho);
                workspace.rho_plus[k] += 0.5 * h;
                workspace.rho_minus.assign(rho);
                workspace.rho_minus[k] -= 0.5 * h;

                let fp = self.half_logh_at(&workspace.rho_plus)?;
                let fm = self.half_logh_at(&workspace.rho_minus)?;
                g_view[k] = (fp - fm) / h;
            }

            Ok(g_view.to_owned())
        }

        const MIN_DMU_DETA: f64 = 1e-6;

        // Accessor methods for private fields
        pub(super) fn x(&self) -> &DesignMatrix {
            &self.x
        }

        #[allow(dead_code)]
        pub(super) fn y(&self) -> ArrayView1<'a, f64> {
            self.y
        }

        #[allow(dead_code)]
        pub(super) fn rs_list_ref(&self) -> &Vec<Array2<f64>> {
            &self.rs_list
        }

        pub(super) fn balanced_penalty_root(&self) -> &Array2<f64> {
            &self.balanced_penalty_root
        }

        pub(super) fn weights(&self) -> ArrayView1<'a, f64> {
            self.weights
        }

        #[allow(dead_code)]
        pub(super) fn offset(&self) -> ArrayView1<'_, f64> {
            self.offset.view()
        }

        /// Runs the inner P-IRLS loop, caching the result.
        fn execute_pirls_if_needed(
            &self,
            rho: &Array1<f64>,
        ) -> Result<Arc<PirlsResult>, EstimationError> {
            let use_cache = self.pirls_cache_enabled.load(Ordering::Relaxed);
            // Use sanitized key to handle NaN and -0.0 vs 0.0 issues
            let key_opt = self.rho_key_sanitized(rho);
            if use_cache
                && let Some(key) = &key_opt
                && let Some(cached) = self.cache.write().unwrap().get(key)
            {
                if self.warm_start_enabled.load(Ordering::Relaxed) {
                    self.update_warm_start_from(cached.as_ref());
                }
                return Ok(cached);
            }

            // Run P-IRLS with original matrices to perform fresh reparameterization
            // The returned result will include the transformation matrix qs
            let pirls_result = {
                let warm_start_holder = self.warm_start_beta.read().unwrap();
                let warm_start_ref = if self.warm_start_enabled.load(Ordering::Relaxed) {
                    warm_start_holder.as_ref()
                } else {
                    None
                };
                pirls::fit_model_for_fixed_rho_matrix(
                    LogSmoothingParamsView::new(rho.view()),
                    &self.x,
                    self.offset.view(),
                    self.y,
                    self.weights,
                    &self.rs_list,
                    Some(&self.balanced_penalty_root),
                    Some(&self.reparam_invariant),
                    self.p,
                    &self.config.as_pirls_config(),
                    warm_start_ref,
                    self.coefficient_lower_bounds.as_ref(),
                    self.linear_constraints.as_ref(),
                    None, // No SE for base model
                )
            };

            if let Err(e) = &pirls_result {
                println!("[GAM COST]   -> P-IRLS INNER LOOP FAILED. Error: {e:?}");
                if self.warm_start_enabled.load(Ordering::Relaxed) {
                    self.warm_start_beta.write().unwrap().take();
                }
            }

            let (pirls_result, _) = pirls_result?; // Propagate error if it occurred
            let pirls_result = Arc::new(pirls_result);
            self.enforce_constraint_kkt(pirls_result.as_ref())?;

            // Check the status returned by the P-IRLS routine.
            match pirls_result.status {
                pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                    self.update_warm_start_from(pirls_result.as_ref());
                    // This is a successful fit. Cache only if key is valid (not NaN).
                    if use_cache && let Some(key) = key_opt {
                        self.cache
                            .write()
                            .unwrap()
                            .insert(key, Arc::clone(&pirls_result));
                    }
                    Ok(pirls_result)
                }
                pirls::PirlsStatus::Unstable => {
                    if self.warm_start_enabled.load(Ordering::Relaxed) {
                        self.warm_start_beta.write().unwrap().take();
                    }
                    // The fit was unstable. This is where we throw our specific, user-friendly error.
                    // Pass the diagnostic info into the error
                    Err(EstimationError::PerfectSeparationDetected {
                        iteration: pirls_result.iteration,
                        max_abs_eta: pirls_result.max_abs_eta,
                    })
                }
                pirls::PirlsStatus::MaxIterationsReached => {
                    if self.warm_start_enabled.load(Ordering::Relaxed) {
                        self.warm_start_beta.write().unwrap().take();
                    }
                    if pirls_result.last_gradient_norm > 1.0 {
                        // The fit timed out and gradient is large.
                        log::error!(
                            "P-IRLS failed convergence check: gradient norm {} > 1.0 (iter {})",
                            pirls_result.last_gradient_norm,
                            pirls_result.iteration
                        );
                        Err(EstimationError::PirlsDidNotConverge {
                            max_iterations: pirls_result.iteration,
                            last_change: pirls_result.last_gradient_norm,
                        })
                    } else {
                        // Gradient is acceptable, treat as converged but with warning if needed
                        log::warn!(
                            "P-IRLS reached max iterations but gradient norm {:.3e} is acceptable.",
                            pirls_result.last_gradient_norm
                        );
                        Ok(pirls_result)
                    }
                }
            }
        }
    }
    impl<'a> RemlState<'a> {
        /// Compute the objective function for BFGS optimization.
        ///
        /// FULL OBJECTIVE REFERENCE
        /// ------------------------
        /// This function returns the scalar outer cost minimized over ρ.
        ///
        /// Non-Gaussian branch (negative LAML form used by optimizer):
        ///   V_LAML(ρ) =
        ///      -ℓ(β̂(ρ))
        ///      + 0.5 β̂(ρ)ᵀ S(ρ) β̂(ρ)
        ///      + 0.5 log|H(ρ)|
        ///      - 0.5 log|S(ρ)|_+
        ///      + const
        ///
        /// where:
        ///   S(ρ) = Σ_k exp(ρ_k) S_k + δI
        ///   H(ρ) = -∇²ℓ(β̂(ρ)) + S(ρ)
        ///
        /// Gaussian identity-link REML branch:
        ///   V_REML(ρ, φ) =
        ///      D_p(ρ)/(2φ)
        ///      + (n_r/2) log φ
        ///      + 0.5 log|H(ρ)|
        ///      - 0.5 log|S(ρ)|_+
        ///      + const
        ///
        /// with profiled φ:
        ///   φ̂(ρ) = D_p(ρ)/n_r
        ///   V_REML,prof(ρ) =
        ///      (n_r/2) log D_p(ρ)
        ///      + 0.5 log|H(ρ)|
        ///      - 0.5 log|S(ρ)|_+
        ///      + const.
        ///
        /// Consistency rule enforced throughout:
        ///   The same stabilized matrices/factorizations are used for
        ///   objective and gradient/Hessian terms. Mixing different H/S variants
        ///   causes deterministic gradient mismatch and unstable outer optimization.
        ///
        /// Determinant conventions:
        ///   - log|H| may use positive-part/stabilized spectrum conventions when needed.
        ///   - log|S|_+ follows fixed-rank pseudo-determinant conventions in the
        ///     transformed penalty basis, optionally including ridge policy.
        /// These conventions are mirrored in gradient code via corresponding trace terms.
        pub fn compute_cost(&self, p: &Array1<f64>) -> Result<f64, EstimationError> {
            let cost_call_idx = {
                let mut calls = self.cost_eval_count.write().unwrap();
                *calls += 1;
                *calls
            };
            let bundle = match self.obtain_eval_bundle(p) {
                Ok(bundle) => bundle,
                Err(EstimationError::ModelIsIllConditioned { .. }) => {
                    self.current_eval_bundle.write().unwrap().take();
                    // Inner linear algebra says "too singular" — treat as barrier.
                    log::warn!(
                        "P-IRLS flagged ill-conditioning for current rho; returning +inf cost to retreat."
                    );
                    // Diagnostics: which rho are at bounds
                    let at_lower: Vec<usize> = p
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &v)| {
                            if v <= -RHO_BOUND + 1e-8 {
                                Some(i)
                            } else {
                                None
                            }
                        })
                        .collect();
                    let at_upper: Vec<usize> = p
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &v)| if v >= RHO_BOUND - 1e-8 { Some(i) } else { None })
                        .collect();
                    if !(at_lower.is_empty() && at_upper.is_empty()) {
                        eprintln!(
                            "[Diag] rho bounds: lower={:?} upper={:?}",
                            at_lower, at_upper
                        );
                    }
                    return Ok(f64::INFINITY);
                }
                Err(e) => {
                    self.current_eval_bundle.write().unwrap().take();
                    // Other errors still bubble up
                    // Provide bounds diagnostics here too
                    let at_lower: Vec<usize> = p
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &v)| {
                            if v <= -RHO_BOUND + 1e-8 {
                                Some(i)
                            } else {
                                None
                            }
                        })
                        .collect();
                    let at_upper: Vec<usize> = p
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &v)| if v >= RHO_BOUND - 1e-8 { Some(i) } else { None })
                        .collect();
                    if !(at_lower.is_empty() && at_upper.is_empty()) {
                        eprintln!(
                            "[Diag] rho bounds: lower={:?} upper={:?}",
                            at_lower, at_upper
                        );
                    }
                    return Err(e);
                }
            };
            let pirls_result = bundle.pirls_result.as_ref();
            let ridge_used = bundle.ridge_passport.delta;

            let lambdas = p.mapv(f64::exp);
            let free_basis_opt = self.active_constraint_free_basis(pirls_result);
            let mut h_eff_eval = bundle.h_eff.as_ref().clone();
            let mut h_total_eval = bundle.h_total.as_ref().clone();
            let mut e_eval = pirls_result.reparam_result.e_transformed.clone();
            if let Some(z) = free_basis_opt.as_ref() {
                h_eff_eval = Self::project_with_basis(bundle.h_eff.as_ref(), z);
                h_total_eval = Self::project_with_basis(bundle.h_total.as_ref(), z);
                e_eval = pirls_result.reparam_result.e_transformed.dot(z);
            }
            let h_eff = &h_eff_eval;

            // Sanity check: penalty dimension consistency across lambdas, R_k, and det1.
            if !p.is_empty() {
                let k_lambda = p.len();
                let k_r = pirls_result.reparam_result.rs_transformed.len();
                let k_d = pirls_result.reparam_result.det1.len();
                if !(k_lambda == k_r && k_r == k_d) {
                    return Err(EstimationError::LayoutError(format!(
                        "Penalty dimension mismatch: lambdas={}, R={}, det1={}",
                        k_lambda, k_r, k_d
                    )));
                }
                if self.nullspace_dims.len() != k_lambda {
                    return Err(EstimationError::LayoutError(format!(
                        "Nullspace dimension mismatch: expected {} entries, got {}",
                        k_lambda,
                        self.nullspace_dims.len()
                    )));
                }
            }

            // Don't barrier on non-PD; we'll stabilize and continue like mgcv
            // Only check eigenvalues if we needed to add a ridge
            const MIN_ACCEPTABLE_HESSIAN_EIGENVALUE: f64 = 1e-12;
            let want_hot_diag = self.should_compute_hot_diagnostics(cost_call_idx);
            if ridge_used > 0.0
                && let Ok((eigs, _)) = pirls_result.penalized_hessian_transformed.eigh(Side::Lower)
                && let Some(min_eig) = eigs.iter().cloned().reduce(f64::min)
            {
                if should_emit_h_min_eig_diag(min_eig) {
                    eprintln!(
                        "[Diag] H min_eig={:.3e} (ridge={:.3e})",
                        min_eig, ridge_used
                    );
                }

                if min_eig <= 0.0 {
                    log::warn!(
                        "Penalized Hessian not PD (min eig <= 0) before stabilization; proceeding with ridge {:.3e}.",
                        ridge_used
                    );
                }

                if want_hot_diag
                    && (!min_eig.is_finite() || min_eig <= MIN_ACCEPTABLE_HESSIAN_EIGENVALUE)
                {
                    let condition_number =
                        calculate_condition_number(&pirls_result.penalized_hessian_transformed)
                            .ok()
                            .unwrap_or(f64::INFINITY);

                    log::warn!(
                        "Penalized Hessian extremely ill-conditioned (cond={:.3e}); continuing with stabilized Hessian.",
                        condition_number
                    );
                }
            }
            // Use stable penalty calculation - no need to reconstruct matrices
            // The penalty term is already calculated stably in the P-IRLS loop

            match self.config.link_function() {
                LinkFunction::Identity => {
                    let ridge_passport = pirls_result.ridge_passport;
                    // From Wood (2017), Chapter 6, Eq. 6.24:
                    // V_r(λ) = D_p/(2φ) + (r/2φ) + ½log|X'X/φ + S_λ/φ| - ½log|S_λ/φ|_+
                    // where D_p = ||y - Xβ̂||² + β̂'S_λβ̂ is the PENALIZED deviance
                    //
                    // With profiled dispersion φ̂ = D_p/(n-M_p), this becomes:
                    //   V_REML(ρ) =
                    //     D_p/(2φ̂)
                    //   + 0.5 log|H|
                    //   - 0.5 log|S|_+
                    //   + ((n-M_p)/2) log(2πφ̂),
                    // where H = XᵀW0X + S(ρ), S(ρ)=Σ_k exp(ρ_k) S_k + δI.
                    //
                    // Because Gaussian identity has c=d=0, there is no third/fourth derivative
                    // correction in H_k: ∂H/∂ρ_k = S_k^ρ exactly.

                    // Check condition number with improved thresholds per Wood (2011)
                    const MAX_CONDITION_NUMBER: f64 = 1e12;
                    if want_hot_diag {
                        let cond = pirls_result
                            .penalized_hessian_transformed
                            .eigh(Side::Lower)
                            .ok()
                            .map(|(evals, _)| {
                                let max_ev = evals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                                let min_ev = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                                if min_ev <= 1e-12 {
                                    f64::INFINITY
                                } else {
                                    max_ev / min_ev
                                }
                            })
                            .unwrap_or(f64::NAN);
                        *self.gaussian_cond_snapshot.write().unwrap() = cond;
                    }
                    let condition_number = *self.gaussian_cond_snapshot.read().unwrap();
                    if condition_number.is_finite() {
                        if condition_number > MAX_CONDITION_NUMBER {
                            log::warn!(
                                "Penalized Hessian very ill-conditioned (cond={:.2e}); proceeding despite poor conditioning.",
                                condition_number
                            );
                        } else if condition_number > 1e8 {
                            log::warn!(
                                "Penalized Hessian is ill-conditioned but proceeding: condition number = {condition_number:.2e}"
                            );
                        }
                    }

                    // STRATEGIC DESIGN DECISION: Use unweighted sample count for mgcv parity
                    // In standard WLS theory, one might use sum(weights) as effective sample size.
                    // However, mgcv deliberately uses the unweighted count 'n.true' in gam.fit3.
                    let n = self.y.len() as f64;
                    // Number of coefficients (transformed basis)

                    // Calculate PENALIZED deviance D_p = ||y - Xβ̂||² + β̂'S_λβ̂
                    let rss = pirls_result.deviance; // Unpenalized ||y - μ||²
                    // Use stable penalty term calculated in P-IRLS
                    let penalty = pirls_result.stable_penalty_term;

                    let dp = rss + penalty;

                    // Calculate EDF = p - tr((X'X + S_λ)⁻¹S_λ)
                    // Work directly in the transformed basis for efficiency and numerical stability
                    // This avoids transforming matrices back to the original basis unnecessarily
                    // Penalty roots are available in reparam_result if needed

                    // Nullspace dimension M_p is constant with respect to ρ.  Use it to profile φ
                    // following the standard REML identity φ = D_p / (n - M_p).
                    let (penalty_rank, log_det_s_plus) =
                        self.fixed_subspace_penalty_rank_and_logdet(&e_eval, ridge_passport)?;
                    let p_eff_dim = h_eff.ncols();
                    let mp = p_eff_dim.saturating_sub(penalty_rank) as f64;

                    // EDF diagnostics are expensive; compute only when diagnostics are enabled.
                    if want_hot_diag {
                        let edf = self.edf_from_h_and_e(&e_eval, lambdas.view(), h_eff)?;
                        log::debug!("[Diag] EDF total={:.3}", edf);
                        if n - edf < 1.0 {
                            log::warn!("Effective DoF exceeds samples; model may be overfit.");
                        }
                    }

                    let denom = (n - mp).max(LAML_RIDGE);
                    let (dp_c, _) = smooth_floor_dp(dp);
                    if dp < DP_FLOOR {
                        log::warn!(
                            "Penalized deviance {:.3e} fell below DP_FLOOR; clamping to maintain REML stability.",
                            dp
                        );
                    }
                    let phi = dp_c / denom;

                    // log |H| = log |X'X + S_λ + ridge I| using the single effective
                    // Hessian shared with the gradient. Ridge is already baked into h_eff.
                    //
                    // This is the same stabilized H used in compute_gradient;
                    // otherwise the chain-rule pieces and determinant pieces are taken on
                    // different objective surfaces and the optimizer sees inconsistent derivatives.
                    let h_for_det = h_eff.clone();

                    let chol = h_for_det.cholesky(Side::Lower).map_err(|_| {
                        let min_eig = h_eff
                            .clone()
                            .eigh(Side::Lower)
                            .ok()
                            .and_then(|(eigs, _)| eigs.iter().cloned().reduce(f64::min))
                            .unwrap_or(f64::NAN);
                        EstimationError::HessianNotPositiveDefinite {
                            min_eigenvalue: min_eig,
                        }
                    })?;
                    let log_det_h = 2.0 * chol.diag().mapv(f64::ln).sum();

                    // log |S_λ + ridge I|_+ (pseudo-determinant) to match the
                    // stabilized penalty used by PIRLS.
                    //
                    // Fixed-rank rule: unpenalized/null directions do not contribute to the
                    // pseudo-logdet. This keeps the objective continuous in ρ when S is singular
                    // (or near-singular before ridge augmentation).
                    // Standard REML expression from Wood (2017), Section 6.5.1
                    // V = (n/2)log(2πσ²) + D_p/(2σ²) + ½log|H| - ½log|S_λ|_+ + (M_p-1)/2 log(2πσ²)
                    // Simplifying: V = D_p/(2φ) + ½log|H| - ½log|S_λ|_+ + ((n-M_p)/2) log(2πφ)
                    let reml = dp_c / (2.0 * phi)
                        + 0.5 * (log_det_h - log_det_s_plus)
                        + ((n - mp) / 2.0) * (2.0 * std::f64::consts::PI * phi).ln();

                    let prior_cost = self.compute_soft_prior_cost(p);

                    Ok(reml + prior_cost)
                }
                _ => {
                    // For non-Gaussian GLMs, use the LAML approximation
                    // Note: Deviance = -2 * log-likelihood + C. So -0.5 * Deviance = log-likelihood - C/2.
                    // Use stable penalty term calculated in P-IRLS
                    let mut penalised_ll =
                        -0.5 * pirls_result.deviance - 0.5 * pirls_result.stable_penalty_term;

                    let ridge_passport = pirls_result.ridge_passport;
                    // Include Firth log-det term in LAML for consistency with inner PIRLS
                    if self.config.firth_bias_reduction
                        && matches!(self.config.link_function(), LinkFunction::Logit)
                    {
                        if let Some(firth_log_det) = pirls_result.firth_log_det {
                            penalised_ll += firth_log_det; // Jeffreys prior contribution
                        }
                    }

                    // Use the stabilized log|Sλ|_+ from the reparameterization (consistent with gradient)
                    let (_penalty_rank, log_det_s) =
                        self.fixed_subspace_penalty_rank_and_logdet(&e_eval, ridge_passport)?;

                    // Log-determinant of the effective Hessian.
                    // HESSIAN PASSPORT: Use the pre-computed h_total and its factorization
                    // from the bundle to ensure exact consistency with gradient computation.
                    // For Firth: h_total = h_eff - h_phi (computed in prepare_eval_bundle)
                    // For non-Firth: h_total = h_eff
                    //
                    // LAML objective:
                    //   V_LAML(ρ) =
                    //      -ℓ(β̂) + 0.5 β̂ᵀSβ̂
                    //    - 0.5 log|S|_+
                    //    + 0.5 log|H|
                    //    + const.
                    //
                    // For non-Gaussian families, H depends on ρ both directly through S and
                    // indirectly through β̂(ρ), which induces the dH/dρ_k third-derivative term in
                    // the exact gradient path (documented in compute_gradient).
                    let log_det_h = if free_basis_opt.is_some() {
                        if h_total_eval.nrows() == 0 {
                            0.0
                        } else {
                            let (evals, _) = h_total_eval
                                .eigh(Side::Lower)
                                .map_err(EstimationError::EigendecompositionFailed)?;
                            let floor = 1e-10;
                            evals.iter().filter(|&&v| v > floor).map(|&v| v.ln()).sum()
                        }
                    } else {
                        bundle.h_total_log_det
                    };

                    // Mp is null space dimension (number of unpenalized coefficients)
                    // For logit, scale parameter is typically fixed at 1.0, but include for completeness
                    let phi = 1.0; // Logit family typically has dispersion parameter = 1

                    // Compute null space dimension using the TRANSFORMED, STABLE basis
                    // Use the rank of the lambda-weighted transformed penalty root (e_transformed)
                    // to determine M_p with the transformed penalty basis.
                    let (penalty_rank, _) =
                        self.fixed_subspace_penalty_rank_and_logdet(&e_eval, ridge_passport)?;
                    let p_eff_dim = h_eff.ncols();
                    let mp = p_eff_dim.saturating_sub(penalty_rank) as f64;

                    let laml = penalised_ll + 0.5 * log_det_s - 0.5 * log_det_h
                        + (mp / 2.0) * (2.0 * std::f64::consts::PI * phi).ln();

                    // Diagnostics below are expensive and not needed for objective value.
                    let (edf, trace_h_inv_s_lambda, stab_cond) = if want_hot_diag {
                        let p_eff = h_eff.ncols() as f64;
                        let edf = self.edf_from_h_and_e(&e_eval, lambdas.view(), h_eff)?;
                        let trace_h_inv_s_lambda = (p_eff - edf).max(0.0);
                        let stab_cond = pirls_result
                            .penalized_hessian_transformed
                            .eigh(Side::Lower)
                            .ok()
                            .map(|(evals, _)| {
                                let min = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                                let max = evals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                                max / min.max(1e-12)
                            })
                            .unwrap_or(f64::NAN);
                        (edf, trace_h_inv_s_lambda, stab_cond)
                    } else {
                        (f64::NAN, f64::NAN, f64::NAN)
                    };

                    // Raw-condition diagnostics are rate-limited in this loop.
                    // We only refresh occasionally, and keep the last snapshot otherwise.
                    let raw_cond = if matches!(self.x(), DesignMatrix::Dense(_)) && want_hot_diag {
                        let x_orig_arc = self.x().to_dense_arc();
                        let x_orig = x_orig_arc.as_ref();
                        let w_orig = self.weights();
                        let sqrt_w = w_orig.mapv(|w| w.max(0.0).sqrt());
                        let wx = x_orig * &sqrt_w.insert_axis(Axis(1));
                        let mut h_raw = fast_ata(&wx);
                        for (k, &lambda) in lambdas.iter().enumerate() {
                            let s_k = &self.s_full_list[k];
                            if lambda != 0.0 {
                                h_raw.scaled_add(lambda, s_k);
                            }
                        }
                        let raw = h_raw
                            .eigh(Side::Lower)
                            .ok()
                            .map(|(evals, _)| {
                                let min = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                                let max = evals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                                max / min.max(1e-12)
                            })
                            .unwrap_or(f64::NAN);
                        *self.raw_cond_snapshot.write().unwrap() = raw;
                        raw
                    } else {
                        *self.raw_cond_snapshot.read().unwrap()
                    };
                    if want_hot_diag {
                        self.log_gam_cost(
                            &p,
                            lambdas.as_slice().unwrap_or(&[]),
                            laml,
                            stab_cond,
                            raw_cond,
                            edf,
                            trace_h_inv_s_lambda,
                        );
                    }

                    let prior_cost = self.compute_soft_prior_cost(p);

                    Ok(-laml + prior_cost)
                }
            }
        }

        ///
        /// -------------------------------------------------------------------------
        /// Exact non-Laplace evidence identities (reference comments; not runtime path)
        /// -------------------------------------------------------------------------
        /// We optimize a Laplace-style outer objective for scalability, but the exact
        /// marginal likelihood for non-Gaussian models can be written analytically as:
        ///
        ///   L(ρ) = ∫ exp(l(β) - 0.5 βᵀ S(ρ) β) dβ,   S(ρ)=Σ_k exp(ρ_k) S_k.
        ///
        /// Universal exact gradient identity (when differentiation under the integral
        /// is justified and L(ρ) < ∞):
        ///
        ///   ∂_{ρ_k} log L(ρ)
        ///   = -0.5 * exp(ρ_k) * E_{π(β|y,ρ)}[ βᵀ S_k β ].
        ///
        /// Laplace bridge to implemented terms:
        /// - If π(β|y,ρ) is approximated locally by N(β̂, H^{-1}), then
        ///     E[βᵀ S_k β] ≈ β̂ᵀ S_k β̂ + tr(H^{-1} S_k),
        ///   giving the familiar quadratic + trace structure.
        /// - In this code those appear as:
        ///     0.5 * β̂ᵀ S_k^ρ β̂,
        ///     -0.5 * tr(S^+ S_k^ρ),
        ///     +0.5 * tr(H^{-1} H_k).
        ///
        /// Why this does NOT collapse to only tr(H^{-1}S_k):
        /// - The exact identity differentiates the true integral measure.
        /// - LAML differentiates a moving approximation:
        ///     V_LAML(ρ) = -ℓ(β̂(ρ)) + 0.5 β̂(ρ)ᵀ S(ρ) β̂(ρ)
        ///                 + 0.5 log|H(ρ)| - 0.5 log|S(ρ)|_+.
        /// - Here both center β̂(ρ) and curvature H(ρ) move with ρ.
        /// - For non-Gaussian families, H_k includes the third-derivative tensor path
        ///   through β̂(ρ), i.e. H_k != S_k^ρ. These are the explicit dH/dρ_k terms
        ///   retained below to differentiate the Laplace objective exactly.
        ///
        /// For Bernoulli-logit, an exact Pólya-Gamma augmentation gives:
        ///
        ///   L(ρ) = 2^{-n} (2π)^{p/2}
        ///          E_{ω_i ~ PG(1,0)} [ |Q(ω,ρ)|^{-1/2} exp(0.5 bᵀ Q^{-1} b) ],
        ///   Q(ω,ρ)=S(ρ)+XᵀΩX, b=Xᵀ(y-1/2).
        ///
        /// and
        ///
        ///   ∂_{ρ_k} log L
        ///   = -0.5 * exp(ρ_k) *
        ///     E_{ω|y,ρ}[ tr(S_k Q^{-1}) + μᵀ S_k μ ],  μ=Q^{-1}b.
        /// Equivalently, since β|ω,y,ρ ~ N(μ,Q^{-1}):
        ///   E[βᵀS_kβ | ω,y,ρ] = tr(S_k Q^{-1}) + μᵀS_kμ.
        ///
        /// yielding exact (but high-dimensional) contour integrals / series after
        /// analytically integrating β.
        ///
        /// Practical note:
        /// - These are exact equalities but generally not polynomial-time tractable
        ///   for arbitrary dense (X, n, p).
        /// - This code therefore uses deterministic Laplace/implicit-differentiation
        ///   machinery for the main optimizer path, with exact tensor terms where
        ///   feasible (H_k, H_{kℓ}, c/d arrays), and scalable trace backends.
        ///
        /// FULL OUTER-DERIVATIVE REFERENCE (exact system, sign convention used here)
        /// -------------------------------------------------------------------------
        /// This optimizer minimizes an outer cost V(ρ).
        ///
        /// Common definitions:
        ///   λ_k = exp(ρ_k)
        ///   S(ρ) = Σ_k λ_k S_k + δI
        ///   A_k = ∂S/∂ρ_k = λ_k S_k
        ///   A_{kℓ} = ∂²S/(∂ρ_k∂ρ_ℓ) = δ_{kℓ} A_k
        ///
        /// Inner mode (β̂):
        ///   ∇_β ℓ(β̂) - S(ρ) β̂ = 0
        ///
        /// Curvature:
        ///   H(ρ) = -∇²_β ℓ(β̂(ρ)) + S(ρ)
        ///
        ///   w_i = -∂²ℓ_i/∂η_i²
        ///   d_i = -∂³ℓ_i/∂η_i³
        ///   e_i = -∂⁴ℓ_i/∂η_i⁴
        ///
        /// Then:
        ///   H_k = A_k + Xᵀ diag(d ⊙ u_k) X,     u_k := X B_k
        ///   H_{kℓ} = δ_{kℓ}A_k + Xᵀ diag(e ⊙ u_k ⊙ u_ℓ + d ⊙ u_{kℓ}) X
        ///
        /// with implicit derivatives:
        ///   B_k := ∂β̂/∂ρ_k = -H^{-1}(A_k β̂)
        ///   H B_{kℓ} = -(H_ℓ B_k + δ_{kℓ}A_k β̂ + A_k B_ℓ)
        ///
        /// Non-Gaussian negative LAML cost:
        ///   V(ρ) = -ℓ(β̂) + 0.5 β̂ᵀSβ̂ + 0.5 log|H| - 0.5 log|S|_+
        ///
        /// Exact gradient:
        ///   g_k = 0.5 β̂ᵀA_kβ̂ + 0.5 tr(H^{-1}H_k) - 0.5 ∂_k log|S|_+
        ///
        /// Exact Hessian decomposition:
        ///   ∂²V/(∂ρ_k∂ρ_ℓ) = Q_{kℓ} + L_{kℓ} + P_{kℓ}
        ///
        ///   Q_{kℓ} = 0.5 δ_{kℓ} β̂ᵀA_kβ̂ - B_ℓᵀ H B_k
        ///
        ///   L_{kℓ} = 0.5 [ tr(H^{-1}H_{kℓ}) - tr(H^{-1}H_k H^{-1}H_ℓ) ]
        ///
        ///   P_{kℓ} = -0.5 ∂²_{kℓ} log|S|_+
        ///
        /// Here, this function computes the exact gradient terms (including dH/dρ_k via d_i).
        /// The full exact Hessian is not assembled in this loop because it requires B_{kℓ}
        /// solves and fourth-derivative terms for every (k,ℓ) pair.
        ///
        /// Gaussian REML note:
        ///   In identity-link Gaussian, d=e=0 so H_k=A_k and H_{kℓ}=δ_{kℓ}A_k.
        ///   With profiled φ, use either:
        ///   - explicit profiled objective derivatives, or
        ///   - Schur complement in (ρ, log φ):
        ///       H_prof = H_{ρρ} - H_{ρα} H_{αα}^{-1} H_{αρ}.
        ///
        /// Pseudo-determinant note:
        ///   The code uses fixed-rank/stabilized conventions for log|S|_+ to keep objective
        ///   derivatives smooth and consistent with the transformed penalty basis used by PIRLS.
        ///
        /// This is the core of the outer optimization loop and provides the search direction for the BFGS algorithm.
        /// The calculation differs significantly between the Gaussian (REML) and non-Gaussian (LAML) cases.
        ///
        /// # Mathematical Basis (Gaussian/REML Case)
        ///
        /// For Gaussian models (Identity link), we minimize the negative REML log-likelihood, which serves as our cost function.
        /// From Wood (2011, JRSSB, Eq. 4), the cost function to minimize is:
        ///
        ///   Cost(ρ) = -l_r(ρ) = D_p / (2φ) + (1/2)log|XᵀWX + S(ρ)| - (1/2)log|S(ρ)|_+
        ///
        /// where D_p is the penalized deviance, H = XᵀWX + S(ρ) is the penalized Hessian, S(ρ) is the total
        /// penalty matrix, and |S(ρ)|_+ is the pseudo-determinant.
        ///
        /// The gradient ∇Cost(ρ) is computed term-by-term. A key simplification for the Gaussian case is the
        /// **envelope theorem**: at the P-IRLS optimum for β̂, the derivative of the cost function with respect to β̂ is zero.
        /// This means we only need the *partial* derivatives with respect to ρ, and the complex indirect derivatives
        /// involving ∂β̂/∂ρ can be ignored.
        ///
        /// # Mathematical Basis (Non-Gaussian/LAML Case)
        ///
        /// For non-Gaussian models, the envelope theorem does not apply because the weight matrix W depends on β̂.
        /// The gradient requires calculating the full derivative, including the indirect term (∂V/∂β̂)ᵀ(∂β̂/∂ρ).
        /// This leads to a different final formula involving derivatives of the weight matrix, as detailed in
        /// Wood (2011, Appendix D).
        ///
        /// This method handles two distinct statistical criteria for marginal likelihood optimization:
        ///
        /// - For Gaussian models (Identity link), this calculates the exact REML gradient
        ///   (Restricted Maximum Likelihood).
        /// - For non-Gaussian GLMs, this calculates the LAML gradient (Laplace Approximate
        ///   Marginal Likelihood) as derived in Wood (2011, Appendix C & D).
        ///
        /// # Mathematical Theory
        ///
        /// The gradient calculation requires careful application of the chain rule and envelope theorem
        /// due to the nested optimization structure of GAMs:
        ///
        /// - The inner loop (P-IRLS) finds coefficients β̂ that maximize the penalized log-likelihood
        ///   for a fixed set of smoothing parameters ρ.
        /// - The outer loop (BFGS) finds smoothing parameters ρ that maximize the marginal likelihood.
        ///
        /// Since β̂ is an implicit function of ρ, the total derivative is:
        ///
        ///    dV_R/dρ_k = (∂V_R/∂β̂)ᵀ(∂β̂/∂ρ_k) + ∂V_R/∂ρ_k
        ///
        /// By the envelope theorem, (∂V_R/∂β̂) = 0 at the optimum β̂, so the first term vanishes.
        ///
        /// # Key Distinction Between REML and LAML Gradients
        ///
        /// - Gaussian (REML): by the envelope theorem the indirect β̂ terms vanish. The deviance
        ///   contribution reduces to the penalty-only derivative, yielding the familiar
        ///   (β̂ᵀS_kβ̂)/σ² piece in the gradient.
        /// - Non-Gaussian (LAML): there is no cancellation of the penalty derivative within the
        ///   deviance component. The derivative of the penalized deviance contains both
        ///   d(D)/dρ_k and d(βᵀSβ)/dρ_k. Our implementation follows mgcv’s gdi1: we add the penalty
        ///   derivative to the deviance derivative before applying the 1/2 factor.
        // Stage: Start with the chain rule for any λₖ,
        //     dV/dλₖ = ∂V/∂λₖ  (holding β̂ fixed)  +  (∂V/∂β̂)ᵀ · (∂β̂/∂λₖ).
        //     The first summand is called the direct part, the second the indirect part.
        //
        // Stage: Note the two outer criteria—Gaussian likelihood maximizes REML, while non-Gaussian likelihood
        //     maximizes a Laplace approximation to the marginal likelihood (LAML). These objectives respond differently to β̂.
        //
        //     2.1  Gaussian case, REML.
        //          The REML construction integrates the fixed effects out of the likelihood.  At the optimum
        //          the partial derivative ∂V/∂β̂ is exactly zero.  The indirect part therefore vanishes.
        //          What remains is the direct derivative of the penalty and determinant terms.  The penalty
        //          contribution is found by differentiating −½ β̂ᵀ S_λ β̂ / σ² with respect to λₖ; this yields
        //          −½ β̂ᵀ Sₖ β̂ / σ².  No opposing term exists, so the quantity stays in the REML gradient.
        //          The code path selected by LinkFunction::Identity therefore computes
        //          beta_term = β̂ᵀ Sₖ β̂ and places it inside
        //          gradient[k] = 0.5 * λₖ * (beta_term / σ² − trace_term).
        //
        //     2.2  Non-Gaussian case, LAML.
        //          The Laplace objective contains −½ log |H_p| with H_p = Xᵀ W(β̂) X + S_λ.  Because W
        //          depends on β̂, the total derivative includes dW/dλₖ via β̂.  Differentiating the
        //          optimality condition for β̂ gives
        //          ∂β̂/∂λₖ = −λₖ H_p⁻¹ Sₖ β̂.  The penalized log-likelihood L(β̂, λ) still obeys the
        //          envelope theorem, so dL/dλₖ = −½ β̂ᵀ Sₖ β̂ (no implicit term).
        //          The resulting cost gradient combines four pieces:
        //            +½ λₖ β̂ᵀ Sₖ β̂
        //            +½ λₖ tr(H_p⁻¹ Sₖ)
        //            +½ tr(H_p⁻¹ Xᵀ ∂W/∂λₖ X)
        //            −½ λₖ tr(S_λ⁺ Sₖ)
        //
        // Stage: Remember that the sign of ∂β̂/∂λₖ matters; from the implicit-function theorem the linear solve reads
        //     −H_p (∂β̂/∂λₖ) = λₖ Sₖ β̂, giving the minus sign used above.  With that sign the indirect and
        //     direct quadratic pieces are exact negatives, which is what the algebra requires.
        pub fn compute_gradient(&self, p: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
            if self.uses_objective_consistent_fd_gradient(p) {
                // Fixed-choice gradient definition for this fit (no mid-flight swapping):
                // use objective-consistent FD in the known sign-unstable 1D non-Gaussian path.
                return compute_fd_gradient_internal(self, p, false, false);
            }
            // Get the converged P-IRLS result for the current rho (`p`)
            let bundle = match self.obtain_eval_bundle(p) {
                Ok(bundle) => bundle,
                Err(err @ EstimationError::ModelIsIllConditioned { .. }) => {
                    self.current_eval_bundle.write().unwrap().take();
                    return Err(err);
                }
                Err(e) => {
                    self.current_eval_bundle.write().unwrap().take();
                    return Err(e);
                }
            };
            let analytic = self.compute_gradient_with_bundle(p, &bundle)?;
            Ok(analytic)
        }

        #[inline]
        fn uses_objective_consistent_fd_gradient(&self, rho: &Array1<f64>) -> bool {
            self.config.link_function() != LinkFunction::Identity
                && (self.config.objective_consistent_fd_gradient || rho.len() == 1)
        }

        /// Helper function that computes gradient using a shared evaluation bundle
        /// so cost and gradient reuse the identical stabilized Hessian and PIRLS state.
        ///
        /// # Exact Outer-Gradient Identity Used by This Function
        ///
        /// Notation:
        /// - `rho[k]` are log-smoothing parameters; `lambda[k] = exp(rho[k])`.
        /// - `S(rho) = Σ_k lambda[k] S_k`.
        /// - `A_k = ∂S/∂rho_k = lambda[k] S_k`.
        /// - `beta_hat(rho)` is the inner PIRLS mode.
        /// - `H(rho)` is the Laplace curvature matrix used by this objective path.
        ///
        /// Outer objective:
        ///   V(rho) = [penalized data-fit at beta_hat]
        ///          + 0.5 log|H(rho)| - 0.5 log|S(rho)|_+.
        ///
        /// Exact derivative form:
        ///   dV/drho_k
        ///   = 0.5 * beta_hat^T A_k beta_hat
        ///   + 0.5 * tr(H^{-1} H_k)
        ///   - 0.5 * tr(S^+ A_k),
        /// where H_k = dH/drho_k is the *total* derivative (includes beta_hat movement).
        ///
        /// Important implementation point:
        /// - We do NOT add a separate `(∇_beta V)^T (d beta_hat / d rho_k)` term on top of
        ///   `tr(H^{-1} H_k)`. That dependence is already inside `H_k`.
        ///
        /// Variable mapping in this function:
        /// - `beta_terms[k]`     => beta_hat^T A_k beta_hat
        /// - `det1_values[k]`    => tr(S^+ A_k)
        /// - `trace_terms[k]`    => tr(H^{-1} H_k) / lambda[k] (before the outer lambda factor)
        /// - final assembly       => 0.5*beta_terms + 0.5*lambda*trace_terms - 0.5*det1
        ///
        /// ## Exact non-Gaussian Hessian system (reference for this implementation)
        ///
        /// For outer parameters ρ with λ_k = exp(ρ_k), A_k = ∂S/∂ρ_k = λ_k S_k, and
        /// H = -∇²ℓ(β̂(ρ)) + S(ρ), exact derivatives are:
        ///
        ///   B_k := ∂β̂/∂ρ_k = -H^{-1}(A_k β̂)
        ///
        ///   H_k := ∂H/∂ρ_k = A_k + D(-∇²ℓ)[B_k]
        ///
        ///   B_{kℓ} solves:
        ///     H B_{kℓ} = -(H_ℓ B_k + δ_{kℓ} A_k β̂ + A_k B_ℓ)
        ///
        ///   H_{kℓ} := ∂²H/(∂ρ_k∂ρ_ℓ)
        ///     = δ_{kℓ}A_k + D²(-∇²ℓ)[B_k,B_ℓ] + D(-∇²ℓ)[B_{kℓ}]
        ///
        /// Then the exact outer Hessian for V(ρ) = -ℓ(β̂)+0.5β̂ᵀSβ̂+0.5log|H|-0.5log|S|_+ is:
        ///
        ///   ∂²V/(∂ρ_k∂ρ_ℓ)
        ///     = 0.5 δ_{kℓ} β̂ᵀA_kβ̂ - B_ℓᵀ H B_k
        ///       + 0.5[ tr(H^{-1}H_{kℓ}) - tr(H^{-1}H_k H^{-1}H_ℓ) ]
        ///       - 0.5 ∂² log|S|_+ /(∂ρ_k∂ρ_ℓ)
        ///
        /// This function computes the exact gradient terms (including the third-derivative
        /// contribution in H_k for logit). Full explicit H_{kℓ} assembly is not
        /// performed in the hot optimization loop because it requires B_{kℓ} solves and
        /// fourth-derivative likelihood terms for every (k,ℓ) pair.
        fn compute_gradient_with_bundle(
            &self,
            p: &Array1<f64>,
            bundle: &EvalShared,
        ) -> Result<Array1<f64>, EstimationError> {
            // If there are no penalties (zero-length rho), the gradient in rho-space is empty.
            if p.is_empty() {
                return Ok(Array1::zeros(0));
            }

            let pirls_result = bundle.pirls_result.as_ref();
            let ridge_passport = bundle.ridge_passport;

            let free_basis_opt = self.active_constraint_free_basis(pirls_result);
            let reparam_result = &pirls_result.reparam_result;
            let mut h_eff_eval = bundle.h_eff.as_ref().clone();
            let mut e_eval = reparam_result.e_transformed.clone();
            let mut beta_eval = pirls_result.beta_transformed.as_ref().clone();
            let mut rs_eval = reparam_result.rs_transformed.clone();
            let mut x_transformed_eval = pirls_result.x_transformed.clone();
            let mut h_pos_factor_w_eval = bundle.h_pos_factor_w.as_ref().clone();

            if let Some(z) = free_basis_opt.as_ref() {
                h_eff_eval = Self::project_with_basis(bundle.h_eff.as_ref(), z);
                e_eval = reparam_result.e_transformed.dot(z);
                beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
                rs_eval = reparam_result
                    .rs_transformed
                    .iter()
                    .map(|r| r.dot(z))
                    .collect();
                let x_dense_arc = pirls_result.x_transformed.to_dense_arc();
                x_transformed_eval = DesignMatrix::Dense(x_dense_arc.as_ref().dot(z));

                let (eigvals, eigvecs) = h_eff_eval
                    .eigh(Side::Lower)
                    .map_err(EstimationError::EigendecompositionFailed)?;
                let max_ev = eigvals.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs()));
                let tol = (h_eff_eval.nrows().max(1) as f64) * f64::EPSILON * max_ev.max(1.0);
                let valid_indices: Vec<usize> = eigvals
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &val)| if val > tol { Some(idx) } else { None })
                    .collect();
                let p_eff = h_eff_eval.nrows();
                let mut w = Array2::<f64>::zeros((p_eff, valid_indices.len()));
                for (w_col_idx, &eig_idx) in valid_indices.iter().enumerate() {
                    let val = eigvals[eig_idx];
                    let scale = 1.0 / val.sqrt();
                    let u_col = eigvecs.column(eig_idx);
                    let mut w_col = w.column_mut(w_col_idx);
                    Zip::from(&mut w_col)
                        .and(&u_col)
                        .for_each(|w_elem, &u_elem| *w_elem = u_elem * scale);
                }
                h_pos_factor_w_eval = w;
            }
            let h_eff = &h_eff_eval;

            // Sanity check: penalty dimension consistency across lambdas, R_k, and det1.
            let k_lambda = p.len();
            let k_r = rs_eval.len();
            let k_d = pirls_result.reparam_result.det1.len();
            if !(k_lambda == k_r && k_r == k_d) {
                return Err(EstimationError::LayoutError(format!(
                    "Penalty dimension mismatch: lambdas={}, R={}, det1={}",
                    k_lambda, k_r, k_d
                )));
            }
            if self.nullspace_dims.len() != k_lambda {
                return Err(EstimationError::LayoutError(format!(
                    "Nullspace dimension mismatch: expected {} entries, got {}",
                    k_lambda,
                    self.nullspace_dims.len()
                )));
            }

            // --- Extract stable transformed quantities ---
            let beta_transformed = &beta_eval;
            // Use cached X·Qs from PIRLS
            let rs_transformed = &rs_eval;

            let includes_prior = false;
            let (gradient_result, gradient_snapshot, applied_truncation_corrections) = {
                let mut workspace_ref = self.workspace.lock().unwrap();
                let workspace = &mut *workspace_ref;
                let len = p.len();
                workspace.reset_for_eval(len);
                workspace.set_lambda_values(p);
                workspace.zero_cost_gradient(len);
                let lambdas = workspace.lambda_view(len).to_owned();
                let mut applied_truncation_corrections: Option<Vec<f64>> = None;

                // Fixed structural-rank pseudo-determinant derivatives:
                // d/dρ_k log|S|_+ and d²/(dρ_k dρ_ℓ) log|S|_+ are evaluated on a
                // reduced structural subspace (rank = e_transformed.nrows()) with a
                // smooth floor in that reduced block. This avoids adaptive rank flips.
                let (det1_values, _) = self.structural_penalty_logdet_derivatives(
                    rs_transformed,
                    &lambdas,
                    e_eval.nrows(),
                    ridge_passport.penalty_logdet_ridge(),
                )?;

                // --- Use Single Stabilized Hessian from P-IRLS ---
                // Use the same effective Hessian as the cost function for consistency.
                if ridge_passport.laplace_hessian_ridge() > 0.0 {
                    log::debug!(
                        "Gradient path using PIRLS-stabilized Hessian (ridge {:.3e})",
                        ridge_passport.laplace_hessian_ridge()
                    );
                }

                // Check that the stabilized effective Hessian is still numerically valid.
                // If even the ridged matrix is indefinite, the PIRLS fit is unreliable and we retreat.
                if let Ok((eigenvalues, _)) = h_eff.eigh(Side::Lower) {
                    let min_eig = eigenvalues.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    const SEVERE_INDEFINITENESS: f64 = -1e-4; // Threshold for severe problems
                    if min_eig < SEVERE_INDEFINITENESS {
                        // The matrix was severely indefinite - signal a need to retreat
                        log::warn!(
                            "Severely indefinite Hessian detected in gradient (min_eig={:.2e}); returning robust retreat gradient.",
                            min_eig
                        );
                        // Generate an informed retreat direction based on current parameters
                        let retreat_grad = p.mapv(|v| -(v.abs() + 1.0));
                        return Ok(retreat_grad);
                    }
                }

                // --- Extract common components ---

                let n = self.y.len() as f64;

                // -------------------------------------------------------------------------
                // Math map to user derivation (Section A):
                //   A.0: λ_k = exp(ρ_k), A_k = ∂S/∂ρ_k = λ_k S_k.
                //   A.2: Envelope theorem at inner stationarity removes explicit dβ̂/dρ term
                //        from the penalized-fit block.
                //   A.3: Outer gradient assembly
                //        ∂V/∂ρ_k = 0.5 β̂^T A_k β̂ + 0.5 tr(H_+^† H_k) - 0.5 tr(S_+^† A_k).
                //   A.1/A.4: H_k differs by family:
                //        Gaussian: H_k = A_k.
                //        Non-Gaussian: H_k = A_k + d(X^T W(η̂) X)/dρ_k (third-derivative path).
                // -------------------------------------------------------------------------
                // Reference: gam.fit3.R line 778: REML1 <- oo$D1/(2*scale*gamma) + oo$trA1/2 - rp$det1/2

                match self.config.link_function() {
                    LinkFunction::Identity => {
                        // GAUSSIAN REML GRADIENT - Wood (2011) Section 6.6.1

                        // Calculate scale parameter using the regular REML profiling
                        // φ = D_p / (n - M_p), where M_p is the penalty nullspace dimension.
                        let rss = pirls_result.deviance;

                        // Use stable penalty term calculated in P-IRLS
                        let penalty = pirls_result.stable_penalty_term;
                        let dp = rss + penalty; // Penalized deviance (a.k.a. D_p)
                        let (dp_c, dp_c_grad) = smooth_floor_dp(dp);

                        let penalty_rank = e_eval.nrows();
                        let mp = h_eff.ncols().saturating_sub(penalty_rank) as f64;
                        let scale = dp_c / (n - mp).max(LAML_RIDGE);
                        // Gaussian profiled-scale identity used by this branch:
                        //   φ̂(ρ) = D_p(ρ)/(n-M_p), with D_p = rss + β̂ᵀSβ̂.
                        // The gradient therefore includes the profiled contribution
                        //   (n-M_p)/2 * D_k / D_p
                        // which is exactly represented by `deviance_grad_term` below.
                        // (Equivalent to Schur-complement profiling in (ρ, log φ).)

                        if dp_c <= DP_FLOOR + DP_FLOOR_SMOOTH_WIDTH {
                            eprintln!(
                                "[REML WARNING] Penalized deviance {:.3e} near DP_FLOOR; using central differences for entire gradient.",
                                dp_c
                            );
                            let mut grad_total_view =
                                workspace.grad_secondary.slice_mut(s![..lambdas.len()]);
                            grad_total_view.fill(0.0);
                            for k in 0..lambdas.len() {
                                let h = 1e-3_f64 * (1.0 + p[k].abs());
                                if h == 0.0 {
                                    continue;
                                }
                                workspace.rho_plus.assign(p);
                                workspace.rho_plus[k] += h;
                                workspace.rho_minus.assign(p);
                                workspace.rho_minus[k] -= h;
                                let cost_plus = self.compute_cost(&workspace.rho_plus)?;
                                let cost_minus = self.compute_cost(&workspace.rho_minus)?;
                                grad_total_view[k] = (cost_plus - cost_minus) / (2.0 * h);
                            }
                            return Ok(grad_total_view.to_owned());
                        }

                        // Three-term gradient computation following mgcv gdi1
                        // for k in 0..lambdas.len() {
                        //   We'll calculate s_k_beta for all cases, as it's needed for both paths
                        //   For Identity link, this is all we need due to envelope theorem
                        //   For other links, we'll use it to compute dβ/dρ_k

                        //   Use transformed penalty matrix for consistent gradient calculation
                        //   let s_k_beta = reparam_result.rs_transformed[k].dot(beta);

                        // For the Gaussian/REML case, the Envelope Theorem applies: at the P-IRLS optimum,
                        // the indirect derivative through β cancels out for the deviance part, leaving only
                        // the direct penalty term derivative. This simplification is not available for
                        // non-Gaussian models where the weight matrix depends on β.

                        // factor_g already computed above; reuse it for trace terms

                        // When the penalized deviance collapses to the numerical floor, the Hessian
                        // can become so ill-conditioned that the analytic ½·log|H| derivative loses
                        // fidelity.  Switch to an exact finite-difference evaluation in that regime
                        // to match the cost function.
                        let use_numeric_logh = dp_c <= DP_FLOOR + DP_FLOOR_SMOOTH_WIDTH;
                        let numeric_logh_grad = if use_numeric_logh {
                            eprintln!(
                                "[REML WARNING] Switching ½·log|H| gradient to numeric finite differences; dp_c={:.3e}.",
                                dp_c
                            );
                            Some(self.numeric_half_logh_grad_with_workspace(p, workspace)?)
                        } else {
                            None
                        };

                        let numeric_logh_grad_ref = numeric_logh_grad.as_ref();
                        let det1_values = &det1_values;
                        let beta_ref = beta_transformed;
                        // Use the same positive-part Hessian factor as cost evaluation:
                        //   H_+^† = W W^T.
                        // Then tr(H_+^† A_k) = λ_k ||R_k W||_F^2 directly, with no separate
                        // truncated-subspace subtraction term.
                        let w_pos = &h_pos_factor_w_eval;
                        // Exact Gaussian identity REML gradient (profiled scale) in log-smoothing coordinates:
                        //
                        //   V_REML(ρ) =
                        //     0.5 * log|H|
                        //   - 0.5 * log|S|_+
                        //   + ((n - M_p)/2) * log(2π φ̂)
                        //   + const,
                        //
                        // where H = Xᵀ W0 X + S(ρ), S(ρ) = Σ_k λ_k S_k + δI, λ_k = exp(ρ_k),
                        // and φ̂ = D_p / (n - M_p), D_p = ||W0^(1/2)(y - Xβ̂ - o)||² + β̂ᵀ S β̂.
                        //
                        // Because Gaussian identity has c_i = d_i = 0, we have:
                        //   H_k := ∂H/∂ρ_k = S_k^ρ = λ_k S_k.
                        // Envelope theorem at β̂(ρ) gives:
                        //   ∂D_p/∂ρ_k = β̂ᵀ S_k^ρ β̂.
                        // Therefore:
                        //   ∂V_REML/∂ρ_k =
                        //     0.5 * tr(H^{-1} S_k^ρ)
                        //   - 0.5 * tr(S^+ S_k^ρ)
                        //   + (1/(2 φ̂)) * β̂ᵀ S_k^ρ β̂.
                        //
                        // Mapping to variables below:
                        //   d1 / (2*scale)                     -> (1/(2 φ̂)) * β̂ᵀ S_k^ρ β̂
                        //   log_det_h_grad_term (or numeric)   -> 0.5 * tr(H^{-1} S_k^ρ)
                        //   0.5 * det1_values[k]               -> 0.5 * tr(S^+ S_k^ρ)
                        let compute_gaussian_grad = |k: usize| -> f64 {
                            let r_k = &rs_transformed[k];
                            // Avoid forming S_k: compute S_k β = Rᵀ (R β)
                            let r_beta = r_k.dot(beta_ref);
                            let s_k_beta_transformed = r_k.t().dot(&r_beta);

                            // Component 1 derivation (profiled Gaussian REML):
                            //
                            //   V_prof includes (n-M_p)/2 * log D_p(ρ), so
                            //   ∂V_prof/∂ρ_k contributes (n-M_p)/2 * D_k / D_p = D_k/(2φ̂),
                            //   φ̂ = D_p/(n-M_p).
                            //
                            // At β̂, envelope cancellation gives:
                            //   D_k = β̂ᵀ A_k β̂ = λ_k β̂ᵀ S_k β̂.
                            //
                            // `d1` stores D_k, and the expression below is D_k/(2φ̂)
                            // with the smooth-floor derivative factor `dp_c_grad`.
                            let d1 = lambdas[k] * beta_ref.dot(&s_k_beta_transformed);
                            let deviance_grad_term = dp_c_grad * (d1 / (2.0 * scale));

                            // A.3/A.5 Component 2 derivation:
                            //   ∂/∂ρ_k [0.5 log|H|_+] = 0.5 tr(H_+^† H_k),
                            // and for Gaussian identity H_k = A_k = λ_k S_k.
                            //
                            // Root form on kept subspace:
                            //   tr(H_+^† A_k) = λ_k tr(H_+^† R_kᵀR_k)
                            //                = λ_k ||R_k W||_F², H_+^†=W W^T.
                            let log_det_h_grad_term = if let Some(g) = numeric_logh_grad_ref {
                                g[k]
                            } else {
                                let rkw = r_k.dot(w_pos);
                                let trace_h_pos_inv_s_k: f64 = rkw.iter().map(|v| v * v).sum();
                                0.5 * lambdas[k] * trace_h_pos_inv_s_k
                            };

                            let corrected_log_det_h = log_det_h_grad_term;

                            // Component 3 derivation:
                            //   -0.5 * ∂/∂ρ_k log|S|_+,
                            // with `det1_values[k]` already equal to ∂ log|S|_+ / ∂ρ_k.
                            let log_det_s_grad_term = 0.5 * det1_values[k];

                            deviance_grad_term + corrected_log_det_h - log_det_s_grad_term
                        };

                        {
                            let mut grad_view = workspace.cost_gradient_view(len);
                            for k in 0..lambdas.len() {
                                grad_view[k] = compute_gaussian_grad(k);
                            }
                        }
                        // No explicit truncation correction vector is needed in this branch:
                        // the H_+^† trace is evaluated directly on the kept subspace.
                        applied_truncation_corrections = None;
                    }
                    _ => {
                        // NON-GAUSSIAN LAML GRADIENT (A.4 exact dH/dρ path)
                        //
                        // Objective:
                        //   V_LAML(ρ) =
                        //     -ℓ(β̂) + 0.5 β̂ᵀ S β̂
                        //   - 0.5 log|S|_+
                        //   + 0.5 log|H|
                        //   + const
                        //
                        // with H(ρ) = J(β̂(ρ)) + S(ρ), J = Xᵀ diag(b) X.
                        //
                        // Exact gradient (cost minimization convention):
                        //   ∂V/∂ρ_k =
                        //     0.5 β̂ᵀ S_k^ρ β̂
                        //   - 0.5 tr(S^+ S_k^ρ)
                        //   + 0.5 tr(H^{-1} H_k)
                        //
                        // where:
                        //   S_k^ρ = λ_k S_k, λ_k = exp(ρ_k),
                        //   b_k   = ∂β̂/∂ρ_k = -H^{-1}(S_k^ρ β̂),
                        //   v_k   = H^{-1}(S_k^ρ β̂) = -b_k,
                        //   H_k   = S_k^ρ + Xᵀ diag(w' ⊙ X b_k) X
                        //         = S_k^ρ - Xᵀ diag(w' ⊙ (X v_k)) X,
                        // and c_i = -∂^3 ℓ_i / ∂η_i^3.
                        //
                        // Derivation anchor:
                        //   V(ρ) = -ℓ(β̂) + 0.5 β̂ᵀ S β̂ + 0.5 log|H|_+ - 0.5 log|S|_+
                        //   with stationarity g(β̂,ρ)=∂/∂β[-ℓ + 0.5 βᵀSβ]=0.
                        // Envelope theorem removes explicit (∂V/∂β̂)(dβ̂/dρ_k) from the
                        // penalized-fit block, but β̂-dependence still enters via dH/dρ_k.
                        // The dH term is exactly what the third-derivative contraction encodes.
                        //
                        // The second term inside H_k is the exact "missing tensor term":
                        //   ∂H/∂ρ_k ≠ S_k^ρ
                        // for non-Gaussian families; dropping it yields the usual approximation.
                        //
                        // Implementation strategy here (logit path):
                        //   1) build S_k β̂ in transformed basis via penalty roots R_k,
                        //   2) solve/apply H_+^† to get v_k and leverage terms,
                        //   3) evaluate tr(H_+^† H_k) as
                        //        tr(H_+^† S_k) - tr(H_+^† Xᵀ diag(c ⊙ X v_k) X),
                        //   4) assemble
                        //        0.5*β̂ᵀA_kβ̂ + 0.5*tr(H_+^†H_k) - 0.5*tr(S^+A_k).
                        //
                        // There is intentionally no extra "(∇_β V)^T dβ/dρ" add-on here:
                        // the beta-dependence path is already encoded in H_k through the
                        // third-derivative contraction term.
                        // Replace FD with implicit differentiation for logit models.
                        // When Firth bias reduction is enabled, the inner objective is:
                        //   L*(beta, rho) = l(beta) - 0.5 * beta' S_lambda beta
                        //                 + 0.5 * log|X' W(beta) X|
                        // with W depending on beta (logit: w_i = mu_i (1 - mu_i)).
                        // Stationarity: grad_beta L* = 0, so the implicit derivative uses
                        // H_total = X' W X + S_lambda - d^2/d beta^2 (0.5 * log|X' W X|).
                        //
                        // Exact Firth derivatives (let K = (X' W X)^{-1}):
                        //   Phi(beta) = 0.5 * log|X' W X|
                        //   grad Phi_j = 0.5 * tr(K X' (dW/d beta_j) X)
                        //             = 0.5 * sum_i h_i * (d w_i / d eta_i) * x_ij
                        //   where h_i = x_i' K x_i (leverages in weighted space).
                        //
                        //   Hessian:
                        //     d^2 Phi / (d beta_j d beta_l) =
                        //       -0.5 * tr(K X' (dW/d beta_l) X K X' (dW/d beta_j) X)
                        //       +0.5 * sum_i h_i * (d^2 w_i / d eta_i^2) * x_ij * x_il
                        //
                        // This curvature enters H_total and therefore d beta_hat / d rho_k.
                        // Our analytic LAML gradient uses H_pen = X' W X + S_lambda only,
                        // so it is inconsistent with the Firth-adjusted objective unless
                        // we add H_phi. Below we compute H_phi and use H_total for the
                        // implicit solve (d beta_hat / d rho). If that fails, we fall
                        // back to H_pen for stability.
                        let w_prime = &pirls_result.solve_c_array;
                        if !w_prime.iter().all(|v| v.is_finite()) {
                            let g_pll =
                                self.numeric_penalised_ll_grad_with_workspace(p, workspace)?;
                            let g_half_logh =
                                self.numeric_half_logh_grad_with_workspace(p, workspace)?;
                            {
                                let mut grad_view = workspace.cost_gradient_view(len);
                                for k in 0..lambdas.len() {
                                    grad_view[k] = g_pll[k] + g_half_logh[k] - 0.5 * det1_values[k];
                                }
                            }
                            // Continue to prior-gradient adjustment below.
                        } else {
                            let clamp_nonsmooth = self.config.firth_bias_reduction
                                && pirls_result
                                    .solve_mu
                                    .iter()
                                    .any(|&mu| mu * (1.0 - mu) < Self::MIN_DMU_DETA);
                            if clamp_nonsmooth {
                                // Keep analytic gradient as the optimizer default even when IRLS
                                // weights are clamped, to avoid FD ridge-jitter artifacts in
                                // line-search/BFGS updates.
                                // Section B note:
                                // hard clamps/floors make the objective only piecewise-smooth;
                                // c_i values then act like a selected generalized derivative
                                // (Clarke-subgradient style), so central FD may disagree at kinks.
                                log::debug!(
                                    "[REML] IRLS weight clamp detected; continuing with analytic gradient"
                                );
                            }
                            let k_count = lambdas.len();
                            let det1_values = &det1_values;
                            let beta_ref = beta_transformed;
                            let mut beta_terms = Array1::<f64>::zeros(k_count);
                            let mut s_k_beta_mat = Array2::<f64>::zeros((beta_ref.len(), k_count));
                            for k in 0..k_count {
                                let r_k = &rs_transformed[k];
                                let r_beta = r_k.dot(beta_ref);
                                let s_k_beta = r_k.t().dot(&r_beta);
                                // Section C.1 step (1):
                                //   q_k = β̂^T A_k β̂ = λ_k β̂^T S_k β̂,
                                // with S_k β̂ assembled as R_k^T (R_k β̂).
                                beta_terms[k] = lambdas[k] * beta_ref.dot(&s_k_beta);
                                s_k_beta_mat.column_mut(k).assign(&s_k_beta);
                            }

                            // Keep outer gradient on the same Hessian surface as PIRLS.
                            // The outer loop uses H_eff consistently (no H_phi subtraction).

                            // P-IRLS already folded any stabilization ridge into h_eff.

                            // TRACE TERM COMPUTATION (exact non-Gaussian/logit dH term):
                            //   tr(H_+^\dagger H_k), with
                            //   H_k = S_k - X^T diag(c ⊙ (X v_k)) X,  v_k = H_+^\dagger (S_k beta).
                            //
                            // We evaluate this without explicit third-derivative tensors:
                            //   tr(H_+^\dagger S_k) = ||R_k W||_F^2
                            //   tr(H_+^\dagger X^T diag(t_k) X) = Σ_i t_k[i] * h_i,
                            // where t_k = c ⊙ (X v_k), h_i = x_i^T H_+^\dagger x_i, and H_+^\dagger = W W^T.
                            //
                            // This is the matrix-free realization of the exact identity:
                            //   tr(H^{-1}H_k) = tr(H^{-1}A_k) + tr(H^{-1}D(-∇²ℓ)[B_k]),
                            // with B_k = -H^{-1}(A_kβ̂).
                            //
                            //   D(-∇²ℓ)[B_k] = Xᵀ diag(d ⊙ (X B_k)) X,
                            // where d_i = -∂³ℓ_i/∂η_i³. Here `c_vec` stores this per-observation
                            // third derivative quantity in the stabilized logit path.
                            let w_pos = &h_pos_factor_w_eval;
                            let n_obs = pirls_result.solve_mu.len();

                            // c_i = dW_ii/dη_i for H = Xᵀ W X + S.
                            // In smooth regimes this matches the required third-derivative object
                            // in dH/dρ. In clamped/floored regimes c_i may behave like a subgradient
                            // proxy rather than a classical derivative; see pirls.rs comments.
                            let c_vec = w_prime;

                            // h_i = x_i^T H_+^\dagger x_i = ||(XW)_{i,*}||^2.
                            let mut leverage_h_pos = Array1::<f64>::zeros(n_obs);
                            if w_pos.ncols() > 0 {
                                match &x_transformed_eval {
                                    DesignMatrix::Dense(x_dense) => {
                                        let xw = x_dense.dot(w_pos);
                                        for i in 0..xw.nrows() {
                                            leverage_h_pos[i] =
                                                xw.row(i).iter().map(|v| v * v).sum();
                                        }
                                    }
                                    DesignMatrix::Sparse(_) => {
                                        for col in 0..w_pos.ncols() {
                                            let w_col = w_pos.column(col).to_owned();
                                            let xw_col =
                                                x_transformed_eval.matrix_vector_multiply(&w_col);
                                            Zip::from(&mut leverage_h_pos)
                                                .and(&xw_col)
                                                .for_each(|h, &v| *h += v * v);
                                        }
                                    }
                                }
                            }

                            // Precompute r = X^T (c ⊙ h) once:
                            //   trace_third_k = (c ⊙ h)^T (X v_k) = r^T v_k.
                            // This removes the per-k O(np) multiply X*v_k from the hot loop.
                            // Section C.1 step (4): r := X^T (w' ⊙ h).
                            let c_times_h = c_vec * &leverage_h_pos;
                            let r_third = x_transformed_eval.transpose_vector_multiply(&c_times_h);

                            // Batch all v_k = H_+^† (S_k beta) into one BLAS-3 path:
                            //   V = W (W^T [S_1 beta, ..., S_K beta]).
                            let v_all = if w_pos.ncols() > 0 && k_count > 0 {
                                let wt_sk_beta_all = w_pos.t().dot(&s_k_beta_mat);
                                w_pos.dot(&wt_sk_beta_all)
                            } else {
                                Array2::<f64>::zeros((beta_ref.len(), k_count))
                            };

                            let trace_mode = std::env::var("GAM_DIAG_TRACE_THIRD_MODE")
                                .unwrap_or_else(|_| "minus".to_string());
                            let trace_mode_code = match trace_mode.as_str() {
                                "plus" => 1u8,
                                "zero" => 2u8,
                                _ => 0u8,
                            };
                            {
                                let mut grad_view = workspace.cost_gradient_view(len);
                                for k_idx in 0..k_count {
                                    let r_k = &rs_transformed[k_idx];
                                    if r_k.ncols() == 0 || w_pos.ncols() == 0 {
                                        let log_det_h_grad_term = 0.0;
                                        let log_det_s_grad_term = 0.5 * det1_values[k_idx];
                                        grad_view[k_idx] = 0.5 * beta_terms[k_idx]
                                            + log_det_h_grad_term
                                            - log_det_s_grad_term;
                                        continue;
                                    }

                                    // First piece:
                                    //   tr(H_+^† S_k) = ||R_k W||_F^2, with H_+^† = W W^T.
                                    let rkw = r_k.dot(w_pos);
                                    let trace_h_inv_s_k: f64 = rkw.iter().map(|v| v * v).sum();

                                    // Exact third-derivative contraction:
                                    //   tr(H_+^† X^T diag(c ⊙ X v_k) X) = r^T v_k.
                                    let v_k = v_all.column(k_idx);
                                    let trace_third = r_third.dot(&v_k);

                                    // Diagnostic switch for term-by-term identification of
                                    // analytic-vs-FD disagreement. Production behavior is "minus",
                                    // matching the smooth-theory formula tr(H^{-1}A_k) - tr(H^{-1}Xᵀdiag(c⊙Xv_k)X).
                                    let trace_term = match trace_mode_code {
                                        1 => trace_h_inv_s_k + trace_third,
                                        2 => trace_h_inv_s_k,
                                        _ => trace_h_inv_s_k - trace_third,
                                    };
                                    let log_det_h_grad_term = 0.5 * lambdas[k_idx] * trace_term;
                                    let corrected_log_det_h = log_det_h_grad_term;
                                    let log_det_s_grad_term = 0.5 * det1_values[k_idx];

                                    // Exact LAML gradient assembly for the implemented objective:
                                    //   g_k = 0.5 * β̂ᵀ A_k β̂ - 0.5 * tr(S^+ A_k) + 0.5 * tr(H^{-1} H_k)
                                    // where A_k = ∂S/∂ρ_k = λ_k S_k and H_k is the total derivative.
                                    grad_view[k_idx] = 0.5 * beta_terms[k_idx]
                                        + corrected_log_det_h
                                        - log_det_s_grad_term;
                                }
                            }
                        }
                    }
                }

                if !includes_prior {
                    let (_, prior_grad_view) = workspace.soft_prior_cost_and_grad(p);
                    let prior_grad = prior_grad_view.to_owned();
                    {
                        let mut cost_gradient_view = workspace.cost_gradient_view(len);
                        cost_gradient_view += &prior_grad;
                    }
                }

                // Capture the gradient snapshot before releasing the workspace borrow so
                // that diagnostics can continue without holding the RefCell borrow.
                let gradient_result = workspace.cost_gradient_view_const(len).to_owned();
                let gradient_snapshot = if p.is_empty() {
                    None
                } else {
                    Some(gradient_result.clone())
                };

                (
                    gradient_result,
                    gradient_snapshot,
                    applied_truncation_corrections,
                )
            };

            // The gradient buffer stored in the workspace already holds -∇V(ρ),
            // which is exactly what the optimizer needs.
            // No final negation is needed.

            // Comprehensive gradient diagnostics (all four strategies)
            if let Some(gradient_snapshot) = gradient_snapshot
                && !p.is_empty()
            {
                // Run all diagnostics and emit a single summary if issues found
                self.run_gradient_diagnostics(
                    p,
                    bundle,
                    &gradient_snapshot,
                    applied_truncation_corrections.as_deref(),
                );
            }

            if self.should_use_stochastic_exact_gradient(bundle, &gradient_result) {
                match self.compute_logit_stochastic_exact_gradient(p, bundle) {
                    Ok(stochastic_grad) => {
                        log::warn!(
                            "[REML] using stochastic exact log-marginal gradient fallback (posterior-sampled expectation)"
                        );
                        return Ok(stochastic_grad);
                    }
                    Err(err) => {
                        log::warn!(
                            "[REML] stochastic exact gradient fallback failed; keeping analytic gradient: {:?}",
                            err
                        );
                    }
                }
            }

            Ok(gradient_result)
        }

        fn should_use_stochastic_exact_gradient(
            &self,
            bundle: &EvalShared,
            gradient: &Array1<f64>,
        ) -> bool {
            // Gate for the posterior-sampled gradient path.
            // This predicate checks for non-finite or unstable analytic states.
            if self.config.link_function() != LinkFunction::Logit {
                return false;
            }
            if self.config.firth_bias_reduction {
                // Firth-adjusted inner objective does not match the plain PG/NUTS posterior target here.
                return false;
            }
            if gradient.is_empty() {
                return false;
            }
            if !gradient.iter().all(|g| g.is_finite()) {
                return true;
            }
            let pirls = bundle.pirls_result.as_ref();
            if matches!(pirls.status, pirls::PirlsStatus::Unstable) {
                return true;
            }
            let kkt_like = pirls.last_gradient_norm;
            if !kkt_like.is_finite() || kkt_like > 1e2 {
                return true;
            }
            let grad_inf = gradient.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            !grad_inf.is_finite() || grad_inf > 1e9
        }

        fn compute_logit_stochastic_exact_gradient(
            &self,
            p: &Array1<f64>,
            bundle: &EvalShared,
        ) -> Result<Array1<f64>, EstimationError> {
            // Derivation sketch (sign convention used by this minimization objective):
            //
            // 1) Penalized evidence identity (logit):
            //      Z(ρ) = ∫ exp(l(β) - 0.5 βᵀS(ρ)β) dβ,   S(ρ)=Σ_j exp(ρ_j) S_j.
            //
            // 2) Fisher/PG identity for each coordinate:
            //      ∂/∂ρ_k log Z(ρ) = -0.5 * λ_k * E_{π(β|y,ρ)}[βᵀ S_k β],   λ_k=exp(ρ_k).
            //
            // 3) This code optimizes a cost that includes the pseudo-determinant
            //    normalization of the improper Gaussian penalty, yielding:
            //      g_k = ∂Cost/∂ρ_k
            //          = 0.5 * λ_k * E[βᵀS_kβ] - 0.5 * λ_k * tr(S(ρ)^+ S_k).
            //
            // 4) Root-factor rewrite used numerically:
            //      S_k = R_kᵀR_k  =>  βᵀS_kβ = ||R_kβ||².
            //
            // 5) Implementation mapping:
            //      PG-Rao-Blackwell average of tr(S_kQ^{-1})+μᵀS_kμ -> E[βᵀS_kβ],
            //      det1_values[k]                                 -> λ_k tr(S(ρ)^+S_k),
            //      grad[k]                                        -> g_k.
            // Equation-to-code map for this fallback path (logit, fixed ρ):
            //   g_k := ∂Cost/∂ρ_k
            //      = 0.5 * λ_k * E_{π(β|y,ρ)}[βᵀ S_k β]
            //        - 0.5 * λ_k * tr(S(ρ)^+ S_k),
            //   λ_k = exp(ρ_k).
            //
            // The first expectation is evaluated by PG Gibbs + Rao-Blackwellization.
            // The second term is deterministic via structural pseudo-logdet derivatives.
            let pirls_result = bundle.pirls_result.as_ref();
            let beta_mode = pirls_result.beta_transformed.as_ref();
            let s_transformed = &pirls_result.reparam_result.s_transformed;
            let x_arc = pirls_result.x_transformed.to_dense_arc();
            let x_dense = x_arc.as_ref();
            let y = self.y;
            let weights = self.weights;
            let h_eff = bundle.h_eff.as_ref();

            // PG-Gibbs Rao-Blackwell fallback: fewer samples are needed than β-NUTS
            // because each retained ω state contributes the analytic conditional moment
            // tr(S_k Q^{-1}) + μᵀ S_k μ instead of a raw quadratic draw.
            let pg_cfg = crate::hmc::NutsConfig {
                n_samples: 24,
                n_warmup: 48,
                n_chains: 2,
                target_accept: 0.85,
                seed: 17_391,
            };

            let len = p.len();
            let mut lambda = Array1::<f64>::zeros(len);
            for k in 0..len {
                // Outer parameters are ρ; penalties are λ = exp(ρ).
                lambda[k] = p[k].exp();
            }

            let (det1_values, _) = self.structural_penalty_logdet_derivatives(
                &pirls_result.reparam_result.rs_transformed,
                &lambda,
                pirls_result.reparam_result.e_transformed.nrows(),
                bundle.ridge_passport.penalty_logdet_ridge(),
            )?;
            // det1_values[k] = ∂ log|S(ρ)|_+ / ∂ρ_k = λ_k tr(S(ρ)^+ S_k).

            let rb_terms_result = crate::hmc::estimate_logit_pg_rao_blackwell_terms(
                x_dense.view(),
                y,
                weights,
                s_transformed.view(),
                beta_mode.view(),
                &pirls_result.reparam_result.rs_transformed,
                &pg_cfg,
            );

            let mut grad = Array1::<f64>::zeros(len);
            match rb_terms_result {
                Ok(rb_terms) => {
                    for k in 0..len {
                        // Rao-Blackwellized exact identity:
                        //   g_k = 0.5 * λ_k * E_ω[ tr(S_k Q^{-1}) + μᵀ S_k μ ] - 0.5 * det1_values[k].
                        grad[k] = 0.5 * lambda[k] * rb_terms[k] - 0.5 * det1_values[k];
                    }
                }
                Err(err) => {
                    log::warn!(
                        "[REML] PG Rao-Blackwell fallback failed ({}); reverting to NUTS beta averaging",
                        err
                    );

                    let nuts_cfg = crate::hmc::NutsConfig {
                        n_samples: 120,
                        n_warmup: 160,
                        n_chains: 2,
                        target_accept: 0.85,
                        seed: 17_391,
                    };

                    let nuts_result = crate::hmc::run_nuts_sampling_flattened_family(
                        crate::types::LikelihoodFamily::BinomialLogit,
                        crate::hmc::FamilyNutsInputs::Glm(crate::hmc::GlmFlatInputs {
                            x: x_dense.view(),
                            y,
                            weights,
                            penalty_matrix: s_transformed.view(),
                            mode: beta_mode.view(),
                            hessian: h_eff.view(),
                            firth_bias_reduction: self.config.firth_bias_reduction,
                        }),
                        &nuts_cfg,
                    )
                    .map_err(EstimationError::InvalidInput)?;

                    let samples = &nuts_result.samples;
                    let n_draws = samples.nrows().max(1);
                    let mut expected_quad = vec![0.0_f64; len];
                    for draw in 0..samples.nrows() {
                        let beta_draw = samples.row(draw).to_owned();
                        for k in 0..len {
                            let r_k = &pirls_result.reparam_result.rs_transformed[k];
                            let r_beta = r_k.dot(&beta_draw);
                            expected_quad[k] += r_beta.dot(&r_beta);
                        }
                    }
                    let inv_draws = 1.0 / (n_draws as f64);
                    for v in &mut expected_quad {
                        *v *= inv_draws;
                    }
                    for k in 0..len {
                        grad[k] = 0.5 * lambda[k] * expected_quad[k] - 0.5 * det1_values[k];
                    }
                }
            }
            grad += &self.compute_soft_prior_grad(p);
            Ok(grad)
        }

        fn xt_diag_x_dense_into(
            x: &Array2<f64>,
            diag: &Array1<f64>,
            weighted: &mut Array2<f64>,
        ) -> Array2<f64> {
            let n = x.nrows();
            weighted.assign(x);
            for i in 0..n {
                let w = diag[i];
                for j in 0..x.ncols() {
                    weighted[[i, j]] *= w;
                }
            }
            fast_atb(x, weighted)
        }

        fn trace_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
            debug_assert_eq!(a.nrows(), b.ncols());
            debug_assert_eq!(a.ncols(), b.nrows());
            let elems = a.nrows().saturating_mul(a.ncols());
            if elems >= 32 * 32 {
                let a_view = FaerArrayView::new(a);
                let b_view = FaerArrayView::new(b);
                return faer_frob_inner(a_view.as_ref(), b_view.as_ref().transpose());
            }
            let m = a.nrows();
            let n = a.ncols();
            kahan_sum((0..m).map(|i| {
                let mut acc = 0.0_f64;
                for j in 0..n {
                    acc += a[[i, j]] * b[[j, i]];
                }
                acc
            }))
        }

        fn bilinear_form(
            mat: &Array2<f64>,
            left: ndarray::ArrayView1<'_, f64>,
            right: ndarray::ArrayView1<'_, f64>,
        ) -> f64 {
            let n = mat.nrows();
            debug_assert_eq!(mat.ncols(), n);
            debug_assert_eq!(left.len(), n);
            debug_assert_eq!(right.len(), n);
            let mut acc = KahanSum::default();
            for i in 0..n {
                let mut row_dot = 0.0_f64;
                for j in 0..n {
                    row_dot += mat[[i, j]] * right[j];
                }
                acc.add(left[i] * row_dot);
            }
            acc.sum()
        }

        fn select_trace_backend(n_obs: usize, p_dim: usize, k_count: usize) -> TraceBackend {
            // Workload-aware policy driven by (n, p, K):
            // - Exact for moderate total complexity.
            // - Hutchinson/Hutch++ as n·p·K and p²·K² costs grow.
            //
            // Proxies:
            //   w_npk   ~ n*p*K   (X/Xᵀ + diagonal contractions)
            //   w_pk2   ~ p*K²    (pairwise rho-Hessian assembly)
            let k = k_count.max(1);
            let w_npk = (n_obs as u128)
                .saturating_mul(p_dim as u128)
                .saturating_mul(k as u128);
            let w_pk2 = (p_dim as u128).saturating_mul((k as u128).saturating_mul(k as u128));

            if p_dim <= 700 && k <= 20 && w_npk <= 220_000_000 && w_pk2 <= 20_000_000 {
                return TraceBackend::Exact;
            }

            let very_large =
                p_dim >= 1_800 || k >= 28 || w_npk >= 1_100_000_000 || w_pk2 >= 85_000_000;
            if very_large {
                let sketch = if p_dim >= 3_500 || w_npk >= 2_500_000_000 {
                    12
                } else {
                    8
                };
                let probes = if k >= 36 || w_pk2 >= 150_000_000 {
                    28
                } else {
                    22
                };
                return TraceBackend::HutchPP { probes, sketch };
            }

            let probes = if w_npk >= 700_000_000 || k >= 24 {
                34
            } else if w_npk >= 350_000_000 {
                28
            } else {
                22
            };
            TraceBackend::Hutchinson { probes }
        }

        #[inline]
        fn splitmix64(mut x: u64) -> u64 {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }

        fn rademacher_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
            let mut out = Array2::<f64>::zeros((rows, cols));
            for j in 0..cols {
                for i in 0..rows {
                    let h = Self::splitmix64(
                        seed ^ ((i as u64).wrapping_mul(0x9E37))
                            ^ ((j as u64).wrapping_mul(0x85EB)),
                    );
                    out[[i, j]] = if (h & 1) == 0 { -1.0 } else { 1.0 };
                }
            }
            out
        }

        fn orthonormalize_columns(a: &Array2<f64>, tol: f64) -> Array2<f64> {
            let p = a.nrows();
            let c = a.ncols();
            let mut q = Array2::<f64>::zeros((p, c));
            let mut kept = 0usize;
            for j in 0..c {
                let mut v = a.column(j).to_owned();
                for t in 0..kept {
                    let qt = q.column(t);
                    let proj = qt.dot(&v);
                    v -= &qt.mapv(|x| x * proj);
                }
                let nrm = v.dot(&v).sqrt();
                if nrm > tol {
                    q.column_mut(kept).assign(&v.mapv(|x| x / nrm));
                    kept += 1;
                }
            }
            if kept == c {
                q
            } else {
                q.slice(ndarray::s![.., 0..kept]).to_owned()
            }
        }

        fn structural_penalty_logdet_derivatives(
            &self,
            rs_transformed: &[Array2<f64>],
            lambdas: &Array1<f64>,
            structural_rank: usize,
            ridge: f64,
        ) -> Result<(Array1<f64>, Array2<f64>), EstimationError> {
            // Section A.1/A.3 (penalty pseudo-logdet path):
            //   det1[k] = ∂/∂ρ_k log|S(ρ)|_+ = tr(S_+^† A_k),
            //   A_k = λ_k S_k, S_k = R_k^T R_k.
            //
            // This helper computes det1/det2 on a fixed-rank structural subspace to
            // keep the objective differentiable w.r.t. ρ under the implemented
            // positive-part convention (A+-fixed active subspace assumption).
            let k_count = lambdas.len();
            if rs_transformed.len() != k_count {
                return Err(EstimationError::LayoutError(format!(
                    "Penalty root/lambda count mismatch in structural logdet derivatives: roots={}, lambdas={}",
                    rs_transformed.len(),
                    k_count
                )));
            }
            if k_count == 0 {
                return Ok((Array1::zeros(k_count), Array2::zeros((k_count, k_count))));
            }

            // IMPORTANT: dimensions must follow the *actual* transformed coefficient frame
            // presented by callers (possibly active-constraint projected), not self.p.
            let p_dim = rs_transformed[0].ncols();
            for (k, r_k) in rs_transformed.iter().enumerate() {
                if r_k.ncols() != p_dim {
                    return Err(EstimationError::LayoutError(format!(
                        "Inconsistent penalty root width at k={k}: got {}, expected {}",
                        r_k.ncols(),
                        p_dim
                    )));
                }
            }
            if p_dim == 0 || structural_rank == 0 {
                return Ok((Array1::zeros(k_count), Array2::zeros((k_count, k_count))));
            }

            let rank = structural_rank.min(p_dim);
            if rank == 0 {
                return Ok((Array1::zeros(k_count), Array2::zeros((k_count, k_count))));
            }

            let mut s_k_full = Vec::with_capacity(k_count);
            let mut s_lambda = Array2::<f64>::zeros((p_dim, p_dim));
            for k in 0..k_count {
                let r_k = &rs_transformed[k];
                // Path: rs_transformed[k] is already in transformed coefficient frame.
                let s_k = r_k.t().dot(r_k);
                s_lambda += &s_k.mapv(|v| lambdas[k] * v);
                s_k_full.push(s_k);
            }
            if ridge > 0.0 {
                for i in 0..p_dim {
                    s_lambda[[i, i]] += ridge;
                }
            }

            let (evals, evecs) = s_lambda
                .eigh(Side::Lower)
                .map_err(EstimationError::EigendecompositionFailed)?;
            let mut order: Vec<usize> = (0..p_dim).collect();
            order.sort_by(|&a, &b| {
                evals[b]
                    .partial_cmp(&evals[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });

            let mut u1 = Array2::<f64>::zeros((p_dim, rank));
            for (col_out, &col_in) in order.iter().take(rank).enumerate() {
                u1.column_mut(col_out).assign(&evecs.column(col_in));
            }
            let mut s_r = u1.t().dot(&s_lambda).dot(&u1);
            let max_diag = s_r
                .diag()
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let eps = 1e-12 * max_diag;
            for i in 0..rank {
                s_r[[i, i]] += eps;
            }
            let s_r_inv = matrix_inverse_with_regularization(&s_r, "structural penalty block")
                .ok_or_else(|| EstimationError::ModelIsIllConditioned {
                    condition_number: f64::INFINITY,
                })?;

            let mut s_k_reduced = Vec::with_capacity(k_count);
            let mut det1 = Array1::<f64>::zeros(k_count);
            for k in 0..k_count {
                let s_kr = u1.t().dot(&s_k_full[k]).dot(&u1);
                // tr(S_r^{-1} S_{k,r}) = tr(S_+^† S_k) on kept subspace.
                let tr = kahan_sum((0..rank).map(|i| {
                    let mut acc = 0.0;
                    for j in 0..rank {
                        acc += s_r_inv[[i, j]] * s_kr[[j, i]];
                    }
                    acc
                }));
                // A_k = λ_k S_k => tr(S_+^† A_k) = λ_k tr(S_+^† S_k).
                det1[k] = lambdas[k] * tr;
                s_k_reduced.push(s_kr);
            }

            let mut det2 = Array2::<f64>::zeros((k_count, k_count));
            for k in 0..k_count {
                for l in 0..=k {
                    let a = s_r_inv.dot(&s_k_reduced[k]);
                    let b = s_r_inv.dot(&s_k_reduced[l]);
                    let tr_ab = kahan_sum((0..rank).map(|i| {
                        let mut acc = 0.0;
                        for j in 0..rank {
                            acc += a[[i, j]] * b[[j, i]];
                        }
                        acc
                    }));
                    let mut val = -lambdas[k] * lambdas[l] * tr_ab;
                    if k == l {
                        val += det1[k];
                    }
                    det2[[k, l]] = val;
                    det2[[l, k]] = val;
                }
            }
            Ok((det1, det2))
        }

        fn compute_hessian_fd_from_active_gradient(
            &self,
            rho: &Array1<f64>,
        ) -> Result<Array2<f64>, EstimationError> {
            let k = rho.len();
            let mut h = Array2::<f64>::zeros((k, k));
            if k == 0 {
                return Ok(h);
            }

            for j in 0..k {
                let h_step = (1e-4 * (1.0 + rho[j].abs())).max(1e-6);
                let mut rho_plus = rho.clone();
                rho_plus[j] += h_step;
                let mut rho_minus = rho.clone();
                rho_minus[j] -= h_step;

                let grad_plus = self.compute_gradient(&rho_plus)?;
                let grad_minus = self.compute_gradient(&rho_minus)?;
                if grad_plus.len() != k || grad_minus.len() != k {
                    return Err(EstimationError::RemlOptimizationFailed(
                        "FD Hessian gradient length mismatch".to_string(),
                    ));
                }

                for i in 0..k {
                    h[[i, j]] = (grad_plus[i] - grad_minus[i]) / (2.0 * h_step);
                }
            }

            for i in 0..k {
                for j in 0..i {
                    let avg = 0.5 * (h[[i, j]] + h[[j, i]]);
                    h[[i, j]] = avg;
                    h[[j, i]] = avg;
                }
            }

            if h.iter().any(|v| !v.is_finite()) {
                return Err(EstimationError::RemlOptimizationFailed(
                    "FD Hessian produced non-finite values".to_string(),
                ));
            }
            Ok(h)
        }

        pub(super) fn compute_laml_hessian_consistent(
            &self,
            rho: &Array1<f64>,
        ) -> Result<Array2<f64>, EstimationError> {
            if self.uses_objective_consistent_fd_gradient(rho) {
                return self.compute_hessian_fd_from_active_gradient(rho);
            }
            match self.compute_laml_hessian_exact(rho) {
                Ok(h) => Ok(h),
                Err(err) => {
                    log::warn!(
                        "Exact LAML Hessian unavailable ({}); falling back to FD Hessian from active gradient.",
                        err
                    );
                    self.compute_hessian_fd_from_active_gradient(rho)
                }
            }
        }

        pub(super) fn compute_laml_hessian_exact(
            &self,
            rho: &Array1<f64>,
        ) -> Result<Array2<f64>, EstimationError> {
            // Exact non-Gaussian outer Hessian components (ρ-space):
            //
            //   B_k   = ∂β̂/∂ρ_k = -H^{-1}(A_k β̂)
            //   B_{kℓ}= ∂²β̂/(∂ρ_k∂ρ_ℓ) from
            //          H B_{kℓ} = -(H_ℓ B_k + A_k B_ℓ + δ_{kℓ} A_k β̂)
            //
            //   H_k   = A_k + Xᵀ diag(c ⊙ u_k) X,        u_k   = X B_k
            //   H_{kℓ}= δ_{kℓ}A_k + Xᵀ diag(d ⊙ u_k ⊙ u_ℓ + c ⊙ u_{kℓ}) X,
            //           where u_{kℓ}=X B_{kℓ}
            //
            // Here `c` and `d` are the per-observation 3rd/4th eta-derivative arrays
            // prepared by PIRLS (`solve_c_array`, `solve_d_array`).
            //
            // Full exact Hessian entry used below:
            //
            //   ∂²V/(∂ρ_k∂ρ_ℓ) = Q_{kℓ} + L_{kℓ} + P_{kℓ}
            //
            // with
            //   Q_{kℓ} = B_ℓᵀ A_k β̂ + 0.5 δ_{kℓ} β̂ᵀ A_k β̂
            //   L_{kℓ} = 0.5 [ -tr(H^{-1}H_ℓ H^{-1}H_k) + tr(H^{-1}H_{kℓ}) ]
            //   P_{kℓ} = -0.5 ∂² log|S|_+ /(∂ρ_k∂ρ_ℓ)
            //
            // Numerically, this function computes:
            // - Q exactly from B_k solves,
            // - P exactly from reduced-penalty logdet derivatives,
            // - L either exactly or stochastically, depending on workload.
            //
            // The objective also includes the separable soft rho prior used by
            // compute_cost/compute_gradient; its exact diagonal Hessian is added
            // to every return path below for full objective consistency.
            //
            // Stochastic trace identities used when backend != Exact:
            //   tr(A) = E[zᵀAz],  z_i∈{±1}.
            //   tr(H^{-1}H_ℓH^{-1}H_k) estimated by shared-probe contractions.
            //   tr(H^{-1}H_{kℓ}) estimated by probe bilinear forms.
            // Hutch++ augments this with a low-rank deflation subspace Q to reduce
            // variance before Hutchinson residual estimation.
            let bundle = self.obtain_eval_bundle(rho)?;
            let pirls_result = bundle.pirls_result.as_ref();
            let reparam_result = &pirls_result.reparam_result;

            // Active-constraint-aware exact Hessian path:
            // Evaluate all non-Gaussian second-order terms on the current active-free subspace
            // span(Z), consistent with cost/gradient projection.
            let free_basis_opt = self.active_constraint_free_basis(pirls_result);
            let mut beta_eval = pirls_result.beta_transformed.as_ref().clone();
            let mut rs_eval = reparam_result.rs_transformed.clone();
            let x_dense_orig_arc = pirls_result.x_transformed.to_dense_arc();
            let mut x_dense_eval = x_dense_orig_arc.as_ref().to_owned();
            let mut h_total_eval = bundle.h_total.as_ref().clone();
            let mut e_eval = reparam_result.e_transformed.clone();

            if let Some(z) = free_basis_opt.as_ref() {
                h_total_eval = Self::project_with_basis(bundle.h_total.as_ref(), z);
                beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
                rs_eval = reparam_result
                    .rs_transformed
                    .iter()
                    .map(|r| r.dot(z))
                    .collect();
                x_dense_eval = x_dense_orig_arc.as_ref().dot(z);
                e_eval = reparam_result.e_transformed.dot(z);
            }

            let beta = &beta_eval;
            let rs_transformed = &rs_eval;
            let h_total = &h_total_eval;
            let use_cached_factor = free_basis_opt.is_none();
            let h_factor_cached = if use_cached_factor {
                Some(self.get_faer_factor(rho, h_total))
            } else {
                None
            };
            let h_factor_local = if use_cached_factor {
                None
            } else {
                Some(self.factorize_faer(h_total))
            };
            let solve_h = |rhs: &Array2<f64>| -> Array2<f64> {
                let mut out = rhs.clone();
                let mut out_view = array2_to_mat_mut(&mut out);
                if let Some(f) = h_factor_cached.as_ref() {
                    f.solve_in_place(out_view.as_mut());
                } else if let Some(f) = h_factor_local.as_ref() {
                    f.solve_in_place(out_view.as_mut());
                }
                out
            };

            let k_count = rho.len();
            if k_count == 0 {
                return Ok(Array2::zeros((0, 0)));
            }
            let lambdas = rho.mapv(f64::exp);
            let x_dense = &x_dense_eval;
            let n = x_dense.nrows();
            let p_dim = x_dense.ncols();
            if p_dim == 0 {
                let (_, d2logs) = self.structural_penalty_logdet_derivatives(
                    rs_transformed,
                    &lambdas,
                    e_eval.nrows(),
                    bundle.ridge_passport.penalty_logdet_ridge(),
                )?;
                let mut hess = Array2::<f64>::zeros((k_count, k_count));
                for l in 0..k_count {
                    for k in 0..k_count {
                        hess[[k, l]] = -0.5 * d2logs[[k, l]];
                    }
                }
                self.add_soft_prior_hessian_in_place(rho, &mut hess);
                return Ok(hess);
            }
            let c = &pirls_result.solve_c_array;
            let d = &pirls_result.solve_d_array;
            if c.len() != n || d.len() != n {
                return Err(EstimationError::InvalidInput(format!(
                    "Exact Hessian derivative arrays size mismatch: n={}, c.len()={}, d.len()={}",
                    n,
                    c.len(),
                    d.len()
                )));
            }

            let mut a_k_mats = Vec::with_capacity(k_count);
            let mut a_k_beta = Vec::with_capacity(k_count);
            let mut rhs_bk = Array2::<f64>::zeros((p_dim, k_count));
            let mut q_diag = vec![0.0; k_count];
            for k in 0..k_count {
                let r_k = &rs_transformed[k];
                let s_k = r_k.t().dot(r_k);
                let r_beta = r_k.dot(beta);
                let s_k_beta = r_k.t().dot(&r_beta);
                let a_k = s_k.mapv(|v| lambdas[k] * v);
                let a_kb = s_k_beta.mapv(|v| lambdas[k] * v);
                q_diag[k] = beta.dot(&a_kb);
                rhs_bk.column_mut(k).assign(&a_kb.mapv(|v| -v));
                a_k_mats.push(a_k);
                a_k_beta.push(a_kb);
            }

            let b_mat = solve_h(&rhs_bk);
            let u_mat = fast_ab(x_dense, &b_mat);

            let mut h_k = Vec::with_capacity(k_count);
            let mut weighted_xtdx = Array2::<f64>::zeros(x_dense.raw_dim());
            for k in 0..k_count {
                let mut diag = Array1::<f64>::zeros(n);
                for i in 0..n {
                    diag[i] = c[i] * u_mat[[i, k]];
                }
                let mut hk = a_k_mats[k].clone();
                hk += &Self::xt_diag_x_dense_into(x_dense, &diag, &mut weighted_xtdx);
                h_k.push(hk);
            }
            let s_cols: Vec<Array1<f64>> = (0..k_count)
                .map(|k| {
                    let mut s = Array1::<f64>::zeros(n);
                    for i in 0..n {
                        s[i] = c[i] * u_mat[[i, k]];
                    }
                    s
                })
                .collect();

            let trace_backend = Self::select_trace_backend(n, p_dim, k_count);
            let (exact_trace_mode, n_probe, n_sketch) = match trace_backend {
                TraceBackend::Exact => (true, 0usize, 0usize),
                TraceBackend::Hutchinson { probes } => (false, probes.max(1), 0usize),
                TraceBackend::HutchPP { probes, sketch } => (false, probes.max(1), sketch.max(1)),
            };
            let use_hutchpp = matches!(trace_backend, TraceBackend::HutchPP { .. });
            // Backend semantics:
            // - Exact: deterministic traces via explicit H^{-1} contractions.
            // - Hutchinson/Hutch++: Monte-Carlo trace estimators (unbiased/low-bias in
            //   expectation) trading tiny stochastic noise for major scaling gains.

            let h_inv = if exact_trace_mode {
                Some(solve_h(&Array2::<f64>::eye(p_dim)))
            } else {
                None
            };
            let m_k: Option<Vec<Array2<f64>>> = h_inv
                .as_ref()
                .map(|hinv| h_k.iter().map(|hk| hinv.dot(hk)).collect());

            let mut probe_z: Option<Array2<f64>> = None;
            let mut probe_u: Option<Array2<f64>> = None;
            let mut probe_xz: Option<Array2<f64>> = None;
            let mut probe_xu: Option<Array2<f64>> = None;
            let mut sketch_q: Option<Array2<f64>> = None;
            let mut sketch_uq: Option<Array2<f64>> = None;
            let mut sketch_xq: Option<Array2<f64>> = None;
            let mut sketch_xuq: Option<Array2<f64>> = None;

            if !exact_trace_mode {
                let mut z = Self::rademacher_matrix(p_dim, n_probe, 0xC0DEC0DE5EEDu64);
                if use_hutchpp && n_sketch > 0 {
                    let g = Self::rademacher_matrix(p_dim, n_sketch, 0xBADC0FFEE0DDF00Du64);
                    let y = solve_h(&g);
                    let q = Self::orthonormalize_columns(&y, 1e-10);
                    if q.ncols() > 0 {
                        for r in 0..n_probe {
                            let mut zr = z.column(r).to_owned();
                            let qt_z = q.t().dot(&zr);
                            let proj = q.dot(&qt_z);
                            zr -= &proj;
                            z.column_mut(r).assign(&zr);
                        }
                        let uq = solve_h(&q);
                        let xq = fast_ab(x_dense, &q);
                        let xuq = fast_ab(x_dense, &uq);
                        sketch_q = Some(q);
                        sketch_uq = Some(uq);
                        sketch_xq = Some(xq);
                        sketch_xuq = Some(xuq);
                    }
                }
                let u = solve_h(&z);
                let xz = fast_ab(x_dense, &z);
                let xu = fast_ab(x_dense, &u);
                probe_z = Some(z);
                probe_u = Some(u);
                probe_xz = Some(xz);
                probe_xu = Some(xu);
            }

            let mut t1_mat = Array2::<f64>::zeros((k_count, k_count));
            if exact_trace_mode {
                let mk = m_k.as_ref().expect("m_k present in exact mode");
                for l in 0..k_count {
                    for k in 0..k_count {
                        t1_mat[[l, k]] = Self::trace_product(&mk[l], &mk[k]);
                    }
                }
            } else {
                if let (Some(q), Some(uq), Some(xq), Some(xuq)) = (
                    sketch_q.as_ref(),
                    sketch_uq.as_ref(),
                    sketch_xq.as_ref(),
                    sketch_xuq.as_ref(),
                ) {
                    let rdim = q.ncols();
                    for j in 0..rdim {
                        let qj = q.column(j).to_owned();
                        let uqj = uq.column(j).to_owned();
                        let xqj = xq.column(j).to_owned();
                        let xuqj = xuq.column(j).to_owned();
                        let mut bq = Array2::<f64>::zeros((p_dim, k_count));
                        for k in 0..k_count {
                            let mut hkq = a_k_mats[k].dot(&qj);
                            let weighted = &s_cols[k] * &xqj;
                            hkq += &x_dense.t().dot(&weighted);
                            bq.column_mut(k).assign(&hkq);
                        }
                        let wq = solve_h(&bq);
                        let xwq = fast_ab(x_dense, &wq);
                        for l in 0..k_count {
                            let alu = a_k_mats[l].dot(&uqj);
                            let sxu = &s_cols[l] * &xuqj;
                            for k in 0..k_count {
                                let val = alu.dot(&wq.column(k)) + sxu.dot(&xwq.column(k));
                                t1_mat[[l, k]] += val;
                            }
                        }
                    }
                }
                let z = probe_z.as_ref().expect("probes present in stochastic mode");
                let u = probe_u
                    .as_ref()
                    .expect("solved probes present in stochastic mode");
                let xz = probe_xz
                    .as_ref()
                    .expect("X probes present in stochastic mode");
                let xu = probe_xu
                    .as_ref()
                    .expect("X solved probes present in stochastic mode");
                for r in 0..n_probe {
                    let zr = z.column(r).to_owned();
                    let ur = u.column(r).to_owned();
                    let xzr = xz.column(r).to_owned();
                    let xur = xu.column(r).to_owned();
                    let mut bz = Array2::<f64>::zeros((p_dim, k_count));
                    for k in 0..k_count {
                        let mut hkz = a_k_mats[k].dot(&zr);
                        let weighted = &s_cols[k] * &xzr;
                        hkz += &x_dense.t().dot(&weighted);
                        bz.column_mut(k).assign(&hkz);
                    }
                    let wz = solve_h(&bz);
                    let xwz = fast_ab(x_dense, &wz);
                    for l in 0..k_count {
                        let alu = a_k_mats[l].dot(&ur);
                        let sxu = &s_cols[l] * &xur;
                        for k in 0..k_count {
                            let val = alu.dot(&wz.column(k)) + sxu.dot(&xwz.column(k));
                            t1_mat[[l, k]] += val / (n_probe as f64);
                        }
                    }
                }
            }
            for i in 0..k_count {
                for j in 0..i {
                    let avg = 0.5 * (t1_mat[[i, j]] + t1_mat[[j, i]]);
                    t1_mat[[i, j]] = avg;
                    t1_mat[[j, i]] = avg;
                }
            }

            let (_, d2logs) = self.structural_penalty_logdet_derivatives(
                rs_transformed,
                &lambdas,
                e_eval.nrows(),
                bundle.ridge_passport.penalty_logdet_ridge(),
            )?;

            let mut hess = Array2::<f64>::zeros((k_count, k_count));
            for l in 0..k_count {
                let bl = b_mat.column(l).to_owned();
                let mut rhs_kl_all = Array2::<f64>::zeros((p_dim, k_count));
                for k in l..k_count {
                    let bk = b_mat.column(k).to_owned();
                    let mut rhs_kl = -h_k[l].dot(&bk);
                    rhs_kl -= &a_k_mats[k].dot(&bl);
                    if k == l {
                        rhs_kl -= &a_k_beta[k];
                    }
                    rhs_kl_all.column_mut(k).assign(&rhs_kl);
                }
                let b_kl_all = solve_h(&rhs_kl_all);
                let u_kl_all = fast_ab(x_dense, &b_kl_all);

                let mut weighted_xtdx_kl = Array2::<f64>::zeros(x_dense.raw_dim());
                for k in l..k_count {
                    let mut diag = Array1::<f64>::zeros(n);
                    for i in 0..n {
                        diag[i] = d[i] * u_mat[[i, k]] * u_mat[[i, l]] + c[i] * u_kl_all[[i, k]];
                    }

                    let q = bl.dot(&a_k_beta[k]) + if k == l { 0.5 * q_diag[k] } else { 0.0 };
                    let t1 = t1_mat[[l, k]];
                    let t2 = if exact_trace_mode {
                        let mut h_kl = if k == l {
                            a_k_mats[k].clone()
                        } else {
                            Array2::<f64>::zeros((p_dim, p_dim))
                        };
                        h_kl += &Self::xt_diag_x_dense_into(x_dense, &diag, &mut weighted_xtdx_kl);
                        let h_inv_ref = h_inv.as_ref().expect("h_inv present in exact mode");
                        Self::trace_product(h_inv_ref, &h_kl)
                    } else {
                        let mut t2_acc = 0.0_f64;
                        if let (Some(q), Some(uq), Some(xq), Some(xuq)) = (
                            sketch_q.as_ref(),
                            sketch_uq.as_ref(),
                            sketch_xq.as_ref(),
                            sketch_xuq.as_ref(),
                        ) {
                            for j in 0..q.ncols() {
                                let qj = q.column(j);
                                let uqj = uq.column(j);
                                let xqj = xq.column(j);
                                let xuqj = xuq.column(j);
                                let mut term = 0.0_f64;
                                if k == l {
                                    term += Self::bilinear_form(&a_k_mats[k], uqj, qj);
                                }
                                let mut quad = 0.0_f64;
                                for i in 0..n {
                                    quad += xuqj[i] * diag[i] * xqj[i];
                                }
                                term += quad;
                                t2_acc += term;
                            }
                        }
                        let z = probe_z.as_ref().expect("probes present in stochastic mode");
                        let u = probe_u
                            .as_ref()
                            .expect("solved probes present in stochastic mode");
                        let xz = probe_xz
                            .as_ref()
                            .expect("X probes present in stochastic mode");
                        let xu = probe_xu
                            .as_ref()
                            .expect("X solved probes present in stochastic mode");
                        let mut res = 0.0_f64;
                        for r in 0..n_probe {
                            let zr = z.column(r);
                            let ur = u.column(r);
                            let xzr = xz.column(r);
                            let xur = xu.column(r);
                            let mut term = 0.0_f64;
                            if k == l {
                                term += Self::bilinear_form(&a_k_mats[k], ur, zr);
                            }
                            let mut quad = 0.0_f64;
                            for i in 0..n {
                                quad += xur[i] * diag[i] * xzr[i];
                            }
                            term += quad;
                            res += term;
                        }
                        t2_acc + res / (n_probe as f64)
                    };
                    let l_term = 0.5 * (-t1 + t2);
                    let p_term = -0.5 * d2logs[[k, l]];
                    let val = q + l_term + p_term;
                    hess[[k, l]] = val;
                    hess[[l, k]] = val;
                }
            }
            self.add_soft_prior_hessian_in_place(rho, &mut hess);
            Ok(hess)
        }

        pub(super) fn compute_smoothing_correction_auto(
            &self,
            final_rho: &Array1<f64>,
            final_fit: &PirlsResult,
            base_covariance: Option<&Array2<f64>>,
            final_grad_norm: f64,
        ) -> Option<Array2<f64>> {
            // Always compute the fast first-order correction first.
            let first_order = super::compute_smoothing_correction(self, final_rho, final_fit);
            let n_rho = final_rho.len();
            if n_rho == 0 {
                return first_order;
            }
            if n_rho > AUTO_CUBATURE_MAX_RHO_DIM {
                return first_order;
            }
            if final_fit.beta_transformed.len() > AUTO_CUBATURE_MAX_BETA_DIM {
                return first_order;
            }

            let near_boundary = final_rho
                .iter()
                .any(|&v| (RHO_BOUND - v.abs()) <= AUTO_CUBATURE_BOUNDARY_MARGIN);
            let grad_norm = if final_grad_norm.is_finite() {
                final_grad_norm
            } else {
                0.0
            };
            let high_grad = grad_norm > 1e-3;
            if !near_boundary && !high_grad {
                // Keep the hot path cheap when the local linearization is likely sufficient.
                return first_order;
            }

            // Build V_rho from the outer Hessian around rho_hat.
            let mut hessian_rho = match self.compute_laml_hessian_consistent(final_rho) {
                Ok(h) => h,
                Err(err) => {
                    log::debug!("Auto cubature skipped: rho Hessian unavailable ({}).", err);
                    return first_order;
                }
            };
            for i in 0..n_rho {
                for j in (i + 1)..n_rho {
                    let avg = 0.5 * (hessian_rho[[i, j]] + hessian_rho[[j, i]]);
                    hessian_rho[[i, j]] = avg;
                    hessian_rho[[j, i]] = avg;
                }
            }
            let ridge = 1e-8
                * hessian_rho
                    .diag()
                    .iter()
                    .map(|&v| v.abs())
                    .fold(0.0, f64::max)
                    .max(1e-8);
            for i in 0..n_rho {
                hessian_rho[[i, i]] += ridge;
            }
            let hessian_rho_inv =
                match matrix_inverse_with_regularization(&hessian_rho, "auto cubature rho Hessian")
                {
                    Some(v) => v,
                    None => return first_order,
                };

            let max_rho_var = hessian_rho_inv
                .diag()
                .iter()
                .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
            if !near_boundary && !high_grad && max_rho_var < 0.1 {
                return first_order;
            }

            use crate::faer_ndarray::FaerEigh;
            use faer::Side;
            let (evals, evecs) = match hessian_rho_inv.eigh(Side::Lower) {
                Ok(x) => x,
                Err(_) => return first_order,
            };
            let mut eig_pairs: Vec<(usize, f64)> = evals
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, v)| v.is_finite() && *v > 1e-12)
                .collect();
            if eig_pairs.is_empty() {
                return first_order;
            }
            eig_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let total_var: f64 = eig_pairs.iter().map(|(_, v)| *v).sum();
            if !total_var.is_finite() || total_var <= 0.0 {
                return first_order;
            }

            let mut rank = 0usize;
            let mut captured = 0.0_f64;
            for (_, eig) in eig_pairs
                .iter()
                .take(AUTO_CUBATURE_MAX_EIGENVECTORS.min(eig_pairs.len()))
            {
                captured += *eig;
                rank += 1;
                if captured / total_var >= AUTO_CUBATURE_TARGET_VAR_FRAC {
                    break;
                }
            }
            if rank == 0 {
                return first_order;
            }

            let base_cov = match base_covariance {
                Some(v) => v,
                None => return first_order,
            };
            let p = base_cov.nrows();
            let radius = (rank as f64).sqrt();
            let mut sigma_points: Vec<Array1<f64>> = Vec::with_capacity(2 * rank);
            for (eig_idx, eig_val) in eig_pairs.iter().take(rank) {
                let axis = evecs.column(*eig_idx).to_owned();
                let scale = radius * eig_val.sqrt();
                let delta = axis.mapv(|v| v * scale);

                for sign in [1.0_f64, -1.0_f64] {
                    let mut rho_point = final_rho.clone();
                    for i in 0..n_rho {
                        rho_point[i] = (rho_point[i] + sign * delta[i])
                            .clamp(-RHO_BOUND + 1e-8, RHO_BOUND - 1e-8);
                    }
                    sigma_points.push(rho_point);
                }
            }
            if sigma_points.is_empty() {
                return first_order;
            }

            // Disable warm-start and PIRLS-cache coupling while evaluating sigma
            // points in parallel. Cache lookups/inserts use an exclusive lock in
            // execute_pirls_if_needed(), so leaving cache enabled serializes this
            // block under contention.
            struct FlagRestoreGuard<'a> {
                flag: &'a AtomicBool,
                prev: bool,
            }
            impl Drop for FlagRestoreGuard<'_> {
                fn drop(&mut self) {
                    self.flag.store(self.prev, Ordering::SeqCst);
                }
            }
            let prev_cache = self.pirls_cache_enabled.swap(false, Ordering::SeqCst);
            let _cache_guard = FlagRestoreGuard {
                flag: &self.pirls_cache_enabled,
                prev: prev_cache,
            };
            let prev_warm_start = self.warm_start_enabled.swap(false, Ordering::SeqCst);
            let _warm_start_guard = FlagRestoreGuard {
                flag: &self.warm_start_enabled,
                prev: prev_warm_start,
            };
            let point_results: Vec<Option<(Array2<f64>, Array1<f64>)>> = (0..sigma_points.len())
                .into_par_iter()
                .map(|idx| {
                    let fit_point = self.execute_pirls_if_needed(&sigma_points[idx]).ok()?;
                    let h_point = map_hessian_to_original_basis(fit_point.as_ref()).ok()?;
                    let cov_point =
                        matrix_inverse_with_regularization(&h_point, "auto cubature point")?;
                    let beta_point = fit_point
                        .reparam_result
                        .qs
                        .dot(fit_point.beta_transformed.as_ref());
                    Some((cov_point, beta_point))
                })
                .collect();

            if point_results.iter().any(|r| r.is_none()) {
                return first_order;
            }

            let w = 1.0 / (sigma_points.len() as f64);
            let mut mean_hinv = Array2::<f64>::zeros((p, p));
            let mut mean_beta = Array1::<f64>::zeros(p);
            let mut second_beta = Array2::<f64>::zeros((p, p));
            for (cov_point, beta_point) in point_results.into_iter().flatten() {
                mean_hinv += &cov_point.mapv(|v| w * v);
                mean_beta += &beta_point.mapv(|v| w * v);
                for i in 0..p {
                    let bi = beta_point[i];
                    for j in 0..p {
                        second_beta[[i, j]] += w * bi * beta_point[j];
                    }
                }
            }

            let mut var_beta = second_beta;
            for i in 0..p {
                for j in 0..p {
                    var_beta[[i, j]] -= mean_beta[i] * mean_beta[j];
                }
            }

            let mut total_cov = mean_hinv + var_beta;
            for i in 0..p {
                for j in (i + 1)..p {
                    let avg = 0.5 * (total_cov[[i, j]] + total_cov[[j, i]]);
                    total_cov[[i, j]] = avg;
                    total_cov[[j, i]] = avg;
                }
            }
            if !total_cov.iter().all(|v| v.is_finite()) {
                return first_order;
            }

            let mut corr = total_cov - base_cov;
            for i in 0..p {
                for j in (i + 1)..p {
                    let avg = 0.5 * (corr[[i, j]] + corr[[j, i]]);
                    corr[[i, j]] = avg;
                    corr[[j, i]] = avg;
                }
            }

            log::info!(
                "Using adaptive cubature smoothing correction (rank={}, points={}, near_boundary={}, grad_norm={:.2e}, max_var={:.2e})",
                rank,
                2 * rank,
                near_boundary,
                grad_norm,
                max_rho_var
            );
            Some(corr)
        }

        /// Run comprehensive gradient diagnostics implementing four strategies:
        /// 1. KKT/Envelope Theorem Audit
        /// 2. Component-wise Finite Difference
        /// 3. Spectral Bleed Trace
        /// 4. Dual-Ridge Consistency
        ///
        /// Only prints a summary when issues are detected.
        fn run_gradient_diagnostics(
            &self,
            rho: &Array1<f64>,
            bundle: &EvalShared,
            analytic_grad: &Array1<f64>,
            applied_truncation_corrections: Option<&[f64]>,
        ) {
            use crate::diagnostics::{
                DiagnosticConfig, GradientDiagnosticReport, compute_dual_ridge_check,
                compute_envelope_audit, compute_spectral_bleed,
            };

            let config = DiagnosticConfig::default();
            let mut report = GradientDiagnosticReport::new();

            let pirls_result = bundle.pirls_result.as_ref();
            let ridge_used = bundle.ridge_passport.delta;
            let beta = pirls_result.beta_transformed.as_ref();
            let lambdas: Array1<f64> = rho.mapv(f64::exp);

            // === Strategy 4: Dual-Ridge Consistency Check ===
            // Compare the PIRLS ridge with the ridge used by cost/gradient paths.
            let dual_ridge = compute_dual_ridge_check(
                pirls_result.ridge_passport.delta, // Ridge from PIRLS passport
                ridge_used,                        // Ridge passed to cost
                ridge_used,                        // Ridge passed to gradient (same bundle)
                beta,
            );
            report.dual_ridge = Some(dual_ridge);

            // === Strategy 1: KKT/Envelope Theorem Audit ===
            // Check if the inner solver actually reached stationarity
            let reparam = &pirls_result.reparam_result;
            let penalty_grad = reparam.s_transformed.dot(beta);

            let envelope_audit = compute_envelope_audit(
                pirls_result.last_gradient_norm,
                &penalty_grad,
                pirls_result.ridge_passport.delta,
                ridge_used, // What gradient assumes
                beta,
                config.kkt_tolerance,
                config.rel_error_threshold,
            );
            report.envelope_audit = Some(envelope_audit);

            // === Strategy 3: Spectral Bleed Trace ===
            // Check if truncated eigenspace corrections are adequate
            // Diagnostics must compare quantities in a common frame.
            // `u_truncated`, `h_eff`, and `rs_transformed` are all in transformed coordinates.
            let u_truncated = reparam.u_truncated.clone();
            let truncated_count = u_truncated.ncols();
            // Path/coordinate contract for diagnostics:
            // - `u_truncated` comes from ReparamResult (already transformed).
            // - `h_eff` and `reparam.rs_transformed` are transformed as well.

            if truncated_count > 0
                && let Some(applied_values) = applied_truncation_corrections
            {
                let h_eff = bundle.h_eff.as_ref();

                // Solve H⁻¹ U_⊥ for spectral bleed calculation
                let h_view = FaerArrayView::new(h_eff);
                if let Ok(chol) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
                    let mut h_inv_u = u_truncated.clone();
                    let mut rhs_view = array2_to_mat_mut(&mut h_inv_u);
                    chol.solve_in_place(rhs_view.as_mut());

                    for (k, r_k) in reparam.rs_transformed.iter().enumerate() {
                        let applied_correction = applied_values.get(k).copied().unwrap_or(0.0);
                        let bleed = compute_spectral_bleed(
                            k,
                            r_k.view(),
                            u_truncated.view(),
                            h_inv_u.view(),
                            lambdas[k],
                            applied_correction,
                            config.rel_error_threshold,
                        );
                        if bleed.has_bleed || bleed.truncated_energy.abs() > 1e-4 {
                            report.spectral_bleed.push(bleed);
                        }
                    }
                }
            }

            // === Strategy 2: Component-wise FD (only if we detected other issues) ===
            // This is expensive, so rate-limit it unless diagnostics are severe.
            let eval_idx = (*self.cost_eval_count.read().unwrap()).max(1);
            let severe_envelope = report.envelope_audit.as_ref().is_some_and(|a| {
                a.kkt_residual_norm > GRAD_DIAG_SEVERE_KKT_NORM
                    || (a.inner_ridge - a.outer_ridge).abs() > GRAD_DIAG_SEVERE_RIDGE_MISMATCH
            });
            let severe_bleed = report
                .spectral_bleed
                .iter()
                .any(|b| b.has_bleed && b.truncated_energy.abs() > GRAD_DIAG_SEVERE_BLEED_ENERGY);
            let severe_ridge = report.dual_ridge.as_ref().is_some_and(|r| {
                r.has_mismatch
                    && (r.ridge_impact.abs() > GRAD_DIAG_SEVERE_RIDGE_IMPACT
                        || r.phantom_penalty.abs() > GRAD_DIAG_SEVERE_PHANTOM_PENALTY)
            });
            let periodic_sample = should_sample_gradient_diag_fd(eval_idx);
            let run_component_fd = report.has_issues()
                && (severe_envelope || severe_bleed || severe_ridge || periodic_sample);
            if run_component_fd {
                struct CacheToggleGuard<'a> {
                    flag: &'a AtomicBool,
                    prev: bool,
                }
                impl Drop for CacheToggleGuard<'_> {
                    fn drop(&mut self) {
                        self.flag.store(self.prev, Ordering::Relaxed);
                    }
                }
                let prev_cache = self.pirls_cache_enabled.swap(false, Ordering::Relaxed);
                let _cache_guard = CacheToggleGuard {
                    flag: &self.pirls_cache_enabled,
                    prev: prev_cache,
                };

                let h = config.fd_step_size;
                let mut numeric_grad = Array1::<f64>::zeros(rho.len());

                for k in 0..rho.len() {
                    let mut rho_plus = rho.clone();
                    rho_plus[k] += h;
                    let mut rho_minus = rho.clone();
                    rho_minus[k] -= h;

                    let fp = self.compute_cost(&rho_plus).unwrap_or(f64::INFINITY);
                    let fm = self.compute_cost(&rho_minus).unwrap_or(f64::INFINITY);
                    numeric_grad[k] = (fp - fm) / (2.0 * h);
                }

                report.analytic_gradient = Some(analytic_grad.clone());
                report.numeric_gradient = Some(numeric_grad.clone());

                // Compute per-component relative errors
                let mut rel_errors = Array1::<f64>::zeros(rho.len());
                for k in 0..rho.len() {
                    let denom = analytic_grad[k].abs().max(numeric_grad[k].abs()).max(1e-8);
                    rel_errors[k] = (analytic_grad[k] - numeric_grad[k]).abs() / denom;
                }
                report.component_rel_errors = Some(rel_errors);
            } else if report.has_issues() {
                log::debug!(
                    "[REML] skipping full FD gradient diagnostics at eval {} (sampled every {} evals unless severe).",
                    eval_idx,
                    GRAD_DIAG_FD_INTERVAL
                );
            }

            // === Output Summary (single print, not in a loop) ===
            if report.has_issues() {
                println!("\n[GRADIENT DIAGNOSTICS] Issues detected:");
                println!("{}", report.summary());

                // Also log total gradient comparison
                if let (Some(analytic), Some(numeric)) =
                    (&report.analytic_gradient, &report.numeric_gradient)
                {
                    let diff = analytic - numeric;
                    let rel_l2 = diff.dot(&diff).sqrt() / numeric.dot(numeric).sqrt().max(1e-8);
                    println!(
                        "[GRADIENT DIAGNOSTICS] Total gradient rel. L2 error: {:.2e}",
                        rel_l2
                    );
                }
            }
        }

        /// Implements the stable re-parameterization algorithm from Wood (2011) Appendix B
        /// This replaces naive summation S_λ = Σ λᵢSᵢ with similarity transforms
        /// to avoid "dominant machine zero leakage" between penalty components
        ///
        // Helper for boundary perturbation
        // Returns (perturbed_rho, optional_corrected_covariance_in_transformed_basis)
        // The covariance is V'_beta_trans
        #[allow(dead_code)]
        pub(super) fn perform_boundary_perturbation_correction(
            &self,
            initial_rho: &Array1<f64>,
        ) -> Result<(Array1<f64>, Option<Array2<f64>>), EstimationError> {
            // 1. Identify boundary parameters and perturb
            let mut current_rho = initial_rho.clone();
            let mut perturbed = false;

            // Target cost increase: 0.01 log-likelihood units (statistically insignificant)
            let target_diff = 0.01;

            for k in 0..current_rho.len() {
                // Check if at upper boundary (high smoothing -> linear)
                // RHO_BOUND is 30.0.
                if current_rho[k] > RHO_BOUND - 1.0 {
                    // Compute base_cost fresh for each parameter to handle multiple boundary cases
                    let base_cost = self.compute_cost(&current_rho)?;

                    log::info!(
                        "[Boundary] rho[{}] = {:.2} is at boundary. Perturbing...",
                        k,
                        current_rho[k]
                    );

                    // Search inwards (decreasing rho)
                    // We want delta > 0 such that Cost(rho - delta) approx Base + 0.01
                    let mut lower = 0.0;
                    let mut upper = 15.0;
                    let mut best_delta = 0.0;

                    // Initial check: if upper is not enough, just take upper
                    let mut rho_test = current_rho.clone();
                    rho_test[k] -= upper;
                    if let Ok(c) = self.compute_cost(&rho_test) {
                        if (c - base_cost).abs() < target_diff {
                            // Even big change doesn't change cost much?
                            // This implies extremely flat surface. Just move away from boundary significantly.
                            best_delta = upper;
                        }
                    }

                    if best_delta == 0.0 {
                        // Bisection
                        for _ in 0..15 {
                            let mid = (lower + upper) * 0.5;
                            rho_test[k] = current_rho[k] - mid;
                            if let Ok(c) = self.compute_cost(&rho_test) {
                                let diff = c - base_cost;
                                if diff < target_diff {
                                    // Need more change -> larger delta
                                    lower = mid;
                                } else {
                                    // Too much change -> smaller delta
                                    upper = mid;
                                }
                            } else {
                                // Error computing cost, assume strictly worse (too far?)
                                upper = mid;
                            }
                        }
                        best_delta = (lower + upper) * 0.5;
                    }

                    current_rho[k] -= best_delta;
                    perturbed = true;
                    log::info!(
                        "[Boundary] rho[{}] moved to {:.2} (delta={:.3})",
                        k,
                        current_rho[k],
                        best_delta
                    );
                }
            }

            if !perturbed {
                return Ok((current_rho, None));
            }

            let n_rho = current_rho.len();
            let mut laml_hessian = match self.compute_laml_hessian_consistent(&current_rho) {
                Ok(h) => h,
                Err(err) => {
                    log::warn!(
                        "Boundary Hessian unavailable ({}); falling back to FD Hessian.",
                        err
                    );
                    let h_step = 1e-4;
                    let mut h_fd = Array2::<f64>::zeros((n_rho, n_rho));
                    let grad_center = self.compute_gradient(&current_rho)?;
                    for j in 0..n_rho {
                        let mut rho_plus = current_rho.clone();
                        rho_plus[j] += h_step;
                        let grad_plus = self.compute_gradient(&rho_plus)?;
                        let col_diff = (&grad_plus - &grad_center) / h_step;
                        for i in 0..n_rho {
                            h_fd[[i, j]] = col_diff[i];
                        }
                    }
                    for i in 0..n_rho {
                        for j in 0..i {
                            let avg = 0.5 * (h_fd[[i, j]] + h_fd[[j, i]]);
                            h_fd[[i, j]] = avg;
                            h_fd[[j, i]] = avg;
                        }
                    }
                    h_fd
                }
            };

            // Invert local Hessian to obtain V_ρ.
            // Stabilization ridge is applied before Cholesky to control near-singularity
            // in weakly identified smoothing directions.
            let mut v_rho = Array2::<f64>::zeros((n_rho, n_rho));
            {
                use crate::faer_ndarray::{FaerArrayView, array2_to_mat_mut};
                use faer::Side;

                // Ensure PD
                crate::pirls::ensure_positive_definite_with_label(
                    &mut laml_hessian,
                    "LAML Hessian",
                )?;

                let h_view = FaerArrayView::new(&laml_hessian);
                if let Ok(chol) = faer::linalg::solvers::Llt::new(h_view.as_ref(), Side::Lower) {
                    let mut eye = Array2::<f64>::eye(n_rho);
                    let mut eye_view = array2_to_mat_mut(&mut eye);
                    chol.solve_in_place(eye_view.as_mut());
                    v_rho.assign(&eye);
                } else {
                    // Fallback: SVD or pseudoinverse? Or just fail correction.
                    log::warn!(
                        "LAML Hessian not invertible even after stabilization. Skipping correction."
                    );
                    return Ok((current_rho, None));
                }
            }

            // 3. Compute smoothing-parameter uncertainty correction: J * V_rho * J^T.
            //
            // Notation mapping to the exact Gaussian-mixture identity:
            //   rho ~ N(mu, Sigma),  mu = rho_hat,  Sigma = V_rho
            //   A(rho) = H_rho^{-1},  b(rho) = beta_hat_rho
            //   Var(beta) = E[A(rho)] + Var(b(rho))   (exact, no truncation)
            //
            // This implementation uses the standard first-order truncation around mu:
            //   E[A(rho)]      ≈ A(mu) = H_p^{-1} = V_beta_cond
            //   Var(b(rho))    ≈ J * V_rho * J^T,  J = dbeta_hat/drho |_{rho=mu}
            // so:
            //   V_total ≈ V_beta_cond + J * V_rho * J^T.
            //
            // Exact higher-order terms from the heat-operator / Wick expansion are
            // not included here.
            //
            // Jacobian identity used here:
            //   d(beta_hat)/d(rho_k) = -H_p^{-1}(S_k^rho * beta_hat), S_k^rho = lambda_k S_k.
            // This is the same implicit derivative used in the main gradient code.

            // We need H_p and beta at the perturbed rho.
            let pirls_res = self.execute_pirls_if_needed(&current_rho)?;

            let beta = pirls_res.beta_transformed.as_ref();
            let h_p = &pirls_res.penalized_hessian_transformed;
            let lambdas = current_rho.mapv(f64::exp);
            let rs = &pirls_res.reparam_result.rs_transformed;

            let p_dim = beta.len();

            // Invert H_p to get V_beta_cond = H_p^{-1}, i.e. A(mu) in the
            // first-order approximation above.
            let mut v_beta_cond = Array2::<f64>::zeros((p_dim, p_dim));
            {
                use crate::faer_ndarray::{FaerArrayView, array2_to_mat_mut};
                use faer::Side;
                let h_view = FaerArrayView::new(h_p);
                // At convergence H_p is typically PD.
                if let Ok(chol) = faer::linalg::solvers::Llt::new(h_view.as_ref(), Side::Lower) {
                    let mut eye = Array2::<f64>::eye(p_dim);
                    let mut eye_view = array2_to_mat_mut(&mut eye);
                    chol.solve_in_place(eye_view.as_mut());
                    v_beta_cond.assign(&eye);
                } else {
                    // Use LDLT if LLT fails
                    if let Ok(ldlt) = faer::linalg::solvers::Ldlt::new(h_view.as_ref(), Side::Lower)
                    {
                        let mut eye = Array2::<f64>::eye(p_dim);
                        let mut eye_view = array2_to_mat_mut(&mut eye);
                        ldlt.solve_in_place(eye_view.as_mut());
                        v_beta_cond.assign(&eye);
                    } else {
                        log::warn!("Penalized Hessian not invertible. Skipping correction.");
                        return Ok((current_rho, None));
                    }
                }
            }

            // Compute Jacobian columns:
            //   J[:,k] = -H_p^{-1}(S_k^ρ β̂)
            //          = -V_beta_cond * (S_k β̂ * λ_k)
            // with S_k β̂ assembled as R_kᵀ(R_k β̂).
            // S_k = R_k^T R_k.
            let mut jacobian = Array2::<f64>::zeros((p_dim, n_rho));

            for k in 0..n_rho {
                let r_k = &rs[k];
                if r_k.ncols() == 0 {
                    continue;
                }

                let lambda = lambdas[k];
                // S_k beta = R_k^T (R_k beta)
                let r_beta = r_k.dot(beta);
                let s_beta = r_k.t().dot(&r_beta);

                let term = s_beta.mapv(|v| v * lambda);

                // col = - V_beta_cond * term
                let col = v_beta_cond.dot(&term).mapv(|v| -v);

                jacobian.column_mut(k).assign(&col);
            }

            // V_corr approximates Var(b(rho)) under first-order linearization.
            // V_corr = J * V_rho * J^T.
            let temp = jacobian.dot(&v_rho); // (p, k) * (k, k) -> (p, k)
            let v_corr = temp.dot(&jacobian.t()); // (p, k) * (k, p) -> (p, p)

            log::info!(
                "[Boundary] Correction computed. Max element in V_corr: {:.3e}",
                v_corr.iter().fold(0.0_f64, |a, &b| a.max(b.abs()))
            );

            // First-order total covariance approximation to Var(beta).
            let v_total = v_beta_cond + v_corr;

            Ok((current_rho, Some(v_total)))
        }
    }
