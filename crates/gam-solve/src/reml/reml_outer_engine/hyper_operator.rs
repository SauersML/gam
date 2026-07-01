use super::*;

pub(crate) fn as_implicit(op: &dyn HyperOperator) -> Option<&ImplicitHyperOperator> {
    op.as_any().downcast_ref::<ImplicitHyperOperator>()
}

pub(crate) fn as_composite(op: &dyn HyperOperator) -> Option<&CompositeHyperOperator> {
    op.as_any().downcast_ref::<CompositeHyperOperator>()
}

pub(crate) fn as_weighted(op: &dyn HyperOperator) -> Option<&WeightedHyperOperator> {
    op.as_any().downcast_ref::<WeightedHyperOperator>()
}

pub(crate) trait DriftDerivTraceExt {
    fn trace_logdet(&self, hop: &dyn HessianOperator) -> f64;

    fn trace_logdet_hessian_cross(&self, rhs: &Self, hop: &dyn HessianOperator) -> f64;
}

impl DriftDerivTraceExt for DriftDerivResult {
    fn trace_logdet(&self, hop: &dyn HessianOperator) -> f64 {
        match self {
            Self::Dense(matrix) => hop.trace_logdet_gradient(matrix),
            Self::Operator(operator) => hop.trace_logdet_operator(operator.as_ref()),
        }
    }

    fn trace_logdet_hessian_cross(&self, rhs: &Self, hop: &dyn HessianOperator) -> f64 {
        match (self, rhs) {
            (Self::Dense(left), Self::Dense(right)) => hop.trace_logdet_hessian_cross(left, right),
            (Self::Dense(left), Self::Operator(right)) => {
                hop.trace_logdet_hessian_cross_matrix_operator(left, right.as_ref())
            }
            (Self::Operator(left), Self::Dense(right)) => {
                hop.trace_logdet_hessian_cross_matrix_operator(right, left.as_ref())
            }
            (Self::Operator(left), Self::Operator(right)) => {
                hop.trace_logdet_hessian_cross_operator(left.as_ref(), right.as_ref())
            }
        }
    }
}

#[derive(Clone)]
pub struct CompositeHyperOperator {
    pub dense: Option<Array2<f64>>,
    pub operators: Vec<Arc<dyn HyperOperator>>,
    pub dim_hint: usize,
}

/// Group composite operators by shared `(implicit_deriv, x_design, w_diag)`
/// so every Duchon ψ-axis built atop the same implicit derivative runs
/// through a single row-kernel sweep via
/// `trace_projected_factor_all_axes_with_xf`. Per-axis `s_psi` and
/// `c_x_psi_beta` are threaded in individually so the batched path matches
/// the per-axis path exactly. Non-implicit operators and singleton groups
/// fall through to the original per-op trace path.
pub(crate) fn composite_trace_implicit_batched(
    operators: &[Arc<dyn HyperOperator>],
    factor: &Array2<f64>,
    cache: Option<&ProjectedFactorCache>,
) -> f64 {
    let mut trace = 0.0;
    let mut group_starts: Vec<Vec<usize>> = Vec::new();
    let mut handled = vec![false; operators.len()];

    for (i, op) in operators.iter().enumerate() {
        if handled[i] {
            continue;
        }
        let Some(impl_i) = as_implicit(op.as_ref()) else {
            continue;
        };
        let mut group = vec![i];
        handled[i] = true;
        for j in (i + 1)..operators.len() {
            if handled[j] {
                continue;
            }
            if let Some(impl_j) = as_implicit(operators[j].as_ref())
                && Arc::ptr_eq(&impl_i.implicit_deriv, &impl_j.implicit_deriv)
                && Arc::ptr_eq(&impl_i.x_design, &impl_j.x_design)
                && Arc::ptr_eq(impl_i.w_diag.as_arc(), impl_j.w_diag.as_arc())
                && impl_i.p == impl_j.p
            {
                group.push(j);
                handled[j] = true;
            }
        }
        group_starts.push(group);
    }

    for group in &group_starts {
        if group.len() >= 2 {
            let lead = as_implicit(operators[group[0]].as_ref()).unwrap();
            let xf = match cache {
                Some(c) => lead.cached_xf(factor, c),
                None => Arc::new(lead.compute_xf(factor)),
            };
            let axes: Vec<(usize, &Array2<f64>, Option<&Array1<f64>>)> = group
                .iter()
                .map(|&k| {
                    let op = as_implicit(operators[k].as_ref()).unwrap();
                    (op.axis, &op.s_psi, op.c_x_psi_beta.as_deref())
                })
                .collect();
            let values = lead.trace_projected_factor_all_axes_with_xf(factor, xf.view(), &axes);
            trace += values.iter().sum::<f64>();
        } else {
            let op = &operators[group[0]];
            trace += match cache {
                Some(c) => op.trace_projected_factor_cached(factor, c),
                None => op.trace_projected_factor(factor),
            };
        }
    }

    for (i, op) in operators.iter().enumerate() {
        if handled[i] {
            continue;
        }
        trace += match cache {
            Some(c) => op.trace_projected_factor_cached(factor, c),
            None => op.trace_projected_factor(factor),
        };
    }

    trace
}

/// Vector form of the implicit-axis trace batching used by
/// [`CompositeHyperOperator`].  It returns one exact `tr(Fᵀ B_i F)` value per
/// input operator while sharing the expensive `X·F` projection and Duchon
/// row-kernel sweeps across sibling implicit ψ/ρ axes.
pub(crate) fn trace_projected_factors_batched(
    operators: &[Arc<dyn HyperOperator>],
    factor: &Array2<f64>,
    cache: &ProjectedFactorCache,
) -> Vec<f64> {
    let mut out = vec![0.0; operators.len()];
    let mut handled = vec![false; operators.len()];

    for i in 0..operators.len() {
        if handled[i] {
            continue;
        }
        let Some(impl_i) = as_implicit(operators[i].as_ref()) else {
            out[i] = operators[i].trace_projected_factor_cached(factor, cache);
            handled[i] = true;
            continue;
        };

        let mut group = vec![i];
        handled[i] = true;
        for j in (i + 1)..operators.len() {
            if handled[j] {
                continue;
            }
            if let Some(impl_j) = as_implicit(operators[j].as_ref())
                && Arc::ptr_eq(&impl_i.implicit_deriv, &impl_j.implicit_deriv)
                && Arc::ptr_eq(&impl_i.x_design, &impl_j.x_design)
                && Arc::ptr_eq(impl_i.w_diag.as_arc(), impl_j.w_diag.as_arc())
                && impl_i.p == impl_j.p
            {
                group.push(j);
                handled[j] = true;
            }
        }

        if group.len() >= 2 {
            let xf = impl_i.cached_xf(factor, cache);
            let axes: Vec<(usize, &Array2<f64>, Option<&Array1<f64>>)> = group
                .iter()
                .map(|&idx| {
                    let op = as_implicit(operators[idx].as_ref()).unwrap();
                    (op.axis, &op.s_psi, op.c_x_psi_beta.as_deref())
                })
                .collect();
            let values = impl_i.trace_projected_factor_all_axes_with_xf(factor, xf.view(), &axes);
            for (&idx, value) in group.iter().zip(values) {
                out[idx] = value;
            }
        } else {
            out[i] = operators[i].trace_projected_factor_cached(factor, cache);
        }
    }

    out
}

pub(crate) fn collect_projected_trace_terms<'a>(
    out_idx: usize,
    weight: f64,
    op: &'a dyn HyperOperator,
    factor: &Array2<f64>,
    dense_acc: &mut [f64],
    terms: &mut Vec<(usize, f64, &'a dyn HyperOperator)>,
) {
    if weight == 0.0 {
        return;
    }
    if let Some(composite) = as_composite(op) {
        if let Some(dense) = composite.dense.as_ref() {
            dense_acc[out_idx] += weight * dense_trace_projected_factor(dense, factor);
        }
        for inner in &composite.operators {
            collect_projected_trace_terms(
                out_idx,
                weight,
                inner.as_ref(),
                factor,
                dense_acc,
                terms,
            );
        }
    } else if let Some(weighted) = as_weighted(op) {
        for (term_weight, inner) in &weighted.terms {
            collect_projected_trace_terms(
                out_idx,
                weight * *term_weight,
                inner.as_ref(),
                factor,
                dense_acc,
                terms,
            );
        }
    } else {
        terms.push((out_idx, weight, op));
    }
}

pub(crate) fn collect_projected_matrix_terms<'a>(
    out_idx: usize,
    weight: f64,
    op: &'a dyn HyperOperator,
    factor: &Array2<f64>,
    dense_acc: &mut [Array2<f64>],
    terms: &mut Vec<(usize, f64, &'a dyn HyperOperator)>,
) {
    if weight == 0.0 {
        return;
    }
    if let Some(composite) = as_composite(op) {
        if let Some(dense) = composite.dense.as_ref() {
            dense_acc[out_idx].scaled_add(weight, &dense_projected_matrix(dense, factor));
        }
        for inner in &composite.operators {
            collect_projected_matrix_terms(
                out_idx,
                weight,
                inner.as_ref(),
                factor,
                dense_acc,
                terms,
            );
        }
    } else if let Some(weighted) = as_weighted(op) {
        for (term_weight, inner) in &weighted.terms {
            collect_projected_matrix_terms(
                out_idx,
                weight * *term_weight,
                inner.as_ref(),
                factor,
                dense_acc,
                terms,
            );
        }
    } else {
        terms.push((out_idx, weight, op));
    }
}

pub(crate) fn trace_projected_operator_terms_batched(
    n_out: usize,
    terms: &[(usize, f64, &dyn HyperOperator)],
    factor: &Array2<f64>,
    cache: &ProjectedFactorCache,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_out];
    let mut handled = vec![false; terms.len()];

    for i in 0..terms.len() {
        if handled[i] {
            continue;
        }
        let Some(impl_i) = as_implicit(terms[i].2) else {
            continue;
        };
        let mut group = vec![i];
        handled[i] = true;
        for j in (i + 1)..terms.len() {
            if handled[j] {
                continue;
            }
            if let Some(impl_j) = as_implicit(terms[j].2)
                && Arc::ptr_eq(&impl_i.implicit_deriv, &impl_j.implicit_deriv)
                && Arc::ptr_eq(&impl_i.x_design, &impl_j.x_design)
                && Arc::ptr_eq(impl_i.w_diag.as_arc(), impl_j.w_diag.as_arc())
                && impl_i.p == impl_j.p
            {
                group.push(j);
                handled[j] = true;
            }
        }

        let lead = as_implicit(terms[group[0]].2).unwrap();
        let xf = lead.cached_xf(factor, cache);
        let axes: Vec<(usize, &Array2<f64>, Option<&Array1<f64>>)> = group
            .iter()
            .map(|&term_idx| {
                let op = as_implicit(terms[term_idx].2).unwrap();
                (op.axis, &op.s_psi, op.c_x_psi_beta.as_deref())
            })
            .collect();
        let values = lead.trace_projected_factor_all_axes_with_xf(factor, xf.view(), &axes);
        for (&term_idx, value) in group.iter().zip(values.iter()) {
            let (out_idx, weight, _) = terms[term_idx];
            out[out_idx] += weight * *value;
        }
    }

    for (i, (out_idx, weight, op)) in terms.iter().enumerate() {
        if handled[i] {
            continue;
        }
        out[*out_idx] += *weight * op.trace_projected_factor_cached(factor, cache);
    }

    out
}

pub(crate) fn projected_operator_terms_batched(
    n_out: usize,
    terms: &[(usize, f64, &dyn HyperOperator)],
    factor: &Array2<f64>,
    cache: &ProjectedFactorCache,
) -> Vec<Array2<f64>> {
    let rank = factor.ncols();
    let mut out: Vec<Array2<f64>> = (0..n_out)
        .map(|_| Array2::<f64>::zeros((rank, rank)))
        .collect();
    for (out_idx, weight, op) in terms.iter() {
        let projected = op.projected_matrix_cached(factor, cache);
        out[*out_idx].scaled_add(*weight, &projected);
    }
    out
}

pub(crate) fn project_hyper_operators_batched(
    n_out: usize,
    terms: &[(usize, f64, &dyn HyperOperator)],
    factor: &Array2<f64>,
    cache: &ProjectedFactorCache,
) -> Vec<Array2<f64>> {
    projected_operator_terms_batched(n_out, terms, factor, cache)
}

pub(crate) fn trace_logdet_drifts_projected_factor_batched(
    drifts: &[DriftDerivResult],
    factor: &Array2<f64>,
    cache: &ProjectedFactorCache,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; drifts.len()];
    let mut terms: Vec<(usize, f64, &dyn HyperOperator)> = Vec::new();
    for (idx, drift) in drifts.iter().enumerate() {
        match drift {
            DriftDerivResult::Dense(matrix) => {
                out[idx] += dense_trace_projected_factor(matrix, factor);
            }
            DriftDerivResult::Operator(op) => {
                collect_projected_trace_terms(idx, 1.0, op.as_ref(), factor, &mut out, &mut terms);
            }
        }
    }
    let batched = trace_projected_operator_terms_batched(drifts.len(), &terms, factor, cache);
    for (dst, value) in out.iter_mut().zip(batched) {
        *dst += value;
    }
    out
}

pub(crate) fn dense_spectral_trace_logdet_drifts_batched(
    ds: &DenseSpectralOperator,
    drifts: &[DriftDerivResult],
) -> Vec<f64> {
    trace_logdet_drifts_projected_factor_batched(drifts, &ds.g_factor, &ds.projected_factor_cache)
}

pub(crate) fn penalty_subspace_trace_factor(kernel: &PenaltySubspaceTrace) -> Array2<f64> {
    let (evals, evecs) = kernel
        .h_proj_inverse
        .eigh(faer::Side::Lower)
        .expect("PenaltySubspaceTrace kernel factor eigendecomposition failed");
    let r = evals.len();
    // F must satisfy F·Fᵀ = K exactly: the batched `tr(FᵀAF)` is consumed as
    // the gradient of the SAME pseudo-logdet criterion whose exact kernel the
    // per-coordinate path contracts via `h_proj_inverse` directly. The kernel
    // eigenvalues are `1/σ_a` over the kept Hessian spectrum, so their
    // dynamic range is the Hessian condition number — clamp ONLY the
    // roundoff-negative tail to zero (K is PSD by construction; a negative
    // eigenvalue is O(ε)·‖K‖ eigensolver noise, and √(max(λ,0)) is the
    // honest PSD square root). A relative floor here is NOT a stabilization:
    // raising `1/σ_max` to `√ε·r·(1/σ_min)` rewrites the criterion's
    // sensitivity along exactly the stiffest directions — where the ρ-drifts
    // `λ_k·S_k` live — inflating the analytic trace by up to `√ε·r·κ(H_pen)`
    // (O(1) once κ ≳ 1e7) while FD differentiates the true criterion. That
    // desync red-lined every iso-κ Duchon probit/logit FD test and starved
    // the spatial κ-optimizer of descent directions; Gaussian was immune
    // because the intrinsic kernel is only installed for c-nontrivial
    // families (#901).
    let mut root = evecs.clone();
    for col in 0..r {
        let scale = evals[col].max(0.0).sqrt();
        for row in 0..r {
            root[[row, col]] *= scale;
        }
    }
    gam_linalg::faer_ndarray::fast_ab(&kernel.u_s, &root)
}

pub(crate) fn penalty_subspace_trace_drifts_batched(
    kernel: &PenaltySubspaceTrace,
    drifts: &[DriftDerivResult],
) -> Vec<f64> {
    drifts
        .iter()
        .map(|drift| {
            match drift {
                // Use the canonical reduced-kernel contraction for dense drifts
                // so the projected logdet value and its trace derivative share
                // exactly the same eigenbasis/inverse. This is essential for
                // composite drifts `B_i + D_βH[β_i]`: evaluating the dense
                // component through a separately factorized square root can
                // make the projected logdet value and gradient describe
                // slightly different kernels.
                DriftDerivResult::Dense(matrix) => kernel.trace_projected_logdet(matrix),
                DriftDerivResult::Operator(op) => kernel.trace_operator(op.as_ref()),
            }
        })
        .collect()
}

pub(crate) fn penalty_subspace_reduce_drifts_batched(
    kernel: &PenaltySubspaceTrace,
    drifts: &[DriftDerivResult],
) -> Vec<Array2<f64>> {
    drifts
        .iter()
        .map(|drift| match drift {
            DriftDerivResult::Dense(matrix) => kernel.reduce(matrix),
            // #901 layer-2 (outer-Hessian path): reduce the operator via
            // `U_Sᵀ·A·U_S = U_Sᵀ·A.mul_mat(U_S)` — NOT `op.to_dense()` then
            // reduce. For the GLM cubic correction `C[v] = Xᵀdiag(c⊙Xv)X` the
            // dense materialization computes near-null quadratic forms by
            // cancelling O(‖C‖) entries, and the spectral kernel's `1/σ_min`
            // then amplifies the roundoff (the +39-vs-−0.30 / ~−7.7e5 blow-up).
            // `reduce_operator` probes through the `X·U_S` matvecs instead, so
            // tiny² stays tiny — the same stability cure as the first-order
            // `trace_operator` path.
            DriftDerivResult::Operator(op) => kernel.reduce_operator(op.as_ref()),
        })
        .collect()
}

pub(crate) fn dense_spectral_trace_logdet_operators_batched(
    ds: &DenseSpectralOperator,
    operators: &[Arc<dyn HyperOperator>],
) -> Vec<f64> {
    if operators.is_empty() {
        return Vec::new();
    }
    if log::log_enabled!(log::Level::Info) {
        let start = std::time::Instant::now();
        let out =
            trace_projected_factors_batched(operators, &ds.g_factor, &ds.projected_factor_cache);
        let implicit_count = operators.iter().filter(|op| op.is_implicit()).count();
        dense_spectral_stage_log(
            &format!(
                "DenseSpectralOperator::trace_logdet_operators_batched dim={} rank={} ops={} implicit_ops={}",
                ds.n_dim,
                ds.g_factor.ncols(),
                operators.len(),
                implicit_count,
            ),
            start.elapsed().as_secs_f64(),
        );
        out
    } else {
        trace_projected_factors_batched(operators, &ds.g_factor, &ds.projected_factor_cache)
    }
}

impl HyperOperator for CompositeHyperOperator {
    fn as_any(&self) -> &(dyn std::any::Any + 'static) {
        self
    }

    fn dim(&self) -> usize {
        self.dim_hint
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v.view(), out.view_mut());
        out
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v, out.view_mut());
        out
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        if self.dense.is_none() && self.operators.len() == 1 {
            self.operators[0].mul_vec_into(v, out);
            return;
        }

        out.fill(0.0);
        if let Some(dense) = self.dense.as_ref() {
            dense_matvec_into(dense, v, out.view_mut());
        }
        for op in &self.operators {
            op.scaled_add_mul_vec(v, 1.0, out.view_mut());
        }
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        if self.dense.is_none() && self.operators.len() == 1 {
            self.operators[0].mul_basis_columns_into(start, out);
            return;
        }

        out.fill(0.0);
        let cols = out.ncols();
        let end = start + cols;
        if let Some(dense) = self.dense.as_ref() {
            out += &dense.slice(ndarray::s![.., start..end]);
        }
        let mut work = Array2::<f64>::zeros((out.nrows(), cols));
        for op in &self.operators {
            op.mul_basis_columns_into(start, work.view_mut());
            out += &work;
        }
    }

    fn scaled_add_mul_vec(
        &self,
        v: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        if scale == 0.0 {
            return;
        }
        if self.dense.is_none() && self.operators.len() == 1 {
            self.operators[0].scaled_add_mul_vec(v, scale, out);
            return;
        }

        if let Some(dense) = self.dense.as_ref() {
            dense_matvec_scaled_add_into(dense, v, scale, out.view_mut());
        }
        for op in &self.operators {
            op.scaled_add_mul_vec(v, scale, out.view_mut());
        }
    }

    /// Forward batched apply to inner operators so their `mul_mat` overrides
    /// (matrix-free Khatri–Rao BLAS3 fuses) fire instead of the default
    /// per-column parallel matvec — which would triple-nest rayon when an
    /// inner op already parallelizes internally.
    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        if self.dense.is_none() && self.operators.len() == 1 {
            return self.operators[0].mul_mat(factor);
        }
        let p = factor.nrows();
        let k = factor.ncols();
        let mut out = Array2::<f64>::zeros((p, k));
        if let Some(dense) = self.dense.as_ref() {
            out += &dense.dot(factor);
        }
        for op in &self.operators {
            out += &op.mul_mat(factor);
        }
        out
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        if self.dense.is_none() && self.operators.len() == 1 {
            return self.operators[0].trace_projected_factor(factor);
        }

        let mut trace = 0.0;
        if let Some(dense) = self.dense.as_ref() {
            let dense_factor = dense.dot(factor);
            trace += factor
                .iter()
                .zip(dense_factor.iter())
                .map(|(&f, &bf)| f * bf)
                .sum::<f64>();
        }
        trace += composite_trace_implicit_batched(&self.operators, factor, None);
        trace
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        if self.dense.is_none() && self.operators.len() == 1 {
            return self.operators[0].trace_projected_factor_cached(factor, cache);
        }

        let mut trace = 0.0;
        if let Some(dense) = self.dense.as_ref() {
            let dense_factor = dense.dot(factor);
            trace += factor
                .iter()
                .zip(dense_factor.iter())
                .map(|(&f, &bf)| f * bf)
                .sum::<f64>();
        }
        trace += composite_trace_implicit_batched(&self.operators, factor, Some(cache));
        trace
    }

    fn projected_matrix(&self, factor: &Array2<f64>) -> Array2<f64> {
        if self.dense.is_none() && self.operators.len() == 1 {
            return self.operators[0].projected_matrix(factor);
        }

        let rank = factor.ncols();
        let mut projected = Array2::<f64>::zeros((rank, rank));
        if let Some(dense) = self.dense.as_ref() {
            let mf = gam_linalg::faer_ndarray::fast_ab(dense, factor);
            projected += &gam_linalg::faer_ndarray::fast_atb(factor, &mf);
        }
        for op in &self.operators {
            projected += &op.projected_matrix(factor);
        }
        projected
    }

    fn projected_matrix_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> Array2<f64> {
        if self.dense.is_none() && self.operators.len() == 1 {
            return self.operators[0].projected_matrix_cached(factor, cache);
        }

        let rank = factor.ncols();
        let mut projected = Array2::<f64>::zeros((rank, rank));
        if let Some(dense) = self.dense.as_ref() {
            let mf = gam_linalg::faer_ndarray::fast_ab(dense, factor);
            projected += &gam_linalg::faer_ndarray::fast_atb(factor, &mf);
        }
        for op in &self.operators {
            projected += &op.projected_matrix_cached(factor, cache);
        }
        projected
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let mut total = 0.0;
        if let Some(dense) = self.dense.as_ref() {
            total += dense_bilinear(dense, v.view(), u.view());
        }
        for op in &self.operators {
            total += op.bilinear(v, u);
        }
        total
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        let mut total = 0.0;
        if let Some(dense) = self.dense.as_ref() {
            total += dense_bilinear(dense, v, u);
        }
        for op in &self.operators {
            total += op.bilinear_view(v, u);
        }
        total
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = self
            .dense
            .clone()
            .unwrap_or_else(|| Array2::<f64>::zeros((self.dim_hint, self.dim_hint)));
        for op in &self.operators {
            out += &op.to_dense();
        }
        out
    }

    fn is_implicit(&self) -> bool {
        self.operators.iter().any(|op| op.is_implicit())
    }
}

/// Implicit Hessian-drift operator for a single anisotropic ψ_d coordinate.
///
/// Computes B_d · v on the fly:
///   B_d · v = (∂X/∂ψ_d)^T (W · (X · v)) + X^T (W · ((∂X/∂ψ_d) · v)) + S_{ψ_d} · v
///
/// The first two terms use the implicit design-derivative operator (no dense
/// (n × p) matrices), and S_{ψ_d} is a dense (p × p) penalty matrix (manageable).
///
/// Storage: the implicit operator holds O(n·k·D) radial jets, plus references
/// to an active-basis X design operator and W (the working weights). The
/// penalty matrix S_{ψ_d} is stored as a dense (p × p) matrix.
/// Thread-local scratch buffers for `ImplicitHyperOperator::mul_vec_into`.
/// Reused across PCG iterations and basis-column sweeps so each matvec
/// avoids three fresh O(n)/O(p) allocations.
mod implicit_matvec_scratch {
    use std::cell::RefCell;

    pub(super) struct Scratch {
        pub x_v: Vec<f64>,
        pub n_work: Vec<f64>,
        pub p_work: Vec<f64>,
    }

    impl Scratch {
        pub(crate) const fn new() -> Self {
            Self {
                x_v: Vec::new(),
                n_work: Vec::new(),
                p_work: Vec::new(),
            }
        }
    }

    thread_local! {
        static SCRATCH: RefCell<Scratch> = const { RefCell::new(Scratch::new()) };
    }

    pub(super) fn with<R>(f: impl FnOnce(&mut Scratch) -> R) -> R {
        SCRATCH.with(|cell| f(&mut cell.borrow_mut()))
    }
}

pub struct ImplicitHyperOperator {
    /// The implicit design-derivative operator (shared across all axes).
    pub implicit_deriv: std::sync::Arc<gam_terms::basis::ImplicitDesignPsiDerivative>,
    /// Which axis this operator is for.
    pub axis: usize,
    /// The active-basis design matrix X. This may be lazy / operator-backed.
    pub(crate) x_design: std::sync::Arc<DesignMatrix>,
    /// Working weights W (diagonal, length n) — observed-information curvature,
    /// signed for non-canonical links. Carried as the owned [`gam_linalg::matrix::SignedWeightsArc`]
    /// newtype so the sign character is construction-enforced at the operator
    /// struct boundary; the function-boundary contract from `linalg/matrix.rs`
    /// is no longer reconstructable accidentally inside `mul_vec`.
    pub(crate) w_diag: gam_linalg::matrix::SignedWeightsArc,
    /// Penalty derivative matrix S_{ψ_d} (p × p), dense.
    pub s_psi: Array2<f64>,
    /// Total basis dimension p.
    pub(crate) p: usize,
    /// Non-Gaussian fixed-β third-derivative correction: c ⊙ (X_{ψ_d} β̂),
    /// length n. When present, the operator additionally applies
    /// `Xᵀ diag(c_x_psi_beta) X v` so that the full B_d formula
    /// `B_d v = (∂X/∂ψ_d)ᵀ W X v + Xᵀ W (∂X/∂ψ_d) v + Xᵀ diag(c ⊙ X_{ψ_d} β̂) X v + S_{ψ_d} v`
    /// is matrix-free for non-Gaussian likelihoods. `None` for Gaussian
    /// identity (c ≡ 0 there).
    pub c_x_psi_beta: Option<std::sync::Arc<Array1<f64>>>,
}

impl HyperOperator for ImplicitHyperOperator {
    fn dim(&self) -> usize {
        self.p
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        // Single canonical path: route every matvec through `mul_vec_into`,
        // which routes through `matvec_with_shared_xz_into`. The four terms of
        // B_d are assembled there, with the third-derivative correction added
        // by `accumulate_c_correction_xt_into` so the four matvec entry points
        // share one inner kernel.
        let mut out = Array1::<f64>::zeros(self.p);
        self.mul_vec_into(v.view(), out.view_mut());
        out
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.p);
        self.mul_vec_into(v, out.view_mut());
        out
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, out: ArrayViewMut1<'_, f64>) {
        assert_eq!(v.len(), self.p);
        let n_obs = self.w_diag.len();
        // Reuse thread-local scratch across repeated matvec calls (e.g.
        // PCG iterations, basis-column sweeps) instead of allocating
        // (2 n_obs + p) f64s every time.
        implicit_matvec_scratch::with(|s| {
            s.x_v.clear();
            s.x_v.resize(n_obs, 0.0);
            s.n_work.clear();
            s.n_work.resize(n_obs, 0.0);
            s.p_work.clear();
            s.p_work.resize(self.p, 0.0);
            let mut x_v_view = ndarray::ArrayViewMut1::from(s.x_v.as_mut_slice());
            let n_work_view = ndarray::ArrayViewMut1::from(s.n_work.as_mut_slice());
            let p_work_view = ndarray::ArrayViewMut1::from(s.p_work.as_mut_slice());
            design_matrix_apply_view_into(&self.x_design, v, x_v_view.view_mut());
            self.matvec_with_shared_xz_into(x_v_view.view(), v, out, n_work_view, p_work_view);
        });
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        let cols = out.ncols();
        assert!(start + cols <= self.p);

        let n_obs = self.w_diag.len();
        let mut basis = Array1::<f64>::zeros(self.p);
        let mut x_col = Array1::<f64>::zeros(n_obs);
        let mut dx_col = Array1::<f64>::zeros(n_obs);
        let mut weighted = Array1::<f64>::zeros(n_obs);
        let mut term = Array1::<f64>::zeros(self.p);

        for local_col in 0..cols {
            let global_col = start + local_col;
            let mut out_col = out.column_mut(local_col);
            out_col.assign(&self.s_psi.column(global_col));

            design_matrix_column_into(&self.x_design, global_col, x_col.view_mut());
            Zip::from(weighted.view_mut())
                .and(self.w_diag.view())
                .and(x_col.view())
                .par_for_each(|dst, &w, &x| *dst = w * x);
            term.assign(
                &self
                    .implicit_deriv
                    .transpose_mul(self.axis, &weighted.view())
                    .expect("radial scalar evaluation failed during implicit hyper transpose_mul"),
            );
            out_col += &term;

            basis[global_col] = 1.0;
            dx_col.assign(
                &self
                    .implicit_deriv
                    .forward_mul(self.axis, &basis.view())
                    .expect("radial scalar evaluation failed during implicit hyper forward_mul"),
            );
            basis[global_col] = 0.0;

            Zip::from(weighted.view_mut())
                .and(self.w_diag.view())
                .and(dx_col.view())
                .par_for_each(|dst, &w, &dx| *dst = w * dx);
            design_matrix_transpose_apply_view_into(
                &self.x_design,
                weighted.view(),
                term.view_mut(),
            );
            out_col += &term;

            // Non-Gaussian third-derivative correction column j: shared kernel.
            self.accumulate_c_correction_xt_into(
                x_col.view(),
                weighted.view_mut(),
                term.view_mut(),
                out_col,
            );
        }
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        self.bilinear_view(v.view(), u.view())
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        assert_eq!(v.len(), self.p);
        assert_eq!(u.len(), self.p);

        let x_v = design_matrix_apply_view(&self.x_design, v);
        let x_u = design_matrix_apply_view(&self.x_design, u);
        let dx_v = self
            .implicit_deriv
            .forward_mul(self.axis, &v)
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");
        let dx_u = self
            .implicit_deriv
            .forward_mul(self.axis, &u)
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");

        let w = &*self.w_diag;
        let mut design = 0.0;
        for i in 0..w.len() {
            design += dx_v[i] * w[i] * x_u[i];
            design += dx_u[i] * w[i] * x_v[i];
        }

        design += self.c_correction_bilinear(&x_v, &x_u);

        let penalty = dense_bilinear(&self.s_psi, v, u);

        design + penalty
    }

    fn is_implicit(&self) -> bool {
        true
    }

    fn as_any(&self) -> &(dyn std::any::Any + 'static) {
        self
    }

    /// Compute `tr(F^T B F)` directly via fused chunked BLAS3 GEMMs on the
    /// shared X and the shared raw kernel matrix, bypassing the rank-many
    /// separate matvecs the default impl would run through the lazy /
    /// operator-backed design.
    ///
    /// **Why this matters:** the default trait impl is
    ///   `let bf = self.mul_mat(F); (F ⊙ bf).sum()`
    /// which calls `mul_vec_into` per column of `F` (rank columns). On a
    /// lazy Duchon / Matérn / CTN design each `mul_vec_into` triggers a
    /// full `O(n · p · kernel_eval)` row-streamed matvec — and with rank ≈ p
    /// at large-scale shape (16D-Duchon-aniso 32 ψ-axes, p ≈ 95, n = 320 K)
    /// the per-axis trace landed at ~30 s. With 32 axes per outer Hessian
    /// eval and ~5 outer iters that's the ~1 hr large-scale timeout.
    ///
    /// Algebra:
    /// ```text
    ///   B_d = D_d^T W X + X^T W D_d  + X^T diag(c) X  + S_psi
    ///   D_d = (∂X/∂ψ_d) = K_d · Z_unproject       (raw kernel · unproject)
    ///   tr(F^T B_d F) = 2 · ⟨W ⊙ DXF, XF⟩ + ⟨c ⊙ XF, XF⟩ + tr(F^T S_psi F)
    /// ```
    /// where `K_d` is the raw (n × n_knots) per-pair kernel scalar matrix
    /// for axis `d` (`q · s_combo + c · coeff_sum · φ` per (i, j) pair) and
    /// `Z_unproject` is the identifiability/padding back-projection.
    ///
    /// We compute `U_knot = unproject_matrix(F)` once at (n_knots × rank),
    /// then for each row chunk do a fused pass:
    ///   * `XF_chunk  = X_chunk · F`        (chunk × rank)  — shared-X GEMM
    ///   * `Kd_chunk  = row_chunk_first_raw`(chunk × n_knots) — raw kernel
    ///   * `DXF_chunk = Kd_chunk · U_knot`  (chunk × rank)  — single GEMM
    /// and immediately accumulate `⟨W ⊙ DXF, XF⟩` and `⟨c ⊙ XF, XF⟩` over
    /// the chunk, never materialising full XF or DXF.
    ///
    /// This replaces the previous `rank`-many `forward_mul` apply loop. On
    /// the large-scale margslope-aniso-duchon16d shard each per-axis trace
    /// drops from ~30 s to a single chunked-GEMM cost.
    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        assert_eq!(factor.nrows(), self.p);
        let n_obs = self.w_diag.len();
        let rank = factor.ncols();
        if rank == 0 || n_obs == 0 {
            return 0.0;
        }
        let xf = self.compute_xf(factor);
        self.trace_projected_factor_with_xf(factor, xf.view())
    }

    /// Cached variant — *the* hot-path optimisation for large-scale outer
    /// gradient/Hessian sweeps. Every ψ-axis built atop the same `x_design`
    /// (e.g. all 32 ψ-axes of a marginal-slope model, or the same axis hit
    /// from `g_factor` and `w_factor` traces) shares one chunked
    /// `X · F` design GEMM per `(x_design, factor)` pair via
    /// [`ProjectedFactorCache`]. With 32 axes per outer-gradient sweep and
    /// O(rank) more cross-axis traces inside the outer-Hessian build, the
    /// cache turns 32× redundant `O(n · p · rank)` GEMMs into a single one
    /// per outer iter. At large-scale shape (`n = 320 K`, `p = rank = 95`) that
    /// is the difference between minutes and seconds of design-GEMM work.
    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        assert_eq!(factor.nrows(), self.p);
        let n_obs = self.w_diag.len();
        let rank = factor.ncols();
        if rank == 0 || n_obs == 0 {
            return 0.0;
        }
        let xf = self.cached_xf(factor, cache);
        self.trace_projected_factor_with_xf(factor, xf.view())
    }
}

/// Row-block size that keeps each streamed `n × cols` chunk near an 8 MiB
/// working set, with a 512-row floor so a wide design still makes useful BLAS-3
/// progress per block, capped at the total row count. Shared by the implicit
/// operator's row-streaming kernels so they cannot drift apart.
pub(crate) fn byte_balanced_row_chunk(cols: usize, n_rows: usize) -> usize {
    const TARGET_BYTES: usize = 8 * 1024 * 1024;
    const MIN_CHUNK_ROWS: usize = 512;
    let bytes_per_row = cols.max(1) * std::mem::size_of::<f64>();
    (TARGET_BYTES / bytes_per_row)
        .max(MIN_CHUNK_ROWS)
        .min(n_rows)
}

impl ImplicitHyperOperator {
    /// Chunked `X · F` via faer SIMD-parallel GEMM. The chunk-row sizing
    /// targets ~8 MiB live blocks so the (chunk_n × p) row slice and
    /// (chunk_n × rank) result both stay in L2/L3 across realistic large-scale
    /// shapes; the kernel mirrors `xt_logdet_kernel_x_diagonal`'s sizing
    /// rule. Caller wraps this in [`Self::cached_xf`] when invariance
    /// across ψ-axes lets one matrix serve every axis at this `(x_design,
    /// factor)` pair.
    pub(crate) fn compute_xf(&self, factor: &Array2<f64>) -> Array2<f64> {
        let n_obs = self.w_diag.len();
        let rank = factor.ncols();
        let mut xf = Array2::<f64>::zeros((n_obs, rank));
        let chunk_rows = byte_balanced_row_chunk(self.p + rank, n_obs);
        let mut start = 0usize;
        while start < n_obs {
            let end = (start + chunk_rows).min(n_obs);
            let rows = self
                .x_design
                .try_row_chunk(start..end)
                // SAFETY: `try_row_chunk` only fails on operator
                // implementation bugs — `start..end` is built from
                // `0..n_obs = 0..x_design.nrows()` with
                // `end = (start+chunk_rows).min(n_obs)`, so the range is
                // always a valid sub-range of `x_design`. Failure means the
                // operator broke its row-chunk contract.
                .unwrap_or_else(|err| {
                    // SAFETY: row range is a valid sub-range of x_design; failure means operator broke contract.
                    reml_contract_panic(format!(
                        "ImplicitHyperOperator::compute_xf row chunk failed: {err}"
                    ))
                });
            let block = gam_linalg::faer_ndarray::fast_ab(&rows, factor);
            xf.slice_mut(ndarray::s![start..end, ..]).assign(&block);
            start = end;
        }
        xf
    }

    /// Look up `X · F` from the [`ProjectedFactorCache`] (compute-on-miss).
    /// Cache key combines the shared `x_design` Arc pointer and the
    /// factor's value fingerprint, so two `ImplicitHyperOperator` instances
    /// built atop the same `x_design` (e.g. axis-0 and axis-1 of a 32-axis
    /// ψ-block) consult the same cache slot and hit after the first
    /// computes.
    pub(crate) fn cached_xf(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> Arc<Array2<f64>> {
        let design_id = Arc::as_ptr(&self.x_design) as usize;
        let key = ProjectedFactorKey::from_factor_view(design_id, factor.view());
        cache.get_or_insert_with(key, || self.compute_xf(factor))
    }

    /// Evaluate `tr(Fᵀ B_d F)` given a precomputed `X · F`. Pulls every
    /// per-axis-redundant `X · F` out of the inner loop so the cache (or
    /// caller-supplied matrix) covers every ψ-axis at once. The remaining
    /// per-axis work is the row-kernel build (`row_chunk_first_raw`),
    /// the `K_d · U_knot` GEMM, the fused `⟨W ⊙ DXF, XF⟩` inner products,
    /// and the small dense penalty contraction.
    pub(crate) fn trace_projected_factor_with_xf(
        &self,
        factor: &Array2<f64>,
        xf: ArrayView2<'_, f64>,
    ) -> f64 {
        let rank = factor.ncols();
        let n_obs = self.w_diag.len();
        assert_eq!(xf.dim(), (n_obs, rank));

        // Once: unproject F to raw knot space → (n_knots × rank).
        let u_knot = self.implicit_deriv.unproject_matrix(&factor.view());

        // Match the chunk sizing `xt_logdet_kernel_x_diagonal` uses so the
        // live block stays in L2/L3 across realistic large-scale shapes.
        let chunk_rows = byte_balanced_row_chunk(self.p + rank, n_obs);

        let w = self.w_diag.as_ref();
        let c_opt = self.c_x_psi_beta.as_ref().map(|arc| arc.as_ref());
        let mut design_total = 0.0_f64;
        let mut correction_total = 0.0_f64;
        let mut start = 0usize;
        while start < n_obs {
            let end = (start + chunk_rows).min(n_obs);
            let chunk_n = end - start;

            // Cached-or-precomputed X·F slice for this chunk.
            let xf_chunk = xf.slice(ndarray::s![start..end, ..]);

            // Raw kernel scalars for axis d on this chunk, then a single
            // (chunk × n_knots) · (n_knots × rank) GEMM gives DXF_chunk.
            let kd_chunk = self
                .implicit_deriv
                .row_chunk_first_raw(self.axis, start..end)
                .expect("radial scalar evaluation failed during implicit hyper forward_mul_matrix");
            let dxf_chunk = gam_linalg::faer_ndarray::fast_ab(&kd_chunk, &u_knot);

            // Fused inner-product accumulation.
            for i_local in 0..chunk_n {
                let i = start + i_local;
                let w_i = w[i];
                let dxf_row = dxf_chunk.row(i_local);
                let xf_row = xf_chunk.row(i_local);
                for k in 0..rank {
                    design_total += dxf_row[k] * w_i * xf_row[k];
                }
                if let Some(c) = c_opt {
                    let c_i = c[i];
                    for k in 0..rank {
                        let v = xf_row[k];
                        correction_total += c_i * v * v;
                    }
                }
            }
            start = end;
        }

        // Penalty trace: tr(F^T S_psi F) via dense BLAS3.
        let s_f = self.s_psi.dot(factor);
        let penalty: f64 = factor.iter().zip(s_f.iter()).map(|(&f, &s)| f * s).sum();

        2.0 * design_total + correction_total + penalty
    }

    /// Batched-axis sibling of [`Self::trace_projected_factor_with_xf`].
    /// Returns `tr(Fᵀ B_d F)` for every `(axis, s_psi, c_x_psi_beta)` triple
    /// in `axes`, sharing the unproject-and-row-sweep work across axes that
    /// only differ in their axis index / penalty matrix / correction vector.
    pub(crate) fn trace_projected_factor_all_axes_with_xf(
        &self,
        factor: &Array2<f64>,
        xf: ArrayView2<'_, f64>,
        axes: &[(usize, &Array2<f64>, Option<&Array1<f64>>)],
    ) -> Vec<f64> {
        let rank = factor.ncols();
        let n_obs = self.w_diag.len();
        assert_eq!(xf.dim(), (n_obs, rank));

        let u_knot = self.implicit_deriv.unproject_matrix(&factor.view());

        let chunk_rows = byte_balanced_row_chunk(self.p + rank, n_obs.max(1));

        let w = self.w_diag.as_ref();
        let mut design_totals = vec![0.0_f64; axes.len()];
        let mut correction_totals = vec![0.0_f64; axes.len()];

        let mut start = 0usize;
        while start < n_obs {
            let end = (start + chunk_rows).min(n_obs);
            let chunk_n = end - start;
            let xf_chunk = xf.slice(ndarray::s![start..end, ..]);

            for (axis_idx, (axis, _s_psi, c_opt_axis)) in axes.iter().enumerate() {
                let kd_chunk = self
                    .implicit_deriv
                    .row_chunk_first_raw(*axis, start..end)
                    .expect(
                        "radial scalar evaluation failed during \
                         trace_projected_factor_all_axes_with_xf",
                    );
                let dxf_chunk = gam_linalg::faer_ndarray::fast_ab(&kd_chunk, &u_knot);

                for i_local in 0..chunk_n {
                    let i = start + i_local;
                    let w_i = w[i];
                    let dxf_row = dxf_chunk.row(i_local);
                    let xf_row = xf_chunk.row(i_local);
                    for k in 0..rank {
                        design_totals[axis_idx] += dxf_row[k] * w_i * xf_row[k];
                    }
                    if let Some(c) = c_opt_axis {
                        let c_i = c[i];
                        for k in 0..rank {
                            let v = xf_row[k];
                            correction_totals[axis_idx] += c_i * v * v;
                        }
                    }
                }
            }
            start = end;
        }

        axes.iter()
            .enumerate()
            .map(|(idx, (_axis, s_psi, _c_opt_axis))| {
                let s_f = s_psi.dot(factor);
                let penalty: f64 = factor.iter().zip(s_f.iter()).map(|(&f, &s)| f * s).sum();
                2.0 * design_totals[idx] + correction_totals[idx] + penalty
            })
            .collect()
    }

    pub(crate) fn accumulate_c_correction_xt_into(
        &self,
        x_col: ArrayView1<'_, f64>,
        mut n_work: ArrayViewMut1<'_, f64>,
        mut p_work: ArrayViewMut1<'_, f64>,
        mut out_col: ArrayViewMut1<'_, f64>,
    ) {
        let Some(c_x_psi_beta) = self.c_x_psi_beta.as_ref() else {
            return;
        };
        let c = c_x_psi_beta.as_ref();
        assert_eq!(x_col.len(), c.len());
        assert_eq!(n_work.len(), c.len());
        assert_eq!(p_work.len(), self.p);

        for i in 0..c.len() {
            n_work[i] = c[i] * x_col[i];
        }
        design_matrix_transpose_apply_view_into(&self.x_design, n_work.view(), p_work.view_mut());
        out_col += &p_work;
    }

    pub(crate) fn c_correction_bilinear(&self, x_v: &Array1<f64>, x_u: &Array1<f64>) -> f64 {
        let Some(c_x_psi_beta) = self.c_x_psi_beta.as_ref() else {
            return 0.0;
        };
        x_v.iter()
            .zip(x_u.iter())
            .zip(c_x_psi_beta.iter())
            .map(|((&xv, &xu), &c)| xv * c * xu)
            .sum()
    }

    /// Compute the design-part bilinear form u^T (X^T C_d X) z using precomputed
    /// shared X-multiplies, avoiding the full B_d matvec.
    ///
    /// The design part of B_d is:
    ///   (∂X/∂ψ_d)^T W X + X^T W (∂X/∂ψ_d)
    ///
    /// For vectors z and u, the bilinear form u^T [design_part] z equals:
    ///   ((∂X/∂ψ_d) u)^T (W (Xz)) + (Xu)^T (W ((∂X/∂ψ_d) z))
    ///   = 2 * (w ⊙ y_vec)^T dx_z       [when u = u, z = z]
    ///
    /// where y_vec = X u, dx_z = (∂X/∂ψ_d) z.
    ///
    /// But the full bilinear form is NOT symmetric in its dependence on z vs u
    /// through the design derivative, so we compute both cross-terms:
    ///   dx_z^T (w ⊙ y_vec) + dx_u^T (w ⊙ x_vec)
    ///
    /// # Arguments
    /// - `x_vec`: X z (precomputed, shared across axes)
    /// - `y_vec`: X u (precomputed, shared across axes)
    /// - `z`: the probe vector (needed for forward_mul and penalty)
    /// - `u`: H⁻¹ z (needed for forward_mul and penalty)
    ///
    /// # Returns
    /// The full bilinear form u^T B_d z = design_part + penalty_part.
    pub fn bilinear_with_shared_x(
        &self,
        x_vec: &Array1<f64>,
        y_vec: &Array1<f64>,
        z: &Array1<f64>,
        u: &Array1<f64>,
    ) -> f64 {
        // Design part: dx_z^T (w ⊙ y_vec) + dx_u^T (w ⊙ x_vec)
        let dx_z = self
            .implicit_deriv
            .forward_mul(self.axis, &z.view())
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");
        let dx_u = self
            .implicit_deriv
            .forward_mul(self.axis, &u.view())
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");

        let mut design = 0.0f64;
        let w = &*self.w_diag;
        for i in 0..x_vec.len() {
            let wi = w[i];
            design += dx_z[i] * wi * y_vec[i];
            design += dx_u[i] * wi * x_vec[i];
        }

        // Non-Gaussian fixed-β third-derivative correction:
        //   uᵀ Xᵀ diag(c ⊙ X_{ψ_d} β̂) X z = Σ_i (X u)_i · c_x_psi_beta_i · (X z)_i
        //   = Σ_i y_vec[i] · c_x_psi_beta[i] · x_vec[i]
        if let Some(c_x_psi_beta) = self.c_x_psi_beta.as_ref() {
            let c = c_x_psi_beta.as_ref();
            for i in 0..x_vec.len() {
                design += y_vec[i] * c[i] * x_vec[i];
            }
        }

        // Penalty part: u^T S_psi z
        let penalty = dense_bilinear(&self.s_psi, z.view(), u.view());

        design + penalty
    }

    /// Compute the design-part contribution to A_d z without the X^T step.
    ///
    /// Returns the n-vector C_d (X z) where C_d encodes the diagonal weighting.
    /// Specifically: (∂X/∂ψ_d)^T maps FROM n-space, but for stochastic trace
    /// estimation we need q_d = A_d z = X^T (C_d x_vec) + P_d z.
    ///
    /// This method computes q_d = A_d z using the shared x_vec = X z:
    ///   q_d = (∂X/∂ψ_d)^T (W (X z)) + X^T (W ((∂X/∂ψ_d) z)) + S_psi z
    /// which is the standard mul_vec but we can share x_vec across axes.
    pub fn matvec_with_shared_xz_into(
        &self,
        x_vec: ArrayView1<'_, f64>,
        z: ArrayView1<'_, f64>,
        mut out: ArrayViewMut1<'_, f64>,
        mut n_work: ArrayViewMut1<'_, f64>,
        mut p_work: ArrayViewMut1<'_, f64>,
    ) {
        assert_eq!(z.len(), self.p);
        assert_eq!(out.len(), self.p);
        assert_eq!(n_work.len(), self.w_diag.len());
        assert_eq!(p_work.len(), self.p);

        let w = &*self.w_diag;
        for i in 0..w.len() {
            n_work[i] = w[i] * x_vec[i];
        }
        let term1 = self
            .implicit_deriv
            .transpose_mul(self.axis, &n_work.view())
            .expect("radial scalar evaluation failed during implicit hyper transpose_mul");
        out.assign(&term1);

        let dx_z = self
            .implicit_deriv
            .forward_mul(self.axis, &z)
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");
        for i in 0..w.len() {
            n_work[i] = w[i] * dx_z[i];
        }
        design_matrix_transpose_apply_view_into(&self.x_design, n_work.view(), p_work.view_mut());
        out += &p_work;

        dense_matvec_into(&self.s_psi, z, p_work.view_mut());
        out += &p_work;

        // Non-Gaussian fixed-β third-derivative correction.
        if let Some(c_x_psi_beta) = self.c_x_psi_beta.as_ref() {
            let c = c_x_psi_beta.as_ref();
            for i in 0..w.len() {
                n_work[i] = c[i] * x_vec[i];
            }
            design_matrix_transpose_apply_view_into(
                &self.x_design,
                n_work.view(),
                p_work.view_mut(),
            );
            out += &p_work;
        }
    }
}

/// Operator-backed fixed-β Hessian drift for sparse-exact τ coordinates.
///
/// This stays in the original sparse/native coefficient basis and computes the
/// exact first-order τ Hessian drift
///   B_τ = X_τᵀ W X + Xᵀ W X_τ + Xᵀ diag(c ⊙ X_τ β̂) X + S_τ − (H_φ)_{τ}|_β
/// without materializing the full dense matrix up front.
pub struct SparseDirectionalHyperOperator {
    /// Original-basis design derivative X_τ.
    pub(crate) x_tau: super::super::HyperDesignDerivative,
    /// Design matrix X in the sparse-native basis.
    pub(crate) x_design: DesignMatrix,
    /// Working weights W (diagonal) — observed-information curvature, signed
    /// for non-canonical links.  Carried as the owned [`gam_linalg::matrix::SignedWeightsArc`]
    /// newtype so the sign character is construction-enforced at the operator
    /// struct boundary.
    pub(crate) w_diag: gam_linalg::matrix::SignedWeightsArc,
    /// Penalty derivative S_τ.
    pub(crate) s_tau: Array2<f64>,
    /// Fixed-β non-Gaussian curvature term c ⊙ (X_τ β̂), if applicable.
    pub(crate) c_x_tau_beta: Option<Array1<f64>>,
    /// Fixed-β Firth partial Hessian drift (H_φ)_{τ}|_β, if applicable.
    pub(crate) firth_hphi_tau_partial: Option<Array2<f64>>,
    /// Total coefficient dimension.
    pub(crate) p: usize,
}

impl HyperOperator for SparseDirectionalHyperOperator {
    fn dim(&self) -> usize {
        self.p
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.p);

        // X v
        let x_v = self.x_design.matrixvectormultiply(v);

        // X_tauᵀ (W (X v))
        let w_x_v = &*self.w_diag * &x_v;
        let term1 = self
            .x_tau
            .transpose_mul_original(&w_x_v)
            .expect("SparseDirectionalHyperOperator transpose product should be shape-consistent");

        // Xᵀ (W (X_tau v))
        let x_tau_v = self
            .x_tau
            .forward_mul_original(v)
            .expect("SparseDirectionalHyperOperator forward product should be shape-consistent");
        let w_x_tau_v = &*self.w_diag * &x_tau_v;
        let term2 = self.x_design.transpose_vector_multiply(&w_x_tau_v);

        // S_tau v
        let term3 = self.s_tau.dot(v);

        let mut out = term1 + term2 + term3;

        // Non-Gaussian fixed-beta curvature: Xᵀ diag(c ⊙ X_tau β̂) X v
        if let Some(c_x_tau_beta) = self.c_x_tau_beta.as_ref() {
            let weighted = c_x_tau_beta * &x_v;
            out += &self.x_design.transpose_vector_multiply(&weighted);
        }

        // Firth fixed-beta partial: subtract (H_φ)_{τ}|_β v
        if let Some(hphi_tau_partial) = self.firth_hphi_tau_partial.as_ref() {
            out -= &hphi_tau_partial.dot(v);
        }

        out
    }

    fn is_implicit(&self) -> bool {
        false
    }
    fn as_any(&self) -> &(dyn std::any::Any + 'static) {
        self
    }
}

/// Matrix-free GLM cubic-correction drift `C[v] = −Xᵀ diag(c ⊙ X v) X`
/// (rows masked to the active Hessian-curvature surface, sign folded into
/// the stored diagonal).
///
/// # Why this must stay an operator (#901 layer 2)
///
/// The spectral logdet kernel evaluates `tr(H⁺ · C)` as
/// `Σ_a (1/σ_a) · u_aᵀ C u_a` over the eigenpairs of `H_pen`. For a
/// near-null eigenvector (`σ_min ~ 1e−4` on the Duchon fixtures) the true
/// quadratic form is tiny — `‖X u_a‖² ≲ σ_a / w_min` — but a DENSE
/// materialization of `C` computes it as a cancellation across entries of
/// magnitude `‖C‖`, leaving roundoff `~ ε‖C‖p` that the kernel then
/// amplifies by `1/σ_min`. On the iso-κ Duchon binomial FD drivers this
/// turned a true cubic trace of `−0.30` into `+39.0`, and `~−7.7e5` on the
/// κ-scaled ψ arms where `‖C‖ ~ λ · ∂S/∂ψ` — the dominant #901 blow-up.
///
/// In operator form the kernel probes `C · u_a = −Xᵀ(d ⊙ (X u_a))`: the
/// cancellation happens inside the `X u_a` matvec (error `~ ε‖X‖‖u_a‖`),
/// and the quadratic form is the *square* of that already-small vector —
/// tiny² stays tiny, so the `1/σ_a` amplification acts on a relatively
/// accurate value. This is the same stability argument as evaluating
/// leverages via `(X u)ᵀ d (X u)` instead of `uᵀ (XᵀdX) u`.
pub struct GlmCurvatureCorrectionOperator {
    /// Design matrix X in the transformed basis (matrix-free capable).
    pub(crate) x_design: DesignMatrix,
    /// Pre-masked, sign-folded diagonal `−(c ⊙ X v)` over active rows.
    pub(crate) neg_c_xv: Array1<f64>,
    /// Total coefficient dimension.
    pub(crate) p: usize,
}

impl HyperOperator for GlmCurvatureCorrectionOperator {
    fn dim(&self) -> usize {
        self.p
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.p);
        let x_v = self.x_design.matrixvectormultiply(v);
        let weighted = &self.neg_c_xv * &x_v;
        self.x_design.transpose_vector_multiply(&weighted)
    }

    fn as_any(&self) -> &(dyn std::any::Any + 'static) {
        self
    }

    fn is_implicit(&self) -> bool {
        false
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Data structures
// ═══════════════════════════════════════════════════════════════════════════
