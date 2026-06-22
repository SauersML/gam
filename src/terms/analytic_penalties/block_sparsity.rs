use super::*;

// ---------------------------------------------------------------------------
// Block sparsity / group-lasso penalty
// ---------------------------------------------------------------------------

/// Group-lasso penalty over predefined latent-axis blocks.
///
/// This is structured L¹ on group L² norms: per-element L¹ can zero isolated
/// coefficients, ARD applies per-axis L² precision, while block sparsity
/// zeroes whole semantic axis groups together. It pairs directly with
/// `LatentIdMode::AuxPriorDimSelection` when the auxiliary prior supplies
/// class-specific active group subsets.
#[derive(Debug, Clone)]
pub struct BlockSparsityPenalty {
    pub target: PsiSlice,
    pub groups: Vec<Vec<usize>>,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major matrix-valued latent block.
    pub n_eff: usize,
    /// Yuan-Lin group-size normalization uses the latent-axis group cardinality
    /// `sqrt(|g|)`; `n_eff` repeats each group across rows inside its norm and
    /// does not participate in this normalization.
    pub smoothing_eps: f64,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

impl BlockSparsityPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        groups: Vec<Vec<usize>>,
        weight: f64,
        n_eff: usize,
        smoothing_eps: f64,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err("BlockSparsityPenalty::new requires a non-empty target".to_string());
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "BlockSparsityPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("BlockSparsityPenalty::new requires n_eff > 0".to_string());
        }
        if !target.len().is_multiple_of(n_eff) {
            return Err(format!(
                "BlockSparsityPenalty::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        let latent_dim = target.len() / n_eff;
        if let Some(expected_dim) = target.latent_dim {
            let expected = n_eff.checked_mul(expected_dim).ok_or_else(|| {
                "BlockSparsityPenalty::new target shape overflows usize".to_string()
            })?;
            if expected != target.len() {
                return Err(format!(
                    "BlockSparsityPenalty::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    expected_dim
                ));
            }
        }
        if !(smoothing_eps.is_finite() && smoothing_eps > 0.0) {
            return Err(format!(
                "BlockSparsityPenalty::new requires finite smoothing_eps > 0, got {smoothing_eps}"
            ));
        }
        if groups.is_empty() {
            return Err("BlockSparsityPenalty::new requires at least one group".to_string());
        }
        let mut seen = vec![false; latent_dim];
        for (group_idx, group) in groups.iter().enumerate() {
            if group.is_empty() {
                return Err(format!(
                    "BlockSparsityPenalty::new groups[{group_idx}] must not be empty"
                ));
            }
            for &axis in group {
                if axis >= latent_dim {
                    return Err(format!(
                        "BlockSparsityPenalty::new groups[{group_idx}] axis {axis} exceeds latent_dim {latent_dim}"
                    ));
                }
                if seen[axis] {
                    return Err(format!(
                        "BlockSparsityPenalty::new axis {axis} appears in more than one group"
                    ));
                }
                seen[axis] = true;
            }
        }
        for (axis, present) in seen.iter().copied().enumerate() {
            if !present {
                return Err(format!(
                    "BlockSparsityPenalty::new groups must partition latent axes; missing axis {axis}"
                ));
            }
        }
        Ok(Self {
            target,
            groups,
            weight,
            n_eff,
            smoothing_eps,
            learnable_weight,
            rho_index: 0,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            assert_eq!(
                target_len % self.n_eff.max(1),
                0,
                "target length must be divisible by n_eff"
            );
            return None;
        }
        Some(target_len / self.n_eff)
    }

    fn target_matrix<'a>(&self, target: ArrayView1<'a, f64>) -> Option<ArrayView2<'a, f64>> {
        let d = self.latent_dim(target.len())?;
        target.into_shape_with_order((self.n_eff, d)).ok()
    }

    fn group_size_factor(group: &[usize]) -> f64 {
        (group.len() as f64).sqrt()
    }

    fn group_norm(&self, t: ArrayView2<'_, f64>, group: &[usize]) -> f64 {
        let mut norm2 = 0.0;
        for n in 0..t.nrows() {
            for &axis in group {
                let x = t[[n, axis]];
                norm2 += x * x;
            }
        }
        (norm2 + self.smoothing_eps * self.smoothing_eps).sqrt()
    }

    pub fn diag_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for group in &self.groups {
            let factor = weight * Self::group_size_factor(group);
            let s = self.group_norm(t.view(), group);
            let inv_s = 1.0 / s;
            let inv_s3 = inv_s * inv_s * inv_s;
            for n in 0..t.nrows() {
                for &axis in group {
                    let x = t[[n, axis]];
                    out[n * t.ncols() + axis] = factor * (inv_s - x * x * inv_s3);
                }
            }
        }
        out
    }

    /// Materialize the group-lasso Hessian for small-block spectral paths.
    pub fn as_dense(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array2<f64> {
        let n = target.len();
        let Some(t) = self.target_matrix(target) else {
            return Array2::<f64>::zeros((n, n));
        };
        let d = t.ncols();
        let weight = self.resolved_weight(rho);
        let mut dense = Array2::<f64>::zeros((n, n));
        for group in &self.groups {
            let factor = weight * Self::group_size_factor(group);
            let s = self.group_norm(t.view(), group);
            let inv_s = 1.0 / s;
            let inv_s3 = inv_s * inv_s * inv_s;
            for row1 in 0..t.nrows() {
                for &col1 in group {
                    let i = row1 * d + col1;
                    let x_i = t[[row1, col1]];
                    for row2 in 0..t.nrows() {
                        for &col2 in group {
                            let j = row2 * d + col2;
                            let mut entry = -x_i * t[[row2, col2]] * inv_s3;
                            if i == j {
                                entry += inv_s;
                            }
                            dense[[i, j]] = factor * entry;
                        }
                    }
                }
            }
        }
        dense
    }
}

impl AnalyticPenalty for BlockSparsityPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(t) = self.target_matrix(target) else {
            return 0.0;
        };
        let mut acc = 0.0;
        for group in &self.groups {
            acc += Self::group_size_factor(group) * self.group_norm(t.view(), group);
        }
        self.resolved_weight(rho) * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut grad = Array2::<f64>::zeros(t.dim());
        for group in &self.groups {
            let s = self.group_norm(t.view(), group);
            let factor = weight * Self::group_size_factor(group) / s;
            for n in 0..t.nrows() {
                for &axis in group {
                    grad[[n, axis]] = factor * t[[n, axis]];
                }
            }
        }
        super::flatten_matrix(&grad)
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let Some(v_mat) = self.target_matrix(v) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array2::<f64>::zeros(t.dim());
        for group in &self.groups {
            let factor = weight * Self::group_size_factor(group);
            let s = self.group_norm(t.view(), group);
            let inv_s = 1.0 / s;
            let inv_s3 = inv_s * inv_s * inv_s;
            let mut inner = 0.0;
            for n in 0..t.nrows() {
                for &axis in group {
                    inner += t[[n, axis]] * v_mat[[n, axis]];
                }
            }
            for n in 0..t.nrows() {
                for &axis in group {
                    out[[n, axis]] =
                        factor * (v_mat[[n, axis]] * inv_s - t[[n, axis]] * inner * inv_s3);
                }
            }
        }
        super::flatten_matrix(&out)
    }

    impl_learnable_weight_grad_rho!();

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "block_sparsity"
    }

    impl_scalar_apply_schedule!(weight);
}

// ---------------------------------------------------------------------------
// Mechanism-sparsity penalty
// ---------------------------------------------------------------------------

/// Per-latent group-lasso sparsity over decoder feature groups.
#[derive(Debug, Clone)]
pub struct MechanismSparsityPenalty {
    pub target: PsiSlice,
    pub feature_groups: Vec<Vec<usize>>,
    pub weight: f64,
    pub smoothing_eps: f64,
    pub n_eff: f64,
    pub weight_schedule: Option<Arc<ScalarWeightSchedule>>,
    pub learnable_weight: bool,
    pub rho_index: usize,
}

impl MechanismSparsityPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        feature_groups: Vec<Vec<usize>>,
        weight: f64,
        smoothing_eps: f64,
        n_eff: f64,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err("MechanismSparsityPenalty::new requires a non-empty target".to_string());
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "MechanismSparsityPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if !(smoothing_eps.is_finite() && smoothing_eps > 0.0) {
            return Err(format!(
                "MechanismSparsityPenalty::new requires finite smoothing_eps > 0, got {smoothing_eps}"
            ));
        }
        if !(n_eff.is_finite() && n_eff > 0.0) {
            return Err(format!(
                "MechanismSparsityPenalty::new requires finite n_eff > 0, got {n_eff}"
            ));
        }
        if feature_groups.is_empty() {
            return Err(
                "MechanismSparsityPenalty::new requires at least one feature group".to_string(),
            );
        }
        let latent_dim = target.latent_dim.ok_or_else(|| {
            "MechanismSparsityPenalty::new requires target.latent_dim".to_string()
        })?;
        if latent_dim == 0 {
            return Err("MechanismSparsityPenalty::new requires latent_dim > 0".to_string());
        }
        let p_features = Self::validate_feature_groups(&feature_groups)?;
        let expected_len = latent_dim.checked_mul(p_features).ok_or_else(|| {
            "MechanismSparsityPenalty::new target shape overflows usize".to_string()
        })?;
        if target.len() != expected_len {
            return Err(format!(
                "MechanismSparsityPenalty::new target length {} does not match latent_dim {} × feature_count {}",
                target.len(),
                latent_dim,
                p_features
            ));
        }
        Ok(Self {
            target,
            feature_groups,
            weight,
            smoothing_eps,
            n_eff,
            weight_schedule: None,
            learnable_weight,
            rho_index: 0,
        })
    }

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(Arc::new(schedule));
        self
    }

    fn validate_feature_groups(feature_groups: &[Vec<usize>]) -> Result<usize, String> {
        let mut max_feature = None::<usize>;
        for (group_idx, group) in feature_groups.iter().enumerate() {
            if group.is_empty() {
                return Err(format!(
                    "MechanismSparsityPenalty::new feature_groups[{group_idx}] must not be empty"
                ));
            }
            for &feature in group {
                max_feature = Some(max_feature.map_or(feature, |current| current.max(feature)));
            }
        }
        let p_features = max_feature
            .and_then(|feature| feature.checked_add(1))
            .ok_or_else(|| {
                "MechanismSparsityPenalty::new feature shape overflows usize".to_string()
            })?;
        let mut seen = vec![false; p_features];
        for (group_idx, group) in feature_groups.iter().enumerate() {
            for &feature in group {
                if seen[feature] {
                    return Err(format!(
                        "MechanismSparsityPenalty::new feature {feature} appears in more than one group"
                    ));
                }
                seen[feature] = true;
            }
            for &feature in group {
                if feature >= p_features {
                    return Err(format!(
                        "MechanismSparsityPenalty::new feature_groups[{group_idx}] feature {feature} exceeds feature_count {p_features}"
                    ));
                }
            }
        }
        for (feature, present) in seen.iter().copied().enumerate() {
            if !present {
                return Err(format!(
                    "MechanismSparsityPenalty::new feature_groups must partition features; missing feature {feature}"
                ));
            }
        }
        Ok(p_features)
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn latent_dim(&self) -> Option<usize> {
        self.target.latent_dim.filter(|&d| d > 0)
    }

    fn feature_count(&self) -> Option<usize> {
        let d = self.latent_dim()?;
        if !self.target.len().is_multiple_of(d) {
            return None;
        }
        Some(self.target.len() / d)
    }

    fn target_matrix<'a>(&self, target: ArrayView1<'a, f64>) -> Option<ArrayView2<'a, f64>> {
        if self.target.range.end > target.len() {
            return None;
        }
        let d = self.latent_dim()?;
        let p = self.feature_count()?;
        let local = target.slice_move(ndarray::s![self.target.range.start..self.target.range.end]);
        local.into_shape_with_order((d, p)).ok()
    }

    fn group_size_factor(group: &[usize]) -> f64 {
        (group.len() as f64).sqrt()
    }

    fn group_norm(&self, w: ArrayView2<'_, f64>, latent: usize, group: &[usize]) -> f64 {
        let mut norm2 = 0.0;
        for &feature in group {
            let x = w[[latent, feature]];
            norm2 += x * x;
        }
        (norm2 + self.smoothing_eps * self.smoothing_eps).sqrt()
    }

    pub fn diag_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let Some(w) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let p = w.ncols();
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for latent in 0..w.nrows() {
            for group in &self.feature_groups {
                let factor = weight * Self::group_size_factor(group);
                let s = self.group_norm(w.view(), latent, group);
                let inv_s = 1.0 / s;
                let inv_s3 = inv_s * inv_s * inv_s;
                for &feature in group {
                    let x = w[[latent, feature]];
                    let idx = self.target.range.start + latent * p + feature;
                    out[idx] = factor * (inv_s - x * x * inv_s3);
                }
            }
        }
        out
    }

    pub fn as_dense(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array2<f64> {
        let n = target.len();
        let Some(w) = self.target_matrix(target) else {
            return Array2::<f64>::zeros((n, n));
        };
        let p = w.ncols();
        let weight = self.resolved_weight(rho);
        let mut dense = Array2::<f64>::zeros((n, n));
        for latent in 0..w.nrows() {
            for group in &self.feature_groups {
                let factor = weight * Self::group_size_factor(group);
                let s = self.group_norm(w.view(), latent, group);
                let inv_s = 1.0 / s;
                let inv_s3 = inv_s * inv_s * inv_s;
                for &feature_i in group {
                    let i = self.target.range.start + latent * p + feature_i;
                    let x_i = w[[latent, feature_i]];
                    for &feature_j in group {
                        let j = self.target.range.start + latent * p + feature_j;
                        let mut entry = -x_i * w[[latent, feature_j]] * inv_s3;
                        if i == j {
                            entry += inv_s;
                        }
                        dense[[i, j]] = factor * entry;
                    }
                }
            }
        }
        dense
    }
}

impl AnalyticPenalty for MechanismSparsityPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Beta
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(w) = self.target_matrix(target) else {
            return 0.0;
        };
        let mut acc = 0.0;
        for latent in 0..w.nrows() {
            for group in &self.feature_groups {
                acc += Self::group_size_factor(group) * self.group_norm(w.view(), latent, group);
            }
        }
        self.resolved_weight(rho) * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(w) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let p = w.ncols();
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for latent in 0..w.nrows() {
            for group in &self.feature_groups {
                let s = self.group_norm(w.view(), latent, group);
                let factor = weight * Self::group_size_factor(group) / s;
                for &feature in group {
                    let idx = self.target.range.start + latent * p + feature;
                    out[idx] = factor * w[[latent, feature]];
                }
            }
        }
        out
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(w) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let Some(v_mat) = self.target_matrix(v) else {
            return Array1::<f64>::zeros(target.len());
        };
        let p = w.ncols();
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for latent in 0..w.nrows() {
            for group in &self.feature_groups {
                let factor = weight * Self::group_size_factor(group);
                let s = self.group_norm(w.view(), latent, group);
                let inv_s = 1.0 / s;
                let inv_s3 = inv_s * inv_s * inv_s;
                let mut inner = 0.0;
                for &feature in group {
                    inner += w[[latent, feature]] * v_mat[[latent, feature]];
                }
                for &feature in group {
                    let idx = self.target.range.start + latent * p + feature;
                    out[idx] = factor
                        * (v_mat[[latent, feature]] * inv_s
                            - w[[latent, feature]] * inner * inv_s3);
                }
            }
        }
        out
    }

    impl_learnable_weight_grad_rho!();

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "mechanism_sparsity"
    }

    fn apply_schedule(&mut self, iter: usize) {
        if let Some(schedule) = self.weight_schedule.as_mut() {
            let schedule = Arc::make_mut(schedule);
            self.weight = schedule.current_weight(iter);
            schedule.iter_count = iter + 1;
        }
    }
}
