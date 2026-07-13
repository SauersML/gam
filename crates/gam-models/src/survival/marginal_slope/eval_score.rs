//! Score-warp and logslope-surface evaluation: querying score-basis
//! dimensions, evaluating the local-cubic score warp at a row, projecting
//! the observed/integration score onto the basis, and reading the per-row
//! logslope surface values.

use super::*;

impl SurvivalMarginalSlopeFamily {
    #[inline]
    pub(crate) fn probit_frailty_scale(&self) -> f64 {
        probit_frailty_scale(self.gaussian_frailty_sd)
    }

    pub(crate) fn z_subsample_key(&self) -> Array1<f64> {
        self.z.column(0).to_owned()
    }

    #[inline]
    pub(crate) fn score_dim(&self) -> usize {
        assert_eq!(self.score_covariance.dim(), self.z.ncols());
        self.z.ncols()
    }

    pub(crate) fn score_warp_basis_dim(&self) -> usize {
        self.score_warp
            .as_ref()
            .map_or(0, DeviationRuntime::basis_dim)
    }

    pub(crate) fn score_warp_coord_basis_index(
        &self,
        local_idx: usize,
    ) -> Result<(usize, usize), String> {
        let basis_dim = self.score_warp_basis_dim();
        if basis_dim == 0 {
            return Err(
                "survival score-warp coordinate lookup without score-warp runtime".to_string(),
            );
        }
        let coord = local_idx / basis_dim;
        if coord >= self.score_dim() {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival score-warp local index {local_idx} exceeds K={} per-z blocks with basis dim {basis_dim}",
                    self.score_dim()
                ),
            }
            .into());
        }
        Ok((coord, local_idx % basis_dim))
    }

    pub(crate) fn score_warp_beta_for_coord(
        &self,
        beta_h: &Array1<f64>,
        coord: usize,
    ) -> Result<Array1<f64>, String> {
        let runtime = self
            .score_warp
            .as_ref()
            .ok_or_else(|| "missing survival score-warp runtime".to_string())?;
        score_warp_component_beta(runtime, beta_h, coord)
    }

    #[inline]
    pub(crate) fn zero_score_warp_span() -> exact_kernel::LocalSpanCubic {
        exact_kernel::LocalSpanCubic {
            left: 0.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        }
    }

    pub(crate) fn add_local_span_cubic(
        left: f64,
        right: f64,
        target: &mut exact_kernel::LocalSpanCubic,
        span: exact_kernel::LocalSpanCubic,
    ) {
        target.left = left;
        target.right = right;
        target.c0 += span.c0;
        target.c1 += span.c1;
        target.c2 += span.c2;
        target.c3 += span.c3;
    }

    pub(crate) fn score_warp_local_cubic_at(
        &self,
        beta_h: Option<&Array1<f64>>,
        value: f64,
    ) -> Result<exact_kernel::LocalSpanCubic, String> {
        let (Some(runtime), Some(beta_h)) = (self.score_warp.as_ref(), beta_h) else {
            return Ok(Self::zero_score_warp_span());
        };
        if self.score_dim() == 1 {
            return runtime.local_cubic_at(beta_h, value);
        }
        let mut sum = Self::zero_score_warp_span();
        for coord in 0..self.score_dim() {
            let local_beta = score_warp_component_beta(runtime, beta_h, coord)?;
            let span = runtime.local_cubic_at(&local_beta, value)?;
            if coord == 0 {
                sum = exact_kernel::LocalSpanCubic {
                    left: span.left,
                    right: span.right,
                    c0: 0.0,
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                };
            }
            Self::add_local_span_cubic(span.left, span.right, &mut sum, span);
        }
        Ok(sum)
    }

    pub(crate) fn score_warp_observed_value(
        &self,
        row: usize,
        beta_h: Option<&Array1<f64>>,
    ) -> Result<f64, String> {
        let (Some(runtime), Some(beta_h)) = (self.score_warp.as_ref(), beta_h) else {
            return Ok(0.0);
        };
        let mut value = 0.0;
        for coord in 0..self.score_dim() {
            let local_beta = score_warp_component_beta(runtime, beta_h, coord)?;
            let z_coord = self.z[[row, coord]];
            value += runtime
                .local_cubic_at(&local_beta, z_coord)?
                .evaluate(z_coord);
        }
        Ok(value)
    }

    pub(crate) fn observed_score_projection(&self, row: usize) -> f64 {
        if self.score_dim() == 1 {
            self.z[[row, 0]]
        } else {
            self.z.row(row).sum()
        }
    }

    pub(crate) fn integration_score_basis_coefficients(
        &self,
        local_idx: usize,
        z_basis: f64,
        multiplier: f64,
    ) -> Result<[f64; 4], String> {
        let runtime = self
            .score_warp
            .as_ref()
            .ok_or_else(|| "missing survival score-warp runtime".to_string())?;
        let (_, basis_idx) = self.score_warp_coord_basis_index(local_idx)?;
        let basis_span = runtime.basis_cubic_at(basis_idx, z_basis)?;
        Ok(exact_kernel::score_basis_cell_coefficients(
            basis_span, multiplier,
        ))
    }

    pub(crate) fn observed_score_basis_coefficients(
        &self,
        row: usize,
        local_idx: usize,
        z_obs: f64,
        multiplier: f64,
    ) -> Result<[f64; 4], String> {
        let runtime = self
            .score_warp
            .as_ref()
            .ok_or_else(|| "missing survival score-warp runtime".to_string())?;
        if self.score_dim() == 1 {
            let basis_span = runtime.basis_cubic_at(local_idx, z_obs)?;
            return Ok(exact_kernel::score_basis_cell_coefficients(
                basis_span, multiplier,
            ));
        }
        let (coord, basis_idx) = self.score_warp_coord_basis_index(local_idx)?;
        let z_coord = self.z[[row, coord]];
        let basis_span = runtime.basis_cubic_at(basis_idx, z_coord)?;
        Ok([multiplier * basis_span.evaluate(z_coord), 0.0, 0.0, 0.0])
    }

    pub(crate) fn per_z_logslope_active(&self) -> bool {
        self.score_dim() > 1 && self.logslope_layout.is_per_score()
    }

    pub(crate) fn logslope_row_workspace(&self) -> Result<LogslopeRowWorkspace, String> {
        self.logslope_layout.row_workspace(self.score_dim())
    }

    pub(crate) fn fill_logslope_values_for_row(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        workspace: &mut LogslopeRowWorkspace,
    ) -> Result<(), String> {
        if self.per_z_logslope_active() {
            return self.logslope_layout.fill_per_score_row(
                row,
                block_states[2].beta.view(),
                workspace,
            );
        }
        if block_states[2].eta.len() != self.n {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "shared survival marginal-slope logslope eta length {} does not match n={}",
                    block_states[2].eta.len(),
                    self.n,
                ),
            }
            .into());
        }
        self.logslope_layout
            .fill_shared_values(block_states[2].eta[row], workspace)
    }

    pub(crate) fn shared_logslope_covariance_scale(&self) -> Result<f64, String> {
        let k = self.score_dim();
        if k == 1 {
            return Ok(1.0);
        }
        let ones = vec![1.0; k];
        self.score_covariance.quadratic_form(&ones).map_err(|err| {
            format!("survival marginal-slope shared log-slope covariance scale: {err}")
        })
    }

    pub(crate) fn exact_shared_score_summary(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        context: &str,
    ) -> Result<(f64, f64), String> {
        let k = self.score_dim();
        if k == 1 {
            return Ok((self.z[[row, 0]], 1.0));
        }
        let logslope_eta_len = block_states[2].eta.len();
        if logslope_eta_len != self.n {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "{context}: survival marginal-slope exact shared-slope calculus for K={k} requires one log-slope eta per row (n={}); got eta len {logslope_eta_len}. Per-z log-slope derivatives require a {}-primary row kernel.",
                    self.n,
                    3 + k
                ),
            }
            .into());
        }
        Ok((
            self.z.row(row).sum(),
            self.shared_logslope_covariance_scale()?,
        ))
    }
}
