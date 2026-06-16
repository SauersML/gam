use super::*;

pub struct BernoulliMarginalSlopePredictor {
    pub beta_marginal: Array1<f64>,
    pub beta_logslope: Array1<f64>,
    pub beta_score_warp: Option<Array1<f64>>,
    pub beta_link_dev: Option<Array1<f64>>,
    pub base_link: InverseLink,
    pub z_column: String,
    pub latent_z_normalization: SavedLatentZNormalization,
    pub latent_measure: LatentMeasureKind,
    pub baseline_marginal: f64,
    pub baseline_logslope: f64,
    pub covariance: Option<Array2<f64>>,
    pub score_warp_runtime: Option<SavedCompiledFlexBlock>,
    pub link_deviation_runtime: Option<SavedCompiledFlexBlock>,
    pub gaussian_frailty_sd: Option<f64>,
    /// Optional rank-INT latent-z calibration. When `Some`, every
    /// predict-time z (after `latent_z_normalization`) is routed through
    /// [`LatentZRankIntCalibration::apply_at_predict`] before entering
    /// the standard-normal closed-form rigid kernel — mirroring the
    /// fit-time transform applied to training z. The map is the exact
    /// monotone, invertible (up to empirical-CDF resolution) piecewise-
    /// linear interpolation on (sorted_z, weighted_cdf) followed by Φ⁻¹,
    /// so the calibrated sample is N(0,1) by construction. `None` means
    /// training-time z already passed the strict normality check and no
    /// transform was applied.
    pub(crate) latent_z_calibration: Option<crate::families::bms::LatentZRankIntCalibration>,
    /// Optional conditional location-scale latent-z calibration (#905). When
    /// `Some`, predict-time z (after `latent_z_normalization`) is replaced by
    /// `ζ = (z − m(C))/√v(C)`, with the conditioning span `a(C)` rebuilt from
    /// the marginal prediction design — mirroring the fit-time correction the
    /// Auto path applied when its conditional `E[z|C]`/`Var(z|C)` Rao gate
    /// detected PC/grouping-dependence. Mutually exclusive with
    /// `latent_z_calibration`.
    pub(crate) latent_z_conditional_calibration:
        Option<crate::families::bms::LatentZConditionalCalibration>,
}

/// Per-runtime predict-time anchor correction matrices.
///
/// Built once per top-level predict call from the marginal + logslope
/// designs at the prediction rows. Each `Array2<f64>` is shaped
/// `n_predict × runtime.basis_dim` and holds `n_row(i) · M` for every
/// prediction row, where `n_row(i)` is the concatenation of the marginal
/// and logslope design rows in the runtime's anchor component order.
///
/// At any `local_cubic_at` / `basis_cubic_at` / `design` call site we
/// subtract the appropriate slice of these matrices from the raw cubic
/// output to apply the cross-block residual `n_row · M` correction.
///
/// `n_anchor_rows` is the underlying `n × d` parametric anchor stack
/// (per-runtime layouts: score_warp gets `[marginal | logslope]`;
/// link_dev gets `[marginal | logslope | score_warp_design(z)]` when the
/// fit-time identifiability stage threaded the score-warp basis in as a
/// flex-evaluation anchor). These layouts must match the column order
/// `install_compiled_flex_block_into_runtime` used at fit time.
#[derive(Default)]
struct BmsAnchorCorrections {
    /// `[marginal | logslope]` at predict rows. `Some` whenever any
    /// runtime carries an anchor residual.
    score_warp_anchor_rows: Option<Array2<f64>>,
    /// `[marginal | logslope | score_warp_design(z)]` at predict rows.
    /// `Some` whenever the link-deviation runtime carries an anchor
    /// residual; the score-warp tail is included iff the saved
    /// link-deviation runtime's residual components include a
    /// `FlexEvaluation` entry.
    link_dev_anchor_rows: Option<Array2<f64>>,
    score_warp: Option<Array2<f64>>,
    link_dev: Option<Array2<f64>>,
}

impl BmsAnchorCorrections {
    fn score_warp_row(&self, row: usize) -> Option<ndarray::ArrayView1<'_, f64>> {
        self.score_warp.as_ref().map(|m| m.row(row))
    }

    fn link_dev_row(&self, row: usize) -> Option<ndarray::ArrayView1<'_, f64>> {
        self.link_dev.as_ref().map(|m| m.row(row))
    }

    fn score_warp_anchor_rows_view(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        self.score_warp_anchor_rows.as_ref().map(|m| m.view())
    }

    fn link_dev_anchor_rows_view(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        self.link_dev_anchor_rows.as_ref().map(|m| m.view())
    }
}

impl BernoulliMarginalSlopePredictor {
    /// Build the anchor correction matrices for a given predict-input batch.
    ///
    /// Returns an empty bundle (all `None`) when neither runtime carries
    /// an anchor residual — this is the fast path for fits without
    /// cross-block residualisation. When at least one runtime has a
    /// residual, materialises the marginal + logslope designs at the
    /// predict rows once and computes the per-runtime correction matrices
    /// against each runtime's stored `M`.
    fn build_anchor_correction_matrices(
        &self,
        input: &PredictInput,
        design_logslope: &DesignMatrix,
        z: &Array1<f64>,
    ) -> Result<BmsAnchorCorrections, EstimationError> {
        use crate::inference::model::SavedAnchorKind;
        let needs_score = self
            .score_warp_runtime
            .as_ref()
            .is_some_and(|r| r.anchor_correction.is_some());
        let needs_link = self
            .link_deviation_runtime
            .as_ref()
            .is_some_and(|r| r.anchor_correction.is_some());
        if !needs_score && !needs_link {
            return Ok(BmsAnchorCorrections::default());
        }
        // Materialise the marginal + logslope designs at predict rows.
        // For large-scale predict batches the caller already chunks via
        // `prediction_chunk_rows`, so this densification is bounded per
        // chunk by `chunk_size × (p_marginal + p_logslope)`.
        let marginal_dense = input
            .design
            .try_to_dense_arc(
                "bernoulli marginal-slope predict-time marginal anchor materialisation",
            )
            .map_err(EstimationError::InvalidInput)?;
        let logslope_dense = design_logslope
            .try_to_dense_arc(
                "bernoulli marginal-slope predict-time logslope anchor materialisation",
            )
            .map_err(EstimationError::InvalidInput)?;
        let n_rows = marginal_dense.nrows();
        if logslope_dense.nrows() != n_rows {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope predict anchor materialisation row mismatch: marginal {} vs logslope {}",
                n_rows,
                logslope_dense.nrows()
            )));
        }
        if z.len() != n_rows {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope predict anchor materialisation: z has {} entries, expected {}",
                z.len(),
                n_rows
            )));
        }
        let p_marginal = marginal_dense.ncols();
        let p_logslope = logslope_dense.ncols();
        let d_parametric = p_marginal + p_logslope;
        let mut parametric_rows = Array2::<f64>::zeros((n_rows, d_parametric));
        parametric_rows
            .slice_mut(ndarray::s![.., 0..p_marginal])
            .assign(&marginal_dense.view());
        parametric_rows
            .slice_mut(ndarray::s![.., p_marginal..d_parametric])
            .assign(&logslope_dense.view());

        // Score-warp anchor layout is `[marginal | logslope]` (parametric
        // only; flex-flex anchoring goes the other direction).
        let score_warp = if needs_score {
            let runtime = self.score_warp_runtime.as_ref().unwrap();
            self.validate_runtime_anchor_layout_parametric_only(runtime, "score_warp")?;
            runtime
                .anchor_correction_matrix(parametric_rows.view())
                .map_err(EstimationError::from)?
        } else {
            None
        };

        // Link-deviation anchor layout matches the fit-time stacking in
        // `install_compiled_flex_block_into_runtime`: parametric
        // columns first, then (if a FlexEvaluation component is present)
        // the score-warp runtime's reparameterised basis at predict rows.
        let (link_dev_anchor_rows, link_dev) = if needs_link {
            let runtime = self.link_deviation_runtime.as_ref().unwrap();
            // Determine whether the saved link-dev residual carries a
            // FlexEvaluation tail and validate ordering matches the
            // fit-time invariant (all parametric components first, then
            // at most one FlexEvaluation tail).
            let mut saw_flex_tail = false;
            let mut flex_tail_ncols: usize = 0;
            for (idx, component) in runtime.anchor_components.iter().enumerate() {
                match &component.kind {
                    SavedAnchorKind::Parametric { .. } => {
                        if saw_flex_tail {
                            return Err(EstimationError::InvalidInput(format!(
                                "bernoulli marginal-slope link-deviation saved anchor components \
                                 are out of order: parametric component at index {idx} follows \
                                 a FlexEvaluation tail",
                            )));
                        }
                    }
                    SavedAnchorKind::FlexEvaluation { ncols } => {
                        if saw_flex_tail {
                            return Err(EstimationError::InvalidInput(
                                "bernoulli marginal-slope link-deviation saved anchor components \
                                 carry more than one FlexEvaluation tail; fit-time stacking emits \
                                 at most one (score-warp)"
                                    .to_string(),
                            ));
                        }
                        saw_flex_tail = true;
                        flex_tail_ncols = *ncols;
                    }
                }
            }
            let rows = if saw_flex_tail {
                let score_runtime = self.score_warp_runtime.as_ref().ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "bernoulli marginal-slope link-deviation saved anchor includes a \
                         FlexEvaluation tail but the saved score-warp runtime is missing"
                            .to_string(),
                    )
                })?;
                // Evaluate the score-warp runtime at predict-row z. When
                // the score-warp itself carries an anchor residual, route
                // through `design_with_anchor_rows` so the per-row
                // subtraction is applied; otherwise the raw `design(z)`
                // is the reparameterised basis.
                let score_basis = if score_runtime.anchor_correction.is_some() {
                    score_runtime
                        .design_with_anchor_rows(z, parametric_rows.view())
                        .map_err(EstimationError::from)?
                } else {
                    score_runtime.design(z).map_err(EstimationError::from)?
                };
                if score_basis.ncols() != flex_tail_ncols {
                    return Err(EstimationError::InvalidInput(format!(
                        "bernoulli marginal-slope link-deviation FlexEvaluation tail expects \
                         {} score-warp basis columns at predict rows, got {}",
                        flex_tail_ncols,
                        score_basis.ncols()
                    )));
                }
                let mut combined = Array2::<f64>::zeros((n_rows, d_parametric + flex_tail_ncols));
                combined
                    .slice_mut(ndarray::s![.., 0..d_parametric])
                    .assign(&parametric_rows.view());
                combined
                    .slice_mut(ndarray::s![.., d_parametric..])
                    .assign(&score_basis.view());
                combined
            } else {
                parametric_rows.clone()
            };
            let corr = runtime
                .anchor_correction_matrix(rows.view())
                .map_err(EstimationError::from)?;
            (Some(rows), corr)
        } else {
            (None, None)
        };

        Ok(BmsAnchorCorrections {
            score_warp_anchor_rows: Some(parametric_rows),
            link_dev_anchor_rows,
            score_warp,
            link_dev,
        })
    }

    /// Validate that a saved deviation runtime's anchor residual contains
    /// only `Parametric` components (no `FlexEvaluation` tail). Used for
    /// the score-warp runtime, whose fit-time stacking is parametric-only.
    fn validate_runtime_anchor_layout_parametric_only(
        &self,
        runtime: &SavedCompiledFlexBlock,
        runtime_label: &str,
    ) -> Result<(), EstimationError> {
        use crate::inference::model::SavedAnchorKind;
        for (idx, component) in runtime.anchor_components.iter().enumerate() {
            match &component.kind {
                SavedAnchorKind::Parametric { .. } => {}
                SavedAnchorKind::FlexEvaluation { .. } => {
                    return Err(EstimationError::InvalidInput(format!(
                        "bernoulli marginal-slope {runtime_label} saved anchor component at \
                         index {idx} is FlexEvaluation; only Parametric components are \
                         expected for this runtime",
                    )));
                }
            }
        }
        Ok(())
    }

    fn likelihood_family(&self) -> LikelihoodSpec {
        LikelihoodSpec::binomial_probit()
    }

    fn mean_from_eta(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        Ok(eta.mapv(normal_cdf))
    }

    fn mean_derivative_from_eta(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        Ok(eta.mapv(normal_pdf))
    }

    fn probit_frailty_scale(&self) -> f64 {
        marginal_slope_probit_frailty_scale(self.gaussian_frailty_sd)
    }

    /// Apply the (optional) rank-INT latent-z calibration to a batch of
    /// normalized predict-time z values.
    ///
    /// The calibration was fit on the training z + weights as a Blom-
    /// rankit weighted rank inverse-normal transform; the calibrated
    /// sample is N(0, 1) by construction (exact, not approximate), which
    /// is why the BMS standard-normal closed-form kernel is correct on
    /// the calibrated scale. At predict time, every z that flows into a
    /// kernel evaluation site (`final_eta_and_gradient_from_theta`,
    /// `predict_eta_and_q_chain`, and indirectly the per-row `solve_intercept_scalar`
    /// / `evaluate_prediction_calibration` / `observed_denested_cell_partials_at_z`
    /// helpers that consume per-row scalar z values from the closure-
    /// captured `z` array) must be routed through the same monotone
    /// transform. When `latent_z_calibration` is `None`, this returns
    /// the input unchanged — that case corresponds to training-time z
    /// having passed the strict normality check, so no transform was
    /// applied at fit time either.
    fn apply_latent_z_calibration(&self, z: &Array1<f64>) -> Array1<f64> {
        match &self.latent_z_calibration {
            Some(cal) => Array1::from_iter(z.iter().map(|&zi| cal.apply_at_predict(zi))),
            None => z.clone(),
        }
    }

    /// Apply the (optional) conditional location-scale latent-z calibration
    /// (#905) to a batch of normalized predict-time z values.
    ///
    /// When `Some`, training detected a conditional `E[z|C]`/`Var(z|C)` shift
    /// and replaced its latent score by `ζ = (z − m(C))/√v(C)`. The predictor
    /// MUST apply the identical map, rebuilding the conditioning span `a(C)`
    /// from the marginal prediction design (`input.design`) — the same span the
    /// fit regressed z on. `None` ⇒ no conditional calibration was applied at
    /// fit time, so z passes through unchanged (mutually exclusive with the
    /// rank-INT calibration above).
    fn apply_latent_z_conditional_calibration(
        &self,
        z: &Array1<f64>,
        input: &PredictInput,
    ) -> Result<Array1<f64>, EstimationError> {
        let Some(cal) = self.latent_z_conditional_calibration.as_ref() else {
            return Ok(z.clone());
        };
        let a_block = input.design.to_dense();
        cal.apply(z.view(), a_block.view())
            .map_err(EstimationError::InvalidInput)
    }

    fn rigid_intercept_from_marginal(&self, marginal_eta: f64, slope: f64) -> f64 {
        let probit_scale = self.probit_frailty_scale();
        marginal_eta * (1.0 + (probit_scale * slope).powi(2)).sqrt() / probit_scale
    }

    fn empirical_rigid_intercept_and_gradient(
        &self,
        marginal_eta: f64,
        slope: f64,
        nodes: &[f64],
        weights: &[f64],
    ) -> Result<(f64, f64, f64), EstimationError> {
        let marginal = bernoulli_marginal_link_map(&self.base_link, marginal_eta)
            .map_err(EstimationError::InvalidInput)?;
        let scale = self.probit_frailty_scale();
        let intercept = empirical_intercept_from_marginal(
            marginal.mu,
            marginal.q,
            slope,
            scale,
            nodes,
            weights,
            None,
        )
        .map_err(EstimationError::InvalidInput)?;
        let observed_slope = scale * slope;
        let mut f_a = 0.0;
        let mut f_b = 0.0;
        for (&node, &weight) in nodes.iter().zip(weights.iter()) {
            let eta = intercept + observed_slope * node;
            let pdf = normal_pdf(eta);
            f_a += weight * pdf;
            f_b += weight * pdf * scale * node;
        }
        if !(f_a.is_finite() && f_a > 0.0 && f_b.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "empirical latent prediction calibration derivative is invalid: F_a={f_a}, F_b={f_b}"
            )));
        }
        let a_marginal_eta = marginal.mu1 / f_a;
        let a_slope = -f_b / f_a;
        Ok((intercept, a_marginal_eta, a_slope))
    }

    fn local_empirical_mixture_for_point(
        point: &[f64],
        centers: &[Vec<f64>],
        top_k: usize,
        bandwidth: f64,
    ) -> Result<Vec<(usize, f64)>, EstimationError> {
        if centers.is_empty() {
            return Err(EstimationError::InvalidInput(
                "local empirical latent prediction has no centers".to_string(),
            ));
        }
        if top_k == 0 {
            return Err(EstimationError::InvalidInput(
                "local empirical latent prediction top_k must be positive".to_string(),
            ));
        }
        if !(bandwidth.is_finite() && bandwidth > 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "local empirical latent prediction bandwidth must be finite and positive, got {bandwidth}"
            )));
        }
        let bw2 = bandwidth * bandwidth;
        let mut distances = Vec::<(usize, f64)>::with_capacity(centers.len());
        for (idx, center) in centers.iter().enumerate() {
            if center.len() != point.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "local empirical latent prediction center {idx} dimension mismatch: center={}, point={}",
                    center.len(),
                    point.len()
                )));
            }
            let d2 = center
                .iter()
                .zip(point.iter())
                .map(|(&c, &x)| {
                    let delta = x - c;
                    delta * delta
                })
                .sum::<f64>();
            if !d2.is_finite() {
                return Err(EstimationError::InvalidInput(
                    "local empirical latent prediction distance is non-finite".to_string(),
                ));
            }
            distances.push((idx, d2));
        }
        distances.sort_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .expect("validated local empirical distances are finite")
        });
        let k = top_k.min(distances.len());
        let mut mixture = Vec::with_capacity(k);
        let mut total = 0.0;
        for &(idx, d2) in distances.iter().take(k) {
            let weight = (-0.5 * d2 / bw2).exp().max(1e-300);
            mixture.push((idx, weight));
            total += weight;
        }
        if !(total.is_finite() && total > 0.0) {
            return Err(EstimationError::InvalidInput(
                "local empirical latent prediction mixture has non-positive total weight"
                    .to_string(),
            ));
        }
        for (_, weight) in &mut mixture {
            *weight /= total;
        }
        Ok(mixture)
    }

    fn combine_empirical_grids(
        grids: &[EmpiricalZGrid],
        mixture: &[(usize, f64)],
    ) -> Result<EmpiricalZGrid, EstimationError> {
        let total_len = mixture
            .iter()
            .map(|&(idx, _)| grids.get(idx).map_or(0, |grid| grid.nodes.len()))
            .sum::<usize>();
        let mut nodes = Vec::with_capacity(total_len);
        let mut weights = Vec::with_capacity(total_len);
        let mut total_weight = 0.0;
        for &(grid_idx, grid_weight) in mixture {
            if !(grid_weight.is_finite() && grid_weight >= 0.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "local empirical latent prediction mixture weight must be finite and non-negative, got {grid_weight}"
                )));
            }
            let grid = grids.get(grid_idx).ok_or_else(|| {
                EstimationError::InvalidInput(format!(
                    "local empirical latent prediction grid index {grid_idx} is out of bounds for {} grids",
                    grids.len()
                ))
            })?;
            if grid.nodes.len() != grid.weights.len() || grid.nodes.is_empty() {
                return Err(EstimationError::InvalidInput(format!(
                    "local empirical latent prediction grid {grid_idx} is invalid: nodes={}, weights={}",
                    grid.nodes.len(),
                    grid.weights.len()
                )));
            }
            for (node, weight) in grid.pairs() {
                let combined_weight = grid_weight * weight;
                if !(node.is_finite() && combined_weight.is_finite() && combined_weight >= 0.0) {
                    return Err(EstimationError::InvalidInput(
                        "local empirical latent prediction grid contains invalid node/weight"
                            .to_string(),
                    ));
                }
                nodes.push(node);
                weights.push(combined_weight);
                total_weight += combined_weight;
            }
        }
        if !(total_weight.is_finite() && total_weight > 0.0) {
            return Err(EstimationError::InvalidInput(
                "local empirical latent prediction combined grid has non-positive total weight"
                    .to_string(),
            ));
        }
        for weight in &mut weights {
            *weight /= total_weight;
        }
        Ok(EmpiricalZGrid { nodes, weights })
    }

    fn empirical_grid_for_prediction_row(
        &self,
        input: &PredictInput,
        row: usize,
    ) -> Result<Option<EmpiricalZGrid>, EstimationError> {
        match &self.latent_measure {
            LatentMeasureKind::StandardNormal => Ok(None),
            LatentMeasureKind::GlobalEmpirical { grid } => Ok(Some(grid.clone())),
            LatentMeasureKind::LocalEmpirical {
                centers,
                grids,
                top_k,
                bandwidth,
                ..
            } => {
                let conditioning = input.auxiliary_matrix.as_ref().ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "bernoulli marginal-slope local empirical prediction requires auxiliary conditioning matrix"
                            .to_string(),
                    )
                })?;
                if row >= conditioning.nrows() {
                    return Err(EstimationError::InvalidInput(format!(
                        "local empirical latent prediction row {row} is out of bounds for {} conditioning rows",
                        conditioning.nrows()
                    )));
                }
                let expected_dim = centers.first().map_or(0, Vec::len);
                if conditioning.ncols() != expected_dim {
                    return Err(EstimationError::InvalidInput(format!(
                        "local empirical latent prediction conditioning dimension mismatch: got {}, expected {expected_dim}",
                        conditioning.ncols()
                    )));
                }
                let point = conditioning.row(row).to_vec();
                let mixture =
                    Self::local_empirical_mixture_for_point(&point, centers, *top_k, *bandwidth)?;
                Self::combine_empirical_grids(grids, &mixture).map(Some)
            }
        }
    }

    fn transform_internal_eta_to_base_scale(
        &self,
        internal_eta: Array1<f64>,
        internal_grad: Option<Array2<f64>>,
    ) -> Result<(Array1<f64>, Option<Array2<f64>>), EstimationError> {
        Ok((internal_eta, internal_grad))
    }

    fn link_terms_value_d1(
        &self,
        eta0: &Array1<f64>,
        beta_link_dev: Option<&Array1<f64>>,
        link_dev_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
    ) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
        if let (Some(runtime), Some(beta)) = (&self.link_deviation_runtime, beta_link_dev) {
            // When the runtime carries a cross-block anchor residual, every
            // raw-design row needs `n_row · M` subtracted. `correction_for_row`
            // already holds the precomputed `n_row · M` for this predict row
            // (length basis_dim), so the corrected basis contribution to η
            // is `basis · beta - correction.dot(beta)` for every eta0 entry.
            // Derivative paths are unaffected (the anchor argument is a
            // different scalar than eta0).
            let basis = runtime
                .design_uncorrected(eta0)
                .map_err(EstimationError::from)?;
            let mut value = &basis.dot(beta) + eta0;
            if let Some(corr) = link_dev_correction_for_row {
                let offset = corr.dot(beta);
                for v in value.iter_mut() {
                    *v -= offset;
                }
            } else if runtime.anchor_correction.is_some() {
                return Err(EstimationError::InvalidInput(
                    "bernoulli marginal-slope link-deviation runtime has an anchor residual but \
                     no per-row correction was supplied to link_terms_value_d1"
                        .to_string(),
                ));
            }
            let d1 = runtime
                .first_derivative_design(eta0)
                .map_err(EstimationError::from)?;
            Ok((value, d1.dot(beta) + 1.0))
        } else {
            Ok((eta0.clone(), Array1::ones(eta0.len())))
        }
    }

    fn denested_partition_cells(
        &self,
        a: f64,
        b: f64,
        beta_score_warp: Option<&Array1<f64>>,
        beta_link_dev: Option<&Array1<f64>>,
        score_warp_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
        link_dev_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
    ) -> Result<Vec<crate::families::cubic_cell_kernel::DenestedPartitionCell>, EstimationError>
    {
        let score_breaks = if let Some(runtime) = self.score_warp_runtime.as_ref() {
            runtime.breakpoints().map_err(EstimationError::from)?
        } else {
            Vec::new()
        };
        let link_breaks = if let Some(runtime) = self.link_deviation_runtime.as_ref() {
            runtime.breakpoints().map_err(EstimationError::from)?
        } else {
            Vec::new()
        };
        let mut cells =
            crate::families::cubic_cell_kernel::build_denested_partition_cells_with_tails(
                a,
                b,
                &score_breaks,
                &link_breaks,
                |z| {
                    if let (Some(runtime), Some(beta)) =
                        (self.score_warp_runtime.as_ref(), beta_score_warp)
                    {
                        let mut span = runtime.local_cubic_at(beta, z)?;
                        // `local_cubic_at`'s c0 is `Σ_j basis_c0[span][j] · beta[j]`.
                        // The cross-block residual replaces basis_c0 by
                        // basis_c0 − n_row · M, contributing a row-constant
                        // `correction.dot(beta)` to c0. Higher coefficients
                        // (c1..c3) depend on derivatives of the basis w.r.t.
                        // its own argument and are untouched.
                        if let Some(corr) = score_warp_correction_for_row {
                            span.c0 -= corr.dot(beta);
                        }
                        Ok(span)
                    } else {
                        Ok(crate::families::cubic_cell_kernel::LocalSpanCubic {
                            left: 0.0,
                            right: 1.0,
                            c0: 0.0,
                            c1: 0.0,
                            c2: 0.0,
                            c3: 0.0,
                        })
                    }
                },
                |u| {
                    if let (Some(runtime), Some(beta)) =
                        (self.link_deviation_runtime.as_ref(), beta_link_dev)
                    {
                        let mut span = runtime.local_cubic_at(beta, u)?;
                        if let Some(corr) = link_dev_correction_for_row {
                            span.c0 -= corr.dot(beta);
                        }
                        Ok(span)
                    } else {
                        Ok(crate::families::cubic_cell_kernel::LocalSpanCubic {
                            left: 0.0,
                            right: 1.0,
                            c0: 0.0,
                            c1: 0.0,
                            c2: 0.0,
                            c3: 0.0,
                        })
                    }
                },
            )
            .map_err(EstimationError::InvalidInput)?;
        let scale = self.probit_frailty_scale();
        if scale != 1.0 {
            for partition_cell in &mut cells {
                partition_cell.cell.c0 *= scale;
                partition_cell.cell.c1 *= scale;
                partition_cell.cell.c2 *= scale;
                partition_cell.cell.c3 *= scale;
            }
        }
        Ok(cells)
    }

    fn evaluate_denested_calibration(
        &self,
        a: f64,
        marginal_eta: f64,
        slope: f64,
        beta_score_warp: Option<&Array1<f64>>,
        beta_link_dev: Option<&Array1<f64>>,
        score_warp_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
        link_dev_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
    ) -> Result<(f64, f64, f64), EstimationError> {
        let marginal = bernoulli_marginal_link_map(&self.base_link, marginal_eta)
            .map_err(EstimationError::InvalidInput)?;
        let cells = self.denested_partition_cells(
            a,
            slope,
            beta_score_warp,
            beta_link_dev,
            score_warp_correction_for_row,
            link_dev_correction_for_row,
        )?;
        let scale = self.probit_frailty_scale();
        let mut f = -marginal.mu;
        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        for partition_cell in cells {
            let cell = partition_cell.cell;
            let (dc_da_raw, _) =
                crate::families::cubic_cell_kernel::denested_cell_coefficient_partials(
                    partition_cell.score_span,
                    partition_cell.link_span,
                    a,
                    slope,
                );
            let (d2c_da2_raw, _, _) =
                crate::families::cubic_cell_kernel::denested_cell_second_partials(
                    partition_cell.score_span,
                    partition_cell.link_span,
                    a,
                    slope,
                );
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let d2c_da2 = scale_coeff4(d2c_da2_raw, scale);
            // Derive the moment `max_degree` from the contractions consumed
            // below, instead of hardcoding a magic constant. The second-
            // derivative contraction dominates the first-derivative one, so
            // its required degree is the binding bound. Hardcoding 7 here
            // produced 8 moments while the contraction needs 10 (#321).
            let max_degree =
                crate::families::cubic_cell_kernel::cell_second_derivative_required_max_degree(
                    &dc_da, &dc_da, &d2c_da2,
                );
            let state = crate::families::cubic_cell_kernel::evaluate_cell_moments(cell, max_degree)
                .map_err(EstimationError::InvalidInput)?;
            f += state.value;
            f_a += crate::families::cubic_cell_kernel::cell_first_derivative_from_moments(
                &dc_da,
                &state.moments,
            )
            .map_err(EstimationError::InvalidInput)?;
            f_aa += crate::families::cubic_cell_kernel::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &d2c_da2,
                &state.moments,
            )
            .map_err(EstimationError::InvalidInput)?;
        }
        Ok((f, f_a, f_aa))
    }

    fn observed_denested_cell_partials_at_z(
        &self,
        z_value: f64,
        a: f64,
        b: f64,
        beta_score_warp: Option<&Array1<f64>>,
        beta_link_dev: Option<&Array1<f64>>,
        score_warp_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
        link_dev_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
    ) -> Result<ObservedDenestedCellPartials, EstimationError> {
        use crate::families::cubic_cell_kernel as exact;

        let zero_span = exact::LocalSpanCubic {
            left: 0.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        };
        let u_value = a + b * z_value;
        let score_span = if let (Some(runtime), Some(beta)) =
            (self.score_warp_runtime.as_ref(), beta_score_warp)
        {
            let mut span = runtime
                .local_cubic_at(beta, z_value)
                .map_err(EstimationError::from)?;
            if let Some(corr) = score_warp_correction_for_row {
                span.c0 -= corr.dot(beta);
            }
            span
        } else {
            zero_span
        };
        let link_span = if let (Some(runtime), Some(beta)) =
            (self.link_deviation_runtime.as_ref(), beta_link_dev)
        {
            let mut span = runtime
                .local_cubic_at(beta, u_value)
                .map_err(EstimationError::from)?;
            if let Some(corr) = link_dev_correction_for_row {
                span.c0 -= corr.dot(beta);
            }
            span
        } else {
            zero_span
        };
        let scale = self.probit_frailty_scale();
        let coeff = scale_coeff4(
            exact::denested_cell_coefficients(score_span, link_span, a, b),
            scale,
        );
        let (dc_da_raw, dc_db_raw) =
            exact::denested_cell_coefficient_partials(score_span, link_span, a, b);
        let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) =
            exact::denested_cell_second_partials(score_span, link_span, a, b);
        let (dc_daaa, dc_daab, dc_dabb, dc_dbbb) = exact::denested_cell_third_partials(link_span);
        Ok(ObservedDenestedCellPartials {
            coeff,
            dc_da: scale_coeff4(dc_da_raw, scale),
            dc_db: scale_coeff4(dc_db_raw, scale),
            dc_daa: scale_coeff4(dc_daa_raw, scale),
            dc_dab: scale_coeff4(dc_dab_raw, scale),
            dc_dbb: scale_coeff4(dc_dbb_raw, scale),
            dc_daaa: scale_coeff4(dc_daaa, scale),
            dc_daab: scale_coeff4(dc_daab, scale),
            dc_dabb: scale_coeff4(dc_dabb, scale),
            dc_dbbb: scale_coeff4(dc_dbbb, scale),
        })
    }

    fn evaluate_empirical_denested_calibration(
        &self,
        a: f64,
        marginal_eta: f64,
        slope: f64,
        beta_score_warp: Option<&Array1<f64>>,
        beta_link_dev: Option<&Array1<f64>>,
        grid: &EmpiricalZGrid,
        score_warp_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
        link_dev_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
    ) -> Result<(f64, f64, f64), EstimationError> {
        let marginal = bernoulli_marginal_link_map(&self.base_link, marginal_eta)
            .map_err(EstimationError::InvalidInput)?;
        let mut f = -marginal.mu;
        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        for (node, weight) in grid.pairs() {
            let obs = self.observed_denested_cell_partials_at_z(
                node,
                a,
                slope,
                beta_score_warp,
                beta_link_dev,
                score_warp_correction_for_row,
                link_dev_correction_for_row,
            )?;
            let eta = eval_coeff4_at(&obs.coeff, node);
            let eta_a = eval_coeff4_at(&obs.dc_da, node);
            let eta_aa = eval_coeff4_at(&obs.dc_daa, node);
            let pdf = normal_pdf(eta);
            f += weight * normal_cdf(eta);
            f_a += weight * pdf * eta_a;
            f_aa += weight * pdf * (eta_aa - eta * eta_a * eta_a);
        }
        Ok((f, f_a, f_aa))
    }

    fn evaluate_prediction_calibration(
        &self,
        a: f64,
        marginal_eta: f64,
        slope: f64,
        beta_score_warp: Option<&Array1<f64>>,
        beta_link_dev: Option<&Array1<f64>>,
        empirical_grid: Option<&EmpiricalZGrid>,
        score_warp_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
        link_dev_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
    ) -> Result<(f64, f64, f64), EstimationError> {
        if let Some(grid) = empirical_grid {
            self.evaluate_empirical_denested_calibration(
                a,
                marginal_eta,
                slope,
                beta_score_warp,
                beta_link_dev,
                grid,
                score_warp_correction_for_row,
                link_dev_correction_for_row,
            )
        } else {
            self.evaluate_denested_calibration(
                a,
                marginal_eta,
                slope,
                beta_score_warp,
                beta_link_dev,
                score_warp_correction_for_row,
                link_dev_correction_for_row,
            )
        }
    }

    pub fn from_unified(
        unified: &UnifiedFitResult,
        z_column: String,
        latent_z_normalization: SavedLatentZNormalization,
        latent_measure: LatentMeasureKind,
        baseline_marginal: f64,
        baseline_logslope: f64,
        base_link: InverseLink,
        frailty: FrailtySpec,
        score_warp_runtime: Option<SavedCompiledFlexBlock>,
        link_deviation_runtime: Option<SavedCompiledFlexBlock>,
        latent_z_calibration: Option<crate::families::bms::LatentZRankIntCalibration>,
        latent_z_conditional_calibration: Option<
            crate::families::bms::LatentZConditionalCalibration,
        >,
    ) -> Result<Self, String> {
        let gaussian_frailty_sd = match frailty {
            FrailtySpec::None => None,
            FrailtySpec::GaussianShift {
                sigma_fixed: Some(sigma),
            } => Some(sigma),
            FrailtySpec::GaussianShift { sigma_fixed: None } => {
                return Err(
                    "bernoulli marginal-slope predictor requires a fixed GaussianShift sigma"
                        .to_string(),
                );
            }
            FrailtySpec::HazardMultiplier { .. } => {
                return Err(
                    "bernoulli marginal-slope predictor does not support HazardMultiplier frailty"
                        .to_string(),
                );
            }
        };
        if !matches!(
            base_link,
            InverseLink::Standard(crate::types::StandardLink::Probit)
        ) {
            return Err(
                "bernoulli marginal-slope predictor requires link(type=probit); saved non-probit marginal-slope models must be refit"
                    .to_string(),
            );
        }
        if let Some(runtime) = score_warp_runtime.as_ref() {
            runtime.validate_exact_replay_contract().map_err(|e| {
                format!("bernoulli marginal-slope score-warp runtime is invalid: {e}")
            })?;
        }
        if let Some(runtime) = link_deviation_runtime.as_ref() {
            runtime.validate_exact_replay_contract().map_err(|e| {
                format!("bernoulli marginal-slope link-deviation runtime is invalid: {e}")
            })?;
        }
        // Cross-block anchor residuals on either runtime are now applied
        // per-row by every predict-time `local_cubic_at` / `basis_cubic_at`
        // / `design` call site via `build_anchor_correction_matrices`.
        latent_z_normalization
            .validate("bernoulli marginal-slope predictor")
            .map_err(|e| {
                format!("bernoulli marginal-slope predictor latent z normalization is invalid: {e}")
            })?;
        latent_measure
            .validate("bernoulli marginal-slope predictor latent measure")
            .map_err(|e| {
                format!("bernoulli marginal-slope predictor latent measure is invalid: {e}")
            })?;
        let blocks = &unified.blocks;
        let expected_blocks = 2
            + usize::from(score_warp_runtime.is_some())
            + usize::from(link_deviation_runtime.is_some());
        if blocks.len() != expected_blocks {
            return Err(format!(
                "bernoulli marginal-slope predictor requires exactly {expected_blocks} coefficient blocks under the current exact de-nested semantics, got {}",
                blocks.len()
            ));
        }
        let mut cursor = 2usize;
        let beta_score_warp = if score_warp_runtime.is_some() {
            let beta = blocks
                .get(cursor)
                .ok_or_else(|| "missing score-warp coefficient block".to_string())?
                .beta
                .clone();
            cursor += 1;
            Some(beta)
        } else {
            None
        };
        let beta_link_dev = if link_deviation_runtime.is_some() {
            Some(
                blocks
                    .get(cursor)
                    .ok_or_else(|| "missing link-deviation coefficient block".to_string())?
                    .beta
                    .clone(),
            )
        } else {
            None
        };
        Ok(Self {
            beta_marginal: blocks[0].beta.clone(),
            beta_logslope: blocks[1].beta.clone(),
            beta_score_warp,
            beta_link_dev,
            base_link,
            z_column,
            latent_z_normalization,
            latent_measure,
            baseline_marginal,
            baseline_logslope,
            covariance: unified.beta_covariance().cloned(),
            score_warp_runtime,
            link_deviation_runtime,
            gaussian_frailty_sd,
            latent_z_calibration,
            latent_z_conditional_calibration,
        })
    }

    fn theta(&self) -> Array1<f64> {
        let total = self.beta_marginal.len()
            + self.beta_logslope.len()
            + self.beta_score_warp.as_ref().map_or(0, |b| b.len())
            + self.beta_link_dev.as_ref().map_or(0, |b| b.len());
        let mut theta = Array1::<f64>::zeros(total);
        let mut cursor = 0usize;
        theta
            .slice_mut(ndarray::s![cursor..cursor + self.beta_marginal.len()])
            .assign(&self.beta_marginal);
        cursor += self.beta_marginal.len();
        theta
            .slice_mut(ndarray::s![cursor..cursor + self.beta_logslope.len()])
            .assign(&self.beta_logslope);
        cursor += self.beta_logslope.len();
        if let Some(beta) = self.beta_score_warp.as_ref() {
            theta
                .slice_mut(ndarray::s![cursor..cursor + beta.len()])
                .assign(beta);
            cursor += beta.len();
        }
        if let Some(beta) = self.beta_link_dev.as_ref() {
            theta
                .slice_mut(ndarray::s![cursor..cursor + beta.len()])
                .assign(beta);
        }
        theta
    }

    fn split_theta<'a>(
        &'a self,
        theta: &'a Array1<f64>,
    ) -> Result<
        (
            ArrayView1<'a, f64>,
            ArrayView1<'a, f64>,
            Option<ArrayView1<'a, f64>>,
            Option<ArrayView1<'a, f64>>,
        ),
        EstimationError,
    > {
        let expected = self.theta().len();
        if theta.len() != expected {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope theta length mismatch: expected {expected}, got {}",
                theta.len()
            )));
        }
        let mut cursor = 0usize;
        let marginal = theta.slice(ndarray::s![cursor..cursor + self.beta_marginal.len()]);
        cursor += self.beta_marginal.len();
        let logslope = theta.slice(ndarray::s![cursor..cursor + self.beta_logslope.len()]);
        cursor += self.beta_logslope.len();
        let score_warp = self.beta_score_warp.as_ref().map(|beta| {
            let view = theta.slice(ndarray::s![cursor..cursor + beta.len()]);
            cursor += beta.len();
            view
        });
        let link_dev = self
            .beta_link_dev
            .as_ref()
            .map(|beta| theta.slice(ndarray::s![cursor..cursor + beta.len()]));
        Ok((marginal, logslope, score_warp, link_dev))
    }

    /// Safeguarded monotone root solve for the marginal intercept under the
    /// de-nested flexible model
    ///   η(z) = a + b z + b Δ_h(z) + Δ_w(a + b z).
    fn solve_intercept_scalar(
        &self,
        marginal_eta: f64,
        slope: f64,
        link_dev_beta: Option<&Array1<f64>>,
        score_warp_beta: Option<&Array1<f64>>,
        empirical_grid: Option<&EmpiricalZGrid>,
        warm_start_buf: &mut Array1<f64>,
        score_warp_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
        link_dev_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
    ) -> Result<f64, EstimationError> {
        let marginal = bernoulli_marginal_link_map(&self.base_link, marginal_eta)
            .map_err(EstimationError::InvalidInput)?;
        let eval = |a: f64| -> Result<(f64, f64, f64), String> {
            self.evaluate_prediction_calibration(
                a,
                marginal_eta,
                slope,
                score_warp_beta,
                link_dev_beta,
                empirical_grid,
                score_warp_correction_for_row,
                link_dev_correction_for_row,
            )
            .map_err(|err| err.to_string())
        };

        let probit_scale = self.probit_frailty_scale();
        let a_rigid = self.rigid_intercept_from_marginal(marginal.q, slope);
        let mut intercept = a_rigid;
        if let (Some(_), Some(beta)) = (self.link_deviation_runtime.as_ref(), link_dev_beta) {
            warm_start_buf[0] = a_rigid;
            let one_pt = warm_start_buf.slice(ndarray::s![0..1]).to_owned();
            let (l_val, l_d1) =
                self.link_terms_value_d1(&one_pt, Some(beta), link_dev_correction_for_row)?;
            let ell1 = l_d1[0];
            if ell1 > 1e-8 {
                let ell0 = l_val[0] - ell1 * a_rigid;
                let observed_logslope = probit_scale * ell1 * slope;
                intercept = (marginal.q * (1.0 + observed_logslope * observed_logslope).sqrt()
                    / probit_scale
                    - ell0)
                    / ell1;
            }
        }

        // Same adaptive tolerance the acceptance check below uses; passing
        // a tighter `convergence_tol` would just iterate past what we accept.
        let target = marginal.mu;
        let abs_tol = 1e-8_f64.max(1e-4 * target.abs());

        let (root, _, f_best) = crate::families::monotone_root::solve_monotone_root(
            eval,
            intercept,
            "saved bernoulli intercept",
            abs_tol,
            64,
            48,
        )?;

        if f_best.abs() > abs_tol {
            return Err(EstimationError::InvalidInput(format!(
                "saved bernoulli marginal-slope intercept solve failed: residual={f_best:.3e} at a={root:.6}, target mu={target:.6}"
            )));
        }
        Ok(root)
    }

    fn final_eta_and_gradient_from_theta(
        &self,
        input: &PredictInput,
        theta: &Array1<f64>,
        need_gradient: bool,
    ) -> Result<(Array1<f64>, Option<Array2<f64>>), EstimationError> {
        let z_raw = input.auxiliary_scalar.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction requires auxiliary z column '{}'",
                self.z_column
            ))
        })?;
        let z_normalized = self
            .latent_z_normalization
            .apply(z_raw, "bernoulli marginal-slope prediction")
            .map_err(EstimationError::from)?;
        // P4: when training applied a rank-INT calibration to the latent
        // z (so the BMS rigid kernel could use the closed-form
        // standard-normal path), the predictor MUST apply the same
        // monotone transform to predict-time z before any kernel
        // evaluation. The transform is mathematically exact: piecewise-
        // linear interpolation on (sorted_z, weighted_cdf) followed by
        // Φ⁻¹, both strictly monotone and invertible up to the empirical
        // CDF resolution. `None` ⇒ training-time z passed the strict
        // normality check, no transform was applied, leave z unchanged.
        let z = self.apply_latent_z_calibration(&z_normalized);
        // #905: replace z by ζ = (z − m(C))/√v(C) when training engaged the
        // conditional Auto gate (no-op otherwise; mutually exclusive with the
        // rank-INT calibration above).
        let z = self.apply_latent_z_conditional_calibration(&z, input)?;
        let design_logslope = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "bernoulli marginal-slope prediction requires logslope design".to_string(),
            )
        })?;
        let (beta_marginal, beta_logslope, beta_score_warp, beta_link_dev) =
            self.split_theta(theta)?;
        if self.score_warp_runtime.is_some() != beta_score_warp.is_some() {
            return Err(EstimationError::InvalidInput(
                "bernoulli marginal-slope saved score-warp runtime/coefficients are inconsistent"
                    .to_string(),
            ));
        }
        if self.link_deviation_runtime.is_some() != beta_link_dev.is_some() {
            return Err(EstimationError::InvalidInput(
                "bernoulli marginal-slope saved link-deviation runtime/coefficients are inconsistent"
                    .to_string(),
            ));
        }
        let n = z.len();
        if input.offset.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction primary offset length mismatch: rows={n}, offset={}",
                input.offset.len()
            )));
        }
        let logslope_offset = input
            .offset_noise
            .as_ref()
            .map_or_else(|| Array1::zeros(n), Clone::clone);
        if logslope_offset.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction logslope offset length mismatch: rows={n}, offset_noise={}",
                logslope_offset.len()
            )));
        }
        let marginal_eta = input
            .design
            .dot(&beta_marginal.to_owned())
            .mapv(|v| v + self.baseline_marginal)
            + &input.offset;
        let logslope_eta = design_logslope
            .dot(&beta_logslope.to_owned())
            .mapv(|v| v + self.baseline_logslope)
            + &logslope_offset;
        let flex_active =
            self.score_warp_runtime.is_some() || self.link_deviation_runtime.is_some();
        let marginal_dim = self.beta_marginal.len();
        let logslope_dim = self.beta_logslope.len();
        let score_warp_dim = self.beta_score_warp.as_ref().map_or(0, Array1::len);
        let link_dev_dim = self.beta_link_dev.as_ref().map_or(0, Array1::len);
        let logslope_offset = marginal_dim;
        let score_warp_offset = logslope_offset + logslope_dim;
        let link_dev_offset = score_warp_offset + score_warp_dim;
        let chunk_size = prediction_chunk_rows(theta.len(), 1, n);
        let num_chunks = n.div_ceil(chunk_size);
        let scale = self.probit_frailty_scale();
        // Cross-block anchor corrections: when either runtime carries an
        // anchor residual, precompute the per-row correction matrices
        // (n × runtime_basis_dim) once. Each subsequent per-row evaluation
        // subtracts the corresponding row of these matrices from the raw
        // cubic-span basis output. When neither runtime has a residual,
        // the returned bundle is empty and threading is a no-op.
        let anchor_corrections =
            self.build_anchor_correction_matrices(input, design_logslope, &z)?;
        let marginal_map = marginal_eta
            .iter()
            .map(|&eta| {
                bernoulli_marginal_link_map(&self.base_link, eta)
                    .map_err(EstimationError::InvalidInput)
            })
            .collect::<Result<Vec<_>, _>>()?;

        if !flex_active {
            let (final_eta_internal, marginal_scales, logslope_scales) = match &self.latent_measure
            {
                LatentMeasureKind::StandardNormal => {
                    let sb_vec = logslope_eta.mapv(|b| scale * b);
                    let c_vec = sb_vec.mapv(|sb| (1.0 + sb * sb).sqrt());
                    let final_eta_internal = Array1::from_iter(
                        (0..n).map(|i| c_vec[i] * marginal_eta[i] + sb_vec[i] * z[i]),
                    );
                    let marginal_scales = c_vec;
                    let logslope_scales = Array1::from_iter((0..n).map(|i| {
                        marginal_eta[i] * (scale * scale) * logslope_eta[i] / marginal_scales[i]
                            + scale * z[i]
                    }));
                    (final_eta_internal, marginal_scales, logslope_scales)
                }
                LatentMeasureKind::GlobalEmpirical { grid } => {
                    let mut final_eta = Array1::<f64>::zeros(n);
                    let mut marginal_scales = Array1::<f64>::zeros(n);
                    let mut logslope_scales = Array1::<f64>::zeros(n);
                    for i in 0..n {
                        let (intercept, a_marginal, a_slope) = self
                            .empirical_rigid_intercept_and_gradient(
                                marginal_eta[i],
                                logslope_eta[i],
                                &grid.nodes,
                                &grid.weights,
                            )?;
                        final_eta[i] = intercept + scale * logslope_eta[i] * z[i];
                        marginal_scales[i] = a_marginal;
                        logslope_scales[i] = a_slope + scale * z[i];
                    }
                    (final_eta, marginal_scales, logslope_scales)
                }
                LatentMeasureKind::LocalEmpirical { .. } => {
                    let mut final_eta = Array1::<f64>::zeros(n);
                    let mut marginal_scales = Array1::<f64>::zeros(n);
                    let mut logslope_scales = Array1::<f64>::zeros(n);
                    for i in 0..n {
                        let grid = self
                            .empirical_grid_for_prediction_row(input, i)?
                            .ok_or_else(|| {
                                EstimationError::InvalidInput(
                                    "local empirical latent prediction did not produce a row grid"
                                        .to_string(),
                                )
                            })?;
                        let (intercept, a_marginal, a_slope) = self
                            .empirical_rigid_intercept_and_gradient(
                                marginal_eta[i],
                                logslope_eta[i],
                                &grid.nodes,
                                &grid.weights,
                            )?;
                        final_eta[i] = intercept + scale * logslope_eta[i] * z[i];
                        marginal_scales[i] = a_marginal;
                        logslope_scales[i] = a_slope + scale * z[i];
                    }
                    (final_eta, marginal_scales, logslope_scales)
                }
            };

            if !need_gradient {
                return self.transform_internal_eta_to_base_scale(final_eta_internal, None);
            }

            // Chunk Jacobian: one pass per row fills both blocks.
            let mut grad_internal = Array2::<f64>::zeros((n, theta.len()));
            let mut start = 0usize;
            while start < n {
                let end = (start + chunk_size).min(n);
                let mc = input
                    .design
                    .try_row_chunk(start..end)
                    .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;
                let lc = design_logslope
                    .try_row_chunk(start..end)
                    .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;

                for li in 0..(end - start) {
                    let i = start + li;
                    let c = marginal_scales[i];
                    let g_scale = logslope_scales[i];
                    let mut row = grad_internal.row_mut(i);
                    for j in 0..marginal_dim {
                        row[j] = c * mc[[li, j]];
                    }
                    for j in 0..logslope_dim {
                        row[logslope_offset + j] = g_scale * lc[[li, j]];
                    }
                }

                start = end;
            }
            return self
                .transform_internal_eta_to_base_scale(final_eta_internal, Some(grad_internal));
        }

        // ── Flexible path: per-row intercept solve, chunked Jacobians ──
        let score_warp_obs_design = self
            .score_warp_runtime
            .as_ref()
            .map(|runtime| {
                if runtime.anchor_correction.is_some() {
                    let anchor_rows = anchor_corrections
                        .score_warp_anchor_rows_view()
                        .ok_or_else(|| {
                            EstimationError::InvalidInput(
                                "bernoulli marginal-slope score-warp anchor residual present but \
                                 anchor_corrections bundle is missing the parametric anchor rows"
                                    .to_string(),
                            )
                        })?;
                    runtime
                        .design_with_anchor_rows(&z, anchor_rows)
                        .map_err(EstimationError::from)
                } else {
                    runtime.design(&z).map_err(EstimationError::from)
                }
            })
            .transpose()?;
        let score_dev_obs =
            if let (Some(design), Some(beta)) = (score_warp_obs_design.as_ref(), beta_score_warp) {
                design.dot(&beta.to_owned())
            } else {
                Array1::zeros(n)
            };

        // Solve intercepts and (when gradient needed) IFT scalars in chunk-parallel passes.
        // Outputs are preallocated and each parallel worker writes directly into
        // its exclusive `axis_chunks_iter_mut` slice; no per-chunk owned buffer
        // and no serial copy pass over the result chunks.
        let score_warp_beta_owned = beta_score_warp.as_ref().map(|v| v.to_owned());
        let link_dev_beta_owned = beta_link_dev.as_ref().map(|v| v.to_owned());
        let mut intercepts = Array1::<f64>::zeros(n);
        let mut a_q_vec = need_gradient.then(|| Array1::<f64>::zeros(n));
        let mut a_b_vec = need_gradient.then(|| Array1::<f64>::zeros(n));
        let mut a_h_rows = if need_gradient && score_warp_dim > 0 {
            Some(Array2::<f64>::zeros((n, score_warp_dim)))
        } else {
            None
        };
        let mut a_w_rows = if need_gradient && link_dev_dim > 0 {
            Some(Array2::<f64>::zeros((n, link_dev_dim)))
        } else {
            None
        };
        let solve_result: Result<(), EstimationError> = {
            use ndarray::Axis;
            use rayon::iter::IndexedParallelIterator;
            let intercepts_chunks: Vec<ndarray::ArrayViewMut1<f64>> = intercepts
                .axis_chunks_iter_mut(Axis(0), chunk_size)
                .collect();
            let a_q_chunks: Option<Vec<ndarray::ArrayViewMut1<f64>>> = a_q_vec
                .as_mut()
                .map(|a| a.axis_chunks_iter_mut(Axis(0), chunk_size).collect());
            let a_b_chunks: Option<Vec<ndarray::ArrayViewMut1<f64>>> = a_b_vec
                .as_mut()
                .map(|a| a.axis_chunks_iter_mut(Axis(0), chunk_size).collect());
            let a_h_chunks: Option<Vec<ndarray::ArrayViewMut2<f64>>> = a_h_rows
                .as_mut()
                .map(|a| a.axis_chunks_iter_mut(Axis(0), chunk_size).collect());
            let a_w_chunks: Option<Vec<ndarray::ArrayViewMut2<f64>>> = a_w_rows
                .as_mut()
                .map(|a| a.axis_chunks_iter_mut(Axis(0), chunk_size).collect());

            // Bundle per-chunk sinks so each parallel worker owns disjoint mutable
            // views into the shared output arrays.
            struct FlexSolveSink<'a> {
                intercepts: ndarray::ArrayViewMut1<'a, f64>,
                a_q: Option<ndarray::ArrayViewMut1<'a, f64>>,
                a_b: Option<ndarray::ArrayViewMut1<'a, f64>>,
                a_h: Option<ndarray::ArrayViewMut2<'a, f64>>,
                a_w: Option<ndarray::ArrayViewMut2<'a, f64>>,
            }
            let mut sinks: Vec<FlexSolveSink<'_>> = Vec::with_capacity(num_chunks);
            // Move each Option<Vec> into iterators so we can zip them.
            let mut intercepts_iter = intercepts_chunks.into_iter();
            let mut a_q_iter = a_q_chunks.map(|v| v.into_iter());
            let mut a_b_iter = a_b_chunks.map(|v| v.into_iter());
            let mut a_h_iter = a_h_chunks.map(|v| v.into_iter());
            let mut a_w_iter = a_w_chunks.map(|v| v.into_iter());
            for _ in 0..num_chunks {
                sinks.push(FlexSolveSink {
                    intercepts: intercepts_iter.next().expect("chunk count matches"),
                    a_q: a_q_iter
                        .as_mut()
                        .map(|it| it.next().expect("chunk count matches")),
                    a_b: a_b_iter
                        .as_mut()
                        .map(|it| it.next().expect("chunk count matches")),
                    a_h: a_h_iter
                        .as_mut()
                        .map(|it| it.next().expect("chunk count matches")),
                    a_w: a_w_iter
                        .as_mut()
                        .map(|it| it.next().expect("chunk count matches")),
                });
            }

            // Precompute the score-warp basis cubic table once when the latent
            // grid is row-constant (`GlobalEmpirical`). The per-row inner loop
            // calls `basis_cubic_at(j, node)` with `node` taken from the grid,
            // which is identical for every row in this code path, so the
            // n_rows × n_nodes × score_warp_dim table can be hoisted out of
            // the parallel chunk dispatch. Per-row work only touches the
            // basis-function-specific `c0` shift via `score_corr_row`, which
            // stays inside the row loop. Computed at the top level so no
            // OnceLock / lazy init lives inside the par closure (per the
            // OnceLock + nested rayon deadlock rule).
            let global_score_basis_table: Option<
                Vec<Vec<crate::families::cubic_cell_kernel::LocalSpanCubic>>,
            > = if let (LatentMeasureKind::GlobalEmpirical { grid }, Some(runtime)) =
                (&self.latent_measure, self.score_warp_runtime.as_ref())
            {
                let mut table = Vec::with_capacity(score_warp_dim);
                for j in 0..score_warp_dim {
                    let mut row = Vec::with_capacity(grid.nodes.len());
                    for &node in &grid.nodes {
                        row.push(
                            runtime
                                .basis_cubic_at(j, node)
                                .map_err(EstimationError::from)?,
                        );
                    }
                    table.push(row);
                }
                Some(table)
            } else {
                None
            };
            let global_score_basis_table = global_score_basis_table.as_ref();

            sinks
                .into_par_iter()
                .enumerate()
                .try_for_each(|(chunk_idx, mut sink)| -> Result<(), EstimationError> {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(n);
                let rows = end - start;
                // Destructure the sink into independent `&mut` references so we
                // can borrow them disjointly across iterations of the inner row
                // loop without further reborrowing through `Option::as_mut`.
                let intercepts_view = &mut sink.intercepts;
                let mut a_q = sink.a_q.as_mut();
                let mut a_b = sink.a_b.as_mut();
                let mut a_h = sink.a_h.as_mut();
                let mut a_w = sink.a_w.as_mut();
                let mut warm_start_buf = Array1::<f64>::zeros(1);
                let mut f_h_row = vec![0.0; score_warp_dim];
                let mut f_w_row = vec![0.0; link_dev_dim];

                for local_row in 0..rows {
                    let i = start + local_row;
                    let slope = logslope_eta[i];
                    let q = marginal_eta[i];
                    let empirical_grid = self.empirical_grid_for_prediction_row(input, i)?;
                    let score_corr_row = anchor_corrections.score_warp_row(i);
                    let link_corr_row = anchor_corrections.link_dev_row(i);
                    intercepts_view[local_row] = self.solve_intercept_scalar(
                        q,
                        slope,
                        link_dev_beta_owned.as_ref(),
                        score_warp_beta_owned.as_ref(),
                        empirical_grid.as_ref(),
                        &mut warm_start_buf,
                        score_corr_row,
                        link_corr_row,
                    )?;

                    if !need_gradient {
                        continue;
                    }

                    let intercept = intercepts_view[local_row];
                    let (_, m_a_raw, _) = self.evaluate_prediction_calibration(
                        intercept,
                        q,
                        slope,
                        score_warp_beta_owned.as_ref(),
                        link_dev_beta_owned.as_ref(),
                        empirical_grid.as_ref(),
                        score_corr_row,
                        link_corr_row,
                    )?;
                    let m_a = m_a_raw.max(1e-12);
                    a_q.as_mut().expect("a_q allocated when need_gradient")[local_row] =
                        marginal_map[i].mu1 / m_a;
                    let mut f_b = 0.0;
                    f_h_row.fill(0.0);
                    f_w_row.fill(0.0);
                    if let Some(grid) = empirical_grid.as_ref() {
                        for (node_idx, (node, weight)) in grid.pairs().enumerate() {
                            let obs = self.observed_denested_cell_partials_at_z(
                                node,
                                intercept,
                                slope,
                                score_warp_beta_owned.as_ref(),
                                link_dev_beta_owned.as_ref(),
                                score_corr_row,
                                link_corr_row,
                            )?;
                            let eta = eval_coeff4_at(&obs.coeff, node);
                            let pdf = normal_pdf(eta);
                            f_b += weight * pdf * eval_coeff4_at(&obs.dc_db, node);

                            if let Some(runtime) = self.score_warp_runtime.as_ref() {
                                for j in 0..score_warp_dim {
                                    // When the latent grid is row-constant
                                    // (`GlobalEmpirical`), the per-(j, node)
                                    // basis cubic is identical for every row
                                    // and lives in `global_score_basis_table`.
                                    // Otherwise (`LocalEmpirical`) the grid
                                    // varies per row and we fall back to a
                                    // direct `basis_cubic_at` call.
                                    let mut basis_span = if let Some(table) =
                                        global_score_basis_table
                                    {
                                        table[j][node_idx]
                                    } else {
                                        runtime
                                            .basis_cubic_at(j, node)
                                            .map_err(EstimationError::from)?
                                    };
                                    // `basis_cubic_at` returns the j-th basis
                                    // function's local cubic; the residual
                                    // subtracts `correction[j]` from the
                                    // constant term (row-constant, basis-
                                    // function-specific). Higher span
                                    // coefficients are unaffected.
                                    if let Some(corr) = score_corr_row {
                                        basis_span.c0 -= corr[j];
                                    }
                                    let coeffs = crate::families::cubic_cell_kernel::score_basis_cell_coefficients(
                                        basis_span,
                                        slope,
                                    );
                                    let coeffs = scale_coeff4(coeffs, scale);
                                    f_h_row[j] += weight * pdf * eval_coeff4_at(&coeffs, node);
                                }
                            }

                            if let Some(runtime) = self.link_deviation_runtime.as_ref() {
                                for j in 0..link_dev_dim {
                                    let mut basis_span = runtime
                                        .basis_cubic_at(j, intercept + slope * node)
                                        .map_err(EstimationError::from)?;
                                    if let Some(corr) = link_corr_row {
                                        basis_span.c0 -= corr[j];
                                    }
                                    let coeffs = crate::families::cubic_cell_kernel::link_basis_cell_coefficients(
                                        basis_span,
                                        intercept,
                                        slope,
                                    );
                                    let coeffs = scale_coeff4(coeffs, scale);
                                    f_w_row[j] += weight * pdf * eval_coeff4_at(&coeffs, node);
                                }
                            }
                        }
                    } else {
                        let cells = self.denested_partition_cells(
                            intercept,
                            slope,
                            score_warp_beta_owned.as_ref(),
                            link_dev_beta_owned.as_ref(),
                            score_corr_row,
                            link_corr_row,
                        )?;
                        for partition_cell in cells {
                            let cell = partition_cell.cell;
                            let state =
                                crate::families::cubic_cell_kernel::evaluate_cell_moments(
                                    cell, 9,
                                )
                                .map_err(EstimationError::InvalidInput)?;
                            let (_, dc_db_raw) = crate::families::cubic_cell_kernel::denested_cell_coefficient_partials(
                                partition_cell.score_span,
                                partition_cell.link_span,
                                intercept,
                                slope,
                            );
                            // `denested_partition_cells` scales the cell itself for
                            // Gaussian frailty, so every coefficient partial of
                            // F(a, theta) must carry the same probit scale as F_a.
                            let dc_db = scale_coeff4(dc_db_raw, scale);
                            f_b += crate::families::cubic_cell_kernel::cell_first_derivative_from_moments(
                                &dc_db,
                                &state.moments,
                            )
                            .map_err(EstimationError::InvalidInput)?;

                            let mid = 0.5 * (cell.left + cell.right);
                            if let Some(runtime) = self.score_warp_runtime.as_ref() {
                                for j in 0..score_warp_dim {
                                    let mut basis_span = runtime
                                        .basis_cubic_at(j, mid)
                                        .map_err(EstimationError::from)?;
                                    if let Some(corr) = score_corr_row {
                                        basis_span.c0 -= corr[j];
                                    }
                                    let coeffs = crate::families::cubic_cell_kernel::score_basis_cell_coefficients(
                                        basis_span, slope,
                                    );
                                    let coeffs = scale_coeff4(coeffs, scale);
                                    f_h_row[j] += crate::families::cubic_cell_kernel::cell_first_derivative_from_moments(
                                        &coeffs,
                                        &state.moments,
                                    )
                                    .map_err(EstimationError::InvalidInput)?;
                                }
                            }

                            if let Some(runtime) = self.link_deviation_runtime.as_ref() {
                                for j in 0..link_dev_dim {
                                    let mut basis_span = runtime
                                        .basis_cubic_at(j, intercept + slope * mid)
                                        .map_err(EstimationError::from)?;
                                    if let Some(corr) = link_corr_row {
                                        basis_span.c0 -= corr[j];
                                    }
                                    let coeffs = crate::families::cubic_cell_kernel::link_basis_cell_coefficients(
                                        basis_span,
                                        intercept,
                                        slope,
                                    );
                                    let coeffs = scale_coeff4(coeffs, scale);
                                    f_w_row[j] += crate::families::cubic_cell_kernel::cell_first_derivative_from_moments(
                                        &coeffs,
                                        &state.moments,
                                    )
                                    .map_err(EstimationError::InvalidInput)?;
                                }
                            }
                        }
                    }
                    if let Some(a_h_view) = a_h.as_mut() {
                        let factor = -1.0 / m_a;
                        for j in 0..score_warp_dim {
                            a_h_view[[local_row, j]] = factor * f_h_row[j];
                        }
                    }
                    if let Some(a_w_view) = a_w.as_mut() {
                        let factor = -1.0 / m_a;
                        for j in 0..link_dev_dim {
                            a_w_view[[local_row, j]] = factor * f_w_row[j];
                        }
                    }
                    a_b.as_mut().expect("a_b allocated when need_gradient")[local_row] =
                        -f_b / m_a;
                }
                Ok(())
            })
        };
        solve_result?;

        let eta_base = &intercepts + &(&logslope_eta * &z);

        let mut link_c_obs: Option<Array1<f64>> = None;
        let mut link_basis_obs: Option<Array2<f64>> = None;
        let link_dev_obs = if let (Some(runtime), Some(beta_owned)) = (
            self.link_deviation_runtime.as_ref(),
            link_dev_beta_owned.as_ref(),
        ) {
            let basis = if runtime.anchor_correction.is_some() {
                let anchor_rows =
                    anchor_corrections
                        .link_dev_anchor_rows_view()
                        .ok_or_else(|| {
                            EstimationError::InvalidInput(
                            "bernoulli marginal-slope link-deviation anchor residual present but \
                             anchor_corrections bundle is missing the parametric anchor rows"
                                .to_string(),
                        )
                        })?;
                runtime
                    .design_with_anchor_rows(&eta_base, anchor_rows)
                    .map_err(EstimationError::from)?
            } else {
                runtime.design(&eta_base).map_err(EstimationError::from)?
            };
            let dev = basis.dot(beta_owned);
            if need_gradient {
                let d1 = runtime
                    .first_derivative_design(&eta_base)
                    .map_err(EstimationError::from)?;
                let mut c_obs = d1.dot(beta_owned);
                c_obs.mapv_inplace(|v| v + 1.0);
                link_c_obs = Some(c_obs);
                link_basis_obs = Some(basis);
            }
            dev
        } else {
            Array1::zeros(n)
        };
        let final_eta_internal =
            (&eta_base + &(&logslope_eta * &score_dev_obs) + &link_dev_obs).mapv(|v| scale * v);

        if !need_gradient {
            return self.transform_internal_eta_to_base_scale(final_eta_internal, None);
        }

        let a_q_vec = a_q_vec.unwrap();
        let a_b_vec = a_b_vec.unwrap();

        // Emit chunk Jacobians using precomputed scalars; each worker writes
        // directly into its exclusive `axis_chunks_iter_mut` slice of the
        // preallocated `grad` output so no serial copy pass is needed.
        let mut grad = Array2::<f64>::zeros((n, theta.len()));
        {
            use ndarray::Axis;
            use rayon::iter::IndexedParallelIterator;
            let grad_result: Result<(), String> = grad
                .axis_chunks_iter_mut(Axis(0), chunk_size)
                .into_par_iter()
                .enumerate()
                .try_for_each(|(chunk_idx, mut grad_chunk)| -> Result<(), String> {
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(n);
                    let mc = input
                        .design
                        .try_row_chunk(start..end)
                        .map_err(|e| e.to_string())?;
                    let lc = design_logslope
                        .try_row_chunk(start..end)
                        .map_err(|e| e.to_string())?;
                    let rows = end - start;

                    for li in 0..rows {
                        let i = start + li;
                        let mut row = grad_chunk.row_mut(li);

                        let a_q = a_q_vec[i];
                        for j in 0..marginal_dim {
                            row[j] = a_q * mc[[li, j]];
                        }

                        let base_multiplier = link_c_obs.as_ref().map_or(1.0, |c| c[i]);
                        let g_scale = base_multiplier * (a_b_vec[i] + z[i]) + score_dev_obs[i];
                        for j in 0..logslope_dim {
                            row[logslope_offset + j] = g_scale * lc[[li, j]];
                        }

                        if let (Some(a_h_rows), Some(obs_design)) =
                            (a_h_rows.as_ref(), score_warp_obs_design.as_ref())
                        {
                            let slope = logslope_eta[i];
                            for j in 0..score_warp_dim {
                                row[score_warp_offset + j] =
                                    base_multiplier * a_h_rows[[i, j]] + slope * obs_design[[i, j]];
                            }
                        }

                        if let Some(a_w_rows) = a_w_rows.as_ref() {
                            for j in 0..link_dev_dim {
                                row[link_dev_offset + j] = a_w_rows[[i, j]];
                            }
                        }

                        if let (Some(link_c), Some(link_basis)) =
                            (link_c_obs.as_ref(), link_basis_obs.as_ref())
                        {
                            let c = link_c[i];
                            for j in 0..marginal_dim {
                                row[j] *= c;
                            }
                            for j in 0..link_dev_dim {
                                row[link_dev_offset + j] =
                                    c * row[link_dev_offset + j] + link_basis[[i, j]];
                            }
                        }
                    }
                    Ok(())
                });
            grad_result.map_err(EstimationError::InvalidInput)?;
        }
        if scale != 1.0 {
            grad.mapv_inplace(|v| scale * v);
        }
        self.transform_internal_eta_to_base_scale(final_eta_internal, Some(grad))
    }

    /// Per-row final (base-scale) linear predictor for an arbitrary
    /// coefficient vector `theta` in the saved `[marginal | logslope |
    /// score_warp? | link_dev?]` block order. The marginal-slope rigid
    /// kernel is applied exactly per row, so the returned η is the same
    /// object the point predictor consumes — only parameterised by an
    /// external draw instead of `self.theta()`. Used by the posterior
    /// predictive path (#1049) to map each Laplace draw to its η surface
    /// before the shared eta→bands collapse; the response scale is the
    /// probit inverse link `μ = Φ(η)`.
    pub fn final_eta_from_theta(
        &self,
        input: &PredictInput,
        theta: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        let (eta, _) = self.final_eta_and_gradient_from_theta(input, theta, false)?;
        Ok(eta)
    }

    /// Length of the concatenated coefficient vector this predictor
    /// consumes (`marginal + logslope + score_warp? + link_dev?`). The
    /// posterior predictive path validates each saved draw against this
    /// before mapping it through [`Self::final_eta_from_theta`].
    pub fn theta_len(&self) -> usize {
        self.beta_marginal.len()
            + self.beta_logslope.len()
            + self.beta_score_warp.as_ref().map_or(0, Array1::len)
            + self.beta_link_dev.as_ref().map_or(0, Array1::len)
    }

    fn eta_standard_error_from_covariance(
        &self,
        input: &PredictInput,
        covariance: &Array2<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        let theta = self.theta();
        let backend = PredictionCovarianceBackend::from_dense(covariance.view());
        linear_predictor_se_from_backend(&backend, input.design.nrows(), |rows| {
            let chunk_input = slice_predict_input(input, rows).map_err(|e| e.to_string())?;
            let (_, grad) = self
                .final_eta_and_gradient_from_theta(&chunk_input, &theta, true)
                .map_err(|e| e.to_string())?;
            let grad = grad.ok_or_else(|| {
                "bernoulli marginal-slope analytic predictor gradient was not produced".to_string()
            })?;
            Ok(vec![grad])
        })
    }

    fn eta_standard_error_from_backend(
        &self,
        input: &PredictInput,
        backend: &PredictionCovarianceBackend<'_>,
    ) -> Result<Array1<f64>, EstimationError> {
        let theta = self.theta();
        linear_predictor_se_from_backend(backend, input.design.nrows(), |rows| {
            let chunk_input = slice_predict_input(input, rows).map_err(|e| e.to_string())?;
            let (_, grad) = self
                .final_eta_and_gradient_from_theta(&chunk_input, &theta, true)
                .map_err(|e| e.to_string())?;
            let grad = grad.ok_or_else(|| {
                "bernoulli marginal-slope analytic predictor gradient was not produced".to_string()
            })?;
            Ok(vec![grad])
        })
    }

    /// Per-row `(eta, ∂eta/∂q_marginal)` under the exact IFT pull-back.
    ///
    /// Returns the same `eta` as `predict_plugin_response`/`predict_linear_predictor`
    /// plus the analytic derivative of the internal probit index with respect to
    /// the per-row marginal q (the linear predictor before the de-nested
    /// calibration). Survival prediction multiplies the second component by the
    /// per-row `dq/dt` to obtain the exact hazard time derivative under
    /// score-warp / link-deviation flex blocks.
    ///
    /// Rigid path (no flex blocks): `∂eta/∂q = c = sqrt(1 + (s b)^2)`, recovering
    /// the rigid-path probit-frailty composition. Flex path: `∂eta/∂q =
    /// scale · link_c_obs · a_q` where `link_c_obs = 1 + Δ_w'(eta_base)` is the
    /// link-deviation slope at the observed `eta_base = a + b z` and `a_q =
    /// φ(q) / |F_a|` is the implicit-function derivative of the calibration
    /// intercept (mirrors the bernoulli `final_eta_and_gradient_from_theta`
    /// flex branch lines 1399-1593).
    pub fn predict_eta_and_q_chain(
        &self,
        input: &PredictInput,
    ) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
        let z_raw = input.auxiliary_scalar.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction requires auxiliary z column '{}'",
                self.z_column
            ))
        })?;
        let z_normalized = self
            .latent_z_normalization
            .apply(z_raw, "bernoulli marginal-slope prediction")
            .map_err(EstimationError::from)?;
        // P4: see `final_eta_and_gradient_from_theta` for the rationale.
        // The rank-INT calibration is a mathematically exact monotone
        // transform; both the rigid standard-normal kernel and the
        // implicit-function chain rule consume the calibrated z, never
        // the raw normalized z, exactly mirroring fit-time semantics.
        let z = self.apply_latent_z_calibration(&z_normalized);
        // #905: replace z by ζ = (z − m(C))/√v(C) when training engaged the
        // conditional Auto gate (no-op otherwise; mutually exclusive with the
        // rank-INT calibration above).
        let z = self.apply_latent_z_conditional_calibration(&z, input)?;
        let design_logslope = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "bernoulli marginal-slope prediction requires logslope design".to_string(),
            )
        })?;
        let n = z.len();
        if input.offset.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction primary offset length mismatch: rows={n}, offset={}",
                input.offset.len()
            )));
        }
        let logslope_offset = input
            .offset_noise
            .as_ref()
            .map_or_else(|| Array1::zeros(n), Clone::clone);
        if logslope_offset.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction logslope offset length mismatch: rows={n}, offset_noise={}",
                logslope_offset.len()
            )));
        }
        let marginal_eta = input
            .design
            .dot(&self.beta_marginal)
            .mapv(|v| v + self.baseline_marginal)
            + &input.offset;
        let logslope_eta = design_logslope
            .dot(&self.beta_logslope)
            .mapv(|v| v + self.baseline_logslope)
            + &logslope_offset;
        let scale = self.probit_frailty_scale();
        let flex_active =
            self.score_warp_runtime.is_some() || self.link_deviation_runtime.is_some();

        // Rigid path mirrors `final_eta_and_gradient_from_theta` lines 1342-1383:
        //   eta = c·q + s·b·z,  ∂eta/∂q = c.
        if !flex_active {
            match &self.latent_measure {
                LatentMeasureKind::StandardNormal => {
                    // Vectorize: sb = scale·logslope, c = sqrt(1 + sb²),
                    // eta = c·marginal_eta + sb·z, ∂eta/∂q = c.
                    let sb = logslope_eta.mapv(|x| scale * x);
                    let deta_dq = sb.mapv(|s| (1.0 + s * s).sqrt());
                    let eta = &deta_dq * marginal_eta + &sb * z;
                    return Ok((eta, deta_dq));
                }
                _ => {
                    let mut eta = Array1::<f64>::zeros(n);
                    let mut deta_dq = Array1::<f64>::zeros(n);
                    for i in 0..n {
                        let grid = self
                            .empirical_grid_for_prediction_row(input, i)?
                            .ok_or_else(|| {
                                EstimationError::InvalidInput(
                                    "empirical latent prediction did not produce a row grid"
                                        .to_string(),
                                )
                            })?;
                        let (intercept, a_marginal, _) = self
                            .empirical_rigid_intercept_and_gradient(
                                marginal_eta[i],
                                logslope_eta[i],
                                &grid.nodes,
                                &grid.weights,
                            )?;
                        eta[i] = intercept + scale * logslope_eta[i] * z[i];
                        deta_dq[i] = a_marginal;
                    }
                    return Ok((eta, deta_dq));
                }
            }
        }

        // Flex path: solve the per-row intercept, then evaluate
        //   eta = scale · (a + b·z + b·Δ_h(z) + Δ_w(a + b·z))
        //   ∂eta/∂q = scale · (1 + Δ_w'(a + b·z)) · ∂a/∂q,
        //   ∂a/∂q   = φ(q) / |F_a|         (IFT, marginal_link is probit so mu1 = φ(q))
        // Mirrors `final_eta_and_gradient_from_theta` lines 1385-1621.
        let marginal_map = marginal_eta
            .iter()
            .map(|&eta_marg| {
                bernoulli_marginal_link_map(&self.base_link, eta_marg)
                    .map_err(EstimationError::InvalidInput)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Cross-block anchor corrections (see final_eta_and_gradient_from_theta
        // for the design); precompute once before the per-row loop.
        let anchor_corrections =
            self.build_anchor_correction_matrices(input, design_logslope, &z)?;
        // Per-row: solve intercept scalar, evaluate denested calibration,
        // record (intercept, a_q). The `warm_start_buf` is just per-call
        // scratch — give each rayon worker its own buffer via fold init.
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let pairs: Result<Vec<(f64, f64)>, EstimationError> = (0..n)
            .into_par_iter()
            .map_init(
                || Array1::<f64>::zeros(1),
                |warm_start_buf, i| {
                    let q = marginal_eta[i];
                    let slope = logslope_eta[i];
                    let empirical_grid = self.empirical_grid_for_prediction_row(input, i)?;
                    let score_corr_row = anchor_corrections.score_warp_row(i);
                    let link_corr_row = anchor_corrections.link_dev_row(i);
                    let intercept = self.solve_intercept_scalar(
                        q,
                        slope,
                        self.beta_link_dev.as_ref(),
                        self.beta_score_warp.as_ref(),
                        empirical_grid.as_ref(),
                        warm_start_buf,
                        score_corr_row,
                        link_corr_row,
                    )?;
                    let (_, m_a_raw, _) = self.evaluate_prediction_calibration(
                        intercept,
                        q,
                        slope,
                        self.beta_score_warp.as_ref(),
                        self.beta_link_dev.as_ref(),
                        empirical_grid.as_ref(),
                        score_corr_row,
                        link_corr_row,
                    )?;
                    let m_a = m_a_raw.max(1e-12);
                    Ok((intercept, marginal_map[i].mu1 / m_a))
                },
            )
            .collect();
        let pairs = pairs?;
        let mut intercepts = Array1::<f64>::zeros(n);
        let mut a_q = Array1::<f64>::zeros(n);
        for (i, (intercept, a)) in pairs.into_iter().enumerate() {
            intercepts[i] = intercept;
            a_q[i] = a;
        }

        let score_dev_obs = if let (Some(runtime), Some(beta)) = (
            self.score_warp_runtime.as_ref(),
            self.beta_score_warp.as_ref(),
        ) {
            let design = if runtime.anchor_correction.is_some() {
                let anchor_rows = anchor_corrections
                    .score_warp_anchor_rows_view()
                    .ok_or_else(|| {
                        EstimationError::InvalidInput(
                            "bernoulli marginal-slope score-warp anchor residual present but \
                             anchor_corrections bundle is missing the parametric anchor rows"
                                .to_string(),
                        )
                    })?;
                runtime
                    .design_with_anchor_rows(&z, anchor_rows)
                    .map_err(EstimationError::from)?
            } else {
                runtime.design(&z).map_err(EstimationError::from)?
            };
            design.dot(beta)
        } else {
            Array1::zeros(n)
        };
        let eta_base = &intercepts + &(&logslope_eta * &z);
        let (link_dev_obs, link_c_obs) = if let (Some(runtime), Some(beta)) = (
            self.link_deviation_runtime.as_ref(),
            self.beta_link_dev.as_ref(),
        ) {
            let basis = if runtime.anchor_correction.is_some() {
                let anchor_rows =
                    anchor_corrections
                        .link_dev_anchor_rows_view()
                        .ok_or_else(|| {
                            EstimationError::InvalidInput(
                            "bernoulli marginal-slope link-deviation anchor residual present but \
                             anchor_corrections bundle is missing the parametric anchor rows"
                                .to_string(),
                        )
                        })?;
                runtime
                    .design_with_anchor_rows(&eta_base, anchor_rows)
                    .map_err(EstimationError::from)?
            } else {
                runtime.design(&eta_base).map_err(EstimationError::from)?
            };
            let dev = basis.dot(beta);
            let d1 = runtime
                .first_derivative_design(&eta_base)
                .map_err(EstimationError::from)?;
            let mut c_obs = d1.dot(beta);
            c_obs.mapv_inplace(|v| v + 1.0);
            (dev, c_obs)
        } else {
            (Array1::zeros(n), Array1::ones(n))
        };
        let final_eta_internal =
            (&eta_base + &(&logslope_eta * &score_dev_obs) + &link_dev_obs).mapv(|v| scale * v);
        let deta_dq = (&link_c_obs * &a_q).mapv(|v| scale * v);
        Ok((final_eta_internal, deta_dq))
    }
}

impl PredictionTransform for BernoulliMarginalSlopePredictor {
    fn point_state(&self, input: &PredictInput) -> Result<LinearState, EstimationError> {
        let eta = self.final_eta_from_theta(input, &self.theta())?;
        let mean = self.mean_from_eta(&eta)?;
        let (eta_se, mean_se) = if let Some(covariance) = self.covariance.as_ref() {
            let theta = self.theta();
            if covariance.nrows() != theta.len() || covariance.ncols() != theta.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "bernoulli marginal-slope covariance dimension mismatch: expected {}x{}, got {}x{}",
                    theta.len(),
                    theta.len(),
                    covariance.nrows(),
                    covariance.ncols()
                )));
            }
            let eta_se = self.eta_standard_error_from_covariance(input, covariance)?;
            let mean_se = eta_se.clone() * self.mean_derivative_from_eta(&eta)?;
            (Some(eta_se), Some(mean_se))
        } else {
            (None, None)
        };
        Ok(LinearState {
            eta,
            mean,
            eta_se,
            mean_se,
            covariance_corrected_used: false,
        })
    }

    fn linear_state(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        pass: PredictPass,
        covariance_mode: InferenceCovarianceMode,
    ) -> Result<LinearState, EstimationError> {
        let eta = self.final_eta_from_theta(input, &self.theta())?;
        match pass {
            PredictPass::FullUncertainty => {
                // Select the covariance the caller requested (conditional vs.
                // smoothing-corrected) instead of always using the conditional
                // backend, and report which was used.
                let (backend, covariance_corrected_used) = fit.select_uncertainty_backend(
                    self.theta().len(),
                    covariance_mode,
                    "bernoulli marginal-slope",
                )?;
                let eta_se = self.eta_standard_error_from_backend(input, &backend)?;
                let mean = self.mean_from_eta(&eta)?;
                let mean_se = eta_se.clone() * self.mean_derivative_from_eta(&eta)?;
                Ok(LinearState {
                    eta,
                    mean,
                    eta_se: Some(eta_se),
                    mean_se: Some(mean_se),
                    covariance_corrected_used,
                })
            }
            PredictPass::PosteriorMean => {
                // Posterior-mean integration uses the conditional posterior.
                let backend = require_posterior_mean_backend(
                    fit,
                    self.covariance.as_ref(),
                    self.theta().len(),
                    "bernoulli marginal-slope posterior mean",
                )?;
                let eta_se = self.eta_standard_error_from_backend(input, &backend)?;
                let strategy = strategy_for_family(self.likelihood_family(), Some(&self.base_link));
                let quadctx = crate::quadrature::QuadratureContext::new();
                let mean = Array1::from_iter(
                    eta.iter()
                        .zip(eta_se.iter())
                        .map(|(&eta_i, &se)| strategy.posterior_mean(&quadctx, eta_i, se))
                        .collect::<Result<Vec<_>, _>>()?,
                );
                Ok(LinearState {
                    eta,
                    mean,
                    eta_se: Some(eta_se),
                    mean_se: None,
                    covariance_corrected_used: false,
                })
            }
        }
    }

    fn response(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        self.mean_from_eta(eta)
    }

    fn response_jacobian_rows(&self, pass: PredictPass) -> ResponseInterval {
        assert!(std::mem::size_of_val(&pass) > 0);
        ResponseInterval::TransformEta
    }

    fn bounds(&self) -> ResponseBounds {
        ResponseBounds::for_family(&self.likelihood_family().response)
    }

    fn response_family(&self) -> ResponseFamily {
        self.likelihood_family().response.clone()
    }
}

impl PredictableModel for BernoulliMarginalSlopePredictor {
    fn predict_plugin_response(
        &self,
        input: &PredictInput,
    ) -> Result<PredictResult, EstimationError> {
        predict_plugin_response_generic(self, input)
    }

    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
        predict_with_uncertainty_generic(self, input)
    }

    fn predict_noise_scale(
        &self,
        predict_input: &PredictInput,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        assert!(std::mem::size_of_val(predict_input) > 0);
        Ok(None)
    }

    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        predict_full_uncertainty_generic(self, input, fit, options)
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PosteriorMeanOptions,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        predict_posterior_mean_generic(self, input, fit, options)
    }

    fn n_blocks(&self) -> usize {
        2 + usize::from(self.beta_score_warp.is_some()) + usize::from(self.beta_link_dev.is_some())
    }

    fn block_roles(&self) -> Vec<BlockRole> {
        let mut roles = vec![BlockRole::Location, BlockRole::Scale];
        if self.beta_score_warp.is_some() {
            roles.push(BlockRole::Mean);
        }
        if self.beta_link_dev.is_some() {
            roles.push(BlockRole::LinkWiggle);
        }
        roles
    }
}

