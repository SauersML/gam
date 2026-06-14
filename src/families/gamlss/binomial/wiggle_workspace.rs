// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

/// Matrix-free joint-Hessian operator for the 3-block binomial
/// location-scale wiggle family. See `BinomialLocationScaleWiggleHessianRowPieces`
/// for the per-row weight structure.
pub(crate) struct BinomialLocationScaleWiggleHessianWorkspace {
    family: BinomialLocationScaleWiggleFamily,
    block_states: Vec<ParameterBlockState>,
    x_t: Arc<Array2<f64>>,
    x_ls: Arc<Array2<f64>>,
    pieces: BinomialLocationScaleWiggleHessianRowPieces,
}

impl BinomialLocationScaleWiggleHessianWorkspace {
    pub(crate) fn new(
        family: BinomialLocationScaleWiggleFamily,
        block_states: Vec<ParameterBlockState>,
        x_t: Array2<f64>,
        x_ls: Array2<f64>,
    ) -> Result<Self, String> {
        let pieces = family.wiggle_hessian_row_pieces(&block_states)?;
        Ok(Self {
            family,
            block_states,
            x_t: Arc::new(x_t),
            x_ls: Arc::new(x_ls),
            pieces,
        })
    }

    /// Apply a Horvitz–Thompson outer-row subsample mask to the precomputed
    /// per-row coefficient arrays in place.
    ///
    /// Each sampled row's `coeff_*[i]` is multiplied by its
    /// `WeightedOuterRow.weight` (the HT inverse-inclusion factor 1/π_i —
    /// uniform or stratified sampling both supported). All non-sampled rows
    /// are zeroed. Because every downstream assembly (`hessian_dense`,
    /// `hessian_matvec`, `hessian_diagonal`) is row-linear in these arrays
    /// via `Xᵀ diag(W) Y`, the resulting joint-Hessian is an unbiased
    /// estimator of the full-data joint Hessian. The `b0`/`d0` basis matrices
    /// are independent of the per-row weights and remain unchanged.
    pub(crate) fn apply_outer_subsample(
        &mut self,
        rows: &[crate::families::marginal_slope_shared::WeightedOuterRow],
    ) {
        let n = self.pieces.coeff_tt.len();
        let mut mask_tt = Array1::<f64>::zeros(n);
        let mut mask_tl = Array1::<f64>::zeros(n);
        let mut mask_ll = Array1::<f64>::zeros(n);
        let mut mask_tw_b = Array1::<f64>::zeros(n);
        let mut mask_tw_d = Array1::<f64>::zeros(n);
        let mut mask_lw_b = Array1::<f64>::zeros(n);
        let mut mask_lw_d = Array1::<f64>::zeros(n);
        let mut maskww = Array1::<f64>::zeros(n);
        for r in rows {
            let i = r.index;
            let w = r.weight;
            mask_tt[i] = self.pieces.coeff_tt[i] * w;
            mask_tl[i] = self.pieces.coeff_tl[i] * w;
            mask_ll[i] = self.pieces.coeff_ll[i] * w;
            mask_tw_b[i] = self.pieces.coeff_tw_b[i] * w;
            mask_tw_d[i] = self.pieces.coeff_tw_d[i] * w;
            mask_lw_b[i] = self.pieces.coeff_lw_b[i] * w;
            mask_lw_d[i] = self.pieces.coeff_lw_d[i] * w;
            maskww[i] = self.pieces.coeffww[i] * w;
        }
        self.pieces.coeff_tt = mask_tt;
        self.pieces.coeff_tl = mask_tl;
        self.pieces.coeff_ll = mask_ll;
        self.pieces.coeff_tw_b = mask_tw_b;
        self.pieces.coeff_tw_d = mask_tw_d;
        self.pieces.coeff_lw_b = mask_lw_b;
        self.pieces.coeff_lw_d = mask_lw_d;
        self.pieces.coeffww = maskww;
    }
}

impl ExactNewtonJointHessianWorkspace for BinomialLocationScaleWiggleHessianWorkspace {
    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        // Same Hv structure as `hessian_matvec`, but routed through the
        // already-existing `assemble_dense` row-pieces helper (eight GEMMs
        // covering h_tt, h_tl, h_ll, h_tw_b, h_tw_d, h_lw_b, h_lw_d, h_ww).
        // Avoids `total` canonical-basis HVPs in
        // `MatrixFreeSpdOperator::materialize_dense_operator`, which at
        // large scale (n≈320k, p_total≈82) costs ~568s per κ-iter versus
        // ~1s for the dense build.
        let dense = self
            .pieces
            .assemble_dense(self.x_t.as_ref(), self.x_ls.as_ref())?;
        Ok(Some(dense))
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let pw = self.pieces.b0.ncols();
        let total = pt + pls + pw;
        if v.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleWiggle matvec dimension mismatch: got {}, expected {}",
                    v.len(),
                    total
                ),
            }
            .into());
        }
        let v_t = v.slice(s![0..pt]);
        let v_ls = v.slice(s![pt..pt + pls]);
        let v_w = v.slice(s![pt + pls..total]);

        let u_t = self.x_t.dot(&v_t);
        let u_ls = self.x_ls.dot(&v_ls);
        let u_b = self.pieces.b0.dot(&v_w);
        let u_d = self.pieces.d0.dot(&v_w);

        let r_t = &self.pieces.coeff_tt * &u_t
            + &self.pieces.coeff_tl * &u_ls
            + &self.pieces.coeff_tw_b * &u_b
            + &self.pieces.coeff_tw_d * &u_d;
        let r_ls = &self.pieces.coeff_tl * &u_t
            + &self.pieces.coeff_ll * &u_ls
            + &self.pieces.coeff_lw_b * &u_b
            + &self.pieces.coeff_lw_d * &u_d;
        let r_b = &self.pieces.coeff_tw_b * &u_t
            + &self.pieces.coeff_lw_b * &u_ls
            + &self.pieces.coeffww * &u_b;
        let r_d = &self.pieces.coeff_tw_d * &u_t + &self.pieces.coeff_lw_d * &u_ls;

        let out_t = fast_atv(self.x_t.as_ref(), &r_t);
        let out_ls = fast_atv(self.x_ls.as_ref(), &r_ls);
        let out_w = fast_atv(&self.pieces.b0, &r_b) + &fast_atv(&self.pieces.d0, &r_d);

        let mut out = Array1::<f64>::zeros(total);
        out.slice_mut(s![0..pt]).assign(&out_t);
        out.slice_mut(s![pt..pt + pls]).assign(&out_ls);
        out.slice_mut(s![pt + pls..total]).assign(&out_w);
        Ok(Some(out))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let pw = self.pieces.b0.ncols();
        let total = pt + pls + pw;
        let mut diag = Array1::<f64>::zeros(total);
        let n = self.pieces.coeff_tt.len();
        for j in 0..pt {
            let col = self.x_t.column(j);
            let mut acc = 0.0;
            for i in 0..n {
                let v = col[i];
                acc += self.pieces.coeff_tt[i] * v * v;
            }
            diag[j] = acc;
        }
        for j in 0..pls {
            let col = self.x_ls.column(j);
            let mut acc = 0.0;
            for i in 0..n {
                let v = col[i];
                acc += self.pieces.coeff_ll[i] * v * v;
            }
            diag[pt + j] = acc;
        }
        for j in 0..pw {
            let col = self.pieces.b0.column(j);
            let mut acc = 0.0;
            for i in 0..n {
                let v = col[i];
                acc += self.pieces.coeffww[i] * v * v;
            }
            diag[pt + pls + j] = acc;
        }
        Ok(Some(diag))
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative(&self.block_states, d_beta_flat)
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        self.family.bls_wiggle_directional_operator(
            &self.block_states,
            self.x_t.clone(),
            self.x_ls.clone(),
            d_beta_flat,
        )
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &self.block_states,
                d_beta_u_flat,
                d_beta_v_flat,
            )
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        self.family.bls_wiggle_second_directional_operator(
            &self.block_states,
            self.x_t.clone(),
            self.x_ls.clone(),
            d_beta_u,
            d_beta_v,
        )
    }
}

impl CustomFamilyGenerative for BinomialLocationScaleWiggleFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != self.y.len() || eta_ls.len() != self.y.len() || etaw.len() != self.y.len()
        {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleWiggleFamily generative size mismatch".to_string(),
            }
            .into());
        }
        let mean = gamlss_rowwise_map_result(self.y.len(), |i| {
            let sigma = exp_sigma_from_eta_scalar(eta_ls[i]);
            let q0 = binomial_location_scale_q0(eta_t[i], sigma);
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q0 + etaw[i])
                .map_err(|e| format!("location-scale inverse-link evaluation failed: {e}"))?;
            Ok(jet.mu)
        })?;
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Bernoulli,
        })
    }
}
