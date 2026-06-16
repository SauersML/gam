// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

pub(crate) struct BinomialLocationScaleHessianWorkspace {
    pub(crate) family: BinomialLocationScaleFamily,
    pub(crate) x_t: DesignMatrix,
    pub(crate) x_ls: DesignMatrix,
    pub(crate) core: BinomialLocationScaleCore,
    pub(crate) coeff_tt: Array1<f64>,
    pub(crate) coeff_tl: Array1<f64>,
    pub(crate) coeff_ll: Array1<f64>,
    pub(crate) direction_eta_cache: Mutex<HashMap<BinomialDirectionKey, Arc<BinomialDirectionEta>>>,
    pub(crate) first_coeff_cache: Mutex<HashMap<BinomialDirectionKey, Arc<BinomialRowCoeffTriple>>>,
    // No `second_coeff_cache` deliberately: see `second_coefficients` for why
    // the per-pair cache was a memory-only loss at large-scale shape.
}

#[derive(Clone, Eq, Hash, PartialEq)]
pub(crate) struct BinomialDirectionKey {
    pub(crate) bits: Vec<u64>,
}

impl BinomialDirectionKey {
    pub(crate) fn from_array(v: &Array1<f64>) -> Self {
        Self {
            bits: v.iter().map(|value| value.to_bits()).collect(),
        }
    }
}

pub(crate) struct BinomialDirectionEta {
    pub(crate) t: Array1<f64>,
    pub(crate) ls: Array1<f64>,
}

pub(crate) struct BinomialRowCoeffTriple {
    pub(crate) tt: Arc<Array1<f64>>,
    pub(crate) tl: Arc<Array1<f64>>,
    pub(crate) ll: Arc<Array1<f64>>,
}

impl BinomialLocationScaleHessianWorkspace {
    pub(crate) fn new(
        family: BinomialLocationScaleFamily,
        block_states: Vec<ParameterBlockState>,
        x_t: DesignMatrix,
        x_ls: DesignMatrix,
    ) -> Result<Self, String> {
        let eta_t = &block_states[BinomialLocationScaleFamily::BLOCK_T].eta;
        let eta_ls = &block_states[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].eta;
        let core = binomial_location_scale_core(
            &family.y,
            &family.weights,
            eta_t,
            eta_ls,
            None,
            &family.link_kind,
        )?;
        let (coeff_tt, coeff_tl, coeff_ll) =
            family.exact_newton_joint_hessian_row_coefficients(&block_states)?;
        Ok(Self {
            family,
            x_t,
            x_ls,
            core,
            coeff_tt,
            coeff_tl,
            coeff_ll,
            direction_eta_cache: Mutex::new(HashMap::new()),
            first_coeff_cache: Mutex::new(HashMap::new()),
        })
    }

    pub(crate) fn direction_eta(
        &self,
        key: &BinomialDirectionKey,
        d_beta: &Array1<f64>,
        pt: usize,
        total: usize,
    ) -> Arc<BinomialDirectionEta> {
        if let Some(value) = self
            .direction_eta_cache
            .lock()
            .expect("binomial direction eta cache lock poisoned")
            .get(key)
            .cloned()
        {
            return value;
        }
        let value = Arc::new(BinomialDirectionEta {
            t: self
                .x_t
                .matrixvectormultiply(&d_beta.slice(s![0..pt]).to_owned()),
            ls: self
                .x_ls
                .matrixvectormultiply(&d_beta.slice(s![pt..total]).to_owned()),
        });
        let mut cache = self
            .direction_eta_cache
            .lock()
            .expect("binomial direction eta cache lock poisoned");
        cache
            .entry(key.clone())
            .or_insert_with(|| value.clone())
            .clone()
    }

    pub(crate) fn first_coefficients(
        &self,
        key: &BinomialDirectionKey,
        eta: &BinomialDirectionEta,
    ) -> Result<Arc<BinomialRowCoeffTriple>, String> {
        if let Some(value) = self
            .first_coeff_cache
            .lock()
            .expect("binomial first coefficient cache lock poisoned")
            .get(key)
            .cloned()
        {
            return Ok(value);
        }
        let (tt, tl, ll) = binomial_location_scale_first_directional_coefficients(
            &self.family.y,
            &self.family.weights,
            &self.core,
            &eta.t,
            &eta.ls,
            &self.family.link_kind,
        )?;
        let value = Arc::new(BinomialRowCoeffTriple {
            tt: Arc::new(tt),
            tl: Arc::new(tl),
            ll: Arc::new(ll),
        });
        let mut cache = self
            .first_coeff_cache
            .lock()
            .expect("binomial first coefficient cache lock poisoned");
        Ok(cache
            .entry(key.clone())
            .or_insert_with(|| value.clone())
            .clone())
    }

    /// No caching here, deliberately: at large-scale shape (n=320k, K=14 outer
    /// coords) the K² ≈ 196 unique direction-pairs are queried exactly once
    /// per outer Hessian eval, and each cached entry stored 3·n f64s
    /// = ~7.7 MB → ~1.5 GB peak per eval with zero practical hit-rate.
    /// Across outer evals the directions shift with ρ/ψ so cross-eval hits
    /// are nil. Computing on demand is O(n) — under 10 ms at this scale,
    /// dwarfed by the (n × p²) trace work that consumes the result.
    pub(crate) fn second_coefficients(
        &self,
        eta_u: &BinomialDirectionEta,
        eta_v: &BinomialDirectionEta,
    ) -> Result<Arc<BinomialRowCoeffTriple>, String> {
        let (tt, tl, ll) = binomial_location_scalesecond_directional_coefficients(
            &self.family.y,
            &self.family.weights,
            &self.core,
            &eta_u.t,
            &eta_u.ls,
            &eta_v.t,
            &eta_v.ls,
            &self.family.link_kind,
        )?;
        Ok(Arc::new(BinomialRowCoeffTriple {
            tt: Arc::new(tt),
            tl: Arc::new(tl),
            ll: Arc::new(ll),
        }))
    }

    /// Apply a Horvitz–Thompson outer-row subsample mask to the precomputed
    /// per-row coefficient arrays in place. Each sampled row's `coeff_*[i]`
    /// is multiplied by its `WeightedOuterRow.weight` (the HT inverse-
    /// inclusion factor 1/π_i); non-sampled rows are zeroed. Because every
    /// downstream assembly (`hessian_dense`, `hessian_matvec`,
    /// `hessian_diagonal`) is row-linear in these arrays via `Xᵀ diag(W) X`,
    /// the resulting joint-Hessian is an unbiased estimator of the full-data
    /// joint Hessian.
    pub(crate) fn apply_outer_subsample(
        &mut self,
        rows: &[crate::solver::outer_subsample::WeightedOuterRow],
    ) {
        let n = self.coeff_tt.len();
        let mut mask_tt = Array1::<f64>::zeros(n);
        let mut mask_tl = Array1::<f64>::zeros(n);
        let mut mask_ll = Array1::<f64>::zeros(n);
        for r in rows {
            let i = r.index;
            mask_tt[i] = self.coeff_tt[i] * r.weight;
            mask_tl[i] = self.coeff_tl[i] * r.weight;
            mask_ll[i] = self.coeff_ll[i] * r.weight;
        }
        self.coeff_tt = mask_tt;
        self.coeff_tl = mask_tl;
        self.coeff_ll = mask_ll;
    }
}

impl ExactNewtonJointHessianWorkspace for BinomialLocationScaleHessianWorkspace {
    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        // Same Hv structure as `hessian_matvec`, built once via 3 GEMMs:
        //   H_tt = X_tᵀ diag(coeff_tt) X_t,
        //   H_tl = X_tᵀ diag(coeff_tl) X_ls,
        //   H_ll = X_lsᵀ diag(coeff_ll) X_ls,
        // versus letting `MatrixFreeSpdOperator::materialize_dense_operator`
        // reconstruct the dense Hessian via `total` canonical-basis HVPs. At
        // large scale, canonical-basis materialization costs p_total full
        // Hessian-vector products. The design helpers below stream row chunks,
        // so the only dense object retained here is the small p_total×p_total
        // coefficient Hessian.
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let total = pt + pls;
        let h_tt = xt_diag_x_design(&self.x_t, &self.coeff_tt)?;
        let h_tl = xt_diag_y_design(&self.x_t, &self.coeff_tl, &self.x_ls)?;
        let h_ll = xt_diag_x_design(&self.x_ls, &self.coeff_ll)?;
        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pt, 0..pt]).assign(&h_tt);
        h.slice_mut(s![0..pt, pt..total]).assign(&h_tl);
        h.slice_mut(s![pt..total, pt..total]).assign(&h_ll);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let total = pt + pls;
        if v.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScale matvec dimension mismatch: got {}, expected {}",
                    v.len(),
                    total
                ),
            }
            .into());
        }
        // u_t = X_t v_t, u_ls = X_ls v_ls
        let u_t = self
            .x_t
            .matrixvectormultiply(&v.slice(s![0..pt]).to_owned());
        let u_ls = self
            .x_ls
            .matrixvectormultiply(&v.slice(s![pt..total]).to_owned());
        // r_t = D_tt .* u_t + D_tl .* u_ls; r_ls = D_tl .* u_t + D_ll .* u_ls
        let r_t = &self.coeff_tt * &u_t + &self.coeff_tl * &u_ls;
        let r_ls = &self.coeff_tl * &u_t + &self.coeff_ll * &u_ls;
        // (X_t^T r_t, X_ls^T r_ls)
        let out_t = self.x_t.transpose_vector_multiply(&r_t);
        let out_ls = self.x_ls.transpose_vector_multiply(&r_ls);
        let mut out = Array1::<f64>::zeros(total);
        out.slice_mut(s![0..pt]).assign(&out_t);
        out.slice_mut(s![pt..total]).assign(&out_ls);
        Ok(Some(out))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let total = pt + pls;
        let mut diag = Array1::<f64>::zeros(total);
        let diag_t = design_weighted_column_squares(&self.x_t, &self.coeff_tt)?;
        let diag_ls = design_weighted_column_squares(&self.x_ls, &self.coeff_ll)?;
        diag.slice_mut(s![0..pt]).assign(&diag_t);
        diag.slice_mut(s![pt..total]).assign(&diag_ls);
        Ok(Some(diag))
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .directional_derivative_operator(d_beta_flat)?
            .map(|operator| operator.to_dense()))
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::reml_outer_engine::HyperOperator>>, String>
    {
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let total = pt + pls;
        if d_beta_flat.len() != total {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "BinomialLocationScale dH operator: d_beta length {} != {}",
                    d_beta_flat.len(),
                    total
                ),
            }
            .into());
        }
        let key = BinomialDirectionKey::from_array(d_beta_flat);
        let eta = self.direction_eta(&key, d_beta_flat, pt, total);
        let coeffs = self.first_coefficients(&key, &eta)?;
        Ok(Some(Arc::new(make_two_block_design_row_coeff_operator(
            self.x_t.clone(),
            self.x_ls.clone(),
            coeffs.tt.clone(),
            coeffs.tl.clone(),
            coeffs.ll.clone(),
        )?)))
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .second_directional_derivative_operator(d_beta_u_flat, d_beta_v_flat)?
            .map(|operator| operator.to_dense()))
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::reml_outer_engine::HyperOperator>>, String>
    {
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let total = pt + pls;
        if d_beta_u.len() != total || d_beta_v.len() != total {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "BinomialLocationScale d2H operator: d_beta_{{u,v}} length {}/{} != {}",
                    d_beta_u.len(),
                    d_beta_v.len(),
                    total
                ),
            }
            .into());
        }
        let key_u = BinomialDirectionKey::from_array(d_beta_u);
        let key_v = BinomialDirectionKey::from_array(d_beta_v);
        let eta_u = self.direction_eta(&key_u, d_beta_u, pt, total);
        let eta_v = self.direction_eta(&key_v, d_beta_v, pt, total);
        let coeffs = self.second_coefficients(&eta_u, &eta_v)?;
        Ok(Some(Arc::new(make_two_block_design_row_coeff_operator(
            self.x_t.clone(),
            self.x_ls.clone(),
            coeffs.tt.clone(),
            coeffs.tl.clone(),
            coeffs.ll.clone(),
        )?)))
    }
}
