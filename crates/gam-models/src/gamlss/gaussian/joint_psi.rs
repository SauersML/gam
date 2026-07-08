// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

pub(crate) struct LocationScaleJointPsiDirection {
    pub(crate) block_idx: usize,
    pub(crate) local_idx: usize,
    pub(crate) x_primary_psi: PsiDesignMap,
    pub(crate) x_ls_psi: PsiDesignMap,
    pub(crate) z_primary_psi: Array1<f64>,
    pub(crate) z_ls_psi: Array1<f64>,
}

pub(crate) struct LocationScaleJointPsiSecondDrifts {
    pub(crate) x_primary_ab_action: Option<CustomFamilyPsiSecondDesignAction>,
    pub(crate) x_ls_ab_action: Option<CustomFamilyPsiSecondDesignAction>,
    pub(crate) x_primary_ab: Option<Array2<f64>>,
    pub(crate) x_ls_ab: Option<Array2<f64>>,
    pub(crate) z_primary_ab: Array1<f64>,
    pub(crate) z_ls_ab: Array1<f64>,
}

/// Shared interface that the Gaussian and Binomial location-scale families (and
/// their wiggle variants) expose to the unified joint ψ workspace.
///
/// The four families are structurally identical at the workspace level: each
/// owns two dense block designs (location + log-scale), produces a per-ψ
/// direction, and assembles second-order ψ terms and a ψ-Hessian directional
/// derivative from those parts. They differ only in (1) the concrete
/// [`Direction`](Self::Direction) struct produced (Gaussian vs Binomial field
/// names), (2) the family-name fragment in the dense-designs error message, and
/// (3) whether an optional Horvitz–Thompson outer-row subsample is threaded
/// into the per-row weight arrays (Gaussian does; Binomial ignores it and runs
/// the full-data exact path). This single trait gives the generic
/// [`LocationScaleJointPsiWorkspace`] one dispatch surface; each family's impl
/// is a thin delegation to inherent methods it already owns.
pub(crate) trait LocationScaleJointPsiFamily: Clone + Send + Sync + 'static {
    /// Per-ψ joint direction produced by this family.
    type Direction: Send + Sync + 'static;

    /// Family-name fragment used in the workspace's dense-designs error
    /// message so the originating family stays visible after unification.
    const LABEL: &'static str;

    fn ws_policy(&self) -> &gam_runtime::resource::ResourcePolicy;

    fn ws_exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String>;

    fn ws_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        policy: &gam_runtime::resource::ResourcePolicy,
    ) -> Result<Option<Self::Direction>, String>;

    fn ws_psi_second_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &Self::Direction,
        psi_b: &Self::Direction,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        subsample: Option<&[crate::outer_subsample::WeightedOuterRow]>,
    ) -> Result<ExactNewtonJointPsiSecondOrderTerms, String>;

    fn ws_psi_hessian_directional_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        psi_dir: &Self::Direction,
        d_beta_flat: &Array1<f64>,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        subsample: Option<&[crate::outer_subsample::WeightedOuterRow]>,
    ) -> Result<Array2<f64>, String>;
}

impl LocationScaleJointPsiFamily for GaussianLocationScaleFamily {
    type Direction = LocationScaleJointPsiDirection;
    const LABEL: &'static str = "GaussianLocationScaleFamily";

    fn ws_policy(&self) -> &gam_runtime::resource::ResourcePolicy {
        &self.policy
    }

    fn ws_exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        self.exact_joint_dense_block_designs(specs)
    }

    fn ws_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        policy: &gam_runtime::resource::ResourcePolicy,
    ) -> Result<Option<LocationScaleJointPsiDirection>, String> {
        self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            design_loc,
            design_scale,
            policy,
        )
    }

    fn ws_psi_second_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &LocationScaleJointPsiDirection,
        psi_b: &LocationScaleJointPsiDirection,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        subsample: Option<&[crate::outer_subsample::WeightedOuterRow]>,
    ) -> Result<ExactNewtonJointPsiSecondOrderTerms, String> {
        self.exact_newton_joint_psisecond_order_terms_from_parts(
            block_states,
            derivative_blocks,
            psi_a,
            psi_b,
            design_loc,
            design_scale,
            subsample,
        )
    }

    fn ws_psi_hessian_directional_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        psi_dir: &LocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        subsample: Option<&[crate::outer_subsample::WeightedOuterRow]>,
    ) -> Result<Array2<f64>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_from_parts(
            block_states,
            psi_dir,
            d_beta_flat,
            design_loc,
            design_scale,
            subsample,
        )
    }
}

impl LocationScaleJointPsiFamily for GaussianLocationScaleWiggleFamily {
    type Direction = LocationScaleJointPsiDirection;
    const LABEL: &'static str = "GaussianLocationScaleWiggleFamily";

    fn ws_policy(&self) -> &gam_runtime::resource::ResourcePolicy {
        &self.policy
    }

    fn ws_exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        self.exact_joint_dense_block_designs(specs)
    }

    fn ws_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        policy: &gam_runtime::resource::ResourcePolicy,
    ) -> Result<Option<LocationScaleJointPsiDirection>, String> {
        self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            design_loc,
            design_scale,
            policy,
        )
    }

    fn ws_psi_second_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &LocationScaleJointPsiDirection,
        psi_b: &LocationScaleJointPsiDirection,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        _: Option<&[crate::outer_subsample::WeightedOuterRow]>,
    ) -> Result<ExactNewtonJointPsiSecondOrderTerms, String> {
        // Wiggle ψ path: full-data exact (= trivially unbiased). The
        // wiggle-specific second-order from-parts function inlines 30+
        // per-row coefficient arrays (`coeff_mm{,_a,_b,_ab}`,
        // `coeff_ml{,_a,_b,_ab}`, `coeff_ll{,_a,_b,_ab}`, `a{,_a,_b,_ab}`,
        // `c{,_a,_b,_ab}`, `l{,_a,_b,_ab}`, `dw_{a,b,ab}`, `s_mu*`, `s_ls*`,
        // `s_w*`, ...) instead of packing them into a struct like the
        // non-wiggle GLS path's `GaussianJointPsi{First,Second}Weights`.
        // Each is row-linear in `rows.{w,m,n,kappa,...}` and the direction
        // vectors so HT masking is theoretically clean, but threading a mask
        // across that many call sites is brittle (any missed array silently
        // biases the estimator). The outer score remains unbiased without
        // touching the wiggle ψ path: HT-unbiased LL
        // (`log_likelihood_only_with_options`) + HT-unbiased ρ-Hessian
        // (`exact_newton_joint_hessian_workspace_with_options`) +
        // exact-unbiased ψ (this path) = unbiased. Broadening to the wiggle
        // ψ path is a follow-up that should refactor the inline arrays into
        // `WiggleJointPsi{First,Second}Weights` structs mirroring
        // `GaussianJointPsi{First,Second}Weights` so a single
        // `apply_ht_mask_wiggle*` helper can mask everything in one place.
        self.exact_newton_joint_psisecond_order_terms_from_parts(
            block_states,
            derivative_blocks,
            psi_a,
            psi_b,
            design_loc,
            design_scale,
        )
    }

    fn ws_psi_hessian_directional_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        psi_dir: &LocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        _: Option<&[crate::outer_subsample::WeightedOuterRow]>,
    ) -> Result<Array2<f64>, String> {
        // Same rationale as `ws_psi_second_order_terms_from_parts` above:
        // the wiggle ψ-Hessian directional-derivative function also inlines
        // dozens of per-row arrays. Full-data is exact (= trivially
        // unbiased), so the total outer score remains unbiased.
        self.exact_newton_joint_psihessian_directional_derivative_from_parts(
            block_states,
            psi_dir,
            d_beta_flat,
            design_loc,
            design_scale,
        )
    }
}

/// Generic joint exact-Newton ψ workspace shared by every location-scale
/// family (Gaussian / Binomial, with or without a wiggle block) via the
/// [`LocationScaleJointPsiFamily`] trait.
///
/// The workspace owns the two dense block designs as `Arc<Array2<f64>>` (the
/// per-family `ws_exact_joint_dense_block_designs` hands back a `Cow`, which is
/// materialized once here), the per-ψ direction cache, and an optional
/// Horvitz–Thompson outer-row subsample. When the subsample is `Some`, every
/// per-row weight array produced inside the second-order ψ Hessian and the
/// ψ-Hessian directional-derivative computations is masked: each sampled row's
/// contribution is scaled by `WeightedOuterRow.weight = 1/π_i` and non-sampled
/// rows are zeroed. Because every downstream assembly is row-linear in those
/// arrays, the resulting ψ score and ψ Hessian remain unbiased estimators of
/// the full-data quantities. Families that do not thread the subsample (the
/// Binomial families) construct with `new` and the field stays `None`.
pub(crate) struct LocationScaleJointPsiWorkspace<F: LocationScaleJointPsiFamily> {
    pub(crate) family: F,
    pub(crate) block_states: Vec<ParameterBlockState>,
    pub(crate) derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    pub(crate) design_loc: Arc<Array2<f64>>,
    pub(crate) design_scale: Arc<Array2<f64>>,
    pub(crate) psi_directions: ExactNewtonJointPsiDirectCache<F::Direction>,
    pub(crate) outer_score_subsample: Option<Arc<crate::outer_subsample::OuterScoreSubsample>>,
}

impl<F: LocationScaleJointPsiFamily> LocationScaleJointPsiWorkspace<F> {
    pub(crate) fn new(
        family: F,
        block_states: Vec<ParameterBlockState>,
        specs: &[ParameterBlockSpec],
        derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    ) -> Result<Self, String> {
        Self::new_with_subsample(family, block_states, specs, derivative_blocks, None)
    }

    pub(crate) fn new_with_subsample(
        family: F,
        block_states: Vec<ParameterBlockState>,
        specs: &[ParameterBlockSpec],
        derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
        outer_score_subsample: Option<Arc<crate::outer_subsample::OuterScoreSubsample>>,
    ) -> Result<Self, String> {
        let Some((design_loc, design_scale)) =
            family.ws_exact_joint_dense_block_designs(Some(specs))?
        else {
            return Err(GamlssError::UnsupportedConfiguration {
                reason: format!(
                    "{} exact joint psi workspace requires dense block designs",
                    F::LABEL,
                ),
            }
            .into());
        };
        let design_loc = shared_dense_arc(design_loc.as_ref());
        let design_scale = shared_dense_arc(design_scale.as_ref());
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum();
        Ok(Self {
            family,
            block_states,
            derivative_blocks,
            design_loc,
            design_scale,
            psi_directions: ExactNewtonJointPsiDirectCache::new(psi_dim),
            outer_score_subsample,
        })
    }

    pub(crate) fn psi_direction(
        &self,
        psi_index: usize,
    ) -> Result<Option<Arc<F::Direction>>, String> {
        self.psi_directions.get_or_try_init(psi_index, || {
            self.family.ws_psi_direction(
                &self.block_states,
                &self.derivative_blocks,
                psi_index,
                self.design_loc.as_ref(),
                self.design_scale.as_ref(),
                self.family.ws_policy(),
            )
        })
    }

    pub(crate) fn subsample_rows(&self) -> Option<&[crate::outer_subsample::WeightedOuterRow]> {
        self.outer_score_subsample
            .as_ref()
            .map(|s| s.rows.as_ref().as_slice())
    }
}

impl<F> ExactNewtonJointPsiWorkspace for LocationScaleJointPsiWorkspace<F>
where
    F: LocationScaleJointPsiFamily,
{
    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_i) = self.psi_direction(psi_i)? else {
            return Ok(None);
        };
        let Some(dir_j) = self.psi_direction(psi_j)? else {
            return Ok(None);
        };
        Ok(Some(self.family.ws_psi_second_order_terms_from_parts(
            &self.block_states,
            &self.derivative_blocks,
            dir_i.as_ref(),
            dir_j.as_ref(),
            self.design_loc.as_ref(),
            self.design_scale.as_ref(),
            self.subsample_rows(),
        )?))
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<gam_problem::DriftDerivResult>, String> {
        let Some(dir) = self.psi_direction(psi_index)? else {
            return Ok(None);
        };
        Ok(Some(gam_problem::DriftDerivResult::Dense(
            self.family.ws_psi_hessian_directional_from_parts(
                &self.block_states,
                dir.as_ref(),
                d_beta_flat,
                self.design_loc.as_ref(),
                self.design_scale.as_ref(),
                self.subsample_rows(),
            )?,
        )))
    }
}

pub(crate) type GaussianLocationScaleExactNewtonJointPsiWorkspace =
    LocationScaleJointPsiWorkspace<GaussianLocationScaleFamily>;

pub(crate) type GaussianLocationScaleWiggleExactNewtonJointPsiWorkspace =
    LocationScaleJointPsiWorkspace<GaussianLocationScaleWiggleFamily>;

#[derive(Clone)]
pub struct GaussianJointRowScalars {
    pub(crate) obs_weight: Array1<f64>,
    pub(crate) w: Array1<f64>,
    pub(crate) m: Array1<f64>,
    pub(crate) n: Array1<f64>,
    /// κ = (dσ/dη_ls)/σ for the active sigma link.
    /// The cross Hessian block H_{μ,ls} carries an overall κ factor and the
    /// scale-scale block H_{ls,ls} carries κ².
    pub(crate) kappa: Array1<f64>,
    /// κ' = dκ/dη_ls = κ(1−κ) for the logb link. The static H_{ls,ls} block
    /// carries a κ'·(a−n) term, so κ' threads through every dH directional
    /// weight via the chain rule.
    pub(crate) kappa_prime: Array1<f64>,
    /// κ'' = κ(1−κ)(1−2κ); appears in d²H_{ls,ls} via the second
    /// η-derivative of κ'·(a−n).
    pub(crate) kappa_dprime: Array1<f64>,
}

pub(crate) struct GaussianJointPsiFirstWeights {
    pub(crate) objective_psirow: Array1<f64>,
    pub(crate) scoremu: Array1<f64>,
    pub(crate) score_ls: Array1<f64>,
    pub(crate) dscoremu: Array1<f64>,
    pub(crate) dscore_ls: Array1<f64>,
    pub(crate) hmumu: Array1<f64>,
    pub(crate) hmu_ls: Array1<f64>,
    pub(crate) h_ls_ls: Array1<f64>,
    pub(crate) dhmumu: Array1<f64>,
    pub(crate) dhmu_ls: Array1<f64>,
    pub(crate) dh_ls_ls: Array1<f64>,
}

pub(crate) struct GaussianJointPsiSecondWeights {
    pub(crate) objective_psi_psirow: Array1<f64>,
    pub(crate) d2scoremu: Array1<f64>,
    pub(crate) d2score_ls: Array1<f64>,
    pub(crate) d2hmumu: Array1<f64>,
    pub(crate) d2hmu_ls: Array1<f64>,
    pub(crate) d2h_ls_ls: Array1<f64>,
}

pub(crate) struct GaussianJointPsiMixedDriftWeights {
    pub(crate) dhmumu_u: Array1<f64>,
    pub(crate) dhmu_ls_u: Array1<f64>,
    pub(crate) dh_ls_ls_u: Array1<f64>,
    pub(crate) d2hmumu: Array1<f64>,
    pub(crate) d2hmu_ls: Array1<f64>,
    pub(crate) d2h_ls_ls: Array1<f64>,
}

/// Apply a Horvitz–Thompson outer-row subsample mask to every per-row array
/// of a `GaussianJointPsiFirstWeights` in place: each sampled row's
/// contribution is multiplied by `WeightedOuterRow.weight = 1/π_i` and all
/// non-sampled rows are zeroed. Every downstream assembly
/// (`gaussian_joint_psi*_fromweights`,
/// `build_two_block_custom_family_joint_psi_operator_from_actions`) consumes
/// these arrays row-linearly via `Xᵀ diag(W) Y` and `weighted_crossprod_psi_maps`,
/// so the resulting first-order ψ score and Hessian remain unbiased estimators
/// of the full-data quantities.
pub(crate) fn apply_ht_mask_first(
    weights: &mut GaussianJointPsiFirstWeights,
    rows: &[crate::outer_subsample::WeightedOuterRow],
) {
    let n = weights.objective_psirow.len();
    let mut obj = Array1::<f64>::zeros(n);
    let mut smu = Array1::<f64>::zeros(n);
    let mut sls = Array1::<f64>::zeros(n);
    let mut dsmu = Array1::<f64>::zeros(n);
    let mut dsls = Array1::<f64>::zeros(n);
    let mut hmm = Array1::<f64>::zeros(n);
    let mut hml = Array1::<f64>::zeros(n);
    let mut hll = Array1::<f64>::zeros(n);
    let mut dhmm = Array1::<f64>::zeros(n);
    let mut dhml = Array1::<f64>::zeros(n);
    let mut dhll = Array1::<f64>::zeros(n);
    for r in rows {
        let i = r.index;
        let w = r.weight;
        obj[i] = weights.objective_psirow[i] * w;
        smu[i] = weights.scoremu[i] * w;
        sls[i] = weights.score_ls[i] * w;
        dsmu[i] = weights.dscoremu[i] * w;
        dsls[i] = weights.dscore_ls[i] * w;
        hmm[i] = weights.hmumu[i] * w;
        hml[i] = weights.hmu_ls[i] * w;
        hll[i] = weights.h_ls_ls[i] * w;
        dhmm[i] = weights.dhmumu[i] * w;
        dhml[i] = weights.dhmu_ls[i] * w;
        dhll[i] = weights.dh_ls_ls[i] * w;
    }
    weights.objective_psirow = obj;
    weights.scoremu = smu;
    weights.score_ls = sls;
    weights.dscoremu = dsmu;
    weights.dscore_ls = dsls;
    weights.hmumu = hmm;
    weights.hmu_ls = hml;
    weights.h_ls_ls = hll;
    weights.dhmumu = dhmm;
    weights.dhmu_ls = dhml;
    weights.dh_ls_ls = dhll;
}

/// HT mask for `GaussianJointPsiSecondWeights`. Same semantics as
/// `apply_ht_mask_first`: each per-row contribution is scaled by 1/π_i and
/// non-sampled rows are zeroed. Consumed row-linearly by
/// `gaussian_joint_psisecondhessian_fromweights` and the `score_psi_psi`
/// `fast_atv(_, d2score_*)` reductions.
pub(crate) fn apply_ht_mask_second(
    weights: &mut GaussianJointPsiSecondWeights,
    rows: &[crate::outer_subsample::WeightedOuterRow],
) {
    let n = weights.objective_psi_psirow.len();
    let mut obj = Array1::<f64>::zeros(n);
    let mut d2smu = Array1::<f64>::zeros(n);
    let mut d2sls = Array1::<f64>::zeros(n);
    let mut d2hmm = Array1::<f64>::zeros(n);
    let mut d2hml = Array1::<f64>::zeros(n);
    let mut d2hll = Array1::<f64>::zeros(n);
    for r in rows {
        let i = r.index;
        let w = r.weight;
        obj[i] = weights.objective_psi_psirow[i] * w;
        d2smu[i] = weights.d2scoremu[i] * w;
        d2sls[i] = weights.d2score_ls[i] * w;
        d2hmm[i] = weights.d2hmumu[i] * w;
        d2hml[i] = weights.d2hmu_ls[i] * w;
        d2hll[i] = weights.d2h_ls_ls[i] * w;
    }
    weights.objective_psi_psirow = obj;
    weights.d2scoremu = d2smu;
    weights.d2score_ls = d2sls;
    weights.d2hmumu = d2hmm;
    weights.d2hmu_ls = d2hml;
    weights.d2h_ls_ls = d2hll;
}

/// HT mask for `GaussianJointPsiMixedDriftWeights`. Same semantics as the
/// other `apply_ht_mask_*` helpers; consumed row-linearly by
/// `gaussian_joint_psi_mixedhessian_drift_fromweights`.
pub(crate) fn apply_ht_mask_mixed(
    weights: &mut GaussianJointPsiMixedDriftWeights,
    rows: &[crate::outer_subsample::WeightedOuterRow],
) {
    let n = weights.dhmumu_u.len();
    let mut dhmm_u = Array1::<f64>::zeros(n);
    let mut dhml_u = Array1::<f64>::zeros(n);
    let mut dhll_u = Array1::<f64>::zeros(n);
    let mut d2hmm = Array1::<f64>::zeros(n);
    let mut d2hml = Array1::<f64>::zeros(n);
    let mut d2hll = Array1::<f64>::zeros(n);
    for r in rows {
        let i = r.index;
        let w = r.weight;
        dhmm_u[i] = weights.dhmumu_u[i] * w;
        dhml_u[i] = weights.dhmu_ls_u[i] * w;
        dhll_u[i] = weights.dh_ls_ls_u[i] * w;
        d2hmm[i] = weights.d2hmumu[i] * w;
        d2hml[i] = weights.d2hmu_ls[i] * w;
        d2hll[i] = weights.d2h_ls_ls[i] * w;
    }
    weights.dhmumu_u = dhmm_u;
    weights.dhmu_ls_u = dhml_u;
    weights.dh_ls_ls_u = dhll_u;
    weights.d2hmumu = d2hmm;
    weights.d2hmu_ls = d2hml;
    weights.d2h_ls_ls = d2hll;
}

pub(crate) fn gaussian_jointrow_scalars(
    y: &Array1<f64>,
    etamu: &Array1<f64>,
    eta_ls: &Array1<f64>,
    weights: &Array1<f64>,
) -> Result<GaussianJointRowScalars, String> {
    let nobs = y.len();
    if etamu.len() != nobs || eta_ls.len() != nobs || weights.len() != nobs {
        return Err(GamlssError::DimensionMismatch {
            reason: "Gaussian joint row scalar input size mismatch".to_string(),
        }
        .into());
    }
    let mut obs_weight = Array1::<f64>::uninit(nobs);
    let mut w = Array1::<f64>::uninit(nobs);
    let mut m = Array1::<f64>::uninit(nobs);
    let mut n = Array1::<f64>::uninit(nobs);
    let mut kappa = Array1::<f64>::uninit(nobs);
    let mut kappa_prime = Array1::<f64>::uninit(nobs);
    let mut kappa_dprime = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let jet = crate::sigma_link::logb_sigma_jet1_scalar(eta_ls[i]);
        let s = jet.sigma;
        // κ = exp(η)/(b + exp(η)). Use the direct exp(η)/σ form
        // when finite — it preserves the precision of exp(η) at very negative
        // η (where 1 − b/σ catastrophically cancels because b/σ → 1). The
        // η → +∞ branch returns 1 cleanly without hitting ∞/∞ NaN.
        let ki = logb_dlog_sigma_deta(s, jet.d1);
        let kp = ki * (1.0 - ki);
        let kdp = kp * (1.0 - 2.0 * ki);
        let wi = weights[i] / (s * s);
        let ri = y[i] - etamu[i];
        obs_weight[i].write(weights[i]);
        w[i].write(wi);
        m[i].write(ri * wi);
        n[i].write(ri * ri * wi);
        kappa[i].write(ki);
        kappa_prime[i].write(kp);
        kappa_dprime[i].write(kdp);
    }
    // SAFETY: every `MaybeUninit` slot in each of these arrays was written
    // exactly once in the `for i in 0..nobs` loop above; no slot is read,
    // moved, or dropped before this point.
    let (obs_weight, w, m, n, kappa, kappa_prime, kappa_dprime) = unsafe {
        (
            obs_weight.assume_init(),
            w.assume_init(),
            m.assume_init(),
            n.assume_init(),
            kappa.assume_init(),
            kappa_prime.assume_init(),
            kappa_dprime.assume_init(),
        )
    };
    Ok(GaussianJointRowScalars {
        obs_weight,
        w,
        m,
        n,
        kappa,
        kappa_prime,
        kappa_dprime,
    })
}

pub(crate) fn gaussian_joint_first_directionalweights(
    scalars: &GaussianJointRowScalars,
    dotmu: &Array1<f64>,
    dot_eta: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let nobs = scalars.w.len();
    let mut w_u = Array1::<f64>::uninit(nobs);
    let mut c_u = Array1::<f64>::uninit(nobs);
    let mut d_u = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let wi = scalars.w[i];
        let mi = scalars.m[i];
        let ni = scalars.n[i];
        let ki = scalars.kappa[i];
        let kpi = scalars.kappa_prime[i];
        let kdpi = scalars.kappa_dprime[i];
        let ai = scalars.obs_weight[i];
        let dm = dotmu[i];
        let de = dot_eta[i];
        // κ-scaled log-sigma direction.
        let sde = ki * de;
        w_u[i].write(-2.0 * wi * sde);
        // + 2·κ'·m·de: dκ/dη chain-rule from σ = b + e^η.
        c_u[i].write(ki * (-2.0 * wi * dm - 4.0 * mi * sde) + 2.0 * mi * kpi * de);
        // Directional derivative of the OBSERVED h_ll = κ'(a−n) + 2κ²n:
        //   ∂h/∂η_μ = −2m(2κ²−κ'),
        //   ∂h/∂η_ls = κ''(a−n) + 6κκ'n − 4κ³n.
        let a_coef = 2.0 * ki * ki - kpi;
        d_u[i].write(
            -2.0 * mi * a_coef * dm
                + (kdpi * (ai - ni) + (6.0 * ki * kpi - 4.0 * ki * ki * ki) * ni) * de,
        );
    }
    // SAFETY: every slot of `w_u`, `c_u`, `d_u` was written exactly once
    // inside the loop above (one `.write(...)` per index per array).
    let (w_u, c_u, d_u) = unsafe { (w_u.assume_init(), c_u.assume_init(), d_u.assume_init()) };
    (w_u, c_u, d_u)
}

pub(crate) fn gaussian_jointsecond_directionalweights(
    scalars: &GaussianJointRowScalars,
    dotmu_u: &Array1<f64>,
    dot_eta_u: &Array1<f64>,
    dotmuv: &Array1<f64>,
    dot_etav: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let nobs = scalars.w.len();
    let mut w_uv = Array1::<f64>::uninit(nobs);
    let mut c_uv = Array1::<f64>::uninit(nobs);
    let mut d_uv = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let wi = scalars.w[i];
        let mi = scalars.m[i];
        let ki = scalars.kappa[i];
        let kpi = scalars.kappa_prime[i];
        let kdpi = scalars.kappa_dprime[i];
        let ai = scalars.obs_weight[i];
        let dmu = dotmu_u[i];
        let dmv = dotmuv[i];
        let deu = dot_eta_u[i];
        let dev = dot_etav[i];
        // κ-scaled log-sigma directions.
        let sdeu = ki * deu;
        let sdev = ki * dev;
        let de_sym = dmu * dev + dmv * deu;
        let de_eta = deu * dev;
        // − 2·κ'·w·deu·dev: ∂²w/∂η² = 4wκ² − 2wκ'.
        w_uv[i].write(4.0 * wi * sdeu * sdev - 2.0 * wi * kpi * de_eta);
        // − 2·κ'·w·sym + 2·m·(κ''−6·κ·κ')·deu·dev from d²(2mκ).
        c_uv[i].write(
            ki * (4.0 * wi * (dmu * sdev + dmv * sdeu) + 8.0 * mi * sdeu * sdev)
                - 2.0 * wi * kpi * de_sym
                + 2.0 * mi * (kdpi - 6.0 * ki * kpi) * de_eta,
        );
        // d²/du dv of the OBSERVED h_ll = κ'(a−n) + 2κ²n (β-directions are
        // linear, so no direction-curvature terms). With A = 2κ²−κ',
        // E = 6κκ'−4κ³−κ'', and the logb-link identity κ''' = κ''(1−2κ)−2κ'²:
        //   ∂²h/∂η_μ²      = 2wA,
        //   ∂²h/∂η_μ∂η_ls  = m(8κ³ − 12κκ' + 2κ'')   (≡ ∂²h_μls/∂η_ls², ∂³ℓ symmetry),
        //   ∂²h/∂η_ls²     = κ'''a − 2κnE + n(6κ'² + 6κκ'' − 12κ²κ' − κ''').
        let ni = scalars.n[i];
        let a_coef = 2.0 * ki * ki - kpi;
        let e_coef = 6.0 * ki * kpi - 4.0 * ki * ki * ki - kdpi;
        let ktp = kdpi * (1.0 - 2.0 * ki) - 2.0 * kpi * kpi;
        d_uv[i].write(
            2.0 * wi * a_coef * (dmu * dmv)
                + mi * (8.0 * ki * ki * ki - 12.0 * ki * kpi + 2.0 * kdpi) * de_sym
                + (ktp * ai - 2.0 * ki * ni * e_coef
                    + ni * (6.0 * kpi * kpi + 6.0 * ki * kdpi
                        - 12.0 * ki * ki * kpi
                        - ktp))
                    * de_eta,
        );
    }
    // SAFETY: every slot of `w_uv`, `c_uv`, `d_uv` was written exactly once
    // inside the loop above.
    let (w_uv, c_uv, d_uv) =
        unsafe { (w_uv.assume_init(), c_uv.assume_init(), d_uv.assume_init()) };
    (w_uv, c_uv, d_uv)
}

pub(crate) fn gaussian_joint_psi_firstweights(
    scalars: &GaussianJointRowScalars,
    mu_a: &Array1<f64>,
    eta_a: &Array1<f64>,
) -> GaussianJointPsiFirstWeights {
    let nobs = scalars.w.len();
    let mut objective_psirow = Array1::<f64>::uninit(nobs);
    let mut scoremu = Array1::<f64>::uninit(nobs);
    let mut score_ls = Array1::<f64>::uninit(nobs);
    let mut dscoremu = Array1::<f64>::uninit(nobs);
    let mut dscore_ls = Array1::<f64>::uninit(nobs);
    let mut hmumu = Array1::<f64>::uninit(nobs);
    let mut hmu_ls = Array1::<f64>::uninit(nobs);
    let mut h_ls_ls = Array1::<f64>::uninit(nobs);
    let mut dhmumu = Array1::<f64>::uninit(nobs);
    let mut dhmu_ls = Array1::<f64>::uninit(nobs);
    let mut dh_ls_ls = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let mi = scalars.m[i];
        let ni = scalars.n[i];
        let ki = scalars.kappa[i];
        let kpi = scalars.kappa_prime[i];
        let ai = scalars.obs_weight[i];
        let ma = mu_a[i];
        let ea = eta_a[i];
        // κ-scaled log-sigma direction.
        let sea = ki * ea;
        let smu = -mi;
        let sls = ki * (ai - ni);
        let wi = scalars.w[i];
        scoremu[i].write(smu);
        score_ls[i].write(sls);
        dscoremu[i].write(wi * ma + 2.0 * mi * sea);
        // + κ'·(a−n)·η̇ chain-rule term (∂[κ(a−n)]/∂η = κ'(a−n) + 2κ²n).
        dscore_ls[i].write(ki * (2.0 * mi * ma + 2.0 * ni * sea) + kpi * (ai - ni) * ea);
        hmumu[i].write(wi);
        // Cross block: Fisher expectation E[H_{μ,ls}] = 2κ·E[m] = 0 (μ ⊥ σ;
        // see exact_newton_joint_hessian_from_designs / #684). The observed
        // 2mκ is mean-zero noise that would inject spurious μ↔σ coupling into
        // the REML determinant via the Schur complement and over-smooth log σ.
        hmu_ls[i].write(0.0);
        // Fisher/expected (log σ, log σ) information: E[H_{ls,ls}] = 2κ²a.
        // The observed curvature 2κ²n + κ'(a−n) collapses where the fitted
        // residual is small (n→0), under-counting the scale block's EDF and
        // letting REML over-smooth the scale predictor toward a flat constant
        // (#566). Using E[n]=a (true model) gives the residual-free expected
        // information 2κ²a, exactly as gamlss/mgcv gaulss Fisher-score the
        // scale channel and as the diagonal PIRLS kernel already does
        // (gaussian_diagonal_row_kernel: 2·obs_weight·κ²). The score
        // (score_ls/dscore_ls/d2score_ls) stays the exact observed gradient so
        // the joint Newton still converges to the true MLE stationary point;
        // only the (ls,ls) curvature feeding the REML determinant/EDF is the
        // expectation.
        h_ls_ls[i].write(2.0 * ki * ki * ai);
        dhmumu[i].write(-2.0 * wi * sea);
        // Cross block is Fisher 0 (μ ⊥ σ; #684), so its directional derivative
        // is identically 0.
        dhmu_ls[i].write(0.0);
        // Directional derivative of E[H_{ls,ls}]=2κ²a along (μ̇,η̇): no μ
        // dependence; ∂(2κ²a)/∂η = 4κκ'a, so dh_ls_ls = 4κκ'a·η̇.
        dh_ls_ls[i].write(4.0 * ki * kpi * ai * ea);
        objective_psirow[i].write(smu * ma + sls * ea);
    }
    // SAFETY: every `MaybeUninit` slot in each field array was written
    // exactly once inside the `for i in 0..nobs` loop above.
    unsafe {
        GaussianJointPsiFirstWeights {
            objective_psirow: objective_psirow.assume_init(),
            scoremu: scoremu.assume_init(),
            score_ls: score_ls.assume_init(),
            dscoremu: dscoremu.assume_init(),
            dscore_ls: dscore_ls.assume_init(),
            hmumu: hmumu.assume_init(),
            hmu_ls: hmu_ls.assume_init(),
            h_ls_ls: h_ls_ls.assume_init(),
            dhmumu: dhmumu.assume_init(),
            dhmu_ls: dhmu_ls.assume_init(),
            dh_ls_ls: dh_ls_ls.assume_init(),
        }
    }
}

pub(crate) fn gaussian_joint_psisecondweights(
    scalars: &GaussianJointRowScalars,
    mu_a: &Array1<f64>,
    eta_a: &Array1<f64>,
    mu_b: &Array1<f64>,
    eta_b: &Array1<f64>,
    mu_ab: &Array1<f64>,
    eta_ab: &Array1<f64>,
) -> GaussianJointPsiSecondWeights {
    let nobs = scalars.w.len();
    let mut objective_psi_psirow = Array1::<f64>::uninit(nobs);
    let mut d2scoremu = Array1::<f64>::uninit(nobs);
    let mut d2score_ls = Array1::<f64>::uninit(nobs);
    let mut d2hmumu = Array1::<f64>::uninit(nobs);
    let mut d2hmu_ls = Array1::<f64>::uninit(nobs);
    let mut d2h_ls_ls = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let wi = scalars.w[i];
        let mi = scalars.m[i];
        let ni = scalars.n[i];
        let ki = scalars.kappa[i];
        let kpi = scalars.kappa_prime[i];
        let kdpi = scalars.kappa_dprime[i];
        let ai = scalars.obs_weight[i];
        let amn = ai - ni;
        let ma = mu_a[i];
        let mb = mu_b[i];
        let mab = mu_ab[i];
        let ea = eta_a[i];
        let eb = eta_b[i];
        let eab = eta_ab[i];
        // κ-scaled log-sigma directions.
        let sea = ki * ea;
        let seb = ki * eb;
        let seab = ki * eab;
        let cross = ma * seb + mb * sea;
        // Bare-η symmetric form (no κ): needed for κ' chain-rule terms.
        let cross_eta = ma * eb + mb * ea;
        let sea_seb = sea * seb;
        let ea_eb = ea * eb;
        let ma_mb = ma * mb;
        // + κ'·(a−n)·ea·eb: dκ/dη chain-rule contribution from σ = b + e^η.
        objective_psi_psirow[i].write(
            wi * ma_mb + 2.0 * mi * cross + 2.0 * ni * sea_seb - mi * mab
                + ki * amn * eab
                + kpi * amn * ea_eb,
        );
        // + 2·m·κ'·ea·eb: ∂²(−m)/∂η² = −4mκ² + 2mκ'.
        d2scoremu[i].write(
            wi * mab - 2.0 * wi * cross - 4.0 * mi * sea_seb
                + 2.0 * mi * seab
                + 2.0 * mi * kpi * ea_eb,
        );
        // + 2·κ'·m·sym(μ_a η_b) + (κ''(a−n)+6κκ'n)·ea·eb + κ'(a−n)·eab.
        d2score_ls[i].write(
            ki * (-2.0 * wi * ma_mb - 4.0 * mi * cross - 4.0 * ni * sea_seb
                + 2.0 * mi * mab
                + 2.0 * ni * seab)
                + 2.0 * mi * kpi * cross_eta
                + (kdpi * amn + 6.0 * ki * kpi * ni) * ea_eb
                + kpi * amn * eab,
        );
        // − 2·κ'·w·ea·eb: ∂²w/∂η² = 4wκ² − 2wκ'.
        d2hmumu[i].write(4.0 * wi * sea_seb - 2.0 * wi * seab - 2.0 * wi * kpi * ea_eb);
        // Cross block is Fisher 0 (μ ⊥ σ; #684), so its second directional
        // derivative is identically 0.
        d2hmu_ls[i].write(0.0);
        // d²/dψ_a dψ_b of the Fisher (ls,ls) information E[H_{ls,ls}]=2κ²a (#566).
        // No μ dependence; ∂(2κ²a)/∂η=4κκ'a and ∂(4κκ'a)/∂η=4a(κ'²+κκ'')a, so
        // the second directional derivative is 4a(κ'²+κκ'')·ea·eb + 4aκκ'·eab.
        d2h_ls_ls[i].write(4.0 * ai * (kpi * kpi + ki * kdpi) * ea_eb + 4.0 * ai * ki * kpi * eab);
    }
    // SAFETY: every `MaybeUninit` slot in each field array was written
    // exactly once inside the `for i in 0..nobs` loop above.
    unsafe {
        GaussianJointPsiSecondWeights {
            objective_psi_psirow: objective_psi_psirow.assume_init(),
            d2scoremu: d2scoremu.assume_init(),
            d2score_ls: d2score_ls.assume_init(),
            d2hmumu: d2hmumu.assume_init(),
            d2hmu_ls: d2hmu_ls.assume_init(),
            d2h_ls_ls: d2h_ls_ls.assume_init(),
        }
    }
}

pub(crate) fn gaussian_joint_psi_mixed_driftweights(
    scalars: &GaussianJointRowScalars,
    // Only the log-σ–channel directions enter the surviving (μ,μ) and (ls,ls)
    // Fisher blocks; the μ-channel drift directions fed the observed cross
    // block, which is now Fisher 0 (μ ⊥ σ; #684) and no longer assembled.
    dot_eta: &Array1<f64>,
    eta_a: &Array1<f64>,
    dot_eta_a: &Array1<f64>,
) -> GaussianJointPsiMixedDriftWeights {
    let nobs = scalars.w.len();
    let mut dhmumu_u = Array1::<f64>::uninit(nobs);
    let mut dhmu_ls_u = Array1::<f64>::uninit(nobs);
    let mut dh_ls_ls_u = Array1::<f64>::uninit(nobs);
    let mut d2hmumu = Array1::<f64>::uninit(nobs);
    let mut d2hmu_ls = Array1::<f64>::uninit(nobs);
    let mut d2h_ls_ls = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let wi = scalars.w[i];
        let ki = scalars.kappa[i];
        let kpi = scalars.kappa_prime[i];
        let kdpi = scalars.kappa_dprime[i];
        let ai = scalars.obs_weight[i];
        let de = dot_eta[i];
        let ea = eta_a[i];
        let dea = dot_eta_a[i];
        // κ-scaled log-sigma directions.
        let sde = ki * de;
        let sea = ki * ea;
        let sdea = ki * dea;
        let de_ea = de * ea;
        // First directional derivative of Hessian blocks (== Helper A).
        dhmumu_u[i].write(-2.0 * wi * sde);
        // Cross block is Fisher 0 (μ ⊥ σ; #684); its first directional and
        // second mixed directional derivatives are identically 0. The
        // observed-cross drift inputs (m, dotmu, μ_a, dotmu_a) are therefore
        // not read here.
        dhmu_ls_u[i].write(0.0);
        // Directional derivative of Fisher E[H_{ls,ls}]=2κ²a along (dm,de):
        // no μ dependence, ∂(2κ²a)/∂η=4κκ'a ⇒ 4κκ'a·de (#566).
        dh_ls_ls_u[i].write(4.0 * ki * kpi * ai * de);
        // − 2·κ'·w·de·ea: ∂²w/∂η² = 4wκ² − 2wκ'.
        d2hmumu[i].write(4.0 * wi * sde * sea - 2.0 * wi * sdea - 2.0 * wi * kpi * de_ea);
        d2hmu_ls[i].write(0.0);
        // d²/(drift × ψ) of Fisher E[H_{ls,ls}]=2κ²a: 4a(κ'²+κκ'')·de·ea +
        // 4aκκ'·dea (drift direction de, ψ direction ea, mixed dea) (#566).
        d2h_ls_ls[i].write(4.0 * ai * (kpi * kpi + ki * kdpi) * de_ea + 4.0 * ai * ki * kpi * dea);
    }
    // SAFETY: every `MaybeUninit` slot in each field array was written
    // exactly once inside the `for i in 0..nobs` loop above.
    unsafe {
        GaussianJointPsiMixedDriftWeights {
            dhmumu_u: dhmumu_u.assume_init(),
            dhmu_ls_u: dhmu_ls_u.assume_init(),
            dh_ls_ls_u: dh_ls_ls_u.assume_init(),
            d2hmumu: d2hmumu.assume_init(),
            d2hmu_ls: d2hmu_ls.assume_init(),
            d2h_ls_ls: d2h_ls_ls.assume_init(),
        }
    }
}

/// Canonical Gaussian location-scale OBSERVED joint-Hessian row coefficients
/// `(mm, ml, ll)` — the SINGLE source of truth for this curvature, shared by
/// every representation that assembles the value Hessian (the dense
/// `exact_newton_joint_hessian_from_designs` and the matrix-free
/// `GaussianLocationScaleHessianWorkspace`). Exact second derivatives of the
/// row NLL (`r = y−μ`, `w = a/σ²`, `m = rw`, `n = r²w`, `κ = dlogσ/dη`):
///   `mm = ∂²ℓ/∂η_μ²      = w`             (observed ≡ expected — exact),
///   `ml = ∂²ℓ/∂η_μ∂η_ls  = 2κm`           (expectation 0 at the truth),
///   `ll = ∂²ℓ/∂η_ls²     = κ′(a−n) + 2κ²n` (expectation 2κ²a).
/// The LAML criterion `−½log|H+S|` requires the OBSERVED penalized Hessian at
/// β̂ (Wood–Pya–Säfken 2016): the earlier block-Fisher object (#684/#566)
/// zeroed `ml` and expected `ll`, which drops the cross-block Schur deficit
/// `H_σμ(H_μμ+S_μ)⁻¹H_μσ` and the fitted-residual shrinkage `E[n̂]≈a(1−h_μ)`
/// — both overstate σ-block information and bias λ̂_σ upward on the flat scale
/// surface (#1561: log-σ over-smoothing; same dof genus as #2133). At a
/// true-null/flat σ surface `n→a`, `m→0`, so observed → Fisher and null
/// behavior is unchanged (SPEC: defaults recover the null). Indefiniteness of
/// the observed joint Hessian is handled by the existing #365 modified-Newton
/// reflection on the inner path and the spectral PD-floor on the criterion
/// log-det. Routing every path through this one constructor keeps the #684
/// cross-block drift structurally impossible.
pub(crate) fn gaussian_locscale_observed_joint_row_coeffs(
    rows: &GaussianJointRowScalars,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mm = rows.w.clone();
    let ml = 2.0 * &rows.kappa * &rows.m;
    let ll = &rows.kappa_prime * (&rows.obs_weight - &rows.n)
        + 2.0 * &rows.kappa * &rows.kappa * &rows.n;
    (mm, ml, ll)
}

pub(crate) fn gaussian_joint_hessian_from_designs(
    xmu: &DenseOrOperator<'_>,
    x_ls: &DenseOrOperator<'_>,
    hmumu_coeff: &Array1<f64>,
    hmu_ls_coeff: &Array1<f64>,
    h_ls_ls_coeff: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    if xmu.nrows() != hmumu_coeff.len()
        || xmu.nrows() != hmu_ls_coeff.len()
        || xmu.nrows() != h_ls_ls_coeff.len()
        || x_ls.nrows() != xmu.nrows()
    {
        return Err(GamlssError::DimensionMismatch { reason: format!(
            "gaussian_joint_hessian_from_designs dimension mismatch: xmu {}x{}, x_ls {}x{}, coeffs {}/{}/{}",
            xmu.nrows(),
            xmu.ncols(),
            x_ls.nrows(),
            x_ls.ncols(),
            hmumu_coeff.len(),
            hmu_ls_coeff.len(),
            h_ls_ls_coeff.len()
        ) }.into());
    }

    let n = xmu.nrows();
    let pmu = xmu.ncols();
    let p_ls = x_ls.ncols();
    let total = pmu + p_ls;
    let mut out = Array2::<f64>::zeros((total, total));
    for rows in exact_design_row_chunks(n, pmu.max(p_ls)) {
        let xmu_chunk = xmu.row_chunk(rows.clone())?;
        let xls_chunk = x_ls.row_chunk(rows.clone())?;
        let hmumu = hmumu_coeff.slice(s![rows.clone()]);
        let hmu_ls = hmu_ls_coeff.slice(s![rows.clone()]);
        let h_ls_ls = h_ls_ls_coeff.slice(s![rows.clone()]);
        let chunk_hessian =
            fast_joint_hessian_2x2(&xmu_chunk, &xls_chunk, &hmumu, &hmu_ls, &h_ls_ls);
        out += &chunk_hessian;
    }
    Ok(out)
}

pub(crate) fn gaussian_joint_psihessian_fromweights(
    xmu: &Array2<f64>,
    x_ls: &Array2<f64>,
    xmu_psi: CustomFamilyPsiLinearMapRef<'_>,
    x_ls_psi: CustomFamilyPsiLinearMapRef<'_>,
    weights: &GaussianJointPsiFirstWeights,
) -> Result<Array2<f64>, String> {
    // For the symmetric blocks (hmumu, h_ls_ls), the pair
    //   X_psi^T D X  and  X^T D X_psi
    // are transposes of each other, so compute one and add its transpose.
    let a_mu = weighted_crossprod_psi_maps(
        xmu_psi,
        weights.hmumu.view(),
        CustomFamilyPsiLinearMapRef::Dense(xmu),
    )?;
    let hmumu = &a_mu + &a_mu.t() + &xt_diag_x_dense(xmu, &weights.dhmumu)?;
    let hmu_ls = weighted_crossprod_psi_maps(
        xmu_psi,
        weights.hmu_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )? + &weighted_crossprod_psi_maps(
        CustomFamilyPsiLinearMapRef::Dense(xmu),
        weights.hmu_ls.view(),
        x_ls_psi,
    )? + &xt_diag_y_dense(xmu, &weights.dhmu_ls, x_ls)?;
    let a_ls = weighted_crossprod_psi_maps(
        x_ls_psi,
        weights.h_ls_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )?;
    let h_ls_ls = &a_ls + &a_ls.t() + &xt_diag_x_dense(x_ls, &weights.dh_ls_ls)?;
    Ok(gaussian_pack_joint_symmetrichessian(
        &hmumu, &hmu_ls, &h_ls_ls,
    ))
}

pub(crate) fn build_two_block_custom_family_joint_psi_operator_from_actions(
    left_action: Option<CustomFamilyPsiDesignAction>,
    right_action: Option<CustomFamilyPsiDesignAction>,
    left_range: std::ops::Range<usize>,
    right_range: std::ops::Range<usize>,
    left_design: &Array2<f64>,
    right_design: &Array2<f64>,
    left_weights: &Array1<f64>,
    cross_weights: &Array1<f64>,
    right_weights: &Array1<f64>,
    left_drift_weights: &Array1<f64>,
    cross_drift_weights: &Array1<f64>,
    right_drift_weights: &Array1<f64>,
) -> Result<Option<std::sync::Arc<dyn gam_problem::HyperOperator>>, String> {
    if left_action.is_none() && right_action.is_none() {
        return Ok(None);
    }

    let total = left_design.ncols() + right_design.ncols();
    let channels = vec![
        CustomFamilyJointDesignChannel::new(left_range, shared_dense_arc(left_design), left_action),
        CustomFamilyJointDesignChannel::new(
            right_range,
            shared_dense_arc(right_design),
            right_action,
        ),
    ];
    let pair_contributions = vec![
        CustomFamilyJointDesignPairContribution::new(
            0,
            0,
            left_weights.clone(),
            left_drift_weights.clone(),
        ),
        CustomFamilyJointDesignPairContribution::new(
            0,
            1,
            cross_weights.clone(),
            cross_drift_weights.clone(),
        ),
        CustomFamilyJointDesignPairContribution::new(
            1,
            0,
            cross_weights.clone(),
            cross_drift_weights.clone(),
        ),
        CustomFamilyJointDesignPairContribution::new(
            1,
            1,
            right_weights.clone(),
            right_drift_weights.clone(),
        ),
    ];

    Ok(Some(std::sync::Arc::new(
        CustomFamilyJointPsiOperator::new(total, channels, pair_contributions),
    )))
}

pub(crate) fn gaussian_joint_psisecondhessian_fromweights(
    xmu: &Array2<f64>,
    x_ls: &Array2<f64>,
    xmu_i: CustomFamilyPsiLinearMapRef<'_>,
    x_ls_i: CustomFamilyPsiLinearMapRef<'_>,
    xmu_j: CustomFamilyPsiLinearMapRef<'_>,
    x_ls_j: CustomFamilyPsiLinearMapRef<'_>,
    xmu_ab: CustomFamilyPsiLinearMapRef<'_>,
    x_ls_ab: CustomFamilyPsiLinearMapRef<'_>,
    weights_i: &GaussianJointPsiFirstWeights,
    weights_j: &GaussianJointPsiFirstWeights,
    secondweights: &GaussianJointPsiSecondWeights,
) -> Result<Array2<f64>, String> {
    // Exploit transpose symmetry: X_a^T D X_b and X_b^T D X_a are transposes.
    // For each such pair in the symmetric blocks (hmumu, h_ls_ls), compute one
    // and add its transpose, halving the number of O(np²) products.
    let a_ab_mu = weighted_crossprod_psi_maps(
        xmu_ab,
        weights_i.hmumu.view(),
        CustomFamilyPsiLinearMapRef::Dense(xmu),
    )?;
    let a_ij_mu = weighted_crossprod_psi_maps(xmu_i, weights_i.hmumu.view(), xmu_j)?;
    let a_iwj_mu = weighted_crossprod_psi_maps(
        xmu_i,
        weights_j.dhmumu.view(),
        CustomFamilyPsiLinearMapRef::Dense(xmu),
    )?;
    let a_jwi_mu = weighted_crossprod_psi_maps(
        xmu_j,
        weights_i.dhmumu.view(),
        CustomFamilyPsiLinearMapRef::Dense(xmu),
    )?;
    let hmumu = &a_ab_mu
        + &a_ab_mu.t()
        + &a_ij_mu
        + a_ij_mu.t()
        + &a_iwj_mu
        + a_iwj_mu.t()
        + &a_jwi_mu
        + a_jwi_mu.t()
        + &xt_diag_x_dense(xmu, &secondweights.d2hmumu)?;
    let hmu_ls = weighted_crossprod_psi_maps(
        xmu_ab,
        weights_i.hmu_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )? + &weighted_crossprod_psi_maps(xmu_i, weights_i.hmu_ls.view(), x_ls_j)?
        + &weighted_crossprod_psi_maps(xmu_j, weights_i.hmu_ls.view(), x_ls_i)?
        + &weighted_crossprod_psi_maps(
            xmu_i,
            weights_j.dhmu_ls.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?
        + &weighted_crossprod_psi_maps(
            xmu_j,
            weights_i.dhmu_ls.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?
        + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            weights_i.dhmu_ls.view(),
            x_ls_j,
        )?
        + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            weights_j.dhmu_ls.view(),
            x_ls_i,
        )?
        + &xt_diag_y_dense(xmu, &secondweights.d2hmu_ls, x_ls)?
        + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            weights_i.hmu_ls.view(),
            x_ls_ab,
        )?;
    let a_ab_ls = weighted_crossprod_psi_maps(
        x_ls_ab,
        weights_i.h_ls_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )?;
    let a_ij_ls = weighted_crossprod_psi_maps(x_ls_i, weights_i.h_ls_ls.view(), x_ls_j)?;
    let a_iwj_ls = weighted_crossprod_psi_maps(
        x_ls_i,
        weights_j.dh_ls_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )?;
    let a_jwi_ls = weighted_crossprod_psi_maps(
        x_ls_j,
        weights_i.dh_ls_ls.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )?;
    let h_ls_ls = &a_ab_ls
        + &a_ab_ls.t()
        + &a_ij_ls
        + a_ij_ls.t()
        + &a_iwj_ls
        + a_iwj_ls.t()
        + &a_jwi_ls
        + a_jwi_ls.t()
        + &xt_diag_x_dense(x_ls, &secondweights.d2h_ls_ls)?;
    Ok(gaussian_pack_joint_symmetrichessian(
        &hmumu, &hmu_ls, &h_ls_ls,
    ))
}

pub(crate) fn gaussian_joint_psi_mixedhessian_drift_fromweights(
    xmu: &Array2<f64>,
    x_ls: &Array2<f64>,
    xmu_psi: CustomFamilyPsiLinearMapRef<'_>,
    x_ls_psi: CustomFamilyPsiLinearMapRef<'_>,
    mixedweights: &GaussianJointPsiMixedDriftWeights,
) -> Result<Array2<f64>, String> {
    let a_mu = weighted_crossprod_psi_maps(
        xmu_psi,
        mixedweights.dhmumu_u.view(),
        CustomFamilyPsiLinearMapRef::Dense(xmu),
    )?;
    let hmumu = &a_mu + &a_mu.t() + &xt_diag_x_dense(xmu, &mixedweights.d2hmumu)?;
    let hmu_ls = weighted_crossprod_psi_maps(
        xmu_psi,
        mixedweights.dhmu_ls_u.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )? + &weighted_crossprod_psi_maps(
        CustomFamilyPsiLinearMapRef::Dense(xmu),
        mixedweights.dhmu_ls_u.view(),
        x_ls_psi,
    )? + &xt_diag_y_dense(xmu, &mixedweights.d2hmu_ls, x_ls)?;
    let a_ls = weighted_crossprod_psi_maps(
        x_ls_psi,
        mixedweights.dh_ls_ls_u.view(),
        CustomFamilyPsiLinearMapRef::Dense(x_ls),
    )?;
    let h_ls_ls = &a_ls + &a_ls.t() + &xt_diag_x_dense(x_ls, &mixedweights.d2h_ls_ls)?;
    Ok(gaussian_pack_joint_symmetrichessian(
        &hmumu, &hmu_ls, &h_ls_ls,
    ))
}

#[inline]
pub(crate) fn exp_sigma_derivs_up_to_fourth_array(
    eta: ArrayView1<'_, f64>,
) -> (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let n = eta.len();
    let tuples: Vec<(f64, f64, f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| exp_sigma_derivs_up_to_fourth_scalar(eta[i]))
        .collect();
    let mut sigma = Array1::<f64>::zeros(n);
    let mut d1 = Array1::<f64>::zeros(n);
    let mut d2 = Array1::<f64>::zeros(n);
    let mut d3 = Array1::<f64>::zeros(n);
    let mut d4 = Array1::<f64>::zeros(n);
    for (i, (s_i, d1_i, d2_i, d3_i, d4_i)) in tuples.into_iter().enumerate() {
        sigma[i] = s_i;
        d1[i] = d1_i;
        d2[i] = d2_i;
        d3[i] = d3_i;
        d4[i] = d4_i;
    }
    (sigma, d1, d2, d3, d4)
}

#[cfg(test)]
mod observed_single_source_oracle_tests {
    //! #932 doctrine oracle for the Gaussian location-scale OBSERVED joint
    //! Hessian tower (#1561 cutover from block-Fisher).
    //!
    //! The production joint Hessian is the OBSERVED information, built from
    //! hand-written closed-form row coefficients
    //! (`gaussian_locscale_observed_joint_row_coeffs`: `mm = w`, `ml = 2κm`,
    //! `ll = κ'(a−n)+2κ²n`) and their β-directional drifts
    //! (`gaussian_joint_first_directionalweights` /
    //! `gaussian_jointsecond_directionalweights`). Those hand forms are FAST and
    //! STAY in production — but, like every other #932 family, they must be
    //! pinned bit-for-bit to a MECHANICAL single source so a future edit that
    //! drops or mis-weights a term (the #736/#947 bug genus) is caught here
    //! rather than in a silently wrong outer Hessian.
    //!
    //! MECHANICAL SOURCE (no hand math reused):
    //!  * The per-row negative log-likelihood is `ρ(μ,η)=−ℓ(μ,η)`, evaluated by
    //!    the production row kernel `gaussian_diagonal_row_kernel`.
    //!  * Its OBSERVED 2×2 Hessian in `(μ,η_ls)` is taken by central finite
    //!    differences of that kernel — a derivative the test computes from the
    //!    likelihood ALONE, never from the hand coefficients under test.
    //!  * The directional / second-directional drifts are pinned to central
    //!    finite differences of the production observed coefficients along the
    //!    same β-direction.

    use super::*;
    use ndarray::array;

    /// Row negative log-likelihood from the production kernel (likelihood only,
    /// no curvature coefficients involved).
    fn row_nll(y: f64, mu: f64, eta_ls: f64, a: f64) -> f64 {
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        -gaussian_diagonal_row_kernel(y, mu, eta_ls, a, ln2pi).log_likelihood
    }

    /// Observed 2×2 Hessian of the row NLL in `(μ, η_ls)` by central FD.
    /// Returns `(H_μμ, H_{μ,ls}, H_{ls,ls})`.
    fn observed_hessian_fd(y: f64, mu: f64, eta_ls: f64, a: f64, h: f64) -> (f64, f64, f64) {
        let hmm = (row_nll(y, mu + h, eta_ls, a) - 2.0 * row_nll(y, mu, eta_ls, a)
            + row_nll(y, mu - h, eta_ls, a))
            / (h * h);
        let hll = (row_nll(y, mu, eta_ls + h, a) - 2.0 * row_nll(y, mu, eta_ls, a)
            + row_nll(y, mu, eta_ls - h, a))
            / (h * h);
        let hml = (row_nll(y, mu + h, eta_ls + h, a)
            - row_nll(y, mu + h, eta_ls - h, a)
            - row_nll(y, mu - h, eta_ls + h, a)
            + row_nll(y, mu - h, eta_ls - h, a))
            / (4.0 * h * h);
        (hmm, hml, hll)
    }

    /// Production observed joint-Hessian coefficients for a single row.
    fn production_observed_row(y: f64, mu: f64, eta_ls: f64, a: f64) -> (f64, f64, f64) {
        let rows = gaussian_jointrow_scalars(&array![y], &array![mu], &array![eta_ls], &array![a])
            .expect("row scalars");
        let (mm, ml, ll) = gaussian_locscale_observed_joint_row_coeffs(&rows);
        (mm[0], ml[0], ll[0])
    }

    #[test]
    fn observed_joint_row_coeffs_match_likelihood_fd_single_source() {
        // Residual-dependent cases: y ≠ μ so the cross and (ls,ls) observed
        // weights are material (not the Fisher limit m→0, n→a).
        let cases = [
            (0.3_f64, -0.4_f64, 1.0_f64, 0.55_f64),
            (-1.2, 0.7, 2.5, -0.9),
            (0.0, 1.5, 0.4, 0.35),
            (2.4, -1.1, 0.8, 1.7),
            (-0.6, 0.2, 3.3, -1.1),
        ];
        let h = 1e-4;
        for &(mu, eta_ls, a, y) in &cases {
            let (mm_hand, ml_hand, ll_hand) = production_observed_row(y, mu, eta_ls, a);
            let (mm_fd, ml_fd, ll_fd) = observed_hessian_fd(y, mu, eta_ls, a, h);
            assert!(
                (mm_hand - mm_fd).abs() <= 1e-5 * mm_hand.abs().max(1.0),
                "H_μμ observed μ={mu} η={eta_ls} y={y}: hand={mm_hand} fd={mm_fd}"
            );
            assert!(
                (ml_hand - ml_fd).abs() <= 1e-5 * ml_hand.abs().max(1.0),
                "H_μls observed μ={mu} η={eta_ls} y={y}: hand={ml_hand} fd={ml_fd}"
            );
            assert!(
                (ll_hand - ll_fd).abs() <= 1e-5 * ll_hand.abs().max(1.0),
                "H_lsls observed μ={mu} η={eta_ls} y={y}: hand={ll_hand} fd={ll_fd}"
            );
        }
    }

    #[test]
    fn first_directional_weights_match_observed_finite_difference() {
        // Pin (w_u, c_u, d_u) to a central FD of the observed coefficients
        // along the β-direction (μ += t·ξμ, η_ls += t·ξls).
        let cases = [
            (0.3_f64, -0.4_f64, 1.0_f64, 0.5_f64, -0.7_f64, 0.55_f64),
            (-1.2, 0.7, 2.5, 1.1, 0.3, -0.9),
            (0.0, 1.5, 0.4, -0.2, 0.9, 0.35),
            (2.4, -1.1, 0.8, 0.6, -0.4, 1.7),
        ];
        let t = 1e-6;
        for &(mu, eta_ls, a, xi_mu, xi_ls, y) in &cases {
            let rows = gaussian_jointrow_scalars(&array![y], &array![mu], &array![eta_ls], &array![a])
                .expect("row scalars");
            let (w_u, c_u, d_u) =
                gaussian_joint_first_directionalweights(&rows, &array![xi_mu], &array![xi_ls]);
            let coeffs_at = |m: f64, e: f64| -> (f64, f64, f64) {
                production_observed_row(y, m, e, a)
            };
            let (mmp, mlp, llp) = coeffs_at(mu + t * xi_mu, eta_ls + t * xi_ls);
            let (mmm, mlm, llm) = coeffs_at(mu - t * xi_mu, eta_ls - t * xi_ls);
            let fd_w = (mmp - mmm) / (2.0 * t);
            let fd_c = (mlp - mlm) / (2.0 * t);
            let fd_d = (llp - llm) / (2.0 * t);
            assert!(
                (w_u[0] - fd_w).abs() <= 1e-5 * fd_w.abs().max(1.0),
                "dH_μμ drift μ={mu} η={eta_ls}: hand={} fd={fd_w}",
                w_u[0]
            );
            assert!(
                (c_u[0] - fd_c).abs() <= 1e-5 * fd_c.abs().max(1.0),
                "dH_μls drift μ={mu} η={eta_ls}: hand={} fd={fd_c}",
                c_u[0]
            );
            assert!(
                (d_u[0] - fd_d).abs() <= 1e-5 * fd_d.abs().max(1.0),
                "dH_lsls drift μ={mu} η={eta_ls}: hand={} fd={fd_d}",
                d_u[0]
            );
        }
    }

    #[test]
    fn second_directional_weights_match_first_directional_finite_difference() {
        // Pin (w_uv, c_uv, d_uv) to a central FD of the FIRST-directional
        // observed drifts along the v-direction.
        let cases = [
            (
                0.3_f64, -0.4_f64, 1.0_f64, 0.5_f64, -0.7_f64, 0.8_f64, 0.2_f64, 0.55_f64,
            ),
            (-1.2, 0.7, 2.5, 1.1, 0.3, -0.6, 0.9, -0.9),
            (0.0, 1.5, 0.4, -0.2, 0.9, 0.4, -0.5, 0.35),
        ];
        let t = 1e-5;
        for &(mu, eta_ls, a, xi_mu_u, xi_ls_u, xi_mu_v, xi_ls_v, y) in &cases {
            let rows = gaussian_jointrow_scalars(&array![y], &array![mu], &array![eta_ls], &array![a])
                .expect("row scalars");
            let (w_uv, c_uv, d_uv) = gaussian_jointsecond_directionalweights(
                &rows,
                &array![xi_mu_u],
                &array![xi_ls_u],
                &array![xi_mu_v],
                &array![xi_ls_v],
            );
            let first_at = |m: f64, e: f64| -> (f64, f64, f64) {
                let r = gaussian_jointrow_scalars(&array![y], &array![m], &array![e], &array![a])
                    .expect("row scalars");
                let (w_u, c_u, d_u) =
                    gaussian_joint_first_directionalweights(&r, &array![xi_mu_u], &array![xi_ls_u]);
                (w_u[0], c_u[0], d_u[0])
            };
            let (wp, cp, dp) = first_at(mu + t * xi_mu_v, eta_ls + t * xi_ls_v);
            let (wm, cm, dm) = first_at(mu - t * xi_mu_v, eta_ls - t * xi_ls_v);
            let fd_w = (wp - wm) / (2.0 * t);
            let fd_c = (cp - cm) / (2.0 * t);
            let fd_d = (dp - dm) / (2.0 * t);
            assert!(
                (w_uv[0] - fd_w).abs() <= 1e-4 * fd_w.abs().max(1.0),
                "d²H_μμ drift μ={mu} η={eta_ls}: hand={} fd={fd_w}",
                w_uv[0]
            );
            assert!(
                (c_uv[0] - fd_c).abs() <= 1e-4 * fd_c.abs().max(1.0),
                "d²H_μls drift μ={mu} η={eta_ls}: hand={} fd={fd_c}",
                c_uv[0]
            );
            assert!(
                (d_uv[0] - fd_d).abs() <= 1e-4 * fd_d.abs().max(1.0),
                "d²H_lsls drift μ={mu} η={eta_ls}: hand={} fd={fd_d}",
                d_uv[0]
            );
        }
    }
}
