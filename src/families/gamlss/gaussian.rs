// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

pub struct GaussianLocationScaleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub mu_design: Option<DesignMatrix>,
    pub log_sigma_design: Option<DesignMatrix>,
    /// Resource policy threaded into PsiDesignMap construction (and any other
    /// per-call materialization decision) made during exact-Newton joint psi
    /// derivative evaluation. Defaults to `ResourcePolicy::default_library()`
    /// when the family is built without an explicit policy.
    pub policy: crate::resource::ResourcePolicy,
    /// Cached per-observation row scalars keyed by 6-element fingerprint
    /// (first, mid, last elements of both eta vectors).
    /// Avoids recomputing O(n) scalars K+ times per REML gradient/Hessian evaluation.
    pub cached_row_scalars:
        std::sync::RwLock<Option<(f64, f64, f64, f64, f64, f64, Arc<GaussianJointRowScalars>)>>,
}

impl Clone for GaussianLocationScaleFamily {
    pub(crate) fn clone(&self) -> Self {
        Self {
            y: self.y.clone(),
            weights: self.weights.clone(),
            mu_design: self.mu_design.clone(),
            log_sigma_design: self.log_sigma_design.clone(),
            policy: self.policy.clone(),
            cached_row_scalars: std::sync::RwLock::new(
                self.cached_row_scalars
                    .read()
                    .expect("lock poisoned")
                    .clone(),
            ),
        }
    }
}

pub(crate) struct LocationScaleJointPsiDirection {
    block_idx: usize,
    local_idx: usize,
    x_primary_psi: PsiDesignMap,
    x_ls_psi: PsiDesignMap,
    z_primary_psi: Array1<f64>,
    z_ls_psi: Array1<f64>,
}

pub(crate) struct LocationScaleJointPsiSecondDrifts {
    x_primary_ab_action: Option<CustomFamilyPsiSecondDesignAction>,
    x_ls_ab_action: Option<CustomFamilyPsiSecondDesignAction>,
    x_primary_ab: Option<Array2<f64>>,
    x_ls_ab: Option<Array2<f64>>,
    z_primary_ab: Array1<f64>,
    z_ls_ab: Array1<f64>,
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
    pub(crate) const LABEL: &'static str;

    pub(crate) fn ws_policy(&self) -> &crate::resource::ResourcePolicy;

    pub(crate) fn ws_exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String>;

    pub(crate) fn ws_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
    ) -> Result<Option<Self::Direction>, String>;

    pub(crate) fn ws_psi_second_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &Self::Direction,
        psi_b: &Self::Direction,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
    ) -> Result<ExactNewtonJointPsiSecondOrderTerms, String>;

    pub(crate) fn ws_psi_hessian_directional_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        psi_dir: &Self::Direction,
        d_beta_flat: &Array1<f64>,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
    ) -> Result<Array2<f64>, String>;
}

impl LocationScaleJointPsiFamily for GaussianLocationScaleFamily {
    type Direction = LocationScaleJointPsiDirection;
    pub(crate) const LABEL: &'static str = "GaussianLocationScaleFamily";

    pub(crate) fn ws_policy(&self) -> &crate::resource::ResourcePolicy {
        &self.policy
    }

    pub(crate) fn ws_exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        self.exact_joint_dense_block_designs(specs)
    }

    pub(crate) fn ws_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
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

    pub(crate) fn ws_psi_second_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &LocationScaleJointPsiDirection,
        psi_b: &LocationScaleJointPsiDirection,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
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

    pub(crate) fn ws_psi_hessian_directional_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        psi_dir: &LocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
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
    pub(crate) const LABEL: &'static str = "GaussianLocationScaleWiggleFamily";

    pub(crate) fn ws_policy(&self) -> &crate::resource::ResourcePolicy {
        &self.policy
    }

    pub(crate) fn ws_exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        self.exact_joint_dense_block_designs(specs)
    }

    pub(crate) fn ws_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
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

    pub(crate) fn ws_psi_second_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &LocationScaleJointPsiDirection,
        psi_b: &LocationScaleJointPsiDirection,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        outer_rows: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
    ) -> Result<ExactNewtonJointPsiSecondOrderTerms, String> {
        assert!(outer_rows.map_or(true, |r| r.len() <= isize::MAX as usize));
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

    pub(crate) fn ws_psi_hessian_directional_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        psi_dir: &LocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        design_loc: &Array2<f64>,
        design_scale: &Array2<f64>,
        outer_rows: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
    ) -> Result<Array2<f64>, String> {
        assert!(outer_rows.map_or(true, |r| r.len() <= isize::MAX as usize));
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
    family: F,
    block_states: Vec<ParameterBlockState>,
    derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    design_loc: Arc<Array2<f64>>,
    design_scale: Arc<Array2<f64>>,
    psi_directions: ExactNewtonJointPsiDirectCache<F::Direction>,
    outer_score_subsample: Option<Arc<crate::families::marginal_slope_shared::OuterScoreSubsample>>,
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
        outer_score_subsample: Option<
            Arc<crate::families::marginal_slope_shared::OuterScoreSubsample>,
        >,
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

    pub(crate) fn subsample_rows(
        &self,
    ) -> Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]> {
        self.outer_score_subsample
            .as_ref()
            .map(|s| s.rows.as_ref().as_slice())
    }
}

impl<F> ExactNewtonJointPsiWorkspace for LocationScaleJointPsiWorkspace<F>
where
    F: LocationScaleJointPsiFamily,
{
    pub(crate) fn second_order_terms(
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

    pub(crate) fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<crate::solver::estimate::reml::unified::DriftDerivResult>, String> {
        let Some(dir) = self.psi_direction(psi_index)? else {
            return Ok(None);
        };
        Ok(Some(
            crate::solver::estimate::reml::unified::DriftDerivResult::Dense(
                self.family.ws_psi_hessian_directional_from_parts(
                    &self.block_states,
                    dir.as_ref(),
                    d_beta_flat,
                    self.design_loc.as_ref(),
                    self.design_scale.as_ref(),
                    self.subsample_rows(),
                )?,
            ),
        ))
    }
}

pub(crate) type GaussianLocationScaleExactNewtonJointPsiWorkspace =
    LocationScaleJointPsiWorkspace<GaussianLocationScaleFamily>;

pub(crate) type GaussianLocationScaleWiggleExactNewtonJointPsiWorkspace =
    LocationScaleJointPsiWorkspace<GaussianLocationScaleWiggleFamily>;

#[derive(Clone)]
pub struct GaussianJointRowScalars {
    obs_weight: Array1<f64>,
    w: Array1<f64>,
    m: Array1<f64>,
    n: Array1<f64>,
    /// κ = (dσ/dη_ls)/σ for the active sigma link.
    /// The cross Hessian block H_{μ,ls} carries an overall κ factor and the
    /// scale-scale block H_{ls,ls} carries κ².
    kappa: Array1<f64>,
    /// κ' = dκ/dη_ls = κ(1−κ) for the logb link. The static H_{ls,ls} block
    /// carries a κ'·(a−n) term, so κ' threads through every dH directional
    /// weight via the chain rule.
    kappa_prime: Array1<f64>,
    /// κ'' = κ(1−κ)(1−2κ); appears in d²H_{ls,ls} via the second
    /// η-derivative of κ'·(a−n).
    kappa_dprime: Array1<f64>,
}

pub(crate) struct GaussianJointPsiFirstWeights {
    objective_psirow: Array1<f64>,
    scoremu: Array1<f64>,
    score_ls: Array1<f64>,
    dscoremu: Array1<f64>,
    dscore_ls: Array1<f64>,
    hmumu: Array1<f64>,
    hmu_ls: Array1<f64>,
    h_ls_ls: Array1<f64>,
    dhmumu: Array1<f64>,
    dhmu_ls: Array1<f64>,
    dh_ls_ls: Array1<f64>,
}

pub(crate) struct GaussianJointPsiSecondWeights {
    objective_psi_psirow: Array1<f64>,
    d2scoremu: Array1<f64>,
    d2score_ls: Array1<f64>,
    d2hmumu: Array1<f64>,
    d2hmu_ls: Array1<f64>,
    d2h_ls_ls: Array1<f64>,
}

pub(crate) struct GaussianJointPsiMixedDriftWeights {
    dhmumu_u: Array1<f64>,
    dhmu_ls_u: Array1<f64>,
    dh_ls_ls_u: Array1<f64>,
    d2hmumu: Array1<f64>,
    d2hmu_ls: Array1<f64>,
    d2h_ls_ls: Array1<f64>,
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
    rows: &[crate::families::marginal_slope_shared::WeightedOuterRow],
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
    rows: &[crate::families::marginal_slope_shared::WeightedOuterRow],
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
    rows: &[crate::families::marginal_slope_shared::WeightedOuterRow],
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
        let jet = crate::families::sigma_link::logb_sigma_jet1_scalar(eta_ls[i]);
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
        let ki = scalars.kappa[i];
        let kpi = scalars.kappa_prime[i];
        let ai = scalars.obs_weight[i];
        let dm = dotmu[i];
        let de = dot_eta[i];
        // κ-scaled log-sigma direction.
        let sde = ki * de;
        w_u[i].write(-2.0 * wi * sde);
        // + 2·κ'·m·de: dκ/dη chain-rule from σ = b + e^η.
        c_u[i].write(ki * (-2.0 * wi * dm - 4.0 * mi * sde) + 2.0 * mi * kpi * de);
        // Directional derivative of Fisher E[H_{ls,ls}]=2κ²a: 4κκ'a·de (#566).
        d_u[i].write(4.0 * ki * kpi * ai * de);
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
        // d²/du dv of Fisher E[H_{ls,ls}]=2κ²a: bilinear in fixed directions
        // u,v, no μ dependence ⇒ 4a(κ'²+κκ'')·deu·dev (#566).
        d_uv[i].write(4.0 * ai * (kpi * kpi + ki * kdpi) * de_eta);
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

/// Canonical Gaussian location-scale Fisher (expected) joint-Hessian row
/// coefficients `(mm, ml, ll)` — the SINGLE source of truth for this curvature,
/// shared by every representation that assembles the value Hessian (the dense
/// `exact_newton_joint_hessian_from_designs` and the matrix-free
/// `GaussianLocationScaleHessianWorkspace`). The (μ, log σ) information is
/// block-diagonal because location and scale are information-orthogonal:
///   `ml = E[H_{μ,ls}] = 2κ·E[m] = 2κ·E[r]·w/σ² = 0`  (E[r]=0 at any β; #684),
/// and the (log σ, log σ) block is the residual-free Fisher form
///   `ll = E[H_{ls,ls}] = 2κ²a`  (a = obs_weight; #566).
/// Routing both paths through this one constructor makes the cross-block drift
/// that caused #684 — one representation using the observed `2κm`, another the
/// Fisher 0 — structurally impossible: they cannot disagree because they read
/// the same coefficients. The observed SCORE still drives the Newton step
/// (Fisher scoring → exact joint MLE); only the curvature feeding the REML
/// determinant / Newton metric is the orthogonal expectation.
pub(crate) fn gaussian_locscale_fisher_joint_row_coeffs(
    rows: &GaussianJointRowScalars,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mm = rows.w.clone();
    let ml = Array1::<f64>::zeros(rows.kappa.len());
    let ll = 2.0 * &rows.kappa * &rows.kappa * &rows.obs_weight;
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
) -> Result<Option<std::sync::Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
{
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

impl GaussianLocationScaleFamily {
    pub const BLOCK_MU: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;

    pub(crate) fn get_or_compute_row_scalars(
        &self,
        etamu: &Array1<f64>,
        eta_ls: &Array1<f64>,
    ) -> Result<Arc<GaussianJointRowScalars>, String> {
        Ok(Arc::new(gaussian_jointrow_scalars(
            &self.y,
            etamu,
            eta_ls,
            &self.weights,
        )?))
    }

    pub fn parameternames() -> &'static [&'static str] {
        &["mu", "log_sigma"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Identity, ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "gaussian_location_scale",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }

    pub(crate) fn exact_joint_supported(&self) -> bool {
        self.mu_design.is_some() && self.log_sigma_design.is_some()
    }

    pub(crate) fn exact_block_designs(
        &self,
    ) -> Result<(DenseOrOperator<'_>, DenseOrOperator<'_>), String> {
        let mu_design = self.mu_design.as_ref().ok_or_else(|| {
            "GaussianLocationScaleFamily exact path is missing mu design".to_string()
        })?;
        let log_sigma_design = self.log_sigma_design.as_ref().ok_or_else(|| {
            "GaussianLocationScaleFamily exact path is missing log-sigma design".to_string()
        })?;
        let planned = dense_blocks_planned_budget(&[mu_design, log_sigma_design]);
        let xmu = dense_block_or_operator(
            mu_design,
            mu_design.nrows(),
            mu_design.ncols(),
            planned[0],
            &self.policy,
        );
        let x_ls = dense_block_or_operator(
            log_sigma_design,
            log_sigma_design.nrows(),
            log_sigma_design.ncols(),
            planned[1],
            &self.policy,
        );
        Ok((xmu, x_ls))
    }

    pub(crate) fn exact_block_designs_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<(DenseOrOperator<'a>, DenseOrOperator<'a>), String> {
        if specs.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily spec-aware exact path expects 2 specs, got {}",
                    specs.len()
                ),
            }
            .into());
        }
        let mu_design = &specs[Self::BLOCK_MU].design;
        let log_sigma_design = &specs[Self::BLOCK_LOG_SIGMA].design;
        let planned = dense_blocks_planned_budget(&[mu_design, log_sigma_design]);
        let xmu = dense_block_or_operator(
            mu_design,
            mu_design.nrows(),
            mu_design.ncols(),
            planned[0],
            &self.policy,
        );
        let x_ls = dense_block_or_operator(
            log_sigma_design,
            log_sigma_design.nrows(),
            log_sigma_design.ncols(),
            planned[1],
            &self.policy,
        );
        Ok((xmu, x_ls))
    }

    pub(crate) fn exact_joint_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(DenseOrOperator<'a>, DenseOrOperator<'a>)>, String> {
        if self.exact_joint_supported() {
            return self.exact_block_designs().map(Some);
        }
        if let Some(specs) = specs {
            return self.exact_block_designs_fromspecs(specs).map(Some);
        }
        Ok(None)
    }

    pub(crate) fn exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_block_designs(specs)? else {
            return Ok(None);
        };
        let xmu = match xmu {
            DenseOrOperator::Borrowed(dense) => Cow::Borrowed(dense),
            DenseOrOperator::Owned(dense) => Cow::Owned(dense),
            DenseOrOperator::Operator(_) => {
                return Err(
                    "GaussianLocationScaleFamily exact psi path requires chunked operator support for oversized designs"
                        .to_string(),
                );
            }
        };
        let x_ls = match x_ls {
            DenseOrOperator::Borrowed(dense) => Cow::Borrowed(dense),
            DenseOrOperator::Owned(dense) => Cow::Owned(dense),
            DenseOrOperator::Operator(_) => {
                return Err(
                    "GaussianLocationScaleFamily exact psi path requires chunked operator support for oversized designs"
                        .to_string(),
                );
            }
        };
        Ok(Some((xmu, x_ls)))
    }

    pub(crate) fn exact_newton_joint_hessian_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_from_designs(block_states, &xmu, &x_ls)
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_directional_derivative_from_designs(
            block_states,
            &xmu,
            &x_ls,
            d_beta_flat,
        )
    }

    pub(crate) fn exact_newton_joint_hessian_second_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessiansecond_directional_derivative_from_designs(
            block_states,
            &xmu,
            &x_ls,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    pub(crate) fn exact_newton_joint_psi_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psi_terms_from_designs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            &xmu,
            &x_ls,
        )
    }

    pub(crate) fn exact_newton_joint_psisecond_order_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psisecond_order_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &xmu,
            &x_ls,
        )
    }

    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psihessian_directional_derivative_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &xmu,
            &x_ls,
        )
    }

    pub(crate) fn exact_newton_joint_hessian_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &DenseOrOperator<'_>,
        x_ls: &DenseOrOperator<'_>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        // Block-diagonal Gaussian Fisher curvature (μ ⊥ σ ⇒ cross = 0, #684;
        // (ls,ls) = 2κ²a, #566), built from the shared single-source-of-truth
        // constructor so this dense path and the matrix-free workspace can never
        // disagree on the cross block. See `gaussian_locscale_fisher_joint_row_coeffs`.
        let (mm, cross, scale) = gaussian_locscale_fisher_joint_row_coeffs(&rows);
        Ok(Some(gaussian_joint_hessian_from_designs(
            xmu, x_ls, &mm, &cross, &scale,
        )?))
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &DenseOrOperator<'_>,
        x_ls: &DenseOrOperator<'_>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let total = pmu + p_ls;
        if d_beta_flat.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily joint d_beta length mismatch: got {}, expected {}",
                    d_beta_flat.len(),
                    total
                ),
            }
            .into());
        }
        let ximu = xmu.dot(d_beta_flat.slice(s![0..pmu]));
        let xi_ls = x_ls.dot(d_beta_flat.slice(s![pmu..pmu + p_ls]));
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let directional = gaussian_joint_first_directionalweights(&rows, &ximu, &xi_ls);
        let dhmumu = directional.0;
        let dh_ls_ls = directional.2;
        // Fisher cross block E[H_{μ,ls}] ≡ 0 (μ ⊥ σ; see
        // exact_newton_joint_hessian_from_designs / #684), so its directional
        // derivative is identically 0 — keep the Hessian's curvature object the
        // block-diagonal Gaussian Fisher information at every order. The
        // observed-cross directional weight (`directional.1`) is therefore not
        // assembled.
        let dhmu_ls = Array1::<f64>::zeros(dhmumu.len());

        Ok(Some(gaussian_joint_hessian_from_designs(
            xmu, x_ls, &dhmumu, &dhmu_ls, &dh_ls_ls,
        )?))
    }

    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &DenseOrOperator<'_>,
        x_ls: &DenseOrOperator<'_>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let total = pmu + p_ls;
        if d_beta_u_flat.len() != total || d_betav_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleFamily joint second directional derivative length mismatch: got {} and {}, expected {}",
                d_beta_u_flat.len(),
                d_betav_flat.len(),
                total
            ) }.into());
        }
        let ximu_u = xmu.dot(d_beta_u_flat.slice(s![0..pmu]));
        let xi_ls_u = x_ls.dot(d_beta_u_flat.slice(s![pmu..pmu + p_ls]));
        let ximuv = xmu.dot(d_betav_flat.slice(s![0..pmu]));
        let xi_lsv = x_ls.dot(d_betav_flat.slice(s![pmu..pmu + p_ls]));
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let second =
            gaussian_jointsecond_directionalweights(&rows, &ximu_u, &xi_ls_u, &ximuv, &xi_lsv);
        let d2hmumu = second.0;
        let d2h_ls_ls = second.2;
        // Fisher cross block E[H_{μ,ls}] ≡ 0 (μ ⊥ σ; #684), so its second
        // directional derivative is identically 0; `second.1` (observed) is not
        // assembled, keeping the curvature object block-diagonal Fisher.
        let d2hmu_ls = Array1::<f64>::zeros(d2hmumu.len());

        Ok(Some(gaussian_joint_hessian_from_designs(
            xmu, x_ls, &d2hmumu, &d2hmu_ls, &d2h_ls_ls,
        )?))
    }

    pub(crate) fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
    ) -> Result<Option<LocationScaleJointPsiDirection>, String> {
        let Some(parts) = locscale_joint_psi_direction_parts(
            block_states,
            derivative_blocks,
            psi_index,
            self.y.len(),
            xmu.ncols(),
            x_ls.ncols(),
            Self::BLOCK_MU,
            Self::BLOCK_LOG_SIGMA,
            2,
            "GaussianLocationScaleFamily",
            "mu",
            policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LocationScaleJointPsiDirection {
            block_idx: parts.block_idx,
            local_idx: parts.local_idx,
            z_primary_psi: parts.primary_z,
            z_ls_psi: parts.log_sigma_z,
            x_primary_psi: parts.primary_psi,
            x_ls_psi: parts.log_sigma_psi,
        }))
    }

    pub(crate) fn exact_newton_joint_psisecond_design_drifts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &LocationScaleJointPsiDirection,
        psi_b: &LocationScaleJointPsiDirection,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<LocationScaleJointPsiSecondDrifts, String> {
        locscale_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            psi_a,
            psi_b,
            LocScalePsiDriftConfig {
                n: self.y.len(),
                p_primary: xmu.ncols(),
                p_log_sigma: x_ls.ncols(),
                primary_block_idx: Self::BLOCK_MU,
                log_sigma_block_idx: Self::BLOCK_LOG_SIGMA,
                family_name: "GaussianLocationScaleFamily",
                primary_label: "mu",
                policy: &self.policy,
            },
        )
    }

    pub(crate) fn exact_newton_joint_psi_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        if specs.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleFamily joint psi terms expect 2 specs and 2 derivative blocks, got {} and {}",
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        // Gaussian 2-block location-scale family in the unified flattened
        // coefficient space beta = [betamu; beta_sigma]:
        //
        //   mu_i = z_i^T betamu,
        //   ell_i = x_i^T beta_sigma,
        //   s_i = exp(ell_i),
        //   r_i = y_i - mu_i,
        //   q_i = r_i / s_i,
        //   w_i = s_i^{-2},
        //   alpha_i = r_i s_i^{-2},
        //   b_i = q_i^2.
        //
        // The first fixed-beta psi object returned here is likelihood-only:
        //
        //   D_a         = -alpha^T m_a + (1 - b)^T ell_a
        //   D_{beta a}  = [ -Xmu^T alpha_a - X_{mu,a}^T alpha ;
        //                   -X_sigma^T b_a + X_{sigma,a}^T (1-b) ]
        //   D_{bb a}    = [ Xmu^T W_a Xmu + X_{mu,a}^T W Xmu + Xmu^T W X_{mu,a},
        //                   2( Xmu^T A_a X_sigma + X_{mu,a}^T A X_sigma + Xmu^T A X_{sigma,a} );
        //                   sym,
        //                   2( X_sigma^T B_a X_sigma + X_{sigma,a}^T B X_sigma + X_sigma^T B X_{sigma,a} ) ]
        //
        // with m_a = X_{mu,a} betamu, ell_a = X_{sigma,a} beta_sigma and
        // rowwise scalar drifts
        //
        //   w_a     = -2 w * ell_a
        //   alpha_a = -w * m_a - 2 alpha * ell_a
        //   b_a     = -2 alpha * m_a - 2 b * ell_a.
        //
        // Generic code in custom_family.rs promotes these likelihood-only
        // objects to the full fixed-beta V_a / g_a / H_a by adding S_a.
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let weights_a =
            gaussian_joint_psi_firstweights(&rows, &dir_a.z_primary_psi, &dir_a.z_ls_psi);
        let objective_psi = weights_a.objective_psirow.sum();
        let xmu_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let score_mu =
            xmu_map.transpose_mul(weights_a.scoremu.view()) + fast_atv(xmu, &weights_a.dscoremu);
        let score_ls = x_ls_map.transpose_mul(weights_a.score_ls.view())
            + fast_atv(x_ls, &weights_a.dscore_ls);
        let score_psi = gaussian_pack_joint_score(&score_mu, &score_ls);
        let hessian_psi_operator = build_two_block_custom_family_joint_psi_operator_from_actions(
            dir_a.x_primary_psi.cloned_first_action(),
            dir_a.x_ls_psi.cloned_first_action(),
            0..xmu.ncols(),
            xmu.ncols()..xmu.ncols() + x_ls.ncols(),
            xmu,
            x_ls,
            &weights_a.hmumu,
            &weights_a.hmu_ls,
            &weights_a.h_ls_ls,
            &weights_a.dhmumu,
            &weights_a.dhmu_ls,
            &weights_a.dh_ls_ls,
        )?;
        let hessian_psi = if hessian_psi_operator.is_some() {
            Array2::zeros((0, 0))
        } else {
            gaussian_joint_psihessian_fromweights(xmu, x_ls, xmu_map, x_ls_map, &weights_a)?
        };

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator,
        }))
    }

    pub(crate) fn exact_newton_joint_psisecond_order_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_i) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_i,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let Some(dir_j) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_j,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psisecond_order_terms_from_parts(
                block_states,
                derivative_blocks,
                &dir_i,
                &dir_j,
                xmu,
                x_ls,
                None,
            )?,
        ))
    }

    pub(crate) fn exact_newton_joint_psisecond_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        dir_i: &LocationScaleJointPsiDirection,
        dir_j: &LocationScaleJointPsiDirection,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
    ) -> Result<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms, String> {
        let second_drifts = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            dir_i,
            dir_j,
            xmu,
            x_ls,
        )?;
        let n = self.y.len();
        let xmu_i_map = dir_i.x_primary_psi.as_linear_map_ref();
        let x_ls_i_map = dir_i.x_ls_psi.as_linear_map_ref();
        let xmu_j_map = dir_j.x_primary_psi.as_linear_map_ref();
        let x_ls_j_map = dir_j.x_ls_psi.as_linear_map_ref();
        let xmu_ab_map = second_psi_linear_map(
            second_drifts.x_primary_ab_action.as_ref(),
            second_drifts.x_primary_ab.as_ref(),
            n,
            xmu.ncols(),
        );
        let x_ls_ab_map = second_psi_linear_map(
            second_drifts.x_ls_ab_action.as_ref(),
            second_drifts.x_ls_ab.as_ref(),
            n,
            x_ls.ncols(),
        );
        // Second fixed-beta psi objects for the same Gaussian location-scale
        // kernel. Using the notation from the first-order comment, the rowwise
        // second psi drifts are
        //
        //   w_ab     = 4 w * ell_a * ell_b - 2 w * ell_ab
        //   alpha_ab = 2 w * (m_a * ell_b + m_b * ell_a)
        //              + 4 alpha * ell_a * ell_b
        //              - w * m_ab
        //              - 2 alpha * ell_ab
        //   b_ab     = 2 w * m_a * m_b
        //              + 4 alpha * (m_a * ell_b + m_b * ell_a)
        //              + 4 b * ell_a * ell_b
        //              - 2 alpha * m_ab
        //              - 2 b * ell_ab.
        //
        // The exact likelihood-only second-order objects are then:
        //
        //   D_ab,
        //   D_{beta ab},
        //   D_{beta beta ab},
        //
        // assembled from the usual product-rule expansion over realized
        // design motion X_{.,a}, X_{.,b}, X_{.,ab}. Generic code adds S_ab.
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let mut weights_i =
            gaussian_joint_psi_firstweights(&rows, &dir_i.z_primary_psi, &dir_i.z_ls_psi);
        let mut weights_j =
            gaussian_joint_psi_firstweights(&rows, &dir_j.z_primary_psi, &dir_j.z_ls_psi);
        let mut secondweights = gaussian_joint_psisecondweights(
            &rows,
            &dir_i.z_primary_psi,
            &dir_i.z_ls_psi,
            &dir_j.z_primary_psi,
            &dir_j.z_ls_psi,
            &second_drifts.z_primary_ab,
            &second_drifts.z_ls_ab,
        );
        if let Some(sub_rows) = subsample {
            // HT mask: every downstream consumer (gaussian_joint_psisecondhessian_fromweights,
            // weighted_crossprod_psi_maps with weights_*.{hmumu,hmu_ls,h_ls_ls},
            // fast_atv on d2score_* and dscore_*) is row-linear in these arrays, so
            // scaling sampled rows by 1/π_i and zeroing the rest yields an unbiased
            // estimator of the full-data second-order ψ Hessian and ψ score.
            apply_ht_mask_first(&mut weights_i, sub_rows);
            apply_ht_mask_first(&mut weights_j, sub_rows);
            apply_ht_mask_second(&mut secondweights, sub_rows);
        }
        let objective_psi_psi = secondweights.objective_psi_psirow.sum();

        let score_psi_psi = gaussian_pack_joint_score(
            &(xmu_ab_map.transpose_mul(weights_i.scoremu.view())
                + xmu_i_map.transpose_mul(weights_j.dscoremu.view())
                + xmu_j_map.transpose_mul(weights_i.dscoremu.view())
                + fast_atv(xmu, &secondweights.d2scoremu)),
            &(x_ls_ab_map.transpose_mul(weights_i.score_ls.view())
                + x_ls_i_map.transpose_mul(weights_j.dscore_ls.view())
                + x_ls_j_map.transpose_mul(weights_i.dscore_ls.view())
                + fast_atv(x_ls, &secondweights.d2score_ls)),
        );
        let hessian_psi_psi = gaussian_joint_psisecondhessian_fromweights(
            xmu,
            x_ls,
            xmu_i_map,
            x_ls_i_map,
            xmu_j_map,
            x_ls_j_map,
            xmu_ab_map,
            x_ls_ab_map,
            &weights_i,
            &weights_j,
            &secondweights,
        )?;

        Ok(crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
            hessian_psi_psi_operator: None,
        })
    }

    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psihessian_directional_derivative_from_parts(
                block_states,
                &dir_a,
                d_beta_flat,
                xmu,
                x_ls,
                None,
            )?,
        ))
    }

    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        dir_a: &LocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
    ) -> Result<Array2<f64>, String> {
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let xmu_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let total = pmu + p_ls;
        if d_beta_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleFamily joint psi hessian directional derivative length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                total
            ) }.into());
        }
        // Only the log-σ–channel direction enters the surviving Fisher blocks
        // of the mixed drift (the μ-channel direction fed the observed cross
        // block, now Fisher 0; μ ⊥ σ, #684).
        let u_ls = d_beta_flat.slice(s![pmu..pmu + p_ls]);
        let xi_ls = fast_av(x_ls, &u_ls);
        let uza_ls = x_ls_map.forward_mul(u_ls);
        // Mixed drift T_a[u] = D_beta H_a^{(D)}[u] for the Gaussian family.
        //
        // Along u = [umu; u_sigma], define xi = Xmu umu and zeta = X_sigma u_sigma.
        // The first beta-directional drifts of the Gaussian row scalars are
        //
        //   d_u w     = -2 w * zeta
        //   d_u alpha = -w * xi - 2 alpha * zeta
        //   d_u b     = -2 alpha * xi - 2 b * zeta.
        //
        // Differentiating the psi-a scalar drifts once more gives
        //
        //   d_u w_a     = 4 w * ell_a * zeta - 2 w * zeta_a
        //   d_u alpha_a = 2 w * (m_a * zeta + ell_a * xi)
        //                 - w * xi_a
        //                 + 4 alpha * ell_a * zeta
        //                 - 2 alpha * zeta_a
        //   d_u b_a     = 2 w * m_a * xi
        //                 + 4 alpha * (m_a * zeta + ell_a * xi)
        //                 + 4 b * ell_a * zeta
        //                 - 2 alpha * xi_a
        //                 - 2 b * zeta_a.
        //
        // The matrix drift returned here is the exact likelihood-only
        //
        //   T_a[u] = D_beta H_{psi_a}^{(D)}[u],
        //
        // assembled blockwise as
        //
        //   Kmumu,a[u]   = Xmu^T W_a[u] Xmu
        //                   + X_{mu,a}^T W[u] Xmu
        //                   + Xmu^T W[u] X_{mu,a}
        //   Kmusigma,a[u]= 2( Xmu^T A_a[u] X_sigma
        //                   + X_{mu,a}^T A[u] X_sigma
        //                   + Xmu^T A[u] X_{sigma,a} )
        //   K_sigmasigma,a[u]
        //                   = 2( X_sigma^T B_a[u] X_sigma
        //                   + X_{sigma,a}^T B[u] X_sigma
        //                   + X_sigma^T B[u] X_{sigma,a} ).
        //
        // Generic code then combines this with S(theta)-motion and the profile
        // mode responses to form ddot H_{ij}.
        let rows = self.get_or_compute_row_scalars(etamu, eta_ls)?;
        let mut mixedweights =
            gaussian_joint_psi_mixed_driftweights(&rows, &xi_ls, &dir_a.z_ls_psi, &uza_ls);
        if let Some(sub_rows) = subsample {
            // HT mask: `gaussian_joint_psi_mixedhessian_drift_fromweights` is
            // row-linear in every `mixedweights.*` array via `xt_diag_*_dense`
            // and `weighted_crossprod_psi_maps`, so the masked Hessian-drift
            // remains an unbiased estimator of the full-data drift.
            apply_ht_mask_mixed(&mut mixedweights, sub_rows);
        }

        gaussian_joint_psi_mixedhessian_drift_fromweights(
            xmu,
            x_ls,
            xmu_map,
            x_ls_map,
            &mixedweights,
        )
    }

    /// Build the [`BlockEffectiveJacobian`] for block `block_idx` given the
    /// realised block specs.  Returns an [`AdditiveBlockJacobian`] encoding the
    /// linear map η_r[i] = X_r[i,:] · β_r:
    ///
    /// - block 0 (mu):       output 0 = design rows, output 1 = zeros
    /// - block 1 (log_sigma): output 0 = zeros, output 1 = design rows
    pub fn block_effective_jacobian(
        specs: &[ParameterBlockSpec],
        block_idx: usize,
    ) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
        crate::util::block_jacobian::AdditiveWiggleBlockLayout {
            family: "GaussianLocationScaleFamily",
            n_outputs: 2,
            additive_blocks: &[Self::BLOCK_MU, Self::BLOCK_LOG_SIGMA],
            wiggle_block: None,
        }
        .block_effective_jacobian(specs, block_idx)
    }
}

/// Per-subject 2×2 channel Hessian `W_i` for Gaussian location-scale.
///
/// The row negative log-likelihood (with per-row weight `w_i`, response `y_i`,
/// mean predictor `μ_i`, log-scale predictor `s_i = log σ_i`) is
///
/// ```text
/// ρ_i(μ, s) = w_i [s + 0.5·(y_i − μ)²·exp(−2s)]
/// ```
///
/// The 2×2 Hessian in `(μ, s)` coordinates:
///
/// ```text
/// W_i[0,0] = w_i · exp(−2 s_i)                        ∂²ρ/∂μ²
/// W_i[1,1] = w_i · 2·(y_i − μ_i)²·exp(−2 s_i)        ∂²ρ/∂s²
/// W_i[0,1] = W_i[1,0] = w_i · 2·(y_i − μ_i)·exp(−2 s_i)  ∂²ρ/∂μ∂s
/// ```
///
/// The off-diagonal cross-channel term `∂²ρ/∂μ∂s` is nonzero whenever the
/// residual `(y_i − μ_i) ≠ 0`, i.e. away from the fitted mean.
pub struct GaussianLocationScaleChannelHessian {
    /// Row-major `(n × 2 × 2)` PSD-clamped per-subject Hessian.
    h: ndarray::Array3<f64>,
}

impl GaussianLocationScaleChannelHessian {
    /// Construct the raw (un-PSD-clamped) per-subject observed Hessian.
    ///
    /// For Gaussian location-scale the 2×2 observed Hessian
    /// `[[w·e^{-2s}, 2·w·r·e^{-2s}], [2·w·r·e^{-2s}, 2·w·r²·e^{-2s}]]`
    /// has determinant `-2·w²·r²·e^{-4s}` which is non-positive whenever
    /// the residual `r = y − μ ≠ 0`. Tests that finite-difference the row
    /// NLL must compare against this raw observed Hessian — PSD clamping
    /// alters the eigenvalues and the FD-versus-closed-form match fails.
    ///
    /// Production code that needs a PSD matrix (e.g. the canonicalize gate)
    /// must call [`Self::from_pilot`] which PSD-clamps via 2×2
    /// eigendecomposition.
    pub fn from_pilot_observed_unclamped(
        y: &ndarray::Array1<f64>,
        w: &ndarray::Array1<f64>,
        eta_mu: &ndarray::Array1<f64>,
        eta_log_sigma: &ndarray::Array1<f64>,
    ) -> Result<Self, String> {
        let n = y.len();
        if w.len() != n || eta_mu.len() != n || eta_log_sigma.len() != n {
            return Err(format!(
                "GaussianLocationScaleChannelHessian::from_pilot_observed_unclamped: \
                 length mismatch y={n} w={} eta_mu={} eta_log_sigma={}",
                w.len(),
                eta_mu.len(),
                eta_log_sigma.len(),
            ));
        }
        let mut h = ndarray::Array3::<f64>::zeros((n, 2, 2));
        for i in 0..n {
            let wi = w[i];
            let mu_i = eta_mu[i];
            let s_i = eta_log_sigma[i];
            let inv_sigma2 = (-2.0 * s_i).exp();
            let resid = y[i] - mu_i;
            h[[i, 0, 0]] = wi * inv_sigma2;
            h[[i, 1, 1]] = wi * 2.0 * resid * resid * inv_sigma2;
            h[[i, 0, 1]] = wi * 2.0 * resid * inv_sigma2;
            h[[i, 1, 0]] = h[[i, 0, 1]];
        }
        Ok(Self { h })
    }

    /// Construct from pilot predictors (μ and log σ at current β) and data,
    /// with PSD eigenvalue clamping applied per subject.
    ///
    /// `y` is the response, `w` the per-row sample weights, `eta_mu` and
    /// `eta_log_sigma` the current linear predictors. Negative eigenvalues
    /// are projected to zero (PSD clamp) before storage so the resulting
    /// matrix is a valid metric for the W-Gram identifiability compile.
    pub fn from_pilot(
        y: &ndarray::Array1<f64>,
        w: &ndarray::Array1<f64>,
        eta_mu: &ndarray::Array1<f64>,
        eta_log_sigma: &ndarray::Array1<f64>,
    ) -> Result<Self, String> {
        let n = y.len();
        if w.len() != n || eta_mu.len() != n || eta_log_sigma.len() != n {
            return Err(format!(
                "GaussianLocationScaleChannelHessian::from_pilot: \
                 length mismatch y={n} w={} eta_mu={} eta_log_sigma={}",
                w.len(),
                eta_mu.len(),
                eta_log_sigma.len(),
            ));
        }
        let mut h = ndarray::Array3::<f64>::zeros((n, 2, 2));
        for i in 0..n {
            let wi = w[i];
            let mu_i = eta_mu[i];
            let s_i = eta_log_sigma[i];
            let inv_sigma2 = (-2.0 * s_i).exp(); // exp(-2s) = 1/sigma^2
            let resid = y[i] - mu_i;
            // Hessian of w_i * ρ_i
            let h00 = wi * inv_sigma2;
            let h11 = wi * 2.0 * resid * resid * inv_sigma2;
            let h01 = wi * 2.0 * resid * inv_sigma2;
            // PSD clamp via eigendecomposition of 2×2 matrix.
            // psd_clamp_2x2 returns (λ1, λ2, u1[0], u1[1], u2[0], u2[1])
            // where u1 and u2 are unit eigenvectors for λ1 and λ2.
            // Reconstruction: H_psd = λ1·u1·u1ᵀ + λ2·u2·u2ᵀ
            let (e0, e1, u1_0, u1_1, u2_0, u2_1) = psd_clamp_2x2(h00, h01, h11);
            h[[i, 0, 0]] = e0 * u1_0 * u1_0 + e1 * u2_0 * u2_0;
            h[[i, 0, 1]] = e0 * u1_0 * u1_1 + e1 * u2_0 * u2_1;
            h[[i, 1, 0]] = h[[i, 0, 1]];
            h[[i, 1, 1]] = e0 * u1_1 * u1_1 + e1 * u2_1 * u2_1;
        }
        Ok(Self { h })
    }
}

impl FamilyChannelHessian for GaussianLocationScaleChannelHessian {
    pub(crate) fn n_outputs(&self) -> usize {
        2
    }

    pub(crate) fn n_subjects(&self) -> usize {
        self.h.shape()[0]
    }

    pub(crate) fn fill_subject(&self, i: usize, out: &mut [f64]) {
        assert_eq!(out.len(), 4);
        out[0] = self.h[[i, 0, 0]];
        out[1] = self.h[[i, 0, 1]];
        out[2] = self.h[[i, 1, 0]];
        out[3] = self.h[[i, 1, 1]];
    }

    pub(crate) fn evaluate_full(&self) -> ndarray::Array3<f64> {
        self.h.clone()
    }
}

impl CustomFamily for GaussianLocationScaleFamily {
    /// The Gaussian location-scale joint Hessian depends on β because the
    /// cross-block (μ,log σ) and (log σ,log σ) blocks contain the residual
    /// r = y − μ (via the row scalars m = r·w and n = r²·w), which changes
    /// when β_μ moves.  The (μ,μ) block weight w = 1/σ² also depends on
    /// β_{log σ}.  This override is essential for correct M_j[u] drift
    /// corrections when ψ hyperparameters move the design matrices.
    pub(crate) fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    /// Two independent linear predictors: block 0 → μ channel, block 1 → log σ
    /// channel. Declaring the channel topology lets `fit_custom_family` route
    /// the identifiability audit channel-aware even when a caller builds the
    /// blocks by hand (without `build_location_scale_block`'s callbacks), so a
    /// shared μ/log-σ covariate basis is recognised as block-diagonal rather
    /// than mistaken for cross-block intercept aliases (#558).
    pub(crate) fn output_channel_assignment(
        &self,
        specs: &[ParameterBlockSpec],
    ) -> Option<Vec<usize>> {
        // Two-channel families: `[mu, log_sigma]`. The optional trailing
        // zero-channel wiggle block (when present) also drives channel 0.
        Some(
            (0..specs.len())
                .map(|i| usize::from(i == Self::BLOCK_LOG_SIGMA))
                .collect(),
        )
    }

    pub(crate) fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Operator-aware: when the unified evaluator picks the matrix-free
        // joint Hessian path (see `use_joint_matrix_free_path`), the workspace
        // applies the joint Hessian via row-streaming Khatri-Rao matvecs at
        // O(n · (p_t + p_ℓ)) per Hv, never building the dense (p_t + p_ℓ)²
        // matrix. Report the operator work model so diagnostics and
        // first-order-only policies reflect the representation that actually
        // runs.
        crate::families::location_scale_engine::location_scale_coefficient_hessian_cost(
            self.y.len() as u64,
            specs,
        )
    }

    pub(crate) fn evaluate(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_log_sigma.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        // Diagonal IRLS weights for the inner solver.
        //
        // For the location block (identity link): wmu = pw / sigma^2. Since the
        // location link is identity, observed = Fisher --- no correction needed.
        //
        // For the log-sigma block (log link): w_ls = 2 * pw * (dsigma/deta)^2 / sigma^2.
        // This is the Fisher weight. For the outer REML, the joint
        // `exact_newton_joint_hessian` provides the full observed Hessian directly,
        // so these Diagonal weights are only used for the inner IRLS iteration
        // (where Fisher scoring is fine). See response.md Section 3.
        //
        let mut zmu = Array1::<f64>::zeros(n);
        let mut wmu = Array1::<f64>::zeros(n);
        let mut z_ls = Array1::<f64>::zeros(n);
        let mut w_ls = Array1::<f64>::zeros(n);
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let mut ll = 0.0;

        pub(crate) const CHUNK: usize = 1024;
        if let (
            Some(y_s),
            Some(w_s),
            Some(mu_s),
            Some(ls_s),
            Some(zmu_s),
            Some(wmu_s),
            Some(zls_s),
            Some(wls_s),
        ) = (
            self.y.as_slice_memory_order(),
            self.weights.as_slice_memory_order(),
            etamu.as_slice_memory_order(),
            eta_log_sigma.as_slice_memory_order(),
            zmu.as_slice_memory_order_mut(),
            wmu.as_slice_memory_order_mut(),
            z_ls.as_slice_memory_order_mut(),
            w_ls.as_slice_memory_order_mut(),
        ) {
            // Per-row Gaussian LS kernel writes 4 working arrays directly into
            // the output slices; ll is reduced via Rayon's sum. Independent
            // across rows.
            ll += zmu_s
                .par_chunks_mut(CHUNK)
                .zip(wmu_s.par_chunks_mut(CHUNK))
                .zip(zls_s.par_chunks_mut(CHUNK))
                .zip(wls_s.par_chunks_mut(CHUNK))
                .enumerate()
                .map(|(chunk_idx, (((zmu_c, wmu_c), zls_c), wls_c))| {
                    let start = chunk_idx * CHUNK;
                    let mut local_ll = 0.0;
                    for local in 0..zmu_c.len() {
                        let i = start + local;
                        let row =
                            gaussian_diagonal_row_kernel(y_s[i], mu_s[i], ls_s[i], w_s[i], ln2pi);
                        zmu_c[local] = mu_s[i] + row.location_working_shift;
                        wmu_c[local] = row.location_working_weight;
                        zls_c[local] = row.log_sigma_working_response;
                        wls_c[local] = row.log_sigma_working_weight;
                        local_ll += row.log_likelihood;
                    }
                    local_ll
                })
                .sum::<f64>();
        } else {
            // Fallback path: inputs are not contiguous. Outputs (just-allocated
            // Array1::zeros) always are. Reborrow input views into the closure.
            let y_view = self.y.view();
            let w_view = self.weights.view();
            let mu_view = etamu.view();
            let ls_view = eta_log_sigma.view();
            let zmu_s = zmu
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let wmu_s = wmu
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let zls_s = z_ls
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let wls_s = w_ls
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            ll += zmu_s
                .par_chunks_mut(CHUNK)
                .zip(wmu_s.par_chunks_mut(CHUNK))
                .zip(zls_s.par_chunks_mut(CHUNK))
                .zip(wls_s.par_chunks_mut(CHUNK))
                .enumerate()
                .map(|(chunk_idx, (((zmu_c, wmu_c), zls_c), wls_c))| {
                    let start = chunk_idx * CHUNK;
                    let mut local_ll = 0.0;
                    for local in 0..zmu_c.len() {
                        let i = start + local;
                        let row = gaussian_diagonal_row_kernel(
                            y_view[i], mu_view[i], ls_view[i], w_view[i], ln2pi,
                        );
                        zmu_c[local] = mu_view[i] + row.location_working_shift;
                        wmu_c[local] = row.location_working_weight;
                        zls_c[local] = row.log_sigma_working_response;
                        wls_c[local] = row.log_sigma_working_weight;
                        local_ll += row.log_likelihood;
                    }
                    local_ll
                })
                .sum::<f64>();
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::diagonal_checked(zmu, wmu)?,
                BlockWorkingSet::diagonal_checked(z_ls, w_ls)?,
            ],
        })
    }

    pub(crate) fn log_likelihood_only(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<f64, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_log_sigma.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }
        // logb noise link: σ(η_ls) = LOGB_SIGMA_FLOOR + exp(η_ls). σ ≥ b > 0
        // bounds the loglik below (−Σlog σ ≥ −n log b) and bounds 1/σ² by 1/b²,
        // so the previous `inv_s2.min(1e24)` cap is structurally unnecessary.
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let mut ll = 0.0;
        if let (Some(y_s), Some(w_s), Some(mu_s), Some(ls_s)) = (
            self.y.as_slice_memory_order(),
            self.weights.as_slice_memory_order(),
            etamu.as_slice_memory_order(),
            eta_log_sigma.as_slice_memory_order(),
        ) {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            ll += (0..n)
                .into_par_iter()
                .map(|i| {
                    let wi = w_s[i];
                    if wi == 0.0 {
                        return 0.0;
                    }
                    let sigma_i = logb_sigma_from_eta_scalar(ls_s[i]);
                    let inv_s2 = (sigma_i * sigma_i).recip();
                    let r = y_s[i] - mu_s[i];
                    wi * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()))
                })
                .sum::<f64>();
        } else {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            ll += (0..n)
                .into_par_iter()
                .map(|i| {
                    let wi = self.weights[i];
                    if wi == 0.0 {
                        return 0.0;
                    }
                    let sigma_i = logb_sigma_from_eta_scalar(eta_log_sigma[i]);
                    let inv_s2 = (sigma_i * sigma_i).recip();
                    let r = self.y[i] - etamu[i];
                    wi * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()))
                })
                .sum::<f64>();
        }
        Ok(ll)
    }

    /// Outer-only log-likelihood with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `Some`, only the sampled rows
    /// contribute; each row's per-row log-likelihood term is multiplied by
    /// `WeightedOuterRow.weight`, the Horvitz–Thompson inverse-inclusion
    /// factor 1/π_i (uniform or stratified sampling both supported), so the
    /// partial sum is an unbiased estimator of the full-data log-likelihood.
    /// When `None`, this returns the full-data `log_likelihood_only`. Inner
    /// PIRLS line searches never install the subsample option, so they
    /// continue to score the exact full-data log-likelihood.
    pub(crate) fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        let Some(subsample) = options.outer_score_subsample.as_ref() else {
            return self.log_likelihood_only(block_states);
        };
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let etamu = &block_states[Self::BLOCK_MU].eta;
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if etamu.len() != n || eta_log_sigma.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        use rayon::iter::ParallelIterator;
        let ll: f64 = subsample
            .rows
            .par_iter()
            .map(|row| {
                let i = row.index;
                let wi = self.weights[i];
                if wi == 0.0 {
                    return 0.0;
                }
                let sigma_i = logb_sigma_from_eta_scalar(eta_log_sigma[i]);
                let inv_s2 = (sigma_i * sigma_i).recip();
                let r = self.y[i] - etamu[i];
                row.weight * wi * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()))
            })
            .sum();
        Ok(ll)
    }

    pub(crate) fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, None)
    }

    pub(crate) fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    /// The Gaussian location-scale likelihood has no separation /
    /// under-identification regime that the full-span Jeffreys curvature `H_Φ`
    /// is meant to regularize: with the soft floor `σ ≥ b > 0` the per-row
    /// Fisher information `diag(a/σ², 2κ²a)` is bounded and `O(n)` on every
    /// identified direction at every working point, so the well-conditioned-`H`
    /// Jeffreys gate smooth-steps `H_Φ` to ~0 — yet the matching score `∇Φ`
    /// kept leaking a *phantom* penalized-stationarity residual into the inner
    /// joint-Newton (a nonzero `|∇L − Sβ|` paired with a numerically null `H_Φ`
    /// and a full-rank `H_pen`), so the KKT certificate refused every iterate
    /// and the outer REML rejected all seeds — aborting heteroscedastic
    /// location-scale fits (#684–#688). This is the same opt-out
    /// `TransformationNormalFamily` takes for the same structural reason
    /// (continuous response, `O(n)` Fisher information everywhere); it removes
    /// the phantom residual and drops the per-cycle `O(n·p²)` Jeffreys
    /// directional-derivative overhead.
    pub(crate) fn joint_jeffreys_term_required(&self) -> bool {
        false
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_flat,
        )
    }

    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    pub(crate) fn diagonalworking_weights_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_eta: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n || d_eta.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let sigma = eta_ls.mapv(logb_sigma_from_eta_scalar);
        let mut dw = Array1::<f64>::zeros(n);
        match block_idx {
            Self::BLOCK_MU => {
                // Gaussian location block:
                //
                //   wmu = weight / sigma^2.
                //
                // This depends only on the scale predictor, so along a
                // location-only direction d etamu the directional derivative is
                // identically zero.
                Ok(Some(dw))
            }
            Self::BLOCK_LOG_SIGMA => {
                // Gaussian log-sigma block:
                //
                // The PIRLS information weight is
                //
                //   w_ls = max(2 * weight * clamp(g, -1, 1)^2, MIN_WEIGHT),
                //   g    = sigma'(eta_ls) / sigma(eta_ls),
                // with the semantic rule that zero observation weights stay zero.
                //
                // Along a direction d eta_ls,
                //
                //   dw_ls is the directional derivative of that piecewise
                //   definition. On the active clamp branch or active MIN_WEIGHT
                //   floor branch, the returned derivative is zero to match the
                //   selected local piece of the evaluated weight.
                //
                // This is the exact directional derivative needed by the REML
                // trace term
                //
                //   0.5 tr(J^{-1} D_beta J[u])
                //   = 0.5 sum_i (x_i^T J^{-1} x_i) dw_i
                //
                // for diagonal working-set blocks.
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                let dw_vec: Vec<f64> = (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let d1 = crate::families::sigma_link::logb_sigma_jet1_scalar(eta_ls[i]).d1;
                        gaussian_log_sigma_irlsinfo_directional_derivative(
                            self.weights[i],
                            sigma[i],
                            d1,
                            d_eta[i],
                        )
                    })
                    .collect();
                for (i, v) in dw_vec.into_iter().enumerate() {
                    dw[i] = v;
                }
                Ok(Some(dw))
            }
            _ => Ok(None),
        }
    }

    pub(crate) fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, Some(specs))
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_flat,
        )
    }

    pub(crate) fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    pub(crate) fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        self.exact_newton_joint_psi_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
        )
    }

    pub(crate) fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.exact_newton_joint_psisecond_order_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_i,
            psi_j,
        )
    }

    pub(crate) fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    pub(crate) fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if block_states.len() != 2 || specs.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleFamily joint psi workspace expects 2 states, 2 specs, and 2 derivative block lists, got {} / {} / {}",
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        Ok(Some(Arc::new(
            GaussianLocationScaleExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
            )?,
        )))
    }

    /// Outer-aware joint ψ workspace with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `None`, this is byte-identical
    /// to `exact_newton_joint_psi_workspace`. When `Some`, the subsample is
    /// stored in the workspace and forwarded into every per-row weight array
    /// produced by `gaussian_joint_psi_firstweights`,
    /// `gaussian_joint_psisecondweights`, and
    /// `gaussian_joint_psi_mixed_driftweights`: each sampled row's
    /// contribution is multiplied by `WeightedOuterRow.weight = 1/π_i` and
    /// non-sampled rows are zeroed. Every downstream assembly
    /// (`gaussian_joint_psi*_fromweights`, `weighted_crossprod_psi_maps`,
    /// `xt_diag_*_dense`,
    /// `build_two_block_custom_family_joint_psi_operator_from_actions`) is
    /// row-linear in these arrays via `Xᵀ diag(W) Y`, so the resulting
    /// second-order ψ Hessian and ψ-Hessian directional derivative are
    /// unbiased Horvitz–Thompson estimators of the full-data quantities.
    /// Inner-PIRLS and final-covariance paths never install the option.
    pub(crate) fn exact_newton_joint_psi_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if block_states.len() != 2 || specs.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleFamily joint psi workspace expects 2 states, 2 specs, and 2 derivative block lists, got {} / {} / {}",
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        Ok(Some(Arc::new(
            GaussianLocationScaleExactNewtonJointPsiWorkspace::new_with_subsample(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
                options.outer_score_subsample.clone(),
            )?,
        )))
    }

    pub(crate) fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let workspace = GaussianLocationScaleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            xmu.into_owned(),
            x_ls.into_owned(),
        )?;
        Ok(Some(Arc::new(workspace)))
    }

    /// Outer-aware joint-Hessian workspace with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `None`, this is byte-identical
    /// to `exact_newton_joint_hessian_workspace`. When `Some`, the precomputed
    /// per-row coefficient arrays (`coeff_mm`, `coeff_ml`, `coeff_ll`) — which
    /// every downstream assembly (`hessian_dense`, `hessian_matvec`,
    /// `hessian_diagonal`) consumes row-linearly via `Xᵀ diag(W) X` — are
    /// replaced by a Horvitz–Thompson mask: each sampled row's coefficient is
    /// multiplied by `WeightedOuterRow.weight` (the inverse-inclusion factor
    /// 1/π_i; uniform or stratified sampling both supported), and non-sampled
    /// rows are zeroed. The resulting joint Hessian is an unbiased estimator
    /// of the full-data joint Hessian. Inner PIRLS never installs the option,
    /// so the inner solve continues to consume the exact full-data Hessian.
    pub(crate) fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let mut workspace = GaussianLocationScaleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            xmu.into_owned(),
            x_ls.into_owned(),
        )?;
        if let Some(subsample) = options.outer_score_subsample.as_ref() {
            workspace.apply_outer_subsample(subsample.rows.as_ref());
        }
        Ok(Some(Arc::new(workspace)))
    }

    pub(crate) fn inner_coefficient_hessian_hvp_available(
        &self,
        specs: &[ParameterBlockSpec],
    ) -> bool {
        // The Gaussian location-scale workspace is returned by
        // `exact_newton_joint_hessian_workspace` whenever
        // `exact_joint_dense_block_designs` succeeds, which itself depends on
        // both block designs being present. This is only a β-space operator
        // capability; outer θθ Hessian availability is declared separately.
        self.exact_joint_supported()
            && matches!(
                self.exact_joint_dense_block_designs(Some(specs)),
                Ok(Some(_))
            )
    }

    /// Outer-derivative policy: declare HT-subsample capability.
    ///
    /// GaussianLocationScaleFamily overrides
    /// `log_likelihood_only_with_options`,
    /// `exact_newton_joint_hessian_workspace_with_options`, and
    /// `exact_newton_joint_psi_workspace_with_options` to consume
    /// `options.outer_score_subsample` with per-row Horvitz–Thompson weights
    /// (each sampled row's contribution is multiplied by
    /// `WeightedOuterRow.weight = 1/π_i`; non-sampled rows are zeroed),
    /// yielding unbiased estimators of the full-data log-likelihood, joint
    /// Hessian, and second-order ψ Hessian / ψ-Hessian directional
    /// derivative. The ψ-workspace masking happens inside
    /// `apply_ht_mask_first`, `apply_ht_mask_second`, and
    /// `apply_ht_mask_mixed` on the `GaussianJointPsi{First,Second,
    /// MixedDrift}Weights` per-row arrays, immediately after the row-scalar
    /// reductions and before the row-linear `weighted_crossprod_psi_maps` /
    /// `xt_diag_*_dense` assemblies, so the masked outputs remain unbiased.
    /// First-order ψ terms remain full-data exact (= trivially unbiased), so
    /// the total outer score is still unbiased. Inner-PIRLS and final-
    /// covariance paths never install the option, so they continue to
    /// consume the exact full-data quantities.
    pub(crate) fn outer_derivative_subsample_capable(&self) -> bool {
        true
    }
}

impl CustomFamilyGenerative for GaussianLocationScaleFamily {
    pub(crate) fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let mu = block_states[Self::BLOCK_MU].eta.clone();
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let sigma = gamlss_rowwise_map(eta_log_sigma.len(), |i| {
            logb_sigma_from_eta_scalar(eta_log_sigma[i])
        });
        Ok(GenerativeSpec {
            mean: mu,
            noise: NoiseModel::Gaussian { sigma },
        })
    }
}

/// One channel of a `RowCoeffOperator`: a row-major `Arc<Array2<f64>>`
/// design matrix indexed by row coefficient pairs. Channels with the same
/// `block` value contribute their `X^T r` outputs into the same coefficient
/// block of the joint vector (e.g. wiggle's basis B and basis_d1 are two
/// channels that both contribute to the wiggle output block).
pub(crate) struct RowCoeffChannel {
    block: usize,
    design: Arc<Array2<f64>>,
}

/// Symmetric pair coefficients `c_{ab}` for `a ≤ b`. The operator adds
/// `X_a^T diag(c_{ab}) X_b` to block `block_a`'s output and the transpose
/// contribution `X_b^T diag(c_{ab}) X_a` to block `block_b` when `a != b`.
pub(crate) struct RowCoeffPair {
    a: usize,
    b: usize,
    coeff: Array1<f64>,
}

/// Pooled per-call scratch for `RowCoeffOperator::mul_vec`. Each call
/// pops a buffer set; if the pool is empty (parallel callers exhausted
/// it) we allocate fresh — the alloc is amortized as concurrent callers
/// recycle. The pool's `Mutex` is taken only for `pop`/`push` (constant
/// time), never during the matmul.
///
/// **Invariant**: every buffer in `pool[k].u[ch]` and `pool[k].r[ch]` has
/// length `nrows`. `mul_vec` overwrites `u` via `fast_av_into` and
/// zeroes-then-accumulates `r`, leaving both buffers in any state on
/// return — callers must not depend on residual content.
pub(crate) struct RowCoeffScratch {
    u: Vec<Array1<f64>>,
    r: Vec<Array1<f64>>,
}

/// Matrix-free operator for two-block-style joint-Hessian directional
/// derivatives that decompose as `H = sum_{a,b} X_a^T diag(c_{ab}) X_b`
/// with each `X_a` an `n × p_a` design and `c_{ab}` an `n` row coefficient
/// vector. `mul_vec` applies the operator in O(n · sum_a p_a) per call,
/// reusing pre-sized scratch buffers for `u`, `r` from a small lock-pool
/// so concurrent `mul_vec` callers do not serialize on the same scratch.
///
/// `block_offsets` gives the starting column of each output block; the
/// operator dimension is the sum of all block widths. Each channel's
/// `mul_vec` contribution is added into the slice for its output block.
pub(crate) struct RowCoeffOperator {
    channels: Vec<RowCoeffChannel>,
    block_offsets: Vec<usize>,
    block_widths: Vec<usize>,
    dim: usize,
    pair_coeffs: Vec<RowCoeffPair>,
    nrows: usize,
    scratch_pool: std::sync::Mutex<Vec<RowCoeffScratch>>,
}

impl RowCoeffOperator {
    /// One-line constructor for the standard (channels, pair-coeffs)
    /// recipe used by every GAMLSS LS workspace: pass the block widths,
    /// the channel list as `(block_id, design)` tuples, and the pair
    /// list as `(a, b, coeff)` tuples. Pre-allocates one scratch in the
    /// pool so the first warm `mul_vec` call skips allocation.
    pub(crate) fn from_directions(
        block_widths: Vec<usize>,
        channels: Vec<(usize, Arc<Array2<f64>>)>,
        pairs: Vec<(usize, usize, Array1<f64>)>,
        nrows: usize,
    ) -> Self {
        let channels: Vec<RowCoeffChannel> = channels
            .into_iter()
            .map(|(block, design)| RowCoeffChannel { block, design })
            .collect();
        let pair_coeffs: Vec<RowCoeffPair> = pairs
            .into_iter()
            .map(|(a, b, coeff)| RowCoeffPair { a, b, coeff })
            .collect();
        let mut block_offsets = Vec::with_capacity(block_widths.len());
        let mut acc = 0;
        for w in &block_widths {
            block_offsets.push(acc);
            acc += *w;
        }
        let n_ch = channels.len();
        let initial = RowCoeffScratch {
            u: (0..n_ch).map(|_| Array1::<f64>::zeros(nrows)).collect(),
            r: (0..n_ch).map(|_| Array1::<f64>::zeros(nrows)).collect(),
        };
        Self {
            channels,
            block_offsets,
            block_widths,
            dim: acc,
            pair_coeffs,
            nrows,
            scratch_pool: std::sync::Mutex::new(vec![initial]),
        }
    }

    pub(crate) fn acquire_scratch(&self) -> RowCoeffScratch {
        self.scratch_pool
            .lock()
            .expect("RowCoeffOperator scratch pool poisoned")
            .pop()
            .unwrap_or_else(|| {
                let n_ch = self.channels.len();
                RowCoeffScratch {
                    u: (0..n_ch)
                        .map(|_| Array1::<f64>::zeros(self.nrows))
                        .collect(),
                    r: (0..n_ch)
                        .map(|_| Array1::<f64>::zeros(self.nrows))
                        .collect(),
                }
            })
    }

    pub(crate) fn release_scratch(&self, scratch: RowCoeffScratch) {
        self.scratch_pool
            .lock()
            .expect("RowCoeffOperator scratch pool poisoned")
            .push(scratch);
    }

    pub(crate) fn projected_trace(&self, factor: &Array2<f64>) -> f64 {
        let grams = self.projected_pair_gram_table(factor);
        self.trace_from_pair_gram_table(grams.view())
    }

    pub(crate) fn projected_pair_gram_cache_id(&self) -> usize {
        let mut hasher = DefaultHasher::new();
        "RowCoeffOperator::projected_pair_gram_table".hash(&mut hasher);
        self.nrows.hash(&mut hasher);
        self.dim.hash(&mut hasher);
        self.block_widths.hash(&mut hasher);
        self.block_offsets.hash(&mut hasher);
        self.channels.len().hash(&mut hasher);
        self.pair_coeffs.len().hash(&mut hasher);
        for (idx, ch) in self.channels.iter().enumerate() {
            idx.hash(&mut hasher);
            (Arc::as_ptr(&ch.design) as usize).hash(&mut hasher);
            ch.block.hash(&mut hasher);
            ch.design.nrows().hash(&mut hasher);
            ch.design.ncols().hash(&mut hasher);
            self.block_widths[ch.block].hash(&mut hasher);
        }
        for (idx, pair) in self.pair_coeffs.iter().enumerate() {
            idx.hash(&mut hasher);
            pair.a.hash(&mut hasher);
            pair.b.hash(&mut hasher);
        }
        hasher.finish() as usize
    }

    pub(crate) fn projected_pair_gram_table(&self, factor: &Array2<f64>) -> Array2<f64> {
        assert_eq!(
            factor.nrows(),
            self.dim,
            "row-coefficient cached projected trace factor row mismatch: factor rows={} but dim={}",
            factor.nrows(),
            self.dim
        );
        let rank = factor.ncols();
        let pair_count = self.pair_coeffs.len();
        if self.nrows == 0 || rank == 0 || pair_count == 0 {
            return Array2::<f64>::zeros((self.nrows, pair_count));
        }
        let rows_per_chunk =
            gamlss_projected_trace_chunk_rows(rank, self.channels.len(), pair_count)
                .min(self.nrows.max(1));
        let mut grams = Array2::<f64>::zeros((self.nrows, pair_count));
        let fill_chunk = |start: usize, mut out_chunk: ndarray::ArrayViewMut2<'_, f64>| {
            let end = (start + rows_per_chunk).min(self.nrows);
            let rows = start..end;
            let mut projected: Vec<Array2<f64>> = Vec::with_capacity(self.channels.len());
            for ch in &self.channels {
                let block_start = self.block_offsets[ch.block];
                let width = self.block_widths[ch.block];
                let design_chunk = ch.design.slice(s![rows.clone(), ..]);
                let factor_block = factor.slice(s![block_start..block_start + width, ..]);
                projected.push(fast_ab(&design_chunk, &factor_block));
            }
            for (pair_idx, pair) in self.pair_coeffs.iter().enumerate() {
                let u_a = &projected[pair.a];
                let u_b = &projected[pair.b];
                for local_i in 0..u_a.nrows() {
                    let mut value = 0.0;
                    for col in 0..rank {
                        value += u_a[[local_i, col]] * u_b[[local_i, col]];
                    }
                    out_chunk[[local_i, pair_idx]] = value;
                }
            }
        };
        if rayon::current_thread_index().is_none() && self.nrows > rows_per_chunk {
            grams
                .axis_chunks_iter_mut(Axis(0), rows_per_chunk)
                .into_par_iter()
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    fill_chunk(chunk_idx * rows_per_chunk, out_chunk)
                });
        } else {
            for start in (0..self.nrows).step_by(rows_per_chunk) {
                let end = (start + rows_per_chunk).min(self.nrows);
                let out_chunk = grams.slice_mut(s![start..end, ..]);
                fill_chunk(start, out_chunk);
            }
        }
        grams
    }

    pub(crate) fn trace_from_pair_gram_table(&self, grams: ArrayView2<'_, f64>) -> f64 {
        assert_eq!(grams.nrows(), self.nrows);
        assert_eq!(grams.ncols(), self.pair_coeffs.len());
        let mut trace = 0.0;
        for i in 0..self.nrows {
            for (pair_idx, pair) in self.pair_coeffs.iter().enumerate() {
                let multiplier = if pair.a == pair.b { 1.0 } else { 2.0 };
                trace += multiplier * pair.coeff[i] * grams[[i, pair_idx]];
            }
        }
        trace
    }
}

impl crate::solver::estimate::reml::unified::HyperOperator for RowCoeffOperator {
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    pub(crate) fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.dim);
        let mut scratch = self.acquire_scratch();
        let RowCoeffScratch { u, r } = &mut scratch;

        // 1) u_a = X_a · v[block_a slice]. `fast_av_into` writes directly
        //    into the pre-sized scratch buffer — no per-call n-sized
        //    allocation.
        for (k, ch) in self.channels.iter().enumerate() {
            let start = self.block_offsets[ch.block];
            let width = self.block_widths[ch.block];
            assert_eq!(ch.design.ncols(), width);
            let v_slice = v.slice(s![start..start + width]);
            crate::faer_ndarray::fast_av_into(ch.design.as_ref(), &v_slice, &mut u[k]);
        }

        // 2) r_a = sum_b c_{ab} ⊙ u_b. Zero-then-accumulate; pair coeffs
        //    contribute symmetrically when `a != b`.
        for slot in r.iter_mut() {
            slot.fill(0.0);
        }
        for pair in &self.pair_coeffs {
            let a = pair.a;
            let b = pair.b;
            let coeff = pair
                .coeff
                .as_slice()
                .expect("RowCoeffOperator pair coeff must be contiguous");
            // r[a] += coeff ⊙ u[b]; if a != b also r[b] += coeff ⊙ u[a].
            // Split the borrow so r[a] and r[b] (or u[a] and u[b]) can be
            // accessed simultaneously when a != b.
            if a == b {
                let u_a = u[a]
                    .as_slice()
                    .expect("RowCoeffOperator u must be contiguous");
                let r_a = r[a]
                    .as_slice_mut()
                    .expect("RowCoeffOperator r must be contiguous");
                use rayon::prelude::*;
                r_a.par_iter_mut()
                    .zip(coeff.par_iter())
                    .zip(u_a.par_iter())
                    .for_each(|((r, c), u)| *r += c * u);
            } else {
                let (r_a_slice, r_b_slice) = if a < b {
                    let (left, right) = r.split_at_mut(b);
                    (
                        left[a].as_slice_mut().expect("contiguous"),
                        right[0].as_slice_mut().expect("contiguous"),
                    )
                } else {
                    let (left, right) = r.split_at_mut(a);
                    (
                        right[0].as_slice_mut().expect("contiguous"),
                        left[b].as_slice_mut().expect("contiguous"),
                    )
                };
                let u_a = u[a].as_slice().expect("contiguous");
                let u_b = u[b].as_slice().expect("contiguous");
                use rayon::prelude::*;
                r_a_slice
                    .par_iter_mut()
                    .zip(r_b_slice.par_iter_mut())
                    .zip(coeff.par_iter())
                    .zip(u_a.par_iter())
                    .zip(u_b.par_iter())
                    .for_each(|((((ra, rb), c), ua), ub)| {
                        *ra += c * ub;
                        *rb += c * ua;
                    });
            }
        }

        // 3) Output[block] += X_a^T r_a per channel. Single output alloc.
        let mut out = Array1::<f64>::zeros(self.dim);
        for (k, ch) in self.channels.iter().enumerate() {
            let start = self.block_offsets[ch.block];
            let width = self.block_widths[ch.block];
            let mut block = out.slice_mut(s![start..start + width]);
            // Atv into a temporary, then accumulate; `fast_atv` allocates
            // a `width`-sized array, which is bounded and small relative
            // to the n-sized u/r buffers we already reuse.
            let contrib = fast_atv(ch.design.as_ref(), &r[k]);
            block += &contrib;
        }
        self.release_scratch(scratch);
        out
    }

    pub(crate) fn mul_basis_columns_into(
        &self,
        start: usize,
        mut out: ndarray::ArrayViewMut2<'_, f64>,
    ) {
        let cols = out.ncols();
        assert!(start + cols <= self.dim);
        let mut basis = Array1::<f64>::zeros(self.dim);
        for local_col in 0..cols {
            let global_col = start + local_col;
            basis[global_col] = 1.0;
            let col = self.mul_vec(&basis);
            out.column_mut(local_col).assign(&col);
            basis[global_col] = 0.0;
        }
    }

    pub(crate) fn to_dense(&self) -> Array2<f64> {
        // Build by basis-vector probing — small-K materialization path.
        let mut out = Array2::<f64>::zeros((self.dim, self.dim));
        self.mul_basis_columns_into(0, out.view_mut());
        out
    }

    pub(crate) fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        self.projected_trace(factor)
    }

    pub(crate) fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &crate::solver::estimate::reml::unified::ProjectedFactorCache,
    ) -> f64 {
        let key = crate::solver::estimate::reml::unified::ProjectedFactorKey::from_factor_view(
            self.projected_pair_gram_cache_id(),
            factor.view(),
        );
        let grams = cache.get_or_insert_with(key, || self.projected_pair_gram_table(factor));
        self.trace_from_pair_gram_table(grams.view())
    }

    pub(crate) fn is_implicit(&self) -> bool {
        true
    }
}

/// Two-block row-coefficient operator backed by `DesignMatrix`.
///
/// This is the operator-form counterpart to `DesignTwoBlockRowCoeffOperator`'s
/// old dense-array storage: it must keep the realized block designs lazy all
/// the way through `Xv` and `X^T r`. Do not cache `Array2` snapshots here;
/// `NoDensifyOperator` regression tests rely on this type to panic if a future
/// change materializes spec-backed designs.
pub(crate) struct DesignTwoBlockRowCoeffOperator {
    x_a: DesignMatrix,
    x_b: DesignMatrix,
    c_aa: Arc<Array1<f64>>,
    c_ab: Arc<Array1<f64>>,
    c_bb: Arc<Array1<f64>>,
    dim: usize,
    nrows: usize,
    pa: usize,
}

impl crate::solver::estimate::reml::unified::HyperOperator for DesignTwoBlockRowCoeffOperator {
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    pub(crate) fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.dim);
        let v_a = v.slice(s![0..self.pa]);
        let v_b = v.slice(s![self.pa..self.dim]);
        let u_a = self.x_a.matrixvectormultiply(&v_a.to_owned());
        let u_b = self.x_b.matrixvectormultiply(&v_b.to_owned());
        assert_eq!(u_a.len(), self.nrows);
        assert_eq!(u_b.len(), self.nrows);
        let r_a = self.c_aa.as_ref() * &u_a + self.c_ab.as_ref() * &u_b;
        let r_b = self.c_ab.as_ref() * &u_a + self.c_bb.as_ref() * &u_b;
        let out_a = self.x_a.transpose_vector_multiply(&r_a);
        let out_b = self.x_b.transpose_vector_multiply(&r_b);
        let mut out = Array1::<f64>::zeros(self.dim);
        out.slice_mut(s![0..self.pa]).assign(&out_a);
        out.slice_mut(s![self.pa..self.dim]).assign(&out_b);
        out
    }

    pub(crate) fn mul_basis_columns_into(
        &self,
        start: usize,
        mut out: ndarray::ArrayViewMut2<'_, f64>,
    ) {
        let cols = out.ncols();
        assert!(start + cols <= self.dim);
        let mut basis = Array1::<f64>::zeros(self.dim);
        for local_col in 0..cols {
            let global_col = start + local_col;
            basis[global_col] = 1.0;
            let col = self.mul_vec(&basis);
            out.column_mut(local_col).assign(&col);
            basis[global_col] = 0.0;
        }
    }

    pub(crate) fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.dim, self.dim));
        self.mul_basis_columns_into(0, out.view_mut());
        out
    }

    pub(crate) fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        // For the two-block row-coefficient operator
        //   B v = [X_a^T (c_aa·u_a + c_ab·u_b),  X_b^T (c_ab·u_a + c_bb·u_b)]
        // with u_a = X_a v_a, u_b = X_b v_b, the column-wise quadratic form is
        //   F[:,k]^T B F[:,k] = u_a^T r_a + u_b^T r_b
        //                    = Σ_i (c_aa[i] u_a[i]² + 2 c_ab[i] u_a[i] u_b[i]
        //                            + c_bb[i] u_b[i]²)
        // so the projected trace never needs the X^T r step that the default
        // mul_vec path computes, and the per-row coefficients fold the K
        // columns into a single weighted sum once U_a, U_b are formed.
        let grams = self.projected_row_gram_triples(factor);
        self.trace_from_row_gram_triples(grams.view())
    }

    pub(crate) fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &crate::solver::estimate::reml::unified::ProjectedFactorCache,
    ) -> f64 {
        // Validate the factor row count up front. Without this, a caller that
        // hands in a factor whose row count does not equal the joint p slips
        // into the per-column `mul_vec` slicing where a `assert_eq!`
        // panics with the generic `left/right` message — that loses the
        // operator identity and the (pa, pb) split which is the only useful
        // diagnostic when the trace caller's own dimension bookkeeping is
        // off. Validate at the operator boundary so the panic localises the
        // caller, and so this contract is enforced in release builds too
        // (the inner `assert_eq!` is a debug-only safety net).
        assert_eq!(
            factor.nrows(),
            self.dim,
            "two-block cached projected trace factor row mismatch: factor rows={} \
             but joint p={} (pa={}, pb={})",
            factor.nrows(),
            self.dim,
            self.pa,
            self.dim - self.pa,
        );
        let key = crate::solver::estimate::reml::unified::ProjectedFactorKey::from_factor_view(
            self.projected_row_gram_cache_id(),
            factor.view(),
        );
        let grams = cache.get_or_insert_with(key, || self.projected_row_gram_triples(factor));
        self.trace_from_row_gram_triples(grams.view())
    }

    pub(crate) fn is_implicit(&self) -> bool {
        true
    }
}

impl DesignTwoBlockRowCoeffOperator {
    pub(crate) fn design_cache_token(design: &DesignMatrix) -> usize {
        match design {
            DesignMatrix::Dense(DenseDesignMatrix::Materialized(matrix)) => {
                Arc::as_ptr(matrix) as usize
            }
            DesignMatrix::Dense(DenseDesignMatrix::Lazy(op)) => {
                Arc::as_ptr(op) as *const () as usize
            }
            DesignMatrix::Sparse(sparse) => sparse as *const _ as usize,
        }
    }

    pub(crate) fn projected_row_gram_cache_id(&self) -> usize {
        let mut hasher = DefaultHasher::new();
        "DesignTwoBlockRowCoeffOperator::projected_row_gram_triples".hash(&mut hasher);
        Self::design_cache_token(&self.x_a).hash(&mut hasher);
        Self::design_cache_token(&self.x_b).hash(&mut hasher);
        self.nrows.hash(&mut hasher);
        self.pa.hash(&mut hasher);
        self.dim.hash(&mut hasher);
        hasher.finish() as usize
    }

    pub(crate) fn projected_row_gram_triples(&self, factor: &Array2<f64>) -> Array2<f64> {
        assert_eq!(
            factor.nrows(),
            self.dim,
            "two-block cached projected trace factor row mismatch: factor rows={} \
             but joint p={} (pa={}, pb={})",
            factor.nrows(),
            self.dim,
            self.pa,
            self.dim - self.pa,
        );
        let rank = factor.ncols();
        let mut grams = Array2::<f64>::zeros((self.nrows, 3));
        if self.nrows == 0 || rank == 0 {
            return grams;
        }
        let rows_per_chunk = gamlss_projected_trace_chunk_rows(rank, 2, 3).min(self.nrows.max(1));
        let f_a = factor.slice(s![0..self.pa, ..]);
        let f_b = factor.slice(s![self.pa..self.dim, ..]);
        let fill_chunk = |start: usize, mut out_chunk: ndarray::ArrayViewMut2<'_, f64>| {
            let end = (start + rows_per_chunk).min(self.nrows);
            let rows = start..end;
            let x_a_chunk = self
                .x_a
                .try_row_chunk(rows.clone())
                .expect("two-block projected trace x_a row chunk materialization failed");
            let x_b_chunk = self
                .x_b
                .try_row_chunk(rows.clone())
                .expect("two-block projected trace x_b row chunk materialization failed");
            let u_a = fast_ab(&x_a_chunk, &f_a);
            let u_b = fast_ab(&x_b_chunk, &f_b);
            for local_i in 0..u_a.nrows() {
                let mut aa = 0.0;
                let mut ab = 0.0;
                let mut bb = 0.0;
                for col in 0..rank {
                    let a = u_a[[local_i, col]];
                    let b = u_b[[local_i, col]];
                    aa += a * a;
                    ab += a * b;
                    bb += b * b;
                }
                out_chunk[[local_i, 0]] = aa;
                out_chunk[[local_i, 1]] = ab;
                out_chunk[[local_i, 2]] = bb;
            }
        };
        if rayon::current_thread_index().is_none() && self.nrows > rows_per_chunk {
            grams
                .axis_chunks_iter_mut(Axis(0), rows_per_chunk)
                .into_par_iter()
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    fill_chunk(chunk_idx * rows_per_chunk, out_chunk)
                });
        } else {
            for start in (0..self.nrows).step_by(rows_per_chunk) {
                let end = (start + rows_per_chunk).min(self.nrows);
                let out_chunk = grams.slice_mut(s![start..end, ..]);
                fill_chunk(start, out_chunk);
            }
        }
        grams
    }

    pub(crate) fn trace_from_row_gram_triples(&self, grams: ArrayView2<'_, f64>) -> f64 {
        assert_eq!(grams.nrows(), self.nrows);
        assert_eq!(grams.ncols(), 3);
        let c_aa = self
            .c_aa
            .as_slice()
            .expect("c_aa is constructed contiguous");
        let c_ab = self
            .c_ab
            .as_slice()
            .expect("c_ab is constructed contiguous");
        let c_bb = self
            .c_bb
            .as_slice()
            .expect("c_bb is constructed contiguous");
        let mut trace = 0.0;
        for i in 0..self.nrows {
            trace +=
                c_aa[i] * grams[[i, 0]] + 2.0 * c_ab[i] * grams[[i, 1]] + c_bb[i] * grams[[i, 2]];
        }
        trace
    }
}

/// Matrix-free joint-Hessian operator for the two-block Gaussian
/// location-scale family. The dense Hessian decomposes as
///
///   H = [[X_mu^T diag(w) X_mu,    X_mu^T diag(cross) X_ls],
///        [X_ls^T diag(cross) X_mu, X_ls^T diag(scale) X_ls]],
///
/// with `cross = 0` and `scale = 2κ²a` — the block-diagonal Gaussian Fisher
/// (expected) information (μ ⊥ σ, #684; residual-free (log σ, log σ) block,
/// #566). This MUST match the dense `exact_newton_joint_hessian_from_designs`
/// curvature object exactly: the observed cross term `2κm` (mean-zero noise)
/// over-smooths the scale and is its Fisher expectation 0. The matvec applies
/// each block by a single design-matrix multiply on each side, so the cost
/// is Θ(n (p_mu + p_ls)) per `Hv` rather than Θ(n (p_mu + p_ls)²) to form
/// the dense matrix.
pub(crate) struct GaussianLocationScaleHessianWorkspace {
    family: GaussianLocationScaleFamily,
    block_states: Vec<ParameterBlockState>,
    xmu: Arc<Array2<f64>>,
    x_ls: Arc<Array2<f64>>,
    coeff_mm: Array1<f64>,
    coeff_ml: Array1<f64>,
    coeff_ll: Array1<f64>,
}

impl GaussianLocationScaleHessianWorkspace {
    pub(crate) fn new(
        family: GaussianLocationScaleFamily,
        block_states: Vec<ParameterBlockState>,
        xmu: Array2<f64>,
        x_ls: Array2<f64>,
    ) -> Result<Self, String> {
        let etamu = &block_states[GaussianLocationScaleFamily::BLOCK_MU].eta;
        let eta_ls = &block_states[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA].eta;
        let rows = family.get_or_compute_row_scalars(etamu, eta_ls)?;
        // Single source of truth shared with the dense
        // `exact_newton_joint_hessian_from_designs`: μ ⊥ σ ⇒ cross = 0 (#684),
        // (ls,ls) = 2κ²a (#566). Reading the same coefficients as the dense path
        // makes the cross-block drift that caused #684 structurally impossible.
        let (coeff_mm, coeff_ml, coeff_ll) = gaussian_locscale_fisher_joint_row_coeffs(&rows);
        Ok(Self {
            family,
            block_states,
            xmu: Arc::new(xmu),
            x_ls: Arc::new(x_ls),
            coeff_mm,
            coeff_ml,
            coeff_ll,
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
    /// via `Xᵀ diag(W) X`, the resulting joint-Hessian is an unbiased
    /// estimator of the full-data joint Hessian.
    pub(crate) fn apply_outer_subsample(
        &mut self,
        rows: &[crate::families::marginal_slope_shared::WeightedOuterRow],
    ) {
        let n = self.coeff_mm.len();
        let mut mask_mm = Array1::<f64>::zeros(n);
        let mut mask_ml = Array1::<f64>::zeros(n);
        let mut mask_ll = Array1::<f64>::zeros(n);
        for r in rows {
            let i = r.index;
            mask_mm[i] = self.coeff_mm[i] * r.weight;
            mask_ml[i] = self.coeff_ml[i] * r.weight;
            mask_ll[i] = self.coeff_ll[i] * r.weight;
        }
        self.coeff_mm = mask_mm;
        self.coeff_ml = mask_ml;
        self.coeff_ll = mask_ll;
    }
}

impl ExactNewtonJointHessianWorkspace for GaussianLocationScaleHessianWorkspace {
    pub(crate) fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        // Same Hv structure as `hessian_matvec`, but built once via 3 GEMMs
        // (`Xᵀ diag(W) X` per block) instead of letting
        // `MatrixFreeSpdOperator::materialize_dense_operator` reconstruct the
        // dense Hessian via `total` canonical-basis HVPs. At large scale
        // (n≈320k, p_total≈82) the canonical-basis path takes ~568s per κ-iter
        // while the dense build via fast_xt_diag_x/y is ~1s.
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let total = pmu + p_ls;
        let h_mm = xt_diag_x_dense(self.xmu.as_ref(), &self.coeff_mm)?;
        let h_ml = xt_diag_y_dense(self.xmu.as_ref(), &self.coeff_ml, self.x_ls.as_ref())?;
        let h_ll = xt_diag_x_dense(self.x_ls.as_ref(), &self.coeff_ll)?;
        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pmu, 0..pmu]).assign(&h_mm);
        h.slice_mut(s![0..pmu, pmu..total]).assign(&h_ml);
        h.slice_mut(s![pmu..total, pmu..total]).assign(&h_ll);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    pub(crate) fn hessian_matvec_available(&self) -> bool {
        true
    }

    pub(crate) fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let total = pmu + p_ls;
        if v.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScale matvec dimension mismatch: got {}, expected {}",
                    v.len(),
                    total
                ),
            }
            .into());
        }
        let u_mu = fast_av(self.xmu.as_ref(), &v.slice(s![0..pmu]));
        let u_ls = fast_av(self.x_ls.as_ref(), &v.slice(s![pmu..total]));
        let r_mu = &self.coeff_mm * &u_mu + &self.coeff_ml * &u_ls;
        let r_ls = &self.coeff_ml * &u_mu + &self.coeff_ll * &u_ls;
        let out_mu = fast_atv(self.xmu.as_ref(), &r_mu);
        let out_ls = fast_atv(self.x_ls.as_ref(), &r_ls);
        let mut out = Array1::<f64>::zeros(total);
        out.slice_mut(s![0..pmu]).assign(&out_mu);
        out.slice_mut(s![pmu..total]).assign(&out_ls);
        Ok(Some(out))
    }

    pub(crate) fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let total = pmu + p_ls;
        // Per-column reduction is independent; parallelize across columns.
        let diag_mu: Vec<f64> = (0..pmu)
            .into_par_iter()
            .map(|j| {
                let col = self.xmu.column(j);
                col.iter()
                    .zip(self.coeff_mm.iter())
                    .map(|(&v, &c)| c * v * v)
                    .sum()
            })
            .collect();
        let diag_ls: Vec<f64> = (0..p_ls)
            .into_par_iter()
            .map(|j| {
                let col = self.x_ls.column(j);
                col.iter()
                    .zip(self.coeff_ll.iter())
                    .map(|(&v, &c)| c * v * v)
                    .sum()
            })
            .collect();
        let mut diag = Array1::<f64>::zeros(total);
        for (j, v) in diag_mu.into_iter().enumerate() {
            diag[j] = v;
        }
        for (j, v) in diag_ls.into_iter().enumerate() {
            diag[pmu + j] = v;
        }
        Ok(Some(diag))
    }

    pub(crate) fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_from_designs(
                &self.block_states,
                &DenseOrOperator::Borrowed(self.xmu.as_ref()),
                &DenseOrOperator::Borrowed(self.x_ls.as_ref()),
                d_beta_flat,
            )
    }

    pub(crate) fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        let n = self.xmu.nrows();
        let pmu = self.xmu.ncols();
        let pls = self.x_ls.ncols();
        let total = pmu + pls;
        if d_beta_flat.len() != total {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "GaussianLocationScale dH operator: d_beta length {} != {}",
                    d_beta_flat.len(),
                    total
                ),
            }
            .into());
        }
        let etamu = &self.block_states[GaussianLocationScaleFamily::BLOCK_MU].eta;
        let eta_ls = &self.block_states[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA].eta;
        let rows = self.family.get_or_compute_row_scalars(etamu, eta_ls)?;
        let ximu = fast_av(self.xmu.as_ref(), &d_beta_flat.slice(s![0..pmu]));
        let xi_ls = fast_av(self.x_ls.as_ref(), &d_beta_flat.slice(s![pmu..total]));
        let directional = gaussian_joint_first_directionalweights(&rows, &ximu, &xi_ls);
        let c_mm = directional.0;
        let c_ll = directional.2;
        // Fisher cross block ≡ 0 (μ ⊥ σ; #684), so its directional derivative is
        // identically 0 — matching the dense
        // `exact_newton_joint_hessian_directional_derivative_from_designs`, which
        // likewise does not assemble `directional.1`.
        let c_ml = Array1::<f64>::zeros(c_mm.len());
        Ok(Some(Arc::new(make_two_block_row_coeff_operator(
            self.xmu.clone(),
            self.x_ls.clone(),
            c_mm,
            c_ml,
            c_ll,
            n,
        ))))
    }

    pub(crate) fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_from_designs(
                &self.block_states,
                &DenseOrOperator::Borrowed(self.xmu.as_ref()),
                &DenseOrOperator::Borrowed(self.x_ls.as_ref()),
                d_beta_u_flat,
                d_beta_v_flat,
            )
    }

    pub(crate) fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        let n = self.xmu.nrows();
        let pmu = self.xmu.ncols();
        let pls = self.x_ls.ncols();
        let total = pmu + pls;
        if d_beta_u.len() != total || d_beta_v.len() != total {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "GaussianLocationScale d2H operator: d_beta_{{u,v}} length {}/{} != {}",
                    d_beta_u.len(),
                    d_beta_v.len(),
                    total
                ),
            }
            .into());
        }
        let etamu = &self.block_states[GaussianLocationScaleFamily::BLOCK_MU].eta;
        let eta_ls = &self.block_states[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA].eta;
        let rows = self.family.get_or_compute_row_scalars(etamu, eta_ls)?;
        let ximu_u = fast_av(self.xmu.as_ref(), &d_beta_u.slice(s![0..pmu]));
        let xi_ls_u = fast_av(self.x_ls.as_ref(), &d_beta_u.slice(s![pmu..total]));
        let ximu_v = fast_av(self.xmu.as_ref(), &d_beta_v.slice(s![0..pmu]));
        let xi_ls_v = fast_av(self.x_ls.as_ref(), &d_beta_v.slice(s![pmu..total]));
        let directional =
            gaussian_jointsecond_directionalweights(&rows, &ximu_u, &xi_ls_u, &ximu_v, &xi_ls_v);
        let c_mm = directional.0;
        let c_ll = directional.2;
        // Fisher cross block ≡ 0 (μ ⊥ σ; #684); its second directional
        // derivative is identically 0 too — match the dense path (which does not
        // assemble `directional.1`).
        let c_ml = Array1::<f64>::zeros(c_mm.len());
        Ok(Some(Arc::new(make_two_block_row_coeff_operator(
            self.xmu.clone(),
            self.x_ls.clone(),
            c_mm,
            c_ml,
            c_ll,
            n,
        ))))
    }
}

/// Build a `RowCoeffOperator` for the standard two-block GAMLSS structure
/// with one design per block and three pair coefficients (a,a), (a,b), (b,b).
/// The resulting matrix mirrors the dense
/// `X_a^T diag(c_aa) X_a + X_a^T diag(c_ab) X_b + X_b^T diag(c_ab) X_a + X_b^T diag(c_bb) X_b`
/// assembly emitted by `gaussian_joint_hessian_from_designs` (Gaussian path)
/// and the `xt_diag_*` block writers (binomial path).
pub(crate) fn make_two_block_row_coeff_operator(
    x_a: Arc<Array2<f64>>,
    x_b: Arc<Array2<f64>>,
    c_aa: Array1<f64>,
    c_ab: Array1<f64>,
    c_bb: Array1<f64>,
    nrows: usize,
) -> RowCoeffOperator {
    let pa = x_a.ncols();
    let pb = x_b.ncols();
    RowCoeffOperator::from_directions(
        vec![pa, pb],
        vec![(0, x_a), (1, x_b)],
        vec![(0, 0, c_aa), (0, 1, c_ab), (1, 1, c_bb)],
        nrows,
    )
}

pub(crate) fn make_two_block_design_row_coeff_operator(
    x_a: DesignMatrix,
    x_b: DesignMatrix,
    c_aa: Arc<Array1<f64>>,
    c_ab: Arc<Array1<f64>>,
    c_bb: Arc<Array1<f64>>,
) -> Result<DesignTwoBlockRowCoeffOperator, String> {
    let nrows = x_a.nrows();
    if x_b.nrows() != nrows || c_aa.len() != nrows || c_ab.len() != nrows || c_bb.len() != nrows {
        return Err(GamlssError::DimensionMismatch { reason: format!(
            "two-block row coefficient operator dimension mismatch: rows a={}, b={}, coeffs={}/{}/{}",
            nrows,
            x_b.nrows(),
            c_aa.len(),
            c_ab.len(),
            c_bb.len()
        ) }.into());
    }
    let pa = x_a.ncols();
    let pb = x_b.ncols();
    Ok(DesignTwoBlockRowCoeffOperator {
        x_a,
        x_b,
        c_aa,
        c_ab,
        c_bb,
        dim: pa + pb,
        nrows,
        pa,
    })
}

pub(crate) struct GaussianLocationScaleWiggleGeometry {
    basis: Array2<f64>,
    basis_d1: Array2<f64>,
    basis_d2: Array2<f64>,
    basis_d3: Array2<f64>,
    dq_dq0: Array1<f64>,
    d2q_dq02: Array1<f64>,
    d3q_dq03: Array1<f64>,
    d4q_dq04: Array1<f64>,
}

/// Per-row pieces of the 3-block Gaussian location-scale-wiggle joint
/// Hessian. Both the dense path and the matrix-free workspace share these
/// row coefficients; only the assembly differs.
pub(crate) struct GaussianLocationScaleWiggleHessianRowPieces {
    coeff_mm: Array1<f64>,
    coeff_ml: Array1<f64>,
    coeff_ll: Array1<f64>,
    coeff_mw_b: Array1<f64>,
    coeff_mw_d: Array1<f64>,
    coeff_lw_b: Array1<f64>,
    coeff_ww: Array1<f64>,
    basis: Array2<f64>,
    basis_d1: Array2<f64>,
}

impl GaussianLocationScaleWiggleHessianRowPieces {
    pub(crate) fn assemble_dense(
        &self,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let h_mm = xt_diag_x_dense(xmu, &self.coeff_mm)?;
        let h_ml = xt_diag_y_dense(xmu, &self.coeff_ml, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &self.coeff_ll)?;
        let h_mw = xt_diag_y_dense(xmu, &self.coeff_mw_b, &self.basis)?
            + &xt_diag_y_dense(xmu, &self.coeff_mw_d, &self.basis_d1)?;
        let h_lw = xt_diag_y_dense(x_ls, &self.coeff_lw_b, &self.basis)?;
        let h_ww = xt_diag_x_dense(&self.basis, &self.coeff_ww)?;
        Ok(gaussian_pack_wiggle_joint_symmetrichessian(
            &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
        ))
    }
}

pub struct GaussianLocationScaleWiggleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub mu_design: Option<DesignMatrix>,
    pub log_sigma_design: Option<DesignMatrix>,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    /// Resource policy threaded into PsiDesignMap construction (and any other
    /// per-call materialization decision) made during exact-Newton joint psi
    /// derivative evaluation. Defaults to `ResourcePolicy::default_library()`
    /// when the family is built without an explicit policy.
    pub policy: crate::resource::ResourcePolicy,
    cached_row_scalars:
        std::sync::RwLock<Option<(f64, f64, f64, f64, f64, f64, Arc<GaussianJointRowScalars>)>>,
}

impl Clone for GaussianLocationScaleWiggleFamily {
    pub(crate) fn clone(&self) -> Self {
        Self {
            y: self.y.clone(),
            weights: self.weights.clone(),
            mu_design: self.mu_design.clone(),
            log_sigma_design: self.log_sigma_design.clone(),
            wiggle_knots: self.wiggle_knots.clone(),
            wiggle_degree: self.wiggle_degree,
            policy: self.policy.clone(),
            cached_row_scalars: std::sync::RwLock::new(
                self.cached_row_scalars
                    .read()
                    .expect("lock poisoned")
                    .clone(),
            ),
        }
    }
}

impl GaussianLocationScaleWiggleFamily {
    pub const BLOCK_MU: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;
    pub const BLOCK_WIGGLE: usize = 2;

    pub fn parameternames() -> &'static [&'static str] {
        &["mu", "log_sigma", "wiggle"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[
            ParameterLink::Identity,
            ParameterLink::Log,
            ParameterLink::Wiggle,
        ]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "gaussian_location_scalewiggle",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }

    pub(crate) fn exact_joint_supported(&self) -> bool {
        self.mu_design.is_some() && self.log_sigma_design.is_some()
    }

    pub(crate) fn wiggle_basiswith_options(
        &self,
        q0: ArrayView1<'_, f64>,
        options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            options.derivative_order,
        )
    }

    pub(crate) fn wiggle_design(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        self.wiggle_basiswith_options(q0, BasisOptions::value())
    }

    pub(crate) fn wiggle_dq_dq0(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d1 = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        if d1.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d1.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d1.dot(&beta_link_wiggle) + 1.0)
    }

    pub(crate) fn wiggle_d2q_dq02(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d2 = self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        if d2.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle second-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d2.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d2.dot(&beta_link_wiggle))
    }

    pub(crate) fn wiggle_d3basis_constrained(
        &self,
        q0: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(q0, &self.wiggle_knots, self.wiggle_degree, 3)
    }

    pub(crate) fn wiggle_d3q_dq03(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d3 = self.wiggle_d3basis_constrained(q0)?;
        if d3.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle third-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d3.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d3.dot(&beta_link_wiggle))
    }

    pub(crate) fn wiggle_d4q_dq04(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d4 = monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            4,
        )?;
        if d4.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle fourth-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d4.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d4.dot(&beta_link_wiggle))
    }

    pub(crate) fn wiggle_geometry(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<GaussianLocationScaleWiggleGeometry, String> {
        let basis = self.wiggle_design(q0)?;
        let basis_d1 = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        let basis_d2 = self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        let basis_d3 = self.wiggle_d3basis_constrained(q0)?;
        let dq_dq0 = self.wiggle_dq_dq0(q0, beta_link_wiggle)?;
        let d2q_dq02 = self.wiggle_d2q_dq02(q0, beta_link_wiggle)?;
        let d3q_dq03 = self.wiggle_d3q_dq03(q0, beta_link_wiggle)?;
        let d4q_dq04 = self.wiggle_d4q_dq04(q0, beta_link_wiggle)?;
        Ok(GaussianLocationScaleWiggleGeometry {
            basis,
            basis_d1,
            basis_d2,
            basis_d3,
            dq_dq0,
            d2q_dq02,
            d3q_dq03,
            d4q_dq04,
        })
    }

    pub(crate) fn get_or_compute_row_scalars(
        &self,
        q: &Array1<f64>,
        eta_ls: &Array1<f64>,
    ) -> Result<Arc<GaussianJointRowScalars>, String> {
        Ok(Arc::new(gaussian_jointrow_scalars(
            &self.y,
            q,
            eta_ls,
            &self.weights,
        )?))
    }

    pub(crate) fn dense_block_designs(
        &self,
    ) -> Result<(Cow<'_, Array2<f64>>, Cow<'_, Array2<f64>>), String> {
        dense_locscale_block_designs_cached(
            self.mu_design.as_ref(),
            self.log_sigma_design.as_ref(),
            "GaussianLocationScaleWiggleFamily",
            "GaussianLocationScaleWiggle",
            "mu",
            &self.policy.material_policy(),
        )
    }
    pub(crate) fn dense_block_designs_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>), String> {
        dense_locscale_block_designs_fromspecs(
            specs,
            3,
            "GaussianLocationScaleWiggleFamily",
            "GaussianLocationScaleWiggle",
            Self::BLOCK_MU,
            Self::BLOCK_LOG_SIGMA,
            "mu",
            &self.policy.material_policy(),
        )
    }

    pub(crate) fn exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        if self.exact_joint_supported() {
            return self.dense_block_designs().map(Some);
        }
        if let Some(specs) = specs {
            return self.dense_block_designs_fromspecs(specs).map(Some);
        }
        Ok(None)
    }

    /// Build the [`BlockEffectiveJacobian`] for block `block_idx`.
    ///
    /// The wiggle block (block 2) modulates the inverse link nonlinearly and
    /// does not contribute a linear additive term to any output η; its
    /// Jacobian is an `(2 * n, p_wiggle)` zero matrix.
    ///
    /// - block 0 (mu):        output 0 = design rows, output 1 = zeros
    /// - block 1 (log_sigma): output 0 = zeros, output 1 = design rows
    /// - block 2 (wiggle):    all zeros (nonlinear link modulation)
    pub fn block_effective_jacobian(
        specs: &[ParameterBlockSpec],
        block_idx: usize,
    ) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
        crate::util::block_jacobian::AdditiveWiggleBlockLayout {
            family: "GaussianLocationScaleWiggleFamily",
            n_outputs: 2,
            additive_blocks: &[Self::BLOCK_MU, Self::BLOCK_LOG_SIGMA],
            wiggle_block: Some(Self::BLOCK_WIGGLE),
        }
        .block_effective_jacobian(specs, block_idx)
    }
}

/// Row-coefficient bundle for the GLS Wiggle joint second directional
/// derivative, shared by the matrix-free operator and the dense
/// `_from_designs` assemblies. Holds exactly the quantities both consumers
/// read downstream of the (identical) coefficient computation.
pub(crate) struct GlsWiggleSecondDirCoeffs {
    coeff_mm_uv: Array1<f64>,
    coeff_ml_uv: Array1<f64>,
    coeff_ll_uv: Array1<f64>,
    a_u: Array1<f64>,
    a_v: Array1<f64>,
    a_uv: Array1<f64>,
    c_u: Array1<f64>,
    c_v: Array1<f64>,
    c_uv: Array1<f64>,
    l_u: Array1<f64>,
    l_v: Array1<f64>,
    l_uv: Array1<f64>,
    dw_u: Array1<f64>,
    dw_v: Array1<f64>,
    dw_uv: Array1<f64>,
}

/// The two probe directions resolved to row space for the GLS Wiggle joint
/// second directional derivative: `xi`/`zeta` are the X_mu/X_ls contractions,
/// and `q`/`s1`/`g2` are the mixed first/second-derivative wiggle pieces.
pub(crate) struct GlsWiggleDirPieces<'a> {
    zeta_u: &'a Array1<f64>,
    zeta_v: &'a Array1<f64>,
    q_u: &'a Array1<f64>,
    q_v: &'a Array1<f64>,
    q_uv: &'a Array1<f64>,
    s1_u: &'a Array1<f64>,
    s1_v: &'a Array1<f64>,
    s1_uv: &'a Array1<f64>,
    g2_u: &'a Array1<f64>,
    g2_v: &'a Array1<f64>,
    g2_uv: &'a Array1<f64>,
}

/// Compute the shared GLS Wiggle second-directional row coefficients from the
/// per-row scalars, wiggle geometry, and the resolved probe directions.
pub(crate) fn gls_wiggle_second_directional_coeffs(
    rows: &GaussianJointRowScalars,
    geom: &GaussianLocationScaleWiggleGeometry,
    dir: &GlsWiggleDirPieces<'_>,
) -> GlsWiggleSecondDirCoeffs {
    let GlsWiggleDirPieces {
        zeta_u,
        zeta_v,
        q_u,
        q_v,
        q_uv,
        s1_u,
        s1_v,
        s1_uv,
        g2_u,
        g2_v,
        g2_uv,
    } = *dir;
    let szeta_u = &rows.kappa * zeta_u;
    let szeta_v = &rows.kappa * zeta_v;
    let zeta_u_zeta_v = zeta_u * zeta_v;
    let dw_u = -2.0 * &rows.w * &szeta_u;
    let dw_v = -2.0 * &rows.w * &szeta_v;
    let dw_uv =
        4.0 * &rows.w * &(&szeta_u * &szeta_v) - 2.0 * &rows.w * &rows.kappa_prime * &zeta_u_zeta_v;
    let dm_u = -(&rows.w * q_u) - &(2.0 * &rows.m * &szeta_u);
    let dm_v = -(&rows.w * q_v) - &(2.0 * &rows.m * &szeta_v);
    let dm_uv = &(2.0 * &rows.w * &(q_u * &szeta_v + q_v * &szeta_u)) - &(&rows.w * q_uv)
        + &(4.0 * &rows.m * &(&szeta_u * &szeta_v))
        - 2.0 * &rows.m * &rows.kappa_prime * &zeta_u_zeta_v;
    let coeff_mm_uv = &(&dw_uv * &geom.dq_dq0.mapv(|v| v * v))
        + &(2.0 * &dw_u * &geom.dq_dq0 * s1_v)
        + &(2.0 * &dw_v * &geom.dq_dq0 * s1_u)
        + &(2.0 * &rows.w * s1_u * s1_v)
        + &(2.0 * &rows.w * &geom.dq_dq0 * s1_uv)
        - &(&dm_uv * &geom.d2q_dq02)
        - &(&dm_u * g2_v)
        - &(&dm_v * g2_u)
        - &(&rows.m * g2_uv);
    let n = rows.m.len();
    // H_{μ,ls} ≡ Fisher 0 (mean⊥scale orthogonality; the wiggle and μ both
    // enter the mean, log σ is the only scale block), so every β-directional
    // derivative — including this second-order one — is identically 0.
    let coeff_ml_uv = Array1::<f64>::zeros(n);
    // Second directional derivative of the Fisher (log σ, log σ) block
    // coeff_ll = 2κ²a (#566). η_ls is linear in β (no zeta_uv), so the only
    // surviving term is ∂²(2κ²a)/∂η² · zeta_u·zeta_v = 4a(κ'²+κκ'')·zeta_u·zeta_v
    // — matching the dense helper `d_uv` (gaussian_jointsecond_directionalweights).
    let coeff_ll_uv = 4.0
        * &rows.obs_weight
        * &(&rows.kappa_prime * &rows.kappa_prime + &rows.kappa * &rows.kappa_dprime)
        * &zeta_u_zeta_v;

    let a_u = &dw_u * &geom.dq_dq0 + &rows.w * s1_u;
    let a_v = &dw_v * &geom.dq_dq0 + &rows.w * s1_v;
    let a_uv = &dw_uv * &geom.dq_dq0 + &dw_u * s1_v + &dw_v * s1_u + &rows.w * s1_uv;
    let c_u = -&dm_u;
    let c_v = -&dm_v;
    let c_uv = -&dm_uv;
    // H_{ls,w} ≡ Fisher 0 (wiggle is mean-side; mean⊥scale), so all of its
    // β-directional derivatives are 0.
    let l_u = Array1::<f64>::zeros(n);
    let l_v = Array1::<f64>::zeros(n);
    let l_uv = Array1::<f64>::zeros(n);

    GlsWiggleSecondDirCoeffs {
        coeff_mm_uv,
        coeff_ml_uv,
        coeff_ll_uv,
        a_u,
        a_v,
        a_uv,
        c_u,
        c_v,
        c_uv,
        l_u,
        l_v,
        l_uv,
        dw_u,
        dw_v,
        dw_uv,
    }
}

impl GaussianLocationScaleWiggleFamily {
    pub(crate) fn exact_newton_joint_hessian_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_from_designs(block_states, &xmu, &x_ls)
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_directional_derivative_from_designs(
            block_states,
            &xmu,
            &x_ls,
            d_beta_flat,
        )
    }

    pub(crate) fn exact_newton_joint_hessian_second_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessiansecond_directional_derivative_from_designs(
            block_states,
            &xmu,
            &x_ls,
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }

    pub(crate) fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
    ) -> Result<Option<LocationScaleJointPsiDirection>, String> {
        let Some(parts) = locscale_joint_psi_direction_parts(
            block_states,
            derivative_blocks,
            psi_index,
            self.y.len(),
            xmu.ncols(),
            x_ls.ncols(),
            Self::BLOCK_MU,
            Self::BLOCK_LOG_SIGMA,
            3,
            "GaussianLocationScaleWiggleFamily",
            "mu",
            policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LocationScaleJointPsiDirection {
            block_idx: parts.block_idx,
            local_idx: parts.local_idx,
            z_primary_psi: parts.primary_z,
            z_ls_psi: parts.log_sigma_z,
            x_primary_psi: parts.primary_psi,
            x_ls_psi: parts.log_sigma_psi,
        }))
    }

    pub(crate) fn exact_newton_joint_psisecond_design_drifts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &LocationScaleJointPsiDirection,
        psi_b: &LocationScaleJointPsiDirection,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<LocationScaleJointPsiSecondDrifts, String> {
        locscale_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            psi_a,
            psi_b,
            LocScalePsiDriftConfig {
                n: self.y.len(),
                p_primary: xmu.ncols(),
                p_log_sigma: x_ls.ncols(),
                primary_block_idx: Self::BLOCK_MU,
                log_sigma_block_idx: Self::BLOCK_LOG_SIGMA,
                family_name: "GaussianLocationScaleWiggleFamily",
                primary_label: "mu",
                policy: &self.policy,
            },
        )
    }

    /// Compute the rowwise Hessian pieces shared by the dense path and the
    /// matrix-free workspace operator. The same coefficients reconstruct the
    /// dense p×p matrix or apply `Hv` directly without ever forming it.
    pub(crate) fn wiggle_hessian_row_pieces(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GaussianLocationScaleWiggleHessianRowPieces, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if q0.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        if geom.basis.ncols() != betaw.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleWiggleFamily wiggle basis/beta mismatch: basis has {} columns but beta has {} entries",
                geom.basis.ncols(),
                betaw.len()
            ) }.into());
        }
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;
        let coeff_mm = &rows.w * &geom.dq_dq0.mapv(|v| v * v) - &rows.m * &geom.d2q_dq02;
        // Gaussian mean⊥scale Fisher orthogonality. μ (mu) AND the wiggle both
        // enter the MEAN q = q0 + B(q0)·βw (see `let q = q0 + etaw`); log σ is
        // the only scale-side block. The Fisher (expected) cross between any
        // mean-side parameter and log σ is exactly 0: H_{μ,ls} = 2κm·dq_dq0 and
        // H_{ls,w} = 2κm both carry m = r·w = (y−q)·weight/σ², and E[m] =
        // E[r]·w = 0. The dense and matrix-free workspace paths SHARE these row
        // pieces, so setting the cross coeffs to 0 fixes the curvature object
        // (the observed 2κm value) for both. Diagonal/same-side blocks
        // (coeff_mm within mean, coeff_ll within scale, coeff_mw_* within mean,
        // coeff_ww within mean) are untouched.
        let coeff_ml = Array1::<f64>::zeros(n);
        // Fisher/expected (log σ, log σ) information E[H_{ls,ls}] = 2κ²a (#566):
        // the observed 2κ²n + κ'(a−n) collapses at small residuals and
        // over-smooths the scale; E[n]=a gives the residual-free 2κ²a.
        let coeff_ll = 2.0 * &rows.kappa * &rows.kappa * &rows.obs_weight;
        let coeff_mw_b = &rows.w * &geom.dq_dq0;
        let coeff_mw_d = -&rows.m;
        // ls↔wiggle is a mean⊥scale cross (wiggle is mean-side): Fisher 0.
        let coeff_lw_b = Array1::<f64>::zeros(n);
        let coeff_ww = rows.w.clone();
        Ok(GaussianLocationScaleWiggleHessianRowPieces {
            coeff_mm,
            coeff_ml,
            coeff_ll,
            coeff_mw_b,
            coeff_mw_d,
            coeff_lw_b,
            coeff_ww,
            basis: geom.basis,
            basis_d1: geom.basis_d1,
        })
    }

    pub(crate) fn exact_newton_joint_hessian_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let pieces = self.wiggle_hessian_row_pieces(block_states)?;
        Ok(Some(pieces.assemble_dense(xmu, x_ls)?))
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) = layout.split_three(
            d_beta_flat,
            "GaussianLocationScaleWiggleFamily exact joint directional Hessian",
        )?;
        if q0.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;
        let xi = fast_av(xmu, &umu);
        let zeta = fast_av(x_ls, &u_ls);
        // logb κ-scaled η_ls direction; κ' = dκ/dη_ls = κ(1−κ).
        let szeta = &rows.kappa * &zeta;
        let phi = fast_av(&geom.basis, &uw);
        let mut q_u = &geom.dq_dq0 * &xi;
        q_u += &phi;
        let mut s1_u = &geom.d2q_dq02 * &xi;
        s1_u += &fast_av(&geom.basis_d1, &uw);
        let mut g2_u = &geom.d3q_dq03 * &xi;
        g2_u += &fast_av(&geom.basis_d2, &uw);
        let basis_u = scale_matrix_rows(&geom.basis_d1, &xi)?;
        let basis1_u = scale_matrix_rows(&geom.basis_d2, &xi)?;
        let dw_u = -2.0 * &rows.w * &szeta;
        let dm_u = -(&rows.w * &q_u) - &(2.0 * &rows.m * &szeta);

        let coeff_mm_u = &(&dw_u * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_u)
            - &(&dm_u * &geom.d2q_dq02)
            - &(&rows.m * &g2_u);
        // Static blocks: H_{μ,ls} = Fisher 0 (mean⊥scale); H_{ls,ls} = Fisher
        // 2κ²a (#566). H_{μ,ls} ≡ 0 for all β, so its directional derivative is
        // also identically 0. The Fisher (ls,ls) block 2κ²a depends only on
        // η_ls (a is the constant prior weight), so its directional derivative
        // is 4κκ'a·zeta.
        let coeff_ml_u = Array1::<f64>::zeros(n);
        let coeff_ll_u = 4.0 * &rows.kappa * &rows.kappa_prime * &(&zeta * &rows.obs_weight);
        let a_u = &dw_u * &geom.dq_dq0 + &rows.w * &s1_u;
        let c_u = -&dm_u;
        // ls↔wiggle cross block: Fisher 0 (wiggle is mean-side), so its
        // directional derivative is 0 as well.
        let l_u = Array1::<f64>::zeros(n);
        let zeros_ls_b1 = Array1::<f64>::zeros(n);

        let h_mm = xt_diag_x_dense(xmu, &coeff_mm_u)?;
        let h_ml = xt_diag_y_dense(xmu, &coeff_ml_u, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &coeff_ll_u)?;
        let h_mw = xt_diag_y_dense(xmu, &a_u, &geom.basis)?
            + &xt_diag_y_dense(xmu, &(&rows.w * &geom.dq_dq0), &basis_u)?
            + &xt_diag_y_dense(xmu, &c_u, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &(-&rows.m), &basis1_u)?;
        let h_lw = xt_diag_y_dense(x_ls, &l_u, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &zeros_ls_b1, &basis_u)?;
        let a_ww = xt_diag_y_dense(&basis_u, &rows.w, &geom.basis)?;
        let h_ww = &a_ww + &a_ww.t() + &xt_diag_x_dense(&geom.basis, &dw_u)?;
        Ok(Some(gaussian_pack_wiggle_joint_symmetrichessian(
            &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
        )))
    }

    /// Build a matrix-free `RowCoeffOperator` for the GLS Wiggle joint
    /// directional derivative `D_β H_L[u]`. Output dimension is
    /// `pmu + p_ls + pw`. Channels (in order): X_mu, X_ls, B, B', B''.
    pub(crate) fn gls_wiggle_directional_operator(
        &self,
        block_states: &[ParameterBlockState],
        xmu_arc: Arc<Array2<f64>>,
        x_ls_arc: Arc<Array2<f64>>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let pmu = xmu_arc.ncols();
        let p_ls = x_ls_arc.ncols();
        let q0_eta = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) =
            layout.split_three(d_beta_flat, "GLS Wiggle joint dH operator d_beta")?;
        if q0_eta.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = q0_eta + etaw;
        let geom = self.wiggle_geometry(q0_eta.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;
        let xi = fast_av(xmu_arc.as_ref(), &umu);
        let zeta = fast_av(x_ls_arc.as_ref(), &u_ls);
        let szeta = &rows.kappa * &zeta;
        let phi = fast_av(&geom.basis, &uw);
        let mut q_u = &geom.dq_dq0 * &xi;
        q_u += &phi;
        let mut s1_u = &geom.d2q_dq02 * &xi;
        s1_u += &fast_av(&geom.basis_d1, &uw);
        let mut g2_u = &geom.d3q_dq03 * &xi;
        g2_u += &fast_av(&geom.basis_d2, &uw);
        let dw_u = -2.0 * &rows.w * &szeta;
        let dm_u = -(&rows.w * &q_u) - &(2.0 * &rows.m * &szeta);

        let coeff_mm_u = &(&dw_u * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_u)
            - &(&dm_u * &geom.d2q_dq02)
            - &(&rows.m * &g2_u);
        // H_{μ,ls} ≡ Fisher 0 (mean⊥scale); its directional derivative is 0.
        let coeff_ml_u = Array1::<f64>::zeros(n);
        // Fisher (ls,ls) 2κ²a directional derivative: 4κκ'a·zeta (#566).
        let coeff_ll_u = 4.0 * &rows.kappa * &rows.kappa_prime * &(&zeta * &rows.obs_weight);
        let a_u = &dw_u * &geom.dq_dq0 + &rows.w * &s1_u;
        let c_u = -&dm_u;
        // H_{ls,w} ≡ Fisher 0 (wiggle is mean-side); its derivative is 0 in
        // both the B channel (l_u) and the B' channel (coeff_ls_b1).
        let l_u = Array1::<f64>::zeros(n);

        // Pair-coefficient bundles. For (0=X_mu, 3=B'): combine
        // `xt_diag_y_dense(xmu, &(w·dq_dq0), &basis_u=diag(xi)·B')`
        // (giving coeff `w·dq_dq0·xi`) with `xt_diag_y_dense(xmu, &c_u, &B')`
        // (coeff `c_u`).
        let coeff_m_b1 = &(&rows.w * &geom.dq_dq0 * &xi) + &c_u;
        // (0=X_mu, 4=B''): from `xt_diag_y_dense(xmu, &(-m), &basis1_u=diag(xi)·B'')`.
        let coeff_m_b2 = -(&rows.m * &xi);
        // (1=X_ls, 3=B'): ls↔wiggle Fisher-0 cross → zero.
        let coeff_ls_b1 = Array1::<f64>::zeros(n);
        // (2=B, 3=B'): a_ww + a_ww^T where a_ww = (diag(xi)·B')^T diag(w) B
        // = B'^T diag(w·xi) B. The symmetric pair contribution in
        // `RowCoeffOperator` reproduces a_ww + a_ww^T with c = w·xi.
        let coeff_b_b1 = &rows.w * &xi;

        let basis: Arc<Array2<f64>> = Arc::new(geom.basis.clone());
        let basis_d1: Arc<Array2<f64>> = Arc::new(geom.basis_d1.clone());
        let basis_d2: Arc<Array2<f64>> = Arc::new(geom.basis_d2.clone());
        let pw = basis.ncols();

        Ok(Some(Arc::new(RowCoeffOperator::from_directions(
            vec![pmu, p_ls, pw],
            vec![
                (0, xmu_arc),
                (1, x_ls_arc),
                (2, basis),
                (2, basis_d1),
                (2, basis_d2),
            ],
            vec![
                // (X_mu, X_mu) ← `xt_diag_x_dense(xmu, &coeff_mm_u)`
                (0, 0, coeff_mm_u),
                // (X_mu, X_ls) ← `xt_diag_y_dense(xmu, &coeff_ml_u, x_ls)`
                (0, 1, coeff_ml_u),
                // (X_ls, X_ls) ← `xt_diag_x_dense(x_ls, &coeff_ll_u)`
                (1, 1, coeff_ll_u),
                // (X_mu, B) ← `xt_diag_y_dense(xmu, &a_u, &geom.basis)`
                (0, 2, a_u),
                // (X_mu, B') ← `xt_diag_y_dense(xmu, w·dq_dq0, basis_u=diag(ξ)·B') + xt_diag_y_dense(xmu, c_u, B')`
                (0, 3, coeff_m_b1),
                // (X_mu, B'') ← `xt_diag_y_dense(xmu, -m, basis1_u=diag(ξ)·B'')`
                (0, 4, coeff_m_b2),
                // (X_ls, B) ← `xt_diag_y_dense(x_ls, &l_u, &geom.basis)`
                (1, 2, l_u),
                // (X_ls, B') ← ls↔wiggle is mean⊥scale Fisher 0, so coeff_ls_b1 = 0
                (1, 3, coeff_ls_b1),
                // (B, B) ← `xt_diag_x_dense(&geom.basis, &dw_u)`
                (2, 2, dw_u),
                // (B, B') ← a_ww + a_ww^T = B^T diag(w·ξ) B' + B'^T diag(w·ξ) B
                (2, 3, coeff_b_b1),
            ],
            n,
        ))))
    }

    /// Build a matrix-free `RowCoeffOperator` for the GLS Wiggle joint
    /// second directional derivative `D²_β H_L[u, v]`. Channels: X_mu,
    /// X_ls, B, B', B'', B'''. Pair list mirrors the 8-term `xt_diag_*`
    /// assembly in `_from_designs`, with row-coefficient bundles that
    /// absorb the `ξ_u, ξ_v, ξ_u·ξ_v` row factors arising from
    /// `basis_u = diag(ξ_u)·B'`, `basis_uv = diag(ξ_u·ξ_v)·B''`, etc.
    pub(crate) fn gls_wiggle_second_directional_operator(
        &self,
        block_states: &[ParameterBlockState],
        xmu_arc: Arc<Array2<f64>>,
        x_ls_arc: Arc<Array2<f64>>,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let pmu = xmu_arc.ncols();
        let p_ls = x_ls_arc.ncols();
        let q0_eta = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) = layout.split_three(d_beta_u, "GLS Wiggle d2H operator (u)")?;
        let (vmu, v_ls, vw) = layout.split_three(d_beta_v, "GLS Wiggle d2H operator (v)")?;
        if q0_eta.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = q0_eta + etaw;
        let geom = self.wiggle_geometry(q0_eta.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;

        let xi_u = fast_av(xmu_arc.as_ref(), &umu);
        let xi_v = fast_av(xmu_arc.as_ref(), &vmu);
        let zeta_u = fast_av(x_ls_arc.as_ref(), &u_ls);
        let zeta_v = fast_av(x_ls_arc.as_ref(), &v_ls);
        let phi_u = fast_av(&geom.basis, &uw);
        let phi_v = fast_av(&geom.basis, &vw);
        let b1u = fast_av(&geom.basis_d1, &uw);
        let b1v = fast_av(&geom.basis_d1, &vw);
        let b2u = fast_av(&geom.basis_d2, &uw);
        let b2v = fast_av(&geom.basis_d2, &vw);
        let b3u = fast_av(&geom.basis_d3, &uw);
        let b3v = fast_av(&geom.basis_d3, &vw);

        let mut q_u = &geom.dq_dq0 * &xi_u;
        q_u += &phi_u;
        let mut q_v = &geom.dq_dq0 * &xi_v;
        q_v += &phi_v;
        let mut s1_u = &geom.d2q_dq02 * &xi_u;
        s1_u += &b1u;
        let mut s1_v = &geom.d2q_dq02 * &xi_v;
        s1_v += &b1v;
        let mut g2_u = &geom.d3q_dq03 * &xi_u;
        g2_u += &b2u;
        let mut g2_v = &geom.d3q_dq03 * &xi_v;
        g2_v += &b2v;
        let q_uv = &(&geom.d2q_dq02 * &(&xi_u * &xi_v)) + &(&b1u * &xi_v) + &(&b1v * &xi_u);
        let s1_uv = &(&geom.d3q_dq03 * &(&xi_u * &xi_v)) + &(&b2u * &xi_v) + &(&b2v * &xi_u);
        let g2_uv = &(&geom.d4q_dq04 * &(&xi_u * &xi_v)) + &(&b3u * &xi_v) + &(&b3v * &xi_u);

        let GlsWiggleSecondDirCoeffs {
            coeff_mm_uv,
            coeff_ml_uv,
            coeff_ll_uv,
            a_u,
            a_v,
            a_uv,
            c_u,
            c_v,
            c_uv,
            l_u,
            l_v,
            l_uv,
            dw_u,
            dw_v,
            dw_uv,
        } = gls_wiggle_second_directional_coeffs(
            &rows,
            &geom,
            &GlsWiggleDirPieces {
                zeta_u: &zeta_u,
                zeta_v: &zeta_v,
                q_u: &q_u,
                q_v: &q_v,
                q_uv: &q_uv,
                s1_u: &s1_u,
                s1_v: &s1_v,
                s1_uv: &s1_uv,
                g2_u: &g2_u,
                g2_v: &g2_v,
                g2_uv: &g2_uv,
            },
        );

        // Pair-coefficient bundles. Cross-block (mu, B'/B'') absorb basis_u/v/uv row scaling.
        let xi_u_xi_v = &xi_u * &xi_v;
        let coeff_m_b1 = &(&a_u * &xi_v) + &(&a_v * &xi_u) + &c_uv;
        let coeff_m_b2 = &(&rows.w * &geom.dq_dq0 * &xi_u_xi_v) + &(&c_u * &xi_v) + &(&c_v * &xi_u);
        let coeff_m_b3 = -(&rows.m * &xi_u_xi_v);
        // ls↔wiggle is Fisher-0 (mean⊥scale): the B' (coeff_ls_b1) and B''
        // (coeff_ls_b2) channels of its second directional derivative vanish.
        let coeff_ls_b1 = &(&l_u * &xi_v) + &(&l_v * &xi_u);
        let coeff_ls_b2 = Array1::<f64>::zeros(n);
        // Wiggle-wiggle from a_ab + a_ab^T + a_ij + a_ij^T + a_iwj + a_iwj^T + a_jwi + a_jwi^T:
        //   a_ab = B''^T diag(w·ξ_uξ_v) B    → pair (B, B'', w·ξ_uξ_v)
        //   a_ij = B'^T diag(w·ξ_uξ_v) B'   → pair (B', B', 2·w·ξ_uξ_v)  (a_ij + a_ij^T)
        //   a_iwj+a_jwi = B'^T diag(dw_v·ξ_u + dw_u·ξ_v) B → pair (B, B', sum)
        let coeff_b_b1 = &(&dw_u * &xi_v) + &(&dw_v * &xi_u);
        let coeff_b_b2 = &rows.w * &xi_u_xi_v;
        let coeff_b1_b1 = 2.0 * &(&rows.w * &xi_u_xi_v);

        let basis: Arc<Array2<f64>> = Arc::new(geom.basis.clone());
        let basis_d1: Arc<Array2<f64>> = Arc::new(geom.basis_d1.clone());
        let basis_d2: Arc<Array2<f64>> = Arc::new(geom.basis_d2.clone());
        let basis_d3: Arc<Array2<f64>> = Arc::new(geom.basis_d3.clone());
        let pw = basis.ncols();

        Ok(Some(Arc::new(RowCoeffOperator::from_directions(
            vec![pmu, p_ls, pw],
            vec![
                (0, xmu_arc),
                (1, x_ls_arc),
                (2, basis),
                (2, basis_d1),
                (2, basis_d2),
                (2, basis_d3),
            ],
            vec![
                // (X_mu, X_mu) ← `xt_diag_x_dense(xmu, &coeff_mm_uv)`
                (0, 0, coeff_mm_uv),
                // (X_mu, X_ls) ← `xt_diag_y_dense(xmu, &coeff_ml_uv, x_ls)`
                (0, 1, coeff_ml_uv),
                // (X_ls, X_ls) ← `xt_diag_x_dense(x_ls, &coeff_ll_uv)`
                (1, 1, coeff_ll_uv),
                // (X_mu, B) ← `xt_diag_y_dense(xmu, &a_uv, &geom.basis)`
                (0, 2, a_uv),
                // (X_mu, B') ← combined `a_u·ξ_v + a_v·ξ_u + c_uv` from
                // `xt_diag_y_dense(xmu, a_u, basis_v) + xt_diag_y_dense(xmu,
                // a_v, basis_u) + xt_diag_y_dense(xmu, c_uv, B')`
                (0, 3, coeff_m_b1),
                // (X_mu, B'') ← `xt_diag_y_dense(xmu, w·dq_dq0, basis_uv) +
                // xt_diag_y_dense(xmu, c_u, basis1_v) + xt_diag_y_dense(xmu,
                // c_v, basis1_u)` (basis_uv = diag(ξ_uξ_v)·B'';
                // basis1_{u,v} = diag(ξ_{u,v})·B'')
                (0, 4, coeff_m_b2),
                // (X_mu, B''') ← `xt_diag_y_dense(xmu, -m, basis1_uv)`
                // with basis1_uv = diag(ξ_uξ_v)·B'''
                (0, 5, coeff_m_b3),
                // (X_ls, B) ← `xt_diag_y_dense(x_ls, &l_uv, &geom.basis)`
                (1, 2, l_uv),
                // (X_ls, B') ← combined from `xt_diag_y_dense(x_ls, l_u,
                // basis_v) + xt_diag_y_dense(x_ls, l_v, basis_u)` =
                // `l_u·ξ_v + l_v·ξ_u`
                (1, 3, coeff_ls_b1),
                // (X_ls, B'') ← ls↔wiggle is mean⊥scale Fisher 0, so coeff_ls_b2 = 0
                (1, 4, coeff_ls_b2),
                // (B, B) ← `xt_diag_x_dense(&geom.basis, &dw_uv)`
                (2, 2, dw_uv),
                // (B, B') ← combined `a_iwj + a_iwj^T + a_jwi + a_jwi^T` =
                // B^T diag(dw_u·ξ_v + dw_v·ξ_u) B' + B'^T diag(...) B
                (2, 3, coeff_b_b1),
                // (B, B'') ← `a_ab + a_ab^T` with a_ab = B''^T diag(w·ξ_uξ_v) B
                (2, 4, coeff_b_b2),
                // (B', B') ← `a_ij + a_ij^T = 2·B'^T diag(w·ξ_uξ_v) B'`;
                // diagonal pair coeff doubles to absorb the factor of 2
                (3, 3, coeff_b1_b1),
            ],
            n,
        ))))
    }

    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) = layout.split_three(
            d_beta_u_flat,
            "GaussianLocationScaleWiggleFamily exact joint second directional Hessian (u)",
        )?;
        let (vmu, v_ls, vw) = layout.split_three(
            d_beta_v_flat,
            "GaussianLocationScaleWiggleFamily exact joint second directional Hessian (v)",
        )?;
        if q0.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;

        let xi_u = fast_av(xmu, &umu);
        let xi_v = fast_av(xmu, &vmu);
        let zeta_u = fast_av(x_ls, &u_ls);
        let zeta_v = fast_av(x_ls, &v_ls);
        let phi_u = fast_av(&geom.basis, &uw);
        let phi_v = fast_av(&geom.basis, &vw);
        let b1u = fast_av(&geom.basis_d1, &uw);
        let b1v = fast_av(&geom.basis_d1, &vw);
        let b2u = fast_av(&geom.basis_d2, &uw);
        let b2v = fast_av(&geom.basis_d2, &vw);
        let b3u = fast_av(&geom.basis_d3, &uw);
        let b3v = fast_av(&geom.basis_d3, &vw);

        let mut q_u = &geom.dq_dq0 * &xi_u;
        q_u += &phi_u;
        let mut q_v = &geom.dq_dq0 * &xi_v;
        q_v += &phi_v;
        let mut s1_u = &geom.d2q_dq02 * &xi_u;
        s1_u += &b1u;
        let mut s1_v = &geom.d2q_dq02 * &xi_v;
        s1_v += &b1v;
        let mut g2_u = &geom.d3q_dq03 * &xi_u;
        g2_u += &b2u;
        let mut g2_v = &geom.d3q_dq03 * &xi_v;
        g2_v += &b2v;
        let q_uv = &(&geom.d2q_dq02 * &(&xi_u * &xi_v)) + &(&b1u * &xi_v) + &(&b1v * &xi_u);
        let s1_uv = &(&geom.d3q_dq03 * &(&xi_u * &xi_v)) + &(&b2u * &xi_v) + &(&b2v * &xi_u);
        let g2_uv = &(&geom.d4q_dq04 * &(&xi_u * &xi_v)) + &(&b3u * &xi_v) + &(&b3v * &xi_u);

        let basis_u = scale_matrix_rows(&geom.basis_d1, &xi_u)?;
        let basis_v = scale_matrix_rows(&geom.basis_d1, &xi_v)?;
        let basis_uv = scale_matrix_rows(&geom.basis_d2, &(&xi_u * &xi_v))?;
        let basis1_u = scale_matrix_rows(&geom.basis_d2, &xi_u)?;
        let basis1_v = scale_matrix_rows(&geom.basis_d2, &xi_v)?;
        let basis1_uv = scale_matrix_rows(&geom.basis_d3, &(&xi_u * &xi_v))?;

        // Shared κ-aware second-directional row coefficients (κ' = κ(1−κ),
        // κ'' = κ(1−κ)(1−2κ), κ''' = κ''(1−2κ) − 2(κ')²): identical to the
        // matrix-free operator path, factored into one helper.
        let GlsWiggleSecondDirCoeffs {
            coeff_mm_uv,
            coeff_ml_uv,
            coeff_ll_uv,
            a_u,
            a_v,
            a_uv,
            c_u,
            c_v,
            c_uv,
            l_u,
            l_v,
            l_uv,
            dw_u,
            dw_v,
            dw_uv,
        } = gls_wiggle_second_directional_coeffs(
            &rows,
            &geom,
            &GlsWiggleDirPieces {
                zeta_u: &zeta_u,
                zeta_v: &zeta_v,
                q_u: &q_u,
                q_v: &q_v,
                q_uv: &q_uv,
                s1_u: &s1_u,
                s1_v: &s1_v,
                s1_uv: &s1_uv,
                g2_u: &g2_u,
                g2_v: &g2_v,
                g2_uv: &g2_uv,
            },
        );

        let h_mm = xt_diag_x_dense(xmu, &coeff_mm_uv)?;
        let h_ml = xt_diag_y_dense(xmu, &coeff_ml_uv, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &coeff_ll_uv)?;
        let h_mw = xt_diag_y_dense(xmu, &a_uv, &geom.basis)?
            + &xt_diag_y_dense(xmu, &a_u, &basis_v)?
            + &xt_diag_y_dense(xmu, &a_v, &basis_u)?
            + &xt_diag_y_dense(xmu, &(&rows.w * &geom.dq_dq0), &basis_uv)?
            + &xt_diag_y_dense(xmu, &c_uv, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &c_u, &basis1_v)?
            + &xt_diag_y_dense(xmu, &c_v, &basis1_u)?
            + &xt_diag_y_dense(xmu, &(-&rows.m), &basis1_uv)?;
        // H_{ls,w} ≡ Fisher 0 (mean⊥scale): l_uv/l_u/l_v are 0 (shared helper)
        // and the 2κm·B'' channel vanishes too.
        let zeros_ls_b2 = Array1::<f64>::zeros(n);
        let h_lw = xt_diag_y_dense(x_ls, &l_uv, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &l_u, &basis_v)?
            + &xt_diag_y_dense(x_ls, &l_v, &basis_u)?
            + &xt_diag_y_dense(x_ls, &zeros_ls_b2, &basis_uv)?;
        let a_ab = xt_diag_y_dense(&basis_uv, &rows.w, &geom.basis)?;
        let a_ij = xt_diag_y_dense(&basis_u, &rows.w, &basis_v)?;
        let a_iwj = xt_diag_y_dense(&basis_u, &dw_v, &geom.basis)?;
        let a_jwi = xt_diag_y_dense(&basis_v, &dw_u, &geom.basis)?;
        let h_ww = &a_ab
            + &a_ab.t()
            + &a_ij
            + a_ij.t()
            + &a_iwj
            + a_iwj.t()
            + &a_jwi
            + a_jwi.t()
            + &xt_diag_x_dense(&geom.basis, &dw_uv)?;
        Ok(Some(gaussian_pack_wiggle_joint_symmetrichessian(
            &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
        )))
    }

    pub(crate) fn exact_newton_joint_psi_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;
        let xmu_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();

        let q_a = &geom.dq_dq0 * &dir_a.z_primary_psi;
        let s1_a = &geom.d2q_dq02 * &dir_a.z_primary_psi;
        let g2_a = &geom.d3q_dq03 * &dir_a.z_primary_psi;
        let basis_a = scale_matrix_rows(&geom.basis_d1, &dir_a.z_primary_psi)?;
        let basis1_a = scale_matrix_rows(&geom.basis_d2, &dir_a.z_primary_psi)?;
        // logb κ-chain on η_ls; e_a = ∂η_ls/∂ψ_a row-direction.
        let e_a = &dir_a.z_ls_psi;
        let amn = &rows.obs_weight - &rows.n;
        let dw_a = -2.0 * &rows.w * &rows.kappa * e_a;
        let dm_a = -(&rows.w * &q_a) - &(2.0 * &rows.m * &rows.kappa * e_a);
        let dn_a = -(2.0 * &rows.m * &q_a) - &(2.0 * &rows.n * &rows.kappa * e_a);
        let s_mu = -&rows.m * &geom.dq_dq0;
        let s_mu_a = -(&dm_a * &geom.dq_dq0) - &(&rows.m * &s1_a);
        let s_ls = &rows.kappa * &amn;
        let s_ls_a = &rows.kappa_prime * &(e_a * &amn) - &rows.kappa * &dn_a;
        let s_w = -&rows.m;
        let s_w_a = -&dm_a;

        let objective_psi = (-&rows.m * &q_a + &s_ls * e_a).sum();
        let score_psi = gaussian_pack_wiggle_joint_score(
            &(xmu_map.transpose_mul(s_mu.view()) + fast_atv(xmu, &s_mu_a)),
            &(x_ls_map.transpose_mul(s_ls.view()) + fast_atv(x_ls, &s_ls_a)),
            &(fast_atv(&basis_a, &s_w) + fast_atv(&geom.basis, &s_w_a)),
        );

        // Static blocks under logb. Gaussian mean⊥scale Fisher orthogonality:
        // μ AND the wiggle both enter the MEAN q = q0 + B(q0)·βw, so log σ is
        // the only scale-side block. The Fisher (expected) cross between any
        // mean-side parameter and log σ is exactly 0 because it carries
        // m = r·weight/σ² and E[m] = E[r]·weight/σ² = 0:
        //   coeff_ml = E[H_{μ,ls}] = 0  (observed 2κmD)
        //   l        = E[H_{ls,w}] = 0  (observed 2κm)
        // A function identically 0 has 0 ψ-derivatives, so coeff_ml_a and l_a
        // vanish too. This mirrors the non-wiggle psi path
        // (gaussian_joint_psi_firstweights: hmu_ls = dhmu_ls = 0) and the
        // wiggle Newton/REML Hessian path (wiggle_hessian_row_pieces:
        // coeff_ml = coeff_lw_b = 0). The observed SCORE (s_mu/s_ls/s_w above)
        // stays exact so Fisher scoring still hits the joint MLE; only the
        // curvature feeding the REML determinant / IFT correction is the
        // (orthogonal) expectation. coeff_ll is the residual-free Fisher
        // 2κ²a (#566); its ψ-derivative coeff_ll_a = 4κκ'a·e_a depends only on
        // η_ls. Same-side blocks (coeff_mm within mean, a/c the μ↔wiggle
        // within-mean cross, coeff_ww within mean) are untouched.
        let n = rows.m.len();
        let coeff_mm = &rows.w * &geom.dq_dq0.mapv(|v| v * v) - &rows.m * &geom.d2q_dq02;
        let coeff_mm_a = &(&dw_a * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_a)
            - &(&dm_a * &geom.d2q_dq02)
            - &(&rows.m * &g2_a);
        let coeff_ml = Array1::<f64>::zeros(n);
        let coeff_ml_a = Array1::<f64>::zeros(n);
        let coeff_ll = 2.0 * &rows.kappa * &rows.kappa * &rows.obs_weight;
        let coeff_ll_a = 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * e_a;
        let a = &rows.w * &geom.dq_dq0;
        let a_a = &dw_a * &geom.dq_dq0 + &rows.w * &s1_a;
        let c = -&rows.m;
        let c_a = -&dm_a;
        let l = Array1::<f64>::zeros(n);
        let l_a = Array1::<f64>::zeros(n);
        let h_mm_a1 = weighted_crossprod_psi_maps(
            xmu_map,
            coeff_mm.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let h_mm = &h_mm_a1 + &h_mm_a1.t() + &xt_diag_x_dense(xmu, &coeff_mm_a)?;
        let h_ml = weighted_crossprod_psi_maps(
            xmu_map,
            coeff_ml.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            coeff_ml.view(),
            x_ls_map,
        )? + &xt_diag_y_dense(xmu, &coeff_ml_a, x_ls)?;
        let h_ll_a1 = weighted_crossprod_psi_maps(
            x_ls_map,
            coeff_ll.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let h_ll = &h_ll_a1 + &h_ll_a1.t() + &xt_diag_x_dense(x_ls, &coeff_ll_a)?;
        let h_mw = weighted_crossprod_psi_maps(
            xmu_map,
            a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &xt_diag_y_dense(xmu, &a_a, &geom.basis)?
            + &xt_diag_y_dense(xmu, &a, &basis_a)?
            + &weighted_crossprod_psi_maps(
                xmu_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &xt_diag_y_dense(xmu, &c_a, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &c, &basis1_a)?;
        let h_lw = weighted_crossprod_psi_maps(
            x_ls_map,
            l.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &xt_diag_y_dense(x_ls, &l_a, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &l, &basis_a)?;
        let h_ww_a1 = xt_diag_y_dense(&basis_a, &rows.w, &geom.basis)?;
        let h_ww = &h_ww_a1 + &h_ww_a1.t() + &xt_diag_x_dense(&geom.basis, &dw_a)?;

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: gaussian_pack_wiggle_joint_symmetrichessian(
                &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
            ),
            hessian_psi_operator: None,
        }))
    }

    pub(crate) fn exact_newton_joint_psisecond_order_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_i,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let Some(dir_b) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_j,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psisecond_order_terms_from_parts(
                block_states,
                derivative_blocks,
                &dir_a,
                &dir_b,
                xmu,
                x_ls,
            )?,
        ))
    }

    pub(crate) fn exact_newton_joint_psisecond_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        dir_a: &LocationScaleJointPsiDirection,
        dir_b: &LocationScaleJointPsiDirection,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms, String> {
        let second_drifts = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            dir_a,
            dir_b,
            xmu,
            x_ls,
        )?;
        let n = self.y.len();
        let xmu_a_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_a_map = dir_a.x_ls_psi.as_linear_map_ref();
        let xmu_b_map = dir_b.x_primary_psi.as_linear_map_ref();
        let x_ls_b_map = dir_b.x_ls_psi.as_linear_map_ref();
        let xmu_ab_map = second_psi_linear_map(
            second_drifts.x_primary_ab_action.as_ref(),
            second_drifts.x_primary_ab.as_ref(),
            n,
            xmu.ncols(),
        );
        let x_ls_ab_map = second_psi_linear_map(
            second_drifts.x_ls_ab_action.as_ref(),
            second_drifts.x_ls_ab.as_ref(),
            n,
            x_ls.ncols(),
        );
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;

        let q_a = &geom.dq_dq0 * &dir_a.z_primary_psi;
        let q_b = &geom.dq_dq0 * &dir_b.z_primary_psi;
        let q_ab = &(&geom.dq_dq0 * &second_drifts.z_primary_ab)
            + &(&geom.d2q_dq02 * &(&dir_a.z_primary_psi * &dir_b.z_primary_psi));
        let s1_a = &geom.d2q_dq02 * &dir_a.z_primary_psi;
        let s1_b = &geom.d2q_dq02 * &dir_b.z_primary_psi;
        let s1_ab = &(&geom.d3q_dq03 * &(&dir_a.z_primary_psi * &dir_b.z_primary_psi))
            + &(&geom.d2q_dq02 * &second_drifts.z_primary_ab);
        let g2_a = &geom.d3q_dq03 * &dir_a.z_primary_psi;
        let g2_b = &geom.d3q_dq03 * &dir_b.z_primary_psi;
        let g2_ab = &(&geom.d4q_dq04 * &(&dir_a.z_primary_psi * &dir_b.z_primary_psi))
            + &(&geom.d3q_dq03 * &second_drifts.z_primary_ab);
        let basis_a = scale_matrix_rows(&geom.basis_d1, &dir_a.z_primary_psi)?;
        let basis_b = scale_matrix_rows(&geom.basis_d1, &dir_b.z_primary_psi)?;
        let basis_ab = scale_matrix_rows(&geom.basis_d1, &second_drifts.z_primary_ab)?
            + &scale_matrix_rows(
                &geom.basis_d2,
                &(&dir_a.z_primary_psi * &dir_b.z_primary_psi),
            )?;
        let basis1_a = scale_matrix_rows(&geom.basis_d2, &dir_a.z_primary_psi)?;
        let basis1_b = scale_matrix_rows(&geom.basis_d2, &dir_b.z_primary_psi)?;
        let basis1_ab = scale_matrix_rows(&geom.basis_d2, &second_drifts.z_primary_ab)?
            + &scale_matrix_rows(
                &geom.basis_d3,
                &(&dir_a.z_primary_psi * &dir_b.z_primary_psi),
            )?;

        // logb κ-chain on η_ls; κ' = κ(1−κ), κ'' = κ(1−κ)(1−2κ),
        // κ''' = κ''(1−2κ) − 2(κ')².
        let e_a = &dir_a.z_ls_psi;
        let e_b = &dir_b.z_ls_psi;
        let e_ab = &second_drifts.z_ls_ab;
        let amn = &rows.obs_weight - &rows.n;
        // 4κ² − 2κ' (∂²w/∂η² style coefficient when both directions hit η_ls).
        let four_k2_minus_2kpi = 4.0 * &rows.kappa * &rows.kappa - 2.0 * &rows.kappa_prime;

        // Row drifts under logb. The η_ls direction picks up a κ on each step,
        // and η_ls·η_ls picks up (4κ²−2κ') from differentiating κ on the
        // second leg. The η_ab (z_ls_ab) leg uses just one κ from the chain.
        let dw_a = -2.0 * &rows.w * &rows.kappa * e_a;
        let dw_b = -2.0 * &rows.w * &rows.kappa * e_b;
        let dw_ab =
            &four_k2_minus_2kpi * &rows.w * &(e_a * e_b) - &(2.0 * &rows.w * &rows.kappa * e_ab);
        let dm_a = -(&rows.w * &q_a) - &(2.0 * &rows.m * &rows.kappa * e_a);
        let dm_b = -(&rows.w * &q_b) - &(2.0 * &rows.m * &rows.kappa * e_b);
        let dm_ab = &(2.0 * &rows.w * &rows.kappa * &(&q_a * e_b + &q_b * e_a))
            - &(&rows.w * &q_ab)
            + &(&four_k2_minus_2kpi * &rows.m * &(e_a * e_b))
            - &(2.0 * &rows.m * &rows.kappa * e_ab);
        let dn_a = -(2.0 * &rows.m * &q_a) - &(2.0 * &rows.n * &rows.kappa * e_a);
        let dn_b = -(2.0 * &rows.m * &q_b) - &(2.0 * &rows.n * &rows.kappa * e_b);
        let dn_ab = &(2.0 * &rows.w * &(&q_a * &q_b))
            + &(4.0 * &rows.m * &rows.kappa * &(&q_a * e_b + &q_b * e_a))
            - &(2.0 * &rows.m * &q_ab)
            + &(&four_k2_minus_2kpi * &rows.n * &(e_a * e_b))
            - &(2.0 * &rows.n * &rows.kappa * e_ab);

        let s_mu = -&rows.m * &geom.dq_dq0;
        let s_mu_a = -(&dm_a * &geom.dq_dq0) - &(&rows.m * &s1_a);
        let s_mu_b = -(&dm_b * &geom.dq_dq0) - &(&rows.m * &s1_b);
        let s_mu_ab =
            -(&dm_ab * &geom.dq_dq0) - &(&dm_a * &s1_b) - &(&dm_b * &s1_a) - &(&rows.m * &s1_ab);
        // score_ls = κ(a−n); ψ derivatives carry κ' / κ'' from chain on κ.
        let s_ls = &rows.kappa * &amn;
        let s_ls_a = &rows.kappa_prime * &(e_a * &amn) - &rows.kappa * &dn_a;
        let s_ls_b = &rows.kappa_prime * &(e_b * &amn) - &rows.kappa * &dn_b;
        // s_ls_ab = κ''·e_a·e_b·(a−n) + κ'·e_ab·(a−n)
        //         − κ'·(e_a·n_b + e_b·n_a) − κ·n_ab
        let s_ls_ab = &rows.kappa_dprime * &(e_a * e_b) * &amn + &rows.kappa_prime * e_ab * &amn
            - &rows.kappa_prime * &(e_a * &dn_b + e_b * &dn_a)
            - &rows.kappa * &dn_ab;
        let s_w = -&rows.m;
        let s_w_a = -&dm_a;
        let s_w_b = -&dm_b;
        let s_w_ab = -&dm_ab;

        let objective_psi_psi = (&rows.w * &(&q_a * &q_b)
            + &(2.0 * &rows.m * &rows.kappa * &(&q_a * e_b + &q_b * e_a))
            + &((2.0 * &rows.kappa * &rows.kappa * &rows.n + &rows.kappa_prime * &amn)
                * &(e_a * e_b))
            - &(&rows.m * &q_ab)
            + &(&rows.kappa * &amn * e_ab))
            .sum();

        let score_psi_psi = gaussian_pack_wiggle_joint_score(
            &(xmu_ab_map.transpose_mul(s_mu.view())
                + xmu_a_map.transpose_mul(s_mu_b.view())
                + xmu_b_map.transpose_mul(s_mu_a.view())
                + fast_atv(xmu, &s_mu_ab)),
            &(x_ls_ab_map.transpose_mul(s_ls.view())
                + x_ls_a_map.transpose_mul(s_ls_b.view())
                + x_ls_b_map.transpose_mul(s_ls_a.view())
                + fast_atv(x_ls, &s_ls_ab)),
            &(fast_atv(&basis_ab, &s_w)
                + fast_atv(&basis_a, &s_w_b)
                + fast_atv(&basis_b, &s_w_a)
                + fast_atv(&geom.basis, &s_w_ab)),
        );

        // Static blocks under logb. coeff_mm has no κ; coeff_ll = Fisher 2κ²a
        // (#566). Gaussian mean⊥scale Fisher orthogonality: the wiggle and μ
        // both enter the mean (q = q0 + B·βw), log σ is the only scale block,
        // so coeff_ml = E[H_{μ,ls}] = 0 and l = E[H_{ls,w}] = 0 (observed 2κm,
        // E[m]=0). All of their ψ-directional derivatives (a/b/ab) are 0 since
        // a function identically 0 has 0 derivatives. The Fisher (ls,ls) block
        // depends only on η_ls so its derivatives carry only κ.
        let n = rows.m.len();
        let coeff_mm = &rows.w * &geom.dq_dq0.mapv(|v| v * v) - &rows.m * &geom.d2q_dq02;
        let coeff_ml = Array1::<f64>::zeros(n);
        let coeff_ll = 2.0 * &rows.kappa * &rows.kappa * &rows.obs_weight;
        // coeff_mm_a/b/ab: structurally κ-free; correctness now follows from
        // dw_a/_b/_ab and dm_a/_b/_ab carrying the κ chain on η_ls (above).
        let coeff_mm_a = &(&dw_a * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_a)
            - &(&dm_a * &geom.d2q_dq02)
            - &(&rows.m * &g2_a);
        let coeff_mm_b = &(&dw_b * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_b)
            - &(&dm_b * &geom.d2q_dq02)
            - &(&rows.m * &g2_b);
        let coeff_mm_ab = &(&dw_ab * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &dw_a * &geom.dq_dq0 * &s1_b)
            + &(2.0 * &dw_b * &geom.dq_dq0 * &s1_a)
            + &(2.0 * &rows.w * &s1_a * &s1_b)
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_ab)
            - &(&dm_ab * &geom.d2q_dq02)
            - &(&dm_a * &g2_b)
            - &(&dm_b * &g2_a)
            - &(&rows.m * &g2_ab);
        // coeff_ml (μ↔logσ) is Fisher 0; its 1st/2nd ψ-directional derivatives
        // are 0 as well.
        let coeff_ml_a = Array1::<f64>::zeros(n);
        let coeff_ml_b = Array1::<f64>::zeros(n);
        let coeff_ml_ab = Array1::<f64>::zeros(n);
        // Fisher (ls,ls) coeff_ll = 2κ²a (a constant prior weight) depends only
        // on η_ls (#566): ∂(2κ²a)/∂η = 4κκ'a, so the ψ-first derivatives are
        // 4κκ'a·e_a / e_b. The η_ab leg carries one κ on top.
        let coeff_ll_a = 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * e_a;
        let coeff_ll_b = 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * e_b;
        // coeff_ll_ab = ∂²(2κ²a)/∂a∂b = 4a(κ'²+κκ'')·e_a·e_b + 4κκ'a·e_ab
        // (mirrors the dense helper `d2h_ls_ls`).
        let coeff_ll_ab = 4.0
            * &rows.obs_weight
            * &(&rows.kappa_prime * &rows.kappa_prime + &rows.kappa * &rows.kappa_dprime)
            * &(e_a * e_b)
            + 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * e_ab;
        let a = &rows.w * &geom.dq_dq0;
        let a_a = &dw_a * &geom.dq_dq0 + &rows.w * &s1_a;
        let a_b = &dw_b * &geom.dq_dq0 + &rows.w * &s1_b;
        let a_ab = &dw_ab * &geom.dq_dq0 + &dw_a * &s1_b + &dw_b * &s1_a + &rows.w * &s1_ab;
        let c = -&rows.m;
        let c_a = -&dm_a;
        let c_b = -&dm_b;
        let c_ab = -&dm_ab;
        // l (logσ↔wiggle) is Fisher 0 (wiggle is mean-side; mean⊥scale), so all
        // of its 1st/2nd ψ-directional derivatives vanish.
        let l = Array1::<f64>::zeros(n);
        let l_a = Array1::<f64>::zeros(n);
        let l_b = Array1::<f64>::zeros(n);
        let l_ab = Array1::<f64>::zeros(n);

        let hmm_ab = weighted_crossprod_psi_maps(
            xmu_ab_map,
            coeff_mm.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let hmm_ij = weighted_crossprod_psi_maps(xmu_a_map, coeff_mm.view(), xmu_b_map)?;
        let hmm_iwj = weighted_crossprod_psi_maps(
            xmu_a_map,
            coeff_mm_b.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let hmm_jwi = weighted_crossprod_psi_maps(
            xmu_b_map,
            coeff_mm_a.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let h_mm = &hmm_ab
            + &hmm_ab.t()
            + &hmm_ij
            + hmm_ij.t()
            + &hmm_iwj
            + hmm_iwj.t()
            + &hmm_jwi
            + hmm_jwi.t()
            + &xt_diag_x_dense(xmu, &coeff_mm_ab)?;
        let h_ml = weighted_crossprod_psi_maps(
            xmu_ab_map,
            coeff_ml.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(xmu_a_map, coeff_ml.view(), x_ls_b_map)?
            + &weighted_crossprod_psi_maps(xmu_b_map, coeff_ml.view(), x_ls_a_map)?
            + &weighted_crossprod_psi_maps(
                xmu_a_map,
                coeff_ml_b.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_b_map,
                coeff_ml_a.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(xmu),
                coeff_ml_a.view(),
                x_ls_b_map,
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(xmu),
                coeff_ml_b.view(),
                x_ls_a_map,
            )?
            + &xt_diag_y_dense(xmu, &coeff_ml_ab, x_ls)?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(xmu),
                coeff_ml.view(),
                x_ls_ab_map,
            )?;
        let hll_ab = weighted_crossprod_psi_maps(
            x_ls_ab_map,
            coeff_ll.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let hll_ij = weighted_crossprod_psi_maps(x_ls_a_map, coeff_ll.view(), x_ls_b_map)?;
        let hll_iwj = weighted_crossprod_psi_maps(
            x_ls_a_map,
            coeff_ll_b.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let hll_jwi = weighted_crossprod_psi_maps(
            x_ls_b_map,
            coeff_ll_a.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let h_ll = &hll_ab
            + &hll_ab.t()
            + &hll_ij
            + hll_ij.t()
            + &hll_iwj
            + hll_iwj.t()
            + &hll_jwi
            + hll_jwi.t()
            + &xt_diag_x_dense(x_ls, &coeff_ll_ab)?;
        let h_mw = weighted_crossprod_psi_maps(
            xmu_ab_map,
            a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            xmu_a_map,
            a_b.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            xmu_a_map,
            a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&basis_b),
        )? + &weighted_crossprod_psi_maps(
            xmu_b_map,
            a_a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &xt_diag_y_dense(xmu, &a_ab, &geom.basis)?
            + &xt_diag_y_dense(xmu, &a_a, &basis_b)?
            + &weighted_crossprod_psi_maps(
                xmu_b_map,
                a.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis_a),
            )?
            + &xt_diag_y_dense(xmu, &a_b, &basis_a)?
            + &xt_diag_y_dense(xmu, &a, &basis_ab)?
            + &weighted_crossprod_psi_maps(
                xmu_ab_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_a_map,
                c_b.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_a_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis1_b),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_b_map,
                c_a.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &xt_diag_y_dense(xmu, &c_ab, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &c_a, &basis1_b)?
            + &weighted_crossprod_psi_maps(
                xmu_b_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis1_a),
            )?
            + &xt_diag_y_dense(xmu, &c_b, &basis1_a)?
            + &xt_diag_y_dense(xmu, &c, &basis1_ab)?;
        let h_lw = weighted_crossprod_psi_maps(
            x_ls_ab_map,
            l.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            x_ls_a_map,
            l_b.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            x_ls_a_map,
            l.view(),
            CustomFamilyPsiLinearMapRef::Dense(&basis_b),
        )? + &weighted_crossprod_psi_maps(
            x_ls_b_map,
            l_a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &xt_diag_y_dense(x_ls, &l_ab, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &l_a, &basis_b)?
            + &weighted_crossprod_psi_maps(
                x_ls_b_map,
                l.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis_a),
            )?
            + &xt_diag_y_dense(x_ls, &l_b, &basis_a)?
            + &xt_diag_y_dense(x_ls, &l, &basis_ab)?;
        let hww_ab = xt_diag_y_dense(&basis_ab, &rows.w, &geom.basis)?;
        let hww_ij = xt_diag_y_dense(&basis_a, &rows.w, &basis_b)?;
        let hww_iwj = xt_diag_y_dense(&basis_a, &dw_b, &geom.basis)?;
        let hww_jwi = xt_diag_y_dense(&basis_b, &dw_a, &geom.basis)?;
        let h_ww = &hww_ab
            + &hww_ab.t()
            + &hww_ij
            + hww_ij.t()
            + &hww_iwj
            + hww_iwj.t()
            + &hww_jwi
            + hww_jwi.t()
            + &xt_diag_x_dense(&geom.basis, &dw_ab)?;

        Ok(crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: gaussian_pack_wiggle_joint_symmetrichessian(
                &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
            ),
            hessian_psi_psi_operator: None,
        })
    }

    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            xmu,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psihessian_directional_derivative_from_parts(
                block_states,
                &dir_a,
                d_beta_flat,
                xmu,
                x_ls,
            )?,
        ))
    }

    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        dir_a: &LocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        xmu: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let pmu = xmu.ncols();
        let p_ls = x_ls.ncols();
        let xmu_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let q0 = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let layout = GamlssBetaLayout::withwiggle(pmu, p_ls, betaw.len());
        let (umu, u_ls, uw) = layout.split_three(
            d_beta_flat,
            "GaussianLocationScaleWiggleFamily joint psi hessian directional derivative",
        )?;
        let q = q0 + etaw;
        let geom = self.wiggle_geometry(q0.view(), betaw.view())?;
        let rows = self.get_or_compute_row_scalars(&q, eta_ls)?;

        let xi = fast_av(xmu, &umu);
        let zeta = fast_av(x_ls, &u_ls);
        let zmu_a_u = xmu_map.forward_mul(umu.view());
        let zls_a_u = x_ls_map.forward_mul(u_ls.view());
        let b1u = fast_av(&geom.basis_d1, &uw);
        let b2u = fast_av(&geom.basis_d2, &uw);
        let b3u = fast_av(&geom.basis_d3, &uw);

        let q_u = &(&geom.dq_dq0 * &xi) + &fast_av(&geom.basis, &uw);
        let s1_u = &(&geom.d2q_dq02 * &xi) + &b1u;
        let g2_u = &(&geom.d3q_dq03 * &xi) + &b2u;
        let g3_u = &(&geom.d4q_dq04 * &xi) + &b3u;

        let q_a = &geom.dq_dq0 * &dir_a.z_primary_psi;
        let s1_a = &geom.d2q_dq02 * &dir_a.z_primary_psi;
        let g2_a = &geom.d3q_dq03 * &dir_a.z_primary_psi;
        let q_a_u = &(&s1_u * &dir_a.z_primary_psi) + &(&geom.dq_dq0 * &zmu_a_u);
        let s1_a_u = &(&g2_u * &dir_a.z_primary_psi) + &(&geom.d2q_dq02 * &zmu_a_u);
        let g2_a_u = &(&g3_u * &dir_a.z_primary_psi) + &(&geom.d3q_dq03 * &zmu_a_u);

        let basis_u = scale_matrix_rows(&geom.basis_d1, &xi)?;
        let basis1_u = scale_matrix_rows(&geom.basis_d2, &xi)?;
        let basis_a = scale_matrix_rows(&geom.basis_d1, &dir_a.z_primary_psi)?;
        let basis1_a = scale_matrix_rows(&geom.basis_d2, &dir_a.z_primary_psi)?;
        let basis_a_u = scale_matrix_rows(&geom.basis_d2, &(&xi * &dir_a.z_primary_psi))?
            + &scale_matrix_rows(&geom.basis_d1, &zmu_a_u)?;
        let basis1_a_u = scale_matrix_rows(&geom.basis_d3, &(&xi * &dir_a.z_primary_psi))?
            + &scale_matrix_rows(&geom.basis_d2, &zmu_a_u)?;

        // logb κ-chain on η_ls; e_a = ψ_a's η_ls direction, ζ = β-direction.
        // η_au = zls_a_u is the second mixed derivative (β·ψ).
        let e_a = &dir_a.z_ls_psi;
        let four_k2_minus_2kpi = 4.0 * &rows.kappa * &rows.kappa - 2.0 * &rows.kappa_prime;
        let dw_u = -2.0 * &rows.w * &rows.kappa * &zeta;
        let dm_u = -(&rows.w * &q_u) - &(2.0 * &rows.m * &rows.kappa * &zeta);
        let dw_a = -2.0 * &rows.w * &rows.kappa * e_a;
        let dm_a = -(&rows.w * &q_a) - &(2.0 * &rows.m * &rows.kappa * e_a);
        let dw_a_u = &four_k2_minus_2kpi * &rows.w * &(e_a * &zeta)
            - &(2.0 * &rows.w * &rows.kappa * &zls_a_u);
        let dm_a_u = &(2.0 * &rows.w * &rows.kappa * &(&q_a * &zeta + &q_u * e_a))
            - &(&rows.w * &q_a_u)
            + &(&four_k2_minus_2kpi * &rows.m * &(e_a * &zeta))
            - &(2.0 * &rows.m * &rows.kappa * &zls_a_u);

        let coeff_mm_u = &(&dw_u * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_u)
            - &(&dm_u * &geom.d2q_dq02)
            - &(&rows.m * &g2_u);
        // coeff_ml (μ↔logσ) is mean⊥scale Fisher 0 (E[m]=0), so both its
        // β-drift derivative coeff_ml_u and the mixed coeff_ml_a_u are 0.
        let n = rows.m.len();
        let coeff_ml_u = Array1::<f64>::zeros(n);
        // Fisher (ls,ls) coeff_ll = 2κ²a (#566); ∂(2κ²a)/∂η = 4κκ'a, so the
        // β-drift derivative along ζ is 4κκ'a·ζ.
        let coeff_ll_u = 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * &zeta;
        let coeff_mm_a_u = &(&dw_a_u * &geom.dq_dq0.mapv(|v| v * v))
            + &(2.0 * &dw_a * &geom.dq_dq0 * &s1_u)
            + &(2.0 * &dw_u * &geom.dq_dq0 * &s1_a)
            + &(2.0 * &rows.w * &s1_u * &s1_a)
            + &(2.0 * &rows.w * &geom.dq_dq0 * &s1_a_u)
            - &(&dm_a_u * &geom.d2q_dq02)
            - &(&dm_a * &g2_u)
            - &(&dm_u * &g2_a)
            - &(&rows.m * &g2_a_u);
        // coeff_ml_a_u = ∂²(coeff_ml)/∂a∂u = 0 (coeff_ml ≡ Fisher 0).
        let coeff_ml_a_u = Array1::<f64>::zeros(n);
        // coeff_ll_a_u = ∂²(2κ²a)/∂a∂u for the Fisher (ls,ls) block (#566):
        // 4a(κ'²+κκ'')·e_a·ζ + 4κκ'a·η_au (the η_au=zls_a_u mixed leg), mirroring
        // the dense mixed-drift helper.
        let coeff_ll_a_u = 4.0
            * &rows.obs_weight
            * &(&rows.kappa_prime * &rows.kappa_prime + &rows.kappa * &rows.kappa_dprime)
            * &(e_a * &zeta)
            + 4.0 * &rows.kappa * &rows.kappa_prime * &rows.obs_weight * &zls_a_u;

        let a = &rows.w * &geom.dq_dq0;
        let a_u = &dw_u * &geom.dq_dq0 + &rows.w * &s1_u;
        let a_a = &dw_a * &geom.dq_dq0 + &rows.w * &s1_a;
        let a_a_u = &dw_a_u * &geom.dq_dq0 + &dw_a * &s1_u + &dw_u * &s1_a + &rows.w * &s1_a_u;
        let c = -&rows.m;
        let c_u = -&dm_u;
        let c_a = -&dm_a;
        let c_a_u = -&dm_a_u;
        // l (logσ↔wiggle) is mean⊥scale Fisher 0 (wiggle is mean-side), so its
        // β-drift (l_u), ψ (l_a), and mixed (l_a_u) derivatives all vanish.
        let l = Array1::<f64>::zeros(n);
        let l_u = Array1::<f64>::zeros(n);
        let l_a = Array1::<f64>::zeros(n);
        let l_a_u = Array1::<f64>::zeros(n);

        let hmm_a1 = weighted_crossprod_psi_maps(
            xmu_map,
            coeff_mm_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(xmu),
        )?;
        let h_mm = &hmm_a1 + &hmm_a1.t() + &xt_diag_x_dense(xmu, &coeff_mm_a_u)?;
        let h_ml = weighted_crossprod_psi_maps(
            xmu_map,
            coeff_ml_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(xmu),
            coeff_ml_u.view(),
            x_ls_map,
        )? + &xt_diag_y_dense(xmu, &coeff_ml_a_u, x_ls)?;
        let hll_a1 = weighted_crossprod_psi_maps(
            x_ls_map,
            coeff_ll_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )?;
        let h_ll = &hll_a1 + &hll_a1.t() + &xt_diag_x_dense(x_ls, &coeff_ll_a_u)?;
        let h_mw = weighted_crossprod_psi_maps(
            xmu_map,
            a_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            xmu_map,
            a.view(),
            CustomFamilyPsiLinearMapRef::Dense(&basis_u),
        )? + &xt_diag_y_dense(xmu, &a_a_u, &geom.basis)?
            + &xt_diag_y_dense(xmu, &a_a, &basis_u)?
            + &xt_diag_y_dense(xmu, &a_u, &basis_a)?
            + &xt_diag_y_dense(xmu, &a, &basis_a_u)?
            + &weighted_crossprod_psi_maps(
                xmu_map,
                c_u.view(),
                CustomFamilyPsiLinearMapRef::Dense(&geom.basis_d1),
            )?
            + &weighted_crossprod_psi_maps(
                xmu_map,
                c.view(),
                CustomFamilyPsiLinearMapRef::Dense(&basis1_u),
            )?
            + &xt_diag_y_dense(xmu, &c_a_u, &geom.basis_d1)?
            + &xt_diag_y_dense(xmu, &c_a, &basis1_u)?
            + &xt_diag_y_dense(xmu, &c_u, &basis1_a)?
            + &xt_diag_y_dense(xmu, &c, &basis1_a_u)?;
        let h_lw = weighted_crossprod_psi_maps(
            x_ls_map,
            l_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(&geom.basis),
        )? + &weighted_crossprod_psi_maps(
            x_ls_map,
            l.view(),
            CustomFamilyPsiLinearMapRef::Dense(&basis_u),
        )? + &xt_diag_y_dense(x_ls, &l_a_u, &geom.basis)?
            + &xt_diag_y_dense(x_ls, &l_a, &basis_u)?
            + &xt_diag_y_dense(x_ls, &l_u, &basis_a)?
            + &xt_diag_y_dense(x_ls, &l, &basis_a_u)?;
        let hww_a_u = xt_diag_y_dense(&basis_a_u, &rows.w, &geom.basis)?;
        let hww_aw = xt_diag_y_dense(&basis_a, &dw_u, &geom.basis)?;
        let hww_au = xt_diag_y_dense(&basis_a, &rows.w, &basis_u)?;
        let h_ww = &hww_a_u
            + &hww_a_u.t()
            + &hww_aw
            + hww_aw.t()
            + &hww_au
            + hww_au.t()
            + &xt_diag_x_dense(&geom.basis, &dw_a_u)?;

        Ok(gaussian_pack_wiggle_joint_symmetrichessian(
            &h_mm, &h_ml, &h_mw, &h_ll, &h_lw, &h_ww,
        ))
    }

    pub(crate) fn exact_newton_joint_psi_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psi_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            &xmu,
            &x_ls,
        )
    }

    pub(crate) fn exact_newton_joint_psisecond_order_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psisecond_order_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &xmu,
            &x_ls,
        )
    }

    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psihessian_directional_derivative_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &xmu,
            &x_ls,
        )
    }
}

impl CustomFamily for GaussianLocationScaleWiggleFamily {
    pub(crate) fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    pub(crate) fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Operator-aware (see GaussianLocationScaleFamily for derivation): when
        // `use_joint_matrix_free_path` selects the workspace operator, joint
        // Hv apply is O(n · (p_t + p_ℓ + p_w)) — the row-streaming RowCoeffOperator
        // never materializes the dense (p_t + p_ℓ + p_w)² matrix.
        crate::families::location_scale_engine::location_scale_coefficient_hessian_cost(
            self.y.len() as u64,
            specs,
        )
    }

    pub(crate) fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(None);
        }
        Ok(monotone_wiggle_nonnegative_constraints(spec.design.ncols()))
    }

    pub(crate) fn post_update_block_beta(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(!block_spec.name.is_empty());
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(beta);
        }
        validate_monotone_wiggle_beta_nonnegative(
            &beta,
            "GaussianLocationScaleWiggleFamily post-update",
        )?;
        Ok(beta)
    }

    pub(crate) fn evaluate(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_mu.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        // Per-row kernel emits 6 working values into pre-allocated outputs;
        // ll is reduced via Rayon's sum. Independent across rows. Note
        // wmu == ww (both equal location_working_weight) and the mean+wiggle
        // working responses share row.location_working_shift, applied to
        // eta_mu[i] and etaw[i] respectively. The previous `q = eta_mu + etaw`
        // intermediate is inlined to avoid an extra n-vector allocation.
        let mut zmu = Array1::<f64>::zeros(n);
        let mut wmu = Array1::<f64>::zeros(n);
        let mut zls = Array1::<f64>::zeros(n);
        let mut wls = Array1::<f64>::zeros(n);
        let mut zw = Array1::<f64>::zeros(n);
        let mut ww = Array1::<f64>::zeros(n);
        pub(crate) const CHUNK: usize = 1024;
        let zmu_s = zmu
            .as_slice_memory_order_mut()
            .expect("zeros is contiguous");
        let wmu_s = wmu
            .as_slice_memory_order_mut()
            .expect("zeros is contiguous");
        let zls_s = zls
            .as_slice_memory_order_mut()
            .expect("zeros is contiguous");
        let wls_s = wls
            .as_slice_memory_order_mut()
            .expect("zeros is contiguous");
        let zw_s = zw.as_slice_memory_order_mut().expect("zeros is contiguous");
        let ww_s = ww.as_slice_memory_order_mut().expect("zeros is contiguous");
        let y_view = self.y.view();
        let w_view = self.weights.view();
        let eta_mu_view = eta_mu.view();
        let eta_ls_view = eta_ls.view();
        let etaw_view = etaw.view();
        let ll: f64 = zmu_s
            .par_chunks_mut(CHUNK)
            .zip(wmu_s.par_chunks_mut(CHUNK))
            .zip(zls_s.par_chunks_mut(CHUNK))
            .zip(wls_s.par_chunks_mut(CHUNK))
            .zip(zw_s.par_chunks_mut(CHUNK))
            .zip(ww_s.par_chunks_mut(CHUNK))
            .enumerate()
            .map(
                |(chunk_idx, (((((zmu_c, wmu_c), zls_c), wls_c), zw_c), ww_c))| {
                    let start = chunk_idx * CHUNK;
                    let mut local_ll = 0.0;
                    for local in 0..zmu_c.len() {
                        let i = start + local;
                        let q_i = eta_mu_view[i] + etaw_view[i];
                        let row = gaussian_diagonal_row_kernel(
                            y_view[i],
                            q_i,
                            eta_ls_view[i],
                            w_view[i],
                            ln2pi,
                        );
                        let w_i = row.location_working_weight;
                        let shift = row.location_working_shift;
                        zmu_c[local] = eta_mu_view[i] + shift;
                        wmu_c[local] = w_i;
                        zw_c[local] = etaw_view[i] + shift;
                        ww_c[local] = w_i;
                        zls_c[local] = row.log_sigma_working_response;
                        wls_c[local] = row.log_sigma_working_weight;
                        local_ll += row.log_likelihood;
                    }
                    local_ll
                },
            )
            .sum();

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::diagonal_checked(zmu, wmu)?,
                BlockWorkingSet::diagonal_checked(zls, wls)?,
                BlockWorkingSet::diagonal_checked(zw, ww)?,
            ],
        })
    }

    pub(crate) fn log_likelihood_only(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<f64, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_mu.len() != self.y.len()
            || eta_ls.len() != self.y.len()
            || etaw.len() != self.y.len()
            || self.weights.len() != self.y.len()
        {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let q = eta_mu + etaw;
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let mut ll = 0.0;
        for i in 0..self.y.len() {
            let sigma_i = logb_sigma_from_eta_scalar(eta_ls[i]);
            let inv_s2 = (sigma_i * sigma_i).recip();
            let r = self.y[i] - q[i];
            ll += self.weights[i] * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()));
        }
        Ok(ll)
    }

    /// Outer-only log-likelihood with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `Some`, only the sampled rows
    /// contribute; each row's per-row log-likelihood term is multiplied by
    /// `WeightedOuterRow.weight`, the Horvitz–Thompson inverse-inclusion
    /// factor 1/π_i (uniform or stratified sampling both supported), so the
    /// partial sum is an unbiased estimator of the full-data log-likelihood.
    /// When `None`, this returns the full-data `log_likelihood_only`. Inner
    /// PIRLS line searches never install the subsample option, so they
    /// continue to score the exact full-data log-likelihood.
    pub(crate) fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        let Some(subsample) = options.outer_score_subsample.as_ref() else {
            return self.log_likelihood_only(block_states);
        };
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_mu.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GaussianLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        use rayon::iter::ParallelIterator;
        let ll: f64 = subsample
            .rows
            .par_iter()
            .map(|row| {
                let i = row.index;
                let wi = self.weights[i];
                if wi == 0.0 {
                    return 0.0;
                }
                let sigma_i = logb_sigma_from_eta_scalar(eta_ls[i]);
                let inv_s2 = (sigma_i * sigma_i).recip();
                let r = self.y[i] - eta_mu[i] - etaw[i];
                row.weight * wi * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma_i.ln()))
            })
            .sum();
        Ok(ll)
    }

    pub(crate) fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    pub(crate) fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let pmu = self
            .mu_design
            .as_ref()
            .ok_or_else(|| {
                "GaussianLocationScaleWiggleFamily exact path is missing mu design".to_string()
            })?
            .ncols();
        let p_ls = self
            .log_sigma_design
            .as_ref()
            .ok_or_else(|| {
                "GaussianLocationScaleWiggleFamily exact path is missing log-sigma design"
                    .to_string()
            })?
            .ncols();
        let pw = block_states[Self::BLOCK_WIGGLE].beta.len();
        let total = pmu + p_ls + pw;
        let (start, end) = match block_idx {
            Self::BLOCK_MU => (0usize, pmu),
            Self::BLOCK_LOG_SIGMA => (pmu, pmu + p_ls),
            Self::BLOCK_WIGGLE => (pmu + p_ls, total),
            _ => return Ok(None),
        };
        if d_beta.len() != end - start {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "GaussianLocationScaleWiggleFamily block {block_idx} d_beta length mismatch: got {}, expected {}",
                d_beta.len(),
                end - start
            ) }.into());
        }
        let mut d_beta_flat = Array1::<f64>::zeros(total);
        d_beta_flat.slice_mut(s![start..end]).assign(d_beta);
        let (xmu, x_ls) = self.dense_block_designs()?;
        let d_joint = self
            .exact_newton_joint_hessian_directional_derivative_from_designs(
                block_states,
                &xmu,
                &x_ls,
                &d_beta_flat,
            )?
            .ok_or_else(|| "missing Gaussian wiggle exact joint directional Hessian".to_string())?;
        Ok(Some(d_joint.slice(s![start..end, start..end]).to_owned()))
    }

    pub(crate) fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, None)
    }

    pub(crate) fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_flat,
        )
    }

    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }

    pub(crate) fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, Some(specs))
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_flat,
        )
    }

    pub(crate) fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }

    pub(crate) fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        self.exact_newton_joint_psi_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
        )
    }

    pub(crate) fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.exact_newton_joint_psisecond_order_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_i,
            psi_j,
        )
    }

    pub(crate) fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    pub(crate) fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        Ok(Some(Arc::new(
            GaussianLocationScaleWiggleExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
            )?,
        )))
    }

    /// Outer-aware joint ψ workspace with optional row subsample.
    ///
    /// The wiggle ψ workspace shares the generic `LocationScaleJointPsiWorkspace`
    /// with the non-wiggle GLS family, and the subsample is plumbed through
    /// the trait. The wiggle's `ws_psi_*_from_parts` impls currently drop the
    /// subsample and fall back to the full-data exact wiggle ψ path; see
    /// their inline rationale and the `apply_ht_mask_*` helpers used by the
    /// non-wiggle GLS family. Storing the subsample here keeps the workspace
    /// signature uniform across both families and leaves a hook for the
    /// follow-up that refactors the wiggle inline arrays into a weights
    /// struct so HT masking can be applied in one place. Even without that
    /// refactor, the total outer score under subsampling remains an unbiased
    /// estimator of the full-data outer score: HT-unbiased LL
    /// (`log_likelihood_only_with_options`) + HT-unbiased ρ-Hessian
    /// (`exact_newton_joint_hessian_workspace_with_options`) + exact-unbiased
    /// ψ (the wiggle workspace path) = unbiased.
    pub(crate) fn exact_newton_joint_psi_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        Ok(Some(Arc::new(
            GaussianLocationScaleWiggleExactNewtonJointPsiWorkspace::new_with_subsample(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
                options.outer_score_subsample.clone(),
            )?,
        )))
    }

    pub(crate) fn block_geometry(
        &self,
        block_states: &[ParameterBlockState],
        spec: &ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        if spec.name != "wiggle" {
            return Ok((spec.design.clone(), spec.offset.clone()));
        }
        if block_states.is_empty() {
            return Err(GamlssError::UnsupportedConfiguration {
                reason: "Gaussian wiggle geometry requires mean block".to_string(),
            }
            .into());
        }
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        if eta_mu.len() != self.y.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: "Gaussian wiggle geometry input size mismatch".to_string(),
            }
            .into());
        }
        let x = self.wiggle_design(eta_mu.view())?;
        if x.ncols() != spec.design.ncols() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "Gaussian dynamic wiggle design col mismatch: got {}, expected {}",
                    x.ncols(),
                    spec.design.ncols()
                ),
            }
            .into());
        }
        let nrows = x.nrows();
        Ok((
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
            Array1::zeros(nrows),
        ))
    }

    pub(crate) fn block_geometry_is_dynamic(&self) -> bool {
        true
    }

    pub(crate) fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let workspace = GaussianLocationScaleWiggleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            xmu.into_owned(),
            x_ls.into_owned(),
        )?;
        Ok(Some(Arc::new(workspace)))
    }

    /// Outer-aware joint-Hessian workspace with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `None`, this is byte-identical
    /// to `exact_newton_joint_hessian_workspace`. When `Some`, the precomputed
    /// per-row coefficient arrays in `pieces` (`coeff_mm`, `coeff_ml`,
    /// `coeff_ll`, `coeff_mw_b`, `coeff_mw_d`, `coeff_lw_b`, `coeff_ww`) —
    /// which every downstream assembly (`hessian_dense`, `hessian_matvec`,
    /// `hessian_diagonal`) consumes row-linearly via `Xᵀ diag(W) Y` — are
    /// replaced by a Horvitz–Thompson mask: each sampled row's coefficient
    /// is multiplied by `WeightedOuterRow.weight` (the inverse-inclusion
    /// factor 1/π_i; uniform or stratified sampling both supported), and
    /// non-sampled rows are zeroed. The `basis`/`basis_d1` matrices are
    /// row-weight-independent and remain unchanged. Note that the Gaussian
    /// wiggle has one fewer cross-coefficient than the binomial wiggle
    /// (no `coeff_lw_d`) because the wiggle enters the Gaussian likelihood
    /// only through `q = η_μ + η_w` (no σ-chain). The resulting joint Hessian
    /// is an unbiased estimator of the full-data joint Hessian. Inner PIRLS
    /// never installs the option, so the inner solve continues to consume
    /// the exact full-data Hessian.
    pub(crate) fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((xmu, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let mut workspace = GaussianLocationScaleWiggleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            xmu.into_owned(),
            x_ls.into_owned(),
        )?;
        if let Some(subsample) = options.outer_score_subsample.as_ref() {
            workspace.apply_outer_subsample(subsample.rows.as_ref());
        }
        Ok(Some(Arc::new(workspace)))
    }

    /// Outer-derivative policy: declare HT-subsample capability.
    ///
    /// GaussianLocationScaleWiggleFamily overrides
    /// `log_likelihood_only_with_options` and
    /// `exact_newton_joint_hessian_workspace_with_options` to consume
    /// `options.outer_score_subsample` with per-row Horvitz–Thompson weights
    /// (each sampled row's contribution is multiplied by
    /// `WeightedOuterRow.weight = 1/π_i`; non-sampled rows are zeroed),
    /// yielding unbiased estimators of the full-data log-likelihood and
    /// joint Hessian. The ψ-workspace path is also subsample-aware via
    /// `exact_newton_joint_psi_workspace_with_options`, which threads the
    /// subsample down to per-row weight masking inside the joint-ψ second-
    /// order and directional-derivative reductions. Inner-PIRLS and final-
    /// covariance paths never install the option, so they continue to
    /// consume the exact full-data quantities.
    pub(crate) fn outer_derivative_subsample_capable(&self) -> bool {
        true
    }

    pub(crate) fn inner_coefficient_hessian_hvp_available(
        &self,
        specs: &[ParameterBlockSpec],
    ) -> bool {
        // Same gating as the workspace impl above: matrix-free fires when
        // `exact_joint_dense_block_designs` is satisfiable, which requires
        // both location and scale block designs to be present.  The wiggle
        // block is folded into the operator via the per-row pieces — its
        // presence is implied by reaching the wiggle family in the first
        // place — so the predicate matches the non-wiggle case.
        self.exact_joint_supported()
            && matches!(
                self.exact_joint_dense_block_designs(Some(specs)),
                Ok(Some(_))
            )
    }
}

/// Matrix-free joint-Hessian operator for the 3-block Gaussian
/// location-scale wiggle family. See `GaussianLocationScaleWiggleHessianRowPieces`
/// for the per-row weight structure. The matvec applies
///
///   r_μ  = D_mm u_μ + D_ml u_ls + D_mw_b (B v_w) + D_mw_d (B' v_w),
///   r_ls = D_ml u_μ + D_ll u_ls + D_lw_b (B v_w),
///   r_b  = D_mw_b u_μ + D_lw_b u_ls + D_ww (B v_w),
///   r_d  = D_mw_d u_μ,
///
/// then forms `out_w = B^T r_b + (B')^T r_d`. The ls-wiggle cross block has
/// no B' contribution because the wiggle enters the Gaussian likelihood only
/// through `q = η_μ + η_w` (no σ-chain), so the Gaussian wiggle has one
/// fewer cross-coefficient than the binomial wiggle.
pub(crate) struct GaussianLocationScaleWiggleHessianWorkspace {
    family: GaussianLocationScaleWiggleFamily,
    block_states: Vec<ParameterBlockState>,
    xmu: Arc<Array2<f64>>,
    x_ls: Arc<Array2<f64>>,
    pieces: GaussianLocationScaleWiggleHessianRowPieces,
}

impl GaussianLocationScaleWiggleHessianWorkspace {
    pub(crate) fn new(
        family: GaussianLocationScaleWiggleFamily,
        block_states: Vec<ParameterBlockState>,
        xmu: Array2<f64>,
        x_ls: Array2<f64>,
    ) -> Result<Self, String> {
        let pieces = family.wiggle_hessian_row_pieces(&block_states)?;
        Ok(Self {
            family,
            block_states,
            xmu: Arc::new(xmu),
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
    /// estimator of the full-data joint Hessian. The `basis`/`basis_d1`
    /// matrices are independent of the per-row weights and remain unchanged.
    /// The Gaussian wiggle has 7 coefficient arrays (no `coeff_lw_d`, unlike
    /// the binomial wiggle's 8) because the wiggle enters the Gaussian
    /// likelihood only through `q = η_μ + η_w` (no σ-chain).
    pub(crate) fn apply_outer_subsample(
        &mut self,
        rows: &[crate::families::marginal_slope_shared::WeightedOuterRow],
    ) {
        let n = self.pieces.coeff_mm.len();
        let mut mask_mm = Array1::<f64>::zeros(n);
        let mut mask_ml = Array1::<f64>::zeros(n);
        let mut mask_ll = Array1::<f64>::zeros(n);
        let mut mask_mw_b = Array1::<f64>::zeros(n);
        let mut mask_mw_d = Array1::<f64>::zeros(n);
        let mut mask_lw_b = Array1::<f64>::zeros(n);
        let mut maskww = Array1::<f64>::zeros(n);
        for r in rows {
            let i = r.index;
            let w = r.weight;
            mask_mm[i] = self.pieces.coeff_mm[i] * w;
            mask_ml[i] = self.pieces.coeff_ml[i] * w;
            mask_ll[i] = self.pieces.coeff_ll[i] * w;
            mask_mw_b[i] = self.pieces.coeff_mw_b[i] * w;
            mask_mw_d[i] = self.pieces.coeff_mw_d[i] * w;
            mask_lw_b[i] = self.pieces.coeff_lw_b[i] * w;
            maskww[i] = self.pieces.coeff_ww[i] * w;
        }
        self.pieces.coeff_mm = mask_mm;
        self.pieces.coeff_ml = mask_ml;
        self.pieces.coeff_ll = mask_ll;
        self.pieces.coeff_mw_b = mask_mw_b;
        self.pieces.coeff_mw_d = mask_mw_d;
        self.pieces.coeff_lw_b = mask_lw_b;
        self.pieces.coeff_ww = maskww;
    }
}

impl ExactNewtonJointHessianWorkspace for GaussianLocationScaleWiggleHessianWorkspace {
    pub(crate) fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        // Same Hv structure as `hessian_matvec`, but routed through the
        // already-existing `assemble_dense` row-pieces helper (six GEMMs:
        // h_mm, h_ml, h_mw_b, h_mw_d, h_lw, h_ww). Avoids `total` canonical-
        // basis HVPs in `MatrixFreeSpdOperator::materialize_dense_operator`,
        // which at large scale (n≈320k, p_total≈82) costs ~568s per κ-iter
        // versus ~1s for the dense build.
        let dense = self
            .pieces
            .assemble_dense(self.xmu.as_ref(), self.x_ls.as_ref())?;
        Ok(Some(dense))
    }

    pub(crate) fn hessian_matvec_available(&self) -> bool {
        true
    }

    pub(crate) fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let pw = self.pieces.basis.ncols();
        let total = pmu + p_ls + pw;
        if v.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggle matvec dimension mismatch: got {}, expected {}",
                    v.len(),
                    total
                ),
            }
            .into());
        }
        let v_mu = v.slice(s![0..pmu]);
        let v_ls = v.slice(s![pmu..pmu + p_ls]);
        let v_w = v.slice(s![pmu + p_ls..total]);

        let u_mu = fast_av(self.xmu.as_ref(), &v_mu);
        let u_ls = fast_av(self.x_ls.as_ref(), &v_ls);
        let u_b = fast_av(&self.pieces.basis, &v_w);
        let u_d = fast_av(&self.pieces.basis_d1, &v_w);

        let r_mu = &self.pieces.coeff_mm * &u_mu
            + &self.pieces.coeff_ml * &u_ls
            + &self.pieces.coeff_mw_b * &u_b
            + &self.pieces.coeff_mw_d * &u_d;
        let r_ls = &self.pieces.coeff_ml * &u_mu
            + &self.pieces.coeff_ll * &u_ls
            + &self.pieces.coeff_lw_b * &u_b;
        let r_b = &self.pieces.coeff_mw_b * &u_mu
            + &self.pieces.coeff_lw_b * &u_ls
            + &self.pieces.coeff_ww * &u_b;
        let r_d = &self.pieces.coeff_mw_d * &u_mu;

        let out_mu = fast_atv(self.xmu.as_ref(), &r_mu);
        let out_ls = fast_atv(self.x_ls.as_ref(), &r_ls);
        let out_w = fast_atv(&self.pieces.basis, &r_b) + &fast_atv(&self.pieces.basis_d1, &r_d);

        let mut out = Array1::<f64>::zeros(total);
        out.slice_mut(s![0..pmu]).assign(&out_mu);
        out.slice_mut(s![pmu..pmu + p_ls]).assign(&out_ls);
        out.slice_mut(s![pmu + p_ls..total]).assign(&out_w);
        Ok(Some(out))
    }

    pub(crate) fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let pw = self.pieces.basis.ncols();
        let total = pmu + p_ls + pw;
        // Diagonals are independent column-wise reductions: parallelize.
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let diag_mu: Vec<f64> = (0..pmu)
            .into_par_iter()
            .map(|j| {
                let col = self.xmu.column(j);
                col.iter()
                    .zip(self.pieces.coeff_mm.iter())
                    .map(|(&v, &c)| c * v * v)
                    .sum()
            })
            .collect();
        let diag_ls: Vec<f64> = (0..p_ls)
            .into_par_iter()
            .map(|j| {
                let col = self.x_ls.column(j);
                col.iter()
                    .zip(self.pieces.coeff_ll.iter())
                    .map(|(&v, &c)| c * v * v)
                    .sum()
            })
            .collect();
        let diag_w: Vec<f64> = (0..pw)
            .into_par_iter()
            .map(|j| {
                let col = self.pieces.basis.column(j);
                col.iter()
                    .zip(self.pieces.coeff_ww.iter())
                    .map(|(&v, &c)| c * v * v)
                    .sum()
            })
            .collect();
        let mut diag = Array1::<f64>::zeros(total);
        for (j, v) in diag_mu.into_iter().enumerate() {
            diag[j] = v;
        }
        for (j, v) in diag_ls.into_iter().enumerate() {
            diag[pmu + j] = v;
        }
        for (j, v) in diag_w.into_iter().enumerate() {
            diag[pmu + p_ls + j] = v;
        }
        Ok(Some(diag))
    }

    pub(crate) fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_from_designs(
                &self.block_states,
                self.xmu.as_ref(),
                self.x_ls.as_ref(),
                d_beta_flat,
            )
    }

    pub(crate) fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        self.family.gls_wiggle_directional_operator(
            &self.block_states,
            self.xmu.clone(),
            self.x_ls.clone(),
            d_beta_flat,
        )
    }

    pub(crate) fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_from_designs(
                &self.block_states,
                self.xmu.as_ref(),
                self.x_ls.as_ref(),
                d_beta_u_flat,
                d_beta_v_flat,
            )
    }

    pub(crate) fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        self.family.gls_wiggle_second_directional_operator(
            &self.block_states,
            self.xmu.clone(),
            self.x_ls.clone(),
            d_beta_u,
            d_beta_v,
        )
    }
}

impl CustomFamilyGenerative for GaussianLocationScaleWiggleFamily {
    pub(crate) fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta_mu = &block_states[Self::BLOCK_MU].eta;
        let eta_wiggle = &block_states[Self::BLOCK_WIGGLE].eta;
        let eta_log_sigma = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let n = eta_mu.len();
        let mean = gamlss_rowwise_map(n, |i| eta_mu[i] + eta_wiggle[i]);
        let sigma = gamlss_rowwise_map(n, |i| logb_sigma_from_eta_scalar(eta_log_sigma[i]));
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Gaussian { sigma },
        })
    }
}

pub(crate) fn expect_single_block<'a>(
    block_states: &'a [ParameterBlockState],
    family_name: &str,
) -> Result<&'a ParameterBlockState, String> {
    if block_states.len() != 1 {
        return Err(GamlssError::DimensionMismatch {
            reason: format!("{family_name} expects 1 block, got {}", block_states.len()),
        }
        .into());
    }
    Ok(&block_states[0])
}

#[derive(Clone)]
pub struct BinomialMeanWiggleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    /// Resource policy threaded into PsiDesignMap construction during
    /// exact-Newton joint psi evaluation. Defaults to
    /// `ResourcePolicy::default_library()` when the family is built without
    /// an explicit policy.
    pub policy: crate::resource::ResourcePolicy,
}

pub(crate) struct BinomialMeanWiggleGeometry {
    basis: Array2<f64>,
    basis_d1: Array2<f64>,
    basis_d2: Array2<f64>,
    basis_d3: Array2<f64>,
    dq_dq0: Array1<f64>,
    d2q_dq02: Array1<f64>,
    d3q_dq03: Array1<f64>,
    d4q_dq04: Array1<f64>,
}

pub(crate) struct BinomialMeanWiggleJointPsiDirection {
    x_eta_psi: Option<Array2<f64>>,
    z_eta_psi: Array1<f64>,
}

impl BinomialMeanWiggleFamily {
    pub const BLOCK_ETA: usize = 0;
    pub const BLOCK_WIGGLE: usize = 1;

    pub(crate) fn wiggle_basiswith_options(
        &self,
        q0: ArrayView1<'_, f64>,
        options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            options.derivative_order,
        )
    }

    pub(crate) fn wiggle_design(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        self.wiggle_basiswith_options(q0, BasisOptions::value())
    }

    pub(crate) fn wiggle_dq_dq0(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d_constrained = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        if d_constrained.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d_constrained.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d_constrained.dot(&beta_link_wiggle) + 1.0)
    }

    pub(crate) fn wiggle_d2q_dq02(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d2 = self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        if d2.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle second-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d2.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d2.dot(&beta_link_wiggle))
    }

    pub(crate) fn wiggle_d3basis_constrained(
        &self,
        q0: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(q0, &self.wiggle_knots, self.wiggle_degree, 3)
    }

    pub(crate) fn wiggle_d3q_dq03(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d3 = self.wiggle_d3basis_constrained(q0)?;
        if d3.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle third-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d3.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d3.dot(&beta_link_wiggle))
    }

    pub(crate) fn wiggle_d4q_dq04(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d4 = monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            4,
        )?;
        if d4.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle fourth-derivative/beta mismatch: basis has {} columns but beta_link_wiggle has {} coefficients",
                d4.ncols(),
                beta_link_wiggle.len()
            ) }.into());
        }
        Ok(d4.dot(&beta_link_wiggle))
    }

    pub(crate) fn wiggle_geometry(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<BinomialMeanWiggleGeometry, String> {
        let basis = self.wiggle_design(q0)?;
        let basis_d1 = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        let basis_d2 = self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        let basis_d3 = self.wiggle_d3basis_constrained(q0)?;
        let dq_dq0 = self.wiggle_dq_dq0(q0, beta_link_wiggle)?;
        let d2q_dq02 = self.wiggle_d2q_dq02(q0, beta_link_wiggle)?;
        let d3q_dq03 = self.wiggle_d3q_dq03(q0, beta_link_wiggle)?;
        let d4q_dq04 = self.wiggle_d4q_dq04(q0, beta_link_wiggle)?;
        Ok(BinomialMeanWiggleGeometry {
            basis,
            basis_d1,
            basis_d2,
            basis_d3,
            dq_dq0,
            d2q_dq02,
            d3q_dq03,
            d4q_dq04,
        })
    }

    pub(crate) fn neglog_q_derivatives(
        &self,
        y: f64,
        weight: f64,
        q: f64,
    ) -> Result<(f64, f64, f64), String> {
        let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q)
            .map_err(|e| format!("fixed-link wiggle inverse-link evaluation failed: {e}"))?;
        // Pass μ RAW: the dispatch returns the exact q-derivatives of the
        // evaluated loss for every representable μ in (0,1) and handles the
        // saturated boundary itself. See binomial_location_scalerow (#948).
        Ok(binomial_neglog_q_derivatives_dispatch(
            y,
            weight,
            q,
            jet.mu,
            jet.d1,
            jet.d2,
            jet.d3,
            &self.link_kind,
        ))
    }

    pub(crate) fn neglog_q_fourth_derivative(
        &self,
        y: f64,
        weight: f64,
        q: f64,
    ) -> Result<f64, String> {
        let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q)
            .map_err(|e| format!("fixed-link wiggle inverse-link evaluation failed: {e}"))?;
        // Pass μ RAW — see neglog_q_derivatives above (#948).
        binomial_neglog_q_fourth_derivative_dispatch(
            y,
            weight,
            q,
            jet.mu,
            jet.d1,
            jet.d2,
            jet.d3,
            &self.link_kind,
        )
    }

    pub(crate) fn dense_eta_design_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<Cow<'a, Array2<f64>>, String> {
        if specs.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 specs, got {}",
                    specs.len()
                ),
            }
            .into());
        }
        Ok(match specs[Self::BLOCK_ETA].design.as_dense_ref() {
            Some(d) => Cow::Borrowed(d),
            None => Cow::Owned(
                specs[Self::BLOCK_ETA]
                    .design
                    .try_to_dense_with_policy(
                        &self.policy.material_policy(),
                        "BinomialMeanWiggle dense_eta_design_fromspecs eta",
                    )
                    .map_err(|e| e.to_string())?
                    .as_ref()
                    .clone(),
            ),
        })
    }

    pub(crate) fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_eta: &Array2<f64>,
    ) -> Result<Option<BinomialMeanWiggleJointPsiDirection>, String> {
        if block_states.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialMeanWiggleFamily joint psi direction expects 2 blocks and 2 derivative block lists, got {} and {}",
                block_states.len(),
                derivative_blocks.len()
            ) }.into());
        }
        let n = self.y.len();
        let p_eta = x_eta.ncols();
        let beta_eta = &block_states[Self::BLOCK_ETA].beta;
        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for deriv in block_derivs {
                if global == psi_index {
                    if block_idx != Self::BLOCK_ETA {
                        return Ok(None);
                    }
                    let x_eta_psi_map = resolve_custom_family_x_psi_map(
                        deriv,
                        n,
                        p_eta,
                        0..n,
                        "BinomialMeanWiggleFamily eta",
                        &self.policy,
                    )?;
                    let x_eta_psi = x_eta_psi_map.row_chunk(0..n)?;
                    let z_eta_psi = x_eta_psi.dot(beta_eta);
                    return Ok(Some(BinomialMeanWiggleJointPsiDirection {
                        x_eta_psi: Some(x_eta_psi),
                        z_eta_psi,
                    }));
                }
                global += 1;
            }
        }
        Ok(None)
    }

    pub(crate) fn exact_newton_joint_psi_action(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        p_eta: usize,
    ) -> Result<Option<(CustomFamilyPsiDesignAction, Array1<f64>)>, String> {
        if block_states.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialMeanWiggleFamily joint psi action expects 2 blocks and 2 derivative block lists, got {} and {}",
                block_states.len(),
                derivative_blocks.len()
            ) }.into());
        }
        let n = self.y.len();
        let beta_eta = &block_states[Self::BLOCK_ETA].beta;
        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for deriv in block_derivs {
                if global == psi_index {
                    if block_idx != Self::BLOCK_ETA {
                        return Ok(None);
                    }
                    let action = match CustomFamilyPsiDesignAction::from_first_derivative(
                        deriv,
                        n,
                        p_eta,
                        0..n,
                        "BinomialMeanWiggleFamily eta",
                    ) {
                        Ok(action) => action,
                        Err(_) => return Ok(None),
                    };
                    let z_eta_psi = action.forward_mul(beta_eta.view());
                    return Ok(Some((action, z_eta_psi)));
                }
                global += 1;
            }
        }
        Ok(None)
    }

    pub(crate) fn bmw_static_hessian_operator(
        &self,
        block_states: &[ParameterBlockState],
        x_eta_arc: Arc<Array2<f64>>,
    ) -> Result<Arc<RowCoeffOperator>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta_arc.ncols();
        let pw = geom.basis.ncols();
        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_ww = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, _) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            coeff_eta[row] = hessian_coeff_fromobjective_q_terms(m1, m2, a, a, b);
            coeff_etaw_b[row] = m2 * a;
            coeff_etaw_d1[row] = m1;
            coeff_ww[row] = m2;
        }
        Ok(Arc::new(RowCoeffOperator::from_directions(
            vec![p_eta, pw],
            vec![
                (0, x_eta_arc),
                (1, Arc::new(geom.basis)),
                (1, Arc::new(geom.basis_d1)),
            ],
            vec![
                (0, 0, coeff_eta),
                (0, 1, coeff_etaw_b),
                (0, 2, coeff_etaw_d1),
                (1, 1, coeff_ww),
            ],
            n,
        )))
    }

    pub(crate) fn bmw_directional_operator(
        &self,
        block_states: &[ParameterBlockState],
        x_eta_arc: Arc<Array2<f64>>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta_arc.ncols();
        let pw = geom.basis.ncols();
        let total = p_eta + pw;
        if d_beta_flat.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily joint d_beta length mismatch: got {}, expected {}",
                    d_beta_flat.len(),
                    total
                ),
            }
            .into());
        }
        let u_eta = d_beta_flat.slice(s![0..p_eta]).to_owned();
        let uw = d_beta_flat.slice(s![p_eta..total]).to_owned();
        let xi = fast_av(x_eta_arc.as_ref(), &u_eta);
        let phi = fast_av(&geom.basis, &uw);
        let basis1_u = fast_av(&geom.basis_d1, &uw);
        let basis2_u = fast_av(&geom.basis_d2, &uw);

        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d2 = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let q_u = a * xi[row] + phi[row];
            let a_u = b * xi[row] + basis1_u[row];
            let b_u = c * xi[row] + basis2_u[row];
            coeff_eta[row] = directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, q_u, a, a, b, a_u, a_u, b_u,
            );
            coeff_etaw_b[row] = m3 * q_u * a + m2 * a_u;
            coeff_etaw_d1[row] = m2 * (a * xi[row] + q_u);
            coeff_etaw_d2[row] = m1 * xi[row];
            coeff_ww_bb[row] = m3 * q_u;
            coeff_ww_db[row] = m2 * xi[row];
        }
        Ok(Some(Arc::new(RowCoeffOperator::from_directions(
            vec![p_eta, pw],
            vec![
                (0, x_eta_arc),
                (1, Arc::new(geom.basis)),
                (1, Arc::new(geom.basis_d1)),
                (1, Arc::new(geom.basis_d2)),
            ],
            vec![
                (0, 0, coeff_eta),
                (0, 1, coeff_etaw_b),
                (0, 2, coeff_etaw_d1),
                (0, 3, coeff_etaw_d2),
                (1, 1, coeff_ww_bb),
                (1, 2, coeff_ww_db),
            ],
            n,
        ))))
    }

    pub(crate) fn bmw_second_directional_operator(
        &self,
        block_states: &[ParameterBlockState],
        x_eta_arc: Arc<Array2<f64>>,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta_arc.ncols();
        let pw = geom.basis.ncols();
        let total = p_eta + pw;
        if d_beta_u_flat.len() != total || d_beta_v_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialMeanWiggleFamily joint second d_beta length mismatch: got {} and {}, expected {}",
                d_beta_u_flat.len(),
                d_beta_v_flat.len(),
                total
            ) }.into());
        }
        let u_eta = d_beta_u_flat.slice(s![0..p_eta]).to_owned();
        let v_eta = d_beta_v_flat.slice(s![0..p_eta]).to_owned();
        let uw = d_beta_u_flat.slice(s![p_eta..total]).to_owned();
        let vw = d_beta_v_flat.slice(s![p_eta..total]).to_owned();

        let xi_u = fast_av(x_eta_arc.as_ref(), &u_eta);
        let xi_v = fast_av(x_eta_arc.as_ref(), &v_eta);
        let phi_u = fast_av(&geom.basis, &uw);
        let phi_v = fast_av(&geom.basis, &vw);
        let b1u = fast_av(&geom.basis_d1, &uw);
        let b1v = fast_av(&geom.basis_d1, &vw);
        let b2u = fast_av(&geom.basis_d2, &uw);
        let b2v = fast_av(&geom.basis_d2, &vw);
        let b3u = fast_av(&geom.basis_d3, &uw);
        let b3v = fast_av(&geom.basis_d3, &vw);

        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d2 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d3 = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);
        let mut coeff_ww_ddb = Array1::<f64>::zeros(n);
        let mut coeff_ww_dd = Array1::<f64>::zeros(n);

        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let m4 = self.neglog_q_fourth_derivative(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let d = geom.d4q_dq04[row];

            let q_u = a * xi_u[row] + phi_u[row];
            let a_u = b * xi_u[row] + b1u[row];
            let b_u = c * xi_u[row] + b2u[row];
            let q_v = a * xi_v[row] + phi_v[row];
            let a_v = b * xi_v[row] + b1v[row];
            let b_v = c * xi_v[row] + b2v[row];
            let q_uv = b * xi_u[row] * xi_v[row] + b1u[row] * xi_v[row] + b1v[row] * xi_u[row];
            let a_uv = c * xi_u[row] * xi_v[row] + b2u[row] * xi_v[row] + b2v[row] * xi_u[row];
            let b_uv = d * xi_u[row] * xi_v[row] + b3u[row] * xi_v[row] + b3v[row] * xi_u[row];

            coeff_eta[row] = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, q_u, q_v, q_uv, a, a, b, a_u, a_v, a_u, a_v, a_uv, a_uv, b_u, b_v,
                b_uv,
            );
            let d2_c_b = m4 * q_u * q_v * a + m3 * (q_uv * a + q_u * a_v + q_v * a_u) + m2 * a_uv;
            let dc_b_u = m3 * q_u * a + m2 * a_u;
            let dc_b_v = m3 * q_v * a + m2 * a_v;
            let c_b_static = m2 * a;
            let d2_c_b1 = m3 * q_u * q_v + m2 * q_uv;
            let dc_b1_u = m2 * q_u;
            let dc_b1_v = m2 * q_v;

            coeff_etaw_b[row] = d2_c_b;
            coeff_etaw_d1[row] = dc_b_u * xi_v[row] + dc_b_v * xi_u[row] + d2_c_b1;
            coeff_etaw_d2[row] =
                c_b_static * xi_u[row] * xi_v[row] + dc_b1_u * xi_v[row] + dc_b1_v * xi_u[row];
            coeff_etaw_d3[row] = m1 * xi_u[row] * xi_v[row];

            let dw = m2;
            let dw_u = m3 * q_u;
            let dw_v = m3 * q_v;
            let dw_uv = m4 * q_u * q_v + m3 * q_uv;
            let xixj = xi_u[row] * xi_v[row];
            coeff_ww_bb[row] = dw_uv;
            coeff_ww_db[row] = dw_v * xi_u[row] + dw_u * xi_v[row];
            coeff_ww_ddb[row] = dw * xixj;
            coeff_ww_dd[row] = 2.0 * dw * xixj;
        }

        Ok(Some(Arc::new(RowCoeffOperator::from_directions(
            vec![p_eta, pw],
            vec![
                (0, x_eta_arc),
                (1, Arc::new(geom.basis)),
                (1, Arc::new(geom.basis_d1)),
                (1, Arc::new(geom.basis_d2)),
                (1, Arc::new(geom.basis_d3)),
            ],
            vec![
                (0, 0, coeff_eta),
                (0, 1, coeff_etaw_b),
                (0, 2, coeff_etaw_d1),
                (0, 3, coeff_etaw_d2),
                (0, 4, coeff_etaw_d3),
                (1, 1, coeff_ww_bb),
                (1, 2, coeff_ww_db),
                (1, 3, coeff_ww_ddb),
                (2, 2, coeff_ww_dd),
            ],
            n,
        ))))
    }

    /// Build the [`BlockEffectiveJacobian`] for block `block_idx`.
    ///
    /// `BinomialMeanWiggle` has a single location output (n_outputs = 1):
    /// - block 0 (eta):    output 0 = design rows
    /// - block 1 (wiggle): all zeros (nonlinear link modulation)
    pub fn block_effective_jacobian(
        specs: &[ParameterBlockSpec],
        block_idx: usize,
    ) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
        crate::util::block_jacobian::AdditiveWiggleBlockLayout {
            family: "BinomialMeanWiggleFamily",
            n_outputs: 1,
            additive_blocks: &[Self::BLOCK_ETA],
            wiggle_block: Some(Self::BLOCK_WIGGLE),
        }
        .block_effective_jacobian(specs, block_idx)
    }
}

impl CustomFamily for BinomialMeanWiggleFamily {
    pub(crate) fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    /// The binomial mean link-wiggle refit must NOT carry the full-span
    /// Jeffreys/Firth augmentation, for the same structural reason
    /// `GaussianLocationScaleWiggleFamily` opts out (#684–#688) — and the
    /// binomial wiggle hits it harder. This is a *second-stage* refit: the
    /// pilot binomial mean fit has already converged through the ordinary
    /// PIRLS path (which is itself un-Firthed unless the user opts in — the
    /// standard binomial fit logs `firth=false` / `jeffreys_logdet=none`), so
    /// the wiggle refit only adds a *penalized*, *monotone-constrained*
    /// I-spline link-shape correction `q = η + B(η)·β_w` around an
    /// already-finite mode. Two failure modes follow from leaving the term on
    /// (default `true`):
    ///
    /// 1. **Phantom stationarity residual.** When `H_pen` is full-rank and
    ///    well-conditioned (the normal case — e.g. `cond ≈ 5.5e2` on the #872
    ///    pure-probit repro) the Jeffreys gate smooth-steps the curvature
    ///    `H_Φ → 0`, but the matching score `∇Φ` does not vanish in lock-step,
    ///    so it leaks a nonzero `|∇L − Sβ + ∇Φ|` into the inner joint-Newton
    ///    KKT residual. The certificate then refuses every iterate and the
    ///    outer REML rejects all seeds (exactly the #684–#688 abort signature).
    /// 2. **Saturation barrier / divergence.** `−Φ = −½log|I_J|` is folded into
    ///    the objective and `∇Φ ∝ I_J⁻¹` into the gradient. The I-spline warp
    ///    can drive the binomial linear predictor toward saturation, where the
    ///    reduced Fisher information `I_J` goes singular: `−Φ → +∞` and
    ///    `∇Φ → ∞`. The augmented objective grows a barrier that the joint
    ///    Newton diverges into — the #872 repro runs the full 1200-cycle budget
    ///    with the augmented objective pinned at ~4.6e9 and the augmented
    ///    residual at ~5.8e9 while the plain data gradient is only ~2.3e2,
    ///    aborting the documented `link(type=flexible(...)) + linkwiggle(...)`
    ///    fit.
    ///
    /// Separation robustness is not lost: the wiggle block carries both a
    /// difference penalty (λ selected by REML) and a hard non-negativity
    /// constraint, and the underlying mean is fit by the pilot; a penalized,
    /// constrained refit around a finite pilot mode does not run away to
    /// `β → ∞` the way an unpenalized MLE can. Turning the term off here makes
    /// the wiggle refit consistent with the un-Firthed pilot and removes the
    /// phantom residual that blocked convergence.
    pub(crate) fn joint_jeffreys_term_required(&self) -> bool {
        false
    }

    pub(crate) fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // The mean-wiggle Hessian is exposed as a row-coefficient operator,
        // so the hot representation cost is one Θ(n · (p_eta + p_w)) HVP
        // rather than dense Θ(n · (p_eta + p_w)^2) assembly.
        let p_total = specs
            .iter()
            .map(|s| s.design.ncols() as u64)
            .fold(0u64, |acc, p| acc.saturating_add(p));
        (self.y.len() as u64).saturating_mul(p_total.max(1))
    }

    pub(crate) fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(None);
        }
        Ok(monotone_wiggle_nonnegative_constraints(spec.design.ncols()))
    }

    pub(crate) fn post_update_block_beta(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(!block_spec.name.is_empty());
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(beta);
        }
        validate_monotone_wiggle_beta_nonnegative(&beta, "BinomialMeanWiggleFamily post-update")?;
        Ok(beta)
    }

    pub(crate) fn evaluate(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let dq_dq0 = self.wiggle_dq_dq0(eta.view(), betaw.view())?;
        if dq_dq0.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily dq/dq0 length mismatch: got {}, expected {}",
                    dq_dq0.len(),
                    n
                ),
            }
            .into());
        }

        let mut ll = 0.0;
        let mut z_eta = Array1::<f64>::zeros(n);
        let mut w_eta = Array1::<f64>::zeros(n);
        let mut z_wiggle = Array1::<f64>::zeros(n);
        let mut w_wiggle = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q = eta[i] + etaw[i];
            let (mu_q, d1_q) = inverse_link_mu_d1_for_inverse_link(&self.link_kind, q)
                .map_err(|e| format!("fixed-link wiggle inverse-link evaluation failed: {e}"))?;
            let yi = self.y[i];
            let wi = self.weights[i];
            ll += binomial_location_scale_log_likelihood(yi, wi, q, &self.link_kind, mu_q)?;

            let mu = mu_q.clamp(1e-12, 1.0 - 1e-12);
            let var = (mu * (1.0 - mu)).max(MIN_PROB);
            let dmu_deta = d1_q * dq_dq0[i];
            let dmu_dw = d1_q;
            if wi == 0.0 || !var.is_finite() {
                z_eta[i] = eta[i];
                z_wiggle[i] = etaw[i];
                continue;
            }

            if dmu_deta.is_finite() {
                w_eta[i] = floor_positiveweight(wi * (dmu_deta * dmu_deta / var), MIN_WEIGHT);
                z_eta[i] = eta[i] + (yi - mu) / signedwith_floor(dmu_deta, MIN_DERIV);
            } else {
                z_eta[i] = eta[i];
            }

            if dmu_dw.is_finite() {
                w_wiggle[i] = floor_positiveweight(wi * (dmu_dw * dmu_dw / var), MIN_WEIGHT);
                z_wiggle[i] = etaw[i] + (yi - mu) / signedwith_floor(dmu_dw, MIN_DERIV);
            } else {
                z_wiggle[i] = etaw[i];
            }
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::diagonal_checked(z_eta, w_eta)?,
                BlockWorkingSet::diagonal_checked(z_wiggle, w_wiggle)?,
            ],
        })
    }

    pub(crate) fn block_geometry(
        &self,
        block_states: &[ParameterBlockState],
        spec: &ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        if spec.name != "wiggle" {
            return Ok((spec.design.clone(), spec.offset.clone()));
        }
        if block_states.is_empty() {
            return Err(GamlssError::UnsupportedConfiguration {
                reason: "wiggle geometry requires eta block".to_string(),
            }
            .into());
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        if eta.len() != self.y.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily eta size mismatch".to_string(),
            }
            .into());
        }
        let x = self.wiggle_design(eta.view())?;
        if x.ncols() != spec.design.ncols() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "dynamic wiggle design col mismatch: got {}, expected {}",
                    x.ncols(),
                    spec.design.ncols()
                ),
            }
            .into());
        }
        let nrows = x.nrows();
        Ok((
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
            Array1::zeros(nrows),
        ))
    }

    pub(crate) fn block_geometry_is_dynamic(&self) -> bool {
        true
    }

    pub(crate) fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let x_eta = self.dense_eta_design_fromspecs(specs)?.into_owned();
        let workspace =
            BinomialMeanWiggleHessianWorkspace::new(self.clone(), block_states.to_vec(), x_eta)?;
        Ok(Some(Arc::new(workspace)))
    }

    pub(crate) fn inner_coefficient_hessian_hvp_available(
        &self,
        specs: &[ParameterBlockSpec],
    ) -> bool {
        self.dense_eta_design_fromspecs(specs).is_ok()
    }

    pub(crate) fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let x_eta = self.dense_eta_design_fromspecs(specs)?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta.ncols();
        let pw = geom.basis.ncols();
        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_ww = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, _) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            coeff_eta[row] = hessian_coeff_fromobjective_q_terms(m1, m2, a, a, b);
            coeff_etaw_b[row] = m2 * a;
            coeff_etaw_d1[row] = m1;
            coeff_ww[row] = m2;
        }
        let h_eta_eta = xt_diag_x_dense(&x_eta, &coeff_eta)?;
        let h_eta_w = xt_diag_y_dense(&x_eta, &coeff_etaw_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d1, &geom.basis_d1)?;
        let h_ww = xt_diag_x_dense(&geom.basis, &coeff_ww)?;
        assert_eq!(h_eta_eta.nrows(), p_eta);
        assert_eq!(h_ww.nrows(), pw);
        Ok(Some(binomial_pack_mean_wiggle_joint_symmetrichessian(
            &h_eta_eta, &h_eta_w, &h_ww,
        )))
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let x_eta = self.dense_eta_design_fromspecs(specs)?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta.ncols();
        let pw = geom.basis.ncols();
        if d_beta_flat.len() != p_eta + pw {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily joint d_beta length mismatch: got {}, expected {}",
                    d_beta_flat.len(),
                    p_eta + pw
                ),
            }
            .into());
        }
        let u_eta = d_beta_flat.slice(s![0..p_eta]).to_owned();
        let uw = d_beta_flat.slice(s![p_eta..p_eta + pw]).to_owned();
        let xi = x_eta.dot(&u_eta);
        let phi = geom.basis.dot(&uw);
        let basis1_u = geom.basis_d1.dot(&uw);
        let basis2_u = geom.basis_d2.dot(&uw);

        let mut coeff_eta = Array1::<f64>::zeros(n);
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d2 = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let q_u = a * xi[row] + phi[row];
            let a_u = b * xi[row] + basis1_u[row];
            let b_u = c * xi[row] + basis2_u[row];
            coeff_eta[row] = directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, q_u, a, a, b, a_u, a_u, b_u,
            );
            coeff_etaw_b[row] = m3 * q_u * a + m2 * a_u;
            coeff_etaw_d1[row] = m2 * (a * xi[row] + q_u);
            coeff_etaw_d2[row] = m1 * xi[row];
            coeff_ww_bb[row] = m3 * q_u;
            coeff_ww_db[row] = m2 * xi[row];
        }

        let d_h_eta_eta = xt_diag_x_dense(&x_eta, &coeff_eta)?;
        let d_h_eta_w = xt_diag_y_dense(&x_eta, &coeff_etaw_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d1, &geom.basis_d1)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d2, &geom.basis_d2)?;
        let a_ww = xt_diag_y_dense(&geom.basis_d1, &coeff_ww_db, &geom.basis)?;
        let d_h_ww = xt_diag_x_dense(&geom.basis, &coeff_ww_bb)? + &a_ww + a_ww.t();
        Ok(Some(binomial_pack_mean_wiggle_joint_symmetrichessian(
            &d_h_eta_eta,
            &d_h_eta_w,
            &d_h_ww,
        )))
    }

    /// Exact second-order directional derivative D²H[u,v] of the joint Hessian
    /// for the BinomialMeanWiggle two-block model (eta, wiggle).
    ///
    /// # Mathematical derivation
    ///
    /// The negative log-likelihood Hessian element for indices (a, b) in the
    /// joint coefficient vector is:
    ///
    ///   H_ab = m2 * q_a * q_b + m1 * q_ab
    ///
    /// where m_k = d^k F / dq^k (k-th derivative of the negative log-likelihood
    /// w.r.t. the effective predictor q), q_a = dq/d(beta_a), and q_ab =
    /// d²q/(d(beta_a) d(beta_b)).
    ///
    /// The effective predictor is q = q0 + w(q0) where q0 = X_eta * beta_eta
    /// and w(q0) = B(q0) * beta_w is the link wiggle.  Write:
    ///   a = dq/dq0 = 1 + B'·beta_w       (geometry first derivative)
    ///   b = d²q/dq0² = B''·beta_w         (geometry second derivative)
    ///   c = d³q/dq0³ = B'''·beta_w        (geometry third derivative)
    ///   d = d⁴q/dq0⁴ = B''''·beta_w       (geometry fourth derivative)
    ///
    /// For a perturbation direction u = (u_eta, u_w), the chain-rule
    /// perturbations are:
    ///   q_u   = a·xi_u + phi_u             (first-order predictor perturbation)
    ///   a_u   = b·xi_u + basis1_u          (perturbation of geometry factor a)
    ///   b_u   = c·xi_u + basis2_u          (perturbation of geometry factor b)
    ///   c_u   = d·xi_u + basis3_u          (perturbation of geometry factor c)
    ///
    /// where xi_u = X_eta·u_eta, phi_u = B·u_w, basis_k_u = B^(k)·u_w.
    ///
    /// Mixed second-order perturbations (u,v) are:
    ///   q_uv  = b·xi_u·xi_v + basis1_u·xi_v + basis1_v·xi_u
    ///   a_uv  = c·xi_u·xi_v + basis2_u·xi_v + basis2_v·xi_u
    ///   b_uv  = d·xi_u·xi_v + basis3_u·xi_v + basis3_v·xi_u
    ///
    /// ## Block decomposition
    ///
    /// **eta-eta block** (X_eta' diag(coeff) X_eta):
    ///   The Hessian element for eta indices (i,j) factors as
    ///     H(eta_i, eta_j) = [m2·a² + m1·b] · x_eta(i)·x_eta(j)
    ///   so D²H_eta_eta[u,v] = X_eta' diag(coeff_eta) X_eta
    ///   where coeff_eta uses `second_directionalhessian_coeff_fromobjective_q_terms`
    ///   with q_a=a, q_b=a, q_ab=b and their chain-rule perturbations.
    ///
    /// **eta-w block** (X_eta' diag(...) [B, B', B'', B''']):
    ///   The static Hessian is:
    ///     H(eta_i, w_j) = (m2·a)·x_eta(i)·B_j + m1·x_eta(i)·B'_j
    ///   Taking D²[u,v] requires differentiating both the scalar coefficients
    ///   (m2·a, m1) and the basis matrices (B, B' depend on q0 via the chain
    ///   rule dB_j/du = B'_j·xi_u).  The full product rule gives four basis-matrix
    ///   tiers: B, B', B'', B'''.
    ///
    /// **w-w block** (B' diag(...) B, etc.):
    ///   The static Hessian is H(w_i, w_j) = m2·B_i·B_j.
    ///   D²[u,v] expands via the product rule on m2, B_i, B_j, each of which
    ///   depends on beta through q and q0.  This gives terms involving
    ///   B·B, B'·B, B'·B', and B''·B (all symmetrised).
    pub(crate) fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let x_eta = self.dense_eta_design_fromspecs(specs)?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta.ncols();
        let pw = geom.basis.ncols();
        let total = p_eta + pw;
        if d_beta_u_flat.len() != total || d_beta_v_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialMeanWiggleFamily joint second d_beta length mismatch: got {} and {}, expected {}",
                d_beta_u_flat.len(),
                d_beta_v_flat.len(),
                total
            ) }.into());
        }

        // Split directions into eta and wiggle components.
        let u_eta = d_beta_u_flat.slice(s![0..p_eta]).to_owned();
        let v_eta = d_beta_v_flat.slice(s![0..p_eta]).to_owned();
        let uw = d_beta_u_flat.slice(s![p_eta..total]).to_owned();
        let vw = d_beta_v_flat.slice(s![p_eta..total]).to_owned();

        // Per-row linear-predictor perturbations from each direction.
        let xi_u = x_eta.dot(&u_eta); // eta perturbation in direction u
        let xi_v = x_eta.dot(&v_eta); // eta perturbation in direction v
        let phi_u = geom.basis.dot(&uw); // direct wiggle basis, direction u
        let phi_v = geom.basis.dot(&vw); // direct wiggle basis, direction v
        let b1u = geom.basis_d1.dot(&uw); // first-derivative basis, direction u
        let b1v = geom.basis_d1.dot(&vw);
        let b2u = geom.basis_d2.dot(&uw); // second-derivative basis, direction u
        let b2v = geom.basis_d2.dot(&vw);
        let b3u = geom.basis_d3.dot(&uw); // third-derivative basis, direction u
        let b3v = geom.basis_d3.dot(&vw);

        // Per-row chain-rule perturbations of q, a = dq/dq0, b = d²q/dq0²:
        //   q_u = a·xi_u + phi_u
        //   a_u = b·xi_u + basis1_u
        //   b_u = c·xi_u + basis2_u
        //   c_u = d·xi_u + basis3_u
        // Mixed second-order perturbations:
        //   q_uv = b·xi_u·xi_v + basis1_u·xi_v + basis1_v·xi_u
        //   a_uv = c·xi_u·xi_v + basis2_u·xi_v + basis2_v·xi_u
        //   b_uv = d·xi_u·xi_v + basis3_u·xi_v + basis3_v·xi_u

        // Scaled basis matrices for the cross-product terms in the w-w and eta-w
        // blocks (same pattern as GaussianLocationScaleWiggleFamily).
        let basis_u = scale_matrix_rows(&geom.basis_d1, &xi_u)?; // dB/du = B'·xi_u
        let basis_v = scale_matrix_rows(&geom.basis_d1, &xi_v)?; // dB/dv = B'·xi_v
        let basis_uv = scale_matrix_rows(&geom.basis_d2, &(&xi_u * &xi_v))?; // d²B/dudv = B''·xi_u·xi_v
        // Per-row coefficient arrays for assembling the block-matrix products.
        let mut coeff_eta = Array1::<f64>::zeros(n);

        // Coefficients for the eta-w block: X_eta' diag(c_*) M where M ∈ {B, B', B'', B'''}
        //
        // The static cross-Hessian is:
        //   H(eta_i, w_j) = (m2·a)·x_i·B_j + m1·x_i·B'_j
        // where B_j and B'_j are row evaluations of basis column j.
        //
        // Write C_B = m2·a (scalar coefficient multiplying B in the cross block)
        // and   C_B1 = m1  (scalar coefficient multiplying B' in the cross block).
        //
        // Product rule on C_B·B:
        //   d(C_B·B)/du = (dC_B/du)·B + C_B·B'·xi_u
        //   d²(C_B·B)/dudv = (d²C_B/dudv)·B + (dC_B/du)·B'·xi_v
        //                   + (dC_B/dv)·B'·xi_u + C_B·B''·xi_u·xi_v
        //
        // Product rule on C_B1·B':
        //   d²(C_B1·B')/dudv = (d²C_B1/dudv)·B' + (dC_B1/du)·B''·xi_v
        //                     + (dC_B1/dv)·B''·xi_u + C_B1·B'''·xi_u·xi_v
        //
        // Derivatives of the scalar coefficients:
        //   C_B  = m2·a
        //   dC_B/du  = m3·q_u·a + m2·a_u
        //   dC_B/dv  = m3·q_v·a + m2·a_v
        //   d²C_B/dudv = m4·q_u·q_v·a + m3·(q_uv·a + q_u·a_v + q_v·a_u) + m2·a_uv
        //
        //   C_B1 = m1
        //   dC_B1/du = m2·q_u
        //   dC_B1/dv = m2·q_v
        //   d²C_B1/dudv = m3·q_u·q_v + m2·q_uv
        //
        // Grouping by basis-matrix tier:
        //   B:   d²C_B/dudv
        //   B':  (dC_B/du)·xi_v + (dC_B/dv)·xi_u + d²C_B1/dudv
        //   B'': C_B·xi_u·xi_v + (dC_B1/du)·xi_v + (dC_B1/dv)·xi_u
        //   B''': C_B1·xi_u·xi_v
        let mut coeff_etaw_b = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d1 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d2 = Array1::<f64>::zeros(n);
        let mut coeff_etaw_d3 = Array1::<f64>::zeros(n);

        // Coefficients for the w-w block.
        //
        // The static w-w Hessian is:
        //   H(w_i, w_j) = m2·B_i·B_j
        //
        // Note: there is no m1·q_ij term because d²q/(d(beta_w_i) d(beta_w_j)) = 0
        // (the basis vectors B_i enter q linearly in beta_w).
        //
        // Product rule on m2·B_i·B_j, treating each factor as depending on beta:
        //   d²(m2·B_i·B_j)/dudv
        //     = (d²m2/dudv)·B_i·B_j                        → B'diag B  (symmetrised)
        //     + (dm2/du)·(B'_i·xi_v·B_j + B_i·B'_j·xi_v)  → dw_u terms
        //     + (dm2/dv)·(B'_i·xi_u·B_j + B_i·B'_j·xi_u)  → dw_v terms
        //     + m2·(B''_i·xi_u·xi_v·B_j + B'_i·xi_u·B'_j·xi_v
        //          + B'_i·xi_v·B'_j·xi_u + B_i·B''_j·xi_u·xi_v)
        //
        // where dm2/du = m3·q_u, dm2/dv = m3·q_v, d²m2/dudv = m4·q_u·q_v + m3·q_uv.
        //
        // Following the Gaussian LS wiggle pattern, we express this via:
        //   xt_diag_x_dense(B, dw_uv)                    — coeff: d²m2
        //   xt_diag_y_dense(basis_u, dw_v, B) + transpose — dB/du weighted by dm2/dv
        //   xt_diag_y_dense(basis_v, dw_u, B) + transpose — dB/dv weighted by dm2/du
        //   xt_diag_y_dense(basis_uv, w, B) + transpose   — d²B/dudv weighted by m2
        //   xt_diag_y_dense(basis_u, w, basis_v) + transpose — dB/du·dB/dv weighted by m2
        let mut dw = Array1::<f64>::zeros(n);
        let mut dw_u = Array1::<f64>::zeros(n);
        let mut dw_v = Array1::<f64>::zeros(n);
        let mut dw_uv = Array1::<f64>::zeros(n);

        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let m4 = self.neglog_q_fourth_derivative(self.y[row], self.weights[row], q)?;
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let d = geom.d4q_dq04[row];

            // Chain-rule perturbations in direction u.
            let q_u = a * xi_u[row] + phi_u[row];
            let a_u = b * xi_u[row] + b1u[row];
            let b_u = c * xi_u[row] + b2u[row];

            // Chain-rule perturbations in direction v.
            let q_v = a * xi_v[row] + phi_v[row];
            let a_v = b * xi_v[row] + b1v[row];
            let b_v = c * xi_v[row] + b2v[row];

            // Mixed second-order perturbations.
            let q_uv = b * xi_u[row] * xi_v[row] + b1u[row] * xi_v[row] + b1v[row] * xi_u[row];
            let a_uv = c * xi_u[row] * xi_v[row] + b2u[row] * xi_v[row] + b2v[row] * xi_u[row];
            let b_uv = d * xi_u[row] * xi_v[row] + b3u[row] * xi_v[row] + b3v[row] * xi_u[row];

            // ── eta-eta block ──
            // H(eta_i, eta_j) uses q_a = a, q_b = a, q_ab = b (absorbing x_eta
            // into the matrix product).  The perturbations of these geometric
            // quantities are: dq_a/du = a_u, dq_b/du = a_u (since q_a = q_b = a),
            // dq_ab/du = b_u (since q_ab = b), and analogously for v.
            coeff_eta[row] = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, q_u, q_v, q_uv, a, a, b, // q_a, q_b, q_ab
                a_u, a_v, // dq_a_u, dq_a_v
                a_u, a_v, // dq_b_u, dq_b_v  (q_b = a so same perturbation)
                a_uv, a_uv, // d2q_a_uv, d2q_b_uv
                b_u, b_v,  // dq_ab_u, dq_ab_v  (q_ab = b)
                b_uv, // d2q_ab_uv
            );

            // ── eta-w block coefficients ──
            // See the derivation in the docstring above.  We group by which basis
            // matrix tier (B, B', B'', B''') the coefficient multiplies.

            // d²(m2·a)/dudv
            let d2_c_b = m4 * q_u * q_v * a + m3 * (q_uv * a + q_u * a_v + q_v * a_u) + m2 * a_uv;
            // d(m2·a)/du and d(m2·a)/dv
            let dc_b_u = m3 * q_u * a + m2 * a_u;
            let dc_b_v = m3 * q_v * a + m2 * a_v;
            // m2·a (static coefficient for B in the cross block)
            let c_b_static = m2 * a;
            // d²(m1)/dudv
            let d2_c_b1 = m3 * q_u * q_v + m2 * q_uv;
            // d(m1)/du and d(m1)/dv
            let dc_b1_u = m2 * q_u;
            let dc_b1_v = m2 * q_v;

            coeff_etaw_b[row] = d2_c_b;
            coeff_etaw_d1[row] = dc_b_u * xi_v[row] + dc_b_v * xi_u[row] + d2_c_b1;
            coeff_etaw_d2[row] =
                c_b_static * xi_u[row] * xi_v[row] + dc_b1_u * xi_v[row] + dc_b1_v * xi_u[row];
            coeff_etaw_d3[row] = m1 * xi_u[row] * xi_v[row];

            // ── w-w block coefficients ──
            // The w-w static Hessian coefficient is m2 (for B'diag B).
            dw[row] = m2;
            dw_u[row] = m3 * q_u;
            dw_v[row] = m3 * q_v;
            dw_uv[row] = m4 * q_u * q_v + m3 * q_uv;
        }

        // ── Assemble eta-eta block ──
        let d2_h_eta_eta = xt_diag_x_dense(&x_eta, &coeff_eta)?;

        // ── Assemble eta-w block ──
        // The second-order directional derivative of the cross block H_eta_w is:
        //   d²H_eta_w[u,v] = X_eta' diag(coeff_etaw_b)  B
        //                   + X_eta' diag(coeff_etaw_d1) B'
        //                   + X_eta' diag(coeff_etaw_d2) B''
        //                   + X_eta' diag(coeff_etaw_d3) B'''
        let d2_h_eta_w = xt_diag_y_dense(&x_eta, &coeff_etaw_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d1, &geom.basis_d1)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d2, &geom.basis_d2)?
            + &xt_diag_y_dense(&x_eta, &coeff_etaw_d3, &geom.basis_d3)?;

        // ── Assemble w-w block ──
        // Following the Gaussian LS wiggle pattern (lines 6351-6363), the w-w
        // second directional derivative is assembled from scaled basis products:
        //
        //   d²(m2·B_i·B_j)/dudv decomposition:
        //     (d²m2)     · B_i·B_j        → xt_diag_x(B, dw_uv)
        //     (dm2/du)   · dB_j/dv · B_i  → xt_diag_y(basis_v, dw_u, B) + transpose
        //     (dm2/dv)   · dB_j/du · B_i  → xt_diag_y(basis_u, dw_v, B) + transpose
        //     m2 · d²B_j/dudv · B_i       → xt_diag_y(basis_uv, dw, B) + transpose
        //     m2 · dB_i/du · dB_j/dv      → xt_diag_y(basis_u, dw, basis_v) + transpose
        let a_ab = xt_diag_y_dense(&basis_uv, &dw, &geom.basis)?;
        let a_ij = xt_diag_y_dense(&basis_u, &dw, &basis_v)?;
        let a_iwj = xt_diag_y_dense(&basis_u, &dw_v, &geom.basis)?;
        let a_jwi = xt_diag_y_dense(&basis_v, &dw_u, &geom.basis)?;
        let d2_h_ww = &a_ab
            + &a_ab.t()
            + &a_ij
            + a_ij.t()
            + &a_iwj
            + a_iwj.t()
            + &a_jwi
            + a_jwi.t()
            + &xt_diag_x_dense(&geom.basis, &dw_uv)?;

        Ok(Some(binomial_pack_mean_wiggle_joint_symmetrichessian(
            &d2_h_eta_eta,
            &d2_h_eta_w,
            &d2_h_ww,
        )))
    }

    pub(crate) fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        if block_states.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialMeanWiggleFamily joint psi terms expect 2 blocks and 2 derivative block lists, got {} and {}",
                block_states.len(),
                derivative_blocks.len()
            ) }.into());
        }
        let x_eta = self.dense_eta_design_fromspecs(specs)?;
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let n = self.y.len();
        if eta.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let geom = self.wiggle_geometry(eta.view(), betaw.view())?;
        let p_eta = x_eta.ncols();
        let pw = geom.basis.ncols();
        let implicit_dir =
            self.exact_newton_joint_psi_action(block_states, derivative_blocks, psi_index, p_eta)?;
        let dense_dir = if implicit_dir.is_none() {
            self.exact_newton_joint_psi_direction(
                block_states,
                derivative_blocks,
                psi_index,
                &x_eta,
            )?
        } else {
            None
        };
        let z_eta_psi = if let Some((_, ref z_eta_psi)) = implicit_dir {
            z_eta_psi
        } else if let Some(ref dir_a) = dense_dir {
            &dir_a.z_eta_psi
        } else {
            return Ok(None);
        };

        let mut objective_psi = 0.0;
        let mut score_eta_xa = Array1::<f64>::zeros(n);
        let mut score_eta_x = Array1::<f64>::zeros(n);
        let mut score_w_b = Array1::<f64>::zeros(n);
        let mut score_w_d1 = Array1::<f64>::zeros(n);

        let mut coeff_eta_eta_xx = Array1::<f64>::zeros(n);
        let mut coeff_eta_eta_xa_x = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_xa_b = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_x_b = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_x_d1 = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_xa_d1 = Array1::<f64>::zeros(n);
        let mut coeff_eta_w_x_d2 = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);

        for row in 0..n {
            let q = eta[row] + etaw[row];
            let (m1, m2, m3) = self.neglog_q_derivatives(self.y[row], self.weights[row], q)?;
            let z_a = z_eta_psi[row];
            let a = geom.dq_dq0[row];
            let b = geom.d2q_dq02[row];
            let c = geom.d3q_dq03[row];
            let q_a = a * z_a;

            objective_psi += m1 * q_a;

            score_eta_xa[row] = m1 * a;
            score_eta_x[row] = m2 * q_a * a + m1 * b * z_a;
            score_w_b[row] = m2 * q_a;
            score_w_d1[row] = m1 * z_a;

            coeff_eta_eta_xx[row] =
                m3 * q_a * a * a + m2 * (2.0 * a * b * z_a + q_a * b) + m1 * c * z_a;
            coeff_eta_eta_xa_x[row] = m2 * a * a + m1 * b;
            coeff_eta_w_xa_b[row] = m2 * a;
            coeff_eta_w_x_b[row] = m3 * q_a * a + m2 * b * z_a;
            coeff_eta_w_x_d1[row] = m2 * (a * z_a + q_a);
            coeff_eta_w_xa_d1[row] = m1;
            coeff_eta_w_x_d2[row] = m1 * z_a;
            coeff_ww_bb[row] = m3 * q_a;
            coeff_ww_db[row] = m2 * z_a;
        }

        let score_w = crate::faer_ndarray::fast_atv(&geom.basis, &score_w_b)
            + crate::faer_ndarray::fast_atv(&geom.basis_d1, &score_w_d1);

        if let Some((action, _)) = implicit_dir {
            let score_eta = action.transpose_mul(score_eta_xa.view())
                + crate::faer_ndarray::fast_atv(x_eta.as_ref(), &score_eta_x);
            let score_psi = binomial_pack_mean_wiggle_joint_score(&score_eta, &score_w);
            let x_eta_arc = shared_dense_arc(x_eta.as_ref());
            let basis_arc = Arc::new(geom.basis.clone());
            let basis_d1_arc = Arc::new(geom.basis_d1.clone());
            let basis_d2_arc = Arc::new(geom.basis_d2.clone());
            let zeros = Array1::<f64>::zeros(n);
            let operator = CustomFamilyJointPsiOperator::new(
                p_eta + pw,
                vec![
                    CustomFamilyJointDesignChannel::new(
                        0..p_eta,
                        Arc::clone(&x_eta_arc),
                        Some(action),
                    ),
                    CustomFamilyJointDesignChannel::new(
                        p_eta..p_eta + pw,
                        Arc::clone(&basis_arc),
                        None,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        p_eta..p_eta + pw,
                        Arc::clone(&basis_d1_arc),
                        None,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        p_eta..p_eta + pw,
                        Arc::clone(&basis_d2_arc),
                        None,
                    ),
                ],
                vec![
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        0,
                        coeff_eta_eta_xa_x.clone(),
                        coeff_eta_eta_xx.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        1,
                        coeff_eta_w_xa_b.clone(),
                        coeff_eta_w_x_b.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        0,
                        coeff_eta_w_xa_b.clone(),
                        coeff_eta_w_x_b.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        2,
                        coeff_eta_w_xa_d1.clone(),
                        coeff_eta_w_x_d1.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        0,
                        coeff_eta_w_xa_d1.clone(),
                        coeff_eta_w_x_d1.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        3,
                        zeros.clone(),
                        coeff_eta_w_x_d2.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        3,
                        0,
                        zeros.clone(),
                        coeff_eta_w_x_d2.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        1,
                        zeros.clone(),
                        coeff_ww_bb.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        1,
                        zeros.clone(),
                        coeff_ww_db.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(1, 2, zeros, coeff_ww_db.clone()),
                ],
            );
            return Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
                objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(std::sync::Arc::new(operator)),
            }));
        }

        let dir_a =
            dense_dir.expect("dense psi direction should exist when implicit direction is absent");
        let x_eta_psi = dir_a
            .x_eta_psi
            .as_ref()
            .expect("dense eta psi design should exist when implicit direction is absent");
        let score_psi = binomial_pack_mean_wiggle_joint_score(
            &(crate::faer_ndarray::fast_atv(x_eta_psi, &score_eta_xa)
                + crate::faer_ndarray::fast_atv(x_eta.as_ref(), &score_eta_x)),
            &score_w,
        );
        let a_eta_eta = xt_diag_y_dense(x_eta_psi, &coeff_eta_eta_xa_x, &x_eta)?;
        let h_eta_eta = &a_eta_eta + &a_eta_eta.t() + &xt_diag_x_dense(&x_eta, &coeff_eta_eta_xx)?;
        let h_eta_w = xt_diag_y_dense(x_eta_psi, &coeff_eta_w_xa_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_eta_w_x_b, &geom.basis)?
            + &xt_diag_y_dense(&x_eta, &coeff_eta_w_x_d1, &geom.basis_d1)?
            + &xt_diag_y_dense(x_eta_psi, &coeff_eta_w_xa_d1, &geom.basis_d1)?
            + &xt_diag_y_dense(&x_eta, &coeff_eta_w_x_d2, &geom.basis_d2)?;
        let a_ww = xt_diag_y_dense(&geom.basis_d1, &coeff_ww_db, &geom.basis)?;
        let h_ww = xt_diag_x_dense(&geom.basis, &coeff_ww_bb)? + &a_ww + a_ww.t();

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: binomial_pack_mean_wiggle_joint_symmetrichessian(
                &h_eta_eta, &h_eta_w, &h_ww,
            ),
            hessian_psi_operator: None,
        }))
    }
}

pub(crate) struct BinomialMeanWiggleHessianWorkspace {
    family: BinomialMeanWiggleFamily,
    block_states: Vec<ParameterBlockState>,
    x_eta: Arc<Array2<f64>>,
    hessian_operator: Arc<RowCoeffOperator>,
}

impl BinomialMeanWiggleHessianWorkspace {
    pub(crate) fn new(
        family: BinomialMeanWiggleFamily,
        block_states: Vec<ParameterBlockState>,
        x_eta: Array2<f64>,
    ) -> Result<Self, String> {
        let x_eta = Arc::new(x_eta);
        let hessian_operator = family.bmw_static_hessian_operator(&block_states, x_eta.clone())?;
        Ok(Self {
            family,
            block_states,
            x_eta,
            hessian_operator,
        })
    }
}

impl ExactNewtonJointHessianWorkspace for BinomialMeanWiggleHessianWorkspace {
    pub(crate) fn hessian_matvec_available(&self) -> bool {
        true
    }

    pub(crate) fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(
            crate::solver::estimate::reml::unified::HyperOperator::mul_vec(
                self.hessian_operator.as_ref(),
                v,
            ),
        ))
    }

    pub(crate) fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(None)
    }

    pub(crate) fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .directional_derivative_operator(d_beta_flat)?
            .map(|operator| operator.to_dense()))
    }

    pub(crate) fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        self.family
            .bmw_directional_operator(&self.block_states, self.x_eta.clone(), d_beta_flat)
    }

    pub(crate) fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .second_directional_derivative_operator(d_beta_u_flat, d_beta_v_flat)?
            .map(|operator| operator.to_dense()))
    }

    pub(crate) fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        self.family.bmw_second_directional_operator(
            &self.block_states,
            self.x_eta.clone(),
            d_beta_u,
            d_beta_v,
        )
    }
}

impl CustomFamilyGenerative for BinomialMeanWiggleFamily {
    pub(crate) fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialMeanWiggleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta = &block_states[Self::BLOCK_ETA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta.len() != self.y.len() || etaw.len() != self.y.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialMeanWiggleFamily generative size mismatch".to_string(),
            }
            .into());
        }
        let mean = gamlss_rowwise_map_result(self.y.len(), |i| {
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, eta[i] + etaw[i])
                .map_err(|e| format!("fixed-link wiggle inverse-link evaluation failed: {e}"))?;
            Ok(jet.mu)
        })?;
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Bernoulli,
        })
    }
}

/// Built-in Poisson log-link family (single parameter block).
#[derive(Clone)]
pub struct PoissonLogFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
}

impl PoissonLogFamily {
    pub const BLOCK_ETA: usize = 0;

    pub fn parameternames() -> &'static [&'static str] {
        &["eta"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "poisson_log",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }
}

/// Per-row IRLS contribution that a single-parameter log-link family must
/// produce. The shared driver `evaluate_log_link_diagonal_irls` consumes
/// these and assembles the full `FamilyEvaluation` so the three pieces of
/// code that previously lived inside each family — size validation, per-row
/// y validation + η clamping + saturated `exp`, the active-clamp w/z guard,
/// and the final return — exist in exactly one place.
pub(crate) struct DiagonalIrlsRow {
    /// Weighted contribution to ℓ at this row.
    log_lik_increment: f64,
    /// Unfloored observed Hessian weight (the driver applies `MIN_WEIGHT`).
    observed_weight: f64,
    /// Per-row Newton step on the working response: `z = e + working_step`.
    /// Each family computes this with its own (score, denominator); the
    /// driver only handles the active-clamp / zero-weight guard.
    working_step: f64,
}

/// Trait implemented by single-block log-link families that share the
/// diagonal IRLS structure (Poisson, Gamma). Each impl is responsible only
/// for the family-specific math: validating `y[i]` and producing the
/// per-row triple `(ℓ_increment, observed_weight, working_step)`.
trait LogLinkDiagonalIrlsFamily {
    /// Short, human-readable name used in size-mismatch errors.
    pub(crate) fn family_label(&self) -> &'static str;

    /// Read access to the shared (y, prior weights) buffers.
    pub(crate) fn y(&self) -> &Array1<f64>;
    pub(crate) fn prior_weights(&self) -> &Array1<f64>;

    /// Optional pre-loop validation hook for parameters outside the
    /// (y, weights, eta) triple (e.g. Gamma shape > 0).
    pub(crate) fn validate_self(&self) -> Result<(), String> {
        Ok(())
    }

    /// Validate `y[i]` and return an error message if rejected. Default
    /// implementation enforces only finiteness; concrete families override
    /// to add domain constraints.
    pub(crate) fn validate_yi(&self, yi: f64, idx: usize) -> Result<(), String>;

    /// Family-specific per-row math; `m = saturated_exp_eta(eta_clamped)`
    /// is computed by the driver and handed in.
    pub(crate) fn row_kernel(
        &self,
        yi: f64,
        e_clamped: f64,
        m: f64,
        prior_w: f64,
    ) -> DiagonalIrlsRow;
}

/// Shared IRLS driver for [`LogLinkDiagonalIrlsFamily`]. Centralises the
/// size-check, η-clamp, saturated-exp, active-clamp guard, ll accumulation,
/// and `FamilyEvaluation` assembly so all log-link families with the diagonal
/// structure (Poisson, Gamma) cannot drift apart numerically.
pub(crate) fn evaluate_log_link_diagonal_irls<F: LogLinkDiagonalIrlsFamily + ?Sized>(
    family: &F,
    block_states: &[ParameterBlockState],
) -> Result<FamilyEvaluation, String> {
    let label = family.family_label();
    let eta = &expect_single_block(block_states, label)?.eta;
    let y = family.y();
    let prior_weights = family.prior_weights();
    let n = y.len();
    if eta.len() != n || prior_weights.len() != n {
        return Err(GamlssError::DimensionMismatch {
            reason: format!("{label} input size mismatch"),
        }
        .into());
    }
    family.validate_self()?;

    let mut ll = 0.0;
    let mut z = Array1::<f64>::zeros(n);
    let mut w = Array1::<f64>::zeros(n);

    for i in 0..n {
        let yi = y[i];
        family.validate_yi(yi, i)?;
        let e_raw = eta[i];
        let e = e_raw.clamp(-ETA_HARD_CLAMP, ETA_HARD_CLAMP);
        let active_clamp = e != e_raw;
        let m = saturated_exp_eta(e_raw);
        let prior_w = prior_weights[i];
        let row = family.row_kernel(yi, e, m, prior_w);
        ll += row.log_lik_increment;
        if prior_w == 0.0 || active_clamp {
            w[i] = 0.0;
            z[i] = e_raw;
        } else {
            w[i] = floor_positiveweight(row.observed_weight, MIN_WEIGHT);
            z[i] = e + row.working_step;
        }
    }

    Ok(FamilyEvaluation {
        log_likelihood: ll,
        blockworking_sets: vec![BlockWorkingSet::diagonal_checked(z, w)?],
    })
}

impl LogLinkDiagonalIrlsFamily for PoissonLogFamily {
    pub(crate) fn family_label(&self) -> &'static str {
        "PoissonLogFamily"
    }
    pub(crate) fn y(&self) -> &Array1<f64> {
        &self.y
    }
    pub(crate) fn prior_weights(&self) -> &Array1<f64> {
        &self.weights
    }
    pub(crate) fn validate_yi(&self, yi: f64, idx: usize) -> Result<(), String> {
        if !yi.is_finite() || yi < 0.0 {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "PoissonLogFamily requires non-negative finite y; found y[{idx}]={yi}"
                ),
            }
            .into());
        }
        Ok::<(), _>(())
    }
    #[inline]
    pub(crate) fn row_kernel(
        &self,
        yi: f64,
        e_clamped: f64,
        m: f64,
        prior_w: f64,
    ) -> DiagonalIrlsRow {
        // Drop log(y!) constant in objective.
        let log_lik_increment = prior_w * (yi * e_clamped - m);
        let dmu = m.max(MIN_DERIV);
        let var = m.max(MIN_PROB);
        DiagonalIrlsRow {
            log_lik_increment,
            observed_weight: prior_w * (dmu * dmu / var),
            // (yi - m)/dmu, identical to the previous direct expression.
            working_step: (yi - m) / signedwith_floor(dmu, MIN_DERIV),
        }
    }
}

impl CustomFamily for PoissonLogFamily {
    pub(crate) fn evaluate(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        evaluate_log_link_diagonal_irls(self, block_states)
    }
}

impl CustomFamilyGenerative for PoissonLogFamily {
    pub(crate) fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        let eta = &expect_single_block(block_states, "PoissonLogFamily")?.eta;
        let mean = gamlss_rowwise_map(eta.len(), |i| saturated_exp_eta(eta[i]));
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Poisson,
        })
    }
}

/// Built-in Gamma log-link family (single parameter block, fixed shape).
#[derive(Clone)]
pub struct GammaLogFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub shape: f64,
}

impl GammaLogFamily {
    pub const BLOCK_ETA: usize = 0;

    pub fn parameternames() -> &'static [&'static str] {
        &["eta"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "gamma_log",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }
}

impl LogLinkDiagonalIrlsFamily for GammaLogFamily {
    pub(crate) fn family_label(&self) -> &'static str {
        "GammaLogFamily"
    }
    pub(crate) fn y(&self) -> &Array1<f64> {
        &self.y
    }
    pub(crate) fn prior_weights(&self) -> &Array1<f64> {
        &self.weights
    }
    pub(crate) fn validate_self(&self) -> Result<(), String> {
        if !self.shape.is_finite() || self.shape <= 0.0 {
            return Err(GamlssError::NonFinite {
                reason: "GammaLogFamily shape must be finite and > 0".to_string(),
            }
            .into());
        }
        Ok(())
    }
    pub(crate) fn validate_yi(&self, yi: f64, idx: usize) -> Result<(), String> {
        if !yi.is_finite() || yi <= 0.0 {
            return Err(GamlssError::InvalidInput {
                reason: format!("GammaLogFamily requires positive finite y; found y[{idx}]={yi}"),
            }
            .into());
        }
        Ok::<(), _>(())
    }
    #[inline]
    pub(crate) fn row_kernel(
        &self,
        yi: f64,
        e_clamped: f64,
        m: f64,
        prior_w: f64,
    ) -> DiagonalIrlsRow {
        assert!(e_clamped.is_finite());
        assert!((e_clamped.exp() - m).abs() <= 1.0e-8 * m.abs().max(1.0));
        // Gamma(shape=k, scale=mu/k), dropping eta-independent constants.
        let log_lik_increment = prior_w * (-self.shape * (yi / m + m.ln()));
        // Gamma with log mean is non-canonical. Use the exact observed
        // η-space curvature -d²ℓ/dη² = prior_w * shape * y / μ, not the
        // Fisher weight prior_w * shape, so diagonal REML/LAML Hessians
        // use the true Laplace curvature instead of a PQL/Fisher surrogate.
        let observed_weight = prior_w * self.shape * yi / m;
        let score = prior_w * self.shape * (yi / m - 1.0);
        // Mirror the pre-extraction formula z = e + score / w_floored exactly;
        // the driver applies MIN_WEIGHT *before* writing w[i], but the old
        // code divided by the already-floored w[i] for non-degenerate rows,
        // and the floor only activates on the degenerate `observed_weight <=
        // MIN_WEIGHT` tail. Reproduce that branch here to preserve bitwise
        // step shape on every row that used to hit the floor.
        let w_floored = observed_weight.max(MIN_WEIGHT);
        DiagonalIrlsRow {
            log_lik_increment,
            observed_weight,
            working_step: score / w_floored,
        }
    }
}

impl CustomFamily for GammaLogFamily {
    pub(crate) fn evaluate(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        evaluate_log_link_diagonal_irls(self, block_states)
    }

    pub(crate) fn diagonalworking_weights_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_eta: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        if block_idx != Self::BLOCK_ETA {
            return Ok(None);
        }
        let eta = &expect_single_block(block_states, "GammaLogFamily")?.eta;
        let n = self.y.len();
        if eta.len() != n || self.weights.len() != n || d_eta.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "GammaLogFamily input size mismatch".to_string(),
            }
            .into());
        }
        if !self.shape.is_finite() || self.shape <= 0.0 {
            return Err(GamlssError::NonFinite {
                reason: "GammaLogFamily shape must be finite and > 0".to_string(),
            }
            .into());
        }

        let mut dw = Array1::<f64>::zeros(n);
        for i in 0..n {
            let yi = self.y[i];
            if !yi.is_finite() || yi <= 0.0 {
                return Err(GamlssError::InvalidInput {
                    reason: format!("GammaLogFamily requires positive finite y; found y[{i}]={yi}"),
                }
                .into());
            }
            let e_raw = eta[i];
            let e = e_raw.clamp(-ETA_HARD_CLAMP, ETA_HARD_CLAMP);
            if self.weights[i] == 0.0 || e != e_raw {
                dw[i] = 0.0;
                continue;
            }
            let m = safe_exp(e).max(MIN_WEIGHT);
            let observed_weight = self.weights[i] * self.shape * yi / m;
            // d/dη [prior_weight * shape * y / exp(η)] = -W_obs.
            // If the positive floor is active, match the evaluated local piece.
            if observed_weight <= MIN_WEIGHT {
                dw[i] = 0.0;
            } else {
                dw[i] = -observed_weight * d_eta[i];
            }
        }
        Ok(Some(dw))
    }
}

impl CustomFamilyGenerative for GammaLogFamily {
    pub(crate) fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        let eta = &expect_single_block(block_states, "GammaLogFamily")?.eta;
        let mean = gamlss_rowwise_map(eta.len(), |i| saturated_exp_eta(eta[i]));
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Gamma { shape: self.shape },
        })
    }
}

/// Built-in binomial location-scale family with a configurable inverse link.
///
/// Parameters:
/// - Block 0: threshold/location T(covariates)
/// - Block 1: log-scale log σ(covariates)
#[derive(Clone)]
pub struct BinomialLocationScaleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub threshold_design: Option<DesignMatrix>,
    pub log_sigma_design: Option<DesignMatrix>,
    /// Resource policy threaded into PsiDesignMap construction (and any other
    /// per-call materialization decision) made during exact-Newton joint psi
    /// derivative evaluation. Defaults to `ResourcePolicy::default_library()`
    /// when the family is built without an explicit policy.
    pub policy: crate::resource::ResourcePolicy,
}

/// Both Binomial location-scale families plug into the unified
/// [`LocationScaleJointPsiFamily`] trait with byte-identical thin delegations
/// to inherent methods, differing only in the implementing type and its
/// `LABEL` fragment; generate them from one template. The Binomial families do
/// not thread the outer-row subsample (they run the full-data exact ψ path), so
/// the trait's `subsample` argument is accepted and ignored here.
macro_rules! impl_binomial_location_scale_joint_psi_family {
    ($family:ty, $label:literal) => {
        impl LocationScaleJointPsiFamily for $family {
            type Direction = LocationScaleJointPsiDirection;
            pub(crate) const LABEL: &'static str = $label;

            pub(crate) fn ws_policy(&self) -> &crate::resource::ResourcePolicy {
                &self.policy
            }

            pub(crate) fn ws_exact_joint_dense_block_designs<'a>(
                &'a self,
                specs: Option<&'a [ParameterBlockSpec]>,
            ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
                self.exact_joint_dense_block_designs(specs)
            }

            pub(crate) fn ws_psi_direction(
                &self,
                block_states: &[ParameterBlockState],
                derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
                psi_index: usize,
                design_loc: &Array2<f64>,
                design_scale: &Array2<f64>,
                policy: &crate::resource::ResourcePolicy,
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

            pub(crate) fn ws_psi_second_order_terms_from_parts(
                &self,
                block_states: &[ParameterBlockState],
                derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
                psi_a: &LocationScaleJointPsiDirection,
                psi_b: &LocationScaleJointPsiDirection,
                design_loc: &Array2<f64>,
                design_scale: &Array2<f64>,
                subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
            ) -> Result<ExactNewtonJointPsiSecondOrderTerms, String> {
                assert!(subsample.is_none());
                self.exact_newton_joint_psisecond_order_terms_from_parts(
                    block_states,
                    derivative_blocks,
                    psi_a,
                    psi_b,
                    design_loc,
                    design_scale,
                )
            }

            pub(crate) fn ws_psi_hessian_directional_from_parts(
                &self,
                block_states: &[ParameterBlockState],
                psi_dir: &LocationScaleJointPsiDirection,
                d_beta_flat: &Array1<f64>,
                design_loc: &Array2<f64>,
                design_scale: &Array2<f64>,
                subsample: Option<&[crate::families::marginal_slope_shared::WeightedOuterRow]>,
            ) -> Result<Array2<f64>, String> {
                assert!(subsample.is_none());
                self.exact_newton_joint_psihessian_directional_derivative_from_parts(
                    block_states,
                    psi_dir,
                    d_beta_flat,
                    design_loc,
                    design_scale,
                )
            }
        }
    };
}

impl_binomial_location_scale_joint_psi_family!(
    BinomialLocationScaleFamily,
    "BinomialLocationScaleFamily"
);

impl_binomial_location_scale_joint_psi_family!(
    BinomialLocationScaleWiggleFamily,
    "BinomialLocationScaleWiggleFamily"
);

pub(crate) type BinomialLocationScaleExactNewtonJointPsiWorkspace =
    LocationScaleJointPsiWorkspace<BinomialLocationScaleFamily>;

pub(crate) type BinomialLocationScaleWiggleExactNewtonJointPsiWorkspace =
    LocationScaleJointPsiWorkspace<BinomialLocationScaleWiggleFamily>;
