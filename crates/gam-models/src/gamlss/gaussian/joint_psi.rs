// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;
use gam_row_macros::row_atom;

// Stable local coordinates for the Gaussian location-scale row NLL. At the
// expansion point `delta_mu = delta_eta = 0`, the log-b link gives
// `sigma(eta + delta_eta) / sigma(eta) = (1-kappa) + kappa*exp(delta_eta)`.
// The perturbed standardized residual is therefore
// `(standardized_residual - delta_mu*inv_sigma) / scale_ratio`. Every runtime
// constant is already certified by `gaussian_diagonal_row_kernel`, so this one
// expression retains the production extreme-value semantics while build-time
// differentiation emits exact observed H, contracted t3, and contracted t4.
row_atom! {
    fn gaussian_normalized_row [generic, order2, third, fourth](
        delta_mu,
        delta_eta;
        obs_weight,
        standardized_residual,
        inv_sigma,
        kappa
    ) {
        obs_weight * ln((1.0 - kappa) + kappa * exp(delta_eta))
            + 0.5
                * obs_weight
                * (standardized_residual - delta_mu * inv_sigma)
                * (standardized_residual - delta_mu * inv_sigma)
                / ((1.0 - kappa) + kappa * exp(delta_eta))
                / ((1.0 - kappa) + kappa * exp(delta_eta))
    }
}

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
    #[cfg(test)]
    pub(crate) w: Array1<f64>,
    #[cfg(test)]
    pub(crate) m: Array1<f64>,
    #[cfg(test)]
    pub(crate) n: Array1<f64>,
    /// Stable `(y - mu) / sigma` at the expansion point.
    pub(crate) standardized_residual: Array1<f64>,
    /// Stable `1 / sigma` at the expansion point.
    pub(crate) inv_sigma: Array1<f64>,
    /// κ = (dσ/dη_ls)/σ for the active sigma link.
    /// The cross Hessian block H_{μ,ls} carries an overall κ factor and the
    /// scale-scale block H_{ls,ls} carries κ².
    pub(crate) kappa: Array1<f64>,
    /// κ' = dκ/dη_ls = κ(1−κ) for the logb link. The static H_{ls,ls} block
    /// carries a κ'·(a−n) term, so κ' threads through every dH directional
    /// weight via the chain rule.
    #[cfg(test)]
    pub(crate) kappa_prime: Array1<f64>,
    /// κ'' = κ(1−κ)(1−2κ); appears in d²H_{ls,ls} via the second
    /// η-derivative of κ'·(a−n).
    #[cfg(test)]
    pub(crate) kappa_dprime: Array1<f64>,
}

/// Production [`gam_math::jet_tower::RowProgram`] for the normalized Gaussian
/// location-scale row NLL.
///
/// The program borrows the exact certified row constants consumed by the live
/// observed-Hessian and directional-weight paths. Its generic evaluator and
/// those specialized order-2/third/fourth paths are emitted from the same
/// [`gaussian_normalized_row`] declaration, so the parity oracle cannot retain
/// an independent copy of the likelihood expression.
pub struct GaussianJointRowProgram<'a> {
    rows: &'a GaussianJointRowScalars,
}

impl<'a> GaussianJointRowProgram<'a> {
    /// Bind the generic row program to one certified production scalar batch.
    pub fn new(rows: &'a GaussianJointRowScalars) -> Self {
        Self { rows }
    }

    fn require_row(&self, row: usize) -> Result<(), String> {
        if row >= self.rows.obs_weight.len() {
            return Err(format!(
                "GaussianJointRowProgram row {row} out of range for {} rows",
                self.rows.obs_weight.len()
            ));
        }
        Ok(())
    }

    /// Symbolically lowered value/gradient/Hessian for one certified row.
    ///
    /// The concrete sparsity bits are part of the generated function's type:
    /// both score channels and all three packed Hessian channels are live.
    #[inline(always)]
    pub(crate) fn row_order2(
        &self,
        row: usize,
    ) -> gam_math::jet_scalar::StaticOrder2Atom<2, 3, 3, 7> {
        gaussian_normalized_row_order2(
            0.0,
            0.0,
            self.rows.obs_weight[row],
            self.rows.standardized_residual[row],
            self.rows.inv_sigma[row],
            self.rows.kappa[row],
        )
    }

    /// Symbolically lowered Hessian derivative in one predictor direction.
    #[inline(always)]
    pub(crate) fn row_third_contracted(&self, row: usize, direction: &[f64; 2]) -> [[f64; 2]; 2] {
        gaussian_normalized_row_third_contracted(
            0.0,
            0.0,
            self.rows.obs_weight[row],
            self.rows.standardized_residual[row],
            self.rows.inv_sigma[row],
            self.rows.kappa[row],
            direction,
        )
    }

    /// Symbolically lowered mixed derivative of the row Hessian.
    #[inline(always)]
    pub(crate) fn row_fourth_contracted(
        &self,
        row: usize,
        direction_u: &[f64; 2],
        direction_v: &[f64; 2],
    ) -> [[f64; 2]; 2] {
        gaussian_normalized_row_fourth_contracted(
            0.0,
            0.0,
            self.rows.obs_weight[row],
            self.rows.standardized_residual[row],
            self.rows.inv_sigma[row],
            self.rows.kappa[row],
            direction_u,
            direction_v,
        )
    }
}

#[inline(always)]
fn matrix_vector_2(matrix: &[[f64; 2]; 2], vector: &[f64; 2]) -> [f64; 2] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]
}

#[inline(always)]
fn dot_2(left: &[f64; 2], right: &[f64; 2]) -> f64 {
    left[0] * right[0] + left[1] * right[1]
}

#[inline(always)]
fn add_vector_2(left: [f64; 2], right: [f64; 2]) -> [f64; 2] {
    [left[0] + right[0], left[1] + right[1]]
}

#[inline(always)]
fn add_matrix_2(left: [[f64; 2]; 2], right: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [left[0][0] + right[0][0], left[0][1] + right[0][1]],
        [left[1][0] + right[1][0], left[1][1] + right[1][1]],
    ]
}

/// One order of generated Gaussian row geometry, stored in neutral predictor
/// coordinates. At order zero these are `(g, H)`; in a first/second tower they
/// are the corresponding directional derivatives of `(g, H)`.
pub(crate) struct GaussianRowChannels {
    pub(crate) gradient_mu: Array1<f64>,
    pub(crate) gradient_ls: Array1<f64>,
    pub(crate) hessian_mm: Array1<f64>,
    pub(crate) hessian_ml: Array1<f64>,
    pub(crate) hessian_ll: Array1<f64>,
}

impl GaussianRowChannels {
    fn zeros(n: usize) -> Self {
        Self {
            gradient_mu: Array1::zeros(n),
            gradient_ls: Array1::zeros(n),
            hessian_mm: Array1::zeros(n),
            hessian_ml: Array1::zeros(n),
            hessian_ll: Array1::zeros(n),
        }
    }
}

pub(crate) struct GaussianRowFirstTower {
    pub(crate) base: GaussianRowChannels,
    pub(crate) first: GaussianRowChannels,
}

pub(crate) struct GaussianRowSecondTower {
    pub(crate) base: GaussianRowChannels,
    pub(crate) first_a: GaussianRowChannels,
    pub(crate) first_b: GaussianRowChannels,
    pub(crate) second: GaussianRowChannels,
}

fn write_base_channels(
    channels: &mut GaussianRowChannels,
    row: usize,
    atom: &gam_math::jet_scalar::StaticOrder2Atom<2, 3, 3, 7>,
) -> [[f64; 2]; 2] {
    let gradient = atom.gradient();
    let hessian = [
        [atom.hessian_at(0, 0), atom.hessian_at(0, 1)],
        [atom.hessian_at(1, 0), atom.hessian_at(1, 1)],
    ];
    channels.gradient_mu[row] = gradient[0];
    channels.gradient_ls[row] = gradient[1];
    channels.hessian_mm[row] = hessian[0][0];
    channels.hessian_ml[row] = hessian[0][1];
    channels.hessian_ll[row] = hessian[1][1];
    hessian
}

fn write_directional_channels(
    channels: &mut GaussianRowChannels,
    row: usize,
    gradient: [f64; 2],
    hessian: [[f64; 2]; 2],
) {
    channels.gradient_mu[row] = gradient[0];
    channels.gradient_ls[row] = gradient[1];
    channels.hessian_mm[row] = hessian[0][0];
    channels.hessian_ml[row] = hessian[0][1];
    channels.hessian_ll[row] = hessian[1][1];
}

/// Generated Gaussian row gradient and Hessian with no directional scratch.
pub(crate) fn gaussian_row_channels(rows: &GaussianJointRowScalars) -> GaussianRowChannels {
    let n = rows.obs_weight.len();
    let program = GaussianJointRowProgram::new(rows);
    let mut base = GaussianRowChannels::zeros(n);
    for row in 0..n {
        let atom = program.row_order2(row);
        write_base_channels(&mut base, row, &atom);
    }
    base
}

/// Generated `(g, H)` plus its first derivative along a rowwise predictor
/// direction. The row atom is evaluated once per row.
pub(crate) fn gaussian_row_first_tower(
    rows: &GaussianJointRowScalars,
    direction_mu: &Array1<f64>,
    direction_ls: &Array1<f64>,
) -> GaussianRowFirstTower {
    let n = rows.obs_weight.len();
    let program = GaussianJointRowProgram::new(rows);
    let mut base = GaussianRowChannels::zeros(n);
    let mut first = GaussianRowChannels::zeros(n);
    for row in 0..n {
        let direction = [direction_mu[row], direction_ls[row]];
        let atom = program.row_order2(row);
        let hessian = write_base_channels(&mut base, row, &atom);
        write_directional_channels(
            &mut first,
            row,
            matrix_vector_2(&hessian, &direction),
            program.row_third_contracted(row, &direction),
        );
    }
    GaussianRowFirstTower { base, first }
}

/// Generated `(g, H)` tower through the mixed second derivative along two
/// rowwise directions, including a possibly nonzero mixed predictor leg.
pub(crate) fn gaussian_row_second_tower(
    rows: &GaussianJointRowScalars,
    direction_a_mu: &Array1<f64>,
    direction_a_ls: &Array1<f64>,
    direction_b_mu: &Array1<f64>,
    direction_b_ls: &Array1<f64>,
    direction_ab_mu: &Array1<f64>,
    direction_ab_ls: &Array1<f64>,
) -> GaussianRowSecondTower {
    let n = rows.obs_weight.len();
    let program = GaussianJointRowProgram::new(rows);
    let mut base = GaussianRowChannels::zeros(n);
    let mut first_a = GaussianRowChannels::zeros(n);
    let mut first_b = GaussianRowChannels::zeros(n);
    let mut second = GaussianRowChannels::zeros(n);
    for row in 0..n {
        let direction_a = [direction_a_mu[row], direction_a_ls[row]];
        let direction_b = [direction_b_mu[row], direction_b_ls[row]];
        let direction_ab = [direction_ab_mu[row], direction_ab_ls[row]];
        let atom = program.row_order2(row);
        let hessian = write_base_channels(&mut base, row, &atom);
        let hessian_a = program.row_third_contracted(row, &direction_a);
        let hessian_b = program.row_third_contracted(row, &direction_b);
        write_directional_channels(
            &mut first_a,
            row,
            matrix_vector_2(&hessian, &direction_a),
            hessian_a,
        );
        write_directional_channels(
            &mut first_b,
            row,
            matrix_vector_2(&hessian, &direction_b),
            hessian_b,
        );
        write_directional_channels(
            &mut second,
            row,
            add_vector_2(
                matrix_vector_2(&hessian_a, &direction_b),
                matrix_vector_2(&hessian, &direction_ab),
            ),
            add_matrix_2(
                program.row_fourth_contracted(row, &direction_a, &direction_b),
                program.row_third_contracted(row, &direction_ab),
            ),
        );
    }
    GaussianRowSecondTower {
        base,
        first_a,
        first_b,
        second,
    }
}

impl gam_math::jet_tower::RowProgram<2> for GaussianJointRowProgram<'_> {
    fn n_rows(&self) -> usize {
        self.rows.obs_weight.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
        self.require_row(row)?;
        Ok([0.0, 0.0])
    }

    fn eval<S: gam_math::jet_scalar::JetScalar<2>>(
        &self,
        row: usize,
        p: &[S; 2],
    ) -> Result<S, String> {
        self.require_row(row)?;
        Ok(gaussian_normalized_row(
            &p[0],
            &p[1],
            self.rows.obs_weight[row],
            self.rows.standardized_residual[row],
            self.rows.inv_sigma[row],
            self.rows.kappa[row],
        ))
    }
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
    #[cfg(test)]
    let mut w = Array1::<f64>::uninit(nobs);
    #[cfg(test)]
    let mut m = Array1::<f64>::uninit(nobs);
    #[cfg(test)]
    let mut n = Array1::<f64>::uninit(nobs);
    let mut standardized_residual = Array1::<f64>::uninit(nobs);
    let mut inv_sigma = Array1::<f64>::uninit(nobs);
    let mut kappa = Array1::<f64>::uninit(nobs);
    #[cfg(test)]
    let mut kappa_prime = Array1::<f64>::uninit(nobs);
    #[cfg(test)]
    let mut kappa_dprime = Array1::<f64>::uninit(nobs);
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    // Compute into an indexed temporary first. Parallel collection preserves
    // row order; scanning the results afterward reports the smallest failing
    // row deterministically and publishes no partially initialized scalar set.
    let certified: Vec<Result<GaussianDiagonalRowKernel, String>> = (0..nobs)
        .into_par_iter()
        .map(|i| gaussian_diagonal_row_kernel(i, y[i], etamu[i], eta_ls[i], weights[i], ln2pi))
        .collect();
    for (i, row) in certified.into_iter().enumerate() {
        let row = row?;
        obs_weight[i].write(weights[i]);
        #[cfg(test)]
        w[i].write(row.joint_w);
        #[cfg(test)]
        m[i].write(row.joint_m);
        #[cfg(test)]
        n[i].write(row.joint_n);
        standardized_residual[i].write(row.standardized_residual);
        inv_sigma[i].write(row.inv_sigma);
        kappa[i].write(row.kappa);
        #[cfg(test)]
        kappa_prime[i].write(row.kappa_prime);
        #[cfg(test)]
        kappa_dprime[i].write(row.kappa_dprime);
    }
    // SAFETY: every `MaybeUninit` slot in each of these arrays was written
    // exactly once in the `for i in 0..nobs` loop above; no slot is read,
    // moved, or dropped before this point.
    let (obs_weight, standardized_residual, inv_sigma, kappa) = unsafe {
        (
            obs_weight.assume_init(),
            standardized_residual.assume_init(),
            inv_sigma.assume_init(),
            kappa.assume_init(),
        )
    };
    #[cfg(test)]
    let (w, m, n, kappa_prime, kappa_dprime) = unsafe {
        (
            w.assume_init(),
            m.assume_init(),
            n.assume_init(),
            kappa_prime.assume_init(),
            kappa_dprime.assume_init(),
        )
    };
    Ok(GaussianJointRowScalars {
        obs_weight,
        #[cfg(test)]
        w,
        #[cfg(test)]
        m,
        #[cfg(test)]
        n,
        standardized_residual,
        inv_sigma,
        kappa,
        #[cfg(test)]
        kappa_prime,
        #[cfg(test)]
        kappa_dprime,
    })
}

/// Live third-order observed-Hessian contraction emitted from the same stable
/// [`gaussian_normalized_row`] expression as the observed Hessian.
pub(crate) fn gaussian_joint_first_directionalweights(
    scalars: &GaussianJointRowScalars,
    dotmu: &Array1<f64>,
    dot_eta: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let nobs = scalars.obs_weight.len();
    let program = GaussianJointRowProgram::new(scalars);
    let mut w_u = Array1::<f64>::zeros(nobs);
    let mut c_u = Array1::<f64>::zeros(nobs);
    let mut d_u = Array1::<f64>::zeros(nobs);
    for i in 0..nobs {
        let matrix = program.row_third_contracted(i, &[dotmu[i], dot_eta[i]]);
        w_u[i] = matrix[0][0];
        c_u[i] = matrix[0][1];
        d_u[i] = matrix[1][1];
    }
    (w_u, c_u, d_u)
}

/// Live fourth-order observed-Hessian contraction emitted from the same stable
/// [`gaussian_normalized_row`] expression as every lower curvature channel.
pub(crate) fn gaussian_jointsecond_directionalweights(
    scalars: &GaussianJointRowScalars,
    dotmu_u: &Array1<f64>,
    dot_eta_u: &Array1<f64>,
    dotmuv: &Array1<f64>,
    dot_etav: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let nobs = scalars.obs_weight.len();
    let program = GaussianJointRowProgram::new(scalars);
    let mut w_uv = Array1::<f64>::zeros(nobs);
    let mut c_uv = Array1::<f64>::zeros(nobs);
    let mut d_uv = Array1::<f64>::zeros(nobs);
    for i in 0..nobs {
        let matrix = program.row_fourth_contracted(
            i,
            &[dotmu_u[i], dot_eta_u[i]],
            &[dotmuv[i], dot_etav[i]],
        );
        w_uv[i] = matrix[0][0];
        c_uv[i] = matrix[0][1];
        d_uv[i] = matrix[1][1];
    }
    (w_uv, c_uv, d_uv)
}

pub(crate) fn gaussian_joint_psi_firstweights(
    scalars: &GaussianJointRowScalars,
    mu_a: &Array1<f64>,
    eta_a: &Array1<f64>,
) -> GaussianJointPsiFirstWeights {
    let nobs = scalars.obs_weight.len();
    let program = GaussianJointRowProgram::new(scalars);
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
        let direction = [mu_a[i], eta_a[i]];
        let atom = program.row_order2(i);
        let score = atom.gradient();
        let hessian = [
            [atom.hessian_at(0, 0), atom.hessian_at(0, 1)],
            [atom.hessian_at(1, 0), atom.hessian_at(1, 1)],
        ];
        let score_direction = matrix_vector_2(&hessian, &direction);
        let hessian_direction = program.row_third_contracted(i, &direction);
        objective_psirow[i].write(dot_2(&score, &direction));
        scoremu[i].write(score[0]);
        score_ls[i].write(score[1]);
        dscoremu[i].write(score_direction[0]);
        dscore_ls[i].write(score_direction[1]);
        hmumu[i].write(hessian[0][0]);
        hmu_ls[i].write(hessian[0][1]);
        h_ls_ls[i].write(hessian[1][1]);
        dhmumu[i].write(hessian_direction[0][0]);
        dhmu_ls[i].write(hessian_direction[0][1]);
        dh_ls_ls[i].write(hessian_direction[1][1]);
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
    let nobs = scalars.obs_weight.len();
    let program = GaussianJointRowProgram::new(scalars);
    let mut objective_psi_psirow = Array1::<f64>::uninit(nobs);
    let mut d2scoremu = Array1::<f64>::uninit(nobs);
    let mut d2score_ls = Array1::<f64>::uninit(nobs);
    let mut d2hmumu = Array1::<f64>::uninit(nobs);
    let mut d2hmu_ls = Array1::<f64>::uninit(nobs);
    let mut d2h_ls_ls = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let direction_a = [mu_a[i], eta_a[i]];
        let direction_b = [mu_b[i], eta_b[i]];
        let direction_ab = [mu_ab[i], eta_ab[i]];
        let atom = program.row_order2(i);
        let score = atom.gradient();
        let hessian = [
            [atom.hessian_at(0, 0), atom.hessian_at(0, 1)],
            [atom.hessian_at(1, 0), atom.hessian_at(1, 1)],
        ];
        let hessian_a = program.row_third_contracted(i, &direction_a);
        let hessian_ab = program.row_third_contracted(i, &direction_ab);
        let hessian_a_b = program.row_fourth_contracted(i, &direction_a, &direction_b);
        let score_second = add_vector_2(
            matrix_vector_2(&hessian_a, &direction_b),
            matrix_vector_2(&hessian, &direction_ab),
        );
        let hessian_second = add_matrix_2(hessian_a_b, hessian_ab);
        objective_psi_psirow[i].write(
            dot_2(&direction_a, &matrix_vector_2(&hessian, &direction_b))
                + dot_2(&score, &direction_ab),
        );
        d2scoremu[i].write(score_second[0]);
        d2score_ls[i].write(score_second[1]);
        d2hmumu[i].write(hessian_second[0][0]);
        d2hmu_ls[i].write(hessian_second[0][1]);
        d2h_ls_ls[i].write(hessian_second[1][1]);
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
    dot_mu: &Array1<f64>,
    dot_eta: &Array1<f64>,
    mu_a: &Array1<f64>,
    eta_a: &Array1<f64>,
    dot_mu_a: &Array1<f64>,
    dot_eta_a: &Array1<f64>,
) -> GaussianJointPsiMixedDriftWeights {
    let nobs = scalars.obs_weight.len();
    let program = GaussianJointRowProgram::new(scalars);
    let mut dhmumu_u = Array1::<f64>::uninit(nobs);
    let mut dhmu_ls_u = Array1::<f64>::uninit(nobs);
    let mut dh_ls_ls_u = Array1::<f64>::uninit(nobs);
    let mut d2hmumu = Array1::<f64>::uninit(nobs);
    let mut d2hmu_ls = Array1::<f64>::uninit(nobs);
    let mut d2h_ls_ls = Array1::<f64>::uninit(nobs);
    for i in 0..nobs {
        let drift = [dot_mu[i], dot_eta[i]];
        let psi = [mu_a[i], eta_a[i]];
        let mixed_direction = [dot_mu_a[i], dot_eta_a[i]];
        let hessian_drift = program.row_third_contracted(i, &drift);
        let hessian_mixed = add_matrix_2(
            program.row_fourth_contracted(i, &drift, &psi),
            program.row_third_contracted(i, &mixed_direction),
        );
        dhmumu_u[i].write(hessian_drift[0][0]);
        dhmu_ls_u[i].write(hessian_drift[0][1]);
        dh_ls_ls_u[i].write(hessian_drift[1][1]);
        d2hmumu[i].write(hessian_mixed[0][0]);
        d2hmu_ls[i].write(hessian_mixed[0][1]);
        d2h_ls_ls[i].write(hessian_mixed[1][1]);
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
    let n = rows.obs_weight.len();
    let program = GaussianJointRowProgram::new(rows);
    let mut mm = Array1::<f64>::zeros(n);
    let mut ml = Array1::<f64>::zeros(n);
    let mut ll = Array1::<f64>::zeros(n);
    for row in 0..n {
        let atom = program.row_order2(row);
        mm[row] = atom.hessian_at(0, 0);
        ml[row] = atom.hessian_at(0, 1);
        ll[row] = atom.hessian_at(1, 1);
    }
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
    //! #932 doctrine oracle for the generated Gaussian location-scale row
    //! program and its OBSERVED joint-Hessian tower.
    //!
    //! Every non-wiggle live score/Hessian/ψ channel is a chain-rule projection
    //! of the symbolic order2/third/fourth lowerings emitted from
    //! `gaussian_normalized_row`. The generic nested-jet evaluator and the
    //! likelihood-only finite differences below are independent witnesses.
    //!
    //! MECHANICAL SOURCE (no hand math reused):
    //!  * The per-row negative log-likelihood is `ρ(μ,η)=−ℓ(μ,η)`, evaluated by
    //!    the production row kernel `gaussian_diagonal_row_kernel`.
    //!  * Its OBSERVED 2×2 Hessian in `(μ,η_ls)` and every directional chain are
    //!    taken by central finite differences of that likelihood alone.

    use super::*;
    use ndarray::array;

    /// Row negative log-likelihood from the production kernel (likelihood only,
    /// no curvature coefficients involved).
    fn row_nll(y: f64, mu: f64, eta_ls: f64, a: f64) -> f64 {
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        -gaussian_diagonal_row_kernel(0, y, mu, eta_ls, a, ln2pi)
            .expect("representable Gaussian oracle row")
            .log_likelihood
    }

    fn gradient_fd(y: f64, mu: f64, eta_ls: f64, a: f64, h: f64) -> [f64; 2] {
        [
            (row_nll(y, mu + h, eta_ls, a) - row_nll(y, mu - h, eta_ls, a)) / (2.0 * h),
            (row_nll(y, mu, eta_ls + h, a) - row_nll(y, mu, eta_ls - h, a)) / (2.0 * h),
        ]
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

    fn hessian_fd(y: f64, mu: f64, eta_ls: f64, a: f64, h: f64) -> [[f64; 2]; 2] {
        let (mm, ml, ll) = observed_hessian_fd(y, mu, eta_ls, a, h);
        [[mm, ml], [ml, ll]]
    }

    fn path_point(
        base: [f64; 2],
        direction_a: [f64; 2],
        direction_b: [f64; 2],
        direction_ab: [f64; 2],
        s: f64,
        t: f64,
    ) -> [f64; 2] {
        [
            base[0] + s * direction_a[0] + t * direction_b[0] + s * t * direction_ab[0],
            base[1] + s * direction_a[1] + t * direction_b[1] + s * t * direction_ab[1],
        ]
    }

    fn mixed_value_fd(
        y: f64,
        base: [f64; 2],
        a: f64,
        direction_a: [f64; 2],
        direction_b: [f64; 2],
        direction_ab: [f64; 2],
        h: f64,
    ) -> f64 {
        let pp = path_point(base, direction_a, direction_b, direction_ab, h, h);
        let pm = path_point(base, direction_a, direction_b, direction_ab, h, -h);
        let mp = path_point(base, direction_a, direction_b, direction_ab, -h, h);
        let mm = path_point(base, direction_a, direction_b, direction_ab, -h, -h);
        (row_nll(y, pp[0], pp[1], a) - row_nll(y, pm[0], pm[1], a) - row_nll(y, mp[0], mp[1], a)
            + row_nll(y, mm[0], mm[1], a))
            / (4.0 * h * h)
    }

    fn mixed_gradient_fd(
        y: f64,
        base: [f64; 2],
        a: f64,
        direction_a: [f64; 2],
        direction_b: [f64; 2],
        direction_ab: [f64; 2],
        outer_h: f64,
        inner_h: f64,
    ) -> [f64; 2] {
        let pp = path_point(
            base,
            direction_a,
            direction_b,
            direction_ab,
            outer_h,
            outer_h,
        );
        let pm = path_point(
            base,
            direction_a,
            direction_b,
            direction_ab,
            outer_h,
            -outer_h,
        );
        let mp = path_point(
            base,
            direction_a,
            direction_b,
            direction_ab,
            -outer_h,
            outer_h,
        );
        let mm = path_point(
            base,
            direction_a,
            direction_b,
            direction_ab,
            -outer_h,
            -outer_h,
        );
        let gpp = gradient_fd(y, pp[0], pp[1], a, inner_h);
        let gpm = gradient_fd(y, pm[0], pm[1], a, inner_h);
        let gmp = gradient_fd(y, mp[0], mp[1], a, inner_h);
        let gmm = gradient_fd(y, mm[0], mm[1], a, inner_h);
        std::array::from_fn(|axis| {
            (gpp[axis] - gpm[axis] - gmp[axis] + gmm[axis]) / (4.0 * outer_h * outer_h)
        })
    }

    fn mixed_hessian_fd(
        y: f64,
        base: [f64; 2],
        a: f64,
        direction_a: [f64; 2],
        direction_b: [f64; 2],
        direction_ab: [f64; 2],
        outer_h: f64,
        inner_h: f64,
    ) -> [[f64; 2]; 2] {
        let pp = path_point(
            base,
            direction_a,
            direction_b,
            direction_ab,
            outer_h,
            outer_h,
        );
        let pm = path_point(
            base,
            direction_a,
            direction_b,
            direction_ab,
            outer_h,
            -outer_h,
        );
        let mp = path_point(
            base,
            direction_a,
            direction_b,
            direction_ab,
            -outer_h,
            outer_h,
        );
        let mm = path_point(
            base,
            direction_a,
            direction_b,
            direction_ab,
            -outer_h,
            -outer_h,
        );
        let hpp = hessian_fd(y, pp[0], pp[1], a, inner_h);
        let hpm = hessian_fd(y, pm[0], pm[1], a, inner_h);
        let hmp = hessian_fd(y, mp[0], mp[1], a, inner_h);
        let hmm = hessian_fd(y, mm[0], mm[1], a, inner_h);
        std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                (hpp[row][column] - hpm[row][column] - hmp[row][column] + hmm[row][column])
                    / (4.0 * outer_h * outer_h)
            })
        })
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
        let band = tolerance * actual.abs().max(expected.abs()).max(1.0);
        assert!(
            (actual - expected).abs() <= band,
            "{label}: actual={actual:+.15e} expected={expected:+.15e} band={band:.3e}"
        );
    }

    fn assert_matrix_close(
        actual: &[[f64; 2]; 2],
        expected: &[[f64; 2]; 2],
        tolerance: f64,
        label: &str,
    ) {
        for row in 0..2 {
            for column in 0..2 {
                assert_close(
                    actual[row][column],
                    expected[row][column],
                    tolerance,
                    &format!("{label}[{row},{column}]"),
                );
            }
        }
    }

    /// Production observed joint-Hessian coefficients for a single row.
    fn production_observed_row(y: f64, mu: f64, eta_ls: f64, a: f64) -> (f64, f64, f64) {
        let rows = gaussian_jointrow_scalars(&array![y], &array![mu], &array![eta_ls], &array![a])
            .expect("row scalars");
        let (mm, ml, ll) = gaussian_locscale_observed_joint_row_coeffs(&rows);
        (mm[0], ml[0], ll[0])
    }

    #[test]
    fn generated_gaussian_psi_chain_matches_generic_nested_jet_all_channels_932() {
        use gam_math::jet_tower::{
            program_fourth_contracted, program_row_kernel, program_third_contracted,
        };

        let rows =
            gaussian_jointrow_scalars(&array![0.55], &array![0.3], &array![-0.4], &array![1.7])
                .expect("row scalars");
        let program = GaussianJointRowProgram::new(&rows);
        let direction_a = [0.5, -0.7];
        let direction_b = [-0.3, 0.8];
        let direction_ab = [0.2, -0.15];
        let drift = [0.6, 0.25];
        let psi = [-0.4, 0.75];
        let drift_psi = [0.12, -0.18];

        let atom = program.row_order2(0);
        let (jet_value, jet_score, jet_hessian) =
            program_row_kernel(&program, 0).expect("generic order2");
        assert_close(atom.value(), jet_value, 1e-12, "row value");
        for axis in 0..2 {
            assert_close(
                atom.gradient()[axis],
                jet_score[axis],
                1e-12,
                &format!("row score[{axis}]"),
            );
        }
        let generated_hessian = [
            [atom.hessian_at(0, 0), atom.hessian_at(0, 1)],
            [atom.hessian_at(1, 0), atom.hessian_at(1, 1)],
        ];
        assert_matrix_close(&generated_hessian, &jet_hessian, 1e-12, "row Hessian");

        let jet_third_a =
            program_third_contracted(&program, 0, &direction_a).expect("generic third a");
        let jet_third_ab =
            program_third_contracted(&program, 0, &direction_ab).expect("generic third ab");
        let jet_fourth_ab = program_fourth_contracted(&program, 0, &direction_a, &direction_b)
            .expect("generic fourth ab");
        assert_matrix_close(
            &program.row_third_contracted(0, &direction_a),
            &jet_third_a,
            1e-9,
            "generated dH",
        );
        assert_matrix_close(
            &program.row_fourth_contracted(0, &direction_a, &direction_b),
            &jet_fourth_ab,
            1e-9,
            "generated d2H",
        );

        let first = gaussian_joint_psi_firstweights(
            &rows,
            &array![direction_a[0]],
            &array![direction_a[1]],
        );
        let expected_score_a = matrix_vector_2(&jet_hessian, &direction_a);
        assert_close(
            first.objective_psirow[0],
            dot_2(&jet_score, &direction_a),
            1e-9,
            "first psi objective",
        );
        assert_close(first.scoremu[0], jet_score[0], 1e-12, "first score mu");
        assert_close(first.score_ls[0], jet_score[1], 1e-12, "first score ls");
        assert_close(
            first.dscoremu[0],
            expected_score_a[0],
            1e-9,
            "first dscore mu",
        );
        assert_close(
            first.dscore_ls[0],
            expected_score_a[1],
            1e-9,
            "first dscore ls",
        );
        assert_close(first.hmumu[0], jet_hessian[0][0], 1e-12, "first H mm");
        assert_close(first.hmu_ls[0], jet_hessian[0][1], 1e-12, "first H ml");
        assert_close(first.h_ls_ls[0], jet_hessian[1][1], 1e-12, "first H ll");
        assert_close(first.dhmumu[0], jet_third_a[0][0], 1e-9, "first dH mm");
        assert_close(first.dhmu_ls[0], jet_third_a[0][1], 1e-9, "first dH ml");
        assert_close(first.dh_ls_ls[0], jet_third_a[1][1], 1e-9, "first dH ll");

        let second = gaussian_joint_psisecondweights(
            &rows,
            &array![direction_a[0]],
            &array![direction_a[1]],
            &array![direction_b[0]],
            &array![direction_b[1]],
            &array![direction_ab[0]],
            &array![direction_ab[1]],
        );
        let expected_second_score = add_vector_2(
            matrix_vector_2(&jet_third_a, &direction_b),
            matrix_vector_2(&jet_hessian, &direction_ab),
        );
        let expected_second_hessian = add_matrix_2(jet_fourth_ab, jet_third_ab);
        assert_close(
            second.objective_psi_psirow[0],
            dot_2(&direction_a, &matrix_vector_2(&jet_hessian, &direction_b))
                + dot_2(&jet_score, &direction_ab),
            1e-9,
            "second psi objective",
        );
        assert_close(
            second.d2scoremu[0],
            expected_second_score[0],
            1e-9,
            "second psi score mu",
        );
        assert_close(
            second.d2score_ls[0],
            expected_second_score[1],
            1e-9,
            "second psi score ls",
        );
        assert_close(
            second.d2hmumu[0],
            expected_second_hessian[0][0],
            1e-9,
            "second psi H mm",
        );
        assert_close(
            second.d2hmu_ls[0],
            expected_second_hessian[0][1],
            1e-9,
            "second psi H ml",
        );
        assert_close(
            second.d2h_ls_ls[0],
            expected_second_hessian[1][1],
            1e-9,
            "second psi H ll",
        );

        let mixed = gaussian_joint_psi_mixed_driftweights(
            &rows,
            &array![drift[0]],
            &array![drift[1]],
            &array![psi[0]],
            &array![psi[1]],
            &array![drift_psi[0]],
            &array![drift_psi[1]],
        );
        let jet_third_drift =
            program_third_contracted(&program, 0, &drift).expect("generic third drift");
        let jet_mixed_hessian = add_matrix_2(
            program_fourth_contracted(&program, 0, &drift, &psi).expect("generic fourth mixed"),
            program_third_contracted(&program, 0, &drift_psi)
                .expect("generic third mixed direction"),
        );
        assert_close(
            mixed.dhmumu_u[0],
            jet_third_drift[0][0],
            1e-9,
            "mixed dH mm",
        );
        assert_close(
            mixed.dhmu_ls_u[0],
            jet_third_drift[0][1],
            1e-9,
            "mixed dH ml",
        );
        assert_close(
            mixed.dh_ls_ls_u[0],
            jet_third_drift[1][1],
            1e-9,
            "mixed dH ll",
        );
        assert_close(
            mixed.d2hmumu[0],
            jet_mixed_hessian[0][0],
            1e-9,
            "mixed d2H mm",
        );
        assert_close(
            mixed.d2hmu_ls[0],
            jet_mixed_hessian[0][1],
            1e-9,
            "mixed d2H ml",
        );
        assert_close(
            mixed.d2h_ls_ls[0],
            jet_mixed_hessian[1][1],
            1e-9,
            "mixed d2H ll",
        );
    }

    #[test]
    fn generated_gaussian_psi_chain_matches_likelihood_finite_differences_932() {
        let y = 0.55;
        let base = [0.3, -0.4];
        let weight = 1.7;
        let direction_a = [0.5, -0.7];
        let direction_b = [-0.3, 0.8];
        let direction_ab = [0.2, -0.15];
        let drift = [0.6, 0.25];
        let psi = [-0.4, 0.75];
        let drift_psi = [0.12, -0.18];
        let rows = gaussian_jointrow_scalars(
            &array![y],
            &array![base[0]],
            &array![base[1]],
            &array![weight],
        )
        .expect("row scalars");
        let program = GaussianJointRowProgram::new(&rows);
        let atom = program.row_order2(0);

        let normalized_value =
            0.5 * weight * rows.standardized_residual[0] * rows.standardized_residual[0];
        assert_close(
            atom.value(),
            normalized_value,
            1e-12,
            "normalized row value",
        );
        let score_fd = gradient_fd(y, base[0], base[1], weight, 2e-5);
        let hessian_fd0 = hessian_fd(y, base[0], base[1], weight, 1e-3);
        for axis in 0..2 {
            assert_close(
                atom.gradient()[axis],
                score_fd[axis],
                2e-8,
                &format!("score FD {axis}"),
            );
        }
        let atom_hessian = [
            [atom.hessian_at(0, 0), atom.hessian_at(0, 1)],
            [atom.hessian_at(1, 0), atom.hessian_at(1, 1)],
        ];
        assert_matrix_close(&atom_hessian, &hessian_fd0, 3e-6, "Hessian FD");

        let first = gaussian_joint_psi_firstweights(
            &rows,
            &array![direction_a[0]],
            &array![direction_a[1]],
        );
        let first_step = 1e-2;
        let plus = [
            base[0] + first_step * direction_a[0],
            base[1] + first_step * direction_a[1],
        ];
        let minus = [
            base[0] - first_step * direction_a[0],
            base[1] - first_step * direction_a[1],
        ];
        let objective_first_fd = (row_nll(y, plus[0], plus[1], weight)
            - row_nll(y, minus[0], minus[1], weight))
            / (2.0 * first_step);
        let gradient_plus = gradient_fd(y, plus[0], plus[1], weight, 2e-5);
        let gradient_minus = gradient_fd(y, minus[0], minus[1], weight, 2e-5);
        let score_first_fd: [f64; 2] = std::array::from_fn(|axis| {
            (gradient_plus[axis] - gradient_minus[axis]) / (2.0 * first_step)
        });
        let hessian_plus = hessian_fd(y, plus[0], plus[1], weight, 1e-3);
        let hessian_minus = hessian_fd(y, minus[0], minus[1], weight, 1e-3);
        let hessian_first_fd: [[f64; 2]; 2] = std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                (hessian_plus[row][column] - hessian_minus[row][column]) / (2.0 * first_step)
            })
        });
        assert_close(
            first.objective_psirow[0],
            objective_first_fd,
            2e-4,
            "first objective FD",
        );
        assert_close(
            first.dscoremu[0],
            score_first_fd[0],
            2e-4,
            "first score mu FD",
        );
        assert_close(
            first.dscore_ls[0],
            score_first_fd[1],
            2e-4,
            "first score ls FD",
        );
        assert_close(
            first.dhmumu[0],
            hessian_first_fd[0][0],
            2e-3,
            "first H mm FD",
        );
        assert_close(
            first.dhmu_ls[0],
            hessian_first_fd[0][1],
            2e-3,
            "first H ml FD",
        );
        assert_close(
            first.dh_ls_ls[0],
            hessian_first_fd[1][1],
            2e-3,
            "first H ll FD",
        );

        let second = gaussian_joint_psisecondweights(
            &rows,
            &array![direction_a[0]],
            &array![direction_a[1]],
            &array![direction_b[0]],
            &array![direction_b[1]],
            &array![direction_ab[0]],
            &array![direction_ab[1]],
        );
        let objective_second_fd = mixed_value_fd(
            y,
            base,
            weight,
            direction_a,
            direction_b,
            direction_ab,
            5e-3,
        );
        let score_second_fd = mixed_gradient_fd(
            y,
            base,
            weight,
            direction_a,
            direction_b,
            direction_ab,
            1e-2,
            2e-5,
        );
        let hessian_second_fd = mixed_hessian_fd(
            y,
            base,
            weight,
            direction_a,
            direction_b,
            direction_ab,
            2e-2,
            1e-3,
        );
        assert_close(
            second.objective_psi_psirow[0],
            objective_second_fd,
            5e-4,
            "second objective FD",
        );
        assert_close(
            second.d2scoremu[0],
            score_second_fd[0],
            2e-3,
            "second score mu FD",
        );
        assert_close(
            second.d2score_ls[0],
            score_second_fd[1],
            2e-3,
            "second score ls FD",
        );
        assert_close(
            second.d2hmumu[0],
            hessian_second_fd[0][0],
            3e-2,
            "second H mm FD",
        );
        assert_close(
            second.d2hmu_ls[0],
            hessian_second_fd[0][1],
            3e-2,
            "second H ml FD",
        );
        assert_close(
            second.d2h_ls_ls[0],
            hessian_second_fd[1][1],
            3e-2,
            "second H ll FD",
        );

        let mixed = gaussian_joint_psi_mixed_driftweights(
            &rows,
            &array![drift[0]],
            &array![drift[1]],
            &array![psi[0]],
            &array![psi[1]],
            &array![drift_psi[0]],
            &array![drift_psi[1]],
        );
        let drift_plus = [
            base[0] + first_step * drift[0],
            base[1] + first_step * drift[1],
        ];
        let drift_minus = [
            base[0] - first_step * drift[0],
            base[1] - first_step * drift[1],
        ];
        let drift_hessian_plus = hessian_fd(y, drift_plus[0], drift_plus[1], weight, 1e-3);
        let drift_hessian_minus = hessian_fd(y, drift_minus[0], drift_minus[1], weight, 1e-3);
        let drift_hessian_fd: [[f64; 2]; 2] = std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                (drift_hessian_plus[row][column] - drift_hessian_minus[row][column])
                    / (2.0 * first_step)
            })
        });
        let mixed_hessian = mixed_hessian_fd(y, base, weight, drift, psi, drift_psi, 2e-2, 1e-3);
        assert_close(
            mixed.dhmumu_u[0],
            drift_hessian_fd[0][0],
            2e-3,
            "mixed dH mm FD",
        );
        assert_close(
            mixed.dhmu_ls_u[0],
            drift_hessian_fd[0][1],
            2e-3,
            "mixed dH ml FD",
        );
        assert_close(
            mixed.dh_ls_ls_u[0],
            drift_hessian_fd[1][1],
            2e-3,
            "mixed dH ll FD",
        );
        assert_close(
            mixed.d2hmumu[0],
            mixed_hessian[0][0],
            3e-2,
            "mixed d2H mm FD",
        );
        assert_close(
            mixed.d2hmu_ls[0],
            mixed_hessian[0][1],
            3e-2,
            "mixed d2H ml FD",
        );
        assert_close(
            mixed.d2h_ls_ls[0],
            mixed_hessian[1][1],
            3e-2,
            "mixed d2H ll FD",
        );
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
            let rows =
                gaussian_jointrow_scalars(&array![y], &array![mu], &array![eta_ls], &array![a])
                    .expect("row scalars");
            let (w_u, c_u, d_u) =
                gaussian_joint_first_directionalweights(&rows, &array![xi_mu], &array![xi_ls]);
            let coeffs_at =
                |m: f64, e: f64| -> (f64, f64, f64) { production_observed_row(y, m, e, a) };
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
            let rows =
                gaussian_jointrow_scalars(&array![y], &array![mu], &array![eta_ls], &array![a])
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

    /// #932 jet oracle for the LIVE gaulss third/fourth directional Hessian
    /// drifts. The FD tests above are the independent numerical witness; these
    /// two pin the SAME live hand closed forms
    /// (`gaussian_joint_first_directionalweights` /
    /// `gaussian_jointsecond_directionalweights`) against the universal gam-math
    /// Taylor jet — the mechanical single source — at ≤1e-9, closing the audit
    /// gap that the gam-math gaulss oracle covered only value/∇/observed-H.
    mod jet_third_fourth_oracle {
        use super::*;
        use gam_math::jet_tower::{program_fourth_contracted, program_third_contracted};

        fn close(hand: f64, jet: f64, label: &str) {
            let band = 1e-9 + 1e-9 * hand.abs().max(jet.abs());
            assert!(
                (hand - jet).abs() <= band,
                "{label}: hand {hand:+.15e} vs jet {jet:+.15e} (band {band:.3e})"
            );
        }

        /// `gaussian_joint_first_directionalweights` (the LIVE third-order ∂_dir
        /// of the observed Hessian) equals the jet's contracted third at ≤1e-9.
        #[test]
        fn first_directional_weights_match_jet_third() {
            let cases = [
                (0.3_f64, -0.4_f64, 1.0_f64, 0.5_f64, -0.7_f64, 0.55_f64),
                (-1.2, 0.7, 2.5, 1.1, 0.3, -0.9),
                (0.0, 1.5, 0.4, -0.2, 0.9, 0.35),
                (2.4, -1.1, 0.8, 0.6, -0.4, 1.7),
                (-0.6, 0.2, 3.3, 0.8, -1.0, -1.1),
            ];
            for &(mu, eta_ls, a, xi_mu, xi_ls, y) in &cases {
                let rows =
                    gaussian_jointrow_scalars(&array![y], &array![mu], &array![eta_ls], &array![a])
                        .expect("row scalars");
                let (w_u, c_u, d_u) =
                    gaussian_joint_first_directionalweights(&rows, &array![xi_mu], &array![xi_ls]);
                let prog = crate::gamlss::GaussianJointRowProgram::new(&rows);
                let jt = program_third_contracted(&prog, 0, &[xi_mu, xi_ls]).expect("jet third");
                close(w_u[0], jt[0][0], &format!("dH_μμ μ={mu} η={eta_ls}"));
                close(c_u[0], jt[0][1], &format!("dH_μls μ={mu} η={eta_ls}"));
                close(c_u[0], jt[1][0], &format!("dH_lsμ μ={mu} η={eta_ls}"));
                close(d_u[0], jt[1][1], &format!("dH_lsls μ={mu} η={eta_ls}"));
            }
        }

        /// `gaussian_jointsecond_directionalweights` (the LIVE fourth-order
        /// ∂_u∂_v of the observed Hessian) equals the jet's contracted fourth at
        /// ≤1e-9.
        #[test]
        fn second_directional_weights_match_jet_fourth() {
            let cases = [
                (
                    0.3_f64, -0.4_f64, 1.0_f64, 0.5_f64, -0.7_f64, 0.8_f64, 0.2_f64, 0.55_f64,
                ),
                (-1.2, 0.7, 2.5, 1.1, 0.3, -0.6, 0.9, -0.9),
                (0.0, 1.5, 0.4, -0.2, 0.9, 0.4, -0.5, 0.35),
                (2.4, -1.1, 0.8, 0.6, -0.4, -0.3, 0.7, 1.7),
            ];
            for &(mu, eta_ls, a, xi_mu_u, xi_ls_u, xi_mu_v, xi_ls_v, y) in &cases {
                let rows =
                    gaussian_jointrow_scalars(&array![y], &array![mu], &array![eta_ls], &array![a])
                        .expect("row scalars");
                let (w_uv, c_uv, d_uv) = gaussian_jointsecond_directionalweights(
                    &rows,
                    &array![xi_mu_u],
                    &array![xi_ls_u],
                    &array![xi_mu_v],
                    &array![xi_ls_v],
                );
                let prog = crate::gamlss::GaussianJointRowProgram::new(&rows);
                let jt =
                    program_fourth_contracted(&prog, 0, &[xi_mu_u, xi_ls_u], &[xi_mu_v, xi_ls_v])
                        .expect("jet fourth");
                close(w_uv[0], jt[0][0], &format!("d²H_μμ μ={mu} η={eta_ls}"));
                close(c_uv[0], jt[0][1], &format!("d²H_μls μ={mu} η={eta_ls}"));
                close(c_uv[0], jt[1][0], &format!("d²H_lsμ μ={mu} η={eta_ls}"));
                close(d_uv[0], jt[1][1], &format!("d²H_lsls μ={mu} η={eta_ls}"));
            }
        }
    }
}
