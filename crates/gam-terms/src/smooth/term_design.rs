// Term-collection design construction (#1521): the `build_term_collection_design`
// subgraph relocated DOWN from `gam-models`
// (`fit_orchestration/drivers/design_construction.rs`) into `gam_terms::smooth`,
// where all of its callees and output types already live. This breaks the old
// families -> fit_orchestration::drivers back-edge for design construction.
//
// This is a child module of `gam_terms::smooth` (not an `include!`d flat file),
// so `use super::*` inherits the whole smooth-module import + definition surface
// (prelude imports, `term_specs.rs` spec/design machinery, the
// `structure_analysis` re-exports). Bodies are byte-identical to the gam-models
// original except: (1) `gam_terms::basis::` paths rewritten to `crate::basis::`
// for the in-crate boundary, and (2) the three entry points that staying
// gam-models drivers still call are `pub` (re-exported from `smooth.rs`).
use super::*;

use super::shape_constraints::{
    linear_constraints_from_lower_bounds_global, merge_linear_constraints_global,
};
use super::structure_analysis::smooth_has_frozen_identifiability;
use crate::basis::{
    ConstantCurvatureIdentifiability, MaternIdentifiability, MeasureJetIdentifiability,
    SphericalSplineIdentifiability, orthogonality_transform_for_design,
};
use gam_linalg::matrix::{CoefficientTransformOperator, RandomEffectOperator};
use ndarray::ArrayView1;

/// Empirical L² mass of a scalar basis function under the uniform measure on
/// the observed rows. A linear term has one realized basis column `b`; using
/// `G = n⁻¹ bᵀb` makes its shrinkage energy `β²G = n⁻¹‖bβ‖²`, a property of
/// the fitted function values rather than of the arbitrary coefficient scale.
fn linear_function_mass(column: ArrayView1<'_, f64>, term_name: &str) -> Result<f64, BasisError> {
    if column.is_empty() {
        crate::bail_invalid_basis!(
            "linear term '{term_name}' cannot define a function-space penalty on zero rows"
        );
    }
    let scale = column.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    if !scale.is_finite() {
        crate::bail_invalid_basis!(
            "linear term '{term_name}' has a non-finite realized design column"
        );
    }
    if scale == 0.0 {
        crate::bail_invalid_basis!(
            "linear term '{term_name}' is identically zero and cannot carry a recoverable effect"
        );
    }
    let scaled_mean_square = column
        .iter()
        .map(|&value| {
            let normalized = value / scale;
            normalized * normalized
        })
        .sum::<f64>()
        / column.len() as f64;
    let mass = scale * scale * scaled_mean_square;
    if !mass.is_finite() || mass <= 0.0 {
        crate::bail_invalid_basis!(
            "linear term '{term_name}' has an invalid empirical function mass {mass}"
        );
    }
    Ok(mass)
}

pub fn build_term_collection_design_inner(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
) -> Result<TermCollectionDesign, BasisError> {
    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    build_term_collection_design_inner_with_policy(data, spec, &policy)
}

/// Build a planned term collection while preserving the caller's resource
/// policy through the actual basis realization. The policy must reach the
/// [`BasisWorkspace`]: using it only while lowering the formula spec leaves a
/// later spatial build free to reverse the routing decision under the library
/// default.
pub fn build_term_collection_design_inner_with_policy(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
    policy: &gam_runtime::resource::ResourcePolicy,
) -> Result<TermCollectionDesign, BasisError> {
    use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

    let n = data.nrows();
    let p_intercept = usize::from(!term_collection_has_anchored_bspline(spec));
    let p_lin = spec.linear_terms.len();

    // Smooth construction, random-effect construction, and linear-column
    // extraction are independent at this stage. Run them concurrently, but keep
    // each result in spec order so the final global layout remains stable:
    // [intercept | linear | random_effects | smooth].
    let (smooth_raw_result, (random_blocks_result, linear_block_result)) = rayon::join(
        || {
            let mut ws = crate::basis::BasisWorkspace::with_policy(policy.clone());
            build_smooth_design_withworkspace_unvalidated(data, &spec.smooth_terms, &mut ws)
        },
        || {
            rayon::join(
                || {
                    spec.random_effect_terms
                        .par_iter()
                        .map(|term| build_random_effect_block(data, term))
                        .collect::<Result<Vec<_>, _>>()
                },
                || -> Result<Option<Array2<f64>>, BasisError> {
                    if p_lin == 0 {
                        return Ok(None);
                    }

                    let linear_columns = (0..p_lin)
                        .into_par_iter()
                        .map(|j| {
                            let linear = &spec.linear_terms[j];
                            // `:` interactions carry multiple feature columns; the
                            // materialized column is their elementwise product
                            // (a plain main effect has a single column), gated by
                            // any categorical-level indicators for a factor-aware
                            // `factor:x` expansion. `realized_design_column`
                            // validates bounds and is the single authority every
                            // design path (this one, the incremental realizer in
                            // `build_term_collection_fixed_blocks`, and the
                            // marginal-slope rank check) shares so they agree on
                            // every interaction.
                            linear
                                .realized_design_column(data)
                                .map_err(BasisError::InvalidInput)
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    let mut out = Array2::<f64>::zeros((n, p_lin));
                    for (j, column) in linear_columns.iter().enumerate() {
                        out.column_mut(j).assign(column);
                    }
                    Ok(Some(out))
                },
            )
        },
    );

    let smooth_raw = smooth_raw_result?;
    let random_blocks = random_blocks_result?;
    let linear_block = linear_block_result?;
    // Reuse the TRAINING-time mass when this spec has already been through
    // `freeze_term_collection_from_design` (predict/rebuild calls always pass
    // the frozen `resolvedspec`). Recomputing from `block.column(j)` here would
    // use whatever rows THIS call happens to be building over — a held-out
    // grid, a handful of group/class anchors, a single test row — where a
    // covariate that varies fine across the training set can easily look
    // constant by chance. Only the very first, fit-time build (before the
    // spec is frozen, so `frozen_function_mass` is still `None`) computes the
    // mass fresh from `data`, which is right: at fit time `data` IS the
    // training rows, so a genuine identifiability failure is real (#1561).
    let linear_function_masses = match linear_block.as_ref() {
        Some(block) => spec
            .linear_terms
            .iter()
            .enumerate()
            .map(|(j, term)| -> Result<Option<f64>, BasisError> {
                if !term.double_penalty {
                    return Ok(None);
                }
                if let Some(frozen_mass) = term.frozen_function_mass {
                    return Ok(Some(frozen_mass));
                }
                linear_function_mass(block.column(j), &term.name).map(Some)
            })
            .collect::<Result<Vec<_>, _>>()?,
        None => Vec::new(),
    };

    let (smooth, affine_offset) = apply_global_smooth_identifiability(
        smooth_raw,
        data,
        &spec.linear_terms,
        &spec.smooth_terms,
    )?;

    let p_rand: usize = random_blocks.iter().map(|b| b.num_groups).sum();
    let p_smooth = smooth.total_smooth_cols();
    let p_total = p_intercept + p_lin + p_rand + p_smooth;

    let mut linear_ranges = Vec::<(String, Range<usize>)>::with_capacity(p_lin);
    for (j, linear) in spec.linear_terms.iter().enumerate() {
        let col = p_intercept + j;
        // Column ranges are in the global (full) coordinate system:
        // [intercept | linear | random_effects | smooth]
        linear_ranges.push((linear.name.clone(), col..(col + 1)));
    }

    // Track random-effect column ranges in the global coordinate system.
    // Global layout: [intercept(1) | linear(p_lin) | RE_0(q0) | RE_1(q1) | … | smooth(p_smooth)]
    let mut random_effect_ranges =
        Vec::<(String, Range<usize>)>::with_capacity(random_blocks.len());
    let mut random_effect_levels = Vec::<(String, Vec<u64>)>::with_capacity(random_blocks.len());
    let mut col_cursor = p_intercept + p_lin;
    for block in &random_blocks {
        let q = block.num_groups;
        let end = col_cursor + q;
        random_effect_ranges.push((block.name.clone(), col_cursor..end));
        random_effect_levels.push((block.name.clone(), block.kept_levels.clone()));
        col_cursor = end;
    }

    // ── Assemble the full DesignMatrix ────────────────────────────────
    //
    // Always use a BlockDesignOperator with per-term blocks.  The full
    // (n, p_total) dense matrix is NEVER materialized:
    //
    //   Block 0:     Intercept — zero storage, implicit all-ones column
    //   Block 1:     Linear terms — (n, p_lin) extracted from data
    //   Blocks 2..k: Random-effect operators — O(n) one-hot, no dense storage
    //   Blocks k+1..: Per-smooth-term dense blocks — each (n, p_term)
    //
    // Splitting smooth terms into per-term blocks means cross-block grams
    // are small O(p_i × p_j) BLAS operations.  Tensor product terms with
    // Kronecker structure become DesignBlock::Operator, avoiding the full
    // n × ∏q_j materialization.

    let mut blocks = Vec::<DesignBlock>::new();

    // Block 0: intercept — zero storage. An anchored B-spline (one *or* two
    // sided) consumes the absolute level at its pinned endpoint(s), so a free
    // intercept would float the whole curve off the pin and violate the
    // structural anchor.
    if p_intercept == 1 {
        blocks.push(DesignBlock::Intercept(n));
    }

    // Block 1: linear terms.
    if let Some(lin_block) = linear_block {
        blocks.push(DesignBlock::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(lin_block),
        ));
    }

    // Blocks: random-effect operators — O(n) implicit one-hot.
    for block in &random_blocks {
        let re_op = RandomEffectOperator::new(block.group_ids.clone(), block.num_groups);
        blocks.push(DesignBlock::RandomEffect(Arc::new(re_op)));
    }

    // Blocks: per-smooth-term.  Each smooth term gets its own block so that
    // cross-block grams are tiny. Tensor terms can stay operator-backed all
    // the way from basis construction; dense smooth terms stay dense.
    if p_smooth > 0 {
        for term_design in &smooth.term_designs {
            match term_design {
                DesignMatrix::Dense(dense) => blocks.push(DesignBlock::Dense(dense.clone())),
                DesignMatrix::Sparse(sparse) => blocks.push(DesignBlock::Sparse(sparse.clone())),
            }
        }
    }

    let design = assemble_term_collection_design_matrix(blocks)?;

    let mut penalties = Vec::<BlockwisePenalty>::new();
    let mut nullspace_dims = Vec::<usize>::new();
    let mut penaltyinfo = Vec::<PenaltyBlockInfo>::new();
    let mut dropped_penaltyinfo = Vec::<DroppedPenaltyBlockInfo>::new();
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(p_total, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraintrows = Vec::<Array1<f64>>::new();
    let mut linear_constraint_b = Vec::<f64>::new();

    for (j, linear) in spec.linear_terms.iter().enumerate() {
        let col = p_intercept + j;
        if let Some(lb) = linear.coefficient_min {
            let mut row = Array1::<f64>::zeros(p_total);
            row[col] = 1.0;
            linear_constraintrows.push(row);
            linear_constraint_b.push(lb);
        }
        if let Some(ub) = linear.coefficient_max {
            let mut row = Array1::<f64>::zeros(p_total);
            row[col] = -1.0;
            linear_constraintrows.push(row);
            linear_constraint_b.push(-ub);
        }
    }

    // Every non-intercept effect owns one independent REML coordinate. For a
    // scalar linear basis `b_j`, the physical null-recovery penalty is its
    // empirical function Gram `G_j = n⁻¹b_jᵀb_j`, so `β_j²G_j` is exactly the
    // mean squared fitted effect. Under a harmless rescaling `b_j -> c b_j`,
    // `β_j -> β_j/c`, the quadratic functional is unchanged. Keeping each
    // term in its own one-column block also lets REML remove unsupported
    // effects independently instead of forcing unrelated slopes to share λ.
    for (j, linear) in spec.linear_terms.iter().enumerate() {
        let Some(function_mass) = linear_function_masses.get(j).copied().flatten() else {
            continue;
        };
        let col = p_intercept + j;
        let global_index = penalties.len();
        penalties.push(BlockwisePenalty::new(
            col..(col + 1),
            Array2::from_elem((1, 1), function_mass),
        ));
        nullspace_dims.push(0);
        penaltyinfo.push(PenaltyBlockInfo {
            global_index,
            termname: Some(linear.name.clone()),
            penalty: ActivePenaltyInfo {
                source: PenaltySource::Other("LinearTermRidge".to_string()),
                original_index: j,
                effective_rank: 1,
                normalization_scale: 1.0,
                kronecker_factors: None,
                structural_null_frame: None,
            },
        });
    }

    for (re_idx, (name, range)) in random_effect_ranges.iter().enumerate() {
        if range.is_empty() || !spec.random_effect_terms[re_idx].penalized {
            continue;
        }
        let block_size = range.len();
        let global_index = penalties.len();
        penalties.push(BlockwisePenalty::ridge(range.clone(), 1.0));
        nullspace_dims.push(0);
        penaltyinfo.push(PenaltyBlockInfo {
            global_index,
            termname: Some(name.clone()),
            penalty: ActivePenaltyInfo {
                source: PenaltySource::Other(format!("RandomEffectRidge({name})")),
                original_index: re_idx,
                effective_rank: block_size,
                normalization_scale: 1.0,
                kronecker_factors: None,
                structural_null_frame: None,
            },
        });
    }

    if smooth.penaltyinfo.len() != smooth.penalties.len() {
        gam_problem::bail_invalid_basis!(
            "smooth penalty metadata mismatch: penalties={}, metadata={}",
            smooth.penalties.len(),
            smooth.penaltyinfo.len()
        );
    }
    let smooth_start = p_intercept + p_lin + p_rand;
    for ((bp_smooth, &ns), localinfo) in smooth
        .penalties
        .iter()
        .zip(smooth.nullspace_dims.iter())
        .zip(smooth.penaltyinfo.iter())
    {
        let global_index = penalties.len();
        // Offset the per-term block range from smooth-local to model-global.
        let offset_range =
            (bp_smooth.col_range.start + smooth_start)..(bp_smooth.col_range.end + smooth_start);
        let bp = if let Some(factors) = localinfo.penalty.kronecker_factors.as_ref() {
            BlockwisePenalty::kronecker(offset_range, bp_smooth.local.clone(), factors.clone())
                .with_op(bp_smooth.op.clone())
        } else if matches!(
            localinfo.penalty.source,
            PenaltySource::Other(ref s) if s.starts_with("RandomEffectRidge")
        ) {
            BlockwisePenalty::ridge(offset_range, 1.0)
        } else {
            BlockwisePenalty::new(offset_range, bp_smooth.local.clone())
                .with_op(bp_smooth.op.clone())
        };
        penalties.push(bp);
        nullspace_dims.push(ns);
        penaltyinfo.push(PenaltyBlockInfo {
            global_index,
            termname: localinfo.termname.clone(),
            penalty: localinfo.penalty.clone(),
        });
    }
    dropped_penaltyinfo.extend(smooth.dropped_penaltyinfo.iter().cloned());

    assert_eq!(
        penalties.len(),
        nullspace_dims.len(),
        "term-collection penalty/nullspace bookkeeping diverged"
    );
    assert_eq!(
        penalties.len(),
        penaltyinfo.len(),
        "term-collection penalty metadata bookkeeping diverged"
    );

    if let Some(lb_smooth) = smooth.coefficient_lower_bounds.as_ref() {
        let start = p_intercept + p_lin + p_rand;
        coefficient_lower_bounds
            .slice_mut(s![start..(start + p_smooth)])
            .assign(lb_smooth);
        any_bounds = true;
    }
    if let Some(lin_smooth) = smooth.linear_constraints.as_ref() {
        let mut a_global = Array2::<f64>::zeros((lin_smooth.a.nrows(), p_total));
        let start = p_intercept + p_lin + p_rand;
        a_global
            .slice_mut(s![.., start..(start + p_smooth)])
            .assign(&lin_smooth.a);
        for r in 0..a_global.nrows() {
            linear_constraintrows.push(a_global.row(r).to_owned());
            linear_constraint_b.push(lin_smooth.b[r]);
        }
    }

    // Canonical constraint path: convert any explicit lower bounds into linear
    // inequalities and merge into the global constraint matrix. This keeps fitting
    // behavior independent of user-facing lower-bound options.
    let lower_bound_constraints = if any_bounds {
        linear_constraints_from_lower_bounds_global(&coefficient_lower_bounds)
    } else {
        None
    };
    let explicit_linear_constraints = if linear_constraintrows.is_empty() {
        None
    } else {
        let mut a = Array2::<f64>::zeros((linear_constraintrows.len(), p_total));
        for (i, row) in linear_constraintrows.iter().enumerate() {
            a.row_mut(i).assign(row);
        }
        Some(LinearInequalityConstraints {
            a,
            b: Array1::from_vec(linear_constraint_b),
        })
    };
    let linear_constraints =
        merge_linear_constraints_global(explicit_linear_constraints, lower_bound_constraints)?;

    Ok(TermCollectionDesign {
        design,
        affine_offset,
        penalties,
        nullspace_dims,
        penaltyinfo,
        dropped_penaltyinfo,
        coefficient_lower_bounds: if any_bounds {
            Some(coefficient_lower_bounds)
        } else {
            None
        },
        linear_constraints,
        intercept_range: 0..p_intercept,
        linear_ranges,
        linear_function_masses,
        random_effect_ranges,
        random_effect_levels,
        smooth,
    })
}

/// Whether any smooth term carries an anchored B-spline endpoint (one *or* two
/// sided). Such a term fixes the function's absolute level through its endpoint
/// pin, so it becomes the model's level gauge: the global intercept is
/// suppressed and the term is not additionally sum-to-zero centered.
pub fn term_collection_has_anchored_bspline(spec: &TermCollectionSpec) -> bool {
    spec.smooth_terms
        .iter()
        .any(|term| smooth_basis_has_anchored_bspline(&term.basis))
}

/// Whether any smooth term realizes an inhomogeneous endpoint anchor and thus
/// contributes a fixed affine row offset.
pub fn term_collection_has_nonzero_anchor(spec: &TermCollectionSpec) -> bool {
    spec.smooth_terms
        .iter()
        .any(|term| smooth_basis_has_nonzero_anchor(&term.basis))
}

fn smooth_basis_has_nonzero_anchor(basis: &SmoothBasisSpec) -> bool {
    match basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => smooth_basis_has_nonzero_anchor(inner),
        SmoothBasisSpec::BSpline1D { spec, .. } => spec.boundary_conditions.has_nonzero_anchor(),
        SmoothBasisSpec::BySmooth { smooth, .. } => smooth_basis_has_nonzero_anchor(smooth),
        SmoothBasisSpec::TensorBSpline { spec, .. } => spec
            .marginalspecs
            .iter()
            .any(|marginal| marginal.boundary_conditions.has_nonzero_anchor()),
        SmoothBasisSpec::FactorSmooth { spec } => {
            spec.marginal.boundary_conditions.has_nonzero_anchor()
        }
        SmoothBasisSpec::ThinPlate { .. }
        | SmoothBasisSpec::Sphere { .. }
        | SmoothBasisSpec::ConstantCurvature { .. }
        | SmoothBasisSpec::Matern { .. }
        | SmoothBasisSpec::MeasureJet { .. }
        | SmoothBasisSpec::Duchon { .. }
        | SmoothBasisSpec::Pca { .. } => false,
    }
}

fn smooth_basis_has_anchored_bspline(basis: &SmoothBasisSpec) -> bool {
    match basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            smooth_basis_has_anchored_bspline(inner)
        }
        SmoothBasisSpec::BSpline1D { spec, .. } => {
            bspline_conditions_have_anchor(&spec.boundary_conditions)
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => smooth_basis_has_anchored_bspline(smooth),
        SmoothBasisSpec::TensorBSpline { spec, .. } => spec
            .marginalspecs
            .iter()
            .any(|marginal| bspline_conditions_have_anchor(&marginal.boundary_conditions)),
        SmoothBasisSpec::FactorSmooth { .. }
        | SmoothBasisSpec::ThinPlate { .. }
        | SmoothBasisSpec::Sphere { .. }
        | SmoothBasisSpec::ConstantCurvature { .. }
        | SmoothBasisSpec::Matern { .. }
        | SmoothBasisSpec::MeasureJet { .. }
        | SmoothBasisSpec::Duchon { .. }
        | SmoothBasisSpec::Pca { .. } => false,
    }
}

fn bspline_conditions_have_anchor(conditions: &crate::basis::BSplineBoundaryConditions) -> bool {
    conditions.has_anchor()
}

pub fn build_term_collection_design(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
) -> Result<TermCollectionDesign, BasisError> {
    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    build_term_collection_design_with_policy(data, spec, &policy)
}

/// Policy-aware counterpart to [`build_term_collection_design`]. Center
/// planning is identical; only the basis workspace's materialization contract
/// differs.
pub fn build_term_collection_design_with_policy(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
    policy: &gam_runtime::resource::ResourcePolicy,
) -> Result<TermCollectionDesign, BasisError> {
    validate_term_collection_finite_inputs(data, spec)?;
    let mut planned_specs =
        plan_joint_spatial_centers_for_term_blocks(data, &[spec.smooth_terms.clone()])?;
    let planned_smooth_terms = planned_specs.pop().ok_or_else(|| {
        BasisError::InvalidInput(
            "joint spatial center planner returned no smooth terms for single-spec build"
                .to_string(),
        )
    })?;
    let mut planned_spec = spec.clone();
    planned_spec.smooth_terms = planned_smooth_terms;
    build_term_collection_design_inner_with_policy(data, &planned_spec, policy)
}

/// Exact analytic derivative of an affine term-collection realization.
#[derive(Debug, Clone)]
pub struct TermCollectionDerivativeDesign {
    /// `∂design(x)/∂x_c`, aligned column-for-column with the value design.
    pub design: Array2<f64>,
    /// `∂affine_offset(x)/∂x_c` on the same rows.
    pub affine_offset: Array1<f64>,
}

impl TermCollectionDerivativeDesign {
    /// Evaluate `∂affine_offset/∂x_c + (∂design/∂x_c) * beta`.
    pub fn apply(&self, beta: ArrayView1<'_, f64>) -> Result<Array1<f64>, BasisError> {
        if beta.len() != self.design.ncols() {
            crate::bail_dim_basis!(
                "term-collection derivative coefficient length {} does not match design width {}",
                beta.len(),
                self.design.ncols()
            );
        }
        if self.affine_offset.len() != self.design.nrows() {
            crate::bail_dim_basis!(
                "term-collection derivative affine offset has {} rows but derivative design has {}",
                self.affine_offset.len(),
                self.design.nrows()
            );
        }
        if beta.iter().any(|value| !value.is_finite())
            || self.affine_offset.iter().any(|value| !value.is_finite())
        {
            crate::bail_invalid_basis!(
                "term-collection derivative coefficients and affine offset must be finite"
            );
        }
        Ok(self.design.dot(&beta.to_owned()) + &self.affine_offset)
    }
}

/// Build the EXACT analytic average-derivative realization of a term
/// collection: `D = ∂design/∂x_c` plus the distinct fixed channel
/// `d = ∂affine_offset/∂x_c`. The two are laid out on the same rows as
/// `build_term_collection_design`, so the fitted derivative is `d + D * beta`
/// (#1120/#2297).
///
/// This is a provably exact analytic derivative of the design, so the production
/// path differentiates the model basis (a known analytic function) in closed form.
///
/// The construction differentiates each term's BASIS w.r.t. `deriv_col` and pushes
/// the basis derivative through the SAME frozen identifiability/orthogonalization
/// transform the value design uses. That transform is captured exactly by the
/// per-term `metadata.identifiability_transform` from the value build: for every
/// 1-D B-spline term the term's value design equals `B_raw(x) · M` where
/// `M = metadata.identifiability_transform` is the composed
/// `raw → boundary → sum-to-zero/joint-null/global-orthogonality` chart (it is a
/// pure linear operator with no additive offset — sum-to-zero centering is a
/// column reparameterization `Z`, not a subtracted mean). Differentiating gives
/// `∂(design)/∂x = B'_raw(x) · M`. Additive constants (the intercept column) drop
/// to zero; terms not involving `deriv_col`, and random-effect blocks, contribute
/// zero columns; a linear main effect equal to `deriv_col` differentiates to 1.
///
/// # Supported structure
///
/// The realistic, tested usage is a single 1-D B-spline / P-spline smooth `s(x)`
/// differentiated w.r.t. its one covariate. Supported: `SmoothBasisSpec::BSpline1D`
/// (non-periodic) over the differentiated feature column, the intercept (zero),
/// random-effect blocks (zero), smooths over other columns (zero), and linear
/// terms (analytic product rule). Any other basis that actually involves
/// `deriv_col` (tensor products, `ByVariable`, factor smooths, Duchon/thin-plate,
/// sphere, periodic B-splines, …) returns a clear `Err` rather than a wrong
/// number or a silent numeric approximation.
pub fn build_term_collection_derivative_design(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
    deriv_col: usize,
) -> Result<TermCollectionDerivativeDesign, BasisError> {
    if deriv_col >= data.ncols() {
        return Err(BasisError::InvalidInput(format!(
            "average-derivative column {deriv_col} out of range for data with {} columns",
            data.ncols()
        )));
    }

    // The value design fixes the exact column layout and carries every term's
    // realized identifiability transform in its metadata. Reusing it guarantees
    // the derivative design aligns column-for-column with the fitted β.
    let value = build_term_collection_design(data, spec)?;
    let n = data.nrows();
    let p_total = value.design.ncols();
    let mut d = Array2::<f64>::zeros((n, p_total));
    let mut affine_derivative = Array1::<f64>::zeros(n);

    // Global layout: [intercept | linear | random_effects | smooth].
    let p_intercept = value.intercept_range.len();
    let p_lin = spec.linear_terms.len();
    let p_rand: usize = value
        .random_effect_ranges
        .iter()
        .map(|(_, range)| range.len())
        .sum();

    // Intercept column: constant ⇒ derivative 0 (already zero).
    // Random-effect blocks: piecewise-constant group indicators ⇒ 0 (already zero).

    // Linear terms: analytic product rule for the realized design column.
    for (j, linear) in spec.linear_terms.iter().enumerate() {
        let col = p_intercept + j;
        let derivative = linear_term_derivative_column(data, linear, deriv_col)?;
        if let Some(column) = derivative {
            d.column_mut(col).assign(&column);
        }
    }

    // Smooth terms: differentiate the basis of any term over `deriv_col`.
    let smooth_start = p_intercept + p_lin + p_rand;
    if value.smooth.terms.len() != spec.smooth_terms.len() {
        return Err(BasisError::InvalidInput(format!(
            "average-derivative design: value build produced {} smooth terms but spec has {}",
            value.smooth.terms.len(),
            spec.smooth_terms.len()
        )));
    }
    for (idx, termspec) in spec.smooth_terms.iter().enumerate() {
        let term_value = &value.smooth.terms[idx];
        let feature_cols = smooth_term_feature_cols(termspec);
        if !feature_cols.contains(&deriv_col) {
            // Term does not involve the differentiated covariate ⇒ zero columns.
            continue;
        }
        let (block, term_affine_derivative) =
            smooth_term_first_derivative_block(data, termspec, term_value, deriv_col)?;
        let range = (term_value.coeff_range.start + smooth_start)
            ..(term_value.coeff_range.end + smooth_start);
        if block.ncols() != range.len() {
            return Err(BasisError::DimensionMismatch(format!(
                "average-derivative design: smooth term '{}' derivative block has {} columns \
                 but the fitted block spans {}",
                termspec.name,
                block.ncols(),
                range.len()
            )));
        }
        d.slice_mut(s![.., range]).assign(&block);
        if let Some(term_offset) = term_affine_derivative {
            if term_offset.len() != n {
                return Err(BasisError::DimensionMismatch(format!(
                    "average-derivative design: smooth term '{}' affine derivative has {} rows but the data has {n}",
                    termspec.name,
                    term_offset.len()
                )));
            }
            affine_derivative += &term_offset;
        }
    }

    Ok(TermCollectionDerivativeDesign {
        design: d,
        affine_offset: affine_derivative,
    })
}

/// Analytic `∂/∂x_{deriv_col}` of a linear term's realized design column.
///
/// The realized column is `gate(x) · ∏_k x_{c_k}` where `gate` is a product of
/// categorical-level indicators (constant w.r.t. a continuous covariate) and the
/// `c_k` are the numeric feature columns. The product rule gives
/// `∂/∂x_d = gate · Σ_{j: c_j = d} ∏_{k ≠ j} x_{c_k}`. Returns `None` when the
/// term does not depend on `deriv_col` (its columns are zero).
fn linear_term_derivative_column(
    data: ArrayView2<'_, f64>,
    linear: &LinearTermSpec,
    deriv_col: usize,
) -> Result<Option<Array1<f64>>, BasisError> {
    let numeric_cols: Vec<usize> = if linear.categorical_levels.is_empty() {
        linear.effective_feature_cols()
    } else {
        linear.feature_cols.clone()
    };
    let occurrences = numeric_cols.iter().filter(|&&c| c == deriv_col).count();
    if occurrences == 0 {
        return Ok(None);
    }
    let n = data.nrows();
    let p = data.ncols();
    for &c in &numeric_cols {
        if c >= p {
            return Err(BasisError::InvalidInput(format!(
                "linear term '{}' feature column {c} out of bounds for {p} columns",
                linear.name
            )));
        }
    }

    // gate(x): categorical-level indicators (constant w.r.t. a continuous axis).
    let mut gate = Array1::<f64>::ones(n);
    for &(col, level_bits) in &linear.categorical_levels {
        if col >= p {
            return Err(BasisError::InvalidInput(format!(
                "linear term '{}' categorical column {col} out of bounds for {p} columns",
                linear.name
            )));
        }
        let level_bits = gam_data::canonical_level_bits(f64::from_bits(level_bits));
        for (row, g) in gate.iter_mut().enumerate() {
            if gam_data::canonical_level_bits(data[[row, col]]) != level_bits {
                *g = 0.0;
            }
        }
    }

    // Product rule: sum over each occurrence of `deriv_col`, dropping that factor.
    let mut derivative = Array1::<f64>::zeros(n);
    for (j, &c_j) in numeric_cols.iter().enumerate() {
        if c_j != deriv_col {
            continue;
        }
        let mut term = gate.clone();
        for (k, &c_k) in numeric_cols.iter().enumerate() {
            if k != j {
                term *= &data.column(c_k);
            }
        }
        derivative += &term;
    }
    Ok(Some(derivative))
}

/// Analytic first-derivative design block for a single smooth term over
/// `deriv_col`, aligned column-for-column with that term's value design block.
///
/// Only non-periodic 1-D B-splines are analytically supported. The block is
/// `B'_raw(x) · M` where `B'_raw` is the raw B-spline basis FIRST DERIVATIVE on
/// the term's frozen knots/degree and `M = metadata.identifiability_transform`
/// is the same linear chart the value design applied (see
/// `build_term_collection_derivative_design`).
fn smooth_term_first_derivative_block(
    data: ArrayView2<'_, f64>,
    termspec: &SmoothTermSpec,
    term_value: &SmoothTerm,
    deriv_col: usize,
) -> Result<(Array2<f64>, Option<Array1<f64>>), BasisError> {
    let feature_col = match &termspec.basis {
        SmoothBasisSpec::BSpline1D { feature_col, .. } => *feature_col,
        other => {
            return Err(BasisError::InvalidInput(format!(
                "analytic average-derivative design only supports non-periodic 1-D B-spline \
                 smooths over the differentiated covariate; term '{}' uses unsupported basis {}",
                termspec.name,
                smooth_basis_kind_label(other)
            )));
        }
    };
    if feature_col != deriv_col {
        // The caller only dispatches here when `deriv_col` is one of the term's
        // feature columns, and a `BSpline1D` term has exactly one. This guards
        // the invariant rather than trusting it silently.
        return Err(BasisError::InvalidInput(format!(
            "analytic average-derivative design: B-spline term '{}' is over column {feature_col}, \
             not the differentiated column {deriv_col}",
            termspec.name
        )));
    }

    let (knots, degree, transform, periodic, anchor_offset_coeffs) = match &term_value.metadata {
        BasisMetadata::BSpline1D {
            knots,
            degree,
            identifiability_transform,
            periodic,
            anchor_offset_coeffs,
            ..
        } => (
            knots,
            *degree,
            identifiability_transform.as_ref(),
            periodic,
            anchor_offset_coeffs.as_ref(),
        ),
        other => {
            return Err(BasisError::InvalidInput(format!(
                "analytic average-derivative design expected B-spline metadata for term '{}', \
                 found {other:?}",
                termspec.name
            )));
        }
    };
    if periodic.is_some() {
        return Err(BasisError::InvalidInput(format!(
            "analytic average-derivative design does not support periodic/cyclic B-spline \
             term '{}'",
            termspec.name
        )));
    }
    let degree = degree.ok_or_else(|| {
        BasisError::InvalidInput(format!(
            "B-spline term '{}' metadata is missing its effective degree",
            termspec.name
        ))
    })?;

    // Raw B-spline basis FIRST DERIVATIVE on the frozen knot geometry.
    let (deriv_basis_arc, _) = crate::basis::create_basis::<crate::basis::Dense>(
        data.column(deriv_col),
        crate::basis::KnotSource::Provided(knots.view()),
        degree,
        crate::basis::BasisOptions::first_derivative(),
    )?;
    let deriv_basis = deriv_basis_arc.as_ref();

    let affine_derivative = match anchor_offset_coeffs {
        Some(beta_p) => {
            if deriv_basis.ncols() != beta_p.len() {
                return Err(BasisError::DimensionMismatch(format!(
                    "B-spline term '{}': raw derivative basis has {} columns but the affine anchor lift has {} coefficients",
                    termspec.name,
                    deriv_basis.ncols(),
                    beta_p.len()
                )));
            }
            Some(deriv_basis.dot(beta_p))
        }
        None => None,
    };

    // Push the basis derivative through the SAME frozen linear chart the value
    // design used. The fixed affine channel remains separate and is returned
    // above, so neither channel is mistaken for a fitted coefficient.
    let block = match transform {
        Some(z) => {
            if deriv_basis.ncols() != z.nrows() {
                return Err(BasisError::DimensionMismatch(format!(
                    "B-spline term '{}': raw derivative basis has {} columns but the frozen \
                     identifiability transform has {} rows",
                    termspec.name,
                    deriv_basis.ncols(),
                    z.nrows()
                )));
            }
            gam_linalg::faer_ndarray::fast_ab(deriv_basis, z)
        }
        None => deriv_basis.to_owned(),
    };
    Ok((block, affine_derivative))
}

/// Short human-readable label for a smooth basis variant, used only in the
/// unsupported-basis error of the analytic average-derivative design.
fn smooth_basis_kind_label(basis: &SmoothBasisSpec) -> &'static str {
    match basis {
        SmoothBasisSpec::BSpline1D { .. } => "BSpline1D",
        SmoothBasisSpec::TensorBSpline { .. } => "TensorBSpline",
        SmoothBasisSpec::ByVariable { .. } => "ByVariable",
        SmoothBasisSpec::FactorSumToZero { .. } => "FactorSumToZero",
        SmoothBasisSpec::FactorSmooth { .. } => "FactorSmooth",
        SmoothBasisSpec::BySmooth { .. } => "BySmooth",
        SmoothBasisSpec::ThinPlate { .. } => "ThinPlate",
        SmoothBasisSpec::Duchon { .. } => "Duchon",
        SmoothBasisSpec::Matern { .. } => "Matern",
        SmoothBasisSpec::Sphere { .. } => "Sphere",
        SmoothBasisSpec::ConstantCurvature { .. } => "ConstantCurvature",
        SmoothBasisSpec::MeasureJet { .. } => "MeasureJet",
        SmoothBasisSpec::Pca { .. } => "Pca",
    }
}

/// How one smooth's realized design is made orthogonal to its constraint block.
///
/// The two arms are not a preference. A DELETION costs one coefficient direction
/// per parametric direction and is free only where that direction is inside the
/// design's span; RESIDUALIZATION costs none and is available always. See the
/// fork in [`apply_global_smooth_identifiability`] and the derivation on
/// [`crate::basis::parametric_residualization_for_design`].
enum GlobalIdentifiabilityPlan {
    /// No constraint block for this term.
    Absent,
    /// Every resolvable constraint direction is contained in the design's span,
    /// so `X·Z` loses nothing. This arm is the pre-`76a520c45` path bit for bit.
    Delete { block: Array2<f64> },
    /// At least one direction is not contained: project in row space instead, so
    /// the model keeps the function the deletion would have removed.
    Residualize { block: Array2<f64> },
}

impl GlobalIdentifiabilityPlan {
    /// The frozen, ψ-independent half of this plan, for export onto the term.
    fn as_gauge(
        &self,
        owner_terms: &[usize],
        has_parametric_block: bool,
        local_identifiability_transform: Option<Array2<f64>>,
        coefficient_transform: Array2<f64>,
        local_columns: usize,
    ) -> Option<SmoothCollectionGauge> {
        let (arm, block) = match self {
            Self::Absent => return None,
            Self::Delete { block } => (SmoothCollectionGaugeArm::Delete, block),
            Self::Residualize { block } => (SmoothCollectionGaugeArm::Residualize, block),
        };
        Some(SmoothCollectionGauge {
            arm,
            constraint_block: block.clone(),
            owner_terms: owner_terms.to_vec(),
            has_parametric_block,
            local_identifiability_transform,
            coefficient_transform,
            local_columns,
        })
    }
}

/// The identifiability chart a basis's own build applied, read off its metadata.
///
/// This is `z_local` — the term-local half. After
/// [`realize_smooth_collection_gauge`] runs, the metadata carries the
/// COMPOSITION `z_local · T`, so this must be read BEFORE the gauge composes,
/// which is exactly where [`SmoothCollectionGauge::local_identifiability_transform`]
/// is filled from (gam#2760).
///
/// Every variant that can carry a chart is listed; the ones that cannot report
/// `None` by construction rather than through a wildcard, so a new basis family
/// has to decide this question rather than inherit an answer.
fn basis_local_identifiability_transform(metadata: &BasisMetadata) -> Option<Array2<f64>> {
    match metadata {
        BasisMetadata::BSpline1D {
            identifiability_transform,
            ..
        }
        | BasisMetadata::CubicRegression1D {
            identifiability_transform,
            ..
        }
        | BasisMetadata::ThinPlate {
            identifiability_transform,
            ..
        }
        | BasisMetadata::Matern {
            identifiability_transform,
            ..
        }
        | BasisMetadata::Duchon {
            identifiability_transform,
            ..
        }
        | BasisMetadata::TensorBSpline {
            identifiability_transform,
            ..
        } => identifiability_transform.clone(),
        BasisMetadata::Sphere {
            constraint_transform,
            ..
        }
        | BasisMetadata::ConstantCurvature {
            constraint_transform,
            ..
        }
        | BasisMetadata::MeasureJet {
            constraint_transform,
            ..
        } => constraint_transform.clone(),
        BasisMetadata::Pca { .. }
        | BasisMetadata::SphereHarmonics { .. }
        | BasisMetadata::BySmooth { .. }
        | BasisMetadata::FactorSmooth { .. } => None,
    }
}

/// One smooth term's realized block, put into a COLLECTION gauge.
pub struct RealizedCollectionGauge {
    /// `P_C X_local T0`, represented lazily as `X_local T0 − C R`.
    pub design: DesignMatrix,
    /// The coefficient transform this realization applied, for composition into
    /// the term's basis metadata by the caller.
    pub coefficient_transform: Array2<f64>,
    /// The moving row-space correction in the fixed coefficient chart. Both
    /// arms carry it: away from the reference psi even a `Delete` chart needs a
    /// nonzero left projection while keeping its right-hand section fixed.
    pub residualization: crate::basis::ParametricResidualization,
}

/// Derive the collection's fixed coefficient chart at its reference
/// realization.
///
/// This is the ONLY operation that may choose RRQR/eigenvectors.  A later
/// spatial-psi replay uses [`realize_smooth_collection_gauge`] with the chart
/// stored on [`SmoothCollectionGauge`]; differentiating or replaying freshly
/// chosen vectors would differentiate a numerical coordinate convention rather
/// than the statistical smooth (gam#2760).
fn derive_smooth_collection_coefficient_transform(
    design_local: &DesignMatrix,
    arm: SmoothCollectionGaugeArm,
    block: ArrayView2<'_, f64>,
    has_owner_terms: bool,
) -> Result<Array2<f64>, BasisError> {
    match arm {
        SmoothCollectionGaugeArm::Delete => {
            match orthogonality_transform_for_design(design_local, block, None) {
                Ok(transform) => Ok(transform),
                // Mirrors the collection's historical fallback: a constraint
                // block made entirely of owner columns may consume the whole
                // dependent block.
                Err(BasisError::ConstraintNullspaceCollapsed { .. }) if has_owner_terms => {
                    Ok(Array2::zeros((design_local.ncols(), 0)))
                }
                Err(error) => Err(error),
            }
        }
        SmoothCollectionGaugeArm::Residualize => {
            Ok(crate::basis::parametric_residualization_for_design(
                design_local,
                block,
                None, // fixed subspace: never use iteration-varying PIRLS weights
            )?
            .coefficient_transform)
        }
    }
}

/// Put a freshly built TERM-LOCAL design into a collection's FROZEN gauge.
///
/// This is the single owner of "make this term's realized block orthogonal to
/// `C`" after the collection has decided both `C` and its coefficient chart
/// `T0`. `apply_global_smooth_identifiability` calls it at the reference
/// realization; the spatial outer search calls it at every moving-psi value.
///
/// The statistical design is represented in one fixed coefficient chart:
///
/// `X_g(psi) = P_C X_local(psi) T0`, `P_C = I - C(C'C)^- C'`.
///
/// Only the row-space correction `R(psi)` moves, because it is the value of the
/// fixed projection on the moving design.  It is returned for predict-time
/// replay.  No RRQR/eigenvector is re-derived here.
///
/// The orthogonality the whole step is for is asserted here, at the same
/// relative bar the collection has always used, so neither caller can produce a
/// block that fails it and report success.
pub fn realize_smooth_collection_gauge(
    design_local: DesignMatrix,
    gauge: &SmoothCollectionGauge,
    termname: &str,
) -> Result<RealizedCollectionGauge, BasisError> {
    let block = gauge.constraint_block.view();
    if block.nrows() != design_local.nrows() {
        gam_problem::bail_dim_basis!(
            "collection gauge row mismatch for term '{termname}': the design has {} rows and the frozen constraint block has {}",
            design_local.nrows(),
            block.nrows()
        );
    }
    if gauge.coefficient_transform.nrows() != gauge.local_columns {
        gam_problem::bail_dim_basis!(
            "collection gauge for term '{termname}' declares {} local columns but its fixed coefficient chart has {} rows",
            gauge.local_columns,
            gauge.coefficient_transform.nrows()
        );
    }
    if design_local.ncols() != gauge.local_columns {
        gam_problem::bail_dim_basis!(
            "collection gauge local width mismatch for term '{termname}': the design has {} columns but the fixed chart was derived on {}",
            design_local.ncols(),
            gauge.local_columns
        );
    }
    let coefficient_transform = gauge.coefficient_transform.clone();
    let design = apply_smooth_transform_to_design(
        design_local,
        &coefficient_transform,
        termname,
    )?;
    let projector = crate::basis::FixedRowSpaceProjector::from_constraint_block(block)?;
    let (design, row_space_correction) = projector.project_design(design, termname)?;
    let residualization = crate::basis::ParametricResidualization {
        coefficient_transform: coefficient_transform.clone(),
        row_space_correction,
    };
    assert_orthogonal_to_constraint_block(&design, block, termname)?;
    Ok(RealizedCollectionGauge {
        design,
        coefficient_transform,
        residualization,
    })
}

/// One smooth term as a TERM-LOCAL build leaves it, on its way into a
/// collection gauge. Grouped rather than passed loose because these seven are
/// one object — a realization — and splitting them across a call boundary is
/// how the design and its chart came apart in the first place (#2747).
pub struct LocalTermRealization<'a> {
    /// The rebuilt block, with the basis chart and any joint-null `Q` applied.
    pub design: DesignMatrix,
    /// The local build's metadata; the gauge's transform is composed into it.
    pub metadata: &'a BasisMetadata,
    pub active_penalties: &'a [ActivePenalty],
    pub dropped_penalties: Vec<DroppedPenaltyInfo>,
    pub linear_constraints_local: Option<&'a gam_problem::LinearInequalityConstraints>,
    /// The rotation the local build applied to `design` and reported
    /// separately. It is folded into the returned metadata.
    pub joint_null_rotation: Option<&'a crate::basis::JointNullRotation>,
    pub termname: &'a str,
}

/// One smooth term, rebuilt TERM-LOCALLY and put back into a collection gauge.
pub struct CollectionGaugedTerm {
    pub design: DesignMatrix,
    pub metadata: BasisMetadata,
    pub active_penalties: Vec<ActivePenalty>,
    pub dropped_penalties: Vec<DroppedPenaltyInfo>,
    pub linear_constraints_local: Option<gam_problem::LinearInequalityConstraints>,
    pub parametric_residualization: Option<ParametricResidualizationChart>,
}

/// Put a TERM-LOCAL rebuild back into the gauge its collection decided (#2747).
///
/// This is the whole per-term tail of `apply_global_smooth_identifiability`,
/// available to a caller that holds ONE term rather than a collection: the
/// design through [`realize_smooth_collection_gauge`], the penalties through
/// `penalty_candidates_under_collection_gauge`, the local inequality rows
/// through the same congruence, and the coefficient transform composed into the
/// basis metadata so a later freeze carries it.
///
/// # Why the joint-null rotation is consumed here
///
/// A term-local build applies `Q` to its design and reports it separately; a
/// collection-built term reports `None` and carries `Q · T` inside its metadata,
/// because a chart split across two fields has an ORDER, and the two fields do
/// not record it. Coming out of this function a term is in the collection's
/// convention, which is what makes it substitutable for one.
///
/// The caller must therefore clear the term's `joint_null_rotation`; the value
/// it passes in is folded in here.
pub fn place_term_in_collection_gauge(
    gauge: &SmoothCollectionGauge,
    local: LocalTermRealization<'_>,
) -> Result<CollectionGaugedTerm, BasisError> {
    let LocalTermRealization {
        design,
        metadata,
        active_penalties,
        dropped_penalties,
        linear_constraints_local,
        joint_null_rotation,
        termname,
    } = local;
    let realized = realize_smooth_collection_gauge(design, gauge, termname)?;
    let coefficient_gauge =
        gam_problem::Gauge::from_block_transforms(&[realized.coefficient_transform.clone()]);
    let candidates = penalty_candidates_under_collection_gauge(
        active_penalties,
        Some(&coefficient_gauge),
        termname,
    )?;
    let filtered = filter_penalty_candidates(candidates)?;
    let mut dropped_penalties = dropped_penalties;
    dropped_penalties.extend(filtered.dropped);
    let linear_constraints_local = linear_constraints_local.map(|lin| {
        gam_problem::LinearInequalityConstraints {
            a: lin.a.dot(&coefficient_gauge.block_transform(0)),
            b: lin.b.clone(),
        }
    });
    let realized_transform = match joint_null_rotation {
        Some(rotation) => {
            gam_linalg::faer_ndarray::fast_ab(&rotation.rotation, &realized.coefficient_transform)
        }
        None => realized.coefficient_transform.clone(),
    };
    let metadata = with_identifiability_transform(metadata, Some(&realized_transform))?;
    let parametric_residualization = Some(ParametricResidualizationChart {
        owner_terms: gauge.owner_terms.clone(),
        has_parametric_block: gauge.has_parametric_block,
        correction: realized.residualization.row_space_correction.clone(),
    });
    Ok(CollectionGaugedTerm {
        design: realized.design,
        metadata,
        active_penalties: filtered.active,
        dropped_penalties,
        linear_constraints_local,
        parametric_residualization,
    })
}

/// Largest relative residual tolerated before a constrained design is rejected
/// as not orthogonal to its constraint block.
const ORTHOGONALITY_REL_RESIDUAL_TOL: f64 = 1e-8;

fn assert_orthogonal_to_constraint_block(
    design: &DesignMatrix,
    constraint: ArrayView2<'_, f64>,
    termname: &str,
) -> Result<(), BasisError> {
    let rel = orthogonality_relative_residual_for_design(design, constraint)?;
    if rel > ORTHOGONALITY_REL_RESIDUAL_TOL {
        gam_problem::bail_invalid_basis!(
            "smooth orthogonality residual too large for term '{}': {:.3e} > {:.1e}",
            termname,
            rel,
            ORTHOGONALITY_REL_RESIDUAL_TOL
        );
    }
    Ok(())
}

/// `X − C·R`, keeping `X`'s storage decision: the two are stacked into one
/// [`gam_linalg::matrix::BlockDesignOperator`] and the subtraction becomes the
/// sign of `R` inside a single coefficient transform, so a lazy design stays
/// lazy and the correction is never materialized as an `n × k` block of its own.
fn subtract_row_space_correction(
    design: DesignMatrix,
    constraint: ArrayView2<'_, f64>,
    correction: ArrayView2<'_, f64>,
    termname: &str,
) -> Result<DesignMatrix, BasisError> {
    use gam_linalg::matrix::{BlockDesignOperator, DesignBlock};
    let p = design.ncols();
    let q = constraint.ncols();
    let k = correction.ncols();
    if correction.nrows() != q || p != k {
        return Err(BasisError::InvalidInput(format!(
            "row-space correction shape mismatch for term '{termname}': design is {}x{p}, \
             constraint is {}x{q}, correction is {}x{k}",
            design.nrows(),
            constraint.nrows(),
            correction.nrows(),
        )));
    }
    if q == 0 {
        return Ok(design);
    }
    let design_block = match design {
        DesignMatrix::Dense(inner) => DesignBlock::Dense(inner),
        DesignMatrix::Sparse(inner) => DesignBlock::Sparse(inner),
    };
    let stacked = BlockDesignOperator::new(vec![
        design_block,
        DesignBlock::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            constraint.to_owned(),
        )),
    ])
    .map_err(BasisError::InvalidInput)?;
    // `[I ; −R]`: the identity on the design's own columns, the negated
    // correction on the constraint's.
    let mut transform = Array2::<f64>::zeros((p + q, k));
    for i in 0..p {
        transform[[i, i]] = 1.0;
    }
    for i in 0..q {
        for j in 0..k {
            transform[[p + i, j]] = -correction[[i, j]];
        }
    }
    let operator = gam_linalg::matrix::CoefficientTransformOperator::new(
        gam_linalg::matrix::DenseDesignMatrix::from(Arc::new(stacked)),
        transform,
    )
    .map_err(|e| {
        BasisError::InvalidInput(format!(
            "row-space correction failed for term '{termname}': {e}"
        ))
    })?;
    Ok(DesignMatrix::Dense(
        gam_linalg::matrix::DenseDesignMatrix::from(Arc::new(operator)),
    ))
}

fn build_constraint_block(
    n: usize,
    parametric_block: Option<&Array2<f64>>,
    owner_blocks: &[&DesignMatrix],
) -> Result<Array2<f64>, BasisError> {
    let param_cols = parametric_block.map_or(0, |mat| mat.ncols());
    let owner_cols: usize = owner_blocks.iter().map(|design| design.ncols()).sum();
    let mut block = Array2::<f64>::zeros((n, param_cols + owner_cols));
    let mut col_start = 0usize;
    if let Some(parametric) = parametric_block {
        let col_end = col_start + parametric.ncols();
        block
            .slice_mut(s![.., col_start..col_end])
            .assign(parametric);
        col_start = col_end;
    }
    const CHUNK: usize = 1024;
    for owner in owner_blocks {
        let col_end = col_start + owner.ncols();
        for row_start in (0..n).step_by(CHUNK) {
            let row_end = (row_start + CHUNK).min(n);
            let chunk = (*owner)
                .try_row_chunk(row_start..row_end)
                .map_err(|e| BasisError::InvalidInput(e.to_string()))?;
            block
                .slice_mut(s![row_start..row_end, col_start..col_end])
                .assign(&chunk);
        }
        col_start = col_end;
    }
    Ok(block)
}

fn design_cross_relative_residual(
    lhs: &DesignMatrix,
    rhs: &DesignMatrix,
) -> Result<f64, BasisError> {
    let n = lhs.nrows();
    if rhs.nrows() != n {
        return Err(BasisError::ConstraintMatrixRowMismatch {
            basisrows: n,
            constraintrows: rhs.nrows(),
        });
    }
    const CHUNK: usize = 1024;
    let mut cross = Array2::<f64>::zeros((lhs.ncols(), rhs.ncols()));
    let mut lhs_sumsq = 0.0;
    let mut rhs_sumsq = 0.0;
    for start in (0..n).step_by(CHUNK) {
        let end = (start + CHUNK).min(n);
        let lhs_chunk = lhs
            .try_row_chunk(start..end)
            .map_err(|e| BasisError::InvalidInput(e.to_string()))?;
        let rhs_chunk = rhs
            .try_row_chunk(start..end)
            .map_err(|e| BasisError::InvalidInput(e.to_string()))?;
        cross += &gam_linalg::faer_ndarray::fast_atb(&lhs_chunk, &rhs_chunk);
        lhs_sumsq += lhs_chunk.iter().map(|v| v * v).sum::<f64>();
        rhs_sumsq += rhs_chunk.iter().map(|v| v * v).sum::<f64>();
    }
    let num = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
    let denom = lhs_sumsq.sqrt() * rhs_sumsq.sqrt();
    // A block with no mass overlaps nothing; say so instead of dividing by a
    // floored zero.
    if denom == 0.0 {
        return Ok(0.0);
    }
    Ok(num / denom)
}

fn smooth_has_overlapping_linear_terms(
    linear_terms: &[LinearTermSpec],
    termspec: &SmoothTermSpec,
) -> bool {
    let feature_cols = smooth_term_feature_cols(termspec);
    linear_terms
        .iter()
        .any(|linear| feature_cols.contains(&linear.feature_col))
}

// `pub` so the #1601-orphaned design-assembly regression guards (re-homed into
// gam-models) can assert the intrinsic-parametric column resolution against this
// exact production helper.
pub fn smooth_intrinsic_parametric_feature_cols(
    linear_terms: &[LinearTermSpec],
    term: &SmoothTermSpec,
) -> Vec<usize> {
    // Returns the data columns that should appear in the smooth's parametric
    // constraint block `C = [1, …]`, alongside which the smooth is then
    // ORTHOGONALIZED via `apply_smooth_transform_to_design`.  Every column
    // returned here is therefore a direction that gets projected OUT of the
    // smooth's basis.  The constant intercept is always included by
    // `build_parametric_constraint_block_for_term`, so this function only
    // controls the polynomial axes added on top of that intercept.
    //
    // Ownership rule: explicit linear terms claim their matching axes — and
    // only those axes — for projection.  A standalone smooth (no overlapping
    // linear term) keeps its full polynomial nullspace and is centered only
    // against the implicit intercept; this matches the canonical thin-plate
    // / Duchon model surface where the linear component is part of the
    // smooth itself.
    let feature_cols = smooth_term_feature_cols(term);
    let mut owned = Vec::new();
    for linear in linear_terms {
        if feature_cols.contains(&linear.feature_col) && !owned.contains(&linear.feature_col) {
            owned.push(linear.feature_col);
        }
    }
    owned
}

fn apply_global_smooth_identifiability(
    smooth: RawSmoothDesign,
    data: ArrayView2<'_, f64>,
    linear_terms: &[LinearTermSpec],
    smoothspecs: &[SmoothTermSpec],
) -> Result<(SmoothDesign, Array1<f64>), BasisError> {
    // Global smooth identifiability policy:
    //
    // 1. Any smooth that overlaps explicit linear terms is residualized against
    //    [intercept | overlapping linear columns]. Spatial smooths also keep
    //    their existing parametric orthogonality policy when requested.
    // 2. Higher-order / duplicate smooths are orthogonalized to the realized
    //    design columns of lower-order owned smooths over nested feature sets.
    //
    // This yields a deterministic hierarchical decomposition: lower-order smooths
    // own their subspaces, and broader smooths fit only the residual structure.
    if smoothspecs.len() != smooth.terms.len() {
        gam_problem::bail_dim_basis!(
            "smooth spec count ({}) does not match built term count ({})",
            smoothspecs.len(),
            smooth.terms.len()
        );
    }

    if smooth.terms.is_empty() {
        let RawSmoothDesign {
            term_designs,
            affine_offset,
            penalties,
            nullspace_dims,
            penaltyinfo,
            dropped_penaltyinfo,
            terms,
            coefficient_lower_bounds,
            linear_constraints,
        } = smooth;
        return Ok((
            SmoothDesign {
                term_designs,
                penalties,
                nullspace_dims,
                penaltyinfo,
                dropped_penaltyinfo,
                terms,
                coefficient_lower_bounds,
                linear_constraints,
            },
            affine_offset,
        ));
    }

    let mut local_designs = vec![None; smooth.terms.len()];
    let mut local_active_penalties = vec![Vec::<ActivePenalty>::new(); smooth.terms.len()];
    let mut local_dropped_penalties = vec![Vec::<DroppedPenaltyInfo>::new(); smooth.terms.len()];
    let mut local_metadata = vec![None; smooth.terms.len()];
    let mut local_dims = vec![0usize; smooth.terms.len()];
    let mut local_linear_constraints = vec![None; smooth.terms.len()];
    let mut local_unabsorbed_z = vec![None::<Array2<f64>>; smooth.terms.len()];
    let mut local_residualization =
        vec![None::<ParametricResidualizationChart>; smooth.terms.len()];
    let mut local_collection_gauge = vec![None::<SmoothCollectionGauge>; smooth.terms.len()];

    let SmoothStructureAnalysis {
        ownership_order,
        term_owners,
        ..
    } = analyze_smooth_ownership(smoothspecs);

    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    for &idx in &ownership_order {
        let term = &smooth.terms[idx];
        let termspec = &smoothspecs[idx];
        let design_local = smooth.term_designs[idx].clone();
        // A frozen global-orthogonality chart (#978) is a pure replay: the
        // fit already decided this term's residualization against its owner
        // terms, and that decision is training-row data — rederiving it from
        // new rows would be wrong, and skipping it (the pre-#978 behavior)
        // emitted an unresidualized design wider than the fitted coefficient
        // block. So it bypasses both the owner analysis and the frozen-skip
        // gate below.
        let replay_z = frozen_global_orthogonality(termspec);
        let skip_global_transform = replay_z.is_none()
            && (smooth_has_frozen_identifiability(termspec) || term.lower_bounds_local.is_some());
        // A marginally-centered tensor interaction (`ti(...)`, MarginalSumToZero)
        // has ALREADY removed each axis's main effect analytically, in
        // coefficient space, via its per-margin sum-to-zero reparameterization
        // (B_xZ_x)⊗(B_zZ_z) — exactly mgcv's `ti` construction. Residualizing it
        // a SECOND time against the explicit s(x)/s(z) smooths' realized B-spline
        // column spans is redundant on an exact tensor grid (a no-op there) and
        // actively HARMFUL off-grid: the realized interaction columns share a
        // grid-dependent, jitter-sized projection with the main-effect bases, so
        // the second projection eats genuine pure-interaction curvature the main
        // effects cannot represent. REML then rails the s(x)/s(z) smoothing
        // parameters and the surface under-recovers (~40x, #1470). The analytic
        // marginal centering is the correct and complete main-effect removal, so
        // such a term takes NO owner block.
        let owner_indices = if replay_z.is_some()
            || skip_global_transform
            || termspec.basis.is_marginally_centered_tensor()
            || termspec.basis.is_sum_to_zero_factor_smooth()
        {
            Vec::new()
        } else {
            // Relative cross-residual above which a dependent smooth's design is
            // judged to share column space with an owner term and so needs that
            // owner's block in its identifiability transform.
            const OVERLAP_REL_RESIDUAL_TOL: f64 = 1e-10;
            let owner_cross_checks = term_owners[idx]
                .clone()
                .into_par_iter()
                .map(|owner_idx| {
                    let owner_design = local_designs[owner_idx]
                        .as_ref()
                        .expect("owner design must be available before dependent smooth");
                    design_cross_relative_residual(&design_local, owner_design)
                        .map(|rel| (owner_idx, rel))
                })
                .collect::<Vec<_>>();
            let mut out = Vec::new();
            for check in owner_cross_checks {
                let (owner_idx, rel) = check?;
                if rel > OVERLAP_REL_RESIDUAL_TOL {
                    out.push(owner_idx);
                }
            }
            out
        };
        let owner_blocks = owner_indices
            .iter()
            .map(|owner_idx| {
                local_designs[*owner_idx]
                    .as_ref()
                    .expect("owner design must be available before dependent smooth")
            })
            .collect::<Vec<_>>();
        // A frozen span-preserving residualization (#2747) says this term's
        // realized block is `X·T − C·R`, so `C` must be rebuilt at these rows
        // whatever the transform gates say — the metadata already carried `T`
        // through, and the correction is the half it could not absorb.
        let replay_correction = frozen_parametric_residualization(termspec);
        let needs_parametric_block = match replay_correction {
            Some(chart) => chart.has_parametric_block,
            None => {
                replay_z.is_none()
                    && !skip_global_transform
                    && (smooth_has_overlapping_linear_terms(linear_terms, termspec)
                        || !smooth_intrinsic_parametric_feature_cols(linear_terms, termspec)
                            .is_empty()
                        || smooth_requires_parametric_orthogonality(termspec)
                        // A factor-by-level smooth must always be centered against its
                        // gated level indicator (see `factor_by_level_gate`) so its
                        // within-level constant cannot collide with the treatment-coded
                        // factor main effect — even when no continuous linear term
                        // overlaps it (e.g. `s(x, by=fac)` with no `+ x`).
                        || factor_by_level_gate(termspec).is_some())
            }
        };
        let parametric_block = if !needs_parametric_block {
            None
        } else {
            Some(build_parametric_constraint_block_for_term(
                data,
                linear_terms,
                termspec,
            )?)
        };
        // The replay's own owner blocks, named by the chart rather than
        // re-derived: which owners bound is decided by a cross-residual on the
        // FIT rows, so recomputing it here is the #978 error.
        let replay_owner_blocks = match replay_correction {
            Some(chart) => chart
                .owner_terms
                .iter()
                .map(|owner_idx| {
                    local_designs.get(*owner_idx).and_then(|slot| slot.as_ref()).ok_or_else(|| {
                        BasisError::InvalidInput(format!(
                            "term '{}' replays a parametric residualization against owner term {owner_idx}, which is not available at this point of the rebuild",
                            termspec.name
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => Vec::new(),
        };
        // How this term's design gets made orthogonal to its constraint block.
        //
        // A DELETION — the `X·Z` with `Z` spanning `null((XᵀC)ᵀ)` this step has
        // always applied — removes one coefficient direction per parametric
        // direction the cross resolves, and that is free only under
        // CONTAINMENT: when `C`'s direction is inside `col(X)`, the deleted
        // function IS the parametric column and the parametric block keeps it.
        // Otherwise it removes a function nothing else carries (#2747;
        // `contained_constraint_directions` carries the derivation and the
        // principal-angle test).
        //
        // `76a520c45` withheld the deletion in that case and left NOTHING in
        // its place, which drops the invariant this whole step exists for.
        // Measured under #2747: every
        // Matérn and constant-curvature block then sits at
        // `‖XᵀC‖/(‖X‖‖C‖) = 3e-1 … 5e-1` against the `1e-8` bar asserted twenty
        // lines below whenever a transform IS applied, and — because an owner
        // smooth's realized columns are contained in no other basis's span —
        // `analyze_smooth_ownership`'s hierarchy became inert for EVERY
        // dependent smooth, including the contained (thin-plate) class.
        //
        // So the fork is not delete-or-nothing. It is delete (free, and
        // bit-identical to what shipped) where the block is wholly contained,
        // and RESIDUALIZE — `X̃ = X − C(CᵀC)⁻CᵀX`, span-preserving by
        // construction — everywhere else.
        let plan = if replay_correction.is_some() {
            // A replay decides nothing; the fit already did.
            GlobalIdentifiabilityPlan::Absent
        } else if skip_global_transform
            || (parametric_block.is_none() && owner_blocks.is_empty())
        {
            GlobalIdentifiabilityPlan::Absent
        } else {
                let raw =
                    build_constraint_block(data.nrows(), parametric_block.as_ref(), &owner_blocks)?;
                let contained =
                    crate::basis::contained_constraint_directions(&design_local, raw.view(), None)?;
                if raw.ncols() == 0 {
                    GlobalIdentifiabilityPlan::Absent
                } else if contained.ncols() == raw.ncols() {
                    // Every resolvable direction is contained. `contained` is the
                    // original block verbatim in that case, so this arm is the
                    // pre-`76a520c45` path bit for bit.
                    GlobalIdentifiabilityPlan::Delete { block: contained }
                } else {
                    GlobalIdentifiabilityPlan::Residualize { block: raw }
                }
            };
        // Freeze the COLLECTION decision and its reference coefficient chart.
        // `C`, the arm, and `T0` are invariant under a move of THIS term's basis
        // parameters. Only the value-side row correction `R(psi)` moves, and it
        // is recomputed by projecting `X_local(psi) T0` through the fixed `C`.
        // This keeps the objective out of arbitrary moving RRQR/eigenvector
        // coordinates (#2760).
        // Read the term-local chart BEFORE the gauge composes its own into the
        // metadata below (gam#2760): after `with_identifiability_transform` the
        // two are one matrix and cannot be told apart.
        let collection_coefficient_transform = match &plan {
            GlobalIdentifiabilityPlan::Absent => None,
            GlobalIdentifiabilityPlan::Delete { block } => Some(
                derive_smooth_collection_coefficient_transform(
                    &design_local,
                    SmoothCollectionGaugeArm::Delete,
                    block.view(),
                    !owner_indices.is_empty(),
                )?,
            ),
            GlobalIdentifiabilityPlan::Residualize { block } => Some(
                derive_smooth_collection_coefficient_transform(
                    &design_local,
                    SmoothCollectionGaugeArm::Residualize,
                    block.view(),
                    !owner_indices.is_empty(),
                )?,
            ),
        };
        let collection_gauge = collection_coefficient_transform.map(|transform| {
            plan.as_gauge(
                &owner_indices,
                parametric_block.is_some(),
                basis_local_identifiability_transform(&term.metadata),
                transform,
                design_local.ncols(),
            )
            .expect("a derived collection coefficient chart implies a present gauge")
        });
        let mut residualization: Option<crate::basis::ParametricResidualization> = None;
        let (design_constrained, z_opt) = if let Some(gauge) = collection_gauge.as_ref() {
            // This term takes a gauge, so it is realized through the one entry
            // point that knows how — the same one the outer search's incremental
            // realizer uses, so the two cannot drift.
            //
            // `replay_z`, `skip_global_transform` and a frozen chart all force
            // `plan = Absent` upstream, so reaching here means the collection is
            // DERIVING the gauge on these rows, and nothing frozen is in play.
            let realized = realize_smooth_collection_gauge(design_local, gauge, &term.name)?;
            residualization = Some(realized.residualization);
            (realized.design, Some(realized.coefficient_transform))
        } else {
            // No gauge: either there is nothing to be orthogonal to, or this is
            // a REPLAY, where the fit already decided both halves and only `C`
            // is rebuilt at the new rows.
            let z_opt = if let Some(z) = replay_z {
                if design_local.ncols() != z.nrows() {
                    gam_problem::bail_dim_basis!(
                        "frozen global-orthogonality transform mismatch for term '{}': rebuilt design has {} columns but the persisted fit-time transform has {} rows",
                        term.name,
                        design_local.ncols(),
                        z.nrows()
                    );
                }
                Some(z.clone())
            } else if skip_global_transform {
                None
            } else {
                // No constraint block by construction (`plan` is `Absent`), so
                // this can only return the basis's own frozen chart or nothing.
                maybe_smooth_identifiability_transform(termspec, &design_local, None)?
            };
            let design_transformed = match z_opt.as_ref() {
                Some(z) => apply_smooth_transform_to_design(design_local, z, &term.name)?,
                None => design_local,
            };
            let design_constrained = match replay_correction {
                Some(chart) => {
                    // Predict-time replay. `C` is rebuilt at the NEW rows — the
                    // parametric half from the same spec-deterministic recipe the
                    // fit used, the owner half from the terms the chart NAMES —
                    // and only the correction itself is frozen.
                    let block = build_constraint_block(
                        data.nrows(),
                        parametric_block.as_ref(),
                        &replay_owner_blocks,
                    )?;
                    if block.ncols() != chart.correction.nrows() {
                        gam_problem::bail_dim_basis!(
                            "frozen parametric residualization mismatch for term '{}': rebuilt constraint block has {} columns but the persisted fit-time correction has {} rows",
                            term.name,
                            block.ncols(),
                            chart.correction.nrows()
                        );
                    }
                    subtract_row_space_correction(
                        design_transformed,
                        block.view(),
                        chart.correction.view(),
                        &term.name,
                    )?
                }
                None => design_transformed,
            };
            (design_constrained, z_opt)
        };
        let coefficient_gauge = z_opt
            .as_ref()
            .map(|z| gam_problem::Gauge::from_block_transforms(&[z.clone()]));

        let penalty_candidates = penalty_candidates_under_collection_gauge(
            &term.active_penalties,
            coefficient_gauge.as_ref(),
            &term.name,
        )?;
        let filtered = filter_penalty_candidates(penalty_candidates)?;
        let linear_constraints_constrained =
            if let Some(lin_local) = term.linear_constraints_local.as_ref() {
                if let Some(gauge) = coefficient_gauge.as_ref() {
                    Some(LinearInequalityConstraints {
                        a: lin_local.a.dot(&gauge.block_transform(0)),
                        b: lin_local.b.clone(),
                    })
                } else {
                    Some(lin_local.clone())
                }
            } else {
                None
            };

        // A rebuild that REPLAYED a frozen chart must re-export it, or the next
        // freeze would drop it and the model would stop being predictable.
        local_residualization[idx] = residualization
            .as_ref()
            .map(|plan| ParametricResidualizationChart {
                owner_terms: owner_indices.clone(),
                has_parametric_block: parametric_block.is_some(),
                correction: plan.row_space_correction.clone(),
            })
            .or_else(|| replay_correction.cloned());
        local_collection_gauge[idx] = collection_gauge;
        local_dims[idx] = design_constrained.ncols();
        local_designs[idx] = Some(design_constrained);
        local_active_penalties[idx] = filtered.active;
        local_dropped_penalties[idx] = term.dropped_penalties.clone();
        local_dropped_penalties[idx].extend(filtered.dropped);
        local_linear_constraints[idx] = linear_constraints_constrained;
        let realized_transform = match (term.joint_null_rotation.as_ref(), z_opt.as_ref()) {
            (Some(rotation), Some(z)) => {
                Some(gam_linalg::faer_ndarray::fast_ab(&rotation.rotation, z))
            }
            (Some(rotation), None) => Some(rotation.rotation.clone()),
            (None, Some(z)) => Some(z.clone()),
            (None, None) => None,
        };
        // Factor-smooth kinds cannot absorb the realized transform into their
        // metadata, so it is exported on the term instead and persisted onto
        // the spec by `freeze_term_collection_from_design` (#978):
        //
        // - Block-replicated factor smooths (`bs="sz"` → `FactorSumToZero`)
        //   carry PER-MARGINAL metadata (predict rebuilds the single inner
        //   marginal then re-stacks the `L-1` sum-to-zero deviation blocks).
        //   The realized transform lives in the FULL `p·(L-1)`-column design
        //   space, so it cannot be folded into the per-marginal metadata (the
        //   dimensions don't compose; folding it in both crashed basis
        //   generation and would double-count `Q` on rebuild, #700). The raw
        //   design builder reapplies `Q` deterministically at predict time, so
        //   only the global-orthogonality `Z` (post-`Q` chart) is exported.
        //
        // - `FactorSmooth` (`fs`/`re`) metadata has no transform slot at all
        //   (its `with_identifiability_transform` arm rejects one). Like `sz`,
        //   any stage-2 joint-null `Q` is recomputed by the raw builder on
        //   rebuild (and is typically absent: `fs` penalties are full-rank),
        //   so the exported chart is likewise the post-`Q` `Z` alone.
        //
        // Without this export the overlap residualization of
        // `s(x) + s(g, x, bs=sz)` / `s(x) + fs(x, g)` was silently dropped:
        // the fit used the narrowed `X·Z` design while every predict rebuilt
        // the full-width design, making the model unpredictable (#978).
        match &termspec.basis {
            SmoothBasisSpec::FactorSumToZero { .. } | SmoothBasisSpec::FactorSmooth { .. } => {
                local_metadata[idx] = Some(term.metadata.clone());
                local_unabsorbed_z[idx] = z_opt.clone();
            }
            _ => {
                local_metadata[idx] = Some(with_identifiability_transform(
                    &term.metadata,
                    realized_transform.as_ref(),
                )?);
            }
        }
    }

    let total_p: usize = local_dims.iter().sum();
    let mut terms_out = Vec::<SmoothTerm>::with_capacity(smooth.terms.len());
    let mut penalties_global = Vec::<BlockwisePenalty>::new();
    let mut nullspace_dims_global = Vec::<usize>::new();
    let mut penaltyinfo_global = Vec::<PenaltyBlockInfo>::new();
    let mut dropped_penaltyinfo_global = Vec::<DroppedPenaltyBlockInfo>::new();
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(total_p, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraintsrows: Vec<Array1<f64>> = Vec::new();
    let mut linear_constraints_b: Vec<f64> = Vec::new();

    let mut col_start = 0usize;
    for idx in 0..smooth.terms.len() {
        let p_local = local_dims[idx];
        let col_end = col_start + p_local;

        for active_penalty in &local_active_penalties[idx] {
            let global_index = penalties_global.len();
            penalties_global.push(BlockwisePenalty::new(
                col_start..col_end,
                active_penalty.matrix.clone(),
            ));
            nullspace_dims_global.push(active_penalty.nullity);
            penaltyinfo_global.push(PenaltyBlockInfo {
                global_index,
                termname: Some(smooth.terms[idx].name.clone()),
                penalty: active_penalty.info.clone(),
            });
        }
        for info in &local_dropped_penalties[idx] {
            dropped_penaltyinfo_global.push(DroppedPenaltyBlockInfo {
                termname: Some(smooth.terms[idx].name.clone()),
                penalty: info.clone(),
            });
        }

        terms_out.push(SmoothTerm {
            name: smooth.terms[idx].name.clone(),
            coeff_range: col_start..col_end,
            shape: smooth.terms[idx].shape,
            active_penalties: local_active_penalties[idx].clone(),
            dropped_penalties: local_dropped_penalties[idx].clone(),
            metadata: local_metadata[idx]
                .clone()
                .expect("local metadata must exist for every smooth term"),
            lower_bounds_local: smooth.terms[idx].lower_bounds_local.clone(),
            linear_constraints_local: local_linear_constraints[idx].clone(),
            // Global orthogonality transforms break Kronecker structure.
            kronecker_factored: None,
            // The final raw-basis → coefficient chart, including any
            // stage-2 joint-null Q and global orthogonality Z, is embedded in
            // `metadata` above. Keeping Q separately here would apply it twice
            // on frozen rebuilds and would put derivative operators in a
            // different chart from the value path.
            joint_null_rotation: None,
            // Factor-smooth kinds export the chart their metadata could not
            // absorb; the freeze persists it onto the spec for replay (#978).
            unabsorbed_global_orthogonality: local_unabsorbed_z[idx].clone(),
            parametric_residualization: local_residualization[idx].clone(),
            // The gauge this collection decided, so a term-local rebuild at a
            // new basis parameter can put its result back into it (#2747).
            collection_gauge: local_collection_gauge[idx].clone(),
        });
        if let Some(lin_local) = &local_linear_constraints[idx] {
            for r in 0..lin_local.a.nrows() {
                let mut row = Array1::<f64>::zeros(total_p);
                row.slice_mut(s![col_start..col_end])
                    .assign(&lin_local.a.row(r));
                linear_constraintsrows.push(row);
                linear_constraints_b.push(lin_local.b[r]);
            }
        }
        if let Some(lb_local) = smooth.terms[idx].lower_bounds_local.as_ref()
            && lb_local.len() == p_local
        {
            coefficient_lower_bounds
                .slice_mut(s![col_start..col_end])
                .assign(lb_local);
            any_bounds = true;
        }

        col_start = col_end;
    }

    assert_eq!(
        penalties_global.len(),
        nullspace_dims_global.len(),
        "globally reparameterized smooth penalty/nullspace bookkeeping diverged"
    );
    assert_eq!(
        penalties_global.len(),
        penaltyinfo_global.len(),
        "globally reparameterized smooth penalty metadata bookkeeping diverged"
    );

    Ok((
        SmoothDesign {
            term_designs: local_designs
                .into_iter()
                .map(|design| design.expect("local design must exist for every smooth term"))
                .collect(),
            penalties: penalties_global,
            nullspace_dims: nullspace_dims_global,
            penaltyinfo: penaltyinfo_global,
            dropped_penaltyinfo: dropped_penaltyinfo_global,
            terms: terms_out,
            coefficient_lower_bounds: if any_bounds {
                Some(coefficient_lower_bounds)
            } else {
                None
            },
            linear_constraints: if linear_constraintsrows.is_empty() {
                None
            } else {
                let mut a = Array2::<f64>::zeros((linear_constraintsrows.len(), total_p));
                for (i, row) in linear_constraintsrows.iter().enumerate() {
                    a.row_mut(i).assign(row);
                }
                Some(LinearInequalityConstraints {
                    a,
                    b: Array1::from_vec(linear_constraints_b),
                })
            },
        },
        smooth.affine_offset,
    ))
}

/// If `termspec` is a single-level factor-by smooth (`s(x, by=fac)` expanded
/// into one `ByVariable { kind: Level }` block per factor level), return the
/// `(by_col, value_bits)` pair identifying which rows that level's block gates
/// to. `None` for numeric-by smooths and every other basis.
///
/// A factor-by smooth's per-level block is the inner basis multiplied by the
/// level indicator (zero on every other level's rows). Its column span
/// therefore contains the per-level CONSTANT — a vector that is `1` on this
/// level's rows and `0` elsewhere — which is exactly the column the
/// treatment-coded factor main effect (`build_termspec` auto-adds one as an
/// unpenalized random-effect term) already carries. Centering each level's
/// smooth against the *global* intercept (`build_parametric_constraint_block_for_term`'s
/// default) removes only its global mean, leaving that within-level constant
/// to collide with the factor main effect: a rank-1 collinearity that lets the
/// penalty/ridge split the per-group baseline level between the two blocks and
/// under-recover it (the per-group log-cumulative-hazard offset leaks out — the
/// #900 weibull-AFT-by-factor surface miscalibration). Centering against the
/// gated level indicator instead removes the within-level constant cleanly,
/// leaving the per-group level entirely to the factor main effect (mgcv's
/// by-factor convention), while the per-level slope/curvature deviation stays
/// in the smooth (we deliberately do NOT project the overlapping continuous
/// axis out of a by-level smooth — that deviation is the by-factor signal).
fn factor_by_level_gate(termspec: &SmoothTermSpec) -> Option<(usize, u64)> {
    match &termspec.basis {
        SmoothBasisSpec::ByVariable {
            by_col,
            by: ByVariableSpec::Level { value_bits, .. },
            ..
        } => Some((*by_col, *value_bits)),
        _ => None,
    }
}

fn build_parametric_constraint_block_for_term(
    data: ArrayView2<'_, f64>,
    linear_terms: &[LinearTermSpec],
    termspec: &SmoothTermSpec,
) -> Result<Array2<f64>, BasisError> {
    let n = data.nrows();
    let p_data = data.ncols();

    // Factor-by-level smooth: center against the gated level indicator so the
    // within-level constant is removed (it belongs to the treatment-coded
    // factor main effect), not against the global `[1 | overlapping axes]`.
    if let Some((by_col, value_bits)) = factor_by_level_gate(termspec) {
        if by_col >= p_data {
            gam_problem::bail_dim_basis!(
                "factor-by smooth term '{}' by column {by_col} out of bounds for {p_data} columns",
                termspec.name
            );
        }
        let mut c = Array2::<f64>::zeros((n, 1));
        let by = data.column(by_col);
        let value_bits = gam_data::canonical_level_bits(f64::from_bits(value_bits));
        for (row, &value) in by.iter().enumerate() {
            if gam_data::canonical_level_bits(value) == value_bits {
                c[[row, 0]] = 1.0;
            }
        }
        return Ok(c);
    }

    let feature_cols = smooth_term_feature_cols(termspec);
    let mut parametric_cols = smooth_intrinsic_parametric_feature_cols(linear_terms, termspec);
    for &feature_col in &parametric_cols {
        if feature_col >= p_data {
            gam_problem::bail_dim_basis!(
                "smooth term feature column {feature_col} out of bounds for {p_data} columns"
            );
        }
    }
    for linear in linear_terms
        .iter()
        .filter(|linear| feature_cols.contains(&linear.feature_col))
    {
        if linear.feature_col >= p_data {
            gam_problem::bail_dim_basis!(
                "linear term '{}' feature column {} out of bounds for {} columns",
                linear.name,
                linear.feature_col,
                p_data
            );
        }
        if !parametric_cols.contains(&linear.feature_col) {
            parametric_cols.push(linear.feature_col);
        }
    }

    let mut c = Array2::<f64>::zeros((n, 1 + parametric_cols.len()));
    c.column_mut(0).fill(1.0);
    for (j, &feature_col) in parametric_cols.iter().enumerate() {
        c.column_mut(j + 1).assign(&data.column(feature_col));
    }
    Ok(c)
}

pub fn apply_smooth_transform_to_design(
    design_local: DesignMatrix,
    transform: &Array2<f64>,
    termname: &str,
) -> Result<DesignMatrix, BasisError> {
    match design_local {
        DesignMatrix::Dense(inner) => {
            let op = CoefficientTransformOperator::new(inner, transform.clone()).map_err(|e| {
                BasisError::InvalidInput(format!(
                    "smooth identifiability transform failed for term '{termname}': {e}"
                ))
            })?;
            Ok(DesignMatrix::Dense(
                gam_linalg::matrix::DenseDesignMatrix::from(Arc::new(op)),
            ))
        }
        DesignMatrix::Sparse(inner) => {
            let dense = inner
                .try_to_dense_arc("smooth identifiability sparse transform")
                .map_err(BasisError::InvalidInput)?
                .as_ref()
                .dot(transform);
            Ok(DesignMatrix::Dense(
                gam_linalg::matrix::DenseDesignMatrix::from(dense),
            ))
        }
    }
}

fn design_constraint_cross(
    design: &DesignMatrix,
    constraint_matrix: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, BasisError> {
    let n = design.nrows();
    if constraint_matrix.nrows() != n {
        return Err(BasisError::ConstraintMatrixRowMismatch {
            basisrows: n,
            constraintrows: constraint_matrix.nrows(),
        });
    }
    let mut cross = Array2::<f64>::zeros((design.ncols(), constraint_matrix.ncols()));
    const CHUNK: usize = 1024;
    for start in (0..n).step_by(CHUNK) {
        let end = (start + CHUNK).min(n);
        let design_chunk = design
            .try_row_chunk(start..end)
            .map_err(|e| BasisError::InvalidInput(e.to_string()))?;
        let constraint_chunk = constraint_matrix.slice(s![start..end, ..]).to_owned();
        cross += &gam_linalg::faer_ndarray::fast_atb(&design_chunk, &constraint_chunk);
    }
    Ok(cross)
}

fn design_frobenius_norm(design: &DesignMatrix) -> Result<f64, BasisError> {
    let n = design.nrows();
    const CHUNK: usize = 1024;
    let mut sumsq = 0.0;
    for start in (0..n).step_by(CHUNK) {
        let end = (start + CHUNK).min(n);
        let chunk = design
            .try_row_chunk(start..end)
            .map_err(|e| BasisError::InvalidInput(e.to_string()))?;
        sumsq += chunk.iter().map(|v| v * v).sum::<f64>();
    }
    Ok(sumsq.sqrt())
}

/// The frozen row-space correction `R` for the span-preserving parametric
/// orthogonalization (#2747), when this term carries one.
///
/// Unlike [`frozen_global_orthogonality`] this lives on the TERM spec rather
/// than inside a basis kind, because every basis can take the residualizing arm
/// — the predicate is the geometry of the realized design against its constraint
/// block, not the basis family.
fn frozen_parametric_residualization(
    termspec: &SmoothTermSpec,
) -> Option<&ParametricResidualizationChart> {
    termspec.frozen_parametric_residualization.as_ref()
}

/// The persisted fit-time global-orthogonality chart for a factor-smooth
/// term, if one was frozen onto its spec (#978). `Some` means this term was
/// residualized against owner terms at fit time and prediction/refit rebuilds
/// must replay exactly that column map instead of rederiving anything from
/// the (new) rows.
fn frozen_global_orthogonality(termspec: &SmoothTermSpec) -> Option<&Array2<f64>> {
    match &termspec.basis {
        SmoothBasisSpec::FactorSumToZero {
            frozen_global_orthogonality,
            ..
        } => frozen_global_orthogonality.as_ref(),
        SmoothBasisSpec::FactorSmooth { spec } => spec.frozen_global_orthogonality.as_ref(),
        _ => None,
    }
}

/// Transport one smooth term's active penalties through the COLLECTION gauge.
///
/// Extracted from `apply_global_smooth_identifiability` (#2747) so that the
/// collection build and the outer search's incremental single-term realizer
/// apply the same congruence and the same double-penalty rebuild. Two callers
/// of one gauge is the point: a term realization spliced back into a collection
/// design has to be in that collection's gauge, and a second implementation of
/// "restrict a penalty through `Z`" is a second answer to one question.
///
/// `coefficient_gauge` is `None` exactly when no global transform was applied,
/// in which case the penalties are passed through with their declared
/// structural frames re-attached and nothing else moved.
fn penalty_candidates_under_collection_gauge(
    active_penalties: &[ActivePenalty],
    coefficient_gauge: Option<&gam_problem::Gauge>,
    term_name: &str,
) -> Result<Vec<PenaltyCandidate>, BasisError> {
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    let penalty_candidates = active_penalties
        .par_iter()
        .map(|penalty| -> Result<PenaltyCandidate, BasisError> {
            let raw = ConstructiveQuadratic::try_from_dense_psd(
                penalty.matrix.clone(),
                "global smooth source penalty",
            )?;
            // Re-attach the structural null frame the basis factory
            // declared (#2445): `try_from_dense_psd` sees only the dense
            // matrix, and the declaration must survive this chokepoint so
            // the double-penalty rebuild below decides topology from the
            // carried theorem, not from a rank test on a matrix carrying
            // the Duchon conditioning ridge. `.restricted` transports it
            // through the global gauge.
            let raw = match penalty.info.structural_null_frame.as_ref() {
                Some(frame) => raw.with_structural_null_frame(
                    frame.clone(),
                    "global smooth source penalty structural frame",
                )?,
                None => raw,
            };
            let restricted = if let Some(gauge) = coefficient_gauge {
                raw.restricted(gauge, "global smooth identifiability restriction")?
            } else {
                raw
            };
            let (_, c_new) = normalize_penalty_in_constrained_space(restricted.dense());
            let matrix = restricted.scaled(1.0 / c_new, "normalized global smooth penalty")?;
            Ok(PenaltyCandidate {
                matrix,
                source: penalty.info.source.clone(),
                normalization_scale: penalty.info.normalization_scale * c_new,
                kronecker_factors: None,
                op: None,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    // #1476-class fix (central, basis-agnostic): when a non-trivial GLOBAL
    // identifiability/orthogonalization transform `z_opt` was applied above,
    // it congruence-restricts EVERY penalty — including a Marra & Wood double-
    // penalty null-space shrinkage ridge (`DoublePenaltyNullspace`). A merely-
    // restricted ridge `Zᵀ (Z_null Z_nullᵀ) Z` is NOT the projector onto the
    // null space of the *constrained* bending penalty `Zᵀ S_bend Z`: the
    // sum-to-zero / parametric-orthogonalization `Z` is not norm-preserving and
    // typically DROPS the constant direction, so the restricted ridge is
    // neither idempotent nor aligned with `null(Zᵀ S_bend Z)` and shrinks
    // penalized directions (the #1266/#1476 flat-collapse / EDF mis-allocation
    // class). This is the single chokepoint every basis flows through, so
    // rebuild the ridge here from the null space of the constrained `Primary`
    // penalty, exactly as the 1-D B-spline / tensor / thin-plate paths do in
    // their own local builds. (Idempotent with those local rebuilds: when no
    // further `Primary`-null directions survive, the rebuilt ridge equals the
    // local one; when this global `Z` removes more, only this rebuild is
    // correct.) Scoped to `coefficient_gauge.is_some()`: with no global
    // transform the penalties are untouched and the basis-local ridge already
    // lives in the fit chart.
    let mut penalty_candidates = penalty_candidates;
    if coefficient_gauge.is_some()
        && penalty_candidates
            .iter()
            .any(|c| matches!(c.source, PenaltySource::DoublePenaltyNullspace))
    {
        // Nonzero-row support of a (symmetric) penalty matrix: the coefficient
        // range it actually penalizes. A per-level `by=factor` smooth emits one
        // `Primary`+`DoublePenaltyNullspace` pair PER LEVEL, each confined to
        // that level's disjoint `[off..off+p]` diagonal block (#1427), so a
        // ridge must be rebuilt from the Primary sharing ITS support — not the
        // first global Primary, and not the summed bending (which would collapse
        // the independent per-level λ). For a single smooth term there is one
        // Primary spanning the whole block and this reduces to the simple case.
        const SUPPORT_TOL: f64 = 0.0;
        let support_rows = |m: &Array2<f64>| -> (usize, usize) {
            let n = m.nrows();
            let mut lo = n;
            let mut hi = 0usize;
            for i in 0..n {
                let any = (0..m.ncols()).any(|j| m[[i, j]].abs() > SUPPORT_TOL);
                if any {
                    lo = lo.min(i);
                    hi = hi.max(i + 1);
                }
            }
            (lo, hi)
        };
        // Snapshot each Primary's support + a clone of its matrix (immutable
        // borrow released before we mutate the ridges below).
        let primaries: Vec<((usize, usize), ConstructiveQuadratic)> = penalty_candidates
            .iter()
            .filter(|c| matches!(c.source, PenaltySource::Primary))
            .map(|c| -> Result<_, BasisError> {
                Ok((
                    support_rows(&c.matrix),
                    c.matrix
                        .scaled(c.normalization_scale, "physical global smooth primary")?,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for candidate in &mut penalty_candidates {
            if !matches!(candidate.source, PenaltySource::DoublePenaltyNullspace) {
                continue;
            }
            let q = candidate.matrix.nrows();
            let (rlo, rhi) = support_rows(&candidate.matrix);
            // The Primary whose support CONTAINS this ridge's support (the
            // co-located bending block). Falls back to the unique Primary when
            // the ridge is (numerically) empty.
            let owner = primaries
                .iter()
                .find(|((plo, phi), _)| *plo <= rlo && rhi <= *phi)
                .or_else(|| (primaries.len() == 1).then(|| &primaries[0]))
                .ok_or_else(|| {
                    BasisError::InvalidInput(format!(
                        "double-penalty ridge for smooth '{}' has no co-located primary penalty",
                        term_name
                    ))
                })?;
            let ((plo, phi), s_full) = owner;
            // Rebuild from the physical Primary and ridge submatrices. Rank
            // revelation on the Primary's retained energy factor preserves
            // the structural null space through this global chart, while the
            // restricted ridge supplies the function-metric action. No signed
            // spectrum of a rounded dense congruence is classified (#2318).
            let block = ConstructiveQuadratic::from_energy_factor(
                s_full.factor().slice(s![.., *plo..*phi]).to_owned(),
                "owned global smooth primary block",
            )?;
            // The support-block extraction rebuilds the quadratic from a
            // sliced factor, so re-attach the declared structural frame
            // restricted to the same block (it is `None` when the frame
            // has support outside the block, and the rebuild then falls
            // back to measuring — never guesses).
            let block = match s_full.structural_null_frame_block(*plo, *phi) {
                Some(frame) => block.with_structural_null_frame(
                    frame,
                    "owned global smooth primary block structural frame",
                )?,
                None => block,
            };
            let ridge_full = candidate.matrix.scaled(
                candidate.normalization_scale,
                "physical global smooth null ridge",
            )?;
            let ridge_block = ConstructiveQuadratic::from_energy_factor(
                ridge_full.factor().slice(s![.., *plo..*phi]).to_owned(),
                "owned global smooth null-ridge block",
            )?;
            let rebuilt_block =
                crate::basis::rebuild_metric_consistent_ridge(&block, &ridge_block)?;
            match rebuilt_block {
                Some(ridge_block) => {
                    let mut full_factor =
                        Array2::<f64>::zeros((ridge_block.factor().nrows(), q));
                    full_factor
                        .slice_mut(s![.., *plo..*phi])
                        .assign(ridge_block.factor());
                    let full = ConstructiveQuadratic::from_energy_factor(
                        full_factor,
                        "embedded global smooth null ridge",
                    )?;
                    let (_, scale) = normalize_penalty_in_constrained_space(full.dense());
                    candidate.matrix = full
                        .scaled(1.0 / scale, "normalized embedded global smooth null ridge")?;
                    candidate.normalization_scale = scale;
                    candidate.kronecker_factors = None;
                    candidate.op = None;
                }
                // Constrained bending block is full rank: no null space to
                // shrink. Zero the ridge; the filter drops it.
                None => {
                    candidate.matrix = ConstructiveQuadratic::zero(q);
                    candidate.normalization_scale = 1.0;
                    candidate.kronecker_factors = None;
                    candidate.op = None;
                }
            }
        }
    }
    Ok(penalty_candidates)
}

fn maybe_smooth_identifiability_transform(
    termspec: &SmoothTermSpec,
    design_local: &DesignMatrix,
    constraint_block: Option<ArrayView2<'_, f64>>,
) -> Result<Option<Array2<f64>>, BasisError> {
    if let Some(SpatialIdentifiability::FrozenTransform { transform }) =
        spatial_identifiability_policy(termspec)
    {
        if design_local.ncols() != transform.nrows() {
            gam_problem::bail_dim_basis!(
                "frozen spatial identifiability transform mismatch: design has {} columns but transform has {} rows",
                design_local.ncols(),
                transform.nrows()
            );
        }
        return Ok(Some(transform.clone()));
    }

    if let Some(c) = constraint_block {
        if c.ncols() == 0 {
            Ok(None)
        } else {
            Ok(Some(orthogonality_transform_for_design(
                design_local,
                c,
                None, // fixed subspace: do not use iteration-varying PIRLS weights
            )?))
        }
    } else {
        Ok(None)
    }
}

/// Whether this smooth's *realized* design (the basis evaluated at the n data
/// rows) must be residualized against the model's parametric block (intercept +
/// any overlapping linear columns) by `apply_global_smooth_identifiability`.
///
/// This is the universal identifiability invariant for **kernel / radial**
/// spatial smooths (#531): without this step the smooth and the parametric
/// intercept fight over the same direction. The collision is invisible to the
/// kernels' *own* identifiability constraints because those act in **coefficient
/// space at the K centers**, not on the realized design rows:
///   - Matérn `CenterSumToZero` enforces `1ᵀα = 0` over the centers, so
///     `Kα` evaluated at the data rows still carries a near-constant direction.
///   - Duchon / TPS `OrthogonalToParametric` *defers* its centering to this very
///     step, which is why it is listed here too.
///
/// # This list says which smooths must be ORTHOGONALIZED, not which ones may be
/// # constrained for free
///
/// The text here used to justify the whole class with *"their realized column
/// span contains the constant … a structural rank-1 collision"*, and that
/// sentence is measured false for half of it
/// (the #2747 containment probe: `‖1 − P_X 1‖/‖1‖` on the
/// realized design against the `√ε` bar the deletion is licensed at):
///
/// ```text
///     thinplate                                      9.90e-15   contained
///     duchon                                         1.33e-14   contained
///     matern (both policies, ν = 3/2 and 5/2)   7.8e-4 .. 8.4e-1   NOT
///     curv (κ ∈ {−1,0,+1}, ℓ = 0.2 … 100)       5.1e-2 .. 9.5e-1   NOT
/// ```
///
/// The polynomial-nullspace bases really do contain the constant; the kernel
/// half does not, and for Matérn the residual falls monotonically toward the bar
/// as the range grows (`8.4e-1 → 7.8e-4` over `ℓ = 0.2 → 10`) — with the range an
/// ESTIMATED coordinate, so containment is a function of a fitted parameter and
/// not a property of the family.
///
/// That does not shorten this list. Returning `true` here asks for the smooth to
/// be made ORTHOGONAL to the parametric block, which is licensed for every
/// member; whether that is done by deleting a coefficient direction (free only
/// under containment) or by projecting in row space (always) is decided per
/// build by `apply_global_smooth_identifiability`, on the measured geometry
/// rather than on a claim about the family (#2747).
///
/// Tensor-product and B-spline bases instead apply a realized-design sum-to-zero
/// at basis-build time (`apply_sum_to_zero_constraint`), so they already satisfy
/// the invariant and must NOT be double-constrained — they return `false`.
///
/// The remaining bases are excluded, each for a concrete reason:
///   - **Sphere, Harmonic method**: the real-spherical-harmonic basis starts at
///     degree `l = 1` (`build_spherical_harmonic_basis`), so it never spans the
///     degree-0 constant — no centering is needed.
///   - **Sphere, Wahba method**: INCLUDED (#532). Its raw finite-center kernel
///     chart can span a near-constant realized direction even though the
///     continuous kernel omits the l=0 mode — same collision class as Matérn
///     `CenterSumToZero`. The composed parametric transform is frozen
///     onto `SphericalSplineBasisSpec::identifiability`
///     (`SphericalSplineIdentifiability::FrozenTransform`) and replayed by
///     `build_spherical_spline_basis` at predict time, so the orthogonalization
///     survives save → reload exactly as it does for Matérn.
///   - **PCA**: its `with_identifiability_transform` arm rejects a post-hoc
///     transform (the constraint lives inside the orthonormal loadings), and its
///     constant content is governed by the `centered` flag, not a residualizable
///     design.
///
/// `FrozenTransform` bases are excluded: a transform frozen by *this* pipeline
/// already has the parametric orthogonalization composed in (see
/// `with_identifiability_transform`), and they are gated out upstream by
/// `skip_global_transform` regardless.
fn smooth_requires_parametric_orthogonality(termspec: &SmoothTermSpec) -> bool {
    match &termspec.basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            smooth_requires_parametric_orthogonality(&SmoothTermSpec {
                frozen_parametric_residualization: None,
                name: termspec.name.clone(),
                basis: (**inner).clone(),
                shape: termspec.shape,
                joint_null_rotation: None,
            })
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => {
            smooth_requires_parametric_orthogonality(&SmoothTermSpec {
                frozen_parametric_residualization: None,
                name: termspec.name.clone(),
                basis: (**smooth).clone(),
                shape: termspec.shape,
                joint_null_rotation: None,
            })
        }
        SmoothBasisSpec::ThinPlate { spec, .. } => {
            matches!(
                spec.identifiability,
                SpatialIdentifiability::OrthogonalToParametric
            )
        }
        SmoothBasisSpec::Duchon { spec, .. } => {
            matches!(
                spec.identifiability,
                SpatialIdentifiability::OrthogonalToParametric
            )
        }
        SmoothBasisSpec::Matern { spec, .. } => matches!(
            spec.identifiability,
            MaternIdentifiability::CenterSumToZero | MaternIdentifiability::CenterLinearOrthogonal
        ),
        // Wahba sphere (`bs="sos"`, method=Wahba): the finite-center Sobolev
        // kernel chart can still span a near-constant realized direction on
        // the data rows, so it requires global parametric orthogonalization
        // (#532). Pseudo is resolved by the basis builder to the harmonic
        // engine, whose degree l=1 start never spans the constant; forcing it
        // through this post-build transform would also renormalize away the
        // harmonic engine's physical spectral penalty scale.
        SmoothBasisSpec::Sphere { spec, .. } => {
            matches!(spec.method, crate::basis::SphereMethod::Wahba)
                && !matches!(spec.wahba_kernel, crate::basis::SphereWahbaKernel::Pseudo)
                && matches!(
                    spec.identifiability,
                    SphericalSplineIdentifiability::CenterSumToZero
                )
        }
        // Constant-curvature geodesic kernel: same #531 collision class as the
        // raw finite-center Wahba sphere. Its coefficient-space sum-to-zero `z`
        // leaves the realized `K·z` design carrying a near-constant direction on
        // the data rows, so the global parametric orthogonalization must compose
        // onto `z` (#532). It does NOT span the constant — measured at
        // `5.1e-2 … 9.5e-1` across κ and the range — which is why that
        // orthogonalization is a projection here and not a deletion (#2747).
        SmoothBasisSpec::ConstantCurvature { spec, .. } => matches!(
            spec.identifiability,
            ConstantCurvatureIdentifiability::CenterSumToZero
        ),
        // Measure-jet representer: identical #531 collision class to the raw
        // finite-center Wahba sphere. Gaussian RBF columns times the
        // center-space sum-to-zero `z` still carry a near-constant direction on
        // the data rows, so `z` must absorb the parametric orthogonalization
        // (#532).
        SmoothBasisSpec::MeasureJet { spec, .. } => matches!(
            spec.identifiability,
            MeasureJetIdentifiability::CenterSumToZero
        ),
        SmoothBasisSpec::BSpline1D { .. }
        | SmoothBasisSpec::TensorBSpline { .. }
        | SmoothBasisSpec::Pca { .. }
        | SmoothBasisSpec::FactorSmooth { .. } => false,
    }
}

fn compose_identifiability_transforms(
    existing: Option<&Array2<f64>>,
    extra: Option<&Array2<f64>>,
) -> Result<Option<Array2<f64>>, BasisError> {
    match (existing, extra) {
        (Some(lhs), Some(rhs)) => {
            if lhs.ncols() == rhs.nrows() {
                Ok(Some(lhs.dot(rhs)))
            } else if lhs.nrows() == rhs.nrows() && lhs.ncols() == rhs.ncols() {
                // Rebuilding from an already-frozen spec can surface the same
                // raw->frozen transform twice. Treat that as idempotent
                // metadata, not a sequential Z_left * Z_right composition.
                Ok(Some(rhs.clone()))
            } else {
                Err(BasisError::DimensionMismatch(format!(
                    "identifiability transform mismatch: existing is {}x{}, extra is {}x{}",
                    lhs.nrows(),
                    lhs.ncols(),
                    rhs.nrows(),
                    rhs.ncols(),
                )))
            }
        }
        (Some(lhs), None) => Ok(Some(lhs.clone())),
        (None, Some(rhs)) => Ok(Some(rhs.clone())),
        (None, None) => Ok(None),
    }
}

fn with_identifiability_transform(
    metadata: &BasisMetadata,
    transform: Option<&Array2<f64>>,
) -> Result<BasisMetadata, BasisError> {
    match metadata {
        BasisMetadata::BSpline1D {
            knots,
            identifiability_transform,
            periodic,
            degree,
            auto_shrink_note,
            anchor_offset_coeffs,
        } => Ok(BasisMetadata::BSpline1D {
            knots: knots.clone(),
            periodic: *periodic,
            identifiability_transform: compose_identifiability_transforms(
                identifiability_transform.as_ref(),
                transform,
            )?,
            degree: *degree,
            auto_shrink_note: auto_shrink_note.clone(),
            // The offset coefficients live in the raw-basis chart and are
            // unaffected by an added constrained-chart identifiability
            // transform; carry them through unchanged (#2297).
            anchor_offset_coeffs: anchor_offset_coeffs.clone(),
        }),
        BasisMetadata::CubicRegression1D {
            knots,
            identifiability_transform,
        } => Ok(BasisMetadata::CubicRegression1D {
            knots: knots.clone(),
            identifiability_transform: compose_identifiability_transforms(
                identifiability_transform.as_ref(),
                transform,
            )?,
        }),
        BasisMetadata::ThinPlate {
            centers,
            length_scale,
            periodic,
            identifiability_transform,
            input_scale,
            radial_reparam,
        } => Ok(BasisMetadata::ThinPlate {
            centers: centers.clone(),
            length_scale: *length_scale,
            periodic: periodic.clone(),
            identifiability_transform: compose_identifiability_transforms(
                identifiability_transform.as_ref(),
                transform,
            )?,
            input_scale: *input_scale,
            radial_reparam: radial_reparam.clone(),
        }),
        BasisMetadata::Sphere {
            centers,
            penalty_order,
            method,
            max_degree,
            wahba_kernel,
            constraint_transform,
        } => Ok(BasisMetadata::Sphere {
            centers: centers.clone(),
            penalty_order: *penalty_order,
            method: *method,
            max_degree: *max_degree,
            wahba_kernel: *wahba_kernel,
            constraint_transform: compose_identifiability_transforms(
                constraint_transform.as_ref(),
                transform,
            )?,
        }),
        BasisMetadata::ConstantCurvature {
            centers,
            kappa,
            length_scale,
            constraint_transform,
        } => Ok(BasisMetadata::ConstantCurvature {
            centers: centers.clone(),
            kappa: *kappa,
            length_scale: *length_scale,
            constraint_transform: compose_identifiability_transforms(
                constraint_transform.as_ref(),
                transform,
            )?,
        }),
        BasisMetadata::MeasureJet {
            centers,
            input_scale,
            length_scale,
            eps_band,
            order_s,
            alpha,
            tau0,
            masses,
            support_means,
            penalty_normalization_scales,
            raw_penalty_normalization_scales,
            fused_penalty_normalization_scale,
            constraint_transform,
            sigma_coord,
        } => Ok(BasisMetadata::MeasureJet {
            centers: centers.clone(),
            input_scale: *input_scale,
            length_scale: *length_scale,
            eps_band: eps_band.clone(),
            order_s: *order_s,
            alpha: *alpha,
            tau0: *tau0,
            masses: masses.clone(),
            support_means: support_means.clone(),
            penalty_normalization_scales: penalty_normalization_scales.clone(),
            raw_penalty_normalization_scales: raw_penalty_normalization_scales.clone(),
            fused_penalty_normalization_scale: *fused_penalty_normalization_scale,
            constraint_transform: compose_identifiability_transforms(
                constraint_transform.as_ref(),
                transform,
            )?,
            sigma_coord: *sigma_coord,
        }),
        BasisMetadata::Matern {
            centers,
            length_scale,
            periodic,
            nu,
            include_intercept,
            identifiability_transform,
            input_scale,
            aniso_log_scales,
        } => Ok(BasisMetadata::Matern {
            centers: centers.clone(),
            length_scale: *length_scale,
            periodic: periodic.clone(),
            nu: *nu,
            include_intercept: *include_intercept,
            identifiability_transform: compose_identifiability_transforms(
                identifiability_transform.as_ref(),
                transform,
            )?,
            input_scale: *input_scale,
            aniso_log_scales: aniso_log_scales.clone(),
        }),
        BasisMetadata::Duchon {
            centers,
            length_scale,
            periodic,
            power,
            nullspace_order,
            identifiability_transform,
            input_scale,
            aniso_log_scales,
            operator_collocation_points,
            radial_reparam,
            spectral_basis,
        } => Ok(BasisMetadata::Duchon {
            centers: centers.clone(),
            length_scale: *length_scale,
            periodic: periodic.clone(),
            power: *power,
            nullspace_order: *nullspace_order,
            input_scale: *input_scale,
            aniso_log_scales: aniso_log_scales.clone(),
            operator_collocation_points: operator_collocation_points.clone(),
            radial_reparam: radial_reparam.clone(),
            spectral_basis: spectral_basis.clone(),
            identifiability_transform: compose_identifiability_transforms(
                identifiability_transform.as_ref(),
                transform,
            )?,
        }),
        BasisMetadata::SphereHarmonics {
            max_degree,
            radians,
        } => Ok(BasisMetadata::SphereHarmonics {
            max_degree: *max_degree,
            radians: *radians,
        }),
        BasisMetadata::TensorBSpline {
            feature_cols,
            knots,
            degrees,
            periods,
            is_cr,
            identifiability_transform,
        } => Ok(BasisMetadata::TensorBSpline {
            feature_cols: feature_cols.clone(),
            knots: knots.clone(),
            degrees: degrees.clone(),
            periods: periods.clone(),
            is_cr: is_cr.clone(),
            identifiability_transform: compose_identifiability_transforms(
                identifiability_transform.as_ref(),
                transform,
            )?,
        }),
        BasisMetadata::BySmooth {
            inner,
            by_col,
            levels,
            ordered,
        } => Ok(BasisMetadata::BySmooth {
            inner: Box::new(with_identifiability_transform(inner, transform)?),
            by_col: *by_col,
            levels: levels.clone(),
            ordered: *ordered,
        }),
        BasisMetadata::FactorSmooth {
            continuous_cols,
            group_col,
            knots,
            degree,
            periodic,
            group_levels,
            flavour,
            marginal_is_cr,
        } => {
            // Factor-smooth metadata has no transform slot; the global pass
            // exports its transform via `SmoothTerm::unabsorbed_global_orthogonality`
            // instead (#978). Silently dropping a transform here is what made
            // `s(x) + fs(x, g)` unpredictable — reject loudly so any future
            // caller that reaches this arm with a transform fails at fit time
            // rather than corrupting the saved coefficient chart.
            if transform.is_some() {
                gam_problem::bail_invalid_basis!(
                    "FactorSmooth metadata cannot absorb an identifiability transform; \
                     route it through the term-level frozen_global_orthogonality carrier"
                );
            }
            Ok(BasisMetadata::FactorSmooth {
                continuous_cols: continuous_cols.clone(),
                group_col: *group_col,
                knots: knots.clone(),
                degree: *degree,
                periodic: *periodic,
                group_levels: group_levels.clone(),
                flavour: flavour.clone(),
                marginal_is_cr: *marginal_is_cr,
            })
        }
        BasisMetadata::Pca {
            feature_cols,
            basis_matrix,
            centered,
            smooth_penalty,
            center_mean,
            pca_basis_path,
            chunk_size,
        } => {
            // PCA bases carry an orthonormal projection matrix and do not
            // expose an identifiability transform that can be re-composed
            // (the constraint, if any, lives inside the PCA loadings
            // themselves), so the caller cannot meaningfully attach a
            // post-hoc Z transform here.
            if transform.is_some() {
                gam_problem::bail_invalid_basis!(
                    "PCA bases do not expose a composable identifiability transform"
                );
            }
            Ok(BasisMetadata::Pca {
                feature_cols: feature_cols.clone(),
                basis_matrix: basis_matrix.clone(),
                centered: *centered,
                smooth_penalty: *smooth_penalty,
                center_mean: center_mean.clone(),
                pca_basis_path: pca_basis_path.clone(),
                chunk_size: *chunk_size,
            })
        }
    }
}

// `pub` so the #1601-orphaned design-assembly constraint regression guards
// (re-homed into gam-models) can assert the realized constraint orthogonality
// residual directly against this exact production helper rather than a copy.
pub fn orthogonality_relative_residual_for_design(
    design: &DesignMatrix,
    constraint_matrix: ArrayView2<'_, f64>,
) -> Result<f64, BasisError> {
    let cross = design_constraint_cross(design, constraint_matrix)?;
    let num = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
    let b_norm = design_frobenius_norm(design)?;
    let c_norm = constraint_matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
    let denom = b_norm * c_norm;
    // An empty design or an empty constraint has no overlap to measure.
    if denom == 0.0 {
        return Ok(0.0);
    }
    Ok(num / denom)
}

#[cfg(test)]
mod frozen_linear_term_mass_rebuild_tests {
    use super::*;

    /// One `double_penalty=true` linear term named `x`, no smooth/random-effect
    /// terms — the minimal spec that exercises `linear_function_mass` without
    /// dragging in basis construction.
    fn one_linear_term_spec() -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "x".to_string(),
                feature_col: 0,
                feature_cols: vec![0],
                categorical_levels: vec![],
                double_penalty: true,
                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                coefficient_min: None,
                coefficient_max: None,
                frozen_function_mass: None,
            }],
            random_effect_terms: Vec::new(),
            smooth_terms: Vec::new(),
        }
    }

    fn training_data_varying_x(n: usize) -> Array2<f64> {
        let mut data = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            // Genuinely varying, well away from zero for every row.
            data[[i, 0]] = 1.0 + i as f64;
        }
        data
    }

    fn constant_zero_x(n_rows: usize) -> Array2<f64> {
        Array2::<f64>::zeros((n_rows, 1))
    }

    /// Sanity check that the guard this fix must NOT weaken is still live: an
    /// UNFROZEN spec built directly over a genuinely all-zero column (the
    /// fit-time case — `data` really is what will be fit on) still reports the
    /// "identically zero" identifiability failure instead of silently fitting
    /// an unrecoverable term.
    #[test]
    fn unfrozen_spec_still_rejects_a_genuinely_zero_training_column() {
        let spec = one_linear_term_spec();
        let degenerate_training_data = constant_zero_x(20);
        let err = build_term_collection_design(degenerate_training_data.view(), &spec)
            .expect_err("an unfrozen spec fit directly on an all-zero column must still fail");
        let message = err.to_string();
        assert!(
            message.contains("identically zero"),
            "expected the identifiability guard's message, got: {message}"
        );
    }

    /// The regression this fix targets (#1561 rebuild-design triage, shortlist
    /// item 1): fit on a TRAINING set where `x` genuinely varies, freeze the
    /// spec, then rebuild the design at a small EVALUATION set where `x`
    /// happens to be constant (e.g. an anchor grid that holds a covariate
    /// fixed to isolate another term's effect — the exact pattern in
    /// `quality_vs_mass_ordinal_polr` and
    /// `quality_vs_inla_survival_random_intercept_baseline`). The rebuild must
    /// succeed and must reuse the TRAINING-time mass rather than recomputing
    /// (which would be a bogus "identically zero" recomputed from the
    /// constant evaluation rows).
    #[test]
    fn frozen_spec_rebuilds_at_a_constant_evaluation_column_using_the_training_mass() {
        let spec = one_linear_term_spec();
        let training_data = training_data_varying_x(40);

        let training_design = build_term_collection_design(training_data.view(), &spec)
            .expect("fit-time build over a genuinely varying column must succeed");
        let training_mass = training_design
            .linear_function_masses
            .first()
            .copied()
            .flatten()
            .expect("a double_penalty=true term must report its fit-time function mass");
        assert!(
            training_mass > 0.0,
            "training mass for a genuinely varying column must be strictly positive, got {training_mass}"
        );

        let frozen_spec = freeze_term_collection_from_design(&spec, &training_design)
            .expect("freezing the spec against its own fit-time design must succeed");
        assert_eq!(
            frozen_spec.linear_terms[0].frozen_function_mass,
            Some(training_mass),
            "freezing must persist the exact fit-time mass onto the term"
        );

        // The rebuild-time evaluation grid: `x` is constant (zero) across
        // every one of these rows, exactly like an anchor/group-anchor probe
        // that fixes a covariate to isolate another effect.
        let evaluation_grid = constant_zero_x(3);
        let rebuilt_design = build_term_collection_design(evaluation_grid.view(), &frozen_spec)
            .expect(
                "rebuilding a FROZEN spec's design at a constant-covariate evaluation grid must \
                 succeed — the training-time mass is reused, never recomputed from these rows",
            );
        assert_eq!(
            rebuilt_design
                .linear_function_masses
                .first()
                .copied()
                .flatten(),
            Some(training_mass),
            "the rebuilt design must carry the REUSED training-time mass, not a value \
             recomputed from the (all-zero) evaluation rows"
        );
    }
}
