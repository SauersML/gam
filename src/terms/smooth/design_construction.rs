fn build_term_collection_design_inner(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
) -> Result<TermCollectionDesign, BasisError> {
    use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

    let n = data.nrows();
    let p_intercept = usize::from(!term_collection_has_one_sided_anchored_bspline(spec));
    let p_lin = spec.linear_terms.len();

    // Smooth construction, random-effect construction, and linear-column
    // extraction are independent at this stage. Run them concurrently, but keep
    // each result in spec order so the final global layout remains stable:
    // [intercept | linear | random_effects | smooth].
    let (smooth_raw_result, (random_blocks_result, linear_block_result)) = rayon::join(
        || {
            let mut ws = crate::basis::BasisWorkspace::new();
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

    let smooth = apply_global_smooth_identifiability(
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

    // Block 0: intercept — zero storage. A one-sided anchored B-spline consumes
    // the absolute level at that endpoint, so a free intercept would violate
    // the structural anchor.
    if p_intercept == 1 {
        blocks.push(DesignBlock::Intercept(n));
    }

    // Block 1: linear terms.
    if let Some(lin_block) = linear_block {
        blocks.push(DesignBlock::Dense(crate::matrix::DenseDesignMatrix::from(
            lin_block,
        )));
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

    // Linear terms with `double_penalty=true` are aggregated into a single
    // shared ridge block named "linear" that spans the union of their
    // coefficient columns. Two reasons this is the only correct shape:
    //
    //   (i) Per-term redundancy: a penalized linear term is a single scalar
    //   coefficient β_j; the only shrinkage direction is β_j² itself. The
    //   earlier per-term form emitted two identical 1×1 ridges per term
    //   (`LinearTermRidge` + `LinearTermDoubleRidge`); for the two λ's that
    //   the outer REML/LAML places on those blocks, the total penalty is
    //   `(λ₁ + λ₂) β_j²` and the outer score is flat along
    //   `λ₁ + λ₂ = const`. Unidentifiable hyperparameters by construction.
    //
    //   (ii) Cross-term aliasing: each penalized linear term in its own 1×1
    //   block gives the outer optimizer a separate λ per term, when the
    //   mgcv-equivalent "ridge over the parametric linear columns" is one
    //   shared λ that shrinks every penalized linear coefficient with the
    //   same scale. The shared block emits exactly one outer hyperparameter
    //   and pins each penalized column's diagonal entry to 1.0; unpenalized
    //   linear columns sit at diagonal 0, so they are mathematically
    //   unaffected (S β = 0 on those rows) while keeping the block range
    //   contiguous over the linear coefficient run.
    //
    // The block's `termname` is the literal "linear" so keyed hyperpriors
    // / coefficient groups address the shared lever directly. The
    // per-term identity is preserved through `linear_ranges`, which still
    // names each individual column.
    let any_double_penalty_linear = spec.linear_terms.iter().any(|t| t.double_penalty);
    if any_double_penalty_linear && p_lin > 0 {
        let block_range = p_intercept..(p_intercept + p_lin);
        let mut s = Array2::<f64>::zeros((p_lin, p_lin));
        let mut effective_rank = 0usize;
        for (j, linear) in spec.linear_terms.iter().enumerate() {
            if linear.double_penalty {
                s[[j, j]] = 1.0;
                effective_rank += 1;
            }
        }
        let global_index = penalties.len();
        penalties.push(BlockwisePenalty::new(block_range, s));
        nullspace_dims.push(0);
        penaltyinfo.push(PenaltyBlockInfo {
            global_index,
            termname: Some("linear".to_string()),
            penalty: PenaltyInfo {
                source: PenaltySource::Other("LinearTermRidge".to_string()),
                original_index: 0,
                active: true,
                effective_rank,
                dropped_reason: None,
                nullspace_dim_hint: 0,
                normalization_scale: 1.0,
                kronecker_factors: None,
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
            penalty: PenaltyInfo {
                source: PenaltySource::Other(format!("RandomEffectRidge({name})")),
                original_index: re_idx,
                active: true,
                effective_rank: block_size,
                dropped_reason: None,
                nullspace_dim_hint: 0,
                normalization_scale: 1.0,
                kronecker_factors: None,
            },
        });
    }

    if smooth.penaltyinfo.len() != smooth.penalties.len() {
        crate::bail_invalid_basis!(
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
        let mut penalty = localinfo.penalty.clone();
        penalty.nullspace_dim_hint = ns;
        penaltyinfo.push(PenaltyBlockInfo {
            global_index,
            termname: localinfo.termname.clone(),
            penalty,
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
        merge_linear_constraints_global(explicit_linear_constraints, lower_bound_constraints);

    Ok(TermCollectionDesign {
        design,
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
        random_effect_ranges,
        random_effect_levels,
        smooth,
    })
}

fn term_collection_has_one_sided_anchored_bspline(spec: &TermCollectionSpec) -> bool {
    spec.smooth_terms
        .iter()
        .any(|term| smooth_basis_has_one_sided_anchored_bspline(&term.basis))
}

fn smooth_basis_has_one_sided_anchored_bspline(basis: &SmoothBasisSpec) -> bool {
    match basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            smooth_basis_has_one_sided_anchored_bspline(inner)
        }
        SmoothBasisSpec::BSpline1D { spec, .. } => {
            bspline_conditions_have_one_sided_anchor(&spec.boundary_conditions)
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => {
            smooth_basis_has_one_sided_anchored_bspline(smooth)
        }
        SmoothBasisSpec::TensorBSpline { spec, .. } => spec
            .marginalspecs
            .iter()
            .any(|marginal| bspline_conditions_have_one_sided_anchor(&marginal.boundary_conditions)),
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

fn bspline_conditions_have_one_sided_anchor(
    conditions: &crate::basis::BSplineBoundaryConditions,
) -> bool {
    let left_anchored = matches!(
        conditions.left,
        crate::basis::BSplineEndpointBoundaryCondition::Anchored { .. }
    );
    let right_anchored = matches!(
        conditions.right,
        crate::basis::BSplineEndpointBoundaryCondition::Anchored { .. }
    );
    left_anchored != right_anchored
}

pub fn build_term_collection_design(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
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
    build_term_collection_design_inner(data, &planned_spec)
}

pub fn build_term_collection_designs_joint(
    data: ArrayView2<'_, f64>,
    specs: &[TermCollectionSpec],
) -> Result<Vec<TermCollectionDesign>, BasisError> {
    for spec in specs {
        validate_term_collection_finite_inputs(data, spec)?;
    }
    let smooth_blocks = specs
        .iter()
        .map(|spec| spec.smooth_terms.clone())
        .collect::<Vec<_>>();
    let planned_blocks = plan_joint_spatial_centers_for_term_blocks(data, &smooth_blocks)?;
    let mut out = Vec::with_capacity(specs.len());
    for (spec, planned_terms) in specs.iter().zip(planned_blocks.into_iter()) {
        let mut planned_spec = spec.clone();
        planned_spec.smooth_terms = planned_terms;
        out.push(build_term_collection_design_inner(data, &planned_spec)?);
    }
    Ok(out)
}

pub fn build_term_collection_designs_and_freeze_joint(
    data: ArrayView2<'_, f64>,
    specs: &[TermCollectionSpec],
) -> Result<(Vec<TermCollectionDesign>, Vec<TermCollectionSpec>), EstimationError> {
    let designs = build_term_collection_designs_joint(data, specs)?;
    let mut resolved_specs = Vec::with_capacity(specs.len());
    for (spec, design) in specs.iter().zip(designs.iter()) {
        resolved_specs.push(freeze_term_collection_from_design(spec, design)?);
    }
    Ok((designs, resolved_specs))
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
        cross += &crate::faer_ndarray::fast_atb(&lhs_chunk, &rhs_chunk);
        lhs_sumsq += lhs_chunk.iter().map(|v| v * v).sum::<f64>();
        rhs_sumsq += rhs_chunk.iter().map(|v| v * v).sum::<f64>();
    }
    let num = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
    let denom = (lhs_sumsq.sqrt() * rhs_sumsq.sqrt()).max(1e-300);
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

fn smooth_intrinsic_parametric_feature_cols(
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
) -> Result<SmoothDesign, BasisError> {
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
        crate::bail_dim_basis!(
            "smooth spec count ({}) does not match built term count ({})",
            smoothspecs.len(),
            smooth.terms.len()
        );
    }

    if smooth.terms.is_empty() {
        return Ok(smooth.into());
    }

    let mut local_designs = vec![None; smooth.terms.len()];
    let mut local_penalties = vec![Vec::<Array2<f64>>::new(); smooth.terms.len()];
    let mut local_nullspaces = vec![Vec::<usize>::new(); smooth.terms.len()];
    let mut local_penaltyinfo = vec![Vec::<PenaltyInfo>::new(); smooth.terms.len()];
    let mut local_metadata = vec![None; smooth.terms.len()];
    let mut local_dims = vec![0usize; smooth.terms.len()];
    let mut local_linear_constraints = vec![None; smooth.terms.len()];
    let mut local_unabsorbed_z = vec![None::<Array2<f64>>; smooth.terms.len()];

    let SmoothStructureAnalysis {
        ownership_order,
        term_owners,
        ..
    } = analyze_smooth_ownership(smoothspecs);

    use rayon::iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    };

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
        let needs_parametric_block = replay_z.is_none()
            && !skip_global_transform
            && (smooth_has_overlapping_linear_terms(linear_terms, termspec)
                || !smooth_intrinsic_parametric_feature_cols(linear_terms, termspec).is_empty()
                || smooth_requires_parametric_orthogonality(termspec)
                // A factor-by-level smooth must always be centered against its
                // gated level indicator (see `factor_by_level_gate`) so its
                // within-level constant cannot collide with the treatment-coded
                // factor main effect — even when no continuous linear term
                // overlaps it (e.g. `s(x, by=fac)` with no `+ x`).
                || factor_by_level_gate(termspec).is_some());
        let parametric_block = if !needs_parametric_block {
            None
        } else {
            Some(build_parametric_constraint_block_for_term(
                data,
                linear_terms,
                termspec,
            )?)
        };
        let c_local =
            if skip_global_transform || (parametric_block.is_none() && owner_blocks.is_empty()) {
                None
            } else {
                Some(build_constraint_block(
                    data.nrows(),
                    parametric_block.as_ref(),
                    &owner_blocks,
                )?)
            };
        let z_opt = if let Some(z) = replay_z {
            if design_local.ncols() != z.nrows() {
                crate::bail_dim_basis!(
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
            match maybe_smooth_identifiability_transform(
                termspec,
                &design_local,
                c_local.as_ref().map(|mat| mat.view()),
            ) {
                Ok(z_opt) => z_opt,
                Err(BasisError::ConstraintNullspaceCollapsed { .. })
                    if !owner_blocks.is_empty() =>
                {
                    Some(Array2::zeros((design_local.ncols(), 0)))
                }
                Err(err) => return Err(err),
            }
        };
        let coefficient_gauge = z_opt
            .as_ref()
            .map(|z| crate::solver::gauge::Gauge::from_block_transforms(&[z.clone()]));
        let design_constrained = if let Some(gauge) = coefficient_gauge.as_ref() {
            apply_smooth_transform_to_design(design_local, &gauge.block_transform(0), &term.name)?
        } else {
            design_local
        };

        if let Some(c_ref) = c_local.as_ref() {
            let rel =
                orthogonality_relative_residual_for_design(&design_constrained, c_ref.view())?;
            // Largest relative residual tolerated before the constrained design
            // is rejected as not orthogonal to its sum-to-zero constraint rows.
            const ORTHOGONALITY_REL_RESIDUAL_TOL: f64 = 1e-8;
            let tol = ORTHOGONALITY_REL_RESIDUAL_TOL;
            if rel > tol {
                crate::bail_invalid_basis!(
                    "smooth orthogonality residual too large for term '{}': {:.3e} > {:.1e}",
                    term.name,
                    rel,
                    tol
                );
            }
        }

        let active_penaltyinfo = term
            .penaltyinfo_local
            .iter()
            .filter(|info| info.active)
            .cloned()
            .collect::<Vec<_>>();
        if active_penaltyinfo.len() != term.penalties_local.len() {
            crate::bail_invalid_basis!(
                "internal penalty metadata mismatch for term '{}': activeinfos={}, penalties={}",
                term.name,
                active_penaltyinfo.len(),
                term.penalties_local.len()
            );
        }
        let penalties_constrained: Vec<Array2<f64>> = term
            .penalties_local
            .par_iter()
            .map(|s_local| {
                if let Some(gauge) = coefficient_gauge.as_ref() {
                    gauge.restrict_penalty(s_local)
                } else {
                    s_local.clone()
                }
            })
            .collect();
        let penalty_candidates = penalties_constrained
            .into_par_iter()
            .zip(active_penaltyinfo.into_par_iter())
            .map(|(matrix, info)| {
                let (matrix, c_new) = normalize_penalty_in_constrained_space(&matrix);
                PenaltyCandidate {
                    nullspace_dim_hint: info.nullspace_dim_hint,
                    matrix,
                    source: info.source,
                    normalization_scale: info.normalization_scale * c_new,
                    kronecker_factors: None,
                    op: None,
                }
            })
            .collect::<Vec<_>>();
        let (penalties_constrained, nullspace_constrained, penaltyinfo_constrained) =
            filter_active_penalty_candidates(penalty_candidates)?;
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

        local_dims[idx] = design_constrained.ncols();
        local_designs[idx] = Some(design_constrained);
        local_penalties[idx] = penalties_constrained;
        local_nullspaces[idx] = nullspace_constrained;
        local_penaltyinfo[idx] = penaltyinfo_constrained;
        local_linear_constraints[idx] = linear_constraints_constrained;
        let realized_transform = match (term.joint_null_rotation.as_ref(), z_opt.as_ref()) {
            (Some(rotation), Some(z)) => {
                Some(crate::linalg::faer_ndarray::fast_ab(&rotation.rotation, z))
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
    let mut dropped_penaltyinfo_global = smooth.dropped_penaltyinfo.clone();
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(total_p, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraintsrows: Vec<Array1<f64>> = Vec::new();
    let mut linear_constraints_b: Vec<f64> = Vec::new();

    let mut col_start = 0usize;
    for idx in 0..smooth.terms.len() {
        let p_local = local_dims[idx];
        let col_end = col_start + p_local;

        let activeinfos = local_penaltyinfo[idx]
            .iter()
            .filter(|info| info.active)
            .collect::<Vec<_>>();
        if activeinfos.len() != local_penalties[idx].len() {
            crate::bail_invalid_basis!(
                "internal penalty info mismatch for term '{}': activeinfos={}, penalties={}",
                smooth.terms[idx].name,
                activeinfos.len(),
                local_penalties[idx].len()
            );
        }
        for ((s_local, &ns), info) in local_penalties[idx]
            .iter()
            .zip(local_nullspaces[idx].iter())
            .zip(activeinfos.into_iter())
        {
            let global_index = penalties_global.len();
            penalties_global.push(BlockwisePenalty::new(col_start..col_end, s_local.clone()));
            nullspace_dims_global.push(ns);
            let mut penalty = info.clone();
            penalty.nullspace_dim_hint = ns;
            penaltyinfo_global.push(PenaltyBlockInfo {
                global_index,
                termname: Some(smooth.terms[idx].name.clone()),
                penalty,
            });
        }
        for info in local_penaltyinfo[idx].iter().filter(|info| !info.active) {
            dropped_penaltyinfo_global.push(DroppedPenaltyBlockInfo {
                termname: Some(smooth.terms[idx].name.clone()),
                penalty: info.clone(),
            });
        }

        terms_out.push(SmoothTerm {
            name: smooth.terms[idx].name.clone(),
            coeff_range: col_start..col_end,
            shape: smooth.terms[idx].shape,
            penalties_local: local_penalties[idx].clone(),
            nullspace_dims: local_nullspaces[idx].clone(),
            penaltyinfo_local: local_penaltyinfo[idx].clone(),
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

    Ok(SmoothDesign {
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
    })
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
            crate::bail_dim_basis!(
                "factor-by smooth term '{}' by column {by_col} out of bounds for {p_data} columns",
                termspec.name
            );
        }
        let mut c = Array2::<f64>::zeros((n, 1));
        let by = data.column(by_col);
        for (row, &value) in by.iter().enumerate() {
            if value.to_bits() == value_bits {
                c[[row, 0]] = 1.0;
            }
        }
        return Ok(c);
    }

    let feature_cols = smooth_term_feature_cols(termspec);
    let mut parametric_cols = smooth_intrinsic_parametric_feature_cols(linear_terms, termspec);
    for &feature_col in &parametric_cols {
        if feature_col >= p_data {
            crate::bail_dim_basis!(
                "smooth term feature column {feature_col} out of bounds for {p_data} columns"
            );
        }
    }
    for linear in linear_terms
        .iter()
        .filter(|linear| feature_cols.contains(&linear.feature_col))
    {
        if linear.feature_col >= p_data {
            crate::bail_dim_basis!(
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

fn apply_smooth_transform_to_design(
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
            Ok(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Arc::new(op),
            )))
        }
        DesignMatrix::Sparse(inner) => {
            let dense = inner
                .try_to_dense_arc("smooth identifiability sparse transform")
                .map_err(BasisError::InvalidInput)?
                .as_ref()
                .dot(transform);
            Ok(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                dense,
            )))
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
        cross += &crate::faer_ndarray::fast_atb(&design_chunk, &constraint_chunk);
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

fn maybe_smooth_identifiability_transform(
    termspec: &SmoothTermSpec,
    design_local: &DesignMatrix,
    constraint_block: Option<ArrayView2<'_, f64>>,
) -> Result<Option<Array2<f64>>, BasisError> {
    if let Some(SpatialIdentifiability::FrozenTransform { transform }) =
        spatial_identifiability_policy(termspec)
    {
        if design_local.ncols() != transform.nrows() {
            crate::bail_dim_basis!(
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

fn spatial_identifiability_policy(termspec: &SmoothTermSpec) -> Option<&SpatialIdentifiability> {
    match &termspec.basis {
        SmoothBasisSpec::ThinPlate { spec, .. } => Some(&spec.identifiability),
        SmoothBasisSpec::Duchon { spec, .. } => Some(&spec.identifiability),
        _ => None,
    }
}

/// Whether this smooth's *realized* design (the basis evaluated at the n data
/// rows) must be residualized against the model's parametric block (intercept +
/// any overlapping linear columns) by `apply_global_smooth_identifiability`.
///
/// This is the universal identifiability invariant for **kernel / radial**
/// spatial smooths (#531): their realized column span contains the constant
/// (and, at `Linear` null-space order, the linear monomials), so without this
/// step the smooth and the parametric intercept fight over the same direction —
/// a structural rank-1 collision. The collision is invisible to the kernels'
/// *own* identifiability constraints because those act in **coefficient space at
/// the K centers**, not on the realized design rows:
///   - Matérn `CenterSumToZero` enforces `1ᵀα = 0` over the centers, so
///     `Kα` evaluated at the data rows still spans the constant.
///   - Duchon / TPS `OrthogonalToParametric` *defers* its centering to this very
///     step, which is why it is listed here too.
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
                name: termspec.name.clone(),
                basis: (**inner).clone(),
                shape: termspec.shape,
                joint_null_rotation: None,
            })
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => {
            smooth_requires_parametric_orthogonality(&SmoothTermSpec {
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
        // leaves the realized `K·z` design spanning the constant on the data
        // rows, so the global parametric orthogonalization must compose onto
        // `z` (#532).
        SmoothBasisSpec::ConstantCurvature { spec, .. } => matches!(
            spec.identifiability,
            ConstantCurvatureIdentifiability::CenterSumToZero
        ),
        // Measure-jet representer: identical #531 collision class to the raw
        // finite-center Wahba sphere. Gaussian RBF columns times the
        // center-space sum-to-zero `z` still span the constant on the data rows,
        // so `z` must absorb the parametric orthogonalization (#532).
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
        } => Ok(BasisMetadata::BSpline1D {
            knots: knots.clone(),
            periodic: *periodic,
            identifiability_transform: compose_identifiability_transforms(
                identifiability_transform.as_ref(),
                transform,
            )?,
            degree: *degree,
            auto_shrink_note: auto_shrink_note.clone(),
        }),
        BasisMetadata::ThinPlate {
            centers,
            length_scale,
            periodic,
            identifiability_transform,
            input_scales,
            radial_reparam,
        } => Ok(BasisMetadata::ThinPlate {
            centers: centers.clone(),
            length_scale: *length_scale,
            periodic: periodic.clone(),
            identifiability_transform: compose_identifiability_transforms(
                identifiability_transform.as_ref(),
                transform,
            )?,
            input_scales: input_scales.clone(),
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
            input_scales,
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
        } => Ok(BasisMetadata::MeasureJet {
            centers: centers.clone(),
            input_scales: input_scales.clone(),
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
        }),
        BasisMetadata::Matern {
            centers,
            length_scale,
            periodic,
            nu,
            include_intercept,
            identifiability_transform,
            input_scales,
            aniso_log_scales,
            nullspace_shrinkage_survived,
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
            input_scales: input_scales.clone(),
            aniso_log_scales: aniso_log_scales.clone(),
            nullspace_shrinkage_survived: *nullspace_shrinkage_survived,
        }),
        BasisMetadata::Duchon {
            centers,
            length_scale,
            periodic,
            power,
            nullspace_order,
            identifiability_transform,
            input_scales,
            aniso_log_scales,
            operator_collocation_points,
            radial_reparam,
        } => Ok(BasisMetadata::Duchon {
            centers: centers.clone(),
            length_scale: *length_scale,
            periodic: periodic.clone(),
            power: *power,
            nullspace_order: *nullspace_order,
            input_scales: input_scales.clone(),
            aniso_log_scales: aniso_log_scales.clone(),
            operator_collocation_points: operator_collocation_points.clone(),
            radial_reparam: radial_reparam.clone(),
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
            identifiability_transform,
        } => Ok(BasisMetadata::TensorBSpline {
            feature_cols: feature_cols.clone(),
            knots: knots.clone(),
            degrees: degrees.clone(),
            periods: periods.clone(),
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
        } => {
            // Factor-smooth metadata has no transform slot; the global pass
            // exports its transform via `SmoothTerm::unabsorbed_global_orthogonality`
            // instead (#978). Silently dropping a transform here is what made
            // `s(x) + fs(x, g)` unpredictable — reject loudly so any future
            // caller that reaches this arm with a transform fails at fit time
            // rather than corrupting the saved coefficient chart.
            if transform.is_some() {
                crate::bail_invalid_basis!(
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
                crate::bail_invalid_basis!(
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

fn orthogonality_relative_residual_for_design(
    design: &DesignMatrix,
    constraint_matrix: ArrayView2<'_, f64>,
) -> Result<f64, BasisError> {
    let cross = design_constraint_cross(design, constraint_matrix)?;
    let num = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
    let b_norm = design_frobenius_norm(design)?;
    let c_norm = constraint_matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
    let denom = (b_norm * c_norm).max(1e-300);
    Ok(num / denom)
}

pub fn fit_term_collection_forspec(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    fit_term_collection_forspecwith_heuristic_lambdas(
        data, y, weights, offset, spec, None, family, options,
    )
}

pub fn fit_term_collection_with_coefficient_groups(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    groups: &[CoefficientGroupSpec],
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    if groups.is_empty() {
        return fit_term_collection_forspec(data, y, weights, offset, spec, family, options);
    }
    let design = build_term_collection_design(data, spec)?;
    let base_fit_opts = adaptive_fit_options_base(options, &design);
    let realized = design
        .realize_coefficient_groups(groups, &base_fit_opts.rho_prior)
        .map_err(EstimationError::BasisError)?;
    let mut grouped_options = base_fit_opts.clone();
    grouped_options.rho_prior = realized.rho_prior;
    let fitted = FittedTermCollection {
        fit: crate::estimate::fit_gam_with_penalty_specs(
            design.design.clone(),
            y,
            weights,
            offset,
            realized.penalty_specs,
            realized.nullspace_dims,
            family.clone(),
            &grouped_options,
        )?,
        design,
        adaptive_diagnostics: None,
    };
    enforce_term_constraint_feasibility(&fitted.design, &fitted.fit)?;
    Ok(fitted)
}

pub fn fit_term_collection_with_penalty_block_gamma_prior_callback<F>(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    callback: F,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError>
where
    F: FnMut(&PenaltyBlockGammaPriorMetadata<'_>) -> Option<(f64, f64)>,
{
    let design = build_term_collection_design(data, spec)?;
    let mut fit_opts = adaptive_fit_options_base(options, &design);
    fit_opts.rho_prior = realize_penalty_block_gamma_priors(&design, callback)
        .map_err(EstimationError::BasisError)?;
    let fitted = FittedTermCollection {
        fit: fit_gamwith_heuristic_lambdas(
            design.design.clone(),
            y,
            weights,
            offset,
            &design.penalties,
            None,
            family.clone(),
            &fit_opts,
        )?,
        design,
        adaptive_diagnostics: None,
    };
    enforce_term_constraint_feasibility(&fitted.design, &fitted.fit)?;
    Ok(fitted)
}

pub fn fit_term_collection_with_penalty_block_gamma_priors(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    priors: &[(String, f64, f64)],
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    let design = build_term_collection_design(data, spec)?;
    let mut fit_opts = adaptive_fit_options_base(options, &design);
    fit_opts.rho_prior = realize_keyed_penalty_block_gamma_priors(&design, priors)
        .map_err(EstimationError::BasisError)?;
    let fitted = FittedTermCollection {
        fit: fit_gamwith_heuristic_lambdas(
            design.design.clone(),
            y,
            weights,
            offset,
            &design.penalties,
            None,
            family.clone(),
            &fit_opts,
        )?,
        design,
        adaptive_diagnostics: None,
    };
    enforce_term_constraint_feasibility(&fitted.design, &fitted.fit)?;
    Ok(fitted)
}

pub fn fit_term_collection_with_coefficient_groups_and_penalty_block_gamma_priors(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    groups: &[CoefficientGroupSpec],
    priors: &[(String, f64, f64)],
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    if groups.is_empty() {
        return fit_term_collection_with_penalty_block_gamma_priors(
            data, y, weights, offset, spec, priors, family, options,
        );
    }
    if priors.is_empty() {
        return fit_term_collection_with_coefficient_groups(
            data, y, weights, offset, spec, groups, family, options,
        );
    }

    let design = build_term_collection_design(data, spec)?;
    let base_fit_opts = adaptive_fit_options_base(options, &design);
    let base_rho_prior = realize_keyed_penalty_block_gamma_priors(&design, priors)
        .map_err(EstimationError::BasisError)?;
    let realized = design
        .realize_coefficient_groups(groups, &base_rho_prior)
        .map_err(EstimationError::BasisError)?;
    let mut grouped_options = base_fit_opts.clone();
    grouped_options.rho_prior = realized.rho_prior;
    let fitted = FittedTermCollection {
        fit: crate::estimate::fit_gam_with_penalty_specs(
            design.design.clone(),
            y,
            weights,
            offset,
            realized.penalty_specs,
            realized.nullspace_dims,
            family.clone(),
            &grouped_options,
        )?,
        design,
        adaptive_diagnostics: None,
    };
    enforce_term_constraint_feasibility(&fitted.design, &fitted.fit)?;
    Ok(fitted)
}

fn fit_term_collection_forspecwith_heuristic_lambdas(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    heuristic_lambdas: Option<&[f64]>,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    let base_design = build_term_collection_design(data, spec)?;
    fit_term_collection_on_realized_design(
        y,
        weights,
        offset,
        spec,
        &base_design,
        heuristic_lambdas,
        family,
        options,
    )
}

fn has_bounded_linear_terms(spec: &TermCollectionSpec) -> bool {
    spec.linear_terms.iter().any(|term| {
        matches!(
            term.coefficient_geometry,
            LinearCoefficientGeometry::Bounded { .. }
        )
    })
}

fn fit_term_collection_on_realized_design(
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    heuristic_lambdas: Option<&[f64]>,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    if has_bounded_linear_terms(spec) {
        return fit_bounded_term_collection_with_design(
            y,
            weights,
            offset,
            spec,
            design,
            heuristic_lambdas,
            family,
            options,
        );
    }
    let mut base_fit_opts = adaptive_fit_options_base(options, design);
    // Lift the symmetric log-λ cap off the smoothing coordinates of
    // well-determined Gaussian-identity B-spline / thin-plate / tensor smooths so
    // REML can drive λ to the value the data wants — including λ → ∞ when a
    // term's signal lives in its penalty null space (#1271 single-penalty tp/ps,
    // #1266 double-penalty selection). Length-safe: only fires when the inner ρ
    // aligns 1:1 with the penalty blocks (see `relax_smoothing_rho_prior`).
    base_fit_opts.rho_prior = relax_smoothing_rho_prior(options, &family, design);
    let fitted = FittedTermCollection {
        fit: fit_gamwith_heuristic_lambdas(
            design.design.clone(),
            y,
            weights,
            offset,
            &design.penalties,
            heuristic_lambdas,
            family.clone(),
            &base_fit_opts,
        )?,
        design: design.clone(),
        adaptive_diagnostics: None,
    };
    enforce_term_constraint_feasibility(&fitted.design, &fitted.fit)?;

    let adaptive_opts = options.adaptive_regularization.clone().unwrap_or_default();
    if !adaptive_opts.enabled {
        return Ok(fitted);
    }
    let runtime_caches = extract_spatial_operator_runtime_caches(spec, &fitted.design)?;
    if runtime_caches.is_empty() {
        return Ok(fitted);
    }
    // Spatial-adaptive overlay always runs when the operator caches are
    // non-empty. Catastrophic-overfit protection lives in the operator-log-λ
    // box bound (Fix B at the BFGS bounds construction), which caps maximum
    // unpenalization regardless of n. Production fits at n≈300K must run the
    // overlay; the previous n-gate (n < max(4·p_total, 200)) silently skipped
    // it for any small-n test, contradicting that contract.
    fit_term_collectionwith_exact_spatial_adaptive_regularization(
        fitted,
        y,
        weights,
        offset,
        family,
        options,
        &runtime_caches,
    )
}

#[derive(Clone)]
struct SpatialOperatorRuntimeCache {
    termname: String,
    feature_cols: Vec<usize>,
    coeff_global_range: Range<usize>,
    mass_penalty_global_idx: usize,
    tension_penalty_global_idx: usize,
    stiffness_penalty_global_idx: usize,
    d0: Array2<f64>,
    d1: Array2<f64>,
    d2: Array2<f64>,
    collocation_points: Array2<f64>,
    dimension: usize,
}

#[derive(Clone)]
struct SpatialAdaptiveWeights {
    inv_magweight: Array1<f64>,
    invgradweight: Array1<f64>,
    inv_lapweight: Array1<f64>,
}

#[derive(Clone)]
struct CharbonnierScalarBlockState {
    signal: Array1<f64>,
    radius: Array1<f64>,
    epsilon: f64,
}

impl CharbonnierScalarBlockState {
    fn from_signal(signal: Array1<f64>, epsilon: f64) -> Self {
        let eps = epsilon.max(1e-12);
        let radius = signal.mapv(|t| (t * t + eps * eps).sqrt());
        Self {
            signal,
            radius,
            epsilon: eps,
        }
    }

    fn absolute_signal(&self) -> Array1<f64> {
        self.signal.mapv(f64::abs)
    }

    fn penalty_value(&self) -> f64 {
        self.radius.iter().map(|r| r - self.epsilon).sum::<f64>()
    }

    fn betagradient_coeff(&self) -> Array1<f64> {
        Array1::from_iter(
            self.signal
                .iter()
                .zip(self.radius.iter())
                .map(|(t, r)| t / r),
        )
    }

    fn betahessian_diag(&self) -> Array1<f64> {
        let eps2 = self.epsilon * self.epsilon;
        self.radius.mapv(|r| eps2 / r.powi(3))
    }

    fn log_epsilon_gradient_terms(&self) -> Array1<f64> {
        let epsilon = self.epsilon;
        let eps2 = epsilon * epsilon;
        self.radius.mapv(|r| eps2 / r - epsilon)
    }

    fn log_epsilon_betagradient_coeff(&self) -> Array1<f64> {
        let eps2 = self.epsilon * self.epsilon;
        Array1::from_iter(
            self.signal
                .iter()
                .zip(self.radius.iter())
                .map(|(t, r)| -eps2 * t / r.powi(3)),
        )
    }

    fn log_epsilon_hessian_terms(&self) -> Array1<f64> {
        let epsilon = self.epsilon;
        let eps2 = epsilon * epsilon;
        let eps4 = eps2 * eps2;
        self.radius
            .mapv(|r| 2.0 * eps2 / r - eps4 / r.powi(3) - epsilon)
    }

    fn surrogateweights_posterior_snr(
        &self,
        variance: &Array1<f64>,
        weight_floor: f64,
        weight_ceiling: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        // Posterior-SNR (credible-magnitude) reweighting of the scalar MM
        // majorizer.
        //
        // The magnitude-only surrogate weight uses the *point-estimate* radius
        //
        //   r_k^mag = sqrt( t_k^2 + eps^2 ),   t_k = (D0 beta_hat)_k,
        //   w_k     = 1 / r_k^mag.
        //
        // The weight multiplies the local quadratic surrogate penalty
        // w_k (D0 beta)^2, so a *small* w_k leaves the response un-penalized
        // (treated as a genuine feature) and a *large* w_k pulls it toward zero
        // (enforces flatness). The failure of the point-estimate radius is that
        // a response t_k which is large only because it is poorly determined
        // gets a tiny weight and is left un-penalized — the weight chases noise
        // in low-information regions.
        //
        // Resolution via the posterior second moment under the working-Laplace
        // posterior beta ~ N(beta_hat, Sigma_beta), Sigma_beta = H^{-1}: the
        // variance of the response is
        //
        //   Var( (D0 beta)_k ) = (D0 Sigma_beta D0^T)_kk >= 0,
        //
        // and the *credible* (noise-floor-corrected) squared magnitude is
        //
        //   t_k^credible^2 = max( t_k^2 - Var(...)_k , 0 ),
        //   r_k^snr        = sqrt( t_k^credible^2 + eps^2 ),
        //   w_k            = 1 / r_k^snr.
        //
        // The principled fix evaluates the MM weight at the *credible* (noise-
        // floor-corrected) squared magnitude rather than the raw point estimate.
        // Under the working-Laplace posterior `beta ~ N(beta_hat, Sigma_beta)`,
        // `Sigma_beta = H^{-1}`, the response `t_k = (D0 beta)_k` has posterior
        // mean `t_hat_k` and variance `V_k = (D0 Sigma_beta D0^T)_kk >= 0`. The
        // expected squared response is `E[t_k^2] = t_hat_k^2 + V_k`, so the part
        // of `t_hat_k^2` that exceeds the noise floor `V_k` is the credibly real
        // squared magnitude
        //
        //   t_k^credible^2 = max( t_hat_k^2 - V_k , 0 ),
        //   r_k^snr        = sqrt( t_k^credible^2 + eps^2 ),   w_k = 1 / r_k^snr.
        //
        // This is the correct realization of the intent. Where the point
        // estimate is a *credible* edge (t_hat^2 >> V) the credible magnitude is
        // ~|t_hat| and the weight is essentially `1/|t_hat|` (left un-penalized,
        // edge preserved). Where the large point-estimate magnitude is *noise*
        // (t_hat^2 <~ V) the credible magnitude collapses to 0 and the weight
        // rises to `1/eps` (extra smoothing, noise suppressed). The weight is
        // monotone non-decreasing in `V`, and is bounded above by `1/eps` — the
        // *same* ceiling the magnitude-only weight `1/sqrt(t^2 + eps^2)` already
        // attains at `t = 0` (and clamped by `weight_ceiling`), so it is not an
        // unbounded blow-up: it only moves the noise-dominated rows to the flat-
        // response weight they would have had with a credible estimate of zero
        // curvature. The earlier delta-method form `f + ½ f'' V` was non-monotone
        // (`f''` flips sign at `2t^2 = eps^2`) and unbounded in `V`, which left
        // noisy rows under-penalized and was the source of the SNR regression.
        // With `V == 0` everywhere this degrades exactly to `surrogateweights`
        // (`1/sqrt(t^2 + eps^2)`), so any covariance-unavailable path is
        // unchanged.
        let eps2 = self.epsilon * self.epsilon;
        let weight = Array1::from_iter(self.signal.iter().zip(variance.iter()).map(|(&t, &v)| {
            let credible2 = (t * t - v.max(0.0)).max(0.0);
            let r = (credible2 + eps2).sqrt();
            (1.0 / r).clamp(weight_floor, weight_ceiling)
        }));
        let invweight = weight.mapv(|u| 1.0 / u);
        (weight, invweight)
    }

    fn directionalhessian_diag(&self, direction_signal: &Array1<f64>) -> Array1<f64> {
        // Scalar-image directional third derivative:
        //
        // If t(beta) = A beta and
        //   H(beta) = A^T diag( eps^2 / (t_k(beta)^2 + eps^2)^(3/2) ) A,
        // then for q = A u,
        //
        //   D(H)[u]
        //   = A^T diag( -3 eps^2 t_k q_k / (t_k^2 + eps^2)^(5/2) ) A.
        //
        // This is one of the exact P_{beta,beta,beta}[u] terms needed by the
        // Laplace hypergradient
        //
        //   d/dtheta log det H = tr(H^{-1} Hdot_theta),
        //   Hdot_theta = J_{beta,beta,theta} + D_beta(H)[beta_theta].
        let eps2 = self.epsilon * self.epsilon;
        Array1::from_iter(
            self.signal
                .iter()
                .zip(direction_signal.iter())
                .zip(self.radius.iter())
                .map(|((t, q), r)| -3.0 * eps2 * t * q / r.powi(5)),
        )
    }

    /// Exact scalar-image fourth derivative contracted along two coefficient
    /// directions: with `t(β)=Aβ`, `H(β)=Aᵀ diag(ψ''(t_k)) A`,
    /// `ψ''(t)=ε²/r³`, the second directional derivative of `H` along
    /// `(u, v)` (signals `q1=A u`, `q2=A v`) is
    /// `Aᵀ diag( ψ''''(t_k) q1_k q2_k ) A`, with
    /// `ψ''''(t) = -3 ε² / r⁵ + 15 ε² t² / r⁷`.
    fn second_directionalhessian_diag(
        &self,
        direction1_signal: &Array1<f64>,
        direction2_signal: &Array1<f64>,
    ) -> Array1<f64> {
        let eps2 = self.epsilon * self.epsilon;
        Array1::from_iter(
            self.signal
                .iter()
                .zip(direction1_signal.iter())
                .zip(direction2_signal.iter())
                .zip(self.radius.iter())
                .map(|(((t, q1), q2), r)| {
                    let r2 = r * r;
                    let psi4 = -3.0 * eps2 / r.powi(5) + 15.0 * eps2 * t * t / (r.powi(5) * r2);
                    psi4 * q1 * q2
                }),
        )
    }

    fn log_epsilon_betahessian_diag(&self) -> Array1<f64> {
        let eps2 = self.epsilon * self.epsilon;
        let eps4 = eps2 * eps2;
        Array1::from_iter(
            self.signal
                .iter()
                .zip(self.radius.iter())
                .map(|(_, r)| 2.0 * eps2 / r.powi(3) - 3.0 * eps4 / r.powi(5)),
        )
    }

    fn log_epsilon_beta_mixed_second_coeff(&self) -> Array1<f64> {
        let eps2 = self.epsilon * self.epsilon;
        Array1::from_iter(
            self.signal
                .iter()
                .zip(self.radius.iter())
                .map(|(t, r)| eps2 * t * (eps2 - 2.0 * t * t) / r.powi(5)),
        )
    }

    fn log_epsilon_betahessian_second_diag(&self) -> Array1<f64> {
        let eps2 = self.epsilon * self.epsilon;
        let eps4 = eps2 * eps2;
        let eps6 = eps4 * eps2;
        Array1::from_iter(
            self.radius.iter().map(|r| {
                4.0 * eps2 / r.powi(3) - 18.0 * eps4 / r.powi(5) + 15.0 * eps6 / r.powi(7)
            }),
        )
    }

    fn log_epsilon_betahessian_directional_diag(
        &self,
        direction_signal: &Array1<f64>,
    ) -> Array1<f64> {
        let eps2 = self.epsilon * self.epsilon;
        let eps4 = eps2 * eps2;
        Array1::from_iter(
            self.signal
                .iter()
                .zip(direction_signal.iter())
                .zip(self.radius.iter())
                .map(|((t, q), r)| (-6.0 * eps2 * t / r.powi(5) + 15.0 * eps4 * t / r.powi(7)) * q),
        )
    }
}

#[derive(Clone)]
struct CharbonnierGroupedBlockState {
    norm: Array1<f64>,
    radius: Array1<f64>,
    signal_blocks: Array2<f64>,
    epsilon: f64,
}

impl CharbonnierGroupedBlockState {
    fn from_signal_blocks(signal_blocks: Array2<f64>, epsilon: f64) -> Self {
        let eps = epsilon.max(1e-12);
        let norm = Array1::from_iter(
            signal_blocks
                .rows()
                .into_iter()
                .map(|row| row.iter().map(|v| v * v).sum::<f64>().sqrt()),
        );
        let radius = norm.mapv(|g| (g * g + eps * eps).sqrt());
        Self {
            norm,
            radius,
            signal_blocks,
            epsilon: eps,
        }
    }

    fn penalty_value(&self) -> f64 {
        self.radius.iter().map(|r| r - self.epsilon).sum::<f64>()
    }

    fn norm_signal(&self) -> Array1<f64> {
        self.norm.clone()
    }

    fn betagradient_blocks(&self) -> Array2<f64> {
        let mut out = self.signal_blocks.clone();
        for (k, mut row) in out.rows_mut().into_iter().enumerate() {
            let scale = 1.0 / self.radius[k];
            row.mapv_inplace(|v| v * scale);
        }
        out
    }

    fn betahessian_blocks(&self) -> Vec<Array2<f64>> {
        let mut out = Vec::with_capacity(self.signal_blocks.nrows());
        for (k, row) in self.signal_blocks.rows().into_iter().enumerate() {
            let dim = row.len();
            let mut block = Array2::<f64>::eye(dim);
            block.mapv_inplace(|v| v / self.radius[k]);
            for i in 0..dim {
                for j in 0..dim {
                    block[[i, j]] -= row[i] * row[j] / self.radius[k].powi(3);
                }
            }
            out.push(block);
        }
        out
    }

    fn log_epsilon_gradient_terms(&self) -> Array1<f64> {
        let epsilon = self.epsilon;
        let eps2 = epsilon * epsilon;
        self.radius.mapv(|r| eps2 / r - epsilon)
    }

    fn log_epsilon_betagradient_blocks(&self) -> Array2<f64> {
        let mut out = self.signal_blocks.clone();
        let eps2 = self.epsilon * self.epsilon;
        for (k, mut row) in out.rows_mut().into_iter().enumerate() {
            let scale = -eps2 / self.radius[k].powi(3);
            row.mapv_inplace(|v| v * scale);
        }
        out
    }

    fn log_epsilon_hessian_terms(&self) -> Array1<f64> {
        let epsilon = self.epsilon;
        let eps2 = epsilon * epsilon;
        let eps4 = eps2 * eps2;
        self.radius
            .mapv(|r| 2.0 * eps2 / r - eps4 / r.powi(3) - epsilon)
    }

    fn surrogateweights_posterior_snr(
        &self,
        variance: &Array1<f64>,
        weight_floor: f64,
        weight_ceiling: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        // Grouped posterior-SNR (credible-magnitude) reweighting.
        //
        // The magnitude-only grouped surrogate weight uses the point-estimate
        // block norm
        //
        //   g_k     = ||v_k||_2,   v_k = G_k beta_hat,
        //   r_k^mag = sqrt( g_k^2 + eps^2 ),
        //   w_k     = 1 / r_k^mag.
        //
        // The posterior covariance of the *block* response v_k = G_k beta under
        // beta ~ N(beta_hat, Sigma_beta), Sigma_beta = H^{-1}, has total trace
        //
        //   Cov(v_k)     = G_k Sigma_beta G_k^T   (a block_dim x block_dim block),
        //   variance[k]  = tr(Cov(v_k)) = sum_axis ( G_k[axis] Sigma_beta G_k[axis]^T ),
        //
        // i.e. the variance aggregated over the axis-block in the same way
        // `norm` aggregates ||v_k||^2. As for the scalar block, we deflate the
        // squared block norm by this noise floor to obtain the credible squared
        // magnitude and shrink poorly-determined responses toward zero:
        //
        //   g_k^credible^2 = max( g_k^2 - tr(Cov(v_k)) , 0 ),
        //   r_k^snr        = sqrt( g_k^credible^2 + eps^2 ),   w_k = 1 / r_k^snr.
        //
        // A block whose norm is credibly large (g_k^2 >> tr Cov) keeps a small
        // weight (real feature, left un-penalized); a block whose norm is
        // dominated by posterior variance has its credible norm collapse to 0,
        // raising the weight to `1/eps` (noise suppressed). The weight is
        // monotone non-decreasing in `tr Cov` and bounded above by `1/eps` — the
        // same ceiling the magnitude-only weight already attains at `g = 0`
        // (and clamped by `weight_ceiling`), so it is not an unbounded blow-up.
        //
        // This evaluates the grouped MM weight `f(v) = (||v||^2 + eps^2)^{-1/2}`
        // at the credible block norm rather than at the raw point estimate. The
        // expected squared block norm under `v_k ~ N(v_hat_k, C_k)` is
        // `E[||v_k||^2] = ||v_hat_k||^2 + tr(C_k)`, so the credibly-real squared
        // norm is `max(g_k^2 - tr(C_k), 0)`, identical in form to the scalar
        // path (`block_dim == 1` recovers it exactly). The earlier delta-method
        // correction `½ Σ ∂²f · C` was non-monotone (its sign flips with the
        // Hessian of `f`) and unbounded in `tr C`, which under-penalized noisy
        // blocks and was the source of the SNR regression. With `tr C == 0` it
        // recovers `1/sqrt(g^2 + eps^2)`.
        let eps2 = self.epsilon * self.epsilon;
        let weight = Array1::from_iter(self.norm.iter().zip(variance.iter()).map(|(&g, &v)| {
            let credible2 = (g * g - v.max(0.0)).max(0.0);
            let r = (credible2 + eps2).sqrt();
            (1.0 / r).clamp(weight_floor, weight_ceiling)
        }));
        let invweight = weight.mapv(|u| 1.0 / u);
        (weight, invweight)
    }

    fn directionalhessian_blocks(&self, direction_blocks: &Array2<f64>) -> Vec<Array2<f64>> {
        // Exact grouped directional third derivative for the slope penalty.
        //
        // For each collocation block k:
        //   v_k = G_k beta,
        //   q_k = G_k u,
        //   r_k = sqrt(||v_k||^2 + eps^2),
        //
        // the exact Hessian block for psi(g; eps) = sqrt(g^2 + eps^2) - eps is
        //   B_k,
        //   B_k = (1 / r_k) I - v_k v_k^T / r_k^3.
        //
        // Differentiating B_k along u gives
        //   M_k(u)
        //   = -(v_k^T q_k / r_k^3) I
        //     - (q_k v_k^T + v_k q_k^T) / r_k^3
        //     + 3 (v_k^T q_k) v_k v_k^T / r_k^5.
        //
        // This expression must be symmetric because it is the directional
        // derivative of the symmetric matrix
        //
        //   B_k = (1 / r_k) I - v_k v_k^T / r_k^3.
        //
        // The full directional penalty Hessian map is then
        //   D(H_g)[u] = lambda_g * sum_k G_k^T M_k(u) G_k.
        let mut out = Vec::with_capacity(self.signal_blocks.nrows());
        for (k, (v, q)) in self
            .signal_blocks
            .rows()
            .into_iter()
            .zip(direction_blocks.rows().into_iter())
            .enumerate()
        {
            let dim = v.len();
            let dot = v.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f64>();
            let r3 = self.radius[k].powi(3);
            let r5 = self.radius[k].powi(5);
            let mut block = Array2::<f64>::eye(dim);
            block.mapv_inplace(|x| -dot * x / r3);
            for i in 0..dim {
                for j in 0..dim {
                    block[[i, j]] -= (q[i] * v[j] + v[i] * q[j]) / r3;
                    block[[i, j]] += 3.0 * dot * v[i] * v[j] / r5;
                }
            }
            out.push(block);
        }
        out
    }

    /// Exact grouped second directional derivative of the slope/curvature block
    /// Hessian `B_k = (1/r_k) I − v_k v_kᵀ / r_k³` along two coefficient
    /// directions, with per-block signal images `a_k = G_k u`, `b_k = G_k w`.
    ///
    /// `B_k`'s first directional derivative along `a` is
    ///   `M_k(a) = −(v·a/r³) I − (a vᵀ + v aᵀ)/r³ + 3 (v·a) v vᵀ/r⁵`
    /// (see `directionalhessian_blocks`). Differentiating `M_k(a)` once more
    /// along `b` (i.e. `v ← v + t b`) gives the symmetric block
    ///   `N_k(a,b) = (−a·b/r³ + 3 (v·a)(v·b)/r⁵) I`
    ///            `  − (a bᵀ + b aᵀ)/r³`
    ///            `  + 3 (v·b)(a vᵀ + v aᵀ)/r⁵`
    ///            `  + 3 (a·b) v vᵀ/r⁵`
    ///            `  + 3 (v·a)(b vᵀ + v bᵀ)/r⁵`
    ///            `  − 15 (v·a)(v·b) v vᵀ/r⁷`,
    /// so `D²_β H_g[u,w] = λ_g Σ_k G_kᵀ N_k(a_k,b_k) G_k`. `N_k` is symmetric in
    /// `a ↔ b`, matching `D²H[u,w] = D²H[w,u]`.
    fn second_directionalhessian_blocks(
        &self,
        direction1_blocks: &Array2<f64>,
        direction2_blocks: &Array2<f64>,
    ) -> Vec<Array2<f64>> {
        let mut out = Vec::with_capacity(self.signal_blocks.nrows());
        for ((k, v), (a, b)) in self.signal_blocks.rows().into_iter().enumerate().zip(
            direction1_blocks
                .rows()
                .into_iter()
                .zip(direction2_blocks.rows().into_iter()),
        ) {
            let dim = v.len();
            let dot = |x: ndarray::ArrayView1<'_, f64>, y: ndarray::ArrayView1<'_, f64>| {
                x.iter().zip(y.iter()).map(|(p, q)| p * q).sum::<f64>()
            };
            let sa = dot(v, a);
            let sb = dot(v, b);
            let ab = dot(a, b);
            let r = self.radius[k];
            let r3 = r.powi(3);
            let r5 = r.powi(5);
            let r7 = r5 * r * r;
            let diag = -ab / r3 + 3.0 * sa * sb / r5;
            let mut block = Array2::<f64>::eye(dim);
            block.mapv_inplace(|x| diag * x);
            for i in 0..dim {
                for j in 0..dim {
                    block[[i, j]] -= (a[i] * b[j] + b[i] * a[j]) / r3;
                    block[[i, j]] += 3.0 * sb * (a[i] * v[j] + v[i] * a[j]) / r5;
                    block[[i, j]] += 3.0 * ab * v[i] * v[j] / r5;
                    block[[i, j]] += 3.0 * sa * (b[i] * v[j] + v[i] * b[j]) / r5;
                    block[[i, j]] -= 15.0 * sa * sb * v[i] * v[j] / r7;
                }
            }
            out.push(block);
        }
        out
    }

    fn log_epsilon_betahessian_blocks(&self) -> Vec<Array2<f64>> {
        let mut out = Vec::with_capacity(self.signal_blocks.nrows());
        for (k, row) in self.signal_blocks.rows().into_iter().enumerate() {
            let dim = row.len();
            let r3 = self.radius[k].powi(3);
            let r5 = self.radius[k].powi(5);
            let mut block = Array2::<f64>::eye(dim);
            let eps2 = self.epsilon * self.epsilon;
            block.mapv_inplace(|v| -eps2 * v / r3);
            for i in 0..dim {
                for j in 0..dim {
                    block[[i, j]] += 3.0 * eps2 * row[i] * row[j] / r5;
                }
            }
            out.push(block);
        }
        out
    }

    fn log_epsilon_beta_mixed_second_blocks(&self) -> Array2<f64> {
        let mut out = self.signal_blocks.clone();
        let eps2 = self.epsilon * self.epsilon;
        for (k, mut row) in out.rows_mut().into_iter().enumerate() {
            let norm2 = self.norm[k] * self.norm[k];
            let scale = eps2 * (eps2 - 2.0 * norm2) / self.radius[k].powi(5);
            row.mapv_inplace(|v| v * scale);
        }
        out
    }

    fn log_epsilon_betahessian_second_blocks(&self) -> Vec<Array2<f64>> {
        let mut out = Vec::with_capacity(self.signal_blocks.nrows());
        let eps2 = self.epsilon * self.epsilon;
        for (k, row) in self.signal_blocks.rows().into_iter().enumerate() {
            let dim = row.len();
            let norm2 = self.norm[k] * self.norm[k];
            let r5 = self.radius[k].powi(5);
            let r7 = self.radius[k].powi(7);
            let mut block = Array2::<f64>::eye(dim);
            block.mapv_inplace(|v| eps2 * (eps2 - 2.0 * norm2) * v / r5);
            for i in 0..dim {
                for j in 0..dim {
                    block[[i, j]] += 3.0 * eps2 * (2.0 * norm2 - 3.0 * eps2) * row[i] * row[j] / r7;
                }
            }
            out.push(block);
        }
        out
    }

    fn log_epsilon_betahessian_directional_blocks(
        &self,
        direction_blocks: &Array2<f64>,
    ) -> Vec<Array2<f64>> {
        let mut out = Vec::with_capacity(self.signal_blocks.nrows());
        let eps2 = self.epsilon * self.epsilon;
        for (k, (v, q)) in self
            .signal_blocks
            .rows()
            .into_iter()
            .zip(direction_blocks.rows().into_iter())
            .enumerate()
        {
            let dim = v.len();
            let dot = v.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f64>();
            let r5 = self.radius[k].powi(5);
            let r7 = self.radius[k].powi(7);
            let mut block = Array2::<f64>::eye(dim);
            block.mapv_inplace(|x| 3.0 * eps2 * dot * x / r5);
            for i in 0..dim {
                for j in 0..dim {
                    block[[i, j]] += 3.0 * eps2 * (q[i] * v[j] + v[i] * q[j]) / r5;
                    block[[i, j]] -= 15.0 * eps2 * dot * v[i] * v[j] / r7;
                }
            }
            out.push(block);
        }
        out
    }
}

fn scalar_operatorgradient(operator: &Array2<f64>, coeff: &Array1<f64>) -> Array1<f64> {
    operator.t().dot(coeff)
}

fn scalar_operatorhessian(operator: &Array2<f64>, diag: &Array1<f64>) -> Array2<f64> {
    let mut weighted = operator.clone();
    for (k, &w) in diag.iter().enumerate() {
        weighted.row_mut(k).mapv_inplace(|v| v * w);
    }
    let gram = operator.t().dot(&weighted);
    (&gram + &gram.t().to_owned()) * 0.5
}

fn grouped_operatorgradient(
    d1: &Array2<f64>,
    dimension: usize,
    blocks: &Array2<f64>,
) -> Result<Array1<f64>, EstimationError> {
    if blocks.ncols() != dimension {
        crate::bail_invalid_estim!(
            "grouped gradient block dimension mismatch: got {}, expected {dimension}",
            blocks.ncols()
        );
    }
    if d1.nrows() != blocks.nrows() * dimension {
        crate::bail_invalid_estim!(
            "grouped gradient row mismatch: D1 has {} rows, blocks imply {}",
            d1.nrows(),
            blocks.nrows() * dimension
        );
    }
    let mut out = Array1::<f64>::zeros(d1.ncols());
    for k in 0..blocks.nrows() {
        let gk = d1
            .slice(s![k * dimension..(k + 1) * dimension, ..])
            .to_owned();
        out += &gk.t().dot(&blocks.row(k));
    }
    Ok(out)
}

fn grouped_operatorhessian(
    d1: &Array2<f64>,
    dimension: usize,
    blocks: &[Array2<f64>],
) -> Result<Array2<f64>, EstimationError> {
    if d1.nrows() != blocks.len() * dimension {
        crate::bail_invalid_estim!(
            "grouped Hessian row mismatch: D1 has {} rows, blocks imply {}",
            d1.nrows(),
            blocks.len() * dimension
        );
    }
    let p = d1.ncols();
    let mut out = Array2::<f64>::zeros((p, p));
    for (k, block) in blocks.iter().enumerate() {
        if block.nrows() != dimension || block.ncols() != dimension {
            crate::bail_invalid_estim!(
                "grouped Hessian block {k} has shape {}x{}, expected {}x{}",
                block.nrows(),
                block.ncols(),
                dimension,
                dimension
            );
        }
        let gk = d1
            .slice(s![k * dimension..(k + 1) * dimension, ..])
            .to_owned();
        out += &gk.t().dot(&block.dot(&gk));
    }
    Ok((&out + &out.t().to_owned()) * 0.5)
}

#[derive(Clone)]
struct SpatialPenaltyExactState {
    magnitude: CharbonnierScalarBlockState,
    gradient: CharbonnierGroupedBlockState,
    curvature: CharbonnierGroupedBlockState,
}

fn collocationgradient_blocks(
    gradrows: &Array1<f64>,
    dimension: usize,
) -> Result<Array2<f64>, EstimationError> {
    if dimension == 0 || !gradrows.len().is_multiple_of(dimension) {
        crate::bail_invalid_estim!(
            "invalid collocation gradient layout: rows={}, dimension={dimension}",
            gradrows.len()
        );
    }
    let p = gradrows.len() / dimension;
    let mut out = Array2::<f64>::zeros((p, dimension));
    for k in 0..p {
        for axis in 0..dimension {
            out[[k, axis]] = gradrows[k * dimension + axis];
        }
    }
    Ok(out)
}

fn collocationhessian_blocks(
    hessianrows: &Array1<f64>,
    dimension: usize,
) -> Result<Array2<f64>, EstimationError> {
    let block_dim = dimension.checked_mul(dimension).ok_or_else(|| {
        EstimationError::InvalidInput("invalid collocation Hessian dimension overflow".to_string())
    })?;
    if block_dim == 0 || !hessianrows.len().is_multiple_of(block_dim) {
        crate::bail_invalid_estim!(
            "invalid collocation Hessian layout: rows={}, dimension={dimension}",
            hessianrows.len()
        );
    }
    let p = hessianrows.len() / block_dim;
    let mut out = Array2::<f64>::zeros((p, block_dim));
    for k in 0..p {
        for idx in 0..block_dim {
            out[[k, idx]] = hessianrows[k * block_dim + idx];
        }
    }
    Ok(out)
}

impl SpatialPenaltyExactState {
    fn from_beta_local(
        beta_local: ArrayView1<'_, f64>,
        cache: &SpatialOperatorRuntimeCache,
        epsilons: [f64; 3],
    ) -> Result<Self, EstimationError> {
        // Exact collocation-state extraction for the three Charbonnier penalty blocks.
        //
        // For one spatial smooth term with coefficient vector beta_local, the exact
        // operator-decomposition penalty is built from three collocation images:
        //
        //   magnitude:  f = D0 beta_local
        //   slope:      v_k = G_k beta_local
        //   curvature:  H_k = D2_k beta_local
        //
        // where the gradient operator is stored in row-stacked form:
        //
        //   D1 beta_local in R^(P * d),
        //   row layout = (point 0, axis 0..d-1), (point 1, axis 0..d-1), ...
        //   D2 beta_local in R^(P * d * d),
        //   row layout = (point, Hessian axis_a, Hessian axis_b).
        //
        // so we first reshape that stacked vector into the grouped block array
        //
        //   [v_0^T
        //    ...
        //    v_(P-1)^T]  in R^(P x d).
        //
        // The three exact Charbonnier block states then carry:
        //   - the raw operator signals,
        //   - their radii sqrt(signal^2 + eps^2) or sqrt(||v_k||^2 + eps^2),
        //   - and all exact derivatives derived from those radii.
        //
        // This is the canonical translation from coefficient-space beta to the
        // penalty-side mathematical objects used throughout the implementation.
        let gradientrows = cache.d1.dot(&beta_local);
        let hessianrows = cache.d2.dot(&beta_local);
        Ok(Self {
            magnitude: CharbonnierScalarBlockState::from_signal(
                cache.d0.dot(&beta_local),
                epsilons[0],
            ),
            gradient: CharbonnierGroupedBlockState::from_signal_blocks(
                collocationgradient_blocks(&gradientrows, cache.dimension)?,
                epsilons[1],
            ),
            curvature: CharbonnierGroupedBlockState::from_signal_blocks(
                collocationhessian_blocks(&hessianrows, cache.dimension)?,
                epsilons[2],
            ),
        })
    }

    fn absolute_collocation_magnitudes(&self) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        (
            self.magnitude.absolute_signal(),
            self.gradient.norm_signal(),
            self.curvature.norm_signal(),
        )
    }
}

fn robust_epsilon_from_samples(values: &[f64], min_epsilon_cfg: f64) -> f64 {
    if values.is_empty() {
        return min_epsilon_cfg.max(1e-12);
    }
    let mut clean = values
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .collect::<Vec<_>>();
    if clean.is_empty() {
        return min_epsilon_cfg.max(1e-12);
    }
    clean.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = clean.len();
    let median = quantile_from_sorted(&clean, 0.5);
    let q75 = quantile_from_sorted(&clean, 0.75);
    let q95 = quantile_from_sorted(&clean, 0.95);

    let mut abs_dev = clean
        .iter()
        .map(|v| (v - median).abs())
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    abs_dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = 1.4826 * quantile_from_sorted(&abs_dev, 0.5);

    // Charbonnier/MM requires eps bounded away from zero:
    //   u(t0) = 1 / (2*sqrt(t0^2 + eps^2)) ~ 1/(2*eps) near t0=0.
    // Use robust pilot scale:
    //   s = max(median(z), 1.4826*MAD(z), Q75(z)).
    // If s is tiny (<= delta), fallback to:
    //   s <- max(Q95(z), RMS(z)).
    // If still tiny, fallback to absolute floor s_min.
    // Then eps = kappa * s.
    // Primary robust scale: s = max(median, 1.4826*MAD, Q75).
    let mut scale = median.max(mad).max(q75);

    // Safety threshold delta and absolute floor s_min.
    let delta = (f64::EPSILON.sqrt() * q95.max(1.0))
        .max(min_epsilon_cfg)
        .max(1e-12);
    let s_min = min_epsilon_cfg.max(1e-12);

    // If robust scale is tiny, use high-quantile / RMS fallback.
    if scale <= delta {
        let rms = (clean.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        scale = q95.max(rms);
    }
    if scale <= delta {
        scale = s_min;
    }

    // Start near the observed operator scale so the optimizer begins in a
    // neutral regime where both quadratic and linear behavior are reachable.
    let kappa = 1.0_f64;
    (kappa * scale).max(s_min)
}

fn extract_spatial_operator_runtime_caches(
    spec: &TermCollectionSpec,
    design: &TermCollectionDesign,
) -> Result<Vec<SpatialOperatorRuntimeCache>, EstimationError> {
    let smooth_start = design
        .design
        .ncols()
        .saturating_sub(design.smooth.total_smooth_cols());
    let mut out = Vec::<SpatialOperatorRuntimeCache>::new();
    for (term_idx, (termspec, term_fit)) in spec
        .smooth_terms
        .iter()
        .zip(design.smooth.terms.iter())
        .enumerate()
    {
        let Some(global_base_idx) = smooth_term_penalty_index(spec, design, term_idx) else {
            continue;
        };
        let mut active_local_idx = 0usize;
        let mut mass_local_idx = None;
        let mut tension_local_idx = None;
        let mut stiffness_local_idx = None;
        let mut mass_norm = None;
        let mut tension_norm = None;
        let mut stiffness_norm = None;
        for info in &term_fit.penaltyinfo_local {
            if !info.active {
                continue;
            }
            match info.source {
                PenaltySource::OperatorMass => {
                    mass_local_idx = Some(active_local_idx);
                    mass_norm = Some(info.normalization_scale);
                }
                PenaltySource::OperatorTension => {
                    tension_local_idx = Some(active_local_idx);
                    tension_norm = Some(info.normalization_scale);
                }
                PenaltySource::OperatorStiffness => {
                    stiffness_local_idx = Some(active_local_idx);
                    stiffness_norm = Some(info.normalization_scale);
                }
                _ => {}
            }
            active_local_idx += 1;
        }
        // The Charbonnier adaptive overlay rebuilds the {mass, tension,
        // stiffness} D-operator triplet from explicit collocation derivatives
        // and reweights all three channels in tandem; the stiffness slot in
        // particular is the D2 second-derivative operator. A term that does
        // NOT ship an explicit Stiffness penalty (pure Duchon's RKHS-Primary-
        // curvature layout — `DuchonOperatorPenaltySpec::default()`) has no
        // matching shipped penalty for the Charbonnier D2 surrogate to reweight,
        // so applying the overlay would smuggle a fresh D2 collocation
        // operator into a basis whose curvature is the RKHS Primary Gram (a
        // different mathematical object). Without an explicit Stiffness
        // channel the term must be skipped — the runtime cache for the
        // adaptive overlay simply doesn't apply.
        let (
            Some(mass_local),
            Some(tension_local),
            Some(stiffness_local),
            Some(mass_scale),
            Some(tension_scale),
            Some(stiffness_scale),
        ) = (
            mass_local_idx,
            tension_local_idx,
            stiffness_local_idx,
            mass_norm,
            tension_norm,
            stiffness_norm,
        )
        else {
            continue;
        };
        let mass_global_idx = global_base_idx + mass_local;
        let tension_global_idx = global_base_idx + tension_local;
        let stiffness_global_idx = global_base_idx + stiffness_local;

        let (feature_cols, mut d0, mut d1, mut d2, collocation_points, dim, center_mass_rows) =
            match (&termspec.basis, &term_fit.metadata) {
                (
                    SmoothBasisSpec::Matern { feature_cols, .. },
                    BasisMetadata::Matern {
                        centers,
                        length_scale,
                        nu,
                        include_intercept,
                        identifiability_transform,
                        aniso_log_scales,
                        input_scales,
                        ..
                    },
                ) => {
                    // Match the σ_geom-compensated effective length scale the
                    // design (and shipped penalties) use against the standardized
                    // centers; the raw metadata length_scale lives in original
                    // coordinates and would put this overlay on a different kernel
                    // range than the penalties it scales (#706).
                    let collocation_length_scale = match input_scales.as_deref() {
                        Some(scales) => {
                            compensate_length_scale_for_standardization(*length_scale, scales)
                        }
                        None => *length_scale,
                    };
                    let ops = build_matern_collocation_operator_matrices(
                        centers.view(),
                        None,
                        collocation_length_scale,
                        *nu,
                        *include_intercept,
                        identifiability_transform.as_ref().map(|z| z.view()),
                        aniso_log_scales.as_deref(),
                    )?;
                    (
                        feature_cols.clone(),
                        ops.d0,
                        ops.d1,
                        ops.d2,
                        ops.collocation_points,
                        centers.ncols(),
                        false,
                    )
                }
                (
                    SmoothBasisSpec::Duchon { feature_cols, .. },
                    BasisMetadata::Duchon {
                        centers,
                        length_scale,
                        power,
                        nullspace_order,
                        identifiability_transform,
                        input_scales,
                        aniso_log_scales,
                        operator_collocation_points: Some(collocation_points),
                        ..
                    },
                ) => {
                    let collocation_length_scale = match (length_scale, input_scales.as_deref()) {
                        (Some(ls), Some(scales)) => {
                            Some(compensate_length_scale_for_standardization(*ls, scales))
                        }
                        (Some(ls), None) => Some(*ls),
                        (None, _) => None,
                    };
                    let ops =
                        crate::basis::build_duchon_collocation_operator_matriceswithworkspace(
                            centers.view(),
                            collocation_points.view(),
                            None,
                            collocation_length_scale,
                            *power,
                            *nullspace_order,
                            aniso_log_scales.as_deref(),
                            identifiability_transform.as_ref().map(|z| z.view()),
                            2,
                            &mut BasisWorkspace::default(),
                        )?;
                    (
                        feature_cols.clone(),
                        ops.d0,
                        ops.d1,
                        ops.d2,
                        ops.collocation_points,
                        centers.ncols(),
                        true,
                    )
                }
                _ => continue,
            };
        if center_mass_rows && d0.nrows() > 0 && d0.ncols() > 0 {
            let means = d0.sum_axis(Axis(0)).mapv(|v| v / d0.nrows() as f64);
            for mut row in d0.rows_mut() {
                row -= &means;
            }
        }

        // Runtime operator caches must live on the same normalized penalty scale as the
        // shipped design penalties. The basis builders normalize S0=D0'D0, S1=D1'D1, and
        // S2=D2'D2 before exposing them as smoothing blocks, recording the corresponding
        // Frobenius norms in penaltyinfo_local.normalization_scale. If the exact adaptive
        // path uses raw collocation operators here, then its Charbonnier penalties live on a
        // different geometry from the ordinary Matérn/Duchon penalties:
        //
        //   raw quadratic limit:        beta' (D'D) beta
        //   shipped design penalty:     beta' (D'D / c) beta
        //
        // The correct operator-level normalization is therefore
        //
        //   D_norm = D / sqrt(c),
        //
        // so that D_norm' D_norm = (D'D)/c matches the design penalty exactly. Without this,
        // adaptive lambdas compensate for hidden operator-scale mismatches and are no longer
        // comparable to the baseline smoothing parameters.
        let mass_scale = mass_scale.max(1e-12).sqrt();
        let tension_scale = tension_scale.max(1e-12).sqrt();
        let stiffness_scale = stiffness_scale.max(1e-12).sqrt();
        d0.mapv_inplace(|v| v / mass_scale);
        d1.mapv_inplace(|v| v / tension_scale);
        d2.mapv_inplace(|v| v / stiffness_scale);

        let coeff_global_range =
            (smooth_start + term_fit.coeff_range.start)..(smooth_start + term_fit.coeff_range.end);
        if d0.ncols() != coeff_global_range.len()
            || d1.ncols() != coeff_global_range.len()
            || d2.ncols() != coeff_global_range.len()
        {
            crate::bail_invalid_estim!(
                "spatial operator dimension mismatch for term '{}': D0 cols={}, D1 cols={}, D2 cols={}, coeffs={}",
                term_fit.name,
                d0.ncols(),
                d1.ncols(),
                d2.ncols(),
                coeff_global_range.len()
            );
        }
        out.push(SpatialOperatorRuntimeCache {
            termname: term_fit.name.clone(),
            feature_cols,
            coeff_global_range,
            mass_penalty_global_idx: mass_global_idx,
            tension_penalty_global_idx: tension_global_idx,
            stiffness_penalty_global_idx: stiffness_global_idx,
            d0,
            d1,
            d2,
            collocation_points,
            dimension: dim,
        });
    }
    Ok(out)
}

/// Posterior variance of a scalar collocation operator response under the
/// working-Laplace posterior `beta ~ N(beta_hat, Sigma_local)`.
///
/// For operator row `D_k` (one row of `D0`) acting on the term-local coefficient
/// block, `Var((D beta)_k) = D_k Sigma_local D_k^T = (D Sigma_local D^T)_kk`.
/// We compute it without forming `D Sigma D^T` densely: for each row we evaluate
/// `s_k = Sigma_local D_k^T` (one matrix-vector product) and then `D_k . s_k`.
/// `Sigma_local` is the sub-block of the global conditional covariance
/// `Sigma_beta = H^{-1}` indexed by the term's `coeff_global_range`, i.e. the
/// covariance proxy is the already-materialized inner working-Laplace inverse;
/// no second factorization is formed.
fn scalar_operator_response_variance(
    operator: &Array2<f64>,
    cov_local: &Array2<f64>,
) -> Array1<f64> {
    Array1::from_iter(operator.rows().into_iter().map(|row| {
        let s = cov_local.dot(&row);
        row.dot(&s).max(0.0)
    }))
}

/// Posterior second-moment variance aggregated over each grouped collocation
/// block (gradient/curvature). The grouped operator is stored row-stacked with
/// `block_dim` rows per collocation point (`d` axes for the gradient, `d*d` for
/// the Hessian). For block `k`,
///
///   v_k = G_k beta,   Cov(v_k) = G_k Sigma_local G_k^T   (block_dim x block_dim),
///   variance_k = tr(Cov(v_k)) = sum_axis ( G_k[axis] Sigma_local G_k[axis]^T ),
///
/// which matches how `CharbonnierGroupedBlockState::norm` aggregates
/// `||v_k||^2 = sum_axis (G_k[axis] beta)^2` across the axis-block.
fn grouped_operator_response_variance(
    operator: &Array2<f64>,
    block_dim: usize,
    cov_local: &Array2<f64>,
) -> Result<Array1<f64>, EstimationError> {
    if block_dim == 0 || !operator.nrows().is_multiple_of(block_dim) {
        crate::bail_invalid_estim!(
            "grouped variance row layout invalid: rows={}, block_dim={block_dim}",
            operator.nrows()
        );
    }
    let p = operator.nrows() / block_dim;
    let mut out = Array1::<f64>::zeros(p);
    for k in 0..p {
        let mut acc = 0.0;
        for axis in 0..block_dim {
            let row = operator.row(k * block_dim + axis);
            let s = cov_local.dot(&row);
            acc += row.dot(&s);
        }
        out[k] = acc.max(0.0);
    }
    Ok(out)
}

fn compute_spatial_adaptiveweights_for_beta(
    beta: &Array1<f64>,
    caches: &[SpatialOperatorRuntimeCache],
    epsilon_0: f64,
    epsilon_g: f64,
    epsilon_c: f64,
    weight_floor: f64,
    weight_ceiling: f64,
    beta_covariance: Option<&Array2<f64>>,
) -> Result<Vec<SpatialAdaptiveWeights>, EstimationError> {
    // Charbonnier / pseudo-Huber MM derivation (per collocation scalar t):
    //   psi(t; eps) = sqrt(t^2 + eps^2) - eps
    // and for reference t0 the tangent majorizer in t^2 gives:
    //   psi(t) <= 0.5 * w(t0) * t^2 + const(t0),
    //   w(t0) = 1 / sqrt(t0^2 + eps^2).
    //
    // We apply this to:
    //   t = f_k = |f(z_k)|             (magnitude),
    //   t = g_k = ||nabla f(z_k)||_2   (gradient magnitude),
    //   t = c_k = ||D²f(z_k)||_F       (full Hessian curvature),
    // both computed from beta^(t-1).
    //
    // These w values define the quadratic surrogate penalties:
    //   K0 = D0_con^T W_0 D0_con,  W_0 = diag(w_0)
    //   K1 = D1_con^T W_g D1_con,  W_g = diag(w_g) \otimes I_d  (k,axis order)
    //   K2 = D2_con^T W_c D2_con,  W_c = diag(w_c) \otimes I_(d*d).
    //
    // We clamp w directly, then derive inv_w=1/w for diagnostics and row scaling.
    //
    // Posterior-SNR reweighting (magic by default): when the inner working-Laplace
    // conditional covariance `Sigma_beta = H^{-1}` is available we replace the
    // squared point-estimate radius `t_k^2 + eps^2` by the credible (noise-floor-
    // corrected) second moment `max(t_k^2 - Var((D beta)_k), 0) + eps^2`, with
    // `Var = (D Sigma_beta D^T)_kk`. This stops the weight from leaving derivatives
    // un-penalized just because they are large but poorly determined: such
    // responses are shrunk toward zero (large weight, strong smoothing), while
    // credibly large derivatives (real edges) keep their small weight. `Sigma_beta`
    // here is the already-formed inner Hessian inverse from the final exact-family
    // solve — no second factorization is built; we only reuse the materialized
    // covariance. When the covariance is unavailable (`None`) the variance is zero
    // and this degrades *exactly* to the old magnitude-only radius.
    caches
        .iter()
        .map(|cache| {
            let beta_local = beta.slice(s![cache.coeff_global_range.clone()]);
            let exact = SpatialPenaltyExactState::from_beta_local(
                beta_local,
                cache,
                [epsilon_0, epsilon_g, epsilon_c],
            )?;
            let cov_local = beta_covariance.map(|cov| {
                cov.slice(s![
                    cache.coeff_global_range.clone(),
                    cache.coeff_global_range.clone()
                ])
                .to_owned()
            });
            let dim = cache.dimension;
            let (var_0, var_g, var_c) = match cov_local.as_ref() {
                Some(cov) => (
                    scalar_operator_response_variance(&cache.d0, cov),
                    grouped_operator_response_variance(&cache.d1, dim, cov)?,
                    grouped_operator_response_variance(&cache.d2, dim * dim, cov)?,
                ),
                None => (
                    Array1::<f64>::zeros(exact.magnitude.signal.len()),
                    Array1::<f64>::zeros(exact.gradient.norm.len()),
                    Array1::<f64>::zeros(exact.curvature.norm.len()),
                ),
            };
            let (_, inv_0) = exact.magnitude.surrogateweights_posterior_snr(
                &var_0,
                weight_floor,
                weight_ceiling,
            );
            let (_, inv_g) =
                exact
                    .gradient
                    .surrogateweights_posterior_snr(&var_g, weight_floor, weight_ceiling);
            let (_, inv_c) = exact.curvature.surrogateweights_posterior_snr(
                &var_c,
                weight_floor,
                weight_ceiling,
            );
            Ok(SpatialAdaptiveWeights {
                inv_magweight: inv_0,
                invgradweight: inv_g,
                inv_lapweight: inv_c,
            })
        })
        .collect()
}

fn compute_initial_epsilons(
    beta: &Array1<f64>,
    caches: &[SpatialOperatorRuntimeCache],
    min_epsilon: f64,
) -> Result<(f64, f64, f64), EstimationError> {
    let mut fvals = Vec::<f64>::new();
    let mut gvals = Vec::<f64>::new();
    let mut cvals = Vec::<f64>::new();
    for cache in caches {
        let beta_local = beta.slice(s![cache.coeff_global_range.clone()]);
        let exact = SpatialPenaltyExactState::from_beta_local(
            beta_local,
            cache,
            [min_epsilon, min_epsilon, min_epsilon],
        )?;
        let (f, g, c) = exact.absolute_collocation_magnitudes();
        fvals.extend(f.iter().copied());
        gvals.extend(g.iter().copied());
        cvals.extend(c.iter().copied());
    }
    // Robust epsilon initialization from pilot magnitudes:
    //   s = max(median(z), 1.4826*MAD(z), Q75(z)),
    //   if s is tiny then fallback to max(Q95(z), RMS(z)),
    //   if still tiny then use absolute floor min_epsilon.
    // Epsilon is then kappa * s.
    let eps_0 = robust_epsilon_from_samples(&fvals, min_epsilon);
    let eps_g = robust_epsilon_from_samples(&gvals, min_epsilon);
    let eps_c = robust_epsilon_from_samples(&cvals, min_epsilon);
    Ok((eps_0, eps_g, eps_c))
}

fn exact_spatial_adaptive_penalty_index_set(
    caches: &[SpatialOperatorRuntimeCache],
) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    for cache in caches {
        out.insert(cache.mass_penalty_global_idx);
        out.insert(cache.tension_penalty_global_idx);
        out.insert(cache.stiffness_penalty_global_idx);
    }
    out
}

fn build_spatial_adaptive_hyperspecs(cache_count: usize) -> Vec<SpatialAdaptiveHyperSpec> {
    let mut out = Vec::with_capacity(cache_count * 3 + 3);
    for cache_index in 0..cache_count {
        out.push(SpatialAdaptiveHyperSpec {
            cache_index,
            kind: SpatialAdaptiveHyperKind::LogLambdaMagnitude,
        });
        out.push(SpatialAdaptiveHyperSpec {
            cache_index,
            kind: SpatialAdaptiveHyperKind::LogLambdaGradient,
        });
        out.push(SpatialAdaptiveHyperSpec {
            cache_index,
            kind: SpatialAdaptiveHyperKind::LogLambdaCurvature,
        });
    }
    out.push(SpatialAdaptiveHyperSpec {
        cache_index: 0,
        kind: SpatialAdaptiveHyperKind::LogEpsilonMagnitude,
    });
    out.push(SpatialAdaptiveHyperSpec {
        cache_index: 0,
        kind: SpatialAdaptiveHyperKind::LogEpsilonGradient,
    });
    out.push(SpatialAdaptiveHyperSpec {
        cache_index: 0,
        kind: SpatialAdaptiveHyperKind::LogEpsilonCurvature,
    });
    out
}

fn penalty_matrixwith_local_block(
    total_dim: usize,
    coeff_range: Range<usize>,
    local: &Array2<f64>,
) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((total_dim, total_dim));
    out.slice_mut(s![coeff_range.clone(), coeff_range])
        .assign(local);
    out
}

fn fit_term_collectionwith_exact_spatial_adaptive_regularization(
    baseline: FittedTermCollection,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: LikelihoodSpec,
    options: &FitOptions,
    runtime_caches: &[SpatialOperatorRuntimeCache],
) -> Result<FittedTermCollection, EstimationError> {
    // Exact adaptive-regularization hyperfit.
    //
    // This replaces the old MM-plus-approximate hyperfit with the
    // exact pseudo-Laplace objective agreed in the math notes:
    //
    //   L_tilde(theta)
    //   = J(beta_hat(theta); theta) + 0.5 log det H(beta_hat(theta), theta),
    //
    // where:
    //   - beta_hat(theta) is the exact inner mode of the true nonquadratic
    //     Charbonnier-penalized objective,
    //   - theta contains:
    //       * retained quadratic log-lambdas for non-adaptive penalties,
    //       * one log-lambda per adaptive operator block,
    //       * three global log-epsilons shared by every adaptive spatial term,
    //   - H is the exact beta-Hessian of the true objective at the mode.
    //
    // Implementation structure:
    //   1. keep ordinary quadratic penalties that are unrelated to adaptive
    //      spatial terms in the standard outer-rho path;
    //   2. move the adaptive Charbonnier penalties into a one-block exact-Newton
    //      custom family so the inner solve uses the real model rather than an
    //      MM surrogate;
    //   3. expose exact psi-gradients for adaptive log-lambda / log-epsilon
    //      coordinates through the custom-family pseudo-Laplace hook;
    //   4. refit once at the optimized hyperparameters with all penalties frozen
    //      inside the exact family, so covariance and final diagnostics are
    //      computed on the same exact surface.
    let adaptive_opts = options.adaptive_regularization.clone().unwrap_or_default();
    let adaptive_penalty_indices = exact_spatial_adaptive_penalty_index_set(runtime_caches);
    let p_total = baseline.design.design.ncols();
    struct RetainedPenaltySetup {
        global_idx: usize,
        global_penalty: Array2<f64>,
        nullspace_dim: usize,
        log_lambda: f64,
        col_range: Range<usize>,
        hessian_piece: Array2<f64>,
    }
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
    let retained_setups = baseline
        .design
        .penalties
        .par_iter()
        .enumerate()
        .map(|(idx, bp)| {
            if adaptive_penalty_indices.contains(&idx) {
                return None;
            }
            let lambda = baseline.fit.lambdas[idx];
            Some(RetainedPenaltySetup {
                global_idx: idx,
                global_penalty: bp.to_global(p_total),
                nullspace_dim: baseline
                    .design
                    .nullspace_dims
                    .get(idx)
                    .copied()
                    .unwrap_or(0),
                log_lambda: lambda.max(1e-12).ln(),
                col_range: bp.col_range.clone(),
                hessian_piece: bp.local.mapv(|v| lambda * v),
            })
        })
        .collect::<Vec<_>>();
    let retained_count = retained_setups
        .iter()
        .filter(|setup| setup.is_some())
        .count();
    let mut retained_penalties = Vec::<Array2<f64>>::with_capacity(retained_count);
    let mut retained_nullspace_dims = Vec::<usize>::with_capacity(retained_count);
    let mut retained_log_lambdas = Vec::<f64>::with_capacity(retained_count);
    let mut retained_global_indices = Vec::<usize>::with_capacity(retained_count);
    let mut fixed_quadratichessian = Array2::<f64>::zeros((p_total, p_total));
    for setup in retained_setups.into_iter().flatten() {
        retained_penalties.push(setup.global_penalty);
        retained_nullspace_dims.push(setup.nullspace_dim);
        retained_log_lambdas.push(setup.log_lambda);
        retained_global_indices.push(setup.global_idx);
        fixed_quadratichessian
            .slice_mut(s![setup.col_range.clone(), setup.col_range])
            .scaled_add(1.0, &setup.hessian_piece);
    }

    let (eps_0_init, eps_g_init, eps_c_init) = compute_initial_epsilons(
        &baseline.fit.beta,
        runtime_caches,
        adaptive_opts.min_epsilon,
    )?;
    let mut initial_theta =
        Array1::<f64>::zeros(retained_penalties.len() + runtime_caches.len() * 3 + 3);
    for (idx, value) in retained_log_lambdas.iter().enumerate() {
        initial_theta[idx] = *value;
    }
    let adaptive_log_lambda_components = runtime_caches
        .par_iter()
        .map(|cache| {
            [
                baseline.fit.lambdas[cache.mass_penalty_global_idx]
                    .max(1e-12)
                    .ln(),
                baseline.fit.lambdas[cache.tension_penalty_global_idx]
                    .max(1e-12)
                    .ln(),
                baseline.fit.lambdas[cache.stiffness_penalty_global_idx]
                    .max(1e-12)
                    .ln(),
            ]
        })
        .collect::<Vec<_>>();
    let mut at = retained_penalties.len();
    for logs in &adaptive_log_lambda_components {
        initial_theta[at] = logs[0];
        initial_theta[at + 1] = logs[1];
        initial_theta[at + 2] = logs[2];
        at += 3;
    }
    initial_theta[at] = eps_0_init.max(adaptive_opts.min_epsilon).ln();
    initial_theta[at + 1] = eps_g_init.max(adaptive_opts.min_epsilon).ln();
    initial_theta[at + 2] = eps_c_init.max(adaptive_opts.min_epsilon).ln();

    let hyperspecs = build_spatial_adaptive_hyperspecs(runtime_caches.len());
    let zero_psi_op: std::sync::Arc<dyn crate::custom_family::CustomFamilyPsiDerivativeOperator> =
        std::sync::Arc::new(crate::custom_family::ZeroPsiDerivativeOperator::new(
            baseline.design.design.nrows(),
            baseline.design.design.ncols(),
        ));
    let derivative_blocks = vec![
        hyperspecs
            .par_iter()
            .map(|_| CustomFamilyBlockPsiDerivative {
                penalty_index: None,
                x_psi: Array2::<f64>::zeros((0, 0)),
                s_psi: Array2::<f64>::zeros((0, 0)),
                s_psi_components: None,
                s_psi_penalty_components: None,
                x_psi_psi: None,
                s_psi_psi: None,
                s_psi_psi_components: None,
                s_psi_psi_penalty_components: None,
                implicit_operator: Some(std::sync::Arc::clone(&zero_psi_op)),
                implicit_axis: 0,
                implicit_group_id: None,
            })
            .collect::<Vec<_>>(),
    ];

    let mixture_link_state = options
        .mixture_link
        .clone()
        .as_ref()
        .map(state_fromspec)
        .transpose()
        .map_err(EstimationError::InvalidInput)?;
    let sas_link_state = options
        .sas_link
        .map(|spec| {
            if family.is_binomial_beta_logistic() {
                state_from_beta_logisticspec(spec)
            } else {
                state_from_sasspec(spec)
            }
        })
        .transpose()
        .map_err(EstimationError::InvalidInput)?;
    let latent_cloglog_state = options.latent_cloglog;
    let shared_y = Arc::new(y.to_owned());
    let sharedweights = Arc::new(weights.to_owned());
    let shared_design = baseline
        .design
        .design
        .try_to_dense_arc("spatial adaptive exact hyperfit design")
        .map_err(EstimationError::InvalidInput)?;
    let shared_offset = Arc::new(offset.to_owned());
    let shared_runtime_caches = Arc::new(runtime_caches.to_vec());
    let shared_hyperspecs = Arc::new(hyperspecs.clone());
    let zero_quadratic = Arc::new(Array2::<f64>::zeros((
        baseline.design.design.ncols(),
        baseline.design.design.ncols(),
    )));
    let base_family = SpatialAdaptiveExactFamily {
        family: family.clone(),
        latent_cloglog_state,
        mixture_link_state: mixture_link_state.clone(),
        sas_link_state,
        y: shared_y.clone(),
        weights: sharedweights.clone(),
        design: shared_design.clone(),
        offset: shared_offset.clone(),
        linear_constraints: baseline.design.linear_constraints.clone(),
        runtime_caches: shared_runtime_caches.clone(),
        adaptive_params: Vec::new(),
        fixed_quadratichessian: zero_quadratic.clone(),
        hyperspecs: shared_hyperspecs.clone(),
        exact_eval_cache: Arc::new(Mutex::new(None)),
    };

    let rho_dim = retained_penalties.len();
    let operator_slots_end = rho_dim + runtime_caches.len() * 3;
    // Every slot's box is `initial_theta[idx] ± WINDOW` clamped into a
    // per-slot [floor, cap]. Retained-λ previously used a scale-blind
    // ±30 absolute interval, which on small-n / weakly-identified Duchon
    // fits let those lambdas wander to the exp(-30) floor and produce
    // near-interpolant solutions. Anchoring on baseline log-λ inherits the
    // baseline REML's scale calibration so the overlay can only refine
    // within an exp(±6) ≈ 400× band of the well-posed baseline regime,
    // matching the discipline already applied to operator and epsilon
    // slots.
    const UNIFIED_LOG_WINDOW: f64 = 6.0;
    const RETAINED_LAMBDA_LOG_LOWER_FLOOR: f64 = -30.0;
    const RETAINED_LAMBDA_LOG_UPPER_CAP: f64 = 30.0;
    const OPERATOR_LAMBDA_LOG_LOWER_FLOOR: f64 = -10.0;
    const OPERATOR_LAMBDA_LOG_UPPER_CAP: f64 = 30.0;
    let epsilon_floor_log = adaptive_opts.min_epsilon.max(1e-12).ln();
    let anchored_bound = |idx: usize, sign: f64| -> f64 {
        let raw = initial_theta[idx] + sign * UNIFIED_LOG_WINDOW;
        if idx < rho_dim {
            raw.clamp(
                RETAINED_LAMBDA_LOG_LOWER_FLOOR,
                RETAINED_LAMBDA_LOG_UPPER_CAP,
            )
        } else if idx < operator_slots_end {
            raw.clamp(
                OPERATOR_LAMBDA_LOG_LOWER_FLOOR,
                OPERATOR_LAMBDA_LOG_UPPER_CAP,
            )
        } else {
            raw.max(epsilon_floor_log)
        }
    };
    let eps_lower =
        Array1::from_iter((0..initial_theta.len()).map(|idx| anchored_bound(idx, -1.0)));
    let eps_upper = Array1::from_iter((0..initial_theta.len()).map(|idx| anchored_bound(idx, 1.0)));
    let blockspec = ParameterBlockSpec {
        name: "eta".to_string(),
        design: baseline.design.design.clone(),
        offset: offset.to_owned(),
        penalties: retained_penalties
            .iter()
            .cloned()
            .map(PenaltyMatrix::Dense)
            .collect(),
        nullspace_dims: retained_nullspace_dims.clone(),
        initial_log_lambdas: Array1::from_vec(retained_log_lambdas.clone()),
        initial_beta: Some(baseline.fit.beta.clone()),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let screening_cap = Arc::new(AtomicUsize::new(0));
    let outer_opts = BlockwiseFitOptions {
        inner_max_cycles: options.max_iter,
        inner_tol: options.tol,
        outer_max_iter: options.max_iter,
        outer_tol: options.tol,
        compute_covariance: false,
        screening_max_inner_iterations: Some(Arc::clone(&screening_cap)),
        ..BlockwiseFitOptions::default()
    };

    use crate::solver::rho_optimizer::{
        DeclaredHessianForm, Derivative, HessianResult, OuterEval, OuterProblem,
    };

    struct SpatialAdaptiveOuterState {
        warm_cache: Option<CustomFamilyWarmStart>,
        last_eval: Option<(
            Array1<f64>,
            f64,
            Array1<f64>,
            HessianResult,
            CustomFamilyWarmStart,
        )>,
    }

    let n_theta = initial_theta.len();

    // Clamp theta to the asymmetric epsilon bounds that run_outer's symmetric
    // rho_bound cannot express directly.
    let theta_bounds = Some((eps_lower.clone(), eps_upper.clone()));
    let clamp_theta = {
        let lo = eps_lower;
        let hi = eps_upper;
        move |theta: &Array1<f64>| -> Array1<f64> {
            let mut clamped = theta.clone();
            for i in 0..clamped.len() {
                clamped[i] = clamped[i].clamp(lo[i], hi[i]);
            }
            clamped
        }
    };

    let decode_theta = |theta: &Array1<f64>| -> (Array1<f64>, Vec<SpatialAdaptiveTermHyperParams>) {
        let rho = theta.slice(s![..rho_dim]).to_owned();
        let adaptive_lambda_start = rho_dim;
        let adaptive_lambda_end = adaptive_lambda_start + runtime_caches.len() * 3;
        let eps = [
            theta[adaptive_lambda_end].exp(),
            theta[adaptive_lambda_end + 1].exp(),
            theta[adaptive_lambda_end + 2].exp(),
        ];
        let adaptive_params = runtime_caches
            .iter()
            .enumerate()
            .map(|(cache_idx, _)| SpatialAdaptiveTermHyperParams {
                lambda: [
                    theta[adaptive_lambda_start + cache_idx * 3].exp(),
                    theta[adaptive_lambda_start + cache_idx * 3 + 1].exp(),
                    theta[adaptive_lambda_start + cache_idx * 3 + 2].exp(),
                ],
                epsilon: eps,
            })
            .collect::<Vec<_>>();
        (rho, adaptive_params)
    };
    let analytic_outer_hessian_available =
        crate::custom_family::joint_exact_analytic_outer_hessian_available()
            && base_family
                .exact_outer_derivative_order(std::slice::from_ref(&blockspec), &outer_opts)
                .has_hessian()
            && crate::custom_family::exact_newton_outer_geometry_supports_second_order_solver(
                &base_family,
            );
    let outer_max_iter = crate::custom_family::cost_gated_first_order_max_iter(
        options.max_iter,
        base_family.coefficient_gradient_cost(std::slice::from_ref(&blockspec)),
        analytic_outer_hessian_available,
    );
    if outer_max_iter < options.max_iter {
        log::info!(
            "[OUTER] exact spatial adaptive regularization: first-order work gate reduced outer_max_iter {} -> {}",
            options.max_iter,
            outer_max_iter,
        );
    }
    // Keep the exact outer Hessian whenever the adaptive family can provide it.
    // The Charbonnier pseudo-Laplace surface mixes ordinary log-lambda
    // coordinates with adaptive λ/ε coordinates; exact curvature is the best
    // route when available. If a family cannot provide exact curvature, this
    // builder declares only the true first-order capability.
    let problem = OuterProblem::new(n_theta)
        .with_gradient(Derivative::Analytic)
        .with_hessian(if analytic_outer_hessian_available {
            DeclaredHessianForm::Either
        } else {
            DeclaredHessianForm::Unavailable
        })
        .with_fallback_policy(crate::solver::rho_optimizer::FallbackPolicy::Disabled)
        .with_psi_dim(n_theta.saturating_sub(rho_dim))
        .with_tolerance(options.tol)
        .with_max_iter(outer_max_iter)
        .with_seed_config(crate::seeding::SeedConfig::default())
        .with_screening_cap(Arc::clone(&screening_cap))
        .with_initial_rho(initial_theta.clone());
    let problem = if let Some((lo, hi)) = theta_bounds {
        problem.with_bounds(lo, hi)
    } else {
        problem
    };

    let eval_outer = |st: &mut SpatialAdaptiveOuterState,
                      theta: &Array1<f64>,
                      order: crate::solver::rho_optimizer::OuterEvalOrder|
     -> Result<OuterEval, EstimationError> {
        let theta = clamp_theta(theta);

        if let Some((cached_theta, cached_cost, cached_grad, cached_hess, cached_warm)) =
            &st.last_eval
            && cached_theta.len() == theta.len()
            && cached_theta
                .iter()
                .zip(theta.iter())
                .all(|(&a, &b)| (a - b).abs() <= 1e-12)
            && (!matches!(
                order,
                crate::solver::rho_optimizer::OuterEvalOrder::ValueGradientHessian
            ) || analytic_outer_hessian_available)
        {
            st.warm_cache = Some(cached_warm.clone());
            return Ok(OuterEval {
                cost: *cached_cost,
                gradient: cached_grad.clone(),
                hessian: if matches!(
                    order,
                    crate::solver::rho_optimizer::OuterEvalOrder::ValueGradientHessian
                ) && analytic_outer_hessian_available
                {
                    cached_hess.clone()
                } else {
                    HessianResult::Unavailable
                },
                inner_beta_hint: None,
            });
        }

        let (rho, adaptive_params) = decode_theta(&theta);
        let family_eval = base_family.with_adaptive_params(adaptive_params, zero_quadratic.clone());
        let need_hessian = matches!(
            order,
            crate::solver::rho_optimizer::OuterEvalOrder::ValueGradientHessian
        ) && analytic_outer_hessian_available;
        let result = evaluate_custom_family_joint_hyper(
            &family_eval,
            std::slice::from_ref(&blockspec),
            &outer_opts,
            &rho,
            &derivative_blocks,
            st.warm_cache.as_ref(),
            if need_hessian {
                crate::solver::estimate::reml::reml_outer_engine::EvalMode::ValueGradientHessian
            } else {
                crate::solver::estimate::reml::reml_outer_engine::EvalMode::ValueAndGradient
            },
        )
        .map_err(|e| {
            EstimationError::RemlOptimizationFailed(format!("spatial adaptive eval failed: {e}"))
        })?;
        if !result.inner_converged {
            st.warm_cache = Some(result.warm_start.clone());
            return Err(EstimationError::RemlOptimizationFailed(
                "exact spatial adaptive inner solve did not converge".to_string(),
            ));
        }
        if !result.objective.is_finite() || result.gradient.iter().any(|v| !v.is_finite()) {
            return Err(EstimationError::RemlOptimizationFailed(
                "exact spatial adaptive objective returned non-finite values".to_string(),
            ));
        }
        let hessian_result = if need_hessian {
            if !result.outer_hessian.is_analytic() {
                return Err(EstimationError::RemlOptimizationFailed(
                    "exact spatial adaptive objective did not return an exact outer Hessian"
                        .to_string(),
                ));
            }
            match result.outer_hessian.dim() {
                Some(dim) if dim == theta.len() => {}
                Some(dim) => {
                    return Err(EstimationError::RemlOptimizationFailed(format!(
                        "exact spatial adaptive outer Hessian dimension mismatch: got {dim}, expected {}",
                        theta.len(),
                    )));
                }
                None => {
                    return Err(EstimationError::RemlOptimizationFailed(
                        "exact spatial adaptive objective did not report an outer Hessian dimension"
                            .to_string(),
                    ));
                }
            }
            st.last_eval = Some((
                theta.clone(),
                result.objective,
                result.gradient.clone(),
                result.outer_hessian.clone(),
                result.warm_start.clone(),
            ));
            result.outer_hessian
        } else {
            HessianResult::Unavailable
        };
        st.warm_cache = Some(result.warm_start);
        Ok(OuterEval {
            cost: result.objective,
            gradient: result.gradient,
            hessian: hessian_result,
            inner_beta_hint: None,
        })
    };

    let mut obj = problem.build_objective_with_screening_proxy(
        SpatialAdaptiveOuterState {
            warm_cache: None,
            last_eval: None,
        },
        |st: &mut SpatialAdaptiveOuterState, theta: &Array1<f64>| {
            let theta = clamp_theta(theta);
            let (rho, adaptive_params) = decode_theta(&theta);
            let family_eval =
                base_family.with_adaptive_params(adaptive_params, zero_quadratic.clone());
            let result = evaluate_custom_family_joint_hyper(
                &family_eval,
                std::slice::from_ref(&blockspec),
                &outer_opts,
                &rho,
                &derivative_blocks,
                st.warm_cache.as_ref(),
                crate::solver::estimate::reml::reml_outer_engine::EvalMode::ValueOnly,
            )
            .map_err(|e| {
                EstimationError::RemlOptimizationFailed(format!(
                    "spatial adaptive cost eval failed: {e}"
                ))
            })?;
            if !result.inner_converged {
                st.warm_cache = Some(result.warm_start);
                return Err(EstimationError::RemlOptimizationFailed(
                    "exact spatial adaptive cost inner solve did not converge".to_string(),
                ));
            }
            st.warm_cache = Some(result.warm_start);
            Ok(result.objective)
        },
        |st: &mut SpatialAdaptiveOuterState, theta: &Array1<f64>| {
            eval_outer(
                st,
                theta,
                if analytic_outer_hessian_available {
                    crate::solver::rho_optimizer::OuterEvalOrder::ValueGradientHessian
                } else {
                    crate::solver::rho_optimizer::OuterEvalOrder::ValueAndGradient
                },
            )
        },
        |st: &mut SpatialAdaptiveOuterState,
         theta: &Array1<f64>,
         order: crate::solver::rho_optimizer::OuterEvalOrder| {
            eval_outer(st, theta, order)
        },
        Some(|st: &mut SpatialAdaptiveOuterState| {
            st.warm_cache = None;
            st.last_eval = None;
        }),
        Some(|st: &mut SpatialAdaptiveOuterState, theta: &Array1<f64>| {
            let theta = clamp_theta(theta);
            let (rho, adaptive_params) = decode_theta(&theta);
            let family_eval =
                base_family.with_adaptive_params(adaptive_params, zero_quadratic.clone());
            let result = evaluate_custom_family_joint_hyper_efs(
                &family_eval,
                std::slice::from_ref(&blockspec),
                &outer_opts,
                &rho,
                &derivative_blocks,
                st.warm_cache.as_ref(),
            )
            .map_err(|e| {
                EstimationError::RemlOptimizationFailed(format!(
                    "spatial adaptive EFS eval failed: {e}"
                ))
            })?;
            if !result.inner_converged {
                st.warm_cache = Some(result.warm_start);
                return Err(EstimationError::RemlOptimizationFailed(
                    "exact spatial adaptive EFS inner solve did not converge".to_string(),
                ));
            }
            st.warm_cache = Some(result.warm_start);
            Ok(result.efs_eval)
        }),
        // Seed-screening ranking proxy (#969). The regular cost closure
        // above hard-errors on a non-converged inner solve — correct for
        // line-search costs, but under the screening cap
        // (`screening_max_inner_iterations`, wired into `outer_opts`) the
        // inner solve is truncated BY DESIGN, so screening through that
        // closure rejects every seed and re-creates the all-seeds-rejected
        // front-door failure genus. Screening only RANKS candidates: the
        // penalized objective of the capped solve is a meaningful ranking
        // signal even unconverged (the same contract as the custom-family
        // labeled proxy), so accept it and let the cascade pick the best
        // seed; the selected seed is then fit with the full budget.
        |st: &mut SpatialAdaptiveOuterState, theta: &Array1<f64>| {
            let theta = clamp_theta(theta);
            let (rho, adaptive_params) = decode_theta(&theta);
            let family_eval =
                base_family.with_adaptive_params(adaptive_params, zero_quadratic.clone());
            let result = evaluate_custom_family_joint_hyper(
                &family_eval,
                std::slice::from_ref(&blockspec),
                &outer_opts,
                &rho,
                &derivative_blocks,
                st.warm_cache.as_ref(),
                crate::solver::estimate::reml::reml_outer_engine::EvalMode::ValueOnly,
            )
            .map_err(|e| {
                EstimationError::RemlOptimizationFailed(format!(
                    "spatial adaptive screening eval failed: {e}"
                ))
            })?;
            st.warm_cache = Some(result.warm_start);
            Ok(result.objective)
        },
    );

    let outer_result = problem
        .run(&mut obj, "exact spatial adaptive regularization")
        .map_err(|e| {
            EstimationError::InvalidInput(format!(
                "exact spatial adaptive outer optimization failed: {e}"
            ))
        })?;
    if !outer_result.converged {
        // The strict absolute-floor gradient criterion (`‖g‖_proj ≤ options.tol`)
        // is too tight near the box-constrained boundary of the adaptive
        // Charbonnier pseudo-Laplace objective: as the optimizer pushes ε → ∞
        // (overlay-disabled corner), λ → λ_min, the Hessian's nearly-null
        // direction lets Cauchy/Newton accept ~e-3-magnitude probe steps that
        // give cost changes well below 6-digit precision, and the projected
        // gradient floors at numerical-noise-scale (≈ 5e-6 for n≈500, cost≈
        // 3e2 fits in double precision) rather than at 0. Accept the iterate
        // when the mgcv-style relative-to-cost criterion ‖g‖_proj ≤ τ·(1+|f|)
        // is satisfied — that is the textbook REML convergence rule and is
        // exactly what `opt::GradientTolerance::relative_to_cost(τ)` would
        // have enforced if this OuterProblem path had wired it through. The
        // strict absolute floor is retained as the primary check; the
        // rel-to-cost form only kicks in once the absolute one has timed out
        // at `max_iter`, so unconverged divergent runs (which have large |g|)
        // still surface as errors.
        let rel_to_cost_threshold = options.tol * (1.0_f64 + outer_result.final_value.abs());
        // Rel-to-cost acceptance requires an actual gradient measurement;
        // `None` (cache-hit short-circuit, gradient-free path) cannot satisfy
        // the mgcv-style criterion regardless of magnitude.
        if let Some(final_grad) = outer_result
            .final_grad_norm
            .filter(|v| v.is_finite() && *v <= rel_to_cost_threshold)
        {
            log::info!(
                "[spatial-adaptive] outer optimization hit max_iter={} but \
                 projected gradient norm {:.3e} ≤ τ·(1+|f|) = {:.3e} \
                 (τ={:.3e}, |f|={:.3e}); accepting iterate under the mgcv-style \
                 relative-to-cost REML convergence criterion.",
                outer_result.iterations,
                final_grad,
                rel_to_cost_threshold,
                options.tol,
                outer_result.final_value.abs(),
            );
        } else {
            crate::bail_invalid_estim!(
                "exact spatial adaptive outer optimization did not converge after {} iterations (final_objective={:.6e}, final_grad_norm={})",
                outer_result.iterations,
                outer_result.final_value,
                outer_result.final_grad_norm_report(),
            );
        }
    }
    let outer_iterations = outer_result.iterations;
    // `None` = no gradient measurement (cache-hit / gradient-free); the
    // authoritative convergence signal is `outer_converged`.
    let outer_grad_norm: Option<f64> = outer_result.final_grad_norm;
    let theta_star = outer_result.rho;
    let rho_star = theta_star.slice(s![..rho_dim]).to_owned();
    let adaptive_lambda_start = rho_dim;
    let adaptive_lambda_end = adaptive_lambda_start + runtime_caches.len() * 3;
    let eps_star = [
        theta_star[adaptive_lambda_end].exp(),
        theta_star[adaptive_lambda_end + 1].exp(),
        theta_star[adaptive_lambda_end + 2].exp(),
    ];
    let adaptive_params = runtime_caches
        .iter()
        .enumerate()
        .map(|(cache_idx, _)| SpatialAdaptiveTermHyperParams {
            lambda: [
                theta_star[adaptive_lambda_start + cache_idx * 3].exp(),
                theta_star[adaptive_lambda_start + cache_idx * 3 + 1].exp(),
                theta_star[adaptive_lambda_start + cache_idx * 3 + 2].exp(),
            ],
            epsilon: eps_star,
        })
        .collect::<Vec<_>>();
    let mut fixed_total = Array2::<f64>::zeros((
        baseline.design.design.ncols(),
        baseline.design.design.ncols(),
    ));
    for (idx, penalty) in retained_penalties.iter().enumerate() {
        fixed_total.scaled_add(rho_star[idx].exp(), penalty);
    }
    let final_family =
        base_family.with_adaptive_params(adaptive_params.clone(), Arc::new(fixed_total.clone()));
    let final_blockspec = ParameterBlockSpec {
        name: "eta".to_string(),
        design: baseline.design.design.clone(),
        offset: offset.to_owned(),
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(baseline.fit.beta.clone()),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let final_fit = fit_custom_family(
        &final_family,
        &[final_blockspec],
        &BlockwiseFitOptions {
            inner_max_cycles: options.max_iter,
            inner_tol: options.tol,
            outer_max_iter: 1,
            outer_tol: options.tol,
            compute_covariance: true,
            ..BlockwiseFitOptions::default()
        },
    )
    .map_err(EstimationError::CustomFamily)?;
    let beta = final_fit.block_states[0].beta.clone();
    let final_eval = final_family
        .exact_evaluation(&beta)
        .map_err(EstimationError::InvalidInput)?;
    let penalized_hessian = final_eval
        .totalobjectivehessian(&final_family.design)
        .map_err(EstimationError::InvalidInput)?;
    let beta_covariance = final_fit.covariance_conditional.clone();
    let beta_standard_errors = beta_covariance
        .as_ref()
        .map(|cov| Array1::from_iter((0..cov.nrows()).map(|i| cov[[i, i]].max(0.0).sqrt())));

    let mut full_lambdas = baseline.fit.lambdas.clone();
    for (idx, &global_idx) in retained_global_indices.iter().enumerate() {
        full_lambdas[global_idx] = rho_star[idx].exp();
    }
    for (cache_idx, cache) in runtime_caches.iter().enumerate() {
        full_lambdas[cache.mass_penalty_global_idx] = adaptive_params[cache_idx].lambda[0];
        full_lambdas[cache.tension_penalty_global_idx] = adaptive_params[cache_idx].lambda[1];
        full_lambdas[cache.stiffness_penalty_global_idx] = adaptive_params[cache_idx].lambda[2];
    }

    let deviance = if family.is_gaussian_identity() {
        y.iter()
            .zip(final_eval.obs.mu.iter())
            .zip(weights.iter())
            .map(|((&yy, &mu), &w)| w.max(0.0) * (yy - mu) * (yy - mu))
            .sum()
    } else {
        -2.0 * final_eval.obs.log_likelihood
    };
    let mut local_penalty_blocks =
        Vec::<PenaltySpec>::with_capacity(baseline.design.penalties.len());
    for (global_idx, bp) in baseline.design.penalties.iter().enumerate() {
        if adaptive_penalty_indices.contains(&global_idx) {
            let cache = runtime_caches
                .iter()
                .find(|cache| {
                    cache.mass_penalty_global_idx == global_idx
                        || cache.tension_penalty_global_idx == global_idx
                        || cache.stiffness_penalty_global_idx == global_idx
                })
                .ok_or_else(|| {
                    EstimationError::InvalidInput(format!(
                        "missing runtime cache for adaptive penalty index {global_idx}"
                    ))
                })?;
            let cache_idx = runtime_caches
                .iter()
                .position(|c| {
                    c.mass_penalty_global_idx == global_idx
                        || c.tension_penalty_global_idx == global_idx
                        || c.stiffness_penalty_global_idx == global_idx
                })
                .ok_or_else(|| {
                    EstimationError::InvalidInput(format!(
                        "missing adaptive cache position for penalty index {global_idx}"
                    ))
                })?;
            let state = &final_eval.adaptive_states[cache_idx];
            let local = if cache.mass_penalty_global_idx == global_idx {
                scalar_operatorhessian(&cache.d0, &state.magnitude.betahessian_diag())
                    .mapv(|v| adaptive_params[cache_idx].lambda[0] * v)
            } else if cache.tension_penalty_global_idx == global_idx {
                grouped_operatorhessian(
                    &cache.d1,
                    cache.dimension,
                    &state.gradient.betahessian_blocks(),
                )?
                .mapv(|v| adaptive_params[cache_idx].lambda[1] * v)
            } else {
                grouped_operatorhessian(
                    &cache.d2,
                    cache.dimension * cache.dimension,
                    &state.curvature.betahessian_blocks(),
                )?
                .mapv(|v| adaptive_params[cache_idx].lambda[2] * v)
            };
            // Wrap the pre-scaled global penalty matrix as PenaltySpec::Dense.
            local_penalty_blocks.push(PenaltySpec::Dense(penalty_matrixwith_local_block(
                baseline.design.design.ncols(),
                cache.coeff_global_range.clone(),
                &local,
            )));
        } else {
            local_penalty_blocks.push(PenaltySpec::Dense(
                bp.to_global(p_total).mapv(|v| v * full_lambdas[global_idx]),
            ));
        }
    }
    let (edf_by_block, penalty_block_trace, edf_total) = if let Some(cov) = beta_covariance.as_ref()
    {
        exact_bounded_edf(
            &local_penalty_blocks,
            &Array1::from_elem(local_penalty_blocks.len(), 1.0),
            cov,
        )?
    } else {
        (
            vec![0.0; local_penalty_blocks.len()],
            vec![0.0; local_penalty_blocks.len()],
            0.0,
        )
    };
    let stable_penalty_term =
        2.0 * final_eval.adaptive_penalty_value + beta.dot(&fixed_total.dot(&beta));
    let standard_deviation = if family.is_gaussian_identity() {
        let denom = (y.len() as f64 - edf_total).max(1.0);
        (deviance / denom).sqrt()
    } else {
        1.0
    };
    let maps = compute_spatial_adaptiveweights_for_beta(
        &beta,
        runtime_caches,
        eps_star[0],
        eps_star[1],
        eps_star[2],
        adaptive_opts.weight_floor,
        adaptive_opts.weight_ceiling,
        // Working-Laplace conditional covariance Sigma_beta = H^{-1} from the
        // final exact-family solve, reused here as the posterior-SNR variance
        // source (no second factorization is formed).
        beta_covariance.as_ref(),
    )?
    .into_iter()
    .zip(runtime_caches.iter())
    .map(|(w, cache)| AdaptiveSpatialMap {
        termname: cache.termname.clone(),
        feature_cols: cache.feature_cols.clone(),
        collocation_points: cache.collocation_points.clone(),
        inv_magweight: w.inv_magweight,
        invgradweight: w.invgradweight,
        inv_lapweight: w.inv_lapweight,
    })
    .collect::<Vec<_>>();
    let fitted_link = if family.is_latent_cloglog() {
        FittedLinkState::LatentCLogLog {
            state: latent_cloglog_state
                .expect("BinomialLatentCLogLog requires an explicit latent-cloglog state"),
        }
    } else if family.is_binomial_mixture() {
        mixture_link_state
            .clone()
            .map(|state| FittedLinkState::Mixture {
                state,
                covariance: None,
            })
            .unwrap_or(FittedLinkState::Standard(None))
    } else if family.is_binomial_sas() {
        sas_link_state
            .map(|state| FittedLinkState::Sas {
                state,
                covariance: None,
            })
            .unwrap_or(FittedLinkState::Standard(None))
    } else if family.is_binomial_beta_logistic() {
        sas_link_state
            .map(|state| FittedLinkState::BetaLogistic {
                state,
                covariance: None,
            })
            .unwrap_or(FittedLinkState::Standard(None))
    } else {
        FittedLinkState::Standard(None)
    };
    let max_abs_eta = final_eval
        .obs
        .eta
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let fitted = FittedTermCollection {
        fit: {
            let log_lambdas = full_lambdas.mapv(|v| v.max(1e-300).ln());
            let inf = FitInference {
                edf_by_block,
                penalty_block_trace,
                edf_total,
                smoothing_correction: None,
                // Boundary adapter: wrap the raw `Array2<f64>` Hessian as
                // `UnscaledPrecision` for the newtype storage.
                penalized_hessian: penalized_hessian.clone().into(),
                working_weights: final_eval.obs.fisherweight.clone(),
                working_response: {
                    let mut out = final_eval.obs.eta.clone();
                    for i in 0..out.len() {
                        let wi = final_eval.obs.fisherweight[i].max(1e-12);
                        out[i] += final_eval.obs.score[i] / wi;
                    }
                    out
                },
                reparam_qs: None,
                dispersion: crate::estimate::Dispersion::Known(1.0),
                beta_covariance: beta_covariance
                    .clone()
                    .map(crate::inference::dispersion_cov::PhiScaledCovariance::from),
                beta_standard_errors,
                beta_covariance_corrected: None,
                beta_standard_errors_corrected: None,
                beta_covariance_frequentist: None,
                coefficient_influence: None,
                weighted_gram: None,
                bias_correction_beta: None,
            };
            let geometry = Some(crate::estimate::FitGeometry {
                penalized_hessian: penalized_hessian.into(),
                working_weights: inf.working_weights.clone(),
                working_response: inf.working_response.clone(),
            });
            let covariance_conditional = beta_covariance;
            let pirls_status_val = if final_fit.outer_converged {
                crate::pirls::PirlsStatus::Converged
            } else {
                crate::pirls::PirlsStatus::StalledAtValidMinimum
            };
            UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
                blocks: vec![crate::estimate::FittedBlock {
                    beta: beta.clone(),
                    role: crate::estimate::BlockRole::Mean,
                    edf: edf_total,
                    lambdas: full_lambdas.clone(),
                }],
                log_lambdas,
                lambdas: full_lambdas,
                likelihood_scale: family.default_scale_metadata(),
                likelihood_family: Some(family),
                log_likelihood_normalization:
                    crate::types::LogLikelihoodNormalization::UserProvided,
                log_likelihood: final_eval.obs.log_likelihood,
                deviance,
                reml_score: final_fit.penalized_objective,
                stable_penalty_term,
                penalized_objective: final_fit.penalized_objective,
                used_device: false,
                outer_iterations,
                outer_converged: final_fit.outer_converged,
                outer_gradient_norm: outer_grad_norm,
                standard_deviation,
                covariance_conditional,
                covariance_corrected: None,
                inference: Some(inf),
                fitted_link,
                geometry,
                block_states: Vec::new(),
                pirls_status: pirls_status_val,
                max_abs_eta,
                constraint_kkt: None,
                artifacts: crate::estimate::FitArtifacts {
                    pirls: None,
                    ..Default::default()
                },
                inner_cycles: 0,
            })?
        },
        design: baseline.design,
        adaptive_diagnostics: Some(AdaptiveRegularizationDiagnostics {
            epsilon_0: eps_star[0],
            epsilon_g: eps_star[1],
            epsilon_c: eps_star[2],
            epsilon_outer_iterations: outer_iterations,
            mm_iterations: 0,
            converged: final_fit.outer_converged,
            maps,
        }),
    };
    enforce_term_constraint_feasibility(&fitted.design, &fitted.fit)?;
    Ok(fitted)
}

/// Relax the per-coordinate ρ-prior for terms running in Marra–Wood
/// double-penalty selection mode (#1266).
///
/// The default ρ-prior is a `Normal { mean: 0, sd: 3 }` cap on each log-λ — a
/// stabiliser that keeps ordinary smoothing parameters from drifting to
/// degenerate extremes (gam#893/#1196). For a smooth carrying a
/// `DoublePenaltyNullspace` block (`double_penalty = True`, the default `s(...)`
/// — analogous to mgcv `select = TRUE`) that cap is actively wrong: the whole
/// purpose of the second penalty is to let REML drive an *unsupported* term to
/// `EDF → 0`, which needs both the wiggliness and null-space log-λ to grow
/// large. The `ρ²/(2·9)` cap pulls them back toward 0, so REML settles at a
/// point that leaves the term under-shrunk — the smooth's EDF comes out ABOVE
/// the single-penalty (`double_penalty = False`) EDF instead of at or below it,
/// the exact contract violation in #1266. mgcv's `select = TRUE` applies no
/// such cap to the selection coordinates, and the lower-level term-collection
/// fits already converge correctly under a flat prior.
///
/// We therefore rewrite the prior to `Independent`, holding the base prior on
/// every ordinary coordinate but switching the coordinates of any
/// double-penalty term to `Flat`. Single-penalty terms are byte-for-byte
/// unchanged, and an already-`Flat`/already-`Independent` base prior, or a
/// design with no double-penalty block, is returned untouched.
///
/// Cap-lifting to `Flat`/wide-`Normal` is a Gaussian-identity recovery device
/// and stays gated to that case. The `DoublePenaltyNullspace` *selection* prior
/// — the one-sided select-out barrier that drives an unsupported null space out
/// (mgcv `select=TRUE`) — is family-agnostic, so it is applied for non-Gaussian
/// families too (e.g. the #1426 Gamma/log overfit) whenever the inner ρ is
/// length-safe (no SAS/mixture link-shape coordinate, no moving spatial κ); the
/// range-space coordinate of such a non-Gaussian smooth keeps its base cap.
fn relax_smoothing_rho_prior(
    options: &FitOptions,
    family: &LikelihoodSpec,
    design: &TermCollectionDesign,
) -> crate::types::RhoPrior {
    use crate::terms::basis::BasisMetadata;
    let base = &options.rho_prior;
    // Only a single scalar prior that actually caps log-λ needs relaxing;
    // `Flat` already imposes no cap and `Independent` is assumed caller-built.
    if matches!(
        base,
        crate::types::RhoPrior::Flat | crate::types::RhoPrior::Independent(_)
    ) {
        return base.clone();
    }
    // LENGTH SAFETY (load-bearing). The per-coordinate `Independent` prior is
    // validated against the FULL outer ρ vector and a length disagreement
    // saturates the prior to `+∞`, breaking the fit. The ρ vector this prior is
    // attached to (the inner REML fit at a *fixed* realized design) aligns 1:1
    // with the penalty blocks in `design.penaltyinfo` ONLY when the fit
    // introduces no auxiliary trailing ρ coordinates. Such coordinates come from
    //   * non-Gaussian dispersion / non-identity link machinery,
    //   * SAS ε/δ and mixture-link parameters,
    //   * spatial κ length-scale optimisation that actually moves κ.
    // Gate to the Gaussian-identity, link-aux-free case. Spatial κ optimisation
    // (Matérn / Duchon / sphere / curvature / measure-jet) genuinely appends a
    // moving log-κ coordinate AND needs the cap to stabilise it, so bail if any
    // such term is present. Thin-plate is the exception: its length-scale is a
    // pure radial SCALE that REML cannot identify (the κ optimiser converges to
    // a no-op, leaving `n_params = penalty-block count`), so it adds no trailing
    // coordinate and is safe to relax alongside the B-spline family.
    let gaussian_identity = matches!(family.response, crate::types::ResponseFamily::Gaussian)
        && matches!(
            family.link,
            crate::types::InverseLink::Standard(crate::types::StandardLink::Identity)
        );
    let has_link_aux = options.sas_link.is_some()
        || options.optimize_sas
        || options.mixture_link.is_some()
        || options.optimize_mixture;
    let has_moving_kappa = design.smooth.terms.iter().any(|t| {
        matches!(
            t.metadata,
            BasisMetadata::Matern { .. }
                | BasisMetadata::Duchon { .. }
                | BasisMetadata::Sphere { .. }
                | BasisMetadata::SphereHarmonics { .. }
                | BasisMetadata::ConstantCurvature { .. }
                | BasisMetadata::MeasureJet { .. }
        )
    });
    // LENGTH SAFETY decides only whether the inner ρ aligns 1:1 with the penalty
    // blocks (so an `Independent` prior is valid): it is broken by SAS/mixture
    // link-shape coordinates and by a moving spatial κ, NOT by the response
    // family or link per se. A Gamma/log (or any other non-Gaussian) GAM with no
    // link-aux and no moving κ has exactly `penaltyinfo.len()` ρ coordinates, so
    // the `DoublePenaltyNullspace` selection prior below is length-safe there too.
    let length_safe = !has_link_aux && !has_moving_kappa;
    if !length_safe {
        return base.clone();
    }
    let coords = &design.penaltyinfo;
    if coords.is_empty() {
        return base.clone();
    }
    // WELL-IDENTIFICATION GATE (#1089). The ρ-prior is two things at once: a
    // #1266/#1271-harmful symmetric cap on each smoothing log-λ, AND a
    // #1089-load-bearing stabiliser that makes the outer REML loop terminate on
    // an *under-determined* design (gam#893/#1196/#1089: the n=30 five-`ps` wine
    // fit has p ≈ 51 > n, so without the cap's curvature the outer criterion is
    // flat/degenerate in ρ-space and the loop never certifies a stationary
    // point). Only lift the cap when the data comfortably over-determines the
    // model (`n ≥ 2·p`), so the unregularised REML problem is well-posed on its
    // own; otherwise keep the base prior. The #1266/#1271 cases (n ≈ 800,
    // p ≈ 20–40) clear this by ≥20×; the #1089 wine fit (n < p) keeps its cap.
    let n_obs = design.design.nrows();
    let p_total = design.design.ncols();
    // REGIME of the relaxed prior on the relaxable smooth coordinates.
    //
    // * WELL-DETERMINED (`n ≥ 2·p`): the unregularised REML problem is well
    //   posed on its own, so the relaxable coordinates are freed to `Flat`,
    //   which the runtime resolves to the firth one-sided barrier — byte-flat
    //   on the identified side (pure REML, exactly mgcv) and only a convex wall
    //   against the `λ → 0` degeneracy. This is the #1266/#1271 behaviour.
    //
    // * UNDER-DETERMINED (`n < 2·p`): the design does NOT over-determine the
    //   model (the n≈26 five-`ps` wine fit has p > n), so the firth barrier's
    //   zero curvature on the identified side leaves the outer REML criterion
    //   flat/degenerate in ρ-space and the loop hits `max_iter` at whatever
    //   (under-smoothed) λ it last held — EDF rails up to ≈n, the smooths
    //   interpolate the training rows, and held-out prediction explodes
    //   (#1392: held-out R² as low as −2.5e6 on `wine_gamair`). The previous
    //   stabiliser kept the FULL base prior here — a symmetric
    //   `Normal{mean:0, sd:3}` cap. Its `ρ²/(2·9)` curvature does terminate the
    //   loop, but it is centred at λ=1 with a tight `sd=3`: at the REML optimum
    //   `ρ* ≈ 8–15` (heavy smoothing, which an over-parameterised fit needs and
    //   which mgcv's pure REML reaches), the cap's `ρ*/9` gradient drags λ back
    //   down by `O(1)` in ρ, pinning the fit in the under-smoothed regime.
    //
    //   The fix keeps a stabiliser with strictly positive curvature (so the
    //   loop still certifies a stationary point — the #1089 requirement) but
    //   WIDENS it to `sd = RELAX_UNDERDETERMINED_RHO_SD` so its gradient drag at
    //   the heavily-smoothed optimum is negligible (`ρ*/sd² = O(1/100)`) and
    //   pure REML — not the prior — chooses λ. The wide symmetric Gaussian is
    //   weakly informative: ±2σ spans the whole feasible ρ range (`|ρ| ≤ 30`),
    //   so it adds termination curvature without biasing which λ REML lands on,
    //   restoring the mgcv-like heavy smoothing on the over-parameterised fit.
    let underdetermined = n_obs < 2 * p_total;
    // Relaxable terms: penalized smooths whose smoothing log-λ the symmetric cap
    // wrongly bounds when the term's signal lives in its penalty null space — a
    // straight line under a bending penalty drives λ → ∞ but the cap pulls it
    // back, leaving spurious wiggle. mgcv caps neither. This is exactly the
    // B-spline family (`ps`/`cr`/`cs`/`bs`, BSpline1D), thin-plate (`tp`), and
    // tensor-B-spline (`te`/`ti`) smooths — single- AND double-penalty (#1266 is
    // the double-penalty case, #1271 the single-penalty `tp`/`ps`). EVERY penalty
    // coordinate such a term owns (bending wiggliness AND any null-space
    // shrinkage) is freed to `Flat`, which the runtime resolves to the
    // firth-default one-sided barrier: no high-λ cap, but still a convex wall
    // against the `λ → 0` under-smoothing degeneracy.
    let relaxable_terms: std::collections::HashSet<&str> = design
        .smooth
        .terms
        .iter()
        .filter(|t| {
            matches!(
                t.metadata,
                BasisMetadata::BSpline1D { .. }
                    | BasisMetadata::ThinPlate { .. }
                    | BasisMetadata::TensorBSpline { .. }
            )
            // SHAPE-CONSTRAINED terms must KEEP the cap (#1380). A monotone /
            // convex / concave smooth carries linear-inequality constraints; at
            // the active boundary (e.g. a convex fit pinned at 2nd-diff = 0) the
            // active set collapses the penalized subspace onto the bending
            // penalty's own null space ({1, x}), where the smoothing log-λ is
            // UNIDENTIFIED. Lifting the cap to `Flat` there lets REML rail λ to
            // `RHO_BOUND` (zero curvature → the smooth collapses to a flat/linear
            // fit, R² ≈ 0 on data the constraint is correct for). The constraint
            // already regularizes the term, and the symmetric cap is the
            // #1089-style stabiliser that pins the unidentified λ — so a
            // shape-constrained term needs the cap KEPT, exactly the
            // under-determined case this gate protects. (Unconstrained #1266/#1271
            // selection terms still relax.)
            && matches!(t.shape, crate::terms::smooth::ShapeConstraint::None)
        })
        .map(|t| t.name.as_str())
        .collect();
    let any_relaxed = coords.iter().any(|info| {
        info.termname
            .as_deref()
            .is_some_and(|name| relaxable_terms.contains(name))
    });
    if !any_relaxed {
        return base.clone();
    }
    // Relaxed prior for a relaxable smooth coordinate, chosen by regime (see the
    // block above): the firth one-sided barrier (`Flat`) when the fit is
    // well-determined, a wide-but-curved symmetric Gaussian when it is
    // under-determined and the loop still needs termination curvature.
    let relaxed_prior = if underdetermined {
        crate::types::RhoPrior::Normal {
            mean: 0.0,
            sd: RELAX_UNDERDETERMINED_RHO_SD,
        }
    } else {
        crate::types::RhoPrior::Flat
    };
    // DOUBLE-PENALTY NULL-SPACE SELECTION (#1392, mgcv `select=TRUE`). A
    // double-penalty smooth carries a second `DoublePenaltyNullspace` ridge on
    // the term's penalty null space ({1, x} for a 1-D bend) whose only job is
    // selection: drive its λ → ∞ to shrink the null-space (linear) component
    // OUT when the data does not support it, exactly as mgcv's `select=TRUE`
    // adds a null-space penalty. On an over-parameterized `p > n` fit
    // (`wine_gamair`: 5 `ps` smooths on ~26 rows) the symmetric relaxed prior
    // above leaves this ridge's outer score flat on the select-out side, so REML
    // stalls it at λ ≈ 0.11 — the null space is kept, the EDF rails up, and
    // held-out prediction collapses (#1392). The RANGE-space (`Primary`) bending
    // coordinate's smoothing selection must NOT be touched, so this select-out
    // bias is gated to `DoublePenaltyNullspace` coordinates only and is applied
    // ONLY in the under-determined regime — in the well-determined regime the
    // relaxable coordinates stay byte-flat (`Flat`) so a clean `n > p` fit is
    // unchanged (no regression on ordinary smooth recovery).
    //
    // The select-out bias is the one-sided penalized-complexity barrier: it
    // walls off the `λ → 0` (null-space kept, wiggly) under-smoothing side and
    // leaves only the gentle bounded Occam pull (gradient → +1/2) toward
    // `λ → ∞` on the over-smoothing side, so REML keeps the null space ONLY when
    // the data clearly earns it. Its curvature `(θ/4) e^{-ρ/2}` is strictly
    // positive everywhere, so the outer loop still certifies a stationary point
    // on the `p > n` design (the #1089 requirement); at a well-earned (large-λ)
    // null space the pull is the same O(1/n) shift as any Occam prior, so the
    // bias is negligible exactly where the null space is identified.
    // NON-GAUSSIAN FAMILIES (#1426 / #1477). The base prior is the same
    // Gaussian-tuned symmetric `Normal{0, sd}` cap, but it is NOT a non-Gaussian
    // smoothing-selection device: it is a #1089 termination-curvature stabiliser
    // centred at λ=1. Keeping it on a length-safe non-Gaussian smooth has two
    // failure modes, both fixed here:
    //   * The `DoublePenaltyNullspace` selection ridge — family-agnostic, exists
    //     ONLY to drive the linear/constant null-space component OUT (mgcv
    //     `select=TRUE`) — was left un-pushed under Gamma/log REML, so REML's
    //     interior minimum sat at a near-full-basis overfit (EDF≈24 vs mgcv≈8 on
    //     the #1426 DGP). It now gets the one-sided select-out PC prior in BOTH
    //     determinacy regimes (the #1426 overfit is well-determined, n≫p).
    //   * The BENDING (range-space) coordinate had its symmetric cap dragging the
    //     fitted log-λ off the REML optimum mgcv reaches, shipping a
    //     systematically biased mean with a hard right-boundary blow-up (#1477:
    //     Tweedie + default `ps` predicts ≈2.4× truth at x=1, while gam's own `cr`
    //     and mgcv's same `ps` both recover truth). It now uses the SAME
    //     `relaxed_prior` regime rule as the Gaussian path — `Flat` (firth
    //     one-sided barrier, byte-flat on the identified side = pure REML = mgcv)
    //     when well-determined, the wide #1089 `Normal` when under-determined.
    // Freeing the bending coordinate is NOT a re-tuning toward a particular λ: it
    // REMOVES an unjustified cap and lets pure REML choose, exactly as on the
    // already-relaxed Gaussian path; the firth barrier still walls off the
    // `λ → 0` under-smoothing degeneracy.
    let nullspace_select_prior = crate::types::RhoPrior::PenalizedComplexity {
        upper: NULLSPACE_SELECT_PC_UPPER,
        tail_prob: NULLSPACE_SELECT_PC_TAIL_PROB,
    };
    let per_coord = coords
        .iter()
        .map(|info| {
            let relax = info
                .termname
                .as_deref()
                .is_some_and(|name| relaxable_terms.contains(name));
            if !relax {
                return base.clone();
            }
            let is_nullspace =
                matches!(info.penalty.source, PenaltySource::DoublePenaltyNullspace);
            if !gaussian_identity {
                // NON-GAUSSIAN length-safe smooth. Select the double-penalty
                // null space OUT (#1426), and free the BENDING (range-space)
                // coordinate by the same regime rule the Gaussian path uses
                // (#1477). The symmetric `Normal{0, sd}` base cap is a
                // Gaussian-tuned #1089 termination stabiliser centred at λ=1 with
                // NO smoothing-selection justification on a non-Gaussian smooth:
                // keeping it dragged the bending log-λ off the REML optimum mgcv
                // reaches, shipping a systematically biased mean with a hard
                // right-boundary blow-up (Tweedie + default `ps`: pred ≈ 2.4×
                // truth at x=1, while gam's own `cr` and mgcv's same `ps` both
                // recover truth). Freeing it to `relaxed_prior` resolves to the
                // firth one-sided barrier (`Flat`, well-determined) — byte-flat on
                // the identified side (pure REML = mgcv) with only a convex wall
                // against the `λ → 0` degeneracy — or the wide #1089 `Normal` when
                // under-determined. This is NOT a re-tuning toward a particular λ:
                // it REMOVES an unjustified cap and lets REML choose, exactly as on
                // the (already-relaxed) Gaussian path.
                if is_nullspace {
                    nullspace_select_prior.clone()
                } else {
                    relaxed_prior.clone()
                }
            } else if underdetermined && is_nullspace {
                nullspace_select_prior.clone()
            } else {
                relaxed_prior.clone()
            }
        })
        .collect::<Vec<_>>();
    crate::types::RhoPrior::Independent(per_coord)
}

/// Standard deviation of the wide, weakly-informative symmetric `Normal` prior
/// placed on a relaxable smooth's log-λ coordinates when the fit is
/// under-determined (`n < 2·p`); see [`relax_smoothing_rho_prior`].
///
/// Chosen so that ±2σ spans the entire feasible ρ range (the outer optimiser
/// bounds `|ρ| ≤ 30`): the prior contributes strictly-positive termination
/// curvature `1/sd²` to the outer Hessian (the #1089 requirement that the REML
/// loop certify a stationary point on a `p > n` design) while its gradient drag
/// at the heavily-smoothed REML optimum is negligible, so pure REML — matching
/// mgcv — selects λ. Reducing it toward the old `sd = 3` re-introduces the
/// #1392 under-smoothing drag; widening it further weakens termination
/// curvature without further benefit.
const RELAX_UNDERDETERMINED_RHO_SD: f64 = 15.0;

/// Distance-scale bound `upper` (`P(d > upper) = tail_prob` on the marginal-SD
/// scale `d = exp(-ρ/2)`) of the one-sided penalized-complexity prior placed on
/// a relaxable smooth's `DoublePenaltyNullspace` selection coordinate when the
/// fit is under-determined (`n < 2·p`); see [`relax_smoothing_rho_prior`].
///
/// The null-space ridge exists only to SELECT the linear/constant null-space
/// component out (mgcv `select=TRUE`): we want its `λ` driven UP (`d → 0`)
/// unless the data clearly buys the null-space wiggle. A small `upper` puts the
/// exponential PC wall against the `λ → 0` (null space kept) side close in, so
/// the gentle bounded Occam pull toward `λ → ∞` is what remains; the data can
/// still keep the null space when it genuinely earns it. `0.05` places the wall
/// at a marginal-SD scale two decades below unit, biasing toward select-out on
/// the over-parameterized `p > n` wine fit while staying weakly informative.
const NULLSPACE_SELECT_PC_UPPER: f64 = 0.05;

/// Tail probability `α` (`P(d > upper) = α`) calibrating the rate
/// `θ = −ln(α)/upper` of the [`NULLSPACE_SELECT_PC_UPPER`] penalized-complexity
/// select-out prior. A small `α` makes the wall against the kept-null-space
/// (`λ → 0`) side steep; combined with the small `upper` it yields a strong
/// θ ≈ 92 so REML moves the under-determined null-space ridge off its stalled
/// λ ≈ 0.11 toward select-out, without ever a hard cap (the over-smoothing pull
/// stays bounded at gradient +1/2). See [`relax_smoothing_rho_prior`].
const NULLSPACE_SELECT_PC_TAIL_PROB: f64 = 0.01;

fn adaptive_fit_options_base(options: &FitOptions, design: &TermCollectionDesign) -> FitOptions {
    FitOptions {
        latent_cloglog: options.latent_cloglog,
        mixture_link: options.mixture_link.clone(),
        optimize_mixture: options.optimize_mixture,
        sas_link: options.sas_link,
        optimize_sas: options.optimize_sas,
        compute_inference: options.compute_inference,
        skip_rho_posterior_inference: options.skip_rho_posterior_inference,
        max_iter: options.max_iter,
        tol: options.tol,
        nullspace_dims: design.nullspace_dims.clone(),
        linear_constraints: design.linear_constraints.clone(),
        firth_bias_reduction: options.firth_bias_reduction,
        adaptive_regularization: None,
        penalty_shrinkage_floor: options.penalty_shrinkage_floor,
        // Propagate user-supplied rho_prior so the baseline/refit and the
        // joint optimizer minimize the same REML objective.
        rho_prior: options.rho_prior.clone(),
        kronecker_penalty_system: design.kronecker_penalty_system(),
        kronecker_factored: design
            .smooth
            .terms
            .iter()
            .find_map(|t| t.kronecker_factored.clone()),
        persist_warm_start_disk: options.persist_warm_start_disk,
    }
}

fn superseded_fit_options(options: &FitOptions) -> FitOptions {
    let mut fit_options = options.clone();
    fit_options.skip_rho_posterior_inference = true;
    fit_options
}

#[derive(Clone)]
struct BoundedLinearTermMeta {
    col_idx: usize,
    min: f64,
    max: f64,
    prior: BoundedCoefficientPriorSpec,
}

/// β-dependent effective Jacobian for the bounded-linear fit block.
///
/// Each bounded coefficient enters the linear predictor non-linearly, as
/// `β = min + width·σ(θ)`, and is supplied to the solver through the family
/// adapter's offset rather than the linear design. To keep that contribution
/// out of the *linear* design the fit places a deliberately **zeroed**
/// placeholder column for every bounded term in the block design
/// (see `fit_bounded_term_collection_with_design`). The pre-fit
/// identifiability audit, however, assesses block rank by reading each block's
/// effective Jacobian — and a zeroed column reads as a structural rank
/// deficiency, so without this callback the audit refuses *every* bounded
/// model before fitting begins.
///
/// This callback reports the model's true Jacobian column for each bounded
/// term, `∂η_i/∂θ = (dβ/dθ)·x_i`, so the audit inspects the same geometry the
/// solver actually fits. Because `dβ/dθ = width·σ(θ)(1−σ(θ))` is strictly
/// positive for finite θ and `width > 0`, a bounded column is rank-deficient
/// in the audit exactly when its underlying covariate is genuinely collinear
/// with the rest of the design — never merely because the placeholder was
/// zeroed. The callback is consumed only by the identifiability audit /
/// canonicalisation; the inner PIRLS solve drives η through the
/// [`BoundedLinearFamily`] adapter, so reporting the non-zeroed Jacobian here
/// does not double-count the bounded contribution.
struct BoundedEffectiveJacobian {
    design: Array2<f64>,
    bounded_terms: Vec<BoundedLinearTermMeta>,
}

impl BlockEffectiveJacobian for BoundedEffectiveJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let p = self.design.ncols();
        let n = self.design.nrows();
        let rows = rows.start.min(n)..rows.end.min(n);
        if !state.beta.is_empty() {
            if state.beta.len() != p {
                return Err(format!(
                    "BoundedEffectiveJacobian::effective_jacobian_at: beta length {} != design \
                     ncols {p}",
                    state.beta.len(),
                ));
            }
            if state.beta.iter().any(|v| v.is_nan()) {
                return Err(
                    "BoundedEffectiveJacobian::effective_jacobian_at: beta contains NaN"
                        .to_string(),
                );
            }
        }
        let mut jac = self
            .design
            .slice(ndarray::s![rows.start..rows.end, ..])
            .to_owned();
        for term in &self.bounded_terms {
            let theta = if state.beta.is_empty() {
                0.0
            } else {
                state.beta[term.col_idx]
            };
            let (_, _, db_dtheta, _, _) = bounded_latent_derivatives(theta, term.min, term.max);
            jac.column_mut(term.col_idx).mapv_inplace(|v| v * db_dtheta);
        }
        Ok(jac)
    }
}

#[derive(Clone)]
struct BoundedLinearFamily {
    family: LikelihoodSpec,
    latent_cloglog_state: Option<LatentCLogLogState>,
    mixture_link_state: Option<MixtureLinkState>,
    sas_link_state: Option<SasLinkState>,
    y: Array1<f64>,
    weights: Array1<f64>,
    design: Array2<f64>,
    designzeroed: Array2<f64>,
    offset: Array1<f64>,
    bounded_terms: Vec<BoundedLinearTermMeta>,
}

#[derive(Clone)]
struct StandardFamilyObservationState {
    eta: Array1<f64>,
    mu: Array1<f64>,
    score: Array1<f64>,
    fisherweight: Array1<f64>,
    neghessian_eta: Array1<f64>,
    neghessian_eta_derivative: Array1<f64>,
    log_likelihood: f64,
}

fn bounded_logit(z: f64) -> f64 {
    let zc = z.clamp(1e-12, 1.0 - 1e-12);
    (zc / (1.0 - zc)).ln()
}

fn stable_sigmoid(theta: f64) -> f64 {
    if theta >= 0.0 {
        let exp_neg = (-theta).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = theta.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

fn bounded_latent_to_user(theta: f64, min: f64, max: f64) -> (f64, f64, f64) {
    let z = stable_sigmoid(theta);
    let width = max - min;
    let beta = min + width * z;
    let db_dtheta = width * z * (1.0 - z);
    (beta, z, db_dtheta)
}

/// Invert the bounded interval transform: given a user-scale coefficient
/// `beta` in the open interval `(min, max)`, return the latent coordinate
/// `theta` with `bounded_latent_to_user(theta, min, max).0 == beta`.
///
/// This is the exact inverse of the logistic interval map used by the bounded
/// custom family: `z = (beta - min)/(max - min)` (the normalized position in
/// the interval) and `theta = logit(z)`. The normalized position is clamped
/// strictly inside `(0, 1)` (mirroring `bounded_logit`) so a coefficient that
/// sits numerically at a boundary maps to a large-but-finite latent value
/// rather than `±∞`.
fn bounded_user_to_latent(beta: f64, min: f64, max: f64) -> f64 {
    let width = max - min;
    if width <= 0.0 || !width.is_finite() {
        return 0.0;
    }
    let z = (beta - min) / width;
    bounded_logit(z)
}

/// One bounded coefficient column for posterior sampling: its position in the
/// (internal, conditioned) coefficient vector and the interval bounds expressed
/// on that same internal scale.
#[derive(Debug, Clone, Copy)]
pub struct BoundedSampleColumn {
    /// Column index into the internal (conditioned) coefficient vector.
    pub col_idx: usize,
    /// Lower interval bound on the internal scale.
    pub min: f64,
    /// Upper interval bound on the internal scale.
    pub max: f64,
}

/// Exact posterior draws for a model with `bounded()` coefficients.
///
/// The bounded custom family fits each bounded coefficient as a smooth interval
/// transform `beta = min + (max - min)·sigmoid(theta)` of an unconstrained
/// latent `theta`. The Laplace approximation is *Gaussian on the latent scale*
/// — that is precisely the scale on which the fit treats the coefficient as an
/// unconstrained, locally-quadratic parameter. Sampling a Gaussian directly on
/// the user (bounded) scale is wrong twice over: it can place mass outside
/// `[min, max]`, and it discards the boundary-induced skew that the nonlinear
/// map produces. This routine instead draws `theta ~ N(theta_mode, H_latent^{-1})`
/// and pushes every draw through the *exact* interval map, so user-scale draws
/// always lie strictly inside the interval and carry the correct skew.
///
/// Coordinate bookkeeping. The caller supplies the user-scale mode `beta_user`
/// and the user-scale penalized Hessian `user_hessian` (both in *internal /
/// conditioned* coordinates — i.e. before `backtransform_*` to the original
/// data scale) together with the internal-scale bounds for each bounded column.
/// The user-scale Hessian relates to the latent-scale Hessian by the diagonal
/// delta-method Jacobian `J = diag(db/dtheta)`:
///   `H_user = J^{-1} H_latent J^{-1}`  ⇒  `H_latent = J H_user J`,
/// which is exactly the inverse of `transform_bounded_latent_precision_to_user_internal`.
/// Non-bounded columns have `J_ii = 1`, so they are sampled as the ordinary
/// Gaussian Laplace draw and returned unchanged.
///
/// Returns the draws as a `(n_draws, p)` matrix on the *internal* user scale
/// (still conditioned); the caller back-transforms to the original data scale
/// with the same conditioning it used for the point estimate.
pub fn sample_bounded_latent_posterior_internal(
    beta_user: &Array1<f64>,
    user_hessian: &Array2<f64>,
    bounded_columns: &[BoundedSampleColumn],
    n_draws: usize,
    base_seed: u64,
) -> Result<Array2<f64>, EstimationError> {
    let p = beta_user.len();
    if user_hessian.nrows() != p || user_hessian.ncols() != p {
        crate::bail_invalid_estim!(
            "bounded posterior sampling dimension mismatch: mode has {p} entries, user Hessian is {}x{}",
            user_hessian.nrows(),
            user_hessian.ncols()
        );
    }

    // Latent mode and delta-method Jacobian, column by column.
    let mut theta_mode = beta_user.clone();
    let mut jac_diag = Array1::<f64>::ones(p);
    for bc in bounded_columns {
        if bc.col_idx >= p {
            crate::bail_invalid_estim!(
                "bounded posterior sampling: bounded column index {} out of range for {p} coefficients",
                bc.col_idx
            );
        }
        let theta_i = bounded_user_to_latent(beta_user[bc.col_idx], bc.min, bc.max);
        let (_, _, db_dtheta) = bounded_latent_to_user(theta_i, bc.min, bc.max);
        theta_mode[bc.col_idx] = theta_i;
        // Guard against a degenerate (numerically vanishing) Jacobian at a
        // coefficient pinned hard against a boundary: floor the slope so the
        // latent precision stays finite and the draw simply collapses onto the
        // boundary, which is the correct limiting posterior.
        jac_diag[bc.col_idx] = db_dtheta.max(1e-12);
    }

    // H_latent = J H_user J  (J diagonal). This is the exact inverse of the
    // user-scale precision transform applied at fit time.
    let mut h_latent = user_hessian.clone();
    for i in 0..p {
        let ji = jac_diag[i];
        if ji != 1.0 {
            h_latent.row_mut(i).mapv_inplace(|v| v * ji);
            h_latent.column_mut(i).mapv_inplace(|v| v * ji);
        }
    }

    // Draw theta ~ N(theta_mode, H_latent^{-1}) via the Cholesky of H_latent:
    // L Lᵀ = H_latent, solve Lᵀ δ = ε so Var(δ) = H_latent^{-1}.
    use crate::faer_ndarray::FaerCholesky as _;
    use rand::SeedableRng as _;
    let chol = h_latent.cholesky(faer::Side::Lower).map_err(|err| {
        EstimationError::InvalidInput(format!(
            "bounded posterior sampling: Cholesky of the latent penalized Hessian failed: {err:?}"
        ))
    })?;
    let l = chol.lower_triangular();

    let mut draws = Array2::<f64>::zeros((n_draws, p));
    let mut eps = Array1::<f64>::zeros(p);
    let mut delta = Array1::<f64>::zeros(p);
    let mut rng = rand::rngs::StdRng::seed_from_u64(base_seed);
    for k in 0..n_draws {
        for e in eps.iter_mut() {
            *e = standard_normal_draw(&mut rng);
        }
        solve_lower_transpose_into(&l, &eps, &mut delta);
        for i in 0..p {
            draws[(k, i)] = theta_mode[i] + delta[i];
        }
        // Push bounded columns through the exact interval map so every draw is
        // strictly inside (min, max); leave unconstrained columns untouched.
        for bc in bounded_columns {
            let (beta_draw, _, _) = bounded_latent_to_user(draws[(k, bc.col_idx)], bc.min, bc.max);
            draws[(k, bc.col_idx)] = beta_draw;
        }
    }

    Ok(draws)
}

/// Box-Muller standard-normal draw (kept local so the bounded sampler does not
/// depend on the HMC module's RNG plumbing).
#[inline]
fn standard_normal_draw<R: rand::Rng + ?Sized>(rng: &mut R) -> f64 {
    use rand::RngExt as _;
    let u1 = rng.random::<f64>().max(1e-16);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Solve `Lᵀ x = b` for a lower-triangular `L` (back substitution), writing the
/// result into `out`. Used to turn a standard-normal `b` into a draw with
/// covariance `(L Lᵀ)^{-1}`.
fn solve_lower_transpose_into(l: &Array2<f64>, b: &Array1<f64>, out: &mut Array1<f64>) {
    let p = l.nrows();
    for i in (0..p).rev() {
        let mut acc = b[i];
        for j in (i + 1)..p {
            acc -= l[(j, i)] * out[j];
        }
        let diag = l[(i, i)];
        out[i] = if diag.abs() > 0.0 { acc / diag } else { 0.0 };
    }
}

fn bounded_latent_derivatives(theta: f64, min: f64, max: f64) -> (f64, f64, f64, f64, f64) {
    let z = stable_sigmoid(theta);
    let width = max - min;
    let s = z * (1.0 - z);
    let beta = min + width * z;
    let db_dtheta = width * s;
    let d2b_dtheta2 = width * s * (1.0 - 2.0 * z);
    let d3b_dtheta3 = width * s * (1.0 - 6.0 * z + 6.0 * z * z);
    (beta, z, db_dtheta, d2b_dtheta2, d3b_dtheta3)
}

fn bounded_prior_terms(theta: f64, prior: &BoundedCoefficientPriorSpec) -> (f64, f64, f64, f64) {
    let (a, b) = match prior {
        // `None` means constrained MLE with no extra prior term on the bounded coefficient.
        BoundedCoefficientPriorSpec::None => return (0.0, 0.0, 0.0, 0.0),
        // Uniform on the normalized user-scale coefficient z in (0, 1). In latent space this is
        // exactly the Jacobian term for the logistic transform, up to an additive width constant.
        BoundedCoefficientPriorSpec::Uniform => (1.0, 1.0),
        BoundedCoefficientPriorSpec::Beta { a, b } => (*a, *b),
    };
    let z = stable_sigmoid(theta).clamp(1e-12, 1.0 - 1e-12);
    let logp = a * z.ln() + b * (1.0 - z).ln();
    let grad = a - (a + b) * z;
    let neghess = (a + b) * z * (1.0 - z);
    let neghess_derivative = (a + b) * z * (1.0 - z) * (1.0 - 2.0 * z);
    (logp, grad, neghess, neghess_derivative)
}

fn evaluate_standard_familyobservations(
    family: LikelihoodSpec,
    latent_cloglog_state: Option<&LatentCLogLogState>,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    eta: &Array1<f64>,
) -> Result<StandardFamilyObservationState, EstimationError> {
    const PROB_EPS: f64 = 1e-10;
    const MU_DERIV_EPS: f64 = 1e-12;
    let n = y.len();
    if weights.len() != n || eta.len() != n {
        crate::bail_invalid_estim!("bounded family observation size mismatch");
    }

    let mut mu = Array1::<f64>::zeros(n);
    let mut score = Array1::<f64>::zeros(n);
    let mut fisherweight = Array1::<f64>::zeros(n);
    let mut neghessian_eta = Array1::<f64>::zeros(n);
    let mut neghessian_eta_derivative = Array1::<f64>::zeros(n);
    let mut log_likelihood = 0.0;

    for i in 0..n {
        let w = weights[i].max(0.0);
        let yi = y[i];
        let eta_i = eta[i];
        match (&family.response, &family.link) {
            (ResponseFamily::Gaussian, _) => {
                let resid = yi - eta_i;
                mu[i] = eta_i;
                score[i] = w * resid;
                fisherweight[i] = w.max(MU_DERIV_EPS);
                neghessian_eta[i] = w;
                neghessian_eta_derivative[i] = 0.0;
                log_likelihood += -0.5 * w * resid * resid;
            }
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Logit)) => {
                let jet = logit_inverse_link_jet5(eta_i);
                mu[i] = jet.mu;
                score[i] = w * (yi - jet.mu);
                fisherweight[i] = jet.d1.max(MU_DERIV_EPS);
                neghessian_eta[i] = jet.d1;
                neghessian_eta_derivative[i] = jet.d2;
                let logmu = -crate::linalg::utils::stable_softplus(-eta_i);
                let log_one_minusmu = -crate::linalg::utils::stable_softplus(eta_i);
                log_likelihood += w * (yi * logmu + (1.0 - yi) * log_one_minusmu);
            }
            (ResponseFamily::Binomial, _) => {
                let inverse_link = if let Some(state) = latent_cloglog_state {
                    Some(InverseLink::LatentCLogLog(*state))
                } else if let Some(state) = mixture_link_state {
                    Some(InverseLink::Mixture(state.clone()))
                } else {
                    sas_link_state.map(|state| {
                        if family.is_binomial_beta_logistic() {
                            InverseLink::BetaLogistic(*state)
                        } else {
                            InverseLink::Sas(*state)
                        }
                    })
                };
                let strategy_spec = LikelihoodSpec {
                    response: family.response.clone(),
                    link: inverse_link.clone().unwrap_or_else(|| family.link.clone()),
                };
                let jet = strategy_for_spec(&strategy_spec).inverse_link_jet(eta_i)?;
                let mu_i_raw = jet.mu;
                let dmu_deta_raw = jet.d1;
                let mu_i: f64 = mu_i_raw.clamp(PROB_EPS, 1.0 - PROB_EPS);
                let dmu_deta = dmu_deta_raw.max(MU_DERIV_EPS);
                let d2mu_deta2 = jet.d2;
                let d3mu_deta3 = jet.d3;
                let var = (mu_i * (1.0 - mu_i)).max(PROB_EPS);
                let lmu = (yi - mu_i) / var;
                let lmumu = -(yi / (mu_i * mu_i)) - ((1.0 - yi) / ((1.0 - mu_i) * (1.0 - mu_i)));
                let lmumumu = 2.0 * yi / (mu_i * mu_i * mu_i)
                    - 2.0 * (1.0 - yi) / ((1.0 - mu_i) * (1.0 - mu_i) * (1.0 - mu_i));
                mu[i] = mu_i;
                score[i] = w * lmu * dmu_deta;
                fisherweight[i] = (w * dmu_deta * dmu_deta / var).max(MU_DERIV_EPS);
                neghessian_eta[i] = -w * (lmumu * dmu_deta * dmu_deta + lmu * d2mu_deta2);
                neghessian_eta_derivative[i] = -w
                    * (lmumumu * dmu_deta * dmu_deta * dmu_deta
                        + 3.0 * lmumu * dmu_deta * d2mu_deta2
                        + lmu * d3mu_deta3);
                log_likelihood += w * (yi * mu_i.ln() + (1.0 - yi) * (1.0 - mu_i).ln());
            }
            (ResponseFamily::Poisson, _) => {
                crate::bail_invalid_estim!(
                    "bounded linear terms are not supported for PoissonLog fits"
                );
            }
            (ResponseFamily::Tweedie { .. }, _) => {
                crate::bail_invalid_estim!(
                    "bounded linear terms are not supported for Tweedie fits"
                );
            }
            (ResponseFamily::NegativeBinomial { .. }, _) => {
                crate::bail_invalid_estim!(
                    "bounded linear terms are not supported for NegativeBinomial fits"
                );
            }
            (ResponseFamily::Beta { .. }, _) => {
                crate::bail_invalid_estim!(
                    "bounded linear terms are not supported for BetaLogit fits"
                );
            }
            (ResponseFamily::Gamma, _) => {
                crate::bail_invalid_estim!(
                    "bounded linear terms are not supported for GammaLog fits"
                );
            }
            (ResponseFamily::RoystonParmar, _) => {
                crate::bail_invalid_estim!(
                    "bounded linear terms are not supported for survival model fits"
                );
            }
        }
    }

    Ok(StandardFamilyObservationState {
        eta: eta.clone(),
        mu,
        score,
        fisherweight,
        neghessian_eta,
        neghessian_eta_derivative,
        log_likelihood,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpatialAdaptiveHyperKind {
    LogLambdaMagnitude,
    LogLambdaGradient,
    LogLambdaCurvature,
    LogEpsilonMagnitude,
    LogEpsilonGradient,
    LogEpsilonCurvature,
}

impl SpatialAdaptiveHyperKind {
    fn component_index(self) -> usize {
        match self {
            SpatialAdaptiveHyperKind::LogLambdaMagnitude
            | SpatialAdaptiveHyperKind::LogEpsilonMagnitude => 0,
            SpatialAdaptiveHyperKind::LogLambdaGradient
            | SpatialAdaptiveHyperKind::LogEpsilonGradient => 1,
            SpatialAdaptiveHyperKind::LogLambdaCurvature
            | SpatialAdaptiveHyperKind::LogEpsilonCurvature => 2,
        }
    }

    fn is_log_lambda(self) -> bool {
        matches!(
            self,
            SpatialAdaptiveHyperKind::LogLambdaMagnitude
                | SpatialAdaptiveHyperKind::LogLambdaGradient
                | SpatialAdaptiveHyperKind::LogLambdaCurvature
        )
    }

    fn is_log_epsilon(self) -> bool {
        matches!(
            self,
            SpatialAdaptiveHyperKind::LogEpsilonMagnitude
                | SpatialAdaptiveHyperKind::LogEpsilonGradient
                | SpatialAdaptiveHyperKind::LogEpsilonCurvature
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct SpatialAdaptiveHyperSpec {
    cache_index: usize,
    kind: SpatialAdaptiveHyperKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpatialAdaptiveExplicitSecondOrderKind {
    StructuralZero,
    LocalAlphaAlpha,
    LocalAlphaEta,
    SharedEtaEta,
}

/// Penalty family selected within one adaptive smooth cache. The component index
/// (0/1/2) used throughout the runtime caches maps onto these three operators:
/// the scalar magnitude operator `d0`, the grouped gradient operator `d1`, and
/// the grouped curvature operator `d2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdaptiveComponent {
    Magnitude,
    Gradient,
    Curvature,
}

impl AdaptiveComponent {
    fn from_index(index: usize) -> Result<Self, String> {
        match index {
            0 => Ok(AdaptiveComponent::Magnitude),
            1 => Ok(AdaptiveComponent::Gradient),
            2 => Ok(AdaptiveComponent::Curvature),
            other => Err(SmoothError::invalid_index(format!(
                "invalid adaptive component index {}",
                other
            ))
            .into()),
        }
    }
}

/// Which hyper-derivative of the adaptive penalty's local pieces to assemble.
/// Each variant selects one accessor triple (objective scalar, beta-mixed
/// gradient, beta hessian) on the per-component exact state; the operator
/// embedding around those accessors is identical across variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HyperDerivativeKind {
    /// First derivative in `log lambda` (rho): the bare penalty pieces.
    Rho,
    /// First derivative in `log epsilon`.
    LogEpsilonFirst,
    /// Second derivative in `log epsilon`.
    LogEpsilonSecond,
}

/// Which directional-drift hyper-derivative of the adaptive penalty Hessian to
/// assemble: the bare rho drift, or the shared-`log epsilon` drift. Both share
/// the per-component direction projection, operator embedding, and global
/// embedding; only the directional state accessor differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HyperDriftKind {
    Rho,
    LogEpsilon,
}

impl SpatialAdaptiveHyperSpec {
    fn component_index(self) -> usize {
        self.kind.component_index()
    }

    fn explicit_second_order_kind(self, other: Self) -> SpatialAdaptiveExplicitSecondOrderKind {
        if self.component_index() != other.component_index() {
            return SpatialAdaptiveExplicitSecondOrderKind::StructuralZero;
        }
        match (
            self.kind.is_log_lambda(),
            other.kind.is_log_lambda(),
            self.kind.is_log_epsilon(),
            other.kind.is_log_epsilon(),
        ) {
            (true, true, false, false) if self.cache_index == other.cache_index => {
                SpatialAdaptiveExplicitSecondOrderKind::LocalAlphaAlpha
            }
            (true, false, false, true) | (false, true, true, false) => {
                SpatialAdaptiveExplicitSecondOrderKind::LocalAlphaEta
            }
            (false, false, true, true) => SpatialAdaptiveExplicitSecondOrderKind::SharedEtaEta,
            _ => SpatialAdaptiveExplicitSecondOrderKind::StructuralZero,
        }
    }
}

#[derive(Clone, Debug)]
struct SpatialAdaptiveTermHyperParams {
    lambda: [f64; 3],
    epsilon: [f64; 3],
}

#[derive(Clone)]
struct SpatialAdaptiveExactEvaluation {
    obs: StandardFamilyObservationState,
    adaptive_states: Vec<SpatialPenaltyExactState>,
    adaptive_penalty_value: f64,
    adaptive_penaltygradient: Array1<f64>,
    adaptive_penaltyhessian: Array2<f64>,
    fixed_quadraticvalue: f64,
    fixed_quadraticgradient: Array1<f64>,
    fixed_quadratichessian: Array2<f64>,
}

#[derive(Clone)]
struct CachedSpatialAdaptiveExactEvaluation {
    beta: Array1<f64>,
    eval: Arc<SpatialAdaptiveExactEvaluation>,
}

impl SpatialAdaptiveExactEvaluation {
    fn total_penalty_value(&self) -> f64 {
        self.adaptive_penalty_value + self.fixed_quadraticvalue
    }

    fn total_penaltygradient(&self) -> Array1<f64> {
        &self.adaptive_penaltygradient + &self.fixed_quadraticgradient
    }

    fn total_penaltyhessian(&self) -> Array2<f64> {
        &self.adaptive_penaltyhessian + &self.fixed_quadratichessian
    }

    fn totalobjectivehessian(&self, design: &Array2<f64>) -> Result<Array2<f64>, String> {
        let mut out = xt_diag_x_dense(design.view(), self.obs.neghessian_eta.view())?;
        out += &self.total_penaltyhessian();
        Ok(out)
    }
}

#[derive(Clone)]
struct SpatialAdaptiveExactFamily {
    family: LikelihoodSpec,
    latent_cloglog_state: Option<LatentCLogLogState>,
    mixture_link_state: Option<MixtureLinkState>,
    sas_link_state: Option<SasLinkState>,
    y: Arc<Array1<f64>>,
    weights: Arc<Array1<f64>>,
    design: Arc<Array2<f64>>,
    offset: Arc<Array1<f64>>,
    linear_constraints: Option<LinearInequalityConstraints>,
    runtime_caches: Arc<Vec<SpatialOperatorRuntimeCache>>,
    adaptive_params: Vec<SpatialAdaptiveTermHyperParams>,
    fixed_quadratichessian: Arc<Array2<f64>>,
    hyperspecs: Arc<Vec<SpatialAdaptiveHyperSpec>>,
    exact_eval_cache: Arc<Mutex<Option<CachedSpatialAdaptiveExactEvaluation>>>,
}

impl SpatialAdaptiveExactFamily {
    fn with_adaptive_params(
        &self,
        adaptive_params: Vec<SpatialAdaptiveTermHyperParams>,
        fixed_quadratichessian: Arc<Array2<f64>>,
    ) -> Self {
        Self {
            family: self.family.clone(),
            latent_cloglog_state: self.latent_cloglog_state,
            mixture_link_state: self.mixture_link_state.clone(),
            sas_link_state: self.sas_link_state,
            y: self.y.clone(),
            weights: self.weights.clone(),
            design: self.design.clone(),
            offset: self.offset.clone(),
            linear_constraints: self.linear_constraints.clone(),
            runtime_caches: self.runtime_caches.clone(),
            adaptive_params,
            fixed_quadratichessian,
            hyperspecs: self.hyperspecs.clone(),
            exact_eval_cache: Arc::new(Mutex::new(None)),
        }
    }

    fn total_eta(&self, beta: &Array1<f64>) -> Array1<f64> {
        crate::faer_ndarray::fast_av(self.design.as_ref(), beta) + self.offset.as_ref()
    }

    fn fixed_quadratic_terms(&self, beta: &Array1<f64>) -> (f64, Array1<f64>) {
        let grad = self.fixed_quadratichessian.dot(beta);
        let value = 0.5 * beta.dot(&grad);
        (value, grad)
    }

    fn adaptive_penalty_value_only(&self, beta: &Array1<f64>) -> Result<f64, String> {
        let mut penalty_value = 0.0;
        for (cache_idx, cache) in self.runtime_caches.iter().enumerate() {
            let params = self.adaptive_params.get(cache_idx).ok_or_else(|| {
                format!(
                    "missing adaptive parameter block for cache {}",
                    cache.termname
                )
            })?;
            let beta_local = beta.slice(s![cache.coeff_global_range.clone()]);
            let state =
                SpatialPenaltyExactState::from_beta_local(beta_local, cache, params.epsilon)
                    .map_err(|e| e.to_string())?;
            penalty_value += params.lambda[0] * state.magnitude.penalty_value();
            penalty_value += params.lambda[1] * state.gradient.penalty_value();
            penalty_value += params.lambda[2] * state.curvature.penalty_value();
        }
        Ok(penalty_value)
    }

    fn zero_hyper_parts(&self) -> (Array1<f64>, Array2<f64>) {
        let total_dim = self.design.ncols();
        (
            Array1::<f64>::zeros(total_dim),
            Array2::<f64>::zeros((total_dim, total_dim)),
        )
    }

    fn embed_local_hyper_parts(
        &self,
        coeff_range: &Range<usize>,
        local_grad: &Array1<f64>,
        local_hess: &Array2<f64>,
    ) -> (Array1<f64>, Array2<f64>) {
        let (mut beta_mixed, mut betahessian) = self.zero_hyper_parts();
        beta_mixed
            .slice_mut(s![coeff_range.clone()])
            .assign(local_grad);
        betahessian
            .slice_mut(s![coeff_range.clone(), coeff_range.clone()])
            .assign(local_hess);
        (beta_mixed, betahessian)
    }

    fn embed_local_hyper_hessian(
        &self,
        coeff_range: &Range<usize>,
        local_hess: &Array2<f64>,
    ) -> Array2<f64> {
        let total_dim = self.design.ncols();
        let mut out = Array2::<f64>::zeros((total_dim, total_dim));
        out.slice_mut(s![coeff_range.clone(), coeff_range.clone()])
            .assign(local_hess);
        out
    }

    /// Unified per-block hyper-derivative assembly. Owns the shared cache /
    /// hyperparameter / exact-state lookup, the component -> operator selection
    /// (scalar magnitude `d0`, grouped gradient `d1`, grouped curvature `d2`),
    /// and the global embedding via [`Self::embed_local_hyper_parts`]. The only
    /// piece that varies with `derivative` is the per-component accessor triple
    /// (objective scalar, beta-mixed gradient, beta hessian) read off the exact
    /// state. Returns `(objective, beta_mixed, betahessian)`, each already
    /// scaled by the component's penalty weight `lambda`.
    fn adaptive_block_eval(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        cache_idx: usize,
        component: AdaptiveComponent,
        derivative: HyperDerivativeKind,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let cache = self
            .runtime_caches
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive cache index {} out of bounds", cache_idx))?;
        let params = self
            .adaptive_params
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive hyperparameter block {} out of bounds", cache_idx))?;
        let state = eval
            .adaptive_states
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive exact state index {} out of bounds", cache_idx))?;

        let (objective_local, beta_mixed_local, betahessian_local) = match component {
            AdaptiveComponent::Magnitude => {
                let lambda = params.lambda[0];
                let mag = &state.magnitude;
                let (objective, gradient_coeff, hessian_diag) = match derivative {
                    HyperDerivativeKind::Rho => (
                        mag.penalty_value(),
                        mag.betagradient_coeff(),
                        mag.betahessian_diag(),
                    ),
                    HyperDerivativeKind::LogEpsilonFirst => (
                        mag.log_epsilon_gradient_terms().sum(),
                        mag.log_epsilon_betagradient_coeff(),
                        mag.log_epsilon_betahessian_diag(),
                    ),
                    HyperDerivativeKind::LogEpsilonSecond => (
                        mag.log_epsilon_hessian_terms().sum(),
                        mag.log_epsilon_beta_mixed_second_coeff(),
                        mag.log_epsilon_betahessian_second_diag(),
                    ),
                };
                (
                    lambda * objective,
                    lambda * scalar_operatorgradient(&cache.d0, &gradient_coeff),
                    lambda * scalar_operatorhessian(&cache.d0, &hessian_diag),
                )
            }
            AdaptiveComponent::Gradient => {
                let lambda = params.lambda[1];
                let grad = &state.gradient;
                let (objective, gradient_blocks, hessian_blocks) = match derivative {
                    HyperDerivativeKind::Rho => (
                        grad.penalty_value(),
                        grad.betagradient_blocks(),
                        grad.betahessian_blocks(),
                    ),
                    HyperDerivativeKind::LogEpsilonFirst => (
                        grad.log_epsilon_gradient_terms().sum(),
                        grad.log_epsilon_betagradient_blocks(),
                        grad.log_epsilon_betahessian_blocks(),
                    ),
                    HyperDerivativeKind::LogEpsilonSecond => (
                        grad.log_epsilon_hessian_terms().sum(),
                        grad.log_epsilon_beta_mixed_second_blocks(),
                        grad.log_epsilon_betahessian_second_blocks(),
                    ),
                };
                (
                    lambda * objective,
                    lambda
                        * grouped_operatorgradient(&cache.d1, cache.dimension, &gradient_blocks)
                            .map_err(|e| e.to_string())?,
                    lambda
                        * grouped_operatorhessian(&cache.d1, cache.dimension, &hessian_blocks)
                            .map_err(|e| e.to_string())?,
                )
            }
            AdaptiveComponent::Curvature => {
                let lambda = params.lambda[2];
                let group = cache.dimension * cache.dimension;
                let curv = &state.curvature;
                let (objective, gradient_blocks, hessian_blocks) = match derivative {
                    HyperDerivativeKind::Rho => (
                        curv.penalty_value(),
                        curv.betagradient_blocks(),
                        curv.betahessian_blocks(),
                    ),
                    HyperDerivativeKind::LogEpsilonFirst => (
                        curv.log_epsilon_gradient_terms().sum(),
                        curv.log_epsilon_betagradient_blocks(),
                        curv.log_epsilon_betahessian_blocks(),
                    ),
                    HyperDerivativeKind::LogEpsilonSecond => (
                        curv.log_epsilon_hessian_terms().sum(),
                        curv.log_epsilon_beta_mixed_second_blocks(),
                        curv.log_epsilon_betahessian_second_blocks(),
                    ),
                };
                (
                    lambda * objective,
                    lambda
                        * grouped_operatorgradient(&cache.d2, group, &gradient_blocks)
                            .map_err(|e| e.to_string())?,
                    lambda
                        * grouped_operatorhessian(&cache.d2, group, &hessian_blocks)
                            .map_err(|e| e.to_string())?,
                )
            }
        };

        let (beta_mixed, betahessian) = self.embed_local_hyper_parts(
            &cache.coeff_global_range,
            &beta_mixed_local,
            &betahessian_local,
        );
        Ok((objective_local, beta_mixed, betahessian))
    }

    fn adaptive_shared_log_epsilon_parts(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        component: usize,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        // Exact shared-log-epsilon first-order pieces:
        //
        //   J_{eta_p}         = sum_m lambda_{m,p} U_{m,p,eta},
        //   J_{beta,eta_p}    = sum_m lambda_{m,p} U_{m,p,beta eta},
        //   J_{beta,beta,eta} = sum_m lambda_{m,p} U_{m,p,beta beta eta}.
        self.adaptive_shared_block_eval(eval, component, HyperDerivativeKind::LogEpsilonFirst)
    }

    fn adaptive_shared_log_epsilon_second_parts(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        component: usize,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        // Exact shared-log-epsilon second-order pieces:
        //
        //   J_{eta_p,eta_p}            = sum_m lambda_{m,p} U_{m,p,eta eta},
        //   J_{beta,eta_p,eta_p}       = sum_m lambda_{m,p} U_{m,p,beta eta eta},
        //   J_{beta,beta,eta_p,eta_p}  = sum_m lambda_{m,p} U_{m,p,beta beta eta eta}.
        self.adaptive_shared_block_eval(eval, component, HyperDerivativeKind::LogEpsilonSecond)
    }

    /// Sum a per-block hyper-derivative across every adaptive term for one shared
    /// `log epsilon` coordinate (selected by `component`). The three log-epsilon
    /// coordinates are shared globally by penalty type, so each contributes the
    /// matching component's block from every cache.
    fn adaptive_shared_block_eval(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        component: usize,
        derivative: HyperDerivativeKind,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let component = AdaptiveComponent::from_index(component)?;
        let (mut score, mut hessian) = self.zero_hyper_parts();
        let mut objective = 0.0;
        for cache_idx in 0..self.runtime_caches.len() {
            let (local_objective, local_score, local_hessian) =
                self.adaptive_block_eval(eval, cache_idx, component, derivative)?;
            objective += local_objective;
            score += &local_score;
            hessian += &local_hessian;
        }
        Ok((objective, score, hessian))
    }

    fn adaptive_shared_log_epsilon_drift(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        component: usize,
        direction: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        // Exact shared-log-epsilon Hessian drift:
        //
        //   T_{eta_p}[u] = sum_m lambda_{m,p} D_beta(U_{m,p,beta beta eta})[u].
        let component = AdaptiveComponent::from_index(component)?;
        let total_dim = self.design.ncols();
        let mut total = Array2::<f64>::zeros((total_dim, total_dim));
        for cache_idx in 0..self.runtime_caches.len() {
            total += &self.adaptive_block_drift_eval(
                eval,
                cache_idx,
                component,
                HyperDriftKind::LogEpsilon,
                direction,
            )?;
        }
        Ok(total)
    }

    fn adaptive_explicit_second_order_parts(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        left: SpatialAdaptiveHyperSpec,
        right: SpatialAdaptiveHyperSpec,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        // Structural sparsity from the adaptive penalty algebra:
        //
        //   - alpha_{m,p} / alpha_{n,r} is nonzero only when (m,p) = (n,r),
        //   - alpha_{m,p} / eta_r is nonzero only when p = r,
        //   - eta_p / eta_r is nonzero only when p = r,
        //
        // with eta_p contributions summed over all adaptive terms m because the
        // three log-epsilon coordinates are shared globally by penalty type.
        match left.explicit_second_order_kind(right) {
            SpatialAdaptiveExplicitSecondOrderKind::StructuralZero => {
                let (score, hessian) = self.zero_hyper_parts();
                Ok((0.0, score, hessian))
            }
            SpatialAdaptiveExplicitSecondOrderKind::LocalAlphaAlpha => self.adaptive_block_eval(
                eval,
                left.cache_index,
                AdaptiveComponent::from_index(left.component_index())?,
                HyperDerivativeKind::Rho,
            ),
            SpatialAdaptiveExplicitSecondOrderKind::LocalAlphaEta => {
                let local_alpha = if left.kind.is_log_lambda() {
                    left
                } else {
                    right
                };
                self.adaptive_block_eval(
                    eval,
                    local_alpha.cache_index,
                    AdaptiveComponent::from_index(local_alpha.component_index())?,
                    HyperDerivativeKind::LogEpsilonFirst,
                )
            }
            SpatialAdaptiveExplicitSecondOrderKind::SharedEtaEta => {
                self.adaptive_shared_log_epsilon_second_parts(eval, left.component_index())
            }
        }
    }

    /// Unified per-block directional-drift assembly. Owns the shared cache /
    /// hyperparameter / exact-state lookup, the per-component direction
    /// projection through the collocation operators, the operator embedding, and
    /// the global embedding via [`Self::embed_local_hyper_hessian`]. The only
    /// piece that varies with `drift` is the directional state accessor:
    /// [`HyperDriftKind::Rho`] takes the bare directional Hessian drift, while
    /// [`HyperDriftKind::LogEpsilon`] takes its `log epsilon` derivative.
    fn adaptive_block_drift_eval(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        cache_idx: usize,
        component: AdaptiveComponent,
        drift: HyperDriftKind,
        direction: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let cache = self
            .runtime_caches
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive cache index {} out of bounds", cache_idx))?;
        let params = self
            .adaptive_params
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive hyperparameter block {} out of bounds", cache_idx))?;
        let state = eval
            .adaptive_states
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive exact state index {} out of bounds", cache_idx))?;
        let direction_local = direction.slice(s![cache.coeff_global_range.clone()]);

        let local_hessian = match component {
            AdaptiveComponent::Magnitude => {
                let d0_u = cache.d0.dot(&direction_local);
                let mag = &state.magnitude;
                let diag = match drift {
                    HyperDriftKind::Rho => mag.directionalhessian_diag(&d0_u),
                    HyperDriftKind::LogEpsilon => {
                        mag.log_epsilon_betahessian_directional_diag(&d0_u)
                    }
                };
                params.lambda[0] * scalar_operatorhessian(&cache.d0, &diag)
            }
            AdaptiveComponent::Gradient => {
                let d1_u = cache.d1.dot(&direction_local);
                let direction_blocks = collocationgradient_blocks(&d1_u, cache.dimension)
                    .map_err(|e| e.to_string())?;
                let grad = &state.gradient;
                let blocks = match drift {
                    HyperDriftKind::Rho => grad.directionalhessian_blocks(&direction_blocks),
                    HyperDriftKind::LogEpsilon => {
                        grad.log_epsilon_betahessian_directional_blocks(&direction_blocks)
                    }
                };
                params.lambda[1]
                    * grouped_operatorhessian(&cache.d1, cache.dimension, &blocks)
                        .map_err(|e| e.to_string())?
            }
            AdaptiveComponent::Curvature => {
                let group = cache.dimension * cache.dimension;
                let d2_u = cache.d2.dot(&direction_local);
                let direction_blocks =
                    collocationhessian_blocks(&d2_u, cache.dimension).map_err(|e| e.to_string())?;
                let curv = &state.curvature;
                let blocks = match drift {
                    HyperDriftKind::Rho => curv.directionalhessian_blocks(&direction_blocks),
                    HyperDriftKind::LogEpsilon => {
                        curv.log_epsilon_betahessian_directional_blocks(&direction_blocks)
                    }
                };
                params.lambda[2]
                    * grouped_operatorhessian(&cache.d2, group, &blocks)
                        .map_err(|e| e.to_string())?
            }
        };

        Ok(self.embed_local_hyper_hessian(&cache.coeff_global_range, &local_hessian))
    }

    fn adaptive_hyper_parts(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        hyper: SpatialAdaptiveHyperSpec,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        match hyper.kind {
            // Per-term `log lambda` (rho) hyper-derivative: the bare penalty
            // pieces for this cache's selected component.
            SpatialAdaptiveHyperKind::LogLambdaMagnitude
            | SpatialAdaptiveHyperKind::LogLambdaGradient
            | SpatialAdaptiveHyperKind::LogLambdaCurvature => self.adaptive_block_eval(
                eval,
                hyper.cache_index,
                AdaptiveComponent::from_index(hyper.component_index())?,
                HyperDerivativeKind::Rho,
            ),
            // Shared `log epsilon` hyper-derivative: summed across all terms.
            SpatialAdaptiveHyperKind::LogEpsilonMagnitude
            | SpatialAdaptiveHyperKind::LogEpsilonGradient
            | SpatialAdaptiveHyperKind::LogEpsilonCurvature => {
                self.adaptive_shared_log_epsilon_parts(eval, hyper.component_index())
            }
        }
    }

    fn exact_evaluation_uncached(
        &self,
        beta: &Array1<f64>,
    ) -> Result<SpatialAdaptiveExactEvaluation, String> {
        let eta = self.total_eta(beta);
        let obs = evaluate_standard_familyobservations(
            self.family.clone(),
            self.latent_cloglog_state.as_ref(),
            self.mixture_link_state.as_ref(),
            self.sas_link_state.as_ref(),
            &self.y,
            &self.weights,
            &eta,
        )
        .map_err(|e| e.to_string())?;
        let p = beta.len();
        let mut penalty_value = 0.0;
        let mut penaltygradient = Array1::<f64>::zeros(p);
        let mut penaltyhessian = Array2::<f64>::zeros((p, p));
        let mut adaptive_states = Vec::with_capacity(self.runtime_caches.len());

        for (cache_idx, cache) in self.runtime_caches.iter().enumerate() {
            let params = self.adaptive_params.get(cache_idx).ok_or_else(|| {
                format!(
                    "missing adaptive parameter block for cache {}",
                    cache.termname
                )
            })?;
            let beta_local = beta.slice(s![cache.coeff_global_range.clone()]);
            let state =
                SpatialPenaltyExactState::from_beta_local(beta_local, cache, params.epsilon)
                    .map_err(|e| e.to_string())?;

            let g0 = scalar_operatorgradient(&cache.d0, &state.magnitude.betagradient_coeff());
            let gg = grouped_operatorgradient(
                &cache.d1,
                cache.dimension,
                &state.gradient.betagradient_blocks(),
            )
            .map_err(|e| e.to_string())?;
            let gc = grouped_operatorgradient(
                &cache.d2,
                cache.dimension * cache.dimension,
                &state.curvature.betagradient_blocks(),
            )
            .map_err(|e| e.to_string())?;
            let h0 = scalar_operatorhessian(&cache.d0, &state.magnitude.betahessian_diag());
            let hg = grouped_operatorhessian(
                &cache.d1,
                cache.dimension,
                &state.gradient.betahessian_blocks(),
            )
            .map_err(|e| e.to_string())?;
            let hc = grouped_operatorhessian(
                &cache.d2,
                cache.dimension * cache.dimension,
                &state.curvature.betahessian_blocks(),
            )
            .map_err(|e| e.to_string())?;

            let lambda0 = params.lambda[0];
            let lambdag = params.lambda[1];
            let lambdac = params.lambda[2];

            penalty_value += lambda0 * state.magnitude.penalty_value();
            penalty_value += lambdag * state.gradient.penalty_value();
            penalty_value += lambdac * state.curvature.penalty_value();

            let range = cache.coeff_global_range.clone();
            {
                let mut grad_local = penaltygradient.slice_mut(s![range.clone()]);
                grad_local += &(g0.mapv(|v| lambda0 * v));
                grad_local += &(gg.mapv(|v| lambdag * v));
                grad_local += &(gc.mapv(|v| lambdac * v));
            }
            {
                let mut h_local = penaltyhessian.slice_mut(s![range.clone(), range]);
                h_local += &h0.mapv(|v| lambda0 * v);
                h_local += &hg.mapv(|v| lambdag * v);
                h_local += &hc.mapv(|v| lambdac * v);
            }

            adaptive_states.push(state);
        }

        let (fixed_quadraticvalue, fixed_quadraticgradient) = self.fixed_quadratic_terms(beta);
        Ok(SpatialAdaptiveExactEvaluation {
            obs,
            adaptive_states,
            adaptive_penalty_value: penalty_value,
            adaptive_penaltygradient: penaltygradient,
            adaptive_penaltyhessian: penaltyhessian,
            fixed_quadraticvalue,
            fixed_quadraticgradient,
            fixed_quadratichessian: self.fixed_quadratichessian.as_ref().clone(),
        })
    }

    fn exact_evaluation(
        &self,
        beta: &Array1<f64>,
    ) -> Result<Arc<SpatialAdaptiveExactEvaluation>, String> {
        {
            let cache = self
                .exact_eval_cache
                .lock()
                .map_err(|_| "spatial adaptive exact-evaluation cache lock poisoned".to_string())?;
            if let Some(cached) = cache.as_ref()
                && cached.beta.len() == beta.len()
                && cached
                    .beta
                    .iter()
                    .zip(beta.iter())
                    .all(|(&left, &right)| left == right)
            {
                return Ok(Arc::clone(&cached.eval));
            }
        }

        let eval = Arc::new(self.exact_evaluation_uncached(beta)?);
        let mut cache = self
            .exact_eval_cache
            .lock()
            .map_err(|_| "spatial adaptive exact-evaluation cache lock poisoned".to_string())?;
        *cache = Some(CachedSpatialAdaptiveExactEvaluation {
            beta: beta.clone(),
            eval: Arc::clone(&eval),
        });
        Ok(eval)
    }

    fn exacthessian_directional_derivative_from_evaluation(
        &self,
        beta: &Array1<f64>,
        eval: &SpatialAdaptiveExactEvaluation,
        direction: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        assert_eq!(
            beta.len(),
            direction.len(),
            "beta/direction length mismatch",
        );
        let d_eta = crate::faer_ndarray::fast_av(self.design.as_ref(), direction);
        let mut total = xt_diag_x_dense(
            self.design.view(),
            (&eval.obs.neghessian_eta_derivative * &d_eta).view(),
        )?;
        for (cache_idx, cache) in self.runtime_caches.iter().enumerate() {
            let params = self.adaptive_params.get(cache_idx).ok_or_else(|| {
                format!(
                    "missing adaptive parameter block for cache {}",
                    cache.termname
                )
            })?;
            let state = eval
                .adaptive_states
                .get(cache_idx)
                .ok_or_else(|| format!("missing adaptive state for cache {}", cache.termname))?;
            let direction_local = direction.slice(s![cache.coeff_global_range.clone()]);
            let d0_u = cache.d0.dot(&direction_local);
            let d1_u = cache.d1.dot(&direction_local);
            let d2_u = cache.d2.dot(&direction_local);
            let h0 =
                scalar_operatorhessian(&cache.d0, &state.magnitude.directionalhessian_diag(&d0_u))
                    .mapv(|v| params.lambda[0] * v);
            let hg = grouped_operatorhessian(
                &cache.d1,
                cache.dimension,
                &state.gradient.directionalhessian_blocks(
                    &collocationgradient_blocks(&d1_u, cache.dimension)
                        .map_err(|e| e.to_string())?,
                ),
            )
            .map_err(|e| e.to_string())?
            .mapv(|v| params.lambda[1] * v);
            let hc = grouped_operatorhessian(
                &cache.d2,
                cache.dimension * cache.dimension,
                &state.curvature.directionalhessian_blocks(
                    &collocationhessian_blocks(&d2_u, cache.dimension)
                        .map_err(|e| e.to_string())?,
                ),
            )
            .map_err(|e| e.to_string())?
            .mapv(|v| params.lambda[2] * v);
            let range = cache.coeff_global_range.clone();
            let mut local = total.slice_mut(s![range.clone(), range]);
            local += &h0;
            local += &hg;
            local += &hc;
        }
        Ok(total)
    }

    /// Exact second directional derivative `D²_β H[u, v]` of the joint
    /// (likelihood + adaptive Charbonnier penalty) Hessian, needed so the outer
    /// LAML's joint-Jeffreys curvature drift `D_β H_Φ[β̇]` is exact rather than
    /// silently dropped (which leaves the outer hypergradient inconsistent with
    /// the `½log|H+H_Φ|` objective it folds `H_Φ` into).
    ///
    /// The data block contributes `Xᵀ diag(ℓ'''(η_i) (Xu)_i (Xv)_i) X`, where
    /// `ℓ'''` is the third derivative of the per-observation log-likelihood in
    /// `η`. The observation state exposes the working weight `w=−ℓ''` and its
    /// first `η`-derivative `w'` (`neghessian_eta_derivative`) but not `w''`, so
    /// the exact data term is available only on the **constant-weight** path
    /// (`w' ≡ 0`, e.g. Gaussian identity), where `w'' ≡ 0` and the data block
    /// second derivative vanishes. On a varying-weight family we return `None`
    /// (the safe, pre-existing behavior: the drift degrades to zero rather than
    /// to a wrong value) until the observation contract carries `w''`.
    ///
    /// The penalty block is always exact: with `λ_m G_mᵀ B_m(G_m β) G_m` the
    /// per-component penalty Hessian, `D²_β` is `λ_m Σ_k G_mᵀ N_m,k G_m` using the
    /// scalar (`second_directionalhessian_diag`) / grouped
    /// (`second_directionalhessian_blocks`) fourth-derivative contractions.
    fn exacthessian_second_directional_derivative_from_evaluation(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        direction_u: &Array1<f64>,
        direction_v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let p = self.design.ncols();
        // Data block: exact only when the working weight is constant in η.
        if eval.obs.neghessian_eta_derivative.iter().any(|&w| w != 0.0) {
            return Ok(None);
        }
        let mut total = Array2::<f64>::zeros((p, p));
        for (cache_idx, cache) in self.runtime_caches.iter().enumerate() {
            let params = self.adaptive_params.get(cache_idx).ok_or_else(|| {
                format!(
                    "missing adaptive parameter block for cache {}",
                    cache.termname
                )
            })?;
            let state = eval
                .adaptive_states
                .get(cache_idx)
                .ok_or_else(|| format!("missing adaptive state for cache {}", cache.termname))?;
            let u_local = direction_u.slice(s![cache.coeff_global_range.clone()]);
            let v_local = direction_v.slice(s![cache.coeff_global_range.clone()]);

            // Magnitude (scalar d0).
            let q0_u = cache.d0.dot(&u_local);
            let q0_v = cache.d0.dot(&v_local);
            let h0 = scalar_operatorhessian(
                &cache.d0,
                &state.magnitude.second_directionalhessian_diag(&q0_u, &q0_v),
            )
            .mapv(|x| params.lambda[0] * x);

            // Gradient (grouped d1, block dim = dimension).
            let a1 = collocationgradient_blocks(&cache.d1.dot(&u_local), cache.dimension)
                .map_err(|e| e.to_string())?;
            let b1 = collocationgradient_blocks(&cache.d1.dot(&v_local), cache.dimension)
                .map_err(|e| e.to_string())?;
            let hg = grouped_operatorhessian(
                &cache.d1,
                cache.dimension,
                &state.gradient.second_directionalhessian_blocks(&a1, &b1),
            )
            .map_err(|e| e.to_string())?
            .mapv(|x| params.lambda[1] * x);

            // Curvature (grouped d2, block dim = dimension²).
            let a2 = collocationhessian_blocks(&cache.d2.dot(&u_local), cache.dimension)
                .map_err(|e| e.to_string())?;
            let b2 = collocationhessian_blocks(&cache.d2.dot(&v_local), cache.dimension)
                .map_err(|e| e.to_string())?;
            let hc = grouped_operatorhessian(
                &cache.d2,
                cache.dimension * cache.dimension,
                &state.curvature.second_directionalhessian_blocks(&a2, &b2),
            )
            .map_err(|e| e.to_string())?
            .mapv(|x| params.lambda[2] * x);

            let range = cache.coeff_global_range.clone();
            let mut local = total.slice_mut(s![range.clone(), range]);
            local += &h0;
            local += &hg;
            local += &hc;
        }
        Ok(Some(total))
    }
}

impl CustomFamily for SpatialAdaptiveExactFamily {
    // Preserve the pre-gam#1395 behavior: the trait default flipped to OFF (the
    // flat-prior exact-Newton objective carries no Jeffreys term), so families
    // that historically armed the term by default opt back in explicitly.
    fn joint_jeffreys_term_required(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = &expect_single_block_state(block_states, "spatial adaptive exact family")?.beta;
        let eval = self.exact_evaluation(beta)?;
        let mut gradient = fast_atv(&self.design, &eval.obs.score);
        gradient -= &eval.total_penaltygradient();
        let mut hessian = xt_diag_x_dense(self.design.view(), eval.obs.neghessian_eta.view())?;
        hessian += &eval.total_penaltyhessian();
        Ok(FamilyEvaluation {
            log_likelihood: eval.obs.log_likelihood - eval.total_penalty_value(),
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            }],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        let state = expect_single_block_state(block_states, "spatial adaptive exact family")?;
        let beta = &state.beta;
        let obs = evaluate_standard_familyobservations(
            self.family.clone(),
            self.latent_cloglog_state.as_ref(),
            self.mixture_link_state.as_ref(),
            self.sas_link_state.as_ref(),
            &self.y,
            &self.weights,
            &state.eta,
        )
        .map_err(|e| e.to_string())?;
        let adaptive_penalty = self.adaptive_penalty_value_only(beta)?;
        let (fixed_quadratic, _) = self.fixed_quadratic_terms(beta);
        Ok(obs.log_likelihood - adaptive_penalty - fixed_quadratic)
    }

    fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
        ExactNewtonOuterObjective::StrictPseudoLaplace
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = &expect_single_block_state(block_states, "spatial adaptive exact family")?.beta;
        let eval = self.exact_evaluation(beta)?;
        Ok(Some(eval.totalobjectivehessian(&self.design)?))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        expect_block_idx_zero(block_idx, "spatial adaptive exact family", "")?;
        self.exact_newton_joint_hessian_directional_derivative(block_states, d_beta)
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = &expect_single_block_state(block_states, "spatial adaptive exact family")?.beta;
        if d_beta_flat.len() != beta.len() {
            return Err(SmoothError::dimension_mismatch(format!(
                "spatial adaptive exact family direction length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                beta.len()
            ))
            .into());
        }
        let eval = self.exact_evaluation(beta)?;
        Ok(Some(
            self.exacthessian_directional_derivative_from_evaluation(beta, &eval, d_beta_flat)?,
        ))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = &expect_single_block_state(block_states, "spatial adaptive exact family")?.beta;
        if d_beta_u_flat.len() != beta.len() || d_betav_flat.len() != beta.len() {
            return Err(SmoothError::dimension_mismatch(format!(
                "spatial adaptive exact family second-direction length mismatch: got ({}, {}), expected {}",
                d_beta_u_flat.len(),
                d_betav_flat.len(),
                beta.len()
            ))
            .into());
        }
        let eval = self.exact_evaluation(beta)?;
        self.exacthessian_second_directional_derivative_from_evaluation(
            &eval,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(!block_states.is_empty(), "block_states must be non-empty");
        assert!(
            !block_spec.name.is_empty(),
            "block spec name must be non-empty",
        );
        expect_block_idx_zero(block_idx, "spatial adaptive exact family", "")?;
        Ok(self.linear_constraints.clone())
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if block_states.len() != 1 || specs.len() != 1 || derivative_blocks.len() != 1 {
            return Err(SmoothError::dimension_mismatch(format!(
                "spatial adaptive exact family expects one block/state/spec/psi payload, got states={} specs={} deriv_blocks={}",
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ))
            .into());
        }
        derivative_blocks[0]
            .get(psi_index)
            .ok_or_else(|| format!("adaptive psi index {} out of bounds", psi_index))?;
        let hyper = self
            .hyperspecs
            .get(psi_index)
            .ok_or_else(|| format!("adaptive psi index {} out of bounds", psi_index))?;
        let beta = &block_states[0].beta;
        let eval = self.exact_evaluation(beta)?;
        let (direct, beta_mixed, betahessian_explicit) =
            self.adaptive_hyper_parts(&eval, *hyper)?;

        // Exact pseudo-Laplace psi-gradient.
        //
        // For one hyperparameter coordinate a we use the exact formula
        //
        //   d/da L_tilde
        //   = J_a + 0.5 tr(H^{-1} Hdot_a),
        //
        // with
        //
        //   H u_a   = J_{beta,a},
        //   beta_a  = -u_a,
        //   Hdot_a  = J_{beta,beta,a} + D_beta(H)[beta_a]
        //           = J_{beta,beta,a} - D_beta(H)[u_a].
        //
        // Here:
        //   - `direct` is J_a,
        //   - `beta_mixed` is J_{beta,a},
        //   - `betahessian_explicit` is J_{beta,beta,a},
        //   - `exacthessian_directional_derivative_from_evaluation(..., u)` returns
        //     D_beta(H)[u] for the exact likelihood-plus-Charbonnier model.
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi: direct,
            score_psi: beta_mixed,
            hessian_psi: betahessian_explicit,
            hessian_psi_operator: None,
        }))
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        if block_states.len() != 1 || specs.len() != 1 || derivative_blocks.len() != 1 {
            return Err(SmoothError::dimension_mismatch(format!(
                "spatial adaptive exact family expects one block/state/spec/psi payload, got states={} specs={} deriv_blocks={}",
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ))
            .into());
        }
        derivative_blocks[0]
            .get(psi_i)
            .ok_or_else(|| format!("adaptive psi index {} out of bounds", psi_i))?;
        derivative_blocks[0]
            .get(psi_j)
            .ok_or_else(|| format!("adaptive psi index {} out of bounds", psi_j))?;
        let hyper_i = self
            .hyperspecs
            .get(psi_i)
            .ok_or_else(|| format!("adaptive psi index {} out of bounds", psi_i))?;
        let hyper_j = self
            .hyperspecs
            .get(psi_j)
            .ok_or_else(|| format!("adaptive psi index {} out of bounds", psi_j))?;
        let beta = &block_states[0].beta;
        let eval = self.exact_evaluation(beta)?;
        let (objective_psi_psi, score_psi_psi, hessian_psi_psi) =
            self.adaptive_explicit_second_order_parts(&eval, *hyper_i, *hyper_j)?;

        Ok(Some(
            crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
                objective_psi_psi,
                score_psi_psi,
                hessian_psi_psi,
                hessian_psi_psi_operator: None,
            },
        ))
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 1 || specs.len() != 1 || derivative_blocks.len() != 1 {
            return Err(SmoothError::dimension_mismatch(format!(
                "spatial adaptive exact family expects one block/state/spec/psi payload, got states={} specs={} deriv_blocks={}",
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ))
            .into());
        }
        let beta = &block_states[0].beta;
        if direction.len() != beta.len() {
            return Err(SmoothError::dimension_mismatch(format!(
                "spatial adaptive exact family direction length mismatch: got {}, expected {}",
                direction.len(),
                beta.len()
            ))
            .into());
        }
        derivative_blocks[0]
            .get(psi_index)
            .ok_or_else(|| format!("adaptive psi index {} out of bounds", psi_index))?;
        let hyper = self
            .hyperspecs
            .get(psi_index)
            .ok_or_else(|| format!("adaptive psi index {} out of bounds", psi_index))?;
        let eval = self.exact_evaluation(beta)?;
        let drift = match hyper.kind {
            SpatialAdaptiveHyperKind::LogLambdaMagnitude
            | SpatialAdaptiveHyperKind::LogLambdaGradient
            | SpatialAdaptiveHyperKind::LogLambdaCurvature => self.adaptive_block_drift_eval(
                &eval,
                hyper.cache_index,
                AdaptiveComponent::from_index(hyper.kind.component_index())?,
                HyperDriftKind::Rho,
                direction,
            )?,
            SpatialAdaptiveHyperKind::LogEpsilonMagnitude
            | SpatialAdaptiveHyperKind::LogEpsilonGradient
            | SpatialAdaptiveHyperKind::LogEpsilonCurvature => self
                .adaptive_shared_log_epsilon_drift(
                    &eval,
                    hyper.kind.component_index(),
                    direction,
                )?,
        };
        Ok(Some(drift))
    }
}

fn expect_single_block_state<'a>(
    block_states: &'a [ParameterBlockState],
    family_name: &str,
) -> Result<&'a ParameterBlockState, String> {
    crate::families::block_layout::block_count::validate_block_count::<SmoothError>(
        family_name,
        1,
        block_states.len(),
    )?;
    Ok(&block_states[0])
}

fn expect_block_idx_zero(block_idx: usize, family_name: &str, context: &str) -> Result<(), String> {
    if block_idx != 0 {
        return Err(SmoothError::invalid_index(format!(
            "{family_name} expects block_idx 0{context}, got {block_idx}"
        ))
        .into());
    }
    Ok::<(), _>(())
}

impl BoundedLinearFamily {
    fn bounded_term_derivative_data(
        &self,
        latent_beta: &Array1<f64>,
    ) -> (
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
    ) {
        let p = latent_beta.len();
        let mut beta_user = latent_beta.clone();
        let mut jac_diag = Array1::<f64>::ones(p);
        let mut second_diag = Array1::<f64>::zeros(p);
        let mut third_diag = Array1::<f64>::zeros(p);
        let mut priorthird = Array1::<f64>::zeros(p);
        for term in &self.bounded_terms {
            let (beta, _, db_dtheta, d2b_dtheta2, d3b_dtheta3) =
                bounded_latent_derivatives(latent_beta[term.col_idx], term.min, term.max);
            beta_user[term.col_idx] = beta;
            jac_diag[term.col_idx] = db_dtheta;
            second_diag[term.col_idx] = d2b_dtheta2;
            third_diag[term.col_idx] = d3b_dtheta3;
            let (_, _, _, prior_neghess_derivative) =
                bounded_prior_terms(latent_beta[term.col_idx], &term.prior);
            priorthird[term.col_idx] = prior_neghess_derivative;
        }
        (beta_user, jac_diag, second_diag, third_diag, priorthird)
    }

    fn user_beta_and_jacobian(&self, latent_beta: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let (beta_user, jac_diag, _, _, _) = self.bounded_term_derivative_data(latent_beta);
        (beta_user, jac_diag)
    }

    fn nonlinear_offset_from_latent(&self, latent_beta: &Array1<f64>) -> Array1<f64> {
        let mut offset = self.offset.clone();
        for term in &self.bounded_terms {
            let (beta, _, _) =
                bounded_latent_to_user(latent_beta[term.col_idx], term.min, term.max);
            offset.scaled_add(beta, &self.design.column(term.col_idx));
        }
        offset
    }

    fn effective_design_for_latent(&self, jac_diag: &Array1<f64>) -> Array2<f64> {
        let mut x_eff = self.design.clone();
        for term in &self.bounded_terms {
            x_eff
                .column_mut(term.col_idx)
                .mapv_inplace(|v| v * jac_diag[term.col_idx]);
        }
        x_eff
    }

    fn exacthessian_andgradient(
        &self,
        latent_beta: &Array1<f64>,
    ) -> Result<
        (
            StandardFamilyObservationState,
            Array2<f64>,
            Array1<f64>,
            f64,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
        ),
        String,
    > {
        let (_, jac_diag, second_diag, third_diag, priorthird) =
            self.bounded_term_derivative_data(latent_beta);
        let x_eff = self.effective_design_for_latent(&jac_diag);
        let eta =
            self.designzeroed.dot(latent_beta) + self.nonlinear_offset_from_latent(latent_beta);
        let obs = evaluate_standard_familyobservations(
            self.family.clone(),
            self.latent_cloglog_state.as_ref(),
            self.mixture_link_state.as_ref(),
            self.sas_link_state.as_ref(),
            &self.y,
            &self.weights,
            &eta,
        )
        .map_err(|e| e.to_string())?;

        let mut priorgrad = Array1::<f64>::zeros(latent_beta.len());
        let mut prior_neghess = Array2::<f64>::zeros((latent_beta.len(), latent_beta.len()));
        let mut prior_loglik = 0.0;
        for term in &self.bounded_terms {
            let (logp, grad, neghess, _) =
                bounded_prior_terms(latent_beta[term.col_idx], &term.prior);
            prior_loglik += logp;
            priorgrad[term.col_idx] += grad;
            prior_neghess[[term.col_idx, term.col_idx]] += neghess;
        }

        let mut hessian = xt_diag_x_dense(x_eff.view(), obs.neghessian_eta.view())?;
        let mut gradient = fast_atv(&x_eff, &obs.score);
        for term in &self.bounded_terms {
            let score_beta = self.design.column(term.col_idx).dot(&obs.score);
            hessian[[term.col_idx, term.col_idx]] -= score_beta * second_diag[term.col_idx];
        }
        hessian += &prior_neghess;
        gradient += &priorgrad;

        Ok((
            obs,
            hessian,
            gradient,
            prior_loglik,
            second_diag,
            third_diag,
            priorthird,
        ))
    }

    fn evaluation_from_latent(
        &self,
        latent_beta: &Array1<f64>,
    ) -> Result<
        (
            StandardFamilyObservationState,
            Array2<f64>,
            Array1<f64>,
            f64,
        ),
        String,
    > {
        let (obs, hessian, gradient, prior_loglik, _, _, _) =
            self.exacthessian_andgradient(latent_beta)?;
        Ok((obs, hessian, gradient, prior_loglik))
    }
}

impl CustomFamily for BoundedLinearFamily {
    // Preserve the pre-gam#1395 behavior: the trait default flipped to OFF (the
    // flat-prior exact-Newton objective carries no Jeffreys term), so families
    // that historically armed the term by default opt back in explicitly.
    fn joint_jeffreys_term_required(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let latent_beta = &expect_single_block_state(block_states, "bounded linear family")?.beta;
        let (obs, hessian, gradient, prior_loglik) = self.evaluation_from_latent(latent_beta)?;
        Ok(FamilyEvaluation {
            log_likelihood: obs.log_likelihood + prior_loglik,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            }],
        })
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let latent_beta = &expect_single_block_state(block_states, "bounded linear family")?.beta;
        let (_, hessian, _, _) = self.evaluation_from_latent(latent_beta)?;
        Ok(Some(hessian))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        expect_block_idx_zero(block_idx, "bounded linear family", "")?;
        self.exact_newton_joint_hessian_directional_derivative(block_states, d_beta)
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let latent_beta = &expect_single_block_state(block_states, "bounded linear family")?.beta;
        if d_beta_flat.len() != latent_beta.len() {
            return Err(SmoothError::dimension_mismatch(format!(
                "bounded linear family directional derivative length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                latent_beta.len()
            ))
            .into());
        }

        let (obs, _, _, _, second_diag, third_diag, priorthird) =
            self.exacthessian_andgradient(latent_beta)?;

        let (_, jac_diag, _, _, _) = self.bounded_term_derivative_data(latent_beta);
        let x_eff = self.effective_design_for_latent(&jac_diag);
        let deta = x_eff.dot(d_beta_flat);
        let d_neghess_eta = &obs.neghessian_eta_derivative * &deta;

        let mut dx_eff = Array2::<f64>::zeros(x_eff.raw_dim());
        for term in &self.bounded_terms {
            let scale = second_diag[term.col_idx] * d_beta_flat[term.col_idx];
            if scale != 0.0 {
                let mut col = dx_eff.column_mut(term.col_idx);
                col.assign(&self.design.column(term.col_idx));
                col.mapv_inplace(|v| v * scale);
            }
        }

        let mut dhessian = xt_diag_x_dense(x_eff.view(), d_neghess_eta.view())?;
        let mut wxdx = Array2::<f64>::zeros((x_eff.ncols(), x_eff.ncols()));
        for i in 0..x_eff.nrows() {
            let wi = obs.neghessian_eta[i];
            if wi == 0.0 {
                continue;
            }
            for a in 0..x_eff.ncols() {
                let xa = x_eff[[i, a]];
                for b in 0..x_eff.ncols() {
                    wxdx[[a, b]] += wi * (dx_eff[[i, a]] * x_eff[[i, b]] + xa * dx_eff[[i, b]]);
                }
            }
        }
        dhessian += &wxdx;

        let d_score = -&obs.neghessian_eta * &deta;
        for term in &self.bounded_terms {
            let score_beta = self.design.column(term.col_idx).dot(&obs.score);
            let d_score_beta = self.design.column(term.col_idx).dot(&d_score);
            dhessian[[term.col_idx, term.col_idx]] -= d_score_beta * second_diag[term.col_idx]
                + score_beta * third_diag[term.col_idx] * d_beta_flat[term.col_idx];
            dhessian[[term.col_idx, term.col_idx]] +=
                priorthird[term.col_idx] * d_beta_flat[term.col_idx];
        }

        Ok(Some(dhessian))
    }

    fn block_geometry(
        &self,
        block_states: &[ParameterBlockState],
        spec: &ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        if block_states.is_empty() {
            return Ok((
                DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    self.designzeroed.clone(),
                )),
                self.offset.clone(),
            ));
        }
        let offset = self.nonlinear_offset_from_latent(
            &expect_single_block_state(block_states, "bounded linear family")?.beta,
        );
        let x = if spec.design.ncols() == self.designzeroed.ncols() {
            self.designzeroed.clone()
        } else {
            return Err(SmoothError::dimension_mismatch(
                "bounded linear family design column mismatch",
            )
            .into());
        };
        Ok((
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
            offset,
        ))
    }

    fn block_geometry_is_dynamic(&self) -> bool {
        true
    }

    fn block_geometry_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
        d_beta: &Array1<f64>,
    ) -> Result<Option<BlockGeometryDirectionalDerivative>, String> {
        expect_block_idx_zero(
            block_idx,
            "bounded linear family",
            " for geometry derivative",
        )?;
        expect_single_block_state(block_states, "bounded linear family")?;
        if d_beta.len() != spec.design.ncols() {
            return Err(SmoothError::dimension_mismatch(format!(
                "bounded linear family geometry derivative direction mismatch: got {}, expected {}",
                d_beta.len(),
                spec.design.ncols()
            ))
            .into());
        }
        let (_, jac_diag, _, _, _) = self.bounded_term_derivative_data(&block_states[0].beta);
        let mut d_offset = Array1::<f64>::zeros(self.offset.len());
        let has_drift = self
            .bounded_terms
            .iter()
            .any(|term| jac_diag[term.col_idx] != 0.0 && d_beta[term.col_idx] != 0.0);
        if !has_drift {
            return Ok(Some(BlockGeometryDirectionalDerivative {
                d_design: None,
                d_offset,
            }));
        }
        for term in &self.bounded_terms {
            let col = term.col_idx;
            let drift = jac_diag[col] * d_beta[col];
            if drift != 0.0 {
                d_offset.scaled_add(drift, &self.design.column(col));
            }
        }
        Ok(Some(BlockGeometryDirectionalDerivative {
            d_design: None,
            d_offset,
        }))
    }
}

#[inline]
fn dense_diag_gram_chunkrows(p: usize) -> usize {
    const MIN_ROWS: usize = 512;
    const MAX_ROWS: usize = 2048;
    const TARGET_BYTES: usize = 2 * 1024 * 1024;
    let bytes_per_row = p.max(1) * std::mem::size_of::<f64>();
    (TARGET_BYTES / bytes_per_row).clamp(MIN_ROWS, MAX_ROWS)
}

fn xt_diag_x_dense(x: ArrayView2<'_, f64>, w: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
    if x.nrows() != w.len() {
        return Err(SmoothError::dimension_mismatch("xt_diag_x_dense row mismatch").into());
    }
    let (n, p) = x.dim();
    if n == 0 || p == 0 {
        return Ok(Array2::<f64>::zeros((p, p)));
    }

    const STREAMING_BYTES_THRESHOLD: usize = 8 * 1024 * 1024;
    let dense_work_bytes = n
        .checked_mul(p)
        .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>()))
        .unwrap_or(usize::MAX);
    if dense_work_bytes <= STREAMING_BYTES_THRESHOLD {
        let mut weighted = x.to_owned();
        ndarray::Zip::from(weighted.rows_mut())
            .and(w)
            .par_for_each(|mut row, wi| row *= *wi);
        return Ok(fast_atb(&x, &weighted));
    }

    let chunkrows = dense_diag_gram_chunkrows(p).min(n);
    let mut weighted_chunk = Array2::<f64>::zeros((chunkrows, p));
    let mut out = Array2::<f64>::zeros((p, p));
    for row_start in (0..n).step_by(chunkrows) {
        let rows = (n - row_start).min(chunkrows);
        let x_chunk = x.slice(s![row_start..row_start + rows, ..]);
        {
            let mut chunk = weighted_chunk.slice_mut(s![0..rows, ..]);
            for local_row in 0..rows {
                let scale = w[row_start + local_row];
                if scale == 0.0 {
                    chunk.row_mut(local_row).fill(0.0);
                    continue;
                }
                for col in 0..p {
                    chunk[[local_row, col]] = x_chunk[[local_row, col]] * scale;
                }
            }
        }
        out += &fast_atb(&x_chunk, &weighted_chunk.slice(s![0..rows, ..]));
    }
    Ok(out)
}

fn trace_of_dense_product(a: &Array2<f64>, b: &Array2<f64>) -> Result<f64, String> {
    if a.nrows() != a.ncols() || b.nrows() != b.ncols() || a.nrows() != b.nrows() {
        return Err(
            SmoothError::dimension_mismatch("trace_of_dense_product dimension mismatch").into(),
        );
    }
    let mut trace = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            trace += a[[i, j]] * b[[j, i]];
        }
    }
    Ok(trace)
}

fn exact_bounded_edf(
    penalties: &[PenaltySpec],
    lambdas: &Array1<f64>,
    latent_cov: &Array2<f64>,
) -> Result<(Vec<f64>, Vec<f64>, f64), EstimationError> {
    if penalties.len() != lambdas.len() {
        crate::bail_invalid_estim!(
            "bounded EDF penalty/lambda mismatch: {} penalties vs {} lambdas",
            penalties.len(),
            lambdas.len()
        );
    }
    if latent_cov.nrows() != latent_cov.ncols() {
        crate::bail_invalid_estim!("bounded EDF covariance must be square");
    }

    let p = latent_cov.nrows();
    let mut s_lambda = Array2::<f64>::zeros((p, p));
    let mut edf_by_block = Vec::with_capacity(penalties.len());
    // Raw per-block penalty trace tr_kk = λ_kk·tr(H⁻¹S_kk) (issue #1219).
    let mut penalty_block_trace = Vec::with_capacity(penalties.len());
    let mut trace_sum = 0.0;

    for (k, ps) in penalties.iter().enumerate() {
        let lambda_k = lambdas[k];
        match ps {
            PenaltySpec::Block {
                local, col_range, ..
            } => {
                s_lambda
                    .slice_mut(ndarray::s![col_range.clone(), col_range.clone()])
                    .scaled_add(lambda_k, local);
                // Compute penalty rank from the block-local matrix directly.
                let penalty_rank =
                    local
                        .nrows()
                        .saturating_sub(estimate_penalty_nullity(local).map_err(|e| {
                            EstimationError::InvalidInput(format!("bounded EDF rank failed: {e}"))
                        })?);
                // Trace only involves the block slice of latent_cov.
                let cov_block = latent_cov.slice(ndarray::s![col_range.clone(), col_range.clone()]);
                let trace_k = lambda_k
                    * trace_of_dense_product(&cov_block.to_owned(), local)
                        .map_err(EstimationError::InvalidInput)?;
                trace_sum += trace_k;
                penalty_block_trace.push(trace_k);
                let p_k = penalty_rank as f64;
                edf_by_block.push((p_k - trace_k).clamp(0.0, p_k));
            }
            PenaltySpec::Dense(m) | PenaltySpec::DenseWithMean { matrix: m, .. } => {
                s_lambda.scaled_add(lambda_k, m);
                let penalty_rank = p.saturating_sub(estimate_penalty_nullity(m).map_err(|e| {
                    EstimationError::InvalidInput(format!("bounded EDF rank failed: {e}"))
                })?);
                let trace_k = lambda_k
                    * trace_of_dense_product(latent_cov, m)
                        .map_err(EstimationError::InvalidInput)?;
                trace_sum += trace_k;
                penalty_block_trace.push(trace_k);
                let p_k = penalty_rank as f64;
                edf_by_block.push((p_k - trace_k).clamp(0.0, p_k));
            }
        }
    }

    let nullity_total = estimate_penalty_nullity(&s_lambda)
        .map_err(|e| EstimationError::InvalidInput(format!("bounded EDF nullity failed: {e}")))?
        as f64;
    let edf_total = (p as f64 - trace_sum).clamp(nullity_total, p as f64);
    Ok((edf_by_block, penalty_block_trace, edf_total))
}

/// Symmetric posterior-precision inverse for the bounded-coefficient path.
///
/// The penalised Hessian at a strict posterior maximum is SPD, so its inverse
/// is the posterior covariance. We eigendecompose the symmetric precision and
/// invert the positive-eigenvalue subspace, projecting out the (rare)
/// structural null directions a penalised model leaves flat rather than
/// δ-ridging them — the same honest pseudo-inverse contract the strict
/// pseudo-Laplace covariance uses (gam#748). A genuinely indefinite precision
/// (a negative eigenvalue beyond rounding) means the reported mode is not a
/// posterior maximum and is surfaced as a fit-quality error rather than
/// masked.
fn symmetric_positive_definite_inverse_or_pseudo(
    precision: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    use crate::faer_ndarray::FaerEigh;
    let p = precision.nrows();
    if precision.ncols() != p {
        crate::bail_invalid_estim!(
            "posterior precision inverse requires a square matrix, got {}x{}",
            precision.nrows(),
            precision.ncols()
        );
    }
    if p == 0 {
        return Ok(Array2::<f64>::zeros((0, 0)));
    }
    let symmetric = (precision + &precision.t().to_owned()) * 0.5;
    let (evals, evecs) = symmetric.eigh(faer::Side::Lower).map_err(|e| {
        EstimationError::InvalidInput(format!(
            "posterior precision eigendecomposition failed: {e}"
        ))
    })?;
    let max_abs_eval = evals.iter().fold(0.0_f64, |acc, &ev| acc.max(ev.abs()));
    let tol =
        (10.0 * f64::EPSILON * (p as f64) * (p as f64) * max_abs_eval).max(100.0 * f64::EPSILON);
    if let Some(&min_eval) = evals
        .iter()
        .filter(|&&ev| ev < -tol)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    {
        crate::bail_invalid_estim!(
            "bounded posterior precision is non-PD at the converged optimum (min eigenvalue \
             {min_eval:.6e} < -tol={tol:.6e}); the reported mode is not a strict posterior \
             maximum, so a covariance would be meaningless"
        );
    }
    // Σ = U diag(1/λ_+) Uᵀ over the positive-eigenvalue subspace.
    let mut scaled = evecs.clone();
    for (j, &ev) in evals.iter().enumerate() {
        let inv = if ev > tol { 1.0 / ev } else { 0.0 };
        scaled.column_mut(j).mapv_inplace(|v| v * inv);
    }
    let cov = scaled.dot(&evecs.t());
    Ok((&cov + &cov.t().to_owned()) * 0.5)
}

fn transform_bounded_latent_precision_to_user_internal(
    latent_precision: &Array2<f64>,
    jac_diag: &Array1<f64>,
) -> Result<Array2<f64>, EstimationError> {
    let p = latent_precision.nrows();
    if latent_precision.ncols() != p || jac_diag.len() != p {
        crate::bail_invalid_estim!(
            "bounded precision transform dimension mismatch: precision is {}x{}, jacobian has {} entries",
            latent_precision.nrows(),
            latent_precision.ncols(),
            jac_diag.len()
        );
    }
    let mut out = latent_precision.clone();
    for i in 0..p {
        let scale = jac_diag[i];
        if !scale.is_finite() || scale <= 0.0 {
            crate::bail_invalid_estim!(
                "bounded precision transform requires a positive finite coefficient jacobian; column {i} has {scale}"
            );
        }
        if scale != 1.0 {
            out.row_mut(i).mapv_inplace(|v| v / scale);
            out.column_mut(i).mapv_inplace(|v| v / scale);
        }
    }
    Ok(out)
}

fn fit_bounded_term_collection_with_design(
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    heuristic_lambdas: Option<&[f64]>,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    let conditioning_cols: Vec<usize> = spec
        .linear_terms
        .iter()
        .enumerate()
        .filter_map(|(j, linear)| {
            (!linear.double_penalty).then_some(design.intercept_range.end + j)
        })
        .collect();
    let conditioning = LinearFitConditioning::from_columns(design, &conditioning_cols);
    let dense_design = design.design.to_dense_cow();
    let fit_design = conditioning.apply_to_design(&dense_design);
    let fit_penalties = conditioning
        .transform_blockwise_penalties_to_internal(&design.penalties, design.design.ncols());
    if design.linear_constraints.is_some() {
        crate::bail_invalid_estim!(
            "bounded() terms are not yet compatible with explicit linear constraints"
        );
    }
    let mut bounded_terms = Vec::<BoundedLinearTermMeta>::new();
    for (j, term) in spec.linear_terms.iter().enumerate() {
        if term.double_penalty
            && matches!(
                term.coefficient_geometry,
                LinearCoefficientGeometry::Bounded { .. }
            )
        {
            crate::bail_invalid_estim!(
                "bounded linear term '{}' cannot also use double_penalty",
                term.name
            );
        }
        if let LinearCoefficientGeometry::Bounded { min, max, prior } =
            term.coefficient_geometry.clone()
        {
            let col_idx = design.intercept_range.end + j;
            let (min_internal, max_internal) = conditioning.internal_bounds_for(col_idx, min, max);
            bounded_terms.push(BoundedLinearTermMeta {
                col_idx,
                min: min_internal,
                max: max_internal,
                prior,
            });
        }
    }
    if bounded_terms.is_empty() {
        crate::bail_invalid_estim!("internal bounded fit path called with no bounded terms");
    }

    let mut designzeroed = fit_design.clone();
    let mut initial_beta = Array1::<f64>::zeros(fit_design.ncols());
    for term in &bounded_terms {
        designzeroed.column_mut(term.col_idx).fill(0.0);
        initial_beta[term.col_idx] = bounded_logit(0.5);
    }

    let initial_log_lambdas = heuristic_lambdas
        .map(|vals| Array1::from_vec(vals.to_vec()))
        .unwrap_or_else(|| Array1::zeros(fit_penalties.len()));
    if initial_log_lambdas.len() != fit_penalties.len() {
        crate::bail_invalid_estim!(
            "heuristic lambda length mismatch for bounded model: got {}, expected {}",
            initial_log_lambdas.len(),
            fit_penalties.len()
        );
    }

    let is_beta_logistic = family.is_binomial_beta_logistic();
    let family_adapter = BoundedLinearFamily {
        family: family.clone(),
        latent_cloglog_state: options.latent_cloglog,
        mixture_link_state: options
            .mixture_link
            .clone()
            .as_ref()
            .map(state_fromspec)
            .transpose()
            .map_err(EstimationError::InvalidInput)?,
        sas_link_state: options
            .sas_link
            .map(|spec| {
                if is_beta_logistic {
                    state_from_beta_logisticspec(spec)
                } else {
                    state_from_sasspec(spec)
                }
            })
            .transpose()
            .map_err(EstimationError::InvalidInput)?,
        y: y.to_owned(),
        weights: weights.to_owned(),
        design: fit_design.clone(),
        designzeroed: designzeroed.clone(),
        offset: offset.to_owned(),
        bounded_terms: bounded_terms.clone(),
    };
    let blockspec = ParameterBlockSpec {
        name: "eta".to_string(),
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(designzeroed)),
        offset: offset.to_owned(),
        penalties: fit_penalties
            .iter()
            .map(|ps| match ps {
                PenaltySpec::Block {
                    local, col_range, ..
                } => PenaltyMatrix::Blockwise {
                    local: local.clone(),
                    col_range: col_range.clone(),
                    total_dim: design.design.ncols(),
                },
                PenaltySpec::Dense(m) | PenaltySpec::DenseWithMean { matrix: m, .. } => {
                    PenaltyMatrix::Dense(m.clone())
                }
            })
            .collect(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas,
        initial_beta: Some(initial_beta),
        gauge_priority: 100,
        // Report the true β-dependent Jacobian (bounded columns scaled by
        // dβ/dθ) to the identifiability audit so it does not mistake the
        // deliberately-zeroed placeholder columns for a structural rank
        // deficiency. The inner solve still drives η through the family
        // adapter, so this does not affect the fit geometry.
        jacobian_callback: Some(Arc::new(BoundedEffectiveJacobian {
            design: fit_design.clone(),
            bounded_terms: bounded_terms.clone(),
        })),
        stacked_design: None,
        stacked_offset: None,
    };
    let fit = fit_custom_family(
        &family_adapter,
        &[blockspec],
        &BlockwiseFitOptions {
            inner_max_cycles: options.max_iter,
            inner_tol: options.tol,
            outer_max_iter: options.max_iter,
            outer_tol: options.tol,
            // The bounded path builds its own user-scale covariance below by
            // inverting the user-scale penalised Hessian (delta-method through
            // the bounded transform's Jacobian + the conditioning map), so it
            // does not consume the inner solver's optional canonical-space
            // `covariance_conditional`. Inverting the reported precision
            // directly guarantees `inv(penalized_hessian) == covariance` and
            // works on every bounded fit — including the common no-smoothing
            // path where the inner solve surfaces no covariance at all (the
            // gam#854 "bounded fit emits no user-scale covariance" symptom).
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        },
    )
    .map_err(EstimationError::CustomFamily)?;

    let latent_beta = fit.block_states[0].beta.clone();
    let (beta_user_internal, jac_diag) = family_adapter.user_beta_and_jacobian(&latent_beta);
    let beta_user = conditioning.backtransform_beta(&beta_user_internal);

    let (eta_state, h_data, _, _) = family_adapter
        .evaluation_from_latent(&latent_beta)
        .map_err(EstimationError::InvalidInput)?;
    let p_fit = fit_design.ncols();
    let mut s_lambda_internal = Array2::<f64>::zeros((p_fit, p_fit));
    for (k, penalty) in fit_penalties.iter().enumerate() {
        match penalty {
            PenaltySpec::Block {
                local, col_range, ..
            } => {
                s_lambda_internal
                    .slice_mut(ndarray::s![col_range.clone(), col_range.clone()])
                    .scaled_add(fit.lambdas[k], local);
            }
            PenaltySpec::Dense(m) | PenaltySpec::DenseWithMean { matrix: m, .. } => {
                s_lambda_internal.scaled_add(fit.lambdas[k], m);
            }
        }
    }
    let mut latent_precision = h_data.clone();
    latent_precision += &s_lambda_internal;
    let user_precision_internal =
        transform_bounded_latent_precision_to_user_internal(&latent_precision, &jac_diag)?;
    let penalized_hessian =
        conditioning.transform_penalized_hessian_to_original(&user_precision_internal);

    // User-scale posterior covariance via the delta method. The reported
    // geometry precision `penalized_hessian` is the user-scale penalized
    // Hessian `H_user = C⁻ᵀ J⁻¹ (H_latent + S_λ) J⁻¹ C⁻¹` (latent precision
    // pushed through the bounded transform's Jacobian `J = diag(dβ_user/dθ)`
    // and the conditioning map `C`). The user-scale covariance is its exact
    // inverse `H_user⁻¹`, which IS the delta-method pushforward of the latent
    // posterior covariance `(H_latent + S_λ)⁻¹`. Inverting the same matrix the
    // geometry reports guarantees `inv(penalized_hessian) == covariance`
    // exactly and removes the dependency on the inner solver's optional,
    // canonical-space `covariance_conditional` (which is `None` whenever the
    // bounded blockspec carries no smoothing parameters — the no-rho fit path
    // — leaving a bounded fit with a populated precision but no user-scale
    // covariance, the gam#854 symptom). The latent precision is SPD at a
    // strict posterior maximum; on a marginally-indefinite boundary Hessian we
    // invert the positive-eigenvalue subspace (the structural null space of a
    // penalised model is a flat posterior direction, not something to ridge
    // away), matching the strict-pseudo-Laplace covariance contract (gam#748).
    let beta_covariance = if options.compute_inference {
        Some(symmetric_positive_definite_inverse_or_pseudo(
            &penalized_hessian,
        )?)
    } else {
        None
    };
    let beta_standard_errors = beta_covariance
        .as_ref()
        .map(|cov| Array1::from_iter((0..cov.nrows()).map(|i| cov[[i, i]].max(0.0).sqrt())));
    // EDF `p − Σ_k λ_k tr(H_latent⁻¹ S_k)` is computed in the *latent*
    // (untransformed) coordinate system the penalties `fit_penalties` live in,
    // so it needs the latent posterior covariance `(H_latent + S_λ)⁻¹`, not the
    // user-scale one. Invert the same latent precision that produced the
    // reported user precision so the two are an exact transform pair.
    let latent_cov = if options.compute_inference {
        Some(symmetric_positive_definite_inverse_or_pseudo(
            &latent_precision,
        )?)
    } else {
        None
    };
    let s_lambda_original = weighted_blockwise_penalty_sum(
        &design.penalties,
        fit.lambdas.as_slice().unwrap(),
        design.design.ncols(),
    );
    let penalty_term = beta_user.dot(&s_lambda_original.dot(&beta_user));
    let deviance = if family.is_gaussian_identity() {
        y.iter()
            .zip(eta_state.mu.iter())
            .zip(weights.iter())
            .map(|((&yy, &mu), &w)| w.max(0.0) * (yy - mu) * (yy - mu))
            .sum()
    } else {
        -2.0 * eta_state.log_likelihood
    };
    let (edf_by_block, penalty_block_trace, edf_total) = if let Some(cov) = latent_cov.as_ref() {
        exact_bounded_edf(&fit_penalties, &fit.lambdas, cov)?
    } else {
        (
            vec![0.0; fit_penalties.len()],
            vec![0.0; fit_penalties.len()],
            0.0,
        )
    };
    let geometry = Some(crate::estimate::FitGeometry {
        penalized_hessian: penalized_hessian.clone().into(),
        working_weights: eta_state.fisherweight.clone(),
        working_response: {
            let mut working_response = eta_state.eta.clone();
            for i in 0..working_response.len() {
                let wi = eta_state.fisherweight[i].max(1e-12);
                working_response[i] += eta_state.score[i] / wi;
            }
            working_response
        },
    });
    let max_abs_eta = eta_state
        .eta
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    Ok(FittedTermCollection {
        fit: {
            let log_lambdas = fit.lambdas.mapv(|v| v.max(1e-300).ln());
            let inf = FitInference {
                edf_by_block,
                penalty_block_trace,
                edf_total,
                smoothing_correction: None,
                // Boundary adapter: `penalized_hessian` storage is now
                // `UnscaledPrecision`.
                penalized_hessian: penalized_hessian.clone().into(),
                working_weights: eta_state.fisherweight.clone(),
                working_response: {
                    let mut working_response = eta_state.eta.clone();
                    for i in 0..working_response.len() {
                        let wi = eta_state.fisherweight[i].max(1e-12);
                        working_response[i] += eta_state.score[i] / wi;
                    }
                    working_response
                },
                reparam_qs: None,
                dispersion: crate::estimate::Dispersion::Known(1.0),
                beta_covariance: beta_covariance
                    .clone()
                    .map(crate::inference::dispersion_cov::PhiScaledCovariance::from),
                beta_standard_errors,
                beta_covariance_corrected: None,
                beta_standard_errors_corrected: None,
                beta_covariance_frequentist: None,
                coefficient_influence: None,
                weighted_gram: None,
                bias_correction_beta: None,
            };
            let covariance_conditional = beta_covariance;
            let pirls_status_val = if fit.outer_converged {
                crate::pirls::PirlsStatus::Converged
            } else {
                crate::pirls::PirlsStatus::StalledAtValidMinimum
            };
            UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
                blocks: vec![crate::estimate::FittedBlock {
                    beta: beta_user.clone(),
                    role: crate::estimate::BlockRole::Mean,
                    edf: edf_total,
                    lambdas: fit.lambdas.clone(),
                }],
                log_lambdas,
                lambdas: fit.lambdas,
                likelihood_scale: family.default_scale_metadata(),
                likelihood_family: Some(family),
                log_likelihood_normalization:
                    crate::types::LogLikelihoodNormalization::UserProvided,
                log_likelihood: eta_state.log_likelihood,
                deviance,
                reml_score: fit.penalized_objective,
                stable_penalty_term: penalty_term,
                penalized_objective: fit.penalized_objective,
                used_device: false,
                outer_iterations: fit.outer_iterations,
                outer_converged: fit.outer_converged,
                outer_gradient_norm: fit.outer_gradient_norm,
                standard_deviation: 1.0,
                covariance_conditional,
                covariance_corrected: None,
                inference: Some(inf),
                fitted_link: crate::estimate::FittedLinkState::Standard(None),
                geometry,
                block_states: Vec::new(),
                pirls_status: pirls_status_val,
                max_abs_eta,
                constraint_kkt: None,
                artifacts: crate::estimate::FitArtifacts {
                    pirls: None,
                    ..Default::default()
                },
                inner_cycles: 0,
            })?
        },
        design: design.clone(),
        adaptive_diagnostics: None,
    })
}

fn enforce_term_constraint_feasibility(
    design: &TermCollectionDesign,
    fit: &UnifiedFitResult,
) -> Result<(), EstimationError> {
    // Geometric (per-row-scaled) tolerance, matching the public contract on
    // `ACTIVE_SET_PRIMAL_FEASIBILITY_TOL` and the diagnostic that
    // `compute_constraint_kkt_diagnostics` exposes via `fit.constraint_kkt`.
    // Lower-bound rows are unit-norm (a_i = e_i) so the scale-invariant and
    // raw checks coincide there. Linear-inequality rows generally are NOT
    // unit-norm — e.g. a B-spline endpoint-derivative clamp at k = 12 carries
    // ‖a_i‖ ≈ 38, so a 1e-6 raw residual is only 2.6e-8 in geometric units.
    // Holding this gate to raw 1e-7 while the in-solver acceptance gate
    // measures geometric 1e-8 is the inconsistency that made well-conditioned
    // clamped fits get rejected after they completed cleanly.
    /// Raw (unscaled) constraint-residual tolerance for the post-fit feasibility
    /// audit; kept loose enough to be consistent with the geometric in-solver
    /// acceptance gate on non-unit-norm linear-inequality rows (see comment).
    const CONSTRAINT_FEASIBILITY_RAW_TOL: f64 = 1e-7;
    let tol = CONSTRAINT_FEASIBILITY_RAW_TOL;
    let smooth_start = design
        .design
        .ncols()
        .saturating_sub(design.smooth.total_smooth_cols());
    let mut violations: Vec<String> = Vec::new();
    for term in &design.smooth.terms {
        let gr = (smooth_start + term.coeff_range.start)..(smooth_start + term.coeff_range.end);
        let beta_local = fit.beta.slice(s![gr.clone()]).to_owned();
        if let Some(lb) = term.lower_bounds_local.as_ref() {
            let mut worst = 0.0_f64;
            let mut worst_idx = 0usize;
            for i in 0..lb.len().min(beta_local.len()) {
                if lb[i].is_finite() {
                    let viol = (lb[i] - beta_local[i]).max(0.0);
                    if viol > worst {
                        worst = viol;
                        worst_idx = i;
                    }
                }
            }
            if worst > tol {
                violations.push(format!(
                    "term='{}' kind=lower-bound maxviolation={:.3e} coeff_index={}",
                    term.name, worst, worst_idx
                ));
            }
        }
        if let Some(lin) = term.linear_constraints_local.as_ref() {
            let mut worst = 0.0_f64;
            let mut worstrow = 0usize;
            for i in 0..lin.a.nrows() {
                let norm = lin.a.row(i).dot(&lin.a.row(i)).sqrt();
                let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
                let s = (lin.a.row(i).dot(&beta_local) - lin.b[i]) * inv;
                let viol = (-s).max(0.0);
                if viol > worst {
                    worst = viol;
                    worstrow = i;
                }
            }
            if worst > tol {
                violations.push(format!(
                    "term='{}' kind=linear-inequality maxviolation={:.3e} row={}",
                    term.name, worst, worstrow
                ));
            }
        }
    }

    if !violations.is_empty() {
        let mut msg = format!(
            "constraint violation after fit ({} violating term constraints): {}",
            violations.len(),
            violations.join(" | ")
        );
        if let Some(kkt) = fit.constraint_kkt.as_ref() {
            msg.push_str(&format!(
                "; KKT[primal={:.3e}, dual={:.3e}, comp={:.3e}, stat={:.3e}]",
                kkt.primal_feasibility, kkt.dual_feasibility, kkt.complementarity, kkt.stationarity
            ));
        }
        return Err(EstimationError::ParameterConstraintViolation(msg));
    }
    Ok(())
}

fn stratified_spatial_subsample(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
    target_size: usize,
) -> Vec<usize> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;

    let n = data.nrows();
    if n <= target_size {
        return (0..n).collect();
    }

    let spatial_cols: Option<Vec<usize>> =
        spec.smooth_terms.iter().find_map(|term| match &term.basis {
            SmoothBasisSpec::ThinPlate { feature_cols, .. }
            | SmoothBasisSpec::Matern { feature_cols, .. }
            | SmoothBasisSpec::Duchon { feature_cols, .. } => {
                if !feature_cols.is_empty() {
                    Some(feature_cols.clone())
                } else {
                    None
                }
            }
            _ => None,
        });

    let cols = match spatial_cols {
        Some(c) if !c.is_empty() => c,
        _ => {
            let mut rng = StdRng::seed_from_u64(spatial_subsample_seed(data, &[], target_size));
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);
            indices.truncate(target_size);
            indices.sort_unstable();
            return indices;
        }
    };
    let mut rng = StdRng::seed_from_u64(spatial_subsample_seed(data, &cols, target_size));

    let d = cols.len();
    let mut mins = vec![f64::INFINITY; d];
    let mut maxs = vec![f64::NEG_INFINITY; d];
    for i in 0..n {
        for (ax, &col) in cols.iter().enumerate() {
            let v = data[[i, col]];
            if v < mins[ax] {
                mins[ax] = v;
            }
            if v > maxs[ax] {
                maxs[ax] = v;
            }
        }
    }

    // Aim for roughly this many sampled points per stratification cell so each
    // occupied cell can contribute a representative draw without collapsing the
    // grid to one point per cell.
    const TARGET_POINTS_PER_CELL: usize = 5;
    let total_cells_target = (target_size / TARGET_POINTS_PER_CELL).max(1);
    let cells_per_axis = ((total_cells_target as f64).powf(1.0 / d as f64)).ceil() as usize;
    let cells_per_axis = cells_per_axis.max(1);

    let mut cell_members: std::collections::HashMap<Vec<usize>, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        let mut cell_key = Vec::with_capacity(d);
        for (ax, &col) in cols.iter().enumerate() {
            let range = maxs[ax] - mins[ax];
            let cell = if range <= 0.0 {
                0
            } else {
                let frac = (data[[i, col]] - mins[ax]) / range;
                (frac * cells_per_axis as f64).floor() as usize
            };
            cell_key.push(cell.min(cells_per_axis - 1));
        }
        cell_members.entry(cell_key).or_default().push(i);
    }

    let mut selected: Vec<usize> = Vec::with_capacity(target_size);
    let mut remaining_budget = target_size;
    let mut remaining_population = n;

    let mut cells: Vec<(Vec<usize>, Vec<usize>)> = cell_members.into_iter().collect();
    cells.sort_by(|a, b| a.0.cmp(&b.0));

    for (_, members) in &mut cells {
        if remaining_budget == 0 {
            break;
        }
        let alloc = ((members.len() as f64 / remaining_population as f64) * remaining_budget as f64)
            .round() as usize;
        let alloc = alloc.max(1).min(members.len()).min(remaining_budget);
        members.shuffle(&mut rng);
        selected.extend_from_slice(&members[..alloc]);
        remaining_budget = remaining_budget.saturating_sub(alloc);
        remaining_population = remaining_population.saturating_sub(members.len());
    }

    if selected.len() > target_size {
        selected.shuffle(&mut rng);
        selected.truncate(target_size);
    }

    selected.sort_unstable();
    selected
}

fn spatial_subsample_seed(
    data: ArrayView2<'_, f64>,
    spatial_cols: &[usize],
    target_size: usize,
) -> u64 {
    let mut state = 0x5350_4154_4941_4C53_u64;
    spatial_seed_mix(&mut state, data.nrows() as u64);
    spatial_seed_mix(&mut state, data.ncols() as u64);
    spatial_seed_mix(&mut state, target_size as u64);
    spatial_seed_mix(&mut state, spatial_cols.len() as u64);
    for &col in spatial_cols {
        spatial_seed_mix(&mut state, col as u64);
    }

    if data.nrows() > 0 {
        let mid = data.nrows() / 2;
        let last = data.nrows() - 1;
        for &row in &[0usize, mid, last] {
            for &col in spatial_cols {
                let value = data[[row, col]];
                spatial_seed_mix(&mut state, value.to_bits());
            }
        }
    }
    state
}

#[inline]
fn spatial_seed_mix(state: &mut u64, value: u64) {
    // Canonical SplitMix64 step over `value + state` (the step adds G itself),
    // then an extra rotate-multiply avalanche unique to the spatial seed mix.
    let mut s = value.wrapping_add(*state);
    let z = crate::linalg::utils::splitmix64(&mut s);
    *state ^= z;
    *state = (*state).rotate_left(27).wrapping_mul(0x3C79_AC49_2BA7_B653);
}

fn sampled_rows(data: ArrayView2<'_, f64>, indices: &[usize]) -> Array2<f64> {
    let mut sampled = Array2::<f64>::zeros((indices.len(), data.ncols()));
    for (new_row, &orig_row) in indices.iter().enumerate() {
        sampled.row_mut(new_row).assign(&data.row(orig_row));
    }
    sampled
}

fn spatial_term_user_centers(term: &SmoothTermSpec) -> Option<ArrayView2<'_, f64>> {
    match spatial_term_center_strategy(term) {
        Some(CenterStrategy::UserProvided(centers)) => Some(centers.view()),
        _ => None,
    }
}

fn finite_centered_axis_contrasts(values: &[f64], expected_dim: usize) -> Option<Vec<f64>> {
    if values.len() != expected_dim || expected_dim <= 1 {
        return None;
    }
    if values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(center_aniso_log_scales(values))
}

fn blended_pilot_axis_contrasts(
    pilot_data: ArrayView2<'_, f64>,
    term: &SmoothTermSpec,
    centers: ArrayView2<'_, f64>,
) -> Option<Vec<f64>> {
    let d = centers.ncols();
    if d <= 1 {
        return None;
    }
    let center_eta = initial_aniso_contrasts(centers);
    let data_eta = standardized_spatial_term_data(pilot_data, term)
        .ok()
        .and_then(|x| finite_centered_axis_contrasts(&initial_aniso_contrasts(x.view()), d));
    let center_eta = finite_centered_axis_contrasts(&center_eta, d)?;
    let blended = match data_eta {
        Some(data_eta) => center_eta
            .iter()
            .zip(data_eta.iter())
            .map(|(&from_centers, &from_data)| 0.5 * (from_centers + from_data))
            .collect::<Vec<_>>(),
        None => center_eta,
    };
    finite_centered_axis_contrasts(&blended, d)
}

fn apply_pilot_spatial_psi_reseed(
    pilot_data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
    spatial_terms: &[usize],
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<TermCollectionSpec, EstimationError> {
    let dims_per_term = spatial_dims_per_term(spec, spatial_terms);
    let use_aniso = has_aniso_terms(spec, spatial_terms);
    let log_kappa0 = if use_aniso {
        SpatialLogKappaCoords::from_length_scales_aniso(spec, spatial_terms, kappa_options)
    } else {
        SpatialLogKappaCoords::from_length_scales(spec, spatial_terms, kappa_options)
    };
    let log_kappa0 = log_kappa0.reseed_from_data(pilot_data, spec, spatial_terms, kappa_options);
    let log_kappa_lower = if use_aniso {
        SpatialLogKappaCoords::lower_bounds_aniso_from_data(
            pilot_data,
            spec,
            spatial_terms,
            &dims_per_term,
            kappa_options,
        )
    } else {
        SpatialLogKappaCoords::lower_bounds_from_data(
            pilot_data,
            spec,
            spatial_terms,
            kappa_options,
        )
    };
    let log_kappa_upper = if use_aniso {
        SpatialLogKappaCoords::upper_bounds_aniso_from_data(
            pilot_data,
            spec,
            spatial_terms,
            &dims_per_term,
            kappa_options,
        )
    } else {
        SpatialLogKappaCoords::upper_bounds_from_data(
            pilot_data,
            spec,
            spatial_terms,
            kappa_options,
        )
    };
    log_kappa0
        .clamp_to_bounds(&log_kappa_lower, &log_kappa_upper)
        .apply_tospec(spec, spatial_terms)
}

pub(crate) fn apply_spatial_anisotropy_pilot_initializer(
    data: ArrayView2<'_, f64>,
    spec: &mut TermCollectionSpec,
    spatial_terms: &[usize],
    target_size: usize,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> usize {
    if target_size == 0 || data.nrows() <= target_size.saturating_mul(2) || spatial_terms.is_empty()
    {
        return 0;
    }
    if !has_aniso_terms(spec, spatial_terms) {
        return 0;
    }
    let indices = stratified_spatial_subsample(data, spec, target_size);
    let pilot_data = sampled_rows(data, &indices);
    let mut working = spec.clone();
    let mut updated_terms = 0usize;
    const GEOMETRY_UPDATES: usize = 2;

    for pass in 0..GEOMETRY_UPDATES {
        let planned_terms = match plan_joint_spatial_centers_for_term_blocks(
            pilot_data.view(),
            &[working.smooth_terms.clone()],
        )
        .and_then(|mut blocks| {
            blocks.pop().ok_or_else(|| {
                BasisError::InvalidInput(
                    "pilot geometry initializer produced no smooth-term block".to_string(),
                )
            })
        }) {
            Ok(terms) => terms,
            Err(err) => {
                log::warn!(
                    "[spatial-kappa] pilot geometry initializer skipped after center planning failed: {err}"
                );
                return updated_terms;
            }
        };

        for &term_idx in spatial_terms {
            let Some(current_eta) = get_spatial_aniso_log_scales(&working, term_idx) else {
                continue;
            };
            let Some(d) = get_spatial_feature_dim(&working, term_idx) else {
                continue;
            };
            if d <= 1 || current_eta.len() != d {
                continue;
            }
            let Some(planned_term) = planned_terms.get(term_idx) else {
                continue;
            };
            let Some(centers) = spatial_term_user_centers(planned_term) else {
                continue;
            };
            let Some(eta) = blended_pilot_axis_contrasts(pilot_data.view(), planned_term, centers)
            else {
                continue;
            };
            if set_spatial_aniso_log_scales(&mut working, term_idx, eta).is_ok() {
                updated_terms += usize::from(pass == 0);
            }
        }

        match apply_pilot_spatial_psi_reseed(
            pilot_data.view(),
            &working,
            spatial_terms,
            kappa_options,
        ) {
            Ok(updated) => {
                working = updated;
            }
            Err(err) => {
                log::warn!(
                    "[spatial-kappa] pilot geometry ψ reseed skipped after deterministic initializer error: {err}"
                );
                break;
            }
        }
    }

    if updated_terms > 0 {
        log::info!(
            "[spatial-kappa] initialized anisotropy from {}-row pilot geometry for {} spatial term(s); proceeding to full-data optimization",
            indices.len(),
            updated_terms
        );
        *spec = working;
    }
    updated_terms
}

pub(crate) fn spatial_length_scale_term_indices(spec: &TermCollectionSpec) -> Vec<usize> {
    spec.smooth_terms
        .iter()
        .enumerate()
        .filter_map(|(idx, _)| spatial_term_supports_hyper_optimization(spec, idx).then_some(idx))
        .collect()
}

/// Returns `true` when every spatial term in `spec` has a locked kernel
/// scale (explicit `length_scale=X` without anisotropy) and therefore
/// contributes no outer ψ/κ optimization axis. Empty term collections
/// also return `true` — there are no kappas to optimize.
///
/// Used by family entry points that want to honor a user-supplied scalar
/// length scale exactly: when all spatial terms are locked the n-block
/// joint-spatial outer solver has nothing to optimize, and routing
/// through it merely spends ~80 outer iters chasing a stalled ARC at the
/// user's chosen ρ. Skipping straight to the rho-only path avoids that
/// waste and respects the user's explicit kernel-scale input.
pub fn all_spatial_terms_kappa_fixed(spec: &TermCollectionSpec) -> bool {
    spec.smooth_terms.iter().enumerate().all(|(idx, _)| {
        !spatial_term_supports_hyper_optimization(spec, idx)
            || spatial_term_has_locked_kappa(spec, idx)
    })
}

fn fit_score(fit: &UnifiedFitResult) -> f64 {
    if fit.reml_score.is_finite() {
        return fit.reml_score;
    }
    let score = 0.5 * fit.deviance + 0.5 * fit.stable_penalty_term;
    if score.is_finite() {
        score
    } else {
        f64::INFINITY
    }
}

/// Classify an outer-evaluation error as a *recoverable trial-point
/// infeasibility* versus a genuine fatal failure.
///
/// The spatial-κ / anisotropy outer optimizer probes a sequence of trial
/// hyperparameters. At an extreme trial point the realized kernel design or
/// its ψ-derivatives may simply be non-constructible — e.g. a learned
/// per-axis log-scale stretches the anisotropic distance `r = |Λh|` until the
/// Duchon polyharmonic blocks `r^(2m−d)` overflow, or a degenerate metric
/// collapses two centers onto a non-C² collision. Those points lie outside
/// the model's feasible domain; the principled response is to treat them like
/// the cost-only path already does (objective `+∞`) so the line-search /
/// trust-region solver retreats, rather than aborting the entire REML fit.
///
/// A `BasisError` is exactly this class: it means "the basis/design cannot be
/// built at this hyperparameter". Everything else (PIRLS divergence wired to a
/// distinct variant, layout/topology invariants, over-parameterization) stays
/// fatal so genuine bugs are never masked.
fn is_recoverable_trial_point_error(err: &EstimationError) -> bool {
    matches!(err, EstimationError::BasisError(_))
}

fn require_successful_spatial_optimization_result<T>(
    initial_score: f64,
    result: Result<Option<(T, f64)>, EstimationError>,
) -> Result<T, EstimationError> {
    match result {
        Ok(Some((value, exact_score))) => {
            // Allow rounding-level worsening: REML scores accumulate
            // log-determinant terms whose finite-precision re-evaluation
            // can drift well past 1e-10 absolute near a converged optimum
            // (we have seen ~1e-6 between two evaluations whose printed
            // values round to identical 6-digit scientific). Reject genuine
            // worsenings (>1 unit) but admit anything within ~1e-6
            // absolute / 1e-8 relative — meaningful REML gains are
            // orders of magnitude larger.
            const SCORE_DRIFT_ABS_TOL: f64 = 1e-6;
            const SCORE_DRIFT_REL_TOL: f64 = 1e-8;
            let tol = SCORE_DRIFT_ABS_TOL.max(initial_score.abs() * SCORE_DRIFT_REL_TOL);
            if exact_score <= initial_score + tol {
                Ok(value)
            } else {
                Err(EstimationError::RemlOptimizationFailed(format!(
                    "spatial kappa optimization made REML score worse ({initial_score:.6e} -> {exact_score:.6e})"
                )))
            }
        }
        Ok(None) => Err(EstimationError::RemlOptimizationFailed(
            "spatial kappa optimization is unavailable for one or more eligible spatial terms"
                .to_string(),
        )),
        Err(err) => Err(EstimationError::RemlOptimizationFailed(format!(
            "spatial kappa optimization failed: {err}"
        ))),
    }
}

fn external_opts_for_design(
    family: &LikelihoodSpec,
    design: &TermCollectionDesign,
    options: &FitOptions,
) -> ExternalOptimOptions {
    ExternalOptimOptions {
        family: family.clone(),
        latent_cloglog: options.latent_cloglog,
        mixture_link: options.mixture_link.clone(),
        optimize_mixture: options.optimize_mixture,
        sas_link: options.sas_link,
        optimize_sas: options.optimize_sas,
        compute_inference: options.compute_inference,
        skip_rho_posterior_inference: options.skip_rho_posterior_inference,
        max_iter: options.max_iter,
        tol: options.tol,
        nullspace_dims: design.nullspace_dims.clone(),
        linear_constraints: design.linear_constraints.clone(),
        firth_bias_reduction: Some(options.firth_bias_reduction),
        penalty_shrinkage_floor: options.penalty_shrinkage_floor,
        rho_prior: options.rho_prior.clone(),
        // Propagate Kronecker structure so the joint optimizer minimizes the
        // same REML surface as the baseline/refit (adaptive_fit_options_base).
        kronecker_penalty_system: design.kronecker_penalty_system(),
        kronecker_factored: design
            .smooth
            .terms
            .iter()
            .find_map(|t| t.kronecker_factored.clone()),
        persist_warm_start_disk: options.persist_warm_start_disk,
    }
}

/// Evaluate the joint REML cost, gradient, and Hessian result at a given θ = [ρ, ψ]
/// for a single-block term collection with spatial hyperparameters.
///
/// This provides a direct evaluation of the profiled REML objective using the
/// external-caller interface, which exposes exact cost/gradient/Hessian without
/// running the full outer smoothing loop. The returned tuple is
/// `(cost, gradient, hessian)` in the joint [ρ, ψ] space.
fn evaluate_joint_reml_outer_eval_at_theta(
    evaluator: &mut crate::estimate::ExternalJointHyperEvaluator<'_>,
    design: &TermCollectionDesign,
    theta: &Array1<f64>,
    rho_dim: usize,
    hyper_dirs: Vec<crate::estimate::reml::DirectionalHyperParam>,
    warm_start_beta: Option<ArrayView1<'_, f64>>,
    order: crate::solver::rho_optimizer::OuterEvalOrder,
    design_revision: Option<u64>,
) -> Result<
    (
        f64,
        Array1<f64>,
        crate::solver::rho_optimizer::HessianResult,
    ),
    EstimationError,
> {
    evaluator.evaluate_with_order(
        &design.design,
        &design.penalties,
        &design.nullspace_dims,
        design.linear_constraints.clone(),
        theta,
        rho_dim,
        hyper_dirs,
        warm_start_beta,
        "evaluate_joint_reml_outer_eval_at_theta",
        order,
        design_revision,
    )
}

fn evaluate_joint_reml_efs_at_theta(
    evaluator: &mut crate::estimate::ExternalJointHyperEvaluator<'_>,
    design: &TermCollectionDesign,
    theta: &Array1<f64>,
    rho_dim: usize,
    hyper_dirs: Vec<crate::estimate::reml::DirectionalHyperParam>,
    warm_start_beta: Option<ArrayView1<'_, f64>>,
    design_revision: Option<u64>,
) -> Result<crate::solver::rho_optimizer::EfsEval, EstimationError> {
    evaluator.evaluate_efs(
        &design.design,
        &design.penalties,
        &design.nullspace_dims,
        design.linear_constraints.clone(),
        theta,
        rho_dim,
        hyper_dirs,
        warm_start_beta,
        "evaluate_joint_reml_efs_at_theta",
        design_revision,
    )
}

fn exact_joint_spatial_outer_hessian_available(
    family: &LikelihoodSpec,
    design: &TermCollectionDesign,
) -> bool {
    // Every `LikelihoodSpec` variant (Gaussian, Binomial-*, Poisson, Gamma,
    // Royston-Parmar) routes through the unified evaluator's outer-Hessian
    // path: Gaussian Identity uses the no-correction dense form, all GLM
    // variants supply scalar-GLM derivative ingredients consumed by
    // `compute_outer_hessian` / `build_outer_hessian_operator`, and the
    // (n, p, K) crossover in `prefer_outer_hessian_operator` chooses the
    // matrix-free `HessianResult::Operator` representation at large scale
    // for dense-lazy designs.  The previous `Identity || sparse_design`
    // gate predates that operator routing and forced binomial+logit+Matern
    // (and any other non-Gaussian dense-lazy spatial design) onto the
    // gradient-only BFGS path even though analytic Hessian is fully
    // available — capability check, not cost.  Match every variant
    // explicitly so any future family addition (which may not yet provide
    // outer-Hessian ingredients) forces an authoring decision here rather
    // than silently inheriting `true`.
    // Every supported response (Gaussian, Binomial-*, Poisson, Tweedie,
    // NegativeBinomial, Beta, Gamma, Royston-Parmar) routes through the
    // unified evaluator's outer-Hessian path; the spec-level capability
    // check therefore always succeeds. Match every response explicitly so
    // any future family addition (which may not yet provide outer-Hessian
    // ingredients) forces an authoring decision here rather than silently
    // inheriting `true`.
    let family_supported = match &family.response {
        ResponseFamily::Gaussian
        | ResponseFamily::Binomial
        | ResponseFamily::Poisson
        | ResponseFamily::Tweedie { .. }
        | ResponseFamily::NegativeBinomial { .. }
        | ResponseFamily::Beta { .. }
        | ResponseFamily::Gamma
        | ResponseFamily::RoystonParmar => true,
    };
    // A design with zero columns has no joint outer-Hessian to compute;
    // the analytic path is only meaningful for non-empty parameter blocks.
    family_supported && design.design.ncols() > 0
}

fn smooth_term_penalty_index(
    spec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    term_idx: usize,
) -> Option<usize> {
    if term_idx >= design.smooth.terms.len() || term_idx >= spec.smooth_terms.len() {
        return None;
    }
    if design.smooth.terms[term_idx].penalties_local.is_empty() {
        return None;
    }
    let linear_penalties = spec
        .linear_terms
        .iter()
        .filter(|t| t.double_penalty)
        .count()
        * 2;
    let random_penalties = design
        .random_effect_ranges
        .iter()
        .filter(|(_, range)| !range.is_empty())
        .count();
    let smooth_offset = linear_penalties + random_penalties;
    let local_offset = design
        .smooth
        .terms
        .iter()
        .take(term_idx)
        .map(|term| term.penalties_local.len())
        .sum::<usize>();
    Some(smooth_offset + local_offset)
}

fn try_build_spatial_term_log_kappa_derivativeinfo(
    data: ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    term_idx: usize,
) -> Result<Option<SpatialPsiDerivative>, EstimationError> {
    let Some((
        global_range,
        total_p,
        x_psi_local,
        s_psi_local_check,
        x_psi_psi_local,
        s_psi_psi_local,
        s_psi_components_local,
        s_psi_psi_components_local,
        implicit_operator,
    )) = try_build_spatial_term_log_kappa_derivative(data, resolvedspec, design, term_idx)?
    else {
        return Ok(None);
    };
    let Some(penalty_start) = smooth_term_penalty_index(resolvedspec, design, term_idx) else {
        return Ok(None);
    };
    if s_psi_components_local.is_empty() || s_psi_psi_components_local.is_empty() {
        return Ok(None);
    }
    if s_psi_components_local.len() != s_psi_psi_components_local.len() {
        return Ok(None);
    }
    let penalty_indices = (0..s_psi_components_local.len())
        .map(|j| penalty_start + j)
        .collect::<Vec<_>>();
    let penalty_index = penalty_indices[0];
    if s_psi_local_check.nrows() == 0 || s_psi_psi_local.nrows() == 0 {
        return Ok(None);
    }
    Ok(Some(SpatialPsiDerivative {
        penalty_index,
        penalty_indices,
        global_range,
        total_p,
        x_psi_local,
        s_psi_components_local,
        x_psi_psi_local,
        s_psi_psi_components_local,
        aniso_group_id: None,
        aniso_cross_designs: None,
        aniso_cross_penalty_provider: None,
        implicit_operator,
        implicit_axis: 0,
    }))
}

pub(crate) fn try_build_spatial_log_kappa_derivativeinfo_list(
    data: ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    spatial_terms: &[usize],
) -> Result<Option<Vec<SpatialPsiDerivative>>, EstimationError> {
    let mut out = Vec::new();
    let mut aniso_gid = 0usize;
    for &term_idx in spatial_terms {
        if spatial_term_uses_per_axis_psi(resolvedspec, term_idx) {
            if let Some(entries) = try_build_spatial_term_log_kappa_aniso_derivativeinfos(
                data,
                resolvedspec,
                design,
                term_idx,
                aniso_gid,
            )? {
                aniso_gid += 1;
                out.extend(entries);
                continue;
            } else {
                return Ok(None);
            }
        }
        let Some(info) =
            try_build_spatial_term_log_kappa_derivativeinfo(data, resolvedspec, design, term_idx)?
        else {
            return Ok(None);
        };
        out.push(info);
    }
    Ok(Some(out))
}

/// For an aniso term with d axes, produce d `SpatialPsiDerivative` entries.
fn try_build_spatial_term_log_kappa_aniso_derivativeinfos(
    data: ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    term_idx: usize,
    aniso_group_id: usize,
) -> Result<Option<Vec<SpatialPsiDerivative>>, EstimationError> {
    let Some(smooth_term) = design.smooth.terms.get(term_idx) else {
        return Ok(None);
    };
    let Some(termspec) = resolvedspec.smooth_terms.get(term_idx) else {
        return Ok(None);
    };
    let mut aniso_result = match &termspec.basis {
        SmoothBasisSpec::Sphere { .. } => return Ok(None),
        SmoothBasisSpec::Matern {
            feature_cols,
            spec,
            input_scales,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
            }
            build_matern_basis_log_kappa_aniso_derivatives(x.view(), spec)
                .map_err(EstimationError::from)?
        }
        // Measure-jet: the grouped dial coordinates ride the same per-axis
        // carrier. The producer runs on the FROZEN spec (the driver runs
        // post-freeze), so per-trial rebuilds move only the dials; the
        // coordinate layout, zero design drift, and shared candidate
        // normalization are owned by `build_measure_jet_basis_psi_derivatives`.
        SmoothBasisSpec::MeasureJet {
            feature_cols,
            spec,
            input_scales,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
            }
            build_measure_jet_basis_psi_derivatives(x.view(), spec)
                .map_err(EstimationError::from)?
        }
        _ => return Ok(None),
    };
    // Get number of axes from the shared operator when available; otherwise
    // fall back to the dense design list.
    let d = if let Some(ref op) = aniso_result.implicit_operator {
        op.n_axes()
    } else if !aniso_result.design_first.is_empty() {
        aniso_result.design_first.len()
    } else {
        0
    };
    if d == 0 {
        return Ok(None);
    }
    let Some(penalty_start) = smooth_term_penalty_index(resolvedspec, design, term_idx) else {
        return Ok(None);
    };
    let p_total = design.design.ncols();
    let smooth_start = p_total.saturating_sub(design.smooth.total_smooth_cols());
    let global_range = (smooth_start + smooth_term.coeff_range.start)
        ..(smooth_start + smooth_term.coeff_range.end);
    let num_penalties = aniso_result.penalties_first[0].len();
    let penalty_indices: Vec<usize> = (0..num_penalties).map(|j| penalty_start + j).collect();
    let penalties_cross_provider = aniso_result.penalties_cross_provider.clone();

    // Dense first/diagonal-second matrices may be present even when the shared
    // operator is available. The operator remains the canonical source for
    // exact cross-axis second derivatives.
    let use_implicit_design = aniso_result.design_first.is_empty();
    let implicit_op_arc = aniso_result
        .implicit_operator
        .as_ref()
        .map(|op| std::sync::Arc::new(op.clone()));

    let mut entries = Vec::with_capacity(d);
    for a in 0..d {
        let (x_psi_local, x_psi_psi_local) = if use_implicit_design {
            // Implicit path: design-derivative matvecs will be dispatched through
            // the ImplicitDerivativeOp inside HyperDesignDerivative, so we do NOT
            // need to materialize the dense (n x p) matrices here.  Store empty
            // placeholders — they are never read when the implicit operator is
            // present (spatial_log_kappa_hyper_dirs_frominfo_list uses from_implicit).
            (Array2::<f64>::zeros((0, 0)), Array2::<f64>::zeros((0, 0)))
        } else {
            // Move the dense (n × p) matrices out of aniso_result instead of
            // cloning. Each axis index `a` is read exactly once across the
            // loop, and aniso_result is dropped at function exit, so leaving
            // empty placeholders behind in those vec slots is safe.
            let x_first = std::mem::take(&mut aniso_result.design_first[a]);
            let x_second = std::mem::take(&mut aniso_result.design_second_diag[a]);
            if x_first.ncols() != smooth_term.coeff_range.len() {
                return Ok(None);
            }
            (x_first, x_second)
        };
        let s_psi_components = std::mem::take(&mut aniso_result.penalties_first[a]);
        let s_psi_psi_components = std::mem::take(&mut aniso_result.penalties_second_diag[a]);
        // Build cross-design entries for other axes b != a in this group.
        // These will be indexed by (b, cross_matrix) where b is the axis
        // offset within the d-entry block.
        // Cross-axis second derivatives are sourced from the shared operator,
        // so we only need placeholder entries to preserve the axis layout.
        let cross_designs = if implicit_op_arc.is_some() {
            let mut cd = Vec::with_capacity(d - 1);
            for b in 0..d {
                if b == a {
                    continue;
                }
                cd.push((b, Array2::<f64>::zeros((0, 0))));
            }
            cd
        } else if !aniso_result.design_second_cross.is_empty() {
            let mut cd = Vec::new();
            for (cross_idx, &(pa, pb)) in aniso_result.design_second_cross_pairs.iter().enumerate()
            {
                if pa == a {
                    cd.push((pb, aniso_result.design_second_cross[cross_idx].clone()));
                } else if pb == a {
                    cd.push((pa, aniso_result.design_second_cross[cross_idx].clone()));
                }
            }
            cd
        } else {
            Vec::new()
        };
        let cross_penalty_provider = if d > 1 {
            let penalties_cross_provider = penalties_cross_provider.clone();
            Some(std::sync::Arc::new(
                move |b_axis: usize| -> Result<Vec<Array2<f64>>, EstimationError> {
                    if b_axis == a {
                        return Ok(Vec::new());
                    }
                    let (axis_lo, axis_hi) = if a < b_axis { (a, b_axis) } else { (b_axis, a) };
                    if let Some(provider) = penalties_cross_provider.as_ref() {
                        provider
                            .evaluate(axis_lo, axis_hi)
                            .map_err(EstimationError::from)
                    } else {
                        // No provider: either the pair is unregistered, or it
                        // was registered without data (early-return raw-operator
                        // paths). Both cases contribute no cross penalties.
                        Ok(Vec::new())
                    }
                },
            )
                as std::sync::Arc<
                    dyn Fn(usize) -> Result<Vec<Array2<f64>>, EstimationError>
                        + Send
                        + Sync
                        + 'static,
                >)
        } else {
            None
        };

        entries.push(SpatialPsiDerivative {
            penalty_index: penalty_indices[0],
            penalty_indices: penalty_indices.clone(),
            global_range: global_range.clone(),
            total_p: p_total,
            x_psi_local,
            s_psi_components_local: s_psi_components,
            x_psi_psi_local,
            s_psi_psi_components_local: s_psi_psi_components,
            aniso_group_id: Some(aniso_group_id),
            aniso_cross_designs: if cross_designs.is_empty() {
                None
            } else {
                Some(cross_designs)
            },
            aniso_cross_penalty_provider: cross_penalty_provider,
            implicit_operator: implicit_op_arc.clone(),
            implicit_axis: a,
        });
    }
    Ok(Some(entries))
}
