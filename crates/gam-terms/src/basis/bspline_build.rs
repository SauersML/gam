use super::*;

/// Generate a B-spline knot vector spanning a 1-D seed sample.
///
/// Family-agnostic knot-generation helper relocated DOWN into `gam-terms`
/// under #1521 (was `gam_models::wiggle::initializewiggle_knots_from_seed`): it
/// only drives the basis builder's `KnotSource::Generate` path and carries no
/// model-family type, so the wiggle-bearing families (gamlss / bms /
/// transformation-normal) consume it from the basis layer instead of reaching
/// across the family stack. A degenerate (near-constant) seed is widened to a
/// fixed half-range so the generated knots stay well-conditioned.
pub fn initializewiggle_knots_from_seed(
    seed: ArrayView1<'_, f64>,
    degree: usize,
    num_internal_knots: usize,
) -> Result<Array1<f64>, String> {
    const MIN_WIGGLE_SEED_SPAN: f64 = 1e-8;
    const DEFAULT_WIGGLE_HALF_RANGE: f64 = 3.0;

    let mut seed_min = seed.iter().copied().fold(f64::INFINITY, f64::min);
    let mut seed_max = seed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !seed_min.is_finite() || !seed_max.is_finite() {
        return Err("non-finite seed for wiggle knot initialization".to_string());
    }
    if (seed_max - seed_min).abs() < MIN_WIGGLE_SEED_SPAN {
        let center = 0.5 * (seed_min + seed_max);
        seed_min = center - DEFAULT_WIGGLE_HALF_RANGE;
        seed_max = center + DEFAULT_WIGGLE_HALF_RANGE;
    }
    let (_, knots) = create_basis::<Dense>(
        seed,
        KnotSource::Generate {
            data_range: (seed_min, seed_max),
            num_internal_knots,
        },
        degree,
        BasisOptions::value(),
    )
    .map_err(|e| e.to_string())?;
    Ok(knots)
}

pub fn select_centers_by_strategy(
    data: ArrayView2<'_, f64>,
    strategy: &CenterStrategy,
) -> Result<Array2<f64>, BasisError> {
    match strategy {
        CenterStrategy::Auto(inner) => select_centers_by_strategy(data, inner.as_ref()),
        CenterStrategy::UserProvided(centers) => {
            if centers.ncols() != data.ncols() {
                crate::bail_dim_basis!(
                    "user centers have {} columns but data has {}",
                    centers.ncols(),
                    data.ncols()
                );
            }
            if centers.nrows() == 0 {
                crate::bail_invalid_basis!("user-provided center list cannot be empty");
            }
            Ok(centers.clone())
        }
        CenterStrategy::EqualMass { num_centers } => select_equal_mass_centers(data, *num_centers),
        CenterStrategy::EqualMassCovarRepresentative { num_centers } => {
            select_equal_mass_covar_representative_centers(data, *num_centers)
        }
        CenterStrategy::FarthestPoint { num_centers } => {
            select_thin_plate_knots(data, *num_centers)
        }
        CenterStrategy::KMeans {
            num_centers,
            max_iter,
        } => select_kmeans_centers(data, *num_centers, *max_iter),
        CenterStrategy::UniformGrid { points_per_dim } => {
            select_uniform_grid_centers(data, *points_per_dim)
        }
    }
}

/// Generic 1D B-spline builder returning design + penalty list.
pub fn build_bspline_basis_1d(
    data: ArrayView1<'_, f64>,
    spec: &BSplineBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    // Natural cubic regression spline (bs="cr"/"cs", #1074): a dense
    // value-at-knot basis with its own roughness penalty, not a B-spline
    // difference penalty. Route to the dedicated builder BEFORE the B-spline-only
    // auto-shrink and periodic logic so neither touches a cr spec.
    if let BSplineKnotSpec::NaturalCubicRegression { knots } = &spec.knotspec {
        return build_cubic_regression_basis_1d(data, spec, knots);
    }

    if let OneDimensionalBoundary::Cyclic { start, end } = spec.boundary
        && end <= start
    {
        return Err(BasisError::InvalidRange(start, end));
    }

    // Issue #340: auto-shrink the requested (degree, num_internal_knots) when
    // `n = data.len()` is too small for the user's request. The shrink only
    // touches the auto/data-driven knot specs — when the caller provides an
    // explicit clamped knot vector or periodic geometry, we respect it
    // verbatim (their knots already encode a deliberate degree choice).
    let (spec_owned, auto_shrink_note) = maybe_auto_shrink_bspline_spec(spec, data.len());
    let spec = &spec_owned;

    let periodic_build = match &spec.knotspec {
        BSplineKnotSpec::PeriodicUniform {
            data_range,
            num_basis,
        } => {
            if let Some((boundary_start, boundary_end, _)) = spec.boundary.period() {
                let scale = (boundary_end - boundary_start).abs().max(1.0);
                let tol = 1e-12 * scale;
                if (data_range.0 - boundary_start).abs() > tol
                    || (data_range.1 - boundary_end).abs() > tol
                {
                    crate::bail_invalid_basis!(
                        "periodic B-spline knot range ({}, {}) conflicts with cyclic boundary ({}, {})",
                        data_range.0,
                        data_range.1,
                        boundary_start,
                        boundary_end
                    );
                }
            }
            Some((data_range.0, data_range.1, *num_basis))
        }
        _ => spec.boundary.period().map(|(start, end, _)| {
            let num_basis = match &spec.knotspec {
                BSplineKnotSpec::Generate {
                    num_internal_knots, ..
                } => num_internal_knots + spec.degree + 1,
                BSplineKnotSpec::Automatic {
                    num_internal_knots, ..
                } => {
                    num_internal_knots.unwrap_or_else(|| {
                        default_internal_knot_count_for_data(data.len(), spec.degree)
                    }) + spec.degree
                        + 1
                }
                BSplineKnotSpec::Provided(knots) => knots.len().saturating_sub(spec.degree + 1),
                // cr is routed away by the early dispatch; its basis dimension
                // equals the knot count (no degree offset).
                BSplineKnotSpec::NaturalCubicRegression { knots } => knots.len(),
                BSplineKnotSpec::PeriodicUniform { .. } => {
                    // Filtered upstream by the outer match arm; if we ever
                    // reach this branch, the upstream filter is broken.
                    // Surface a debug-assert in test builds and fall back
                    // to 0 in release so the build does not panic.
                    assert!(
                        false,
                        "PeriodicUniform knotspec should have been handled by the outer match arm"
                    );
                    0
                }
            };
            (start, end, num_basis)
        }),
    };

    if let Some((start, end, num_basis)) = periodic_build {
        if spec.degree < 1 {
            crate::bail_invalid_basis!(
                "cyclic P-splines require degree >= 1, got degree={}",
                spec.degree
            );
        }
        if !spec.boundary_conditions.is_free() {
            crate::bail_invalid_basis!(
                "periodic B-splines cannot also declare endpoint boundary conditions"
            );
        }
        let knots = cyclic_uniform_knot_vector(start, end, spec.degree, num_basis);
        let s_bend_raw = create_cyclic_difference_penalty_matrix(num_basis, spec.penalty_order)?;
        // A cyclic difference penalty has a single null direction — the constant
        // vector — and that direction is removed wholesale by the periodic
        // sum-to-zero identifiability constraint applied below
        // (`apply_bspline_identifiability_policy` / streaming equivalent). The
        // null-space-shrinkage ("double") penalty is, by construction, the
        // projector `z·zᵀ` onto exactly that constant eigenvector, so after the
        // constraint transform `T` (whose columns span the sum-to-zero subspace,
        // orthogonal to the constant) it becomes `Tᵀ(z·zᵀ)T = 0` — an identically
        // zero penalty carrying its own smoothing parameter. A zero penalty block
        // contributes nothing to the REML cost or penalty log-determinant, so its
        // log-λ coordinate is completely unidentified: the outer REML objective is
        // flat along it and the outer Hessian is singular. The outer optimizer
        // then cannot certify a step in that direction and the loop fails to
        // terminate at the (otherwise converged) optimum (#874). mgcv's `bs="cc"`
        // is likewise a SINGLE-penalty smooth for the same reason. Emit only the
        // wiggliness penalty for the cyclic basis regardless of `double_penalty`:
        // there is no free polynomial null space left to shrink.
        //
        // Frobenius-normalize the cyclic wiggliness penalty (recording the norm
        // in `normalization_scale`) so its smoothing parameter `λ` is on the same
        // unit-Frobenius scale as every other basis (cr / duchon / tensor / the
        // open-knot ps path, #1365). The shipped design penalty is `β'(S/c)β`; a
        // raw `S` (scale 1.0) put `λ` on a basis-dependent scale and the outer
        // λ-search heuristics under-smoothed exactly as for the open ps single
        // penalty. Fit-invariant at the REML optimum (only `λ̂` rescales by `c`).
        let (s_bend_norm, s_bend_scale) = normalize_penalty(&s_bend_raw);
        let penalties_raw = vec![PenaltyCandidate {
            matrix: s_bend_norm,
            nullspace_dim_hint: 1,
            source: PenaltySource::Primary,
            normalization_scale: s_bend_scale,
            kronecker_factors: None,
            op: None,
        }];
        let penalties_raw_mats = penalties_raw
            .iter()
            .map(|candidate| candidate.matrix.clone())
            .collect();
        let auto_chunk = auto_streaming_chunk_size_for_dense(data.len(), num_basis);
        let (design, transformed_candidates, identifiability_transform) =
            if let Some(chunk) = auto_chunk {
                log::info!(
                    "B-spline basis auto-streaming evaluator: n={} p={} chunk_size={}",
                    data.len(),
                    num_basis,
                    chunk,
                );
                build_streaming_bspline_design_and_candidates(
                    data,
                    &knots,
                    spec.degree,
                    Some((start, end - start, num_basis)),
                    &spec.identifiability,
                    penalties_raw,
                    penalties_raw_mats,
                    Some(chunk),
                )?
            } else {
                let (basis, _) =
                    create_cyclic_bspline_basis_dense(data, start, end, spec.degree, num_basis)?;
                let (design_c, penalty_mats, identifiability_transform) =
                    apply_bspline_identifiability_policy(
                        basis,
                        penalties_raw_mats,
                        &knots,
                        spec.degree,
                        &spec.identifiability,
                    )?;
                let transformed_candidates = penalty_mats
                    .into_iter()
                    .zip(penalties_raw)
                    .map(|(matrix, candidate)| PenaltyCandidate {
                        nullspace_dim_hint: candidate.nullspace_dim_hint,
                        matrix,
                        source: candidate.source,
                        normalization_scale: candidate.normalization_scale,
                        kronecker_factors: None,
                        op: None,
                    })
                    .collect();
                (
                    DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(design_c)),
                    transformed_candidates,
                    identifiability_transform,
                )
            };
        let transformed_candidates =
            rebuild_double_penalty_nullspace_in_constrained_chart(transformed_candidates)?;
        let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
            filter_active_penalty_candidates_with_ops(renormalize_constrained_penalty_candidates(
                transformed_candidates,
            ))?;
        return Ok(BasisBuildResult {
            design,
            penalties,
            nullspace_dims,
            penaltyinfo,
            metadata: BasisMetadata::BSpline1D {
                knots,
                identifiability_transform,
                periodic: Some((start, end - start, num_basis)),
                degree: Some(spec.degree),
                auto_shrink_note: auto_shrink_note.clone(),
            },
            kronecker_factored: None,
            ops,
            null_eigenvectors,
            joint_null_rotation: None,
        });
    }
    // Auto-streaming decision for non-periodic B-spline: we need `p_raw` to
    // size the dense buffer estimate, which requires materializing the knot
    // vector first. We build it here (cheap relative to a full N×P design)
    // and check the auto threshold; if streaming activates we use the same
    // knots/penalties downstream, otherwise we fall through to the regular
    // dense/sparse path below (which will build its own knots — that path is
    // shared with several knot-spec shapes and is not worth refactoring).
    let auto_chunk_streaming = if spec.boundary_conditions.is_free() {
        let knots_for_estimate = match &spec.knotspec {
            BSplineKnotSpec::Generate {
                data_range,
                num_internal_knots,
            } => Some(internal::generate_full_knot_vector(
                *data_range,
                *num_internal_knots,
                spec.degree,
            )?),
            BSplineKnotSpec::Provided(knots) => Some(knots.clone()),
            // cr is routed away by the early dispatch; the knots index the basis
            // directly, so they are the estimate set verbatim.
            BSplineKnotSpec::NaturalCubicRegression { knots } => Some(knots.clone()),
            BSplineKnotSpec::Automatic {
                num_internal_knots,
                placement,
            } => {
                let inferred = num_internal_knots.unwrap_or_else(|| {
                    default_internal_knot_count_for_data(data.len(), spec.degree)
                });
                Some(match placement {
                    BSplineKnotPlacement::Uniform => {
                        let range = finite_data_range(data)?;
                        internal::generate_full_knot_vector(range, inferred, spec.degree)?
                    }
                    BSplineKnotPlacement::Quantile => {
                        internal::generate_full_knot_vector_quantile(data, inferred, spec.degree)?
                    }
                })
            }
            BSplineKnotSpec::PeriodicUniform { .. } => None,
        };
        match knots_for_estimate {
            Some(knots_est) => {
                let p_raw_est = knots_est
                    .len()
                    .checked_sub(spec.degree + 1)
                    .ok_or_else(|| {
                        BasisError::InvalidInput(
                            "invalid B-spline knot/degree combination".to_string(),
                        )
                    })?;
                auto_streaming_chunk_size_for_dense(data.len(), p_raw_est)
                    .map(|chunk| (knots_est, p_raw_est, chunk))
            }
            None => None,
        }
    } else {
        None
    };
    if let Some((knots, p_raw, chunk)) = auto_chunk_streaming {
        let greville_for_penalty = penalty_greville_abscissae_for_knots(&knots, spec.degree)?;
        let s_bend_raw = create_difference_penalty_matrix(
            p_raw,
            spec.penalty_order,
            greville_for_penalty.as_ref().map(|g| g.view()),
        )?;
        let penalties_raw = bspline_penalty_candidates(&s_bend_raw, spec, &knots)?;
        let penalties_raw_mats = penalties_raw
            .iter()
            .map(|candidate| candidate.matrix.clone())
            .collect();
        log::info!(
            "B-spline basis auto-streaming evaluator: n={} p={} chunk_size={}",
            data.len(),
            p_raw,
            chunk,
        );
        let (design, transformed_candidates, identifiability_transform) =
            build_streaming_bspline_design_and_candidates(
                data,
                &knots,
                spec.degree,
                None,
                &spec.identifiability,
                penalties_raw,
                penalties_raw_mats,
                Some(chunk),
            )?;
        let transformed_candidates =
            rebuild_double_penalty_nullspace_in_constrained_chart(transformed_candidates)?;
        let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
            filter_active_penalty_candidates_with_ops(renormalize_constrained_penalty_candidates(
                transformed_candidates,
            ))?;
        return Ok(BasisBuildResult {
            design,
            penalties,
            nullspace_dims,
            penaltyinfo,
            metadata: BasisMetadata::BSpline1D {
                knots,
                identifiability_transform,
                periodic: None,
                degree: Some(spec.degree),
                auto_shrink_note: auto_shrink_note.clone(),
            },
            kronecker_factored: None,
            ops,
            null_eigenvectors,
            joint_null_rotation: None,
        });
    }
    // Non-free endpoint boundary conditions are structural: they must be baked
    // into the raw B-spline coefficient chart before identifiability handling.
    // The sparse path is reserved for free endpoints; non-free endpoints take
    // the dense path below, where the boundary nullspace transform is composed
    // into the stored raw-basis identifiability transform.
    let prefer_sparse_design = spec.boundary_conditions.is_free()
        && matches!(
            spec.identifiability,
            BSplineIdentifiability::None | BSplineIdentifiability::WeightedSumToZero { .. }
        );
    let (design_sparse_opt, design_dense_opt, knots) = if prefer_sparse_design {
        match &spec.knotspec {
            BSplineKnotSpec::Generate {
                data_range,
                num_internal_knots,
            } => {
                let (basis, knots) = create_basis::<Sparse>(
                    data,
                    KnotSource::Generate {
                        data_range: *data_range,
                        num_internal_knots: *num_internal_knots,
                    },
                    spec.degree,
                    BasisOptions::value(),
                )?;
                (Some(basis), None, knots)
            }
            BSplineKnotSpec::NaturalCubicRegression { knots } => {
                // Unreachable in practice (the early dispatch returns the cr
                // basis), but keeps this match exhaustive and self-consistent.
                return build_cubic_regression_basis_1d(data, spec, knots);
            }
            BSplineKnotSpec::Provided(knots) => {
                let (basis, knots) = create_basis::<Sparse>(
                    data,
                    KnotSource::Provided(knots.view()),
                    spec.degree,
                    BasisOptions::value(),
                )?;
                (Some(basis), None, knots)
            }
            BSplineKnotSpec::PeriodicUniform { .. } => {
                crate::bail_invalid_basis!(
                    "periodic B-spline must be handled before storage selection; \
                     this branch is reserved for non-periodic knot specs"
                        .to_string(),
                );
            }
            BSplineKnotSpec::Automatic {
                num_internal_knots,
                placement,
            } => {
                let inferred = num_internal_knots.unwrap_or_else(|| {
                    default_internal_knot_count_for_data(data.len(), spec.degree)
                });
                let knots = match placement {
                    BSplineKnotPlacement::Uniform => {
                        let range = finite_data_range(data)?;
                        internal::generate_full_knot_vector(range, inferred, spec.degree)?
                    }
                    BSplineKnotPlacement::Quantile => {
                        internal::generate_full_knot_vector_quantile(data, inferred, spec.degree)?
                    }
                };
                let (basis, knots) = create_basis::<Sparse>(
                    data,
                    KnotSource::Provided(knots.view()),
                    spec.degree,
                    BasisOptions::value(),
                )?;
                (Some(basis), None, knots)
            }
        }
    } else {
        match &spec.knotspec {
            BSplineKnotSpec::Generate {
                data_range,
                num_internal_knots,
            } => {
                let (basis, knots) = create_basis::<Dense>(
                    data,
                    KnotSource::Generate {
                        data_range: *data_range,
                        num_internal_knots: *num_internal_knots,
                    },
                    spec.degree,
                    BasisOptions::value(),
                )?;
                (None, Some((*basis).clone()), knots)
            }
            BSplineKnotSpec::NaturalCubicRegression { knots } => {
                // Unreachable in practice (the early dispatch returns the cr
                // basis), but keeps this match exhaustive and self-consistent.
                return build_cubic_regression_basis_1d(data, spec, knots);
            }
            BSplineKnotSpec::Provided(knots) => {
                let (basis, knots) = create_basis::<Dense>(
                    data,
                    KnotSource::Provided(knots.view()),
                    spec.degree,
                    BasisOptions::value(),
                )?;
                (None, Some((*basis).clone()), knots)
            }
            BSplineKnotSpec::PeriodicUniform { .. } => {
                crate::bail_invalid_basis!(
                    "periodic B-spline must be handled before storage selection; \
                     this branch is reserved for non-periodic knot specs"
                        .to_string(),
                );
            }
            BSplineKnotSpec::Automatic {
                num_internal_knots,
                placement,
            } => {
                let inferred = num_internal_knots.unwrap_or_else(|| {
                    default_internal_knot_count_for_data(data.len(), spec.degree)
                });
                let knots = match placement {
                    BSplineKnotPlacement::Uniform => {
                        let range = finite_data_range(data)?;
                        internal::generate_full_knot_vector(range, inferred, spec.degree)?
                    }
                    BSplineKnotPlacement::Quantile => {
                        internal::generate_full_knot_vector_quantile(data, inferred, spec.degree)?
                    }
                };
                let (basis, knots) = create_basis::<Dense>(
                    data,
                    KnotSource::Provided(knots.view()),
                    spec.degree,
                    BasisOptions::value(),
                )?;
                (None, Some((*basis).clone()), knots)
            }
        }
    };
    let p_raw = design_sparse_opt
        .as_ref()
        .map(|basis| basis.ncols())
        .or_else(|| design_dense_opt.as_ref().map(Array2::ncols))
        .expect("B-spline basis should be present");
    let greville_for_penalty = penalty_greville_abscissae_for_knots(&knots, spec.degree)?;
    let s_bend_raw = create_difference_penalty_matrix(
        p_raw,
        spec.penalty_order,
        greville_for_penalty.as_ref().map(|g| g.view()),
    )?;
    let penalties_raw = bspline_penalty_candidates(&s_bend_raw, spec, &knots)?;
    let penalties_raw_mats: Vec<Array2<f64>> = penalties_raw
        .iter()
        .map(|candidate| candidate.matrix.clone())
        .collect();
    let (design, transformed_candidates, identifiability_transform) = if let Some(sparse_basis) =
        design_sparse_opt
    {
        match &spec.identifiability {
            BSplineIdentifiability::None => {
                let transformed_candidates = penalties_raw
                    .into_iter()
                    .map(|candidate| -> Result<PenaltyCandidate, BasisError> {
                        Ok(PenaltyCandidate {
                            nullspace_dim_hint: candidate.nullspace_dim_hint,
                            matrix: candidate.matrix,
                            source: candidate.source,
                            normalization_scale: candidate.normalization_scale,
                            kronecker_factors: None,
                            op: None,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (
                    DesignMatrix::Sparse(gam_linalg::matrix::SparseDesignMatrix::new(sparse_basis)),
                    transformed_candidates,
                    None,
                )
            }
            BSplineIdentifiability::WeightedSumToZero { weights } => {
                let (constrained_basis, z) = apply_sum_to_zero_constraint_sparse(
                    &sparse_basis,
                    weights.as_ref().map(|w| w.view()),
                )?;
                let gauge = gam_problem::Gauge::sum_to_zero(z);
                let z = gauge.block_transform(0);
                let transformed_candidates = penalties_raw
                    .into_iter()
                    .map(|candidate| -> Result<PenaltyCandidate, BasisError> {
                        let matrix = gauge.restrict_penalty(&candidate.matrix);
                        Ok(PenaltyCandidate {
                            nullspace_dim_hint: candidate.nullspace_dim_hint,
                            matrix,
                            source: candidate.source,
                            normalization_scale: candidate.normalization_scale,
                            kronecker_factors: None,
                            op: None,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                // `apply_sum_to_zero_constraint_sparse` now returns a dense
                // constrained basis `B_c = B Z` with orthonormal `Z`. The
                // densification is the honest cost of using an orthonormal
                // null-space basis (so that `ZZᵀ` is a true projector); the
                // post-constraint matrix has `k-1` columns, which is the
                // smooth's typical working dimension, so this stays small.
                (
                    DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Arc::new(
                        constrained_basis,
                    ))),
                    transformed_candidates,
                    Some(z),
                )
            }
            BSplineIdentifiability::RemoveLinearTrend
            | BSplineIdentifiability::OrthogonalToDesignColumns { .. }
            | BSplineIdentifiability::FrozenTransform { .. } => {
                crate::bail_invalid_basis!(
                    "sparse B-spline identifiability only supports None or \
                     WeightedSumToZero; RemoveLinearTrend, \
                     OrthogonalToDesignColumns, and FrozenTransform require \
                     the dense path"
                        .to_string(),
                );
            }
        }
    } else {
        let raw_design = design_dense_opt.expect("dense B-spline basis should be present");
        // A `FrozenTransform` already maps from the RAW knot basis with the
        // endpoint boundary projection baked in (it was composed as
        // `boundary ∘ identifiability` at fit time). Re-deriving and re-applying
        // the boundary nullspace transform here would project the raw basis a
        // second time and shrink its width before the frozen transform replays,
        // so a frozen anchored/clamped spec must NOT re-run the boundary step.
        // Skipping it lets the frozen spec keep its original
        // `boundary_conditions` (the single source of truth the intercept-
        // suppression decision reads, #1238/#1265) without double-projecting.
        let boundary_transform = if matches!(
            spec.identifiability,
            BSplineIdentifiability::FrozenTransform { .. }
        ) {
            None
        } else {
            bspline_boundary_nullspace_transform(&knots, spec.degree, spec.boundary_conditions)?
        };
        let (boundary_design, boundary_penalties) = if let Some(z_bc) = boundary_transform.as_ref()
        {
            (
                fast_ab(&raw_design, z_bc),
                penalties_raw_mats
                    .into_iter()
                    .map(|s| project_penalty_matrix(&s, Some(z_bc)))
                    .collect(),
            )
        } else {
            (raw_design, penalties_raw_mats)
        };
        let (design, penalties, identifiability_local) =
            apply_bspline_identifiability_policy_in_chart(
                boundary_design,
                boundary_penalties,
                &knots,
                spec.degree,
                &spec.identifiability,
                boundary_transform.as_ref(),
            )?;
        let identifiability_transform =
            compose_optional_bspline_transform(boundary_transform, identifiability_local)?;
        let transformed_candidates = penalties
            .into_iter()
            .zip(penalties_raw.into_iter())
            .map(
                |(matrix, candidate)| -> Result<PenaltyCandidate, BasisError> {
                    Ok(PenaltyCandidate {
                        nullspace_dim_hint: candidate.nullspace_dim_hint,
                        matrix,
                        source: candidate.source,
                        normalization_scale: candidate.normalization_scale,
                        kronecker_factors: None,
                        op: None,
                    })
                },
            )
            .collect::<Result<Vec<_>, _>>()?;
        (
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(design)),
            transformed_candidates,
            identifiability_transform,
        )
    };
    let transformed_candidates =
        rebuild_double_penalty_nullspace_in_constrained_chart(transformed_candidates)?;
    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(renormalize_constrained_penalty_candidates(
            transformed_candidates,
        ))?;
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        metadata: BasisMetadata::BSpline1D {
            knots,
            identifiability_transform,
            periodic: None,
            degree: Some(spec.degree),
            auto_shrink_note,
        },
        kronecker_factored: None,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
    })
}

/// Build a natural cubic regression spline (mgcv `bs="cr"`/`"cs"`, #1074) basis
/// from a fixed Lancaster–Salkauskas knot set.
///
/// Mirrors the dense-penalty tail of the other dense bases (design + penalty
/// formed, then identifiability congruence → double-penalty nullspace rebuild →
/// filter), but with the cr design ([`CubicRegressionBasis::design`]) and cr
/// roughness penalty ([`CubicRegressionBasis::penalty`], null space `{const,
/// linear}`, dim 2) instead of the B-spline design + difference penalty, and
/// emits [`BasisMetadata::CubicRegression1D`]. The stored
/// `identifiability_transform` is the SAME raw→constrained transform a
/// `BSpline1D` stores, so predict-time replay reconstructs the fit-time design.
///
/// `cs` (shrinkage) differs from `cr` only via `spec.double_penalty`: when set,
/// the Marra & Wood (2011) null-space ridge is emitted as a separate REML
/// coordinate, then rebuilt in the constrained chart.
pub fn build_cubic_regression_basis_1d(
    data: ArrayView1<'_, f64>,
    spec: &BSplineBasisSpec,
    knots: &Array1<f64>,
) -> Result<BasisBuildResult, BasisError> {
    // cr has no B-spline knot/degree geometry: a `RemoveLinearTrend`
    // identifiability would mis-apply Greville-based linear removal to the
    // value-at-knot parameterization. Reject it explicitly; every other
    // `BSplineIdentifiability` variant is a pure design+penalty congruence and
    // is delegated to the shared dense policy below.
    if matches!(
        spec.identifiability,
        BSplineIdentifiability::RemoveLinearTrend
    ) {
        crate::bail_invalid_basis!(
            "natural cubic regression spline (bs=\"cr\"/\"cs\") does not support \
             RemoveLinearTrend identifiability; use the default sum-to-zero centering"
        );
    }

    let cr = CubicRegressionBasis::new(knots.clone())?;
    let raw_design = cr.design(data);
    let s_bend_raw = cr.penalty();

    // Raw (pre-identifiability) candidates: Frobenius-normalized bending penalty
    // plus, for `cs`/double-penalty, the null-space shrinkage ridge — exactly as
    // `bspline_penalty_candidates` assembles them.
    let want_nullspace = spec.double_penalty && spec.boundary_conditions.is_free();
    let (bend_norm, bend_scale) = normalize_penalty(&s_bend_raw);
    let mut penalties_raw = vec![PenaltyCandidate {
        matrix: bend_norm,
        nullspace_dim_hint: 2,
        source: PenaltySource::Primary,
        normalization_scale: bend_scale,
        kronecker_factors: None,
        op: None,
    }];
    // The cr basis is piecewise cubic between its knots, so the Gram integrand
    // `b_i b_j` has degree 6 per span and a 4-point per-span Gauss–Legendre
    // rule is exact — same function-space shrinkage construction as the
    // B-spline path (SPEC rule 5).
    let cr_shrinkage = if want_nullspace {
        let gram =
            piecewise_polynomial_function_gram(&knots.to_vec(), 4, &mut |pts| Ok(cr.design(pts)))?;
        function_space_nullspace_shrinkage(&s_bend_raw, &gram)?
    } else {
        None
    };
    if let Some(shrinkage) = cr_shrinkage {
        let (ridge_norm, ridge_scale) = normalize_penalty(&shrinkage);
        penalties_raw.push(PenaltyCandidate {
            matrix: ridge_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::DoublePenaltyNullspace,
            normalization_scale: ridge_scale,
            kronecker_factors: None,
            op: None,
        });
    }

    // Apply the identifiability congruence to the dense (design, penalty) pair.
    // `apply_bspline_identifiability_policy` is design-generic for every variant
    // except RemoveLinearTrend (rejected above); the `knots`/`degree` arguments
    // it takes are only consumed by that rejected branch, so passing the cr
    // knots and `spec.degree` here is inert. The returned transform is the raw→
    // constrained map stored in metadata for predict-time replay.
    let raw_penalty_mats: Vec<Array2<f64>> = penalties_raw
        .iter()
        .map(|candidate| candidate.matrix.clone())
        .collect();
    let (design_c, penalty_mats_c, identifiability_transform) =
        apply_bspline_identifiability_policy(
            raw_design,
            raw_penalty_mats,
            knots,
            spec.degree,
            &spec.identifiability,
        )?;

    let transformed_candidates: Vec<PenaltyCandidate> = penalty_mats_c
        .into_iter()
        .zip(penalties_raw)
        .map(|(matrix, candidate)| PenaltyCandidate {
            nullspace_dim_hint: candidate.nullspace_dim_hint,
            matrix,
            source: candidate.source,
            normalization_scale: candidate.normalization_scale,
            kronecker_factors: None,
            op: None,
        })
        .collect();

    // Rebuild the double-penalty ridge in the constrained chart (no-op when no
    // ridge candidate is present) and renormalize every constrained block to
    // unit Frobenius norm, exactly as the 1-D B-spline path does.
    let transformed_candidates =
        rebuild_double_penalty_nullspace_in_constrained_chart(transformed_candidates)?;
    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(renormalize_constrained_penalty_candidates(
            transformed_candidates,
        ))?;

    Ok(BasisBuildResult {
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(design_c)),
        penalties,
        nullspace_dims,
        penaltyinfo,
        metadata: BasisMetadata::CubicRegression1D {
            knots: knots.clone(),
            identifiability_transform,
        },
        kronecker_factored: None,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
    })
}

pub(crate) fn compose_bspline_transform(
    existing: Option<Array2<f64>>,
    next: Array2<f64>,
) -> Result<Array2<f64>, BasisError> {
    match existing {
        Some(prev) => {
            if prev.ncols() != next.nrows() {
                crate::bail_dim_basis!(
                    "B-spline streaming transform composition mismatch: previous is {}x{}, next is {}x{}",
                    prev.nrows(),
                    prev.ncols(),
                    next.nrows(),
                    next.ncols()
                );
            }
            Ok(fast_ab(&prev, &next))
        }
        None => Ok(next),
    }
}

fn compose_optional_bspline_transform(
    existing: Option<Array2<f64>>,
    next: Option<Array2<f64>>,
) -> Result<Option<Array2<f64>>, BasisError> {
    match (existing, next) {
        (Some(prev), Some(next)) => Ok(Some(compose_bspline_transform(Some(prev), next)?)),
        (Some(prev), None) => Ok(Some(prev)),
        (None, Some(next)) => Ok(Some(next)),
        (None, None) => Ok(None),
    }
}

fn bspline_boundary_endpoint(
    knots: &Array1<f64>,
    degree: usize,
    right: bool,
) -> Result<f64, BasisError> {
    if knots.len() <= degree + 1 {
        crate::bail_invalid_basis!("B-spline boundary condition requires a valid knot vector");
    }
    let n_basis = knots.len() - degree - 1;
    Ok(if right { knots[n_basis] } else { knots[degree] })
}

fn bspline_endpoint_value_row(
    knots: &Array1<f64>,
    degree: usize,
    endpoint: f64,
) -> Result<Array1<f64>, BasisError> {
    let point = Array1::from_vec(vec![endpoint]);
    let (raw, _) = create_basis::<Dense>(
        point.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::value(),
    )?;
    Ok(raw.row(0).to_owned())
}

fn bspline_endpoint_derivative_row(
    knots: &Array1<f64>,
    degree: usize,
    endpoint: f64,
) -> Result<Array1<f64>, BasisError> {
    let n_basis = knots
        .len()
        .checked_sub(degree + 1)
        .ok_or_else(|| BasisError::InvalidInput("invalid B-spline knot vector".to_string()))?;
    let mut row = vec![0.0; n_basis];
    evaluate_bspline_derivative_scalar(endpoint, knots.view(), degree, &mut row)?;
    Ok(Array1::from_vec(row))
}

fn push_bspline_boundary_rows_for_endpoint(
    rows: &mut Vec<Array1<f64>>,
    knots: &Array1<f64>,
    degree: usize,
    condition: BSplineEndpointBoundaryCondition,
    right: bool,
) -> Result<(), BasisError> {
    let endpoint = bspline_boundary_endpoint(knots, degree, right)?;
    match condition {
        BSplineEndpointBoundaryCondition::Free => {}
        BSplineEndpointBoundaryCondition::Clamped => {
            rows.push(bspline_endpoint_derivative_row(knots, degree, endpoint)?);
        }
        BSplineEndpointBoundaryCondition::Anchored { value } => {
            if !value.is_finite() || value.abs() > 1e-12 {
                crate::bail_invalid_basis!(
                    "anchored B-spline boundary value must be zero for the structural Hermite pin; non-zero anchors are not supported"
                );
            }
            rows.push(bspline_endpoint_value_row(knots, degree, endpoint)?);
            rows.push(bspline_endpoint_derivative_row(knots, degree, endpoint)?);
        }
    }
    Ok(())
}

fn bspline_boundary_constraint_rows(
    knots: &Array1<f64>,
    degree: usize,
    boundary_conditions: BSplineBoundaryConditions,
) -> Result<Option<Array2<f64>>, BasisError> {
    if boundary_conditions.is_free() {
        return Ok(None);
    }
    let p_raw = knots
        .len()
        .checked_sub(degree + 1)
        .ok_or_else(|| BasisError::InvalidInput("invalid B-spline knot vector".to_string()))?;
    let mut rows = Vec::<Array1<f64>>::new();
    push_bspline_boundary_rows_for_endpoint(
        &mut rows,
        knots,
        degree,
        boundary_conditions.left,
        false,
    )?;
    push_bspline_boundary_rows_for_endpoint(
        &mut rows,
        knots,
        degree,
        boundary_conditions.right,
        true,
    )?;
    if rows.is_empty() {
        return Ok(None);
    }
    let mut c = Array2::<f64>::zeros((rows.len(), p_raw));
    for (i, row) in rows.into_iter().enumerate() {
        if row.len() != p_raw {
            crate::bail_dim_basis!(
                "B-spline boundary row has {} columns but raw basis has {}",
                row.len(),
                p_raw
            );
        }
        c.row_mut(i).assign(&row);
    }
    Ok(Some(c))
}

fn bspline_boundary_nullspace_transform(
    knots: &Array1<f64>,
    degree: usize,
    boundary_conditions: BSplineBoundaryConditions,
) -> Result<Option<Array2<f64>>, BasisError> {
    let Some(c) = bspline_boundary_constraint_rows(knots, degree, boundary_conditions)? else {
        return Ok(None);
    };
    let p_raw = c.ncols();
    let frob = c.iter().map(|v| v * v).sum::<f64>().sqrt();
    let (z, rank) =
        rrqr_nullspace_basis(&c.t(), default_rrqr_rank_alpha()).map_err(BasisError::LinalgError)?;
    if rank >= p_raw || z.ncols() == 0 {
        return Err(BasisError::ConstraintNullspaceCollapsed {
            site: "bspline_boundary_nullspace_transform",
            cross_rank: rank,
            coeff_dim: p_raw,
            cross_frobenius: frob,
            gram_spectrum: "not computed (structural rank collapse before Gram eigendecomposition)"
                .to_string(),
        });
    }
    if rank == 0 { Ok(None) } else { Ok(Some(z)) }
}

fn bspline_geometric_constraint_rows(
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Array2<f64>, BasisError> {
    let g = compute_greville_abscissae(knots, degree)?;
    let k = g.len();
    if k < 3 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: k });
    }
    let mut c_geom = Array2::<f64>::zeros((2, k));
    for j in 0..k {
        c_geom[[0, j]] = 1.0;
        c_geom[[1, j]] = g[j];
    }
    let g_mean = g.mean().unwrap_or(0.0);
    let gvar = g.iter().map(|&x| (x - g_mean).powi(2)).sum::<f64>() / (k as f64);
    let g_std = gvar.sqrt().max(1e-10);
    for j in 0..k {
        c_geom[[1, j]] = (c_geom[[1, j]] - g_mean) / g_std;
    }
    Ok(c_geom)
}

fn compute_geometric_constraint_transform_in_chart(
    knots: &Array1<f64>,
    degree: usize,
    raw_to_current: Option<&Array2<f64>>,
) -> Result<Array2<f64>, BasisError> {
    if let Some(t) = raw_to_current {
        let c_geom_raw = bspline_geometric_constraint_rows(knots, degree)?;
        if c_geom_raw.ncols() != t.nrows() {
            crate::bail_dim_basis!(
                "B-spline geometric constraint transform mismatch: raw constraint has {} columns but transform has {} rows",
                c_geom_raw.ncols(),
                t.nrows()
            );
        }
        let c_geom = fast_ab(&c_geom_raw, t);
        let k = c_geom.ncols();
        let frob = c_geom.iter().map(|v| v * v).sum::<f64>().sqrt();
        let (z, rank) = rrqr_nullspace_basis(&c_geom.t(), default_rrqr_rank_alpha())
            .map_err(BasisError::LinalgError)?;
        if rank >= k || z.ncols() == 0 {
            return Err(BasisError::ConstraintNullspaceCollapsed {
                site: "compute_geometric_constraint_transform_in_chart",
                cross_rank: rank,
                coeff_dim: k,
                cross_frobenius: frob,
                gram_spectrum: "not computed (structural rank collapse before Gram \
                                eigendecomposition)"
                    .to_string(),
            });
        }
        Ok(z)
    } else {
        let (z, _) = compute_geometric_constraint_transform(knots, degree, 2)?;
        Ok(z)
    }
}

pub(crate) fn bspline_sum_to_zero_transform_from_cross(
    c: &Array1<f64>,
) -> Result<Array2<f64>, BasisError> {
    let k = c.len();
    if k < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: k });
    }
    let pivot_abs = c.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if pivot_abs <= 1e-12 {
        return Ok(Array2::eye(k));
    }
    let mut c_mat = Array2::<f64>::zeros((k, 1));
    c_mat.column_mut(0).assign(c);
    let (z, rank) =
        rrqr_nullspace_basis(&c_mat, default_rrqr_rank_alpha()).map_err(BasisError::LinalgError)?;
    if rank >= k {
        return Err(BasisError::ConstraintNullspaceCollapsed {
            site: "bspline_sum_to_zero_transform_from_cross",
            cross_rank: rank,
            coeff_dim: k,
            cross_frobenius: c.iter().map(|v| v * v).sum::<f64>().sqrt(),
            gram_spectrum: "not computed (structural rank collapse before Gram eigendecomposition)"
                .to_string(),
        });
    }
    Ok(z)
}

pub(crate) fn streaming_bspline_current_chunk(
    data: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    periodic: Option<(f64, f64, usize)>,
    transform: Option<&Array2<f64>>,
    start: usize,
    end: usize,
) -> Result<Array2<f64>, BasisError> {
    let raw = bspline_raw_row_chunk(data, knots.view(), degree, periodic, start, end)?;
    Ok(match transform {
        Some(z) => fast_ab(&raw, z),
        None => raw,
    })
}

pub(crate) fn streaming_bspline_sum_cross(
    data: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    periodic: Option<(f64, f64, usize)>,
    transform: Option<&Array2<f64>>,
    weights: Option<ArrayView1<'_, f64>>,
    chunk_size: usize,
) -> Result<Array1<f64>, BasisError> {
    if let Some(w) = weights.as_ref()
        && w.len() != data.len()
    {
        return Err(BasisError::WeightsDimensionMismatch {
            expected: data.len(),
            found: w.len(),
        });
    }
    let cols = transform.map(Array2::ncols).unwrap_or(
        bspline_raw_column_count(knots, degree, periodic).map_err(BasisError::InvalidInput)?,
    );
    let mut out = Array1::<f64>::zeros(cols);
    for start in (0..data.len()).step_by(chunk_size.max(1)) {
        let end = (start + chunk_size.max(1)).min(data.len());
        let current =
            streaming_bspline_current_chunk(data, knots, degree, periodic, transform, start, end)?;
        let w_chunk = match weights.as_ref() {
            Some(w) => w.slice(s![start..end]).to_owned(),
            None => Array1::<f64>::ones(end - start),
        };
        out += &current.t().dot(&w_chunk);
    }
    Ok(out)
}

pub(crate) fn streaming_bspline_orthogonality_transform(
    data: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    periodic: Option<(f64, f64, usize)>,
    transform: Option<&Array2<f64>>,
    columns: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    chunk_size: usize,
) -> Result<Array2<f64>, BasisError> {
    if columns.nrows() != data.len() {
        return Err(BasisError::ConstraintMatrixRowMismatch {
            basisrows: data.len(),
            constraintrows: columns.nrows(),
        });
    }
    if let Some(w) = weights.as_ref()
        && w.len() != data.len()
    {
        return Err(BasisError::WeightsDimensionMismatch {
            expected: data.len(),
            found: w.len(),
        });
    }
    let cols = transform.map(Array2::ncols).unwrap_or(
        bspline_raw_column_count(knots, degree, periodic).map_err(BasisError::InvalidInput)?,
    );
    if columns.ncols() == 0 {
        return Ok(Array2::eye(cols));
    }
    let mut cross = Array2::<f64>::zeros((cols, columns.ncols()));
    let mut gram = Array2::<f64>::zeros((cols, cols));
    for start in (0..data.len()).step_by(chunk_size.max(1)) {
        let end = (start + chunk_size.max(1)).min(data.len());
        let current =
            streaming_bspline_current_chunk(data, knots, degree, periodic, transform, start, end)?;
        let mut weighted_constraints = columns.slice(s![start..end, ..]).to_owned();
        if let Some(w) = weights.as_ref() {
            for (mut row, &weight) in weighted_constraints
                .axis_iter_mut(Axis(0))
                .zip(w.slice(s![start..end]).iter())
            {
                row *= weight;
            }
        }
        cross += &current.t().dot(&weighted_constraints);
        gram += &fast_ata(&current);
    }
    orthogonality_transform_from_cross_and_gram(&cross, &gram)
}

pub(crate) fn build_streaming_bspline_design_and_candidates(
    data: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    periodic: Option<(f64, f64, usize)>,
    identifiability: &BSplineIdentifiability,
    penalties_raw: Vec<PenaltyCandidate>,
    mut penalty_mats: Vec<Array2<f64>>,
    chunk_size: Option<usize>,
) -> Result<(DesignMatrix, Vec<PenaltyCandidate>, Option<Array2<f64>>), BasisError> {
    let chunk = chunk_size.unwrap_or(DEFAULT_STREAMING_CHUNK_ROWS).max(1);
    // Streaming is selected only for free endpoint boundary conditions. Non-free
    // endpoints route through the dense structural boundary transform first.
    let mut transform_opt: Option<Array2<f64>> = None;

    match identifiability {
        BSplineIdentifiability::None => {}
        BSplineIdentifiability::WeightedSumToZero { weights } => {
            let cross = streaming_bspline_sum_cross(
                data,
                knots,
                degree,
                periodic,
                transform_opt.as_ref(),
                weights.as_ref().map(|w| w.view()),
                chunk,
            )?;
            let z = bspline_sum_to_zero_transform_from_cross(&cross)?;
            let gauge = gam_problem::Gauge::sum_to_zero(z);
            let z = gauge.block_transform(0);
            penalty_mats = penalty_mats
                .into_iter()
                .map(|s| gauge.restrict_penalty(&s))
                .collect();
            transform_opt = Some(compose_bspline_transform(transform_opt, z)?);
        }
        BSplineIdentifiability::RemoveLinearTrend => {
            let (z, _) = compute_geometric_constraint_transform(knots, degree, 2)?;
            penalty_mats = penalty_mats
                .into_iter()
                .map(|s| project_penalty_matrix(&s, Some(&z)))
                .collect();
            transform_opt = Some(compose_bspline_transform(transform_opt, z)?);
        }
        BSplineIdentifiability::OrthogonalToDesignColumns { columns, weights } => {
            let z = streaming_bspline_orthogonality_transform(
                data,
                knots,
                degree,
                periodic,
                transform_opt.as_ref(),
                columns.view(),
                weights.as_ref().map(|w| w.view()),
                chunk,
            )?;
            penalty_mats = penalty_mats
                .into_iter()
                .map(|s| project_penalty_matrix(&s, Some(&z)))
                .collect();
            transform_opt = Some(compose_bspline_transform(transform_opt, z)?);
        }
        BSplineIdentifiability::FrozenTransform { transform } => {
            let raw_cols = transform_opt.as_ref().map(Array2::ncols).unwrap_or(
                bspline_raw_column_count(knots, degree, periodic)
                    .map_err(BasisError::InvalidInput)?,
            );
            if raw_cols != transform.nrows() {
                crate::bail_dim_basis!(
                    "frozen identifiability transform mismatch: design has {} columns but transform has {} rows",
                    raw_cols,
                    transform.nrows()
                );
            }
            let z = transform.clone();
            penalty_mats = penalty_mats
                .into_iter()
                .map(|s| project_penalty_matrix(&s, Some(&z)))
                .collect();
            transform_opt = Some(compose_bspline_transform(transform_opt, z)?);
        }
    }

    let transformed_candidates = penalty_mats
        .into_iter()
        .zip(penalties_raw)
        .map(|(matrix, candidate)| PenaltyCandidate {
            nullspace_dim_hint: candidate.nullspace_dim_hint,
            matrix,
            source: candidate.source,
            normalization_scale: candidate.normalization_scale,
            kronecker_factors: None,
            op: None,
        })
        .collect();
    let op = StreamingBSplineEvaluator::new(
        Arc::new(data.to_owned()),
        Arc::new(knots.clone()),
        degree,
        periodic,
        transform_opt.as_ref().map(|z| Arc::new(z.clone())),
        chunk_size,
    )
    .map_err(BasisError::InvalidInput)?;
    Ok((
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Arc::new(op))),
        transformed_candidates,
        transform_opt,
    ))
}

pub(crate) fn apply_bspline_identifiability_policy(
    design: Array2<f64>,
    penalties: Vec<Array2<f64>>,
    knots: &Array1<f64>,
    degree: usize,
    identifiability: &BSplineIdentifiability,
) -> Result<(Array2<f64>, Vec<Array2<f64>>, Option<Array2<f64>>), BasisError> {
    apply_bspline_identifiability_policy_in_chart(
        design,
        penalties,
        knots,
        degree,
        identifiability,
        None,
    )
}

fn apply_bspline_identifiability_policy_in_chart(
    design: Array2<f64>,
    penalties: Vec<Array2<f64>>,
    knots: &Array1<f64>,
    degree: usize,
    identifiability: &BSplineIdentifiability,
    raw_to_current: Option<&Array2<f64>>,
) -> Result<(Array2<f64>, Vec<Array2<f64>>, Option<Array2<f64>>), BasisError> {
    let (design_c, z_opt): (Array2<f64>, Option<Array2<f64>>) = match identifiability {
        BSplineIdentifiability::None => (design, None),
        BSplineIdentifiability::WeightedSumToZero { weights } => {
            let (_, z) =
                apply_sum_to_zero_constraint(design.view(), weights.as_ref().map(|w| w.view()))?;
            let gauge = gam_problem::Gauge::sum_to_zero(z);
            let b_c = gauge.restrict_design(&design);
            let z = gauge.block_transform(0);
            (b_c, Some(z))
        }
        BSplineIdentifiability::RemoveLinearTrend => {
            let z = compute_geometric_constraint_transform_in_chart(knots, degree, raw_to_current)?;
            (fast_ab(&design, &z), Some(z))
        }
        BSplineIdentifiability::OrthogonalToDesignColumns { columns, weights } => {
            let (b_c, z) = applyweighted_orthogonality_constraint(
                design.view(),
                columns.view(),
                weights.as_ref().map(|w| w.view()),
            )?;
            (b_c, Some(z))
        }
        BSplineIdentifiability::FrozenTransform { transform } => {
            let z = transform.clone();
            if design.ncols() != z.nrows() {
                if let Some(t) = raw_to_current {
                    if t.nrows() == z.nrows() {
                        crate::bail_dim_basis!(
                            "frozen B-spline transform already maps from the raw basis; clear boundary_conditions before replaying FrozenTransform"
                        );
                    }
                }
                crate::bail_dim_basis!(
                    "frozen identifiability transform mismatch: design has {} columns but transform has {} rows",
                    design.ncols(),
                    z.nrows()
                );
            }
            (fast_ab(&design, &z), Some(z))
        }
    };

    let penalties_c = if let Some(ref z) = z_opt {
        let gauge = gam_problem::Gauge::from_block_transforms(&[z.clone()]);
        penalties
            .into_iter()
            .map(|s| gauge.restrict_penalty(&s))
            .collect()
    } else {
        penalties
    };

    Ok((design_c, penalties_c, z_opt))
}

pub fn estimate_penalty_nullity(penalty: &Array2<f64>) -> Result<usize, BasisError> {
    if penalty.nrows() != penalty.ncols() {
        crate::bail_dim_basis!("penalty matrix must be square when estimating nullspace");
    }
    if penalty.nrows() == 0 {
        return Ok(0);
    }

    let (sym, evals, _) = spectral_summary(penalty)?;
    let tol = spectral_tolerance(&sym, &evals);
    Ok(SpectralClassification::new(&evals, tol).nullity())
}

#[derive(Debug, Clone)]
pub(crate) struct PsdSpectralSummary {
    pub(crate) min_eigenvalue: f64,
    pub(crate) max_abs_eigenvalue: f64,
    pub(crate) tolerance: f64,
    pub(crate) effective_rank: usize,
}

pub(crate) fn symmetrize_penalty(penalty: &Array2<f64>) -> Array2<f64> {
    gam_linalg::matrix::symmetrize(penalty)
}

/// Project a (nearly-)symmetric matrix to the PSD cone by clamping
/// negative eigenvalues to zero. A PenaltyMatrix is by definition PSD;
/// this enforces that contract against the f64 noise floor so callers
/// downstream (PIRLS, REML/LAML, outer-Hessian assembly) never see a
/// quadratic form that goes negative on legitimate β.
pub(crate) fn project_penalty_to_psd_cone(matrix: &Array2<f64>) -> Array2<f64> {
    let sym = symmetrize_penalty(matrix);
    let n = sym.nrows();
    if n == 0 || n != sym.ncols() {
        return sym;
    }
    let (evals, evecs) = match FaerEigh::eigh(&sym, Side::Lower) {
        Ok(pair) => pair,
        Err(_) => return sym,
    };
    if evals.is_empty() {
        return sym;
    }
    let min_ev = evals.iter().copied().fold(f64::INFINITY, f64::min);
    if min_ev >= 0.0 {
        return sym;
    }
    let mut clamped = sym.clone();
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for k in 0..evals.len() {
                let lam = evals[k];
                if lam > 0.0 {
                    acc += lam * evecs[[i, k]] * evecs[[j, k]];
                }
            }
            clamped[[i, j]] = acc;
        }
    }
    // Final symmetrize to wipe any reconstruction asymmetry at the noise floor.
    for i in 0..n {
        for j in 0..i {
            let v = 0.5 * (clamped[[i, j]] + clamped[[j, i]]);
            clamped[[i, j]] = v;
            clamped[[j, i]] = v;
        }
    }
    clamped
}

pub(crate) fn spectral_tolerance(sym: &Array2<f64>, evals: &Array1<f64>) -> f64 {
    let max_abs_ev = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    // Keep the cutoff in eigenvalue units so uniform penalty scaling does not
    // change PSD/rank decisions for the same spectrum shape.
    (sym.nrows().max(1) as f64) * 1e-10 * max_abs_ev
}

/// Where a single eigenvalue of a symmetric penalty sits relative to the
/// spectral tolerance — the *only* place the three-way convention is defined.
///
/// A symmetric penalty's spectrum has exactly three structural classes, and
/// conflating any two of them is the #1425 defect class:
///
/// * [`Range`](EigenClass::Range) — `ev > tol`: a direction the penalty
///   genuinely penalizes (positive curvature). Spans `range(S)`; its square
///   root enters the penalty root `R` (so `RᵀR = S` on this subspace).
/// * [`Null`](EigenClass::Null) — `|ev| <= tol`: an *unpenalized* direction
///   (`Sβ = 0`). Spans `null(S)`; this is what gets absorbed into the
///   parametric block so the inner solve sees no flat penalty direction.
/// * [`Negative`](EigenClass::Negative) — `ev < -tol`: a direction of genuine
///   *negative* curvature — the penalty is non-PSD there. This is NEITHER
///   range nor null: it must never be square-rooted into `R` (its sqrt is
///   imaginary) and must never be counted toward `nullity` (it is not
///   unpenalized). The old binary `ev <= tol` split had no name for this
///   class, so it silently mislabeled negative curvature as null space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EigenClass {
    Range,
    Null,
    Negative,
}

impl EigenClass {
    /// Classify a single eigenvalue against a (non-negative) spectral
    /// tolerance. This is the canonical, single-definition partition every
    /// penalty-spectrum consumer routes through.
    #[inline]
    pub(crate) fn of(eigenvalue: f64, tol: f64) -> EigenClass {
        if eigenvalue > tol {
            EigenClass::Range
        } else if eigenvalue < -tol {
            EigenClass::Negative
        } else {
            EigenClass::Null
        }
    }
}

/// The complete three-way structural partition of a symmetric penalty's
/// spectrum: which eigen-indices are range, null, and negative-curvature.
///
/// This is the single source of truth for `rank` / `nullity` /
/// `negative_dim` and for every subspace basis (`null_basis`, the joint
/// absorption ordering). Building it once from `(evals, tol)` and deriving
/// all answers from it is what makes the range/null/negative convention
/// impossible to apply inconsistently across call sites (#1425).
#[derive(Debug, Clone)]
pub(crate) struct SpectralClassification {
    /// Indices (into the eigenvalue array) with `ev > tol`.
    pub(crate) range_idx: Vec<usize>,
    /// Indices with `|ev| <= tol` — the genuine null space.
    pub(crate) null_idx: Vec<usize>,
    /// Indices with `ev < -tol` — genuine negative curvature (non-PSD).
    pub(crate) negative_idx: Vec<usize>,
    /// The tolerance the partition was computed against.
    pub(crate) tol: f64,
}

impl SpectralClassification {
    /// Partition `evals` against `tol` into range / null / negative classes.
    pub(crate) fn new(evals: &Array1<f64>, tol: f64) -> SpectralClassification {
        let mut range_idx = Vec::new();
        let mut null_idx = Vec::new();
        let mut negative_idx = Vec::new();
        for (i, &ev) in evals.iter().enumerate() {
            match EigenClass::of(ev, tol) {
                EigenClass::Range => range_idx.push(i),
                EigenClass::Null => null_idx.push(i),
                EigenClass::Negative => negative_idx.push(i),
            }
        }
        SpectralClassification {
            range_idx,
            null_idx,
            negative_idx,
            tol,
        }
    }

    /// Number of positively-penalized directions, `dim(range(S))`.
    #[inline]
    pub(crate) fn rank(&self) -> usize {
        self.range_idx.len()
    }

    /// Number of genuinely unpenalized directions, `dim(null(S))`. Excludes
    /// negative-curvature directions, which are not unpenalized.
    #[inline]
    pub(crate) fn nullity(&self) -> usize {
        self.null_idx.len()
    }

    /// Number of negative-curvature directions. `> 0` iff the penalty is
    /// non-PSD beyond the noise floor.
    #[inline]
    pub(crate) fn negative_dim(&self) -> usize {
        self.negative_idx.len()
    }

    /// True iff the penalty carries genuine negative curvature (non-PSD).
    #[inline]
    pub(crate) fn is_indefinite(&self) -> bool {
        !self.negative_idx.is_empty()
    }

    /// True iff every eigenvalue is within tolerance of zero (the whole
    /// block is numerically the zero matrix).
    #[inline]
    pub(crate) fn iszero(&self) -> bool {
        self.range_idx.is_empty() && self.negative_idx.is_empty()
    }

    /// Eigenvector columns spanning `null(S)`, or `None` when full-rank.
    pub(crate) fn null_basis(&self, evecs: &Array2<f64>) -> Option<Array2<f64>> {
        if self.null_idx.is_empty() {
            return None;
        }
        Some(evecs.select(Axis(1), &self.null_idx))
    }

    /// Column order for the absorption rotation `Q = [U_range | U_neg | U_null]`:
    /// genuinely-penalized directions first (range, then any non-PSD negative
    /// directions — penalized in the sense of *not unpenalized*), with the
    /// `nullity()` genuine null columns LAST so the absorption stage can take
    /// the trailing block. Within range, descending by eigenvalue for a stable,
    /// well-conditioned leading block.
    pub(crate) fn absorption_order(&self, evals: &Array1<f64>) -> Vec<usize> {
        let mut order: Vec<usize> = (0..evals.len()).collect();
        order.sort_by(|&a, &b| {
            let null_a = EigenClass::of(evals[a], self.tol) == EigenClass::Null;
            let null_b = EigenClass::of(evals[b], self.tol) == EigenClass::Null;
            match (null_a, null_b) {
                (false, true) => std::cmp::Ordering::Less,
                (true, false) => std::cmp::Ordering::Greater,
                // Both genuinely null (or both non-null): descending by
                // eigenvalue. NaN/sign ties are unlikely on a symmetric
                // penalty but handled safely.
                _ => evals[b]
                    .partial_cmp(&evals[a])
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        });
        order
    }
}

pub(crate) fn spectral_summary(
    penalty: &Array2<f64>,
) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), BasisError> {
    let sym = symmetrize_penalty(penalty);
    let (evals, evecs) = FaerEigh::eigh(&sym, Side::Lower).map_err(BasisError::LinalgError)?;
    Ok((sym, evals, evecs))
}

pub(crate) fn validate_psd_penalty(
    penalty: &Array2<f64>,
    context: &str,
    guidance: &str,
) -> Result<PsdSpectralSummary, BasisError> {
    if penalty.nrows() != penalty.ncols() {
        crate::bail_dim_basis!("{context}: penalty matrix must be square for PSD validation");
    }
    if penalty.nrows() == 0 {
        return Ok(PsdSpectralSummary {
            min_eigenvalue: 0.0,
            max_abs_eigenvalue: 0.0,
            tolerance: 1e-10,
            effective_rank: 0,
        });
    }

    let (sym, evals, _) = spectral_summary(penalty)?;
    let tolerance = spectral_tolerance(&sym, &evals);
    let classes = SpectralClassification::new(&evals, tolerance);
    let min_eigenvalue = evals.iter().copied().fold(f64::INFINITY, f64::min);
    let max_abs_eigenvalue = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let effective_rank = classes.rank();

    // A PSD penalty has no negative-curvature class. Surface any genuine
    // negative eigendirection as an indefinite-penalty error rather than
    // letting it masquerade as range or null downstream (#1425).
    if classes.is_indefinite() {
        return Err(BasisError::IndefinitePenalty {
            context: context.to_string(),
            min_eigenvalue,
            tolerance,
            guidance: guidance.to_string(),
        });
    }

    Ok(PsdSpectralSummary {
        min_eigenvalue,
        max_abs_eigenvalue,
        tolerance,
        effective_rank,
    })
}

pub fn analyze_penalty_block(penalty: &Array2<f64>) -> Result<CanonicalPenaltyBlock, BasisError> {
    analyze_penalty_block_with_op(penalty, None)
}

pub fn analyze_penalty_block_with_op(
    penalty: &Array2<f64>,
    op: Option<std::sync::Arc<dyn crate::analytic_penalties::PenaltyOp>>,
) -> Result<CanonicalPenaltyBlock, BasisError> {
    if penalty.nrows() != penalty.ncols() {
        crate::bail_dim_basis!("penalty matrix must be square when analyzing penalty");
    }
    if penalty.nrows() == 0 {
        return Ok(CanonicalPenaltyBlock {
            sym_penalty: Array2::<f64>::zeros((0, 0)),
            eigenvalues: Array1::<f64>::zeros(0),
            eigenvectors: Array2::<f64>::zeros((0, 0)),
            rank: 0,
            nullity: 0,
            negative_dim: 0,
            tol: 1e-10,
            iszero: true,
            op,
        });
    }

    let (sym, evals, evecs) = spectral_summary(penalty)?;
    let tol = spectral_tolerance(&sym, &evals);
    // Route the entire range / null / negative-curvature partition through the
    // single canonical classifier so this block can never disagree with any
    // other penalty-spectrum consumer about which directions are unpenalized
    // (the #1425 defect class). `nullity` is the genuine null space
    // (`|ev| <= tol`); negative-curvature directions are tracked separately in
    // `negative_dim` and are neither range nor null.
    let classes = SpectralClassification::new(&evals, tol);
    Ok(CanonicalPenaltyBlock {
        sym_penalty: sym,
        eigenvalues: evals,
        eigenvectors: evecs,
        rank: classes.rank(),
        nullity: classes.nullity(),
        negative_dim: classes.negative_dim(),
        tol,
        iszero: classes.iszero(),
        op,
    })
}

pub fn filter_active_penalty_candidates(
    candidates: Vec<PenaltyCandidate>,
) -> Result<(Vec<Array2<f64>>, Vec<usize>, Vec<PenaltyInfo>), BasisError> {
    let (penalties, nullspace_dims, penaltyinfo, _null_eigenvectors, _ops) =
        filter_active_penalty_candidates_with_ops(candidates)?;
    Ok((penalties, nullspace_dims, penaltyinfo))
}

/// Extract the orthonormal basis of `null(S)` from a `CanonicalPenaltyBlock`.
///
/// Returns `Some(U_null)` with `U_null.ncols() == block.nullity` when the
/// block has a non-trivial null space; `None` when the block is full-rank
/// (`block.nullity == 0`). The columns of `U_null` are the eigenvectors of
/// `block.sym_penalty` at eigenvalues `|ev| ≤ block.tol` (genuine null
/// directions only — never the negative-curvature class) — exactly the
/// directions along which `Sβ = 0` and on which `H_pen = H_loglik + S` carries no
/// curvature from the penalty. These are the directions that must be
/// absorbed into the parametric block at construction time so that the
/// smooth's design is orthogonal to its own null space and the inner Newton
/// solve does not get stuck refusing a non-existent stationary point.
pub(crate) fn nullspace_basis_from_block(block: &CanonicalPenaltyBlock) -> Option<Array2<f64>> {
    if block.nullity == 0 {
        return None;
    }
    // Derive the null basis from the canonical classifier so the columns it
    // selects are exactly the directions counted in `block.nullity`
    // (`|ev| <= tol`), never the negative-curvature directions (#1425).
    SpectralClassification::new(&block.eigenvalues, block.tol).null_basis(&block.eigenvectors)
}

/// Compute the joint-null absorption rotation for a smooth with one or more
/// active penalty blocks.
///
/// Given the smooth's active penalties `S_1, ..., S_K` (all `p × p`,
/// symmetric PSD, in the same local coordinate system), return `Q` such
/// that `Q = [U_range | U_null]` with `U_null` spanning the *joint* null
/// space `null(Σ_k S_k)` — i.e., the directions annihilated by every
/// active penalty.
///
/// Why the joint and not the per-block null: a per-block null
/// `null(S_k)` may be penalized by some other `S_j` (`j ≠ k`), so
/// absorbing it would lose smoothing the data is paying for. Only
/// directions in the joint null are *genuinely* unpenalized and qualify
/// for absorption into the parametric block.
///
/// Returns `None` when there is no penalty (`penalties.is_empty()`),
/// when the local dimension is zero, or when the joint penalty is
/// full-rank (joint nullity = 0). A non-trivial `joint_nullity` is the
/// only state encoded as `Some`.
/// Recompute per-block null-eigenvector matrices from a sequence of penalty
/// matrices. Each output entry `null_eigenvectors[k]` is `Some(U_null)`
/// (eigenvectors of `penalties[k]` at eigenvalues `|ev| ≤ spectral_tolerance`,
/// the genuine null directions) when the block has a non-trivial null space,
/// and `None` otherwise.
///
/// This is the inverse of "consumers update `penalties[k]` without
/// refreshing `null_eigenvectors[k]`": whenever a code path rebuilds a
/// penalty matrix in-place (e.g., the factor-sum-to-zero handler's
/// Kronecker-style `S_big` reconstruction), the parallel `null_eigenvectors`
/// vector becomes stale unless this helper is called. The invariant the
/// pipeline relies on is `null_eigenvectors[k]` always mirrors
/// `penalties[k]`'s spectral null space.
pub fn recompute_null_eigenvectors(
    penalties: &[Array2<f64>],
) -> Result<Vec<Option<Array2<f64>>>, BasisError> {
    penalties
        .iter()
        .map(|s| {
            let block = analyze_penalty_block_with_op(s, None)?;
            Ok(nullspace_basis_from_block(&block))
        })
        .collect()
}

pub fn compute_joint_null_rotation(
    penalties: &[Array2<f64>],
) -> Result<Option<JointNullRotation>, BasisError> {
    if penalties.is_empty() {
        return Ok(None);
    }
    let p = penalties[0].nrows();
    if p == 0 {
        return Ok(None);
    }
    for (k, s) in penalties.iter().enumerate() {
        if s.nrows() != p || s.ncols() != p {
            crate::bail_dim_basis!(
                "compute_joint_null_rotation: penalty[{}] is {}×{}, expected {}×{}",
                k,
                s.nrows(),
                s.ncols(),
                p,
                p
            );
        }
    }
    let mut s_sum = Array2::<f64>::zeros((p, p));
    for s in penalties {
        s_sum += s;
    }
    let (sym, evals, evecs) = spectral_summary(&s_sum)?;
    let tol = spectral_tolerance(&sym, &evals);
    // Classify the joint penalty `Σ_k S_k` through the single canonical
    // partition. Only the genuine joint null (`|ev| <= tol`) is absorbed; a
    // negative joint eigenvalue (`ev < -tol`) is negative curvature, NOT an
    // unpenalized direction, and stays in the leading (non-absorbed) block
    // (#1425).
    let classes = SpectralClassification::new(&evals, tol);
    let joint_nullity = classes.nullity();
    if joint_nullity == 0 {
        return Ok(None);
    }
    // Order columns of Q as [U_range | U_neg | U_null]: the absorption stage
    // takes the trailing `joint_nullity` columns as the absorbed parametric
    // block, so the genuine null columns must come last and negative-curvature
    // columns must NOT be among them.
    let order = classes.absorption_order(&evals);
    let rotation = evecs.select(Axis(1), &order);
    Ok(Some(JointNullRotation {
        rotation,
        joint_nullity,
    }))
}

/// Same filtering pass as [`filter_active_penalty_candidates`] but also
/// returns the per-active-penalty operator handles and null-space bases.
///
/// All three "side-channel" vectors (`nullspace_dims`, `null_eigenvectors`,
/// `ops`) are parallel to `penalties` — same length, same order. `null_eigenvectors[k]`
/// is `Some(U_null)` iff `nullspace_dims[k] > 0`; `ops[k]` is `Some(op)` iff
/// the candidate carried an operator-form handle bit-equivalent to the dense
/// matrix. Construction-side consumers use `null_eigenvectors` to absorb the
/// smooth's penalty null space into the parametric block; PIRLS/REML
/// consumers route through the `ops` `Some` entries for exact operator
/// matvec without materializing the dense `p x p` Gram.
pub fn filter_active_penalty_candidates_with_ops(
    candidates: Vec<PenaltyCandidate>,
) -> Result<
    (
        Vec<Array2<f64>>,
        Vec<usize>,
        Vec<PenaltyInfo>,
        Vec<Option<Array2<f64>>>,
        Vec<Option<std::sync::Arc<dyn crate::analytic_penalties::PenaltyOp>>>,
    ),
    BasisError,
> {
    let mut penalties = Vec::with_capacity(candidates.len());
    let mut nullspace_dims = Vec::with_capacity(candidates.len());
    let mut penaltyinfo = Vec::with_capacity(candidates.len());
    let mut active_null_eigenvectors: Vec<Option<Array2<f64>>> =
        Vec::with_capacity(candidates.len());
    let mut active_ops: Vec<Option<std::sync::Arc<dyn crate::analytic_penalties::PenaltyOp>>> =
        Vec::with_capacity(candidates.len());

    for (original_index, candidate) in candidates.into_iter().enumerate() {
        let analysis = analyze_penalty_block_with_op(&candidate.matrix, candidate.op.clone())?;
        let dropped_reason = if analysis.rank == 0 {
            Some(if analysis.iszero {
                PenaltyDropReason::ZeroMatrix
            } else {
                PenaltyDropReason::NumericalRankZero
            })
        } else {
            None
        };
        let active = dropped_reason.is_none();
        let kronecker_factors =
            validated_kronecker_factors(candidate.kronecker_factors, &analysis.sym_penalty);
        if active {
            let null_basis = nullspace_basis_from_block(&analysis);
            log::debug!(
                "Retained penalty block source={:?} original_index={} rank={} nullspace_dim_hint={} has_op={} has_null_basis={}",
                candidate.source,
                original_index,
                analysis.rank,
                analysis.nullity,
                analysis.op.is_some(),
                null_basis.is_some(),
            );
            penalties.push(analysis.sym_penalty);
            nullspace_dims.push(analysis.nullity);
            active_null_eigenvectors.push(null_basis);
            active_ops.push(analysis.op);
        } else {
            log::debug!(
                "Dropped inactive penalty block source={:?} original_index={} reason={:?}",
                candidate.source,
                original_index,
                dropped_reason
            );
        }
        penaltyinfo.push(PenaltyInfo {
            source: candidate.source,
            original_index,
            active,
            effective_rank: analysis.rank,
            dropped_reason,
            nullspace_dim_hint: analysis.nullity,
            normalization_scale: candidate.normalization_scale,
            kronecker_factors,
        });
    }

    Ok((
        penalties,
        nullspace_dims,
        penaltyinfo,
        active_null_eigenvectors,
        active_ops,
    ))
}

/// Re-normalize already-constrained 1-D B-spline penalty candidates to unit
/// Frobenius norm *in the constrained coordinate frame*.
///
/// The raw (pre-identifiability) wiggliness/ridge penalty is Frobenius-normalized
/// at construction, but the sum-to-zero identifiability transform `Zᵀ S Z`
/// perturbs `‖S‖_F` away from 1 (open `bs="ps"` order-2 drifts to ≈0.99967). The
/// block the REML smoothing parameter `λ` actually multiplies is the *shipped*,
/// constrained penalty, and the REML objective is evaluated entirely in
/// constrained coordinates — so the shipped penalty must carry unit Frobenius
/// norm *there*, matching `normalize_penalty_in_constrained_space` used by
/// cr / duchon / tensor (the #1364/#1365/#1366/#1401 normalization class).
/// Normalizing only the raw penalty (before the constraint) leaves `λ` on a
/// slightly basis-dependent scale; this folds the residual constraint-transform
/// factor out so `‖S‖_F = 1` in the frame that REML scores.
///
/// Fit-invariant at the REML optimum: rescaling `S → S/c` only rescales the
/// recorded `λ̂` by `c`. Scaling a block never changes its rank, so this cannot
/// alter which penalties `filter_active_penalty_candidates_with_ops` keeps
/// active; the `> 1e-12` guard only avoids dividing a numerically-zero block.
fn renormalize_constrained_penalty_candidates(
    mut candidates: Vec<PenaltyCandidate>,
) -> Vec<PenaltyCandidate> {
    for candidate in &mut candidates {
        let frob = candidate.matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
        if frob.is_finite() && frob > 1e-12 {
            candidate.matrix.mapv_inplace(|v| v / frob);
            candidate.normalization_scale *= frob;
        }
    }
    candidates
}

/// Rebuild the double-penalty null-space shrinkage ridge in the FINAL
/// (post-identifiability) coefficient chart.
///
/// The Marra & Wood (2011) null-space shrinkage block is the orthogonal
/// projector `U Uᵀ` onto `null(S_wiggle)`. `bspline_penalty_candidates` builds
/// it in the RAW B-spline coefficient chart, but the identifiability transform
/// `Z` (sum-to-zero centering, boundary projection, ...) is applied to every
/// penalty *afterwards* as the congruence `S → ZᵀSZ`. A congruence does NOT
/// commute with the projector construction: `Zᵀ(UUᵀ)Z = (ZᵀU)(ZᵀU)ᵀ` is no
/// longer a projector onto `null(ZᵀSZ)`. Its nonzero eigenvalues are those of
/// `Uᵀ(ZZᵀ)U`; for an open/clamped B-spline the centering vector `c=Bᵀ1`
/// is NOT in `null(S)`, so one of those eigenvalues is `δ=dist²(ĉ,null(S))>0`
/// (≈0.148 for the k=10 order-2 P-spline). That spurious second null direction
/// lies in the RANGE of the bend penalty, so the "shrinkage" ridge penalizes a
/// genuine curvature mode — the source of the concurvity collapse (#1476) and
/// the Tweedie `bs="ps"` boundary bias (#1477).
///
/// The fix mirrors the box-reparametrization path (term_specs.rs): rebuild the
/// ridge from `null(S_c)` of the *transformed* primary wiggliness penalty, so
/// the rebuilt block has `rank = nullity(S_c)` and `S_c`-range directions carry
/// no shrinkage. After centering the constant direction is gone, so
/// `nullity(S_c)=1`; an UNcentered / constraint-free smooth keeps its genuine
/// 2-D null space, because the rebuild adapts to the actual `null(S_c)`.
///
/// The rebuild is METRIC-CONSISTENT (`rebuild_metric_consistent_ridge`): the
/// raw ridge is the function-space block `G Z (ZᵀGZ)⁻¹ ZᵀG` (penalizing
/// `∫(null component of f)²`, SPEC rule 5), and its congruence transform still
/// acts as the constrained-chart Gram on `null(S_c)`, so the rebuilt block is
/// the constrained-chart function-space ridge — not a coefficient-space
/// projector that would change under basis rescaling. The rebuilt ridge's
/// `normalization_scale` is reset to `1.0`; the subsequent
/// `renormalize_constrained_penalty_candidates` pass folds in its unit-Frobenius
/// scale just as for every other constrained block.
fn rebuild_double_penalty_nullspace_in_constrained_chart(
    mut candidates: Vec<PenaltyCandidate>,
) -> Result<Vec<PenaltyCandidate>, BasisError> {
    let has_ridge = candidates
        .iter()
        .any(|c| matches!(c.source, PenaltySource::DoublePenaltyNullspace));
    if !has_ridge {
        return Ok(candidates);
    }
    // Select the wiggliness penalty by `PenaltySource::Primary` EXPLICITLY rather
    // than "the first non-ridge block". `bspline_penalty_candidates` emits exactly
    // one `Primary` (the bending penalty) plus the optional ridge, so the two are
    // equivalent today. The explicit match is robust to a future per-axis
    // boundary / anchor penalty being added to the 1-D candidate set: such a block
    // would be a non-`Primary`, non-ridge candidate, and a `find(!ridge)` lookup
    // could then mis-pick it and rebuild the projector from the wrong null space.
    // Deriving the ridge from `null(S_c)` only makes sense for the genuine
    // wiggliness penalty, so we pin that selection here.
    let primary_constrained = candidates
        .iter()
        .find(|c| matches!(c.source, PenaltySource::Primary))
        .map(|c| c.matrix.clone());
    let Some(s_c) = primary_constrained else {
        crate::bail_invalid_basis!(
            "double-penalty B-spline has a null-space shrinkage ridge but no primary \
             wiggliness penalty to derive its constrained null space from"
        );
    };
    let p = s_c.nrows();
    for candidate in &mut candidates {
        if matches!(candidate.source, PenaltySource::DoublePenaltyNullspace) {
            // `candidate.matrix` is the congruence-transformed raw ridge; its
            // action on `null(S_c)` is exactly the constrained-chart Gram's,
            // so the rebuild stays in the function-space metric the raw ridge
            // was built with (see `rebuild_metric_consistent_ridge`).
            candidate.matrix = rebuild_metric_consistent_ridge(&s_c, &candidate.matrix)?
                .unwrap_or_else(|| Array2::<f64>::zeros((p, p)));
            candidate.normalization_scale = 1.0;
            candidate.op = None;
        }
    }
    Ok(candidates)
}

pub(crate) fn validated_kronecker_factors(
    factors: Option<Vec<Array2<f64>>>,
    matrix: &Array2<f64>,
) -> Option<Vec<Array2<f64>>> {
    let factors = factors?;
    let Some((first, rest)) = factors.split_first() else {
        return None;
    };
    let mut kron = first.clone();
    for factor in rest {
        kron = crate::kronecker::kronecker_product(&kron, factor);
    }
    if kron.dim() != matrix.dim() {
        return None;
    }

    let scale = kron
        .iter()
        .chain(matrix.iter())
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()))
        .max(1.0);
    let max_abs_diff = kron
        .iter()
        .zip(matrix.iter())
        .fold(0.0_f64, |acc, (&lhs, &rhs)| acc.max((lhs - rhs).abs()));
    (max_abs_diff <= scale * 1e-10).then_some(factors)
}

/// Assemble the raw (pre-identifiability) penalty candidates for a 1-D B-spline.
///
/// The wiggliness penalty `S_bend` is always present. When `double_penalty` is
/// enabled on a free (non-boundary-conditioned) basis we additionally emit the
/// null-space shrinkage block (Marra & Wood 2011, in the FUNCTION-SPACE form
/// `G Z (ZᵀGZ)⁻¹ ZᵀG` with `G` the exact basis Gram) as a *separate* REML
/// coordinate, so that REML can drive an unsupported term's constant/linear
/// part to `EDF → 0` independently of its wiggliness (mgcv `select = TRUE`).
///
/// Both candidates are Frobenius-normalized to unit norm exactly the way the
/// Duchon / constant-curvature / tensor-B-spline paths already normalize their
/// own primary + `DoublePenaltyNullspace` blocks. This normalization is what
/// makes the second smoothing parameter `λ_nullspace` *identifiable*: an
/// un-normalized `Z Zᵀ` (largest eigenvalue 1) sits on a wildly different scale
/// from the raw bending penalty, leaving the outer REML objective nearly flat
/// along the `λ_nullspace` coordinate. Under that flat coordinate REML weakened
/// the wiggliness penalty instead of shrinking the term out, which *inflated*
/// the smooth's EDF rather than reducing it (#1266). With both blocks on a
/// common (unit-Frobenius) scale the coordinate is identified and the double
/// penalty shrinks — never inflates — null-space / unsupported terms.
fn bspline_penalty_candidates(
    s_bend_raw: &Array2<f64>,
    spec: &BSplineBasisSpec,
    knots: &Array1<f64>,
) -> Result<Vec<PenaltyCandidate>, BasisError> {
    let want_nullspace = spec.double_penalty && spec.boundary_conditions.is_free();
    let shrinkage = if want_nullspace {
        // Function-space shrinkage (SPEC rule 5): the ridge penalizes
        // `∫(null component of f)²` via the exact basis Gram, so the penalized
        // quantity is a property of the fitted function, invariant to how the
        // B-spline basis happens to be scaled or parameterized.
        let gram = bspline_function_gram(knots, spec.degree)?;
        function_space_nullspace_shrinkage(s_bend_raw, &gram)?
    } else {
        None
    };

    // Without an active null-space block, still Frobenius-normalize the bending
    // penalty (recording the norm in `normalization_scale`) exactly the way the
    // double-penalty branch below and the cr / duchon / constant-curvature /
    // tensor paths already do. The shipped design penalty is `β'(S/c)β`; the
    // REML smoothing parameter `λ` multiplies that *normalized* block, and the
    // outer optimizer's λ-search (log-λ brackets, seed screening, the implicit
    // prior on λ) is calibrated for a unit-Frobenius penalty. Shipping the raw
    // `S` (scale 1.0) put `λ` on a basis-dependent scale, so REML stopped at a
    // smaller effective `λ` and failed to fully shrink the penalty null space:
    // `s(x, bs="ps")` over-fit data whose signal is the null space (a straight
    // line), landing at EDF ~5 with spurious curvature while the normalized
    // `bs="cr"` correctly collapsed to EDF ≈ 2 on the same data (#1365). At the
    // REML optimum the fit is invariant to this normalization (only the recorded
    // `λ̂` rescales by `c`); it just removes the scale miscalibration of the
    // λ-search heuristics.
    let Some(shrinkage) = shrinkage else {
        let (bend_norm, bend_scale) = normalize_penalty(s_bend_raw);
        return Ok(vec![PenaltyCandidate {
            matrix: bend_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::Primary,
            normalization_scale: bend_scale,
            kronecker_factors: None,
            op: None,
        }]);
    };

    let (bend_norm, bend_scale) = normalize_penalty(s_bend_raw);
    let (ridge_norm, ridge_scale) = normalize_penalty(&shrinkage);
    Ok(vec![
        PenaltyCandidate {
            matrix: bend_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::Primary,
            normalization_scale: bend_scale,
            kronecker_factors: None,
            op: None,
        },
        PenaltyCandidate {
            matrix: ridge_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::DoublePenaltyNullspace,
            normalization_scale: ridge_scale,
            kronecker_factors: None,
            op: None,
        },
    ])
}

/// Exact L² Gram `G = ∫ b(x) b(x)ᵀ dx` of a piecewise-polynomial 1-D basis,
/// assembled span-by-span with a `points_per_span`-point Gauss–Legendre rule.
/// An `m`-point rule is exact for polynomial integrands of degree `≤ 2m − 1`
/// on every span, so choosing `points_per_span = degree + 1` makes the Gram of
/// a degree-`degree` spline basis exact (the integrand `b_i b_j` has degree
/// `2·degree` per span).
pub(crate) fn piecewise_polynomial_function_gram(
    breaks: &[f64],
    points_per_span: usize,
    eval: &mut dyn FnMut(ArrayView1<'_, f64>) -> Result<Array2<f64>, BasisError>,
) -> Result<Array2<f64>, BasisError> {
    if breaks.len() < 2 {
        crate::bail_invalid_basis!("function-space Gram requires at least one knot span");
    }
    let (nodes, weights) = gam_math::special::gauss_legendre(points_per_span);
    let mut quad_x = Vec::with_capacity((breaks.len() - 1) * points_per_span);
    let mut quad_w = Vec::with_capacity(quad_x.capacity());
    for span in breaks.windows(2) {
        let (a, b) = (span[0], span[1]);
        if !(b > a) {
            continue;
        }
        let half = 0.5 * (b - a);
        let mid = 0.5 * (a + b);
        for (&t, &w) in nodes.iter().zip(weights.iter()) {
            quad_x.push(mid + half * t);
            quad_w.push(half * w);
        }
    }
    if quad_x.is_empty() {
        crate::bail_invalid_basis!(
            "function-space Gram: every knot span is degenerate (zero length)"
        );
    }
    let x = Array1::from(quad_x);
    let design = eval(x.view())?;
    if design.nrows() != x.len() {
        crate::bail_dim_basis!(
            "function-space Gram evaluator returned {} rows for {} quadrature points",
            design.nrows(),
            x.len()
        );
    }
    let mut weighted = design.clone();
    for (mut row, &w) in weighted.axis_iter_mut(Axis(0)).zip(quad_w.iter()) {
        row *= w;
    }
    Ok(design.t().dot(&weighted))
}

/// Exact Gram of the raw (free-end) B-spline basis over its modeling interval
/// `[t_d, t_{K−1−d}]` (the span where the partition of unity holds).
pub(crate) fn bspline_function_gram(
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Array2<f64>, BasisError> {
    let k = knots.len();
    if k < 2 * (degree + 1) {
        crate::bail_invalid_basis!(
            "B-spline function Gram requires at least {} knots for degree {degree}, got {k}",
            2 * (degree + 1)
        );
    }
    let mut breaks = Vec::<f64>::with_capacity(k - 2 * degree);
    for i in degree..=(k - 1 - degree) {
        let t = knots[i];
        if breaks.last().is_none_or(|&prev| t > prev) {
            breaks.push(t);
        }
    }
    piecewise_polynomial_function_gram(&breaks, degree + 1, &mut |pts| {
        let (basis, _) = create_basis::<Dense>(
            pts,
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::value(),
        )?;
        Ok((*basis).clone())
    })
}

/// `R = W (NᵀW)⁻¹ Wᵀ` for a (Euclidean-orthonormal) null basis `N` and its
/// metric image `W = G·N` under an SPD function-space metric `G`. Writing
/// `C = NᵀW = NᵀGN = U diag(d) Uᵀ`, the G-orthonormal null frame is
/// `Z = N U diag(d)^{-1/2}` and `R = (GZ)(GZ)ᵀ = G N C⁻¹ Nᵀ G` — the penalty
/// whose quadratic form is `‖G-orthogonal null component of f‖²`. Invariant to
/// the choice of null basis `N`. Returns `Ok(None)` when `C` is not numerically
/// SPD (the metric action carried no usable information on the null space);
/// callers fall back to the Euclidean projector `NNᵀ` in that case.
fn ridge_from_null_metric_action(
    n: &Array2<f64>,
    w: &Array2<f64>,
) -> Result<Option<Array2<f64>>, BasisError> {
    let c_raw = n.t().dot(w);
    let (c_sym, evals, evecs) = spectral_summary(&c_raw)?;
    let tol = spectral_tolerance(&c_sym, &evals);
    if evals.iter().any(|&ev| ev <= tol) {
        return Ok(None);
    }
    let mut gz = w.dot(&evecs);
    for (mut col, &d) in gz.axis_iter_mut(Axis(1)).zip(evals.iter()) {
        col /= d.sqrt();
    }
    Ok(Some(fast_abt(&gz, &gz)))
}

/// Function-space double-penalty ridge: shrink the *function* component that
/// the primary penalty cannot see, not the raw coefficients (SPEC rule 5).
///
/// The Euclidean Marra & Wood projector `ZZᵀ` penalizes `‖null coefficients‖²`,
/// which changes under a harmless basis rescaling/reparameterization even when
/// the represented function is identical. With the basis Gram `G = ∫ b bᵀ`, the
/// ridge `G Z (ZᵀGZ)⁻¹ ZᵀG` penalizes `∫ (null component of f)²` instead — a
/// property of the function alone, covariant under any basis change
/// (`S → MᵀSM`, `G → MᵀGM` maps the ridge to exactly `MᵀRM`).
pub(crate) fn function_space_nullspace_shrinkage(
    penalty: &Array2<f64>,
    gram: &Array2<f64>,
) -> Result<Option<Array2<f64>>, BasisError> {
    if penalty.dim() != gram.dim() || penalty.nrows() != penalty.ncols() {
        crate::bail_dim_basis!(
            "function-space shrinkage: penalty is {}x{} but Gram is {}x{}",
            penalty.nrows(),
            penalty.ncols(),
            gram.nrows(),
            gram.ncols()
        );
    }
    if penalty.nrows() == 0 {
        return Ok(None);
    }
    let (sym, evals, evecs) = spectral_summary(penalty)?;
    let tol = spectral_tolerance(&sym, &evals);
    let zero_idx: Vec<usize> = evals
        .iter()
        .enumerate()
        .filter_map(|(i, &ev)| (ev.abs() <= tol).then_some(i))
        .collect();
    if zero_idx.is_empty() {
        return Ok(None);
    }
    let n = evecs.select(Axis(1), &zero_idx);
    let w = gram.dot(&n);
    Ok(Some(match ridge_from_null_metric_action(&n, &w)? {
        Some(ridge) => ridge,
        None => fast_abt(&n, &n),
    }))
}

/// Rebuild a double-penalty ridge after a coefficient reparameterization while
/// PRESERVING the function-space metric the original ridge encoded.
///
/// For any ridge of the form `R = G Z (ZᵀGZ)⁻¹ ZᵀG` (`G` SPD, `Z` spanning
/// `null(S)`) and any injective transform `M` applied as the congruences
/// `S_c = MᵀSM`, `R_c = MᵀRM`, the identity `R_c v = (MᵀGM) v` holds for every
/// `v ∈ null(S_c)`: `Mv ∈ null(S)` (PSD `S`), and the G-orthogonal projector
/// underlying `R` fixes null vectors. So the correct constrained-chart ridge
/// `G_c N (NᵀG_cN)⁻¹ NᵀG_c` is computable from `(S_c, R_c)` alone — no Gram
/// needs to travel with the candidate:
///
///   `N = null(S_c)`,  `W = R_c N (= G_c N)`,  `ridge = W (NᵀW)⁻¹ Wᵀ`.
///
/// With a Euclidean raw ridge (`G = I`, kernel bases) this reduces to the
/// projector `N(NᵀN)⁻¹Nᵀ` in the `MᵀM` metric, and when the transform removes
/// none of the null directions it is idempotent (`ridge = R_c` exactly).
/// Returns `Ok(None)` when the constrained primary has no null space.
pub(crate) fn rebuild_metric_consistent_ridge(
    primary_constrained: &Array2<f64>,
    ridge_constrained: &Array2<f64>,
) -> Result<Option<Array2<f64>>, BasisError> {
    if primary_constrained.dim() != ridge_constrained.dim()
        || primary_constrained.nrows() != primary_constrained.ncols()
    {
        crate::bail_dim_basis!(
            "metric-consistent ridge rebuild: primary is {}x{} but ridge is {}x{}",
            primary_constrained.nrows(),
            primary_constrained.ncols(),
            ridge_constrained.nrows(),
            ridge_constrained.ncols()
        );
    }
    if primary_constrained.nrows() == 0 {
        return Ok(None);
    }
    let (sym, evals, evecs) = spectral_summary(primary_constrained)?;
    let tol = spectral_tolerance(&sym, &evals);
    let zero_idx: Vec<usize> = evals
        .iter()
        .enumerate()
        .filter_map(|(i, &ev)| (ev.abs() <= tol).then_some(i))
        .collect();
    if zero_idx.is_empty() {
        return Ok(None);
    }
    let n = evecs.select(Axis(1), &zero_idx);
    let w = ridge_constrained.dot(&n);
    Ok(Some(match ridge_from_null_metric_action(&n, &w)? {
        Some(ridge) => ridge,
        None => fast_abt(&n, &n),
    }))
}

/// Build the double-penalty ridge from the structural null space of a PSD penalty.
pub fn build_nullspace_shrinkage_penalty(
    penalty: &Array2<f64>,
) -> Result<Option<CanonicalPenaltyBlock>, BasisError> {
    if penalty.nrows() != penalty.ncols() {
        crate::bail_dim_basis!(
            "penalty matrix must be square when building nullspace shrinkage penalty"
        );
    }
    if penalty.nrows() == 0 {
        return Ok(None);
    }

    let (sym, evals, evecs) = spectral_summary(penalty)?;
    let tol = spectral_tolerance(&sym, &evals);

    let zero_idx: Vec<usize> = evals
        .iter()
        .enumerate()
        .filter_map(|(i, &ev)| (ev.abs() <= tol).then_some(i))
        .collect();
    if zero_idx.is_empty() {
        return Ok(None);
    }
    let z = evecs.select(Axis(1), &zero_idx);
    let shrink = fast_abt(&z, &z);
    Ok(Some(CanonicalPenaltyBlock {
        sym_penalty: shrink,
        eigenvalues: evals,
        eigenvectors: evecs,
        rank: zero_idx.len(),
        nullity: 0,
        negative_dim: 0,
        tol,
        iszero: false,
        op: None,
    }))
}

pub(crate) fn default_internal_knot_count_for_data(n: usize, degree: usize) -> usize {
    if n < 8 {
        return 0;
    }
    let heuristic = if n < 16 { 3 } else { (n / 4).max(3) };
    let max_reasonable = n.saturating_sub(degree + 2);
    heuristic.min(40).min(max_reasonable)
}

/// Auto-shrink a requested B-spline configuration to the largest feasible
/// `(num_internal_knots, degree)` that the available data can support.
///
/// Issue #340: when `n` is small (e.g. `n = 3` with a default cubic + 4
/// interior knots), the upstream auto-knot placement otherwise errors out and
/// forces the user to manually downgrade their model. This helper folds the
/// downgrade decision into the engine so cubic-by-default gracefully degrades
/// to quadratic / linear, and the interior-knot count shrinks toward zero,
/// matching what the data can express.
///
/// Constraints we must satisfy for the generated clamped knot vector to be
/// non-degenerate:
///   * `n >= num_internal_knots + 2`  (one min, one max, one strict interior
///     value per interior knot in the best case)
///   * `n >= degree + 1`              (clamped vector covers `degree + 1`
///     boundary repeats on each side; we need at least that many distinct
///     evaluation points for the basis to span)
///
/// The shrink rule (deterministic, magic-by-default):
///   1. Cap `num_internal_knots` at `n.saturating_sub(2)`.
///   2. While `degree + 1 > n` and `degree > 1`, drop the degree by one
///      (cubic → quadratic → linear).
///   3. If even linear (`degree = 1`, requiring `n >= 2`) is impossible,
///      return `None` and let the caller raise a clear "not enough points"
///      diagnostic.
///
/// Returns `(effective_num_internal_knots, effective_degree)` along with a
/// boolean flagging whether either parameter was actually reduced.
pub(crate) fn auto_shrink_bspline_config(
    n: usize,
    requested_num_internal_knots: usize,
    requested_degree: usize,
) -> Option<(usize, usize, bool)> {
    if n < 2 {
        return None;
    }
    let mut degree = requested_degree.max(1);
    while degree + 1 > n && degree > 1 {
        degree -= 1;
    }
    if degree + 1 > n {
        return None;
    }
    let max_interior = n.saturating_sub(2);
    let num_internal_knots = requested_num_internal_knots.min(max_interior);
    let shrunk =
        num_internal_knots != requested_num_internal_knots || degree != requested_degree.max(1);
    Some((num_internal_knots, degree, shrunk))
}

/// Apply [`auto_shrink_bspline_config`] to a [`BSplineBasisSpec`] when the
/// caller's `knotspec` is data-driven, returning the (possibly mutated) spec.
///
/// Touches only auto / generate-from-range knot specs; `Provided` and
/// `PeriodicUniform` are pass-through because the caller has already
/// hard-committed to a specific basis geometry.
///
/// When a shrink actually happens, an info-level log message is emitted so
/// the decision is visible in fit logs and downstream model summaries.
pub(crate) fn maybe_auto_shrink_bspline_spec(
    spec: &BSplineBasisSpec,
    n: usize,
) -> (BSplineBasisSpec, Option<String>) {
    match &spec.knotspec {
        BSplineKnotSpec::Generate {
            data_range,
            num_internal_knots,
        } => {
            let Some((eff_interior, eff_degree, shrunk)) =
                auto_shrink_bspline_config(n, *num_internal_knots, spec.degree)
            else {
                return (spec.clone(), None);
            };
            if !shrunk {
                return (spec.clone(), None);
            }
            let note = format!(
                "auto-shrink (#340): n={n} too small for requested degree={req_deg}, \
                 interior_knots={req_ki}; using degree={eff_deg}, interior_knots={eff_ki}",
                n = n,
                req_deg = spec.degree,
                req_ki = num_internal_knots,
                eff_deg = eff_degree,
                eff_ki = eff_interior,
            );
            log::info!("B-spline {note} on Generate knotspec");
            let mut shrunk_spec = spec.clone();
            shrunk_spec.degree = eff_degree;
            shrunk_spec.knotspec = BSplineKnotSpec::Generate {
                data_range: *data_range,
                num_internal_knots: eff_interior,
            };
            (shrunk_spec, Some(note))
        }
        BSplineKnotSpec::Automatic {
            num_internal_knots,
            placement,
        } => {
            let requested_interior = num_internal_knots
                .unwrap_or_else(|| default_internal_knot_count_for_data(n, spec.degree));
            let Some((eff_interior, eff_degree, shrunk)) =
                auto_shrink_bspline_config(n, requested_interior, spec.degree)
            else {
                return (spec.clone(), None);
            };
            if !shrunk {
                return (spec.clone(), None);
            }
            let note = format!(
                "auto-shrink (#340): n={n} too small for requested degree={req_deg}, \
                 interior_knots={req_ki}; using degree={eff_deg}, interior_knots={eff_ki}",
                n = n,
                req_deg = spec.degree,
                req_ki = requested_interior,
                eff_deg = eff_degree,
                eff_ki = eff_interior,
            );
            log::info!("B-spline {note} on Automatic knotspec");
            let mut shrunk_spec = spec.clone();
            shrunk_spec.degree = eff_degree;
            shrunk_spec.knotspec = BSplineKnotSpec::Automatic {
                num_internal_knots: Some(eff_interior),
                placement: *placement,
            };
            (shrunk_spec, Some(note))
        }
        // cr/cs knots are frozen (value-at-knot), never auto-shrunk; the
        // explicitly-provided and periodic specs are likewise respected verbatim.
        BSplineKnotSpec::Provided(_)
        | BSplineKnotSpec::PeriodicUniform { .. }
        | BSplineKnotSpec::NaturalCubicRegression { .. } => (spec.clone(), None),
    }
}

pub(crate) fn finite_data_range(data: ArrayView1<'_, f64>) -> Result<(f64, f64), BasisError> {
    if data.is_empty() {
        crate::bail_invalid_basis!("cannot infer knot range from empty data");
    }
    if data.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!("automatic knot placement requires finite data values");
    }
    let mut minv = f64::INFINITY;
    let mut maxv = f64::NEG_INFINITY;
    for &x in data {
        if x < minv {
            minv = x;
        }
        if x > maxv {
            maxv = x;
        }
    }
    Ok((minv, maxv))
}

pub fn expand_periodic_centers(
    centers: &Array2<f64>,
    periodic: Option<&[Option<f64>]>,
) -> Result<Array2<f64>, BasisError> {
    let Some(periodic) = periodic else {
        return Ok(centers.clone());
    };
    if periodic.len() != centers.ncols() {
        crate::bail_dim_basis!(
            "period vector length {} does not match smooth dimension {}",
            periodic.len(),
            centers.ncols()
        );
    }
    let active: Vec<(usize, f64)> = periodic
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.map(|v| (i, v)))
        .collect();
    if active.is_empty() {
        return Ok(centers.clone());
    }
    for (axis, period) in &active {
        if !period.is_finite() || *period <= 0.0 {
            crate::bail_invalid_basis!(
                "period for axis {axis} must be finite and positive, got {period}"
            );
        }
    }
    let shifts = 3usize.pow(active.len() as u32);
    let mut out = Array2::<f64>::zeros((centers.nrows() * shifts, centers.ncols()));
    let mut row_out = 0usize;
    for code in 0..shifts {
        let mut tmp = code;
        let mut offsets = vec![0.0; centers.ncols()];
        for &(axis, period) in &active {
            let digit = tmp % 3;
            tmp /= 3;
            offsets[axis] = match digit {
                0 => -period,
                1 => 0.0,
                _ => period,
            };
        }
        for r in 0..centers.nrows() {
            for c in 0..centers.ncols() {
                out[[row_out, c]] = centers[[r, c]] + offsets[c];
            }
            row_out += 1;
        }
    }
    Ok(out)
}
