
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
        if spec.degree != 3 {
            crate::bail_invalid_basis!(
                "cyclic P-splines currently require cubic degree=3, got degree={}",
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
        let penalties_raw = vec![PenaltyCandidate {
            matrix: s_bend_raw.clone(),
            nullspace_dim_hint: 1,
            source: PenaltySource::Primary,
            normalization_scale: 1.0,
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
                    DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(design_c)),
                    transformed_candidates,
                    identifiability_transform,
                )
            };
        let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
            filter_active_penalty_candidates_with_ops(transformed_candidates)?;
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
    let auto_chunk_streaming = {
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
    };
    if let Some((knots, p_raw, chunk)) = auto_chunk_streaming {
        let greville_for_penalty = penalty_greville_abscissae_for_knots(&knots, spec.degree)?;
        let s_bend_raw = create_difference_penalty_matrix(
            p_raw,
            spec.penalty_order,
            greville_for_penalty.as_ref().map(|g| g.view()),
        )?;
        let mut penalties_raw = vec![PenaltyCandidate {
            matrix: s_bend_raw.clone(),
            nullspace_dim_hint: 0,
            source: PenaltySource::Primary,
            normalization_scale: 1.0,
            kronecker_factors: None,
            op: None,
        }];
        if spec.double_penalty {
            penalties_raw.push(PenaltyCandidate {
                matrix: build_nullspace_shrinkage_penalty(&s_bend_raw)?
                    .map(|shrink| shrink.sym_penalty)
                    .unwrap_or_else(|| Array2::<f64>::zeros(s_bend_raw.raw_dim())),
                nullspace_dim_hint: 0,
                source: PenaltySource::DoublePenaltyNullspace,
                normalization_scale: 1.0,
                kronecker_factors: None,
                op: None,
            });
        }
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
        let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
            filter_active_penalty_candidates_with_ops(transformed_candidates)?;
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
    // Boundary conditions are emitted by the smooth-level paired
    // linear-constraint path (`bspline_boundary_linear_constraints` in
    // smooth.rs), and `build_local_smooth_term` clears `spec.boundary_conditions`
    // before reaching this builder, so the basis builder no longer bakes them.
    // The `is_free` guard is retained defensively: any direct caller that still
    // supplies non-free endpoints must take the dense identifiability path
    // rather than the sparse branch, which never threaded boundary handling.
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
    let mut penalties_raw = vec![PenaltyCandidate {
        matrix: s_bend_raw.clone(),
        nullspace_dim_hint: 0,
        source: PenaltySource::Primary,
        normalization_scale: 1.0,
        kronecker_factors: None,
        op: None,
    }];
    if spec.double_penalty {
        penalties_raw.push(PenaltyCandidate {
            matrix: build_nullspace_shrinkage_penalty(&s_bend_raw)?
                .map(|shrink| shrink.sym_penalty)
                .unwrap_or_else(|| Array2::<f64>::zeros(s_bend_raw.raw_dim())),
            nullspace_dim_hint: 0,
            source: PenaltySource::DoublePenaltyNullspace,
            normalization_scale: 1.0,
            kronecker_factors: None,
            op: None,
        });
    }

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
                    DesignMatrix::Sparse(crate::matrix::SparseDesignMatrix::new(sparse_basis)),
                    transformed_candidates,
                    None,
                )
            }
            BSplineIdentifiability::WeightedSumToZero { weights } => {
                let (constrained_basis, z) = apply_sum_to_zero_constraint_sparse(
                    &sparse_basis,
                    weights.as_ref().map(|w| w.view()),
                )?;
                let transformed_candidates = penalties_raw
                    .into_iter()
                    .map(|candidate| -> Result<PenaltyCandidate, BasisError> {
                        let zt_s = fast_atb(&z, &candidate.matrix);
                        let matrix = fast_ab(&zt_s, &z);
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
                    DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(
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
        // Boundary conditions are no longer baked into the basis null space:
        // they are emitted by the smooth-level paired linear-constraint path
        // (`bspline_boundary_linear_constraints` in smooth.rs), which supports
        // non-zero anchors and composes with the frozen identifiability
        // transform. `build_local_smooth_term` clears `boundary_conditions`
        // before calling this builder, so the basis-level boundary path is no
        // longer reached.
        let (design, penalties, identifiability_transform) = apply_bspline_identifiability_policy(
            design_dense_opt.expect("dense B-spline basis should be present"),
            penalties_raw_mats,
            &knots,
            spec.degree,
            &spec.identifiability,
        )?;
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
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(design)),
            transformed_candidates,
            identifiability_transform,
        )
    };
    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(transformed_candidates)?;
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


fn compose_bspline_transform(
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


fn bspline_sum_to_zero_transform_from_cross(c: &Array1<f64>) -> Result<Array2<f64>, BasisError> {
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
            constrained_gram_max_eigenvalue: f64::NAN,
            constrained_gram_min_eigenvalue: f64::NAN,
            spectral_tolerance: f64::NAN,
        });
    }
    Ok(z)
}


fn streaming_bspline_current_chunk(
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


fn streaming_bspline_sum_cross(
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


fn streaming_bspline_orthogonality_transform(
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


fn build_streaming_bspline_design_and_candidates(
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
    // Boundary conditions are emitted by the smooth-level paired
    // linear-constraint path; the basis builder no longer bakes them, so this
    // streaming path starts from no boundary transform.
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
            penalty_mats = penalty_mats
                .into_iter()
                .map(|s| project_penalty_matrix(&s, Some(&z)))
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
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(op))),
        transformed_candidates,
        transform_opt,
    ))
}


fn apply_bspline_identifiability_policy(
    design: Array2<f64>,
    penalties: Vec<Array2<f64>>,
    knots: &Array1<f64>,
    degree: usize,
    identifiability: &BSplineIdentifiability,
) -> Result<(Array2<f64>, Vec<Array2<f64>>, Option<Array2<f64>>), BasisError> {
    let (design_c, z_opt): (Array2<f64>, Option<Array2<f64>>) = match identifiability {
        BSplineIdentifiability::None => (design, None),
        BSplineIdentifiability::WeightedSumToZero { weights } => {
            let (b_c, z) =
                apply_sum_to_zero_constraint(design.view(), weights.as_ref().map(|w| w.view()))?;
            (b_c, Some(z))
        }
        BSplineIdentifiability::RemoveLinearTrend => {
            let (z, _) = compute_geometric_constraint_transform(knots, degree, 2)?;
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
        penalties
            .into_iter()
            .map(|s| {
                let zt_s = fast_atb(z, &s);
                fast_ab(&zt_s, z)
            })
            .collect()
    } else {
        penalties
    };

    Ok((design_c, penalties_c, z_opt))
}


pub(crate) fn estimate_penalty_nullity(penalty: &Array2<f64>) -> Result<usize, BasisError> {
    if penalty.nrows() != penalty.ncols() {
        crate::bail_dim_basis!("penalty matrix must be square when estimating nullspace");
    }
    if penalty.nrows() == 0 {
        return Ok(0);
    }

    let (sym, evals, _) = spectral_summary(penalty)?;
    let tol = spectral_tolerance(&sym, &evals);
    Ok(evals.iter().filter(|&&ev| ev.abs() <= tol).count())
}


#[derive(Debug, Clone)]
struct PsdSpectralSummary {
    min_eigenvalue: f64,
    max_abs_eigenvalue: f64,
    tolerance: f64,
    effective_rank: usize,
}


fn symmetrize_penalty(penalty: &Array2<f64>) -> Array2<f64> {
    let mut sym = penalty.clone();
    for i in 0..sym.nrows() {
        for j in 0..i {
            let v = 0.5 * (sym[[i, j]] + sym[[j, i]]);
            sym[[i, j]] = v;
            sym[[j, i]] = v;
        }
    }
    sym
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


fn spectral_tolerance(sym: &Array2<f64>, evals: &Array1<f64>) -> f64 {
    let max_abs_ev = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    // Keep the cutoff in eigenvalue units so uniform penalty scaling does not
    // change PSD/rank decisions for the same spectrum shape.
    (sym.nrows().max(1) as f64) * 1e-10 * max_abs_ev
}


fn spectral_summary(
    penalty: &Array2<f64>,
) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), BasisError> {
    let sym = symmetrize_penalty(penalty);
    let (evals, evecs) = FaerEigh::eigh(&sym, Side::Lower).map_err(BasisError::LinalgError)?;
    Ok((sym, evals, evecs))
}


fn validate_psd_penalty(
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
    let min_eigenvalue = evals.iter().copied().fold(f64::INFINITY, f64::min);
    let max_abs_eigenvalue = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let effective_rank = evals.iter().filter(|&&ev| ev > tolerance).count();

    if min_eigenvalue < -tolerance {
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
    op: Option<std::sync::Arc<dyn crate::terms::penalty_op::PenaltyOp>>,
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
            tol: 1e-10,
            iszero: true,
            op,
        });
    }

    let (sym, evals, evecs) = spectral_summary(penalty)?;
    let tol = spectral_tolerance(&sym, &evals);
    let rank = evals.iter().filter(|&&ev| ev > tol).count();
    let nullity = sym.nrows().saturating_sub(rank);
    let max_abs_eigenvalue = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    Ok(CanonicalPenaltyBlock {
        sym_penalty: sym,
        eigenvalues: evals,
        eigenvectors: evecs,
        rank,
        nullity,
        tol,
        iszero: max_abs_eigenvalue <= tol,
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
/// `block.sym_penalty` at eigenvalues `≤ block.tol` — exactly the directions
/// along which `Sβ = 0` and on which `H_pen = H_loglik + S` carries no
/// curvature from the penalty. These are the directions that must be
/// absorbed into the parametric block at construction time so that the
/// smooth's design is orthogonal to its own null space and the inner Newton
/// solve does not get stuck refusing a non-existent stationary point.
fn nullspace_basis_from_block(block: &CanonicalPenaltyBlock) -> Option<Array2<f64>> {
    if block.nullity == 0 {
        return None;
    }
    let null_idx: Vec<usize> = block
        .eigenvalues
        .iter()
        .enumerate()
        .filter_map(|(i, &ev)| (ev <= block.tol).then_some(i))
        .collect();
    if null_idx.is_empty() {
        return None;
    }
    Some(block.eigenvectors.select(Axis(1), &null_idx))
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
/// (eigenvectors of `penalties[k]` at eigenvalues `≤ spectral_tolerance`)
/// when the block has a non-trivial null space, and `None` otherwise.
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
    let joint_nullity = evals.iter().filter(|&&ev| ev <= tol).count();
    if joint_nullity == 0 {
        return Ok(None);
    }
    // Order columns of Q as [U_range | U_null]: ascending-eigenvalue ordering
    // from `eigh` puts null columns first (eigenvalues near zero), so we need
    // to permute so range columns lead and null columns trail. This is the
    // canonical absorption ordering — Stage-3 split takes the last
    // `joint_nullity` columns as the absorbed parametric columns.
    let n = evals.len();
    let mut order: Vec<usize> = (0..n).collect();
    // Stable partition: range indices first (in original order, large
    // eigenvalues first by descending sort), then null indices (small ev).
    order.sort_by(|&a, &b| {
        let ev_a = evals[a];
        let ev_b = evals[b];
        let null_a = ev_a <= tol;
        let null_b = ev_b <= tol;
        match (null_a, null_b) {
            (false, true) => std::cmp::Ordering::Less,
            (true, false) => std::cmp::Ordering::Greater,
            // Both range or both null: descending by eigenvalue, with
            // NaN/sign tie-breaks unlikely on a PSD sum but handled safely.
            _ => ev_b.partial_cmp(&ev_a).unwrap_or(std::cmp::Ordering::Equal),
        }
    });
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
        Vec<Option<std::sync::Arc<dyn crate::terms::penalty_op::PenaltyOp>>>,
    ),
    BasisError,
> {
    let mut penalties = Vec::with_capacity(candidates.len());
    let mut nullspace_dims = Vec::with_capacity(candidates.len());
    let mut penaltyinfo = Vec::with_capacity(candidates.len());
    let mut active_null_eigenvectors: Vec<Option<Array2<f64>>> =
        Vec::with_capacity(candidates.len());
    let mut active_ops: Vec<Option<std::sync::Arc<dyn crate::terms::penalty_op::PenaltyOp>>> =
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


fn validated_kronecker_factors(
    factors: Option<Vec<Array2<f64>>>,
    matrix: &Array2<f64>,
) -> Option<Vec<Array2<f64>>> {
    let factors = factors?;
    let Some((first, rest)) = factors.split_first() else {
        return None;
    };
    let mut kron = first.clone();
    for factor in rest {
        kron = crate::construction::kronecker_product(&kron, factor);
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


/// Build the double-penalty ridge from the structural null space of a PSD penalty.
pub(crate) fn build_nullspace_shrinkage_penalty(
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
        tol,
        iszero: false,
        op: None,
    }))
}


fn default_internal_knot_count_for_data(n: usize, degree: usize) -> usize {
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
fn maybe_auto_shrink_bspline_spec(
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
        BSplineKnotSpec::Provided(_) | BSplineKnotSpec::PeriodicUniform { .. } => {
            (spec.clone(), None)
        }
    }
}


fn finite_data_range(data: ArrayView1<'_, f64>) -> Result<(f64, f64), BasisError> {
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


pub(crate) fn expand_periodic_centers(
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


/// Generic thin-plate builder returning design + penalty list.
pub fn build_thin_plate_basis(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_thin_plate_basiswithworkspace(data, spec, &mut workspace)
}


pub fn build_thin_plate_basiswithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisBuildResult, BasisError> {
    let original_centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let centers = expand_periodic_centers(&original_centers, spec.periodic.as_deref())?;
    // Canonical TPS in dimension d uses penalty order m = ⌊d/2⌋+1 and a
    // polynomial nullspace of size M(d) = C(d+m-1, d). For d=16 this is
    // 735_471, well above any practical knot count. When the requested
    // knot count is below M(d), canonical TPS is mathematically infeasible
    // (the constraint P(C)^T α = 0 is overdetermined). Rather than reject,
    // delegate to a hybrid Matern-Duchon spline — TPS's proper generalization
    // with an additional Riesz fractional smoothness s and a Matern-blended
    // spectrum — using parameters that satisfy Duchon's collocation gates:
    //   2(p + s) > d + max_op   pointwise kernel + collocation existence
    // We pick p = 2 (Linear nullspace, M' = d+1, well below typical k) and
    // the smallest s satisfying the gate. Hybrid (length_scale=Some) is used
    // rather than pure Duchon so the spatial-scale optimizer's log-κ
    // derivatives have a tunable kernel parameter (pure Duchon has none).
    if d_canonical_tps_infeasible(data.ncols(), centers.nrows())
        && let Some((nullspace_order, s)) =
            duchon_thin_plate_fallback_params(data.ncols(), centers.nrows())
    {
        let d = data.ncols();
        // The hybrid-Duchon partial-fraction kernel coefficients scale as
        // `kappa^(-2(p+s-n)) = length_scale^(2(p+s-n))` (see
        // `duchon_partial_fraction_coeffs`). With the high spectral order `s`
        // this auto-promotion selects (s ≥ 3 for d ≥ 6) and the Matern-style
        // auto-init length_scale (`max_range / sqrt(n)`, which is far below the
        // center spacing for moderate n), kappa·r runs large at every center
        // pair, every kernel block underflows toward machine epsilon, and the
        // constrained radial Gram collapses to floating-point noise
        // (`positive_spectral_whitener_from_gram` then rejects a rank-0 smooth —
        // gam#1091). The natural operating scale of a radial kernel is the
        // typical center separation, where kappa·r ≈ O(1) keeps every block
        // O(1); promote at that scale rather than inheriting the (possibly
        // tiny) Matern init. The outer optimizer still tunes psi = log kappa
        // from here, but it now starts from a non-degenerate basis.
        let promotion_length_scale =
            hybrid_duchon_promotion_length_scale(centers.view(), spec.length_scale);
        let duchon_spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(original_centers.clone()),
            periodic: spec.periodic.clone(),
            length_scale: Some(promotion_length_scale),
            power: s as f64,
            nullspace_order,
            identifiability: spec.identifiability.clone(),
            aniso_log_scales: None,
            operator_penalties: DuchonOperatorPenaltySpec::default(),
            boundary: OneDimensionalBoundary::Open,
        };
        log::info!(
            "thin-plate basis auto-promoted to hybrid Duchon ({:?}, s={}) in d={}: \
             canonical TPS would need {} centers but got {} — using Duchon's \
             Riesz-fractional generalization with finite kernel at r=0 \
             (length_scale={:.4e} from center spacing, was {:.4e})",
            nullspace_order,
            s,
            d,
            thin_plate_polynomial_basis_dimension(d),
            centers.nrows(),
            promotion_length_scale,
            spec.length_scale,
        );
        return build_duchon_basiswithworkspace(data, &duchon_spec, workspace);
    }
    let internal_kernel_transform =
        thin_plate_kernel_constraint_nullspace(centers.view(), &mut workspace.cache)?;
    let poly_cols = thin_plate_polynomial_basis_dimension(centers.ncols());
    let base_cols = internal_kernel_transform.ncols() + poly_cols;
    let dense_bytes = dense_design_bytes(data.nrows(), base_cols);
    let use_lazy = should_use_lazy_spatial_design(data.nrows(), base_cols, workspace.policy());
    if use_lazy {
        // log::info! — deliberate memory-saving choice, not an anomaly.
        log::info!(
            "thin-plate basis switching to lazy chunked design: n={} p={} ({:.1} MiB dense)",
            data.nrows(),
            base_cols,
            dense_bytes as f64 / (1024.0 * 1024.0),
        );
    }
    let (design, identifiability_transform, mut candidates, radial_reparam_meta) = if use_lazy {
        let poly_block = thin_plate_polynomial_block(data);
        let d = data.ncols();
        let length_scale_sq = spec.length_scale * spec.length_scale;
        let shared_data = shared_owned_data_matrix(data, &workspace.cache);
        let kernel_fn = move |data_row: &[f64], center_row: &[f64]| -> f64 {
            let mut dist2 = 0.0;
            for axis in 0..d {
                let delta = data_row[axis] - center_row[axis];
                dist2 += delta * delta;
            }
            thin_plate_kernel_from_dist2(dist2 / length_scale_sq, d)
                .expect("validated thin-plate inputs should not fail")
        };
        let base_op = ChunkedKernelDesignOperator::new(
            shared_data,
            Arc::new(centers.clone()),
            kernel_fn,
            Some(Arc::new(internal_kernel_transform.clone())),
            Some(Arc::new(poly_block)),
        )
        .map_err(BasisError::InvalidInput)?;
        let base_design =
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(base_op)));
        let identifiability_transform = thin_plate_identifiability_transform_from_design_matrix(
            &base_design,
            internal_kernel_transform.ncols(),
            poly_cols,
            &spec.identifiability,
        )?;
        let design = if let Some(transform) = identifiability_transform.as_ref() {
            wrap_dense_design_with_transform(base_design, transform, "ThinPlate")?
        } else {
            base_design
        };
        let (penalty_bending, penalty_ridge) = build_thin_plate_penalty_matrices(
            centers.view(),
            spec.length_scale,
            &internal_kernel_transform,
            spec.double_penalty,
        )?;
        let (penalty_bending_norm, c_bending) = normalize_penalty(&penalty_bending);
        let mut candidates = vec![PenaltyCandidate {
            matrix: penalty_bending_norm,
            nullspace_dim_hint: poly_cols,
            source: PenaltySource::Primary,
            normalization_scale: c_bending,
            kronecker_factors: None,
            op: None,
        }];
        if let Some(penalty_ridge) = penalty_ridge {
            let (penalty_ridge_norm, c_ridge) = normalize_penalty(&penalty_ridge);
            candidates.push(PenaltyCandidate {
                matrix: penalty_ridge_norm,
                nullspace_dim_hint: 0,
                source: PenaltySource::DoublePenaltyNullspace,
                normalization_scale: c_ridge,
                kronecker_factors: None,
                op: None,
            });
        }
        (design, identifiability_transform, candidates, None)
    } else {
        let tps = create_thin_plate_spline_basis_scaledwithworkspace(
            data,
            centers.view(),
            spec.length_scale,
            spec.radial_reparam.as_ref(),
            workspace,
        )?;
        let identifiability_transform = thin_plate_identifiability_transform_from_design(
            tps.basis.view(),
            tps.num_kernel_basis,
            tps.num_polynomial_basis,
            &spec.identifiability,
        )?;
        let design = if let Some(z) = identifiability_transform.as_ref() {
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(fast_ab(
                &tps.basis, z,
            )))
        } else {
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(tps.basis.clone()))
        };
        let (penalty_bending_norm, c_bending) = normalize_penalty(&tps.penalty_bending);
        let mut candidates = vec![PenaltyCandidate {
            matrix: penalty_bending_norm,
            nullspace_dim_hint: tps.num_polynomial_basis,
            source: PenaltySource::Primary,
            normalization_scale: c_bending,
            kronecker_factors: None,
            op: None,
        }];
        if spec.double_penalty {
            let (penalty_ridge_norm, c_ridge) = normalize_penalty(&tps.penalty_ridge);
            candidates.push(PenaltyCandidate {
                matrix: penalty_ridge_norm,
                nullspace_dim_hint: 0,
                source: PenaltySource::DoublePenaltyNullspace,
                normalization_scale: c_ridge,
                kronecker_factors: None,
                op: None,
            });
        }
        let radial_reparam_meta = Some(tps.radial_reparam.clone());
        (
            design,
            identifiability_transform,
            candidates,
            radial_reparam_meta,
        )
    };
    if let Some(z) = identifiability_transform.as_ref() {
        candidates = candidates
            .into_iter()
            .map(|candidate| -> Result<PenaltyCandidate, BasisError> {
                let zt_s = z.t().dot(&candidate.matrix);
                let matrix = zt_s.dot(z);
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
    }
    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(candidates)?;
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
        metadata: BasisMetadata::ThinPlate {
            centers: original_centers,
            length_scale: spec.length_scale,
            periodic: spec.periodic.clone(),
            identifiability_transform,
            input_scales: None,
            radial_reparam: radial_reparam_meta,
        },
        kronecker_factored: None,
    })
}


/// Canonical domain guard for Matérn kernel evaluations: distance `r` must be
/// finite and non-negative, length scale must be finite and positive. Single
/// source of truth for the `(r, length_scale)` validity check shared by every
/// Matérn kernel/derivative function below.
#[inline(always)]
fn validate_matern_inputs(r: f64, length_scale: f64) -> Result<(), BasisError> {
    if !r.is_finite() || r < 0.0 {
        crate::bail_invalid_basis!("Matérn kernel distance must be finite and non-negative");
    }
    validate_matern_length_scale(length_scale)
}


/// Canonical guard for the length-scale-only Matérn sites: length scale must be
/// finite and positive. Shared by `validate_matern_inputs` and by callers that
/// validate the distance separately (or have no distance argument).
#[inline(always)]
fn validate_matern_length_scale(length_scale: f64) -> Result<(), BasisError> {
    if !length_scale.is_finite() || length_scale <= 0.0 {
        crate::bail_invalid_basis!("Matérn length_scale must be finite and positive");
    }
    Ok(())
}


#[inline(always)]
fn matern_kernel_from_distance(r: f64, length_scale: f64, nu: MaternNu) -> Result<f64, BasisError> {
    validate_matern_inputs(r, length_scale)?;

    // Parameterization used here:
    //   x = r / length_scale
    //   a = sqrt(2ν) * x
    // and the half-integer Matérn closed forms are in terms of `a`:
    //   ν=1/2: exp(-a)
    //   ν=3/2: (1+a) exp(-a)
    //   ν=5/2: (1+a+a^2/3) exp(-a)
    // (for ν=1/2, a=x since sqrt(2ν)=1).
    let x = r / length_scale;
    let k = match nu {
        MaternNu::Half => stable_nonnegative_poly_times_exp_neg(x, &[1.0]),
        MaternNu::ThreeHalves => {
            let a = 3.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[1.0, 1.0])
        }
        MaternNu::FiveHalves => {
            let a = 5.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[1.0, 1.0, 1.0 / 3.0])
        }
        MaternNu::SevenHalves => {
            let a = 7.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[1.0, 1.0, 2.0 / 5.0, 1.0 / 15.0])
        }
        MaternNu::NineHalves => {
            let a = 9.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(
                a,
                &[1.0, 1.0, 3.0 / 7.0, 2.0 / 21.0, 1.0 / 105.0],
            )
        }
    };
    Ok(k)
}


#[inline(always)]
fn matern_kernel_log_kappa_derivative_from_distance(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<f64, BasisError> {
    validate_matern_inputs(r, length_scale)?;

    let x = r / length_scale;
    let deriv = match nu {
        MaternNu::Half => stable_nonnegative_poly_times_exp_neg(x, &[0.0, -1.0]),
        MaternNu::ThreeHalves => {
            let a = 3.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[0.0, 0.0, -1.0])
        }
        MaternNu::FiveHalves => {
            let a = 5.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[0.0, 0.0, -1.0 / 3.0, -1.0 / 3.0])
        }
        MaternNu::SevenHalves => {
            let a = 7.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(
                a,
                &[0.0, 0.0, -1.0 / 5.0, -1.0 / 5.0, -1.0 / 15.0],
            )
        }
        MaternNu::NineHalves => {
            let a = 9.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(
                a,
                &[0.0, 0.0, -1.0 / 7.0, -1.0 / 7.0, -2.0 / 35.0, -1.0 / 105.0],
            )
        }
    };
    Ok(deriv)
}


#[inline(always)]
fn matern_kernel_log_kappasecond_derivative_from_distance(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<f64, BasisError> {
    validate_matern_inputs(r, length_scale)?;

    let x = r / length_scale;
    let second = match nu {
        MaternNu::Half => stable_nonnegative_poly_times_exp_neg(x, &[0.0, -1.0, 1.0]),
        MaternNu::ThreeHalves => {
            let a = 3.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[0.0, 0.0, -2.0, 1.0])
        }
        MaternNu::FiveHalves => {
            let a = 5.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[0.0, 0.0, -2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0])
        }
        MaternNu::SevenHalves => {
            let a = 7.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(
                a,
                &[0.0, 0.0, -2.0 / 5.0, -2.0 / 5.0, -1.0 / 15.0, 1.0 / 15.0],
            )
        }
        MaternNu::NineHalves => {
            let a = 9.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(
                a,
                &[
                    0.0,
                    0.0,
                    -2.0 / 7.0,
                    -2.0 / 7.0,
                    -3.0 / 35.0,
                    1.0 / 105.0,
                    1.0 / 105.0,
                ],
            )
        }
    };
    Ok(second)
}


#[inline(always)]
fn matern_kernel_radial_tripletwith_safe_ratio(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<(f64, f64, f64, f64), BasisError> {
    validate_matern_inputs(r, length_scale)?;

    // Full derivation used by collocation operators:
    //   phi(r) = P_nu(a) exp(-a), a=sr, s=sqrt(2nu)/length_scale.
    // For nu>=3/2 we use closed-form phi'(r)/r polynomials with finite r->0 limit.
    // For nu=1/2:
    //   phi'(r)/r = -kappa exp(-kappa r)/r,
    // which is genuinely singular at r=0 and must not be regularized here.
    // Closed forms used below (a = s r, E = exp(-a)):
    // nu=1/2:
    //   phi'    = -s E
    //   phi''   =  s^2 E
    //   phi'/r  diverges as -s/r (regularized via r floor).
    // nu=3/2:
    //   phi'    = -s E a
    //   phi''   =  s^2 E (a-1)
    //   phi'/r  = -s^2 E.
    // nu=5/2:
    //   phi'    = -(s/3) E a(a+1)
    //   phi''   =  (s^2/3) E (a^2-a-1)
    //   phi'/r  = -(s^2/3) E (a+1).
    // nu=7/2:
    //   phi'    = -(s/15) E a(a^2+3a+3)
    //   phi''   =  (s^2/15) E (a^3-3a-3)
    //   phi'/r  = -(s^2/15) E (a^2+3a+3).
    // nu=9/2:
    //   phi'    = -(s/105) E a(a^3+6a^2+15a+15)
    //   phi''   =  (s^2/105) E (a^4+2a^3-3a^2-15a-15)
    //   phi'/r  = -(s^2/105) E (a^3+6a^2+15a+15).
    let (phi, phi_r, phi_rr, phi_r_over_r) = match nu {
        MaternNu::Half => {
            let s = 1.0 / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = e;
            let phi_r = -s * e;
            let phi_rr = s * s * e;
            // Safe ratio regularization at r=0 to keep operator assembly finite.
            let r_eff = r.max(1e-12);
            let ratio = phi_r / r_eff;
            (phi, phi_r, phi_rr, ratio)
        }
        MaternNu::ThreeHalves => {
            let s = 3.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a) * e;
            let phi_r = -s * e * a;
            let phi_rr = s * s * e * (a - 1.0);
            let ratio = -s * s * e;
            (phi, phi_r, phi_rr, ratio)
        }
        MaternNu::FiveHalves => {
            let s = 5.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a + (a * a) / 3.0) * e;
            let phi_r = -(s / 3.0) * e * a * (a + 1.0);
            let phi_rr = (s * s / 3.0) * e * (a * a - a - 1.0);
            let ratio = -(s * s / 3.0) * e * (a + 1.0);
            (phi, phi_r, phi_rr, ratio)
        }
        MaternNu::SevenHalves => {
            let s = 7.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a + (2.0 / 5.0) * a * a + (1.0 / 15.0) * a * a * a) * e;
            let phi_r = -(s / 15.0) * e * a * (a * a + 3.0 * a + 3.0);
            let phi_rr = (s * s / 15.0) * e * (a * a * a - 3.0 * a - 3.0);
            let ratio = -(s * s / 15.0) * e * (a * a + 3.0 * a + 3.0);
            (phi, phi_r, phi_rr, ratio)
        }
        MaternNu::NineHalves => {
            let s = 9.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0
                + a
                + (3.0 / 7.0) * a * a
                + (2.0 / 21.0) * a * a * a
                + (1.0 / 105.0) * a * a * a * a)
                * e;
            let phi_r = -(s / 105.0) * e * a * (a * a * a + 6.0 * a * a + 15.0 * a + 15.0);
            let phi_rr = (s * s / 105.0)
                * e
                * (a * a * a * a + 2.0 * a * a * a - 3.0 * a * a - 15.0 * a - 15.0);
            let ratio = -(s * s / 105.0) * e * (a * a * a + 6.0 * a * a + 15.0 * a + 15.0);
            (phi, phi_r, phi_rr, ratio)
        }
    };

    if !phi.is_finite() || !phi_r.is_finite() || !phi_rr.is_finite() || !phi_r_over_r.is_finite() {
        crate::bail_invalid_basis!(
            "non-finite Matérn radial derivatives at r={r}, length_scale={length_scale}, nu={nu:?}"
        );
    }
    Ok((phi, phi_r, phi_rr, phi_r_over_r))
}


/// Extended radial scalars for exact per-axis eta_a derivatives of the Matérn
/// operator collocation matrices D1 (gradient) and D2 (full Hessian).
///
/// Returns `(phi, q, t, dt_dr, d2t_dr2)` where:
///   - phi   = φ(r)                  (kernel value)
///   - q     = φ'(r)/r               (used in D₁)
///   - t     = (phi''(r) - q) / r^2  (Hessian mixed-curvature scalar)
///   - dt_dr = dt/dr                 (needed for second eta-derivatives)
///   - d2t_dr2 = d2t/dr2             (needed for second eta-derivatives)
///
/// At r = 0 (center collision), the function returns zeros for all quantities
/// that would be multiplied by s_a (which also vanishes at collision).
///
/// For ν = 1/2 and ν = 3/2 where t and/or dt_dr diverge at r = 0, the
/// collision entries are safe because D₁ and D₂ derivatives at coincident
/// centers vanish via s_a = 0.
fn matern_aniso_extended_radial_scalars(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<(f64, f64, f64, f64, f64), BasisError> {
    if !r.is_finite() || r < 0.0 {
        crate::bail_invalid_basis!(
            "Matérn extended radial scalar distance must be finite and non-negative"
        );
    }
    validate_matern_length_scale(length_scale)?;

    match nu {
        // ----------------------------------------------------------------
        // ν = 1/2:  φ = exp(-a), a = r / ℓ, s = 1/ℓ
        //   q = -s·E/r  (diverges at r=0)
        //   t = (s²E - q) / r²  (diverges at r=0)
        //   At r=0 all products with s_a vanish, so return 0 for dt_dr, d2t_dr2.
        // ----------------------------------------------------------------
        MaternNu::Half => {
            let s = 1.0 / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = e;
            if r < 1e-14 {
                // Center collision. φ(r) = exp(−s r) has a cusp at r = 0, so
                // the radial scalars q = φ'/r and t = (φ'' − q)/r² diverge.
                // But every consumer multiplies them by displacement factors
                // that vanish identically at a coincident center:
                //   * the design-matrix η-derivatives are q·s_a and t·s_a·s_b
                //     (true value 0 — φ ≡ 1 there, independent of length scale);
                //   * the operator-collocation gradient row is q·h_b (h_b = 0);
                //   * the only term not pre-multiplied by a vanishing factor is
                //     the D₂ operator diagonal q·w_b, which the *value* path
                //     defines via the same convention — `phi_r_over_r = 0` for
                //     ν = 1/2 in 1D (the 1D Laplacian Δφ = φ'' carries no φ'/r
                //     term; see the base assembly below), and bails for d ≥ 2
                //     (already rejected at term construction).
                // Returning the convention-consistent zeros keeps the analytic
                // κ-gradient in lockstep with its own value surface — mirroring
                // the ν = 3/2 branch — rather than hard-erroring on a quantity
                // that is multiplied away.
                return Ok((phi, 0.0, 0.0, 0.0, 0.0));
            }
            let q = -s * e / r;
            let phi_rr = s * s * e;
            let t = (phi_rr - q) / (r * r);
            // t' from: t = f/r² where f = φ'' - q.
            //   f'  = φ''' - q' = -s³E - t·r   (since q' = t·r)
            //   t'  = (f' - 2t·r) / r²  = (-s³E - 3t·r) / r²
            let dt_dr = (-s * s * s * e - 3.0 * t * r) / (r * r);
            // t'' from: t' = g/r² where g = -s³E - 3tr.
            //   g' = s⁴E - 3(t'r + t)
            //   t'' = (g' - 2t'r) / r² = (s⁴E - 3t'r - 3t - 2t'r) / r²
            //        = (s⁴E - 5t'r - 3t) / r²
            let d2t_dr2 = (s.powi(4) * e - 5.0 * dt_dr * r - 3.0 * t) / (r * r);
            Ok((phi, q, t, dt_dr, d2t_dr2))
        }
        // ----------------------------------------------------------------
        // ν = 3/2:  φ = (1 + a)E, a = √3·r/ℓ, s = √3/ℓ
        //   q  = -s²E         (finite at r=0)
        //   t  = s³E/r        (diverges at r=0)
        //   dt/dr = s³E(-sr - 1)/r²  (diverges at r=0)
        //   At r=0, s_a = 0 so all products vanish.
        // ----------------------------------------------------------------
        MaternNu::ThreeHalves => {
            let s = 3.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a) * e;
            let q = -s * s * e;
            if r < 1e-14 {
                return Ok((phi, q, 0.0, 0.0, 0.0));
            }
            let t = s * s * s * e / r;
            // dt/dr: d/dr [s³ E / r] = s³ [-s E r - E] / r² = -s³ E (sr + 1) / r²
            let dt_dr = -s * s * s * e * (a + 1.0) / (r * r);
            // d²t/dr²: d/dr [-s³ E (a+1) / r²]
            //   = -s³ [(-s E)(a+1)r² + s E r² - 2r E(a+1)] / r⁴ ... expand
            // Let g(r) = -s³ E (a+1) / r²
            // g'(r) = -s³ [E'(a+1) + E·s] / r² + 2s³ E(a+1) / r³
            //       = -s³ [-sE(a+1) + sE] / r² + 2s³ E(a+1) / r³
            //       = -s³ · sE[-a-1+1] / r² + 2s³ E(a+1) / r³
            //       = s⁴ a E / r² + 2s³ E(a+1) / r³
            //       = s³ E [s a r + 2(a+1)] / r³
            let d2t_dr2 = s * s * s * e * (s * a * r + 2.0 * (a + 1.0)) / (r * r * r);
            Ok((phi, q, t, dt_dr, d2t_dr2))
        }
        // ----------------------------------------------------------------
        // ν = 5/2:  φ = (1 + a + a²/3)E, a = √5·r/ℓ, s = √5/ℓ
        //   q = -(s²/3)(a+1)E
        //   t = (s⁴/3)E
        //   dt/dr = -(s⁵/3)E
        //   d²t/dr² = (s⁶/3)E
        // ----------------------------------------------------------------
        MaternNu::FiveHalves => {
            let s = 5.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a + (a * a) / 3.0) * e;
            let q = -(s * s / 3.0) * (a + 1.0) * e;
            let t = (s * s * s * s / 3.0) * e;
            let dt_dr = -(s * s * s * s * s / 3.0) * e;
            let d2t_dr2 = (s.powi(6) / 3.0) * e;
            Ok((phi, q, t, dt_dr, d2t_dr2))
        }
        // ----------------------------------------------------------------
        // ν = 7/2:  φ = (1 + a + (2/5)a² + (1/15)a³)E
        //   q = -(s²/15)(a² + 3a + 3)E
        //   t = (s⁴/15)(a + 1)E
        //   dt/dr = -(s⁵/15)aE
        //   d²t/dr² = (s⁶/15)(a - 1)E
        // ----------------------------------------------------------------
        MaternNu::SevenHalves => {
            let s = 7.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a + (2.0 / 5.0) * a * a + (1.0 / 15.0) * a * a * a) * e;
            let q = -(s * s / 15.0) * (a * a + 3.0 * a + 3.0) * e;
            let t = (s * s * s * s / 15.0) * (a + 1.0) * e;
            let dt_dr = -(s.powi(5) / 15.0) * a * e;
            let d2t_dr2 = (s.powi(6) / 15.0) * (a - 1.0) * e;
            Ok((phi, q, t, dt_dr, d2t_dr2))
        }
        // ----------------------------------------------------------------
        // ν = 9/2:  φ = (1 + a + (3/7)a² + (2/21)a³ + (1/105)a⁴)E
        //   q = -(s²/105)(a³ + 6a² + 15a + 15)E
        //   t = (s⁴/105)(a² + 3a + 3)E
        //   dt/dr = -(s⁵/105)a(a + 1)E
        //   d²t/dr² = (s⁶/105)(a² - a - 1)E
        // ----------------------------------------------------------------
        MaternNu::NineHalves => {
            let s = 9.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0
                + a
                + (3.0 / 7.0) * a * a
                + (2.0 / 21.0) * a * a * a
                + (1.0 / 105.0) * a * a * a * a)
                * e;
            let q = -(s * s / 105.0) * (a * a * a + 6.0 * a * a + 15.0 * a + 15.0) * e;
            let t = (s * s * s * s / 105.0) * (a * a + 3.0 * a + 3.0) * e;
            let dt_dr = -(s.powi(5) / 105.0) * a * (a + 1.0) * e;
            let d2t_dr2 = (s.powi(6) / 105.0) * (a * a - a - 1.0) * e;
            Ok((phi, q, t, dt_dr, d2t_dr2))
        }
    }
}


#[inline(always)]
fn hessian_operator_entry(
    q: f64,
    t: f64,
    h_b: f64,
    h_c: f64,
    w_b: f64,
    w_c: f64,
    axis_b: usize,
    axis_c: usize,
) -> f64 {
    let diagonal = if axis_b == axis_c { w_b * q } else { 0.0 };
    diagonal + (w_b * h_b) * (w_c * h_c) * t
}


#[inline(always)]
fn hessian_operator_eta_entry(
    q: f64,
    t: f64,
    t_r: f64,
    r: f64,
    s_a: f64,
    h_b: f64,
    h_c: f64,
    w_b: f64,
    w_c: f64,
    axis_a: usize,
    axis_b: usize,
    axis_c: usize,
) -> f64 {
    let a_is_b = usize::from(axis_a == axis_b) as f64;
    let a_is_c = usize::from(axis_a == axis_c) as f64;
    let q_a = t * s_a;
    let t_a = if r > 1e-14 { t_r * s_a / r } else { 0.0 };
    let diagonal = if axis_b == axis_c {
        w_b * (2.0 * a_is_b * q + q_a)
    } else {
        0.0
    };
    let mixed_multiplier = 2.0 * a_is_b + 2.0 * a_is_c;
    diagonal + (w_b * h_b) * (w_c * h_c) * (mixed_multiplier * t + t_a)
}


#[inline(always)]
fn hessian_operator_eta2_entry(
    q: f64,
    t: f64,
    t_r: f64,
    t_rr: f64,
    r: f64,
    s_a: f64,
    h_b: f64,
    h_c: f64,
    w_b: f64,
    w_c: f64,
    axis_a: usize,
    axis_b: usize,
    axis_c: usize,
) -> f64 {
    let a_is_b = usize::from(axis_a == axis_b) as f64;
    let a_is_c = usize::from(axis_a == axis_c) as f64;
    let q_a = t * s_a;
    let q_aa = if r > 1e-14 {
        t_r * s_a * s_a / r + 2.0 * t * s_a
    } else {
        0.0
    };
    let t_a = if r > 1e-14 { t_r * s_a / r } else { 0.0 };
    let t_aa = if r > 1e-14 {
        ((t_rr * r - t_r) / (r * r * r)) * s_a * s_a + 2.0 * t_r * s_a / r
    } else {
        0.0
    };
    let diagonal = if axis_b == axis_c {
        w_b * (4.0 * a_is_b * q + 4.0 * a_is_b * q_a + q_aa)
    } else {
        0.0
    };
    let mixed_multiplier = 2.0 * a_is_b + 2.0 * a_is_c;
    diagonal
        + (w_b * h_b)
            * (w_c * h_c)
            * (mixed_multiplier * mixed_multiplier * t + 2.0 * mixed_multiplier * t_a + t_aa)
}


#[inline(always)]
fn hessian_operator_eta_cross_entry(
    t: f64,
    t_r: f64,
    t_rr: f64,
    r: f64,
    s_i: f64,
    s_j: f64,
    h_b: f64,
    h_c: f64,
    w_b: f64,
    w_c: f64,
    axis_i: usize,
    axis_j: usize,
    axis_b: usize,
    axis_c: usize,
) -> f64 {
    assert_ne!(axis_i, axis_j);
    let i_is_b = usize::from(axis_i == axis_b) as f64;
    let i_is_c = usize::from(axis_i == axis_c) as f64;
    let j_is_b = usize::from(axis_j == axis_b) as f64;
    let j_is_c = usize::from(axis_j == axis_c) as f64;
    let q_i = t * s_i;
    let q_j = t * s_j;
    let q_ij = if r > 1e-14 { t_r * s_i * s_j / r } else { 0.0 };
    let t_i = if r > 1e-14 { t_r * s_i / r } else { 0.0 };
    let t_j = if r > 1e-14 { t_r * s_j / r } else { 0.0 };
    let t_ij = if r > 1e-14 {
        ((t_rr * r - t_r) / (r * r * r)) * s_i * s_j
    } else {
        0.0
    };
    let diagonal = if axis_b == axis_c {
        w_b * (2.0 * i_is_b * q_j + 2.0 * j_is_b * q_i + q_ij)
    } else {
        0.0
    };
    let m_i = 2.0 * i_is_b + 2.0 * i_is_c;
    let m_j = 2.0 * j_is_b + 2.0 * j_is_c;
    diagonal + (w_b * h_b) * (w_c * h_c) * (m_i * m_j * t + m_i * t_j + m_j * t_i + t_ij)
}


/// Build exact per-axis η_a derivatives of operator penalty matrices for
/// anisotropic Matérn terms.
///
/// Instead of the fractional approximation `dS_op/dη_a ≈ f_a · dS_op/dψ`,
/// this computes exact first and second η_a derivatives of each operator
/// collocation matrix (D₀, D₁, D₂) and assembles the Gram product-rule
/// derivatives:
///   S_{m,a}  = D_{m,a}ᵀ D_m + D_mᵀ D_{m,a}
///   S_{m,aa} = D_{m,aa}ᵀ D_m + 2 D_{m,a}ᵀ D_{m,a} + D_mᵀ D_{m,aa}
///
/// ## Per-axis derivative formulas (y-space operators)
///
/// With r = √(Σ exp(2η_a) h_a²) and s_a = exp(2η_a) h_a²:
///
/// **D₀[k,j] = φ(r):**
///   ∂φ/∂η_a = q · s_a
///   ∂²φ/∂η_a² = t · s_a² + 2q · s_a
///
/// **D₁[(k,b),j] = q(r) · h_b** (y-space gradient):
///   ∂D₁/∂η_a = t · s_a · h_b
///   ∂²D₁/∂η_a² = (dt/dr · s_a²/r + 2t · s_a) · h_b
///
/// **D₂[k,j] = φ''(r) + (d-1)·q(r)** (y-space Laplacian):
///   ∂D₂/∂η_a = [(d+2)·t + dt/dr · r] · s_a
///   ∂²D₂/∂η_a² = [(d+3)·dt/dr/r + d²t/dr²] · s_a² + 2·[(d+2)·t + dt/dr·r] · s_a
struct MaternCrossPenaltyContext {
    centers: Array2<f64>,
    aniso_log_scales: Vec<f64>,
    length_scale: f64,
    nu: MaternNu,
    z_transform: Option<Array2<f64>>,
    penaltyinfo: Vec<PenaltyInfo>,
    d0: Array2<f64>,
    d1: Array2<f64>,
    d2: Array2<f64>,
    d0_eta_proj: Vec<Array2<f64>>,
    d1_eta_proj: Vec<Array2<f64>>,
    d2_eta_proj: Vec<Array2<f64>>,
    op0_s_raw: Array2<f64>,
    op1_s_raw: Array2<f64>,
    op2_s_raw: Array2<f64>,
    op0_c: f64,
    op1_c: f64,
    op2_c: f64,
    op0_s_first_raw: Vec<Array2<f64>>,
    op1_s_first_raw: Vec<Array2<f64>>,
    op2_s_first_raw: Vec<Array2<f64>>,
}


impl MaternCrossPenaltyContext {
    fn project_operator(&self, mat: &Array2<f64>, row_dim: usize) -> Array2<f64> {
        let kernel = if let Some(z) = self.z_transform.as_ref() {
            fast_ab(mat, z)
        } else {
            mat.clone()
        };
        let mut padded = Array2::<f64>::zeros((row_dim, self.d0.ncols()));
        padded.slice_mut(s![.., 0..kernel.ncols()]).assign(&kernel);
        padded
    }

    fn compute_pair(&self, axis_a: usize, axis_b: usize) -> Result<Vec<Array2<f64>>, BasisError> {
        let p = self.centers.nrows();
        let d = self.centers.ncols();
        let mut d0_cross_raw = Array2::<f64>::zeros((p, p));
        let mut d1_cross_raw = Array2::<f64>::zeros((p * d, p));
        let mut d2_cross_raw = Array2::<f64>::zeros((p * d * d, p));
        let metric_weights = centered_aniso_metric_weights(&self.aniso_log_scales);

        for k in 0..p {
            for j in 0..p {
                let ci: Vec<f64> = (0..d).map(|axis| self.centers[[k, axis]]).collect();
                let cj: Vec<f64> = (0..d).map(|axis| self.centers[[j, axis]]).collect();
                let (r, s_vec) = aniso_distance_and_components(&ci, &cj, &self.aniso_log_scales);
                let (_, _, t, dt_dr, d2t_dr2) =
                    matern_aniso_extended_radial_scalars(r, self.length_scale, self.nu)?;
                let s_a = s_vec[axis_a];
                let s_b = s_vec[axis_b];
                let sa_sb = s_a * s_b;

                d0_cross_raw[[k, j]] = t * sa_sb;
                for axis in 0..d {
                    let h_axis = ci[axis] - cj[axis];
                    let w_axis = metric_weights[axis];
                    let row = k * d + axis;
                    d1_cross_raw[[row, j]] = if r > 1e-14 {
                        dt_dr * sa_sb / r * h_axis
                            + if axis == axis_a {
                                2.0 * t * s_b * h_axis
                            } else {
                                0.0
                            }
                            + if axis == axis_b {
                                2.0 * t * s_a * h_axis
                            } else {
                                0.0
                            }
                    } else {
                        0.0
                    } * w_axis;
                }
                for b in 0..d {
                    let h_b = ci[b] - cj[b];
                    let w_b = metric_weights[b];
                    for c in 0..d {
                        let h_c = ci[c] - cj[c];
                        let w_c = metric_weights[c];
                        let row = (k * d + b) * d + c;
                        d2_cross_raw[[row, j]] = hessian_operator_eta_cross_entry(
                            t, dt_dr, d2t_dr2, r, s_a, s_b, h_b, h_c, w_b, w_c, axis_a, axis_b, b,
                            c,
                        );
                    }
                }
            }
        }

        let d0_cross_proj = self.project_operator(&d0_cross_raw, p);
        let d1_cross_proj = self.project_operator(&d1_cross_raw, p * d);
        let d2_cross_proj = self.project_operator(&d2_cross_raw, p * d * d);

        let s0_cross = normalize_penalty_cross_psi_derivative(
            &self.op0_s_raw,
            &self.op0_s_first_raw[axis_a],
            &self.op0_s_first_raw[axis_b],
            &gram_cross_psi_derivative_from_operator(
                &self.d0,
                &self.d0_eta_proj[axis_a],
                &self.d0_eta_proj[axis_b],
                &d0_cross_proj,
            ),
            self.op0_c,
        );
        let s1_cross = normalize_penalty_cross_psi_derivative(
            &self.op1_s_raw,
            &self.op1_s_first_raw[axis_a],
            &self.op1_s_first_raw[axis_b],
            &gram_cross_psi_derivative_from_operator(
                &self.d1,
                &self.d1_eta_proj[axis_a],
                &self.d1_eta_proj[axis_b],
                &d1_cross_proj,
            ),
            self.op1_c,
        );
        let s2_cross = normalize_penalty_cross_psi_derivative(
            &self.op2_s_raw,
            &self.op2_s_first_raw[axis_a],
            &self.op2_s_first_raw[axis_b],
            &gram_cross_psi_derivative_from_operator(
                &self.d2,
                &self.d2_eta_proj[axis_a],
                &self.d2_eta_proj[axis_b],
                &d2_cross_proj,
            ),
            self.op2_c,
        );

        active_operator_penalty_derivatives(
            &self.penaltyinfo,
            &[s0_cross, s1_cross, s2_cross],
            "Matérn-aniso-cross",
        )
    }
}


fn build_matern_operator_penalty_aniso_derivatives(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    z_opt: Option<&Array2<f64>>,
    eta: &[f64],
) -> Result<
    (
        Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)>,
        Vec<(usize, usize)>,
        AnisoPenaltyCrossProvider,
    ),
    BasisError,
> {
    let p = centers.nrows();
    let d = centers.ncols();
    let dim = eta.len();
    assert_eq!(dim, d);

    // Per-axis: build raw D0, D1, and full-Hessian D2 plus their eta_a
    // first/second derivatives.
    let mut d0_raw = Array2::<f64>::zeros((p, p));
    let mut d1_raw = Array2::<f64>::zeros((p * d, p));
    let mut d2_raw = Array2::<f64>::zeros((p * d * d, p));
    let mut d0_raw_eta: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p, p))).collect();
    let mut d1_raw_eta: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p * d, p))).collect();
    let mut d2_raw_eta: Vec<Array2<f64>> =
        (0..dim).map(|_| Array2::zeros((p * d * d, p))).collect();
    let mut d0_raw_eta2: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p, p))).collect();
    let mut d1_raw_eta2: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p * d, p))).collect();
    let mut d2_raw_eta2: Vec<Array2<f64>> =
        (0..dim).map(|_| Array2::zeros((p * d * d, p))).collect();
    let num_cross = dim * (dim - 1) / 2;
    let mut cross_pairs: Vec<(usize, usize)> = Vec::with_capacity(num_cross);
    for a in 0..dim {
        for b in (a + 1)..dim {
            cross_pairs.push((a, b));
        }
    }
    let metric_weights = centered_aniso_metric_weights(eta);

    struct CenterRowAccumulator {
        k: usize,
        d0: Array1<f64>,
        d1: Array2<f64>,
        d2: Array2<f64>,
        d0_eta: Vec<Array1<f64>>,
        d1_eta: Vec<Array2<f64>>,
        d2_eta: Vec<Array2<f64>>,
        d0_eta2: Vec<Array1<f64>>,
        d1_eta2: Vec<Array2<f64>>,
        d2_eta2: Vec<Array2<f64>>,
    }

    let row_accumulators: Vec<CenterRowAccumulator> = (0..p)
        .into_par_iter()
        .map(|k| -> Result<CenterRowAccumulator, BasisError> {
            let ci: Vec<f64> = (0..d).map(|a| centers[[k, a]]).collect();
            let mut d0 = Array1::<f64>::zeros(p);
            let mut d1 = Array2::<f64>::zeros((d, p));
            let mut d2 = Array2::<f64>::zeros((d * d, p));
            let mut d0_eta: Vec<Array1<f64>> = (0..dim).map(|_| Array1::zeros(p)).collect();
            let mut d1_eta: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((d, p))).collect();
            let mut d2_eta: Vec<Array2<f64>> =
                (0..dim).map(|_| Array2::zeros((d * d, p))).collect();
            let mut d0_eta2: Vec<Array1<f64>> = (0..dim).map(|_| Array1::zeros(p)).collect();
            let mut d1_eta2: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((d, p))).collect();
            let mut d2_eta2: Vec<Array2<f64>> =
                (0..dim).map(|_| Array2::zeros((d * d, p))).collect();

            for j in 0..p {
                let cj: Vec<f64> = (0..d).map(|a| centers[[j, a]]).collect();
                let (r, s_vec) = aniso_distance_and_components(&ci, &cj, eta);

                let (phi, q, t, dt_dr, d2t_dr2) =
                    matern_aniso_extended_radial_scalars(r, length_scale, nu)?;

                // --- D₀ ---
                d0[j] = phi;

                // --- D₁ (gradient) ---
                for axis in 0..d {
                    let h_b = ci[axis] - cj[axis];
                    let w_b = metric_weights[axis];
                    d1[[axis, j]] = q * w_b * h_b;
                }

                // --- D₂ (full Hessian, row layout point × axis × axis) ---
                for b in 0..d {
                    let h_b = ci[b] - cj[b];
                    let w_b = metric_weights[b];
                    for c in 0..d {
                        let h_c = ci[c] - cj[c];
                        let w_c = metric_weights[c];
                        let row = b * d + c;
                        d2[[row, j]] = hessian_operator_entry(q, t, h_b, h_c, w_b, w_c, b, c);
                    }
                }

                // --- Per-axis η_a derivatives ---
                for a in 0..dim {
                    let s_a = s_vec[a];

                    // ∂D₀/∂η_a = q · s_a
                    d0_eta[a][j] = q * s_a;
                    // ∂²D₀/∂η_a² = t · s_a² + 2q · s_a
                    d0_eta2[a][j] = t * s_a * s_a + 2.0 * q * s_a;

                    // ∂D₁/∂η_a: for each axis b, ∂(q · h_b)/∂η_a = (dq/dη_a) · h_b = t · s_a · h_b
                    for b in 0..d {
                        let h_b = ci[b] - cj[b];
                        let w_b = metric_weights[b];
                        d1_eta[a][[b, j]] = if a == b {
                            w_b * h_b * (t * s_a + 2.0 * q)
                        } else {
                            w_b * h_b * t * s_a
                        };
                        d1_eta2[a][[b, j]] = if a == b && r > 1e-14 {
                            w_b * h_b * (dt_dr * s_a * s_a / r + 6.0 * t * s_a + 4.0 * q)
                        } else if a == b {
                            0.0
                        } else if r > 1e-14 {
                            w_b * h_b * (dt_dr * s_a * s_a / r + 2.0 * t * s_a)
                        } else {
                            0.0
                        };
                    }

                    for b in 0..d {
                        let h_b = ci[b] - cj[b];
                        let w_b = metric_weights[b];
                        for c in 0..d {
                            let h_c = ci[c] - cj[c];
                            let w_c = metric_weights[c];
                            let row = b * d + c;
                            d2_eta[a][[row, j]] = hessian_operator_eta_entry(
                                q, t, dt_dr, r, s_a, h_b, h_c, w_b, w_c, a, b, c,
                            );
                            d2_eta2[a][[row, j]] = hessian_operator_eta2_entry(
                                q, t, dt_dr, d2t_dr2, r, s_a, h_b, h_c, w_b, w_c, a, b, c,
                            );
                        }
                    }
                }
            }

            Ok(CenterRowAccumulator {
                k,
                d0,
                d1,
                d2,
                d0_eta,
                d1_eta,
                d2_eta,
                d0_eta2,
                d1_eta2,
                d2_eta2,
            })
        })
        .collect::<Result<Vec<_>, BasisError>>()?;

    for row in row_accumulators {
        let k = row.k;
        d0_raw.row_mut(k).assign(&row.d0);
        d1_raw.slice_mut(s![k * d..(k + 1) * d, ..]).assign(&row.d1);
        d2_raw
            .slice_mut(s![k * d * d..(k + 1) * d * d, ..])
            .assign(&row.d2);

        for a in 0..dim {
            d0_raw_eta[a].row_mut(k).assign(&row.d0_eta[a]);
            d1_raw_eta[a]
                .slice_mut(s![k * d..(k + 1) * d, ..])
                .assign(&row.d1_eta[a]);
            d2_raw_eta[a]
                .slice_mut(s![k * d * d..(k + 1) * d * d, ..])
                .assign(&row.d2_eta[a]);
            d0_raw_eta2[a].row_mut(k).assign(&row.d0_eta2[a]);
            d1_raw_eta2[a]
                .slice_mut(s![k * d..(k + 1) * d, ..])
                .assign(&row.d1_eta2[a]);
            d2_raw_eta2[a]
                .slice_mut(s![k * d * d..(k + 1) * d * d, ..])
                .assign(&row.d2_eta2[a]);
        }
    }

    // Project through identifiability transform Z (ψ-independent).
    let project = |mat: Array2<f64>| -> Array2<f64> {
        if let Some(z) = z_opt {
            fast_ab(&mat, z)
        } else {
            mat
        }
    };

    let d0_kernel = project(d0_raw);
    let d1_kernel = project(d1_raw);
    let d2_kernel = project(d2_raw);

    let kernel_cols = d0_kernel.ncols();
    let total_cols = kernel_cols + usize::from(include_intercept);

    // Pad with intercept column.
    let pad = |kernel_mat: Array2<f64>, nrows: usize, add_intercept_ones: bool| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((nrows, total_cols));
        out.slice_mut(s![.., 0..kernel_cols]).assign(&kernel_mat);
        if add_intercept_ones && include_intercept {
            out.column_mut(kernel_cols).fill(1.0);
        }
        out
    };

    let d0 = pad(d0_kernel, p, true);
    let d1 = pad(d1_kernel, p * d, false);
    let d2 = pad(d2_kernel, p * d * d, false);

    // Project and pad all per-axis operator derivative matrices upfront,
    // so they remain available for cross-term computation.
    let d0_eta_all: Vec<Array2<f64>> = d0_raw_eta
        .into_iter()
        .map(|m| pad(project(m), p, false))
        .collect();
    let d1_eta_all: Vec<Array2<f64>> = d1_raw_eta
        .into_iter()
        .map(|m| pad(project(m), p * d, false))
        .collect();
    let d2_eta_all: Vec<Array2<f64>> = d2_raw_eta
        .into_iter()
        .map(|m| pad(project(m), p * d * d, false))
        .collect();
    let d0_eta2_all: Vec<Array2<f64>> = d0_raw_eta2
        .into_iter()
        .map(|m| pad(project(m), p, false))
        .collect();
    let d1_eta2_all: Vec<Array2<f64>> = d1_raw_eta2
        .into_iter()
        .map(|m| pad(project(m), p * d, false))
        .collect();
    let d2_eta2_all: Vec<Array2<f64>> = d2_raw_eta2
        .into_iter()
        .map(|m| pad(project(m), p * d * d, false))
        .collect();

    // Build raw Gram penalties (axis-independent) and their per-axis
    // first/second derivatives + Frobenius norms.
    // We compute these once for axis 0 (the raw Gram S and norm c are the same
    // for all axes) and store them, then reuse c for cross-term normalization.
    struct PerOperatorInfo {
        s_raw: Array2<f64>,
        c: f64,
        s_first: Vec<Array2<f64>>, // per-axis first derivatives (normalized)
        s_second: Vec<Array2<f64>>, // per-axis second derivatives (normalized)
        s_first_raw: Vec<Array2<f64>>, // per-axis first derivatives (raw, for cross normalization)
    }

    let compute_operator_info = |d_op: &Array2<f64>,
                                 d_eta_all: &[Array2<f64>],
                                 d_eta2_all: &[Array2<f64>]|
     -> PerOperatorInfo {
        // Compute the raw Gram and its norm (same for all axes).
        let s_raw = symmetrize(&fast_ata(d_op));
        let fro2: f64 = s_raw.iter().map(|v| v * v).sum();
        let c = fro2.sqrt();

        let mut s_first = Vec::with_capacity(dim);
        let mut s_second = Vec::with_capacity(dim);
        let mut s_first_raw = Vec::with_capacity(dim);
        for a in 0..dim {
            let (_, sa, sa2) =
                gram_and_psi_derivatives_from_operator(d_op, &d_eta_all[a], &d_eta2_all[a]);
            let (_, sa_norm, sa2_norm, _) =
                normalize_penaltywith_psi_derivatives(&s_raw, &sa, &sa2);
            s_first_raw.push(sa);
            s_first.push(sa_norm);
            s_second.push(sa2_norm);
        }

        PerOperatorInfo {
            s_raw,
            c,
            s_first,
            s_second,
            s_first_raw,
        }
    };

    let op0_info = compute_operator_info(&d0, &d0_eta_all, &d0_eta2_all);
    let op1_info = compute_operator_info(&d1, &d1_eta_all, &d1_eta2_all);
    let op2_info = compute_operator_info(&d2, &d2_eta_all, &d2_eta2_all);

    // Build penalty candidates and determine which are active (using axis-0
    // normalized Gram, which is axis-independent).
    let (s0_norm, c0) = if op0_info.c > 1e-12 {
        (op0_info.s_raw.mapv(|v| v / op0_info.c), op0_info.c)
    } else {
        (op0_info.s_raw.clone(), 1.0)
    };
    let (s1_norm, c1) = if op1_info.c > 1e-12 {
        (op1_info.s_raw.mapv(|v| v / op1_info.c), op1_info.c)
    } else {
        (op1_info.s_raw.clone(), 1.0)
    };
    let (s2_norm, c2) = if op2_info.c > 1e-12 {
        (op2_info.s_raw.mapv(|v| v / op2_info.c), op2_info.c)
    } else {
        (op2_info.s_raw.clone(), 1.0)
    };

    let candidates = vec![
        PenaltyCandidate {
            matrix: s0_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorMass,
            normalization_scale: c0,
            kronecker_factors: None,
            op: None,
        },
        PenaltyCandidate {
            matrix: s1_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorTension,
            normalization_scale: c1,
            kronecker_factors: None,
            op: None,
        },
        PenaltyCandidate {
            matrix: s2_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorStiffness,
            normalization_scale: c2,
            kronecker_factors: None,
            op: None,
        },
    ];
    let (_, _, penaltyinfo) = filter_active_penalty_candidates(candidates)?;

    // Build per-axis results.
    let mut per_axis_results = Vec::with_capacity(dim);
    for a in 0..dim {
        let pen_first = active_operator_penalty_derivatives(
            &penaltyinfo,
            &[
                op0_info.s_first[a].clone(),
                op1_info.s_first[a].clone(),
                op2_info.s_first[a].clone(),
            ],
            "Matérn-aniso",
        )?;
        let pen_second = active_operator_penalty_derivatives(
            &penaltyinfo,
            &[
                op0_info.s_second[a].clone(),
                op1_info.s_second[a].clone(),
                op2_info.s_second[a].clone(),
            ],
            "Matérn-aniso",
        )?;
        per_axis_results.push((pen_first, pen_second));
    }

    let cross_ctx = std::sync::Arc::new(MaternCrossPenaltyContext {
        centers: centers.to_owned(),
        aniso_log_scales: eta.to_vec(),
        length_scale,
        nu,
        z_transform: z_opt.cloned(),
        penaltyinfo,
        d0,
        d1,
        d2,
        d0_eta_proj: d0_eta_all,
        d1_eta_proj: d1_eta_all,
        d2_eta_proj: d2_eta_all,
        op0_s_raw: op0_info.s_raw,
        op1_s_raw: op1_info.s_raw,
        op2_s_raw: op2_info.s_raw,
        op0_c: op0_info.c,
        op1_c: op1_info.c,
        op2_c: op2_info.c,
        op0_s_first_raw: op0_info.s_first_raw,
        op1_s_first_raw: op1_info.s_first_raw,
        op2_s_first_raw: op2_info.s_first_raw,
    });
    let cross_provider = AnisoPenaltyCrossProvider::new(move |a: usize, b: usize| {
        let (axis_a, axis_b) = if a < b { (a, b) } else { (b, a) };
        if axis_a == axis_b || axis_b >= cross_ctx.d0_eta_proj.len() {
            return Ok(Vec::new());
        }
        cross_ctx.compute_pair(axis_a, axis_b)
    });

    Ok((per_axis_results, cross_pairs, cross_provider))
}


/// Build exact per-axis η_a derivatives of operator penalty matrices for
/// anisotropic hybrid Duchon terms.
///
/// Analogous to [`build_matern_operator_penalty_aniso_derivatives`] but for
/// the Duchon kernel. Uses `duchon_radial_jets` for the full radial jet
/// `(φ, q, t, t_r, t_rr)`.
///
/// The local y-space operator shape derivatives start from the same formulas as
/// Matérn, but the raw per-axis `psi_a` coordinates also inherit the Duchon
/// isotropic scaling law. After assembling the shape-only pieces, this routine
/// adds the exact raw-`psi` isotropic-share correction implied by
/// `phi(r; kappa) = kappa^delta H(kappa r)`.
fn duchon_kernel_radial_triplet(
    r: f64,
    length_scale: Option<f64>,
    p_order: usize,
    s_order: f64,
    k_dim: usize,
    coeffs: Option<&DuchonPartialFractionCoeffs>,
) -> Result<(f64, f64, f64), BasisError> {
    // Public Duchon (φ, φ_r, φ_rr) triplet.
    //
    // The Duchon spectral kernel is F(ρ) = 1 / [ρ^(2p)·(κ²+ρ²)^s]. The pure
    // case (κ=0) is the κ→0 limit and collapses to a single polyharmonic of
    // order m = p+s — value and both radial derivatives all come from one
    // normalization in `polyharmonic_block_jet4`. The hybrid case (κ>0) is the
    // partial-fraction sum; we route it through `duchon_radial_jets` so the
    // public triplet shares the same Taylor / collision tiering used by the
    // operator scalars (q, lap, t) in the penalty code.
    let triplet = match length_scale {
        None => {
            // Keep the block order in `f64`: fractional `s_order` rides
            // through `pure_duchon_block_order` → `polyharmonic_kernel_triplet`
            // → `polyharmonic_block_jet4` end-to-end. Truncating to
            // `usize` here was the bug — for `s=1.5, p=2, d=4` it
            // collapsed `m=3.5` to `m=3` and tripped the integer-only
            // log-case branch at `m=d/2`, producing NaN at `r=0`.
            let m = pure_duchon_block_order(p_order, s_order);
            polyharmonic_kernel_triplet(r, m, k_dim)?
        }
        Some(length_scale) => {
            if !length_scale.is_finite() || length_scale <= 0.0 {
                crate::bail_invalid_basis!(
                    "Duchon hybrid length_scale must be finite and positive"
                );
            }
            let kappa = 1.0 / length_scale.max(1e-300);
            let coeffs_local;
            let coeffs_ref = match coeffs {
                Some(c) => c,
                None => {
                    coeffs_local = duchon_partial_fraction_coeffs(p_order, s_order as usize, kappa);
                    &coeffs_local
                }
            };
            let jets = duchon_radial_jets(
                r,
                length_scale,
                p_order,
                s_order as usize,
                k_dim,
                coeffs_ref,
            )?;
            (jets.phi, jets.phi_r, jets.phi_rr)
        }
    };

    if !triplet.0.is_finite() || !triplet.1.is_finite() || !triplet.2.is_finite() {
        crate::bail_invalid_basis!(
            "non-finite Duchon radial triplet at r={r}, length_scale={length_scale:?}, p={p_order}, s={s_order}, dim={k_dim}"
        );
    }
    Ok(triplet)
}


#[inline(always)]
fn lower_triangular_offset(row: usize) -> usize {
    if row & 1 == 0 {
        (row / 2)
            .checked_mul(row + 1)
            .expect("lower-triangular row offset overflow")
    } else {
        row.checked_mul(row / 2 + 1)
            .expect("lower-triangular row offset overflow")
    }
}


fn lower_triangular_len(k: usize) -> usize {
    if k & 1 == 0 {
        (k / 2)
            .checked_mul(k.checked_add(1).expect("lower-triangular length overflow"))
            .expect("lower-triangular length overflow")
    } else {
        k.checked_mul(k / 2 + 1)
            .expect("lower-triangular length overflow")
    }
}


fn symmetric_matrix_from_lower_values(k: usize, values: &[f64]) -> Array2<f64> {
    assert_eq!(values.len(), lower_triangular_len(k));
    let mut g = Array2::<f64>::zeros((k, k));
    let mut idx = 0usize;
    for i in 0..k {
        for j in 0..=i {
            let v = values[idx];
            g[[i, j]] = v;
            if i != j {
                g[[j, i]] = v;
            }
            idx += 1;
        }
    }
    g
}


fn transform_closed_form_raw_block(
    raw: &Array2<f64>,
    kernel_nullspace: Option<&Array2<f64>>,
    polynomial_block_cols: usize,
    outer_identifiability: Option<&Array2<f64>>,
) -> Array2<f64> {
    let kernel_block = if let Some(z) = kernel_nullspace {
        let zt = fast_atb(z, raw);
        fast_ab(&zt, z)
    } else {
        raw.clone()
    };
    let kernel_cols = kernel_block.nrows();
    let total_pre = kernel_cols + polynomial_block_cols;
    let padded = if polynomial_block_cols == 0 {
        kernel_block
    } else {
        let mut padded = Array2::<f64>::zeros((total_pre, total_pre));
        padded
            .slice_mut(s![0..kernel_cols, 0..kernel_cols])
            .assign(&kernel_block);
        padded
    };
    let total = if let Some(t) = outer_identifiability {
        let tt = fast_atb(t, &padded);
        fast_ab(&tt, t)
    } else {
        padded
    };
    symmetrize(&total)
}


fn symmetrize(matrix: &Array2<f64>) -> Array2<f64> {
    (matrix + &matrix.t().to_owned()) * 0.5
}


/// Centered design Gram: `(D − 1 μ')^T (D − 1 μ')` where `μ_j` is the
/// column mean of `D` across rows. The constant direction (intercept basis
/// = column of ones) sits in the exact null space because its centered
/// column is identically zero. Used as the spring-measure mass penalty on
/// the scale-free Duchon path so the magnitude term penalizes deviations
/// from the function's row-mean rather than absolute level — the intercept
/// is genuinely unpenalized regardless of what the row-mean is.
fn centered_design_gram(d: &Array2<f64>) -> Array2<f64> {
    let n_rows = d.nrows();
    let n_cols = d.ncols();
    if n_rows == 0 || n_cols == 0 {
        return Array2::<f64>::zeros((n_cols, n_cols));
    }
    let inv_n = 1.0 / n_rows as f64;
    let col_sum = d.sum_axis(Axis(0));
    let g_raw = fast_ata(d);
    // (D − 1μ')'(D − 1μ') = D'D − N μ μ'   where Σ = D'1 = col_sum,
    // so the rank-1 correction is `col_sum col_sum' / N`.
    let mut out = g_raw;
    for i in 0..n_cols {
        let ci = col_sum[i];
        let row = out.row_mut(i);
        // Subtract column i's contribution: out[i, j] -= (ci * col_sum[j]) / N.
        let mut row = row;
        for j in 0..n_cols {
            row[j] -= ci * col_sum[j] * inv_n;
        }
    }
    out
}


fn centered_operator_gram_and_psi_derivatives(
    d: &Array2<f64>,
    d_psi: &Array2<f64>,
    d_psi_psi: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let center_columns = |mat: &Array2<f64>| {
        let n_rows = mat.nrows();
        if n_rows == 0 || mat.ncols() == 0 {
            return mat.clone();
        }
        let means = mat.sum_axis(Axis(0)).mapv(|v| v / n_rows as f64);
        let mut centered = mat.clone();
        for mut row in centered.rows_mut() {
            row -= &means;
        }
        centered
    };
    let d_centered = center_columns(d);
    let d_psi_centered = center_columns(d_psi);
    let d_psi_psi_centered = center_columns(d_psi_psi);
    gram_and_psi_derivatives_from_operator(&d_centered, &d_psi_centered, &d_psi_psi_centered)
}


fn normalize_penalty(matrix: &Array2<f64>) -> (Array2<f64>, f64) {
    let norm = matrix.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    (matrix.mapv(|v| v / norm), norm)
}


pub(crate) fn closed_form_anisotropic_pair_value_with_powers(
    q: usize,
    m: usize,
    s: usize,
    kappa: f64,
    eta_raw: &[f64],
    powers: &closed_form_penalty::AnisoMetricPowers,
    r: &[f64],
    diagonal_epsilon: f64,
) -> f64 {
    assert_eq!(
        eta_raw.len(),
        r.len(),
        "closed_form_anisotropic_pair_value_with_powers: eta/r dimension mismatch"
    );
    let j_prefactor = eta_raw.iter().sum::<f64>().exp();
    if r.iter().all(|&value| value == 0.0) {
        // Exact distributional diagonal first. In the convergent spectral
        // strip the radial pointwise chain is singular at R=0, but the
        // self-pair integral is finite and has a Gamma/Beta closed form.
        // Outside that strip, odd-d hybrid Taylor covers the smooth pointwise
        // diagonal; epsilon regularization is only the final non-convergent
        // diagonal convention.
        if let Some(bundle) =
            closed_form_penalty::analytic_self_pair_bundle(q, m, s, kappa, eta_raw)
        {
            return bundle.value;
        }
        let mut r_eps_buf = vec![0.0_f64; r.len()];
        if !r_eps_buf.is_empty() {
            r_eps_buf[0] = diagonal_epsilon * eta_raw[0].exp();
        }
        return j_prefactor
            * closed_form_penalty::anisotropic_duchon_penalty_radial_with_powers(
                q, m, s as f64, kappa, eta_raw, powers, &r_eps_buf,
            );
    }

    j_prefactor
        * closed_form_penalty::anisotropic_duchon_penalty_radial_with_powers(
            q, m, s as f64, kappa, eta_raw, powers, r,
        )
}


pub fn closed_form_anisotropic_pair_block(
    centers: ArrayView2<'_, f64>,
    q: usize,
    m: usize,
    s: usize,
    kappa: f64,
    aniso_log_scales: Option<&[f64]>,
) -> Array2<f64> {
    // Math team Letter A §9: G_q(η_raw, κ) ≠ G_q(η_centered, κ) in general;
    // the relation involves an exp((2d-4m-4s)μ) prefactor and a κ rescaling
    // that don't reduce to a uniform Jacobian. Use raw η directly so the
    // pair-block matches the closed-form Lebesgue penalty's natural raw-η
    // parameterization, and so its η-derivatives (computed elsewhere via
    // `pair_block_radial_with_j_second_derivatives` with raw η) are already
    // ∂G_q/∂η_raw without any chain-rule conversion.
    let k = centers.nrows();
    let d = centers.ncols();
    let zeros: Vec<f64>;
    let eta_raw: &[f64] = match aniso_log_scales {
        Some(eta) => eta,
        None => {
            zeros = vec![0.0_f64; d];
            &zeros
        }
    };
    let r_eps = if closed_form_penalty::analytic_self_pair_bundle(q, m, s, kappa, eta_raw).is_some()
    {
        0.0
    } else {
        pure_duchon_diagonal_epsilon(centers, eta_raw)
    };
    let powers = closed_form_penalty::AnisoMetricPowers::new(eta_raw);

    // Parallelize by independent lower-triangular rows. This keeps one lag
    // scratch buffer per worker row, avoids the sqrt-based flat-index decode
    // in the hot loop, and still evaluates each symmetric pair only once.
    let n_pairs = lower_triangular_len(k);
    let mut values = vec![0.0_f64; n_pairs];
    let values_ptr = SendPtr(values.as_mut_ptr());
    (0..k).into_par_iter().for_each(|i| {
        let row_ptr = values_ptr.add(lower_triangular_offset(i));
        let mut r_buf: SmallVec<[f64; 16]> = SmallVec::with_capacity(d);
        r_buf.resize(d, 0.0);
        for j in 0..=i {
            for axis in 0..d {
                r_buf[axis] = centers[[i, axis]] - centers[[j, axis]];
            }
            let value = closed_form_anisotropic_pair_value_with_powers(
                q, m, s, kappa, eta_raw, &powers, &r_buf, r_eps,
            );
            // SAFETY: values has k(k+1)/2 slots; for i in 0..k and j ∈ 0..=i,
            // lower_triangular_offset(i)+j is in bounds. Each rayon iteration
            // owns a distinct lower-triangular row, so writes are disjoint.
            unsafe {
                *row_ptr.add(j) = value;
            }
        }
    });

    symmetric_matrix_from_lower_values(k, &values)
}


/// Pure-Duchon (κ=0) variant of [`closed_form_anisotropic_pair_block`].
///
/// Uses the analytic radial-derivative path
/// [`closed_form_penalty::anisotropic_duchon_penalty_radial`] which handles
/// κ=0 cleanly by delegating to pure-Riesz radial derivatives. The
/// Schoenberg path is undefined at κ=0 in low dimensions, so this variant
/// must be used in place of [`closed_form_anisotropic_pair_block`] for
/// `length_scale = None` (pure-Duchon) penalty assembly.
///
/// Self-pairs (R=0) are ε-regularized using a small fraction of the median
/// off-diagonal lag, since the radial form is singular at R=0 and the pure
/// Riesz Schoenberg fallback also doesn't converge for κ=0.
pub fn closed_form_anisotropic_pair_block_pure(
    centers: ArrayView2<'_, f64>,
    q: usize,
    m: usize,
    s: f64,
    aniso_log_scales: Option<&[f64]>,
) -> Array2<f64> {
    let k = centers.nrows();
    let d = centers.ncols();
    let eta_centered: Vec<f64> = if let Some(eta) = aniso_log_scales {
        let mean = centered_aniso_log_scale_mean(eta);
        eta.iter()
            .map(|&v| centered_aniso_log_scale(v, mean))
            .collect()
    } else {
        vec![0.0_f64; d]
    };
    let j_prefactor = eta_centered.iter().sum::<f64>().exp();

    // Median off-diagonal anisotropic distance is needed only when the
    // exact pure-Duchon finite self-pair is unavailable. The integer-only
    // self-pair helper is consulted only when `s` is whole-valued;
    // fractional `s` always falls through to the analytic radial chain
    // below, which now accepts `f64` via the threaded
    // `radial_derivatives_of_isotropic_duchon` cascade.
    let s_int = if s.fract() == 0.0 && s >= 0.0 {
        Some(s as usize)
    } else {
        None
    };
    let pure_diag_exact = s_int
        .and_then(|si| closed_form_penalty::pure_duchon_self_pair_value(q, d, m, si, &eta_centered))
        .is_some();
    let r_eps = if pure_diag_exact {
        0.0
    } else {
        pure_duchon_diagonal_epsilon(centers, &eta_centered)
    };
    let powers = closed_form_penalty::AnisoMetricPowers::new(&eta_centered);

    // Parallelize by independent lower-triangular rows and evaluate each
    // symmetric pair once while reusing a single lag scratch buffer per row.
    let n_pairs = lower_triangular_len(k);
    let eta_slice: &[f64] = eta_centered.as_slice();
    let mut values = vec![0.0_f64; n_pairs];
    let values_ptr = SendPtr(values.as_mut_ptr());
    (0..k).into_par_iter().for_each(|i| {
        let row_ptr = values_ptr.add(lower_triangular_offset(i));
        let mut r_buf: SmallVec<[f64; 16]> = SmallVec::with_capacity(d);
        r_buf.resize(d, 0.0);
        for j in 0..=i {
            for axis in 0..d {
                r_buf[axis] = centers[[i, axis]] - centers[[j, axis]];
            }
            let value = if i == j {
                // Self-pair (R = 0). Prefer the exact finite-part limit
                // when available (integer s only); otherwise use the same
                // ε-regularized convention via the analytic radial chain
                // (which now accepts fractional s end-to-end).
                let closed_self = s_int.and_then(|si| {
                    closed_form_penalty::pure_duchon_self_pair_value(q, d, m, si, eta_slice)
                });
                if let Some(closed) = closed_self {
                    j_prefactor * closed
                } else {
                    let mut r_eps_buf: SmallVec<[f64; 16]> = SmallVec::with_capacity(d);
                    r_eps_buf.resize(d, 0.0);
                    if d > 0 {
                        r_eps_buf[0] = r_eps * eta_slice[0].exp();
                    }
                    j_prefactor
                        * closed_form_penalty::anisotropic_duchon_penalty_radial_with_powers(
                            q, m, s, 0.0, eta_slice, &powers, &r_eps_buf,
                        )
                }
            } else {
                j_prefactor
                    * closed_form_penalty::anisotropic_duchon_penalty_radial_with_powers(
                        q, m, s, 0.0, eta_slice, &powers, &r_buf,
                    )
            };
            // SAFETY: values has k(k+1)/2 slots; for i in 0..k and j ∈ 0..=i,
            // lower_triangular_offset(i)+j is in bounds. Each rayon iteration
            // owns a distinct lower-triangular row, so writes are disjoint.
            unsafe {
                *row_ptr.add(j) = value;
            }
        }
    });

    symmetric_matrix_from_lower_values(k, &values)
}


/// Closed-form pair-block penalty for the pure Matérn basis at operator order
/// `q ∈ {0, 1, 2}`.
///
/// Spectral form: the Matérn kernel has Fourier symbol `K̂(ρ) = c · (κ² + ρ²)^{-ℓ}`
/// with `ℓ = ν + d/2`. For half-integer `MaternNu`, `2ℓ = 2ν + d` is always
/// a positive integer (even when ℓ itself is half-integer in even d). The
/// pair-block penalty is
///
///   `g_q(R; κ) = F^{-1}{|ρ|^{2q} / (κ² + ρ²)^{2ℓ}}(R)`.
///
/// Using the binomial expansion `|ρ|^{2q} = ((κ² + ρ²) − κ²)^q`:
///
///   `g_q(R; κ) = Σ_{j=0}^{q} C(q,j) (−κ²)^{q−j} · M_{2ℓ−j}^d(R; κ)`,
///
/// where each `M_ℓ'^d(R; κ)` is supplied by
/// [`closed_form_penalty::matern_kernel_value`]. Convergence (finite block at
/// R = 0) requires `4ℓ − 2q > d`, i.e. the Sobolev order strictly exceeds
/// the operator order plus dimension/2; otherwise the spectral integrand is
/// not integrable and the resulting matrix is not PSD-by-construction.
///
/// Length scale enters as `κ = √(2ν) / length_scale` (the standard Matérn
/// parameterization that makes `length_scale` the practical correlation
/// scale). Anisotropy is handled via `aniso_log_scales` by rescaling lags:
/// `r_axis ← r_axis · exp(η_axis)`. The spectral form is invariant under
/// this rescaling (Schoenberg) so the penalty matrix remains PSD when the
/// gate accepts.
///
/// Returns `None` when `q > 2` or when the spectral integral diverges
/// (`4ℓ ≤ 2q + d`); the caller should fall back to the collocation
/// `D_q^T D_q` path in those regimes.
pub fn closed_form_matern_pair_block(
    centers: ArrayView2<'_, f64>,
    q: usize,
    length_scale: f64,
    nu: MaternNu,
    aniso_log_scales: Option<&[f64]>,
) -> Option<Array2<f64>> {
    assert!(
        length_scale.is_finite() && length_scale > 0.0,
        "closed_form_matern_pair_block: length_scale must be finite and positive"
    );
    if q > 2 {
        return None;
    }
    let k = centers.nrows();
    let d = centers.ncols();
    if d == 0 || k == 0 {
        return Some(Array2::<f64>::zeros((k, k)));
    }

    // Convert MaternNu (half-integer ν) to integer 2ν, then 2ℓ = 2ν + d.
    let two_nu: usize = match nu {
        MaternNu::Half => 1,
        MaternNu::ThreeHalves => 3,
        MaternNu::FiveHalves => 5,
        MaternNu::SevenHalves => 7,
        MaternNu::NineHalves => 9,
    };
    let two_ell = two_nu + d;

    // IR convergence: 2·(2ℓ) > 2q + d.
    if 2 * two_ell <= 2 * q + d {
        return None;
    }
    // Each building block requires `2ℓ - j ≥ 1` for j ∈ [0, q].
    if two_ell < q + 1 {
        return None;
    }

    // Standard Matérn parameterization: κ = √(2ν) / length_scale.
    let kappa = (two_nu as f64).sqrt() / length_scale;
    let kappa_sq = kappa * kappa;

    // Per-axis multiplicative scale for anisotropic lags.
    let scale_per_axis: Option<Vec<f64>> = aniso_log_scales.map(|eta| {
        assert_eq!(
            eta.len(),
            d,
            "closed_form_matern_pair_block: aniso_log_scales length must match d"
        );
        eta.iter().map(|v| v.exp()).collect()
    });

    // Coefficients C(q, j) · (−κ²)^{q−j} for j = 0..q.
    let mut binom_coeffs: Vec<f64> = Vec::with_capacity(q + 1);
    for j in 0..=q {
        let cqj = crate::probability::binomial_coefficient_f64(q, j);
        let sign_pow = if (q - j).is_multiple_of(2) { 1.0 } else { -1.0 };
        let coeff = cqj * sign_pow * kappa_sq.powi((q - j) as i32);
        binom_coeffs.push(coeff);
    }

    let n_pairs = lower_triangular_len(k);
    let mut values = vec![0.0_f64; n_pairs];
    let values_ptr = SendPtr(values.as_mut_ptr());
    (0..k).into_par_iter().for_each(|i| {
        let row_ptr = values_ptr.add(lower_triangular_offset(i));
        for j in 0..=i {
            // Anisotropic distance: r_eff² = Σ_a (Δ_a · exp(η_a))².
            let mut r2 = 0.0_f64;
            for axis in 0..d {
                let delta = centers[[i, axis]] - centers[[j, axis]];
                let scaled = if let Some(sc) = &scale_per_axis {
                    delta * sc[axis]
                } else {
                    delta
                };
                r2 += scaled * scaled;
            }
            let r = r2.sqrt();

            let mut acc = 0.0_f64;
            for jj in 0..=q {
                let order = two_ell - jj; // ≥ 1 by the gate
                acc +=
                    binom_coeffs[jj] * closed_form_penalty::matern_kernel_value(d, order, kappa, r);
            }
            // SAFETY: values has k(k+1)/2 slots; for i in 0..k and j ∈ 0..=i,
            // lower_triangular_offset(i)+j is in bounds. Each rayon iteration
            // owns a distinct lower-triangular row, so writes are disjoint.
            unsafe {
                *row_ptr.add(j) = acc;
            }
        }
    });
    Some(symmetric_matrix_from_lower_values(k, &values))
}


/// Median off-diagonal anisotropic lag scaled by 1e-6, used for
/// regularizing self-pair R=0 evaluations in pure-Duchon (κ=0) closed-form
/// penalties. Matches the magnitude used by hybrid κ>0 collocation builders
/// where the ε-regularization is implicit in the Matérn kernel finiteness.
pub(crate) fn pure_duchon_diagonal_epsilon(
    centers: ArrayView2<'_, f64>,
    eta_log_scales: &[f64],
) -> f64 {
    let k = centers.nrows();
    let d = centers.ncols();
    if k <= 1 || d == 0 {
        return 1e-12;
    }
    let mut lags = Vec::with_capacity(k * (k - 1) / 2);
    for i in 0..k {
        for j in 0..i {
            let mut acc = 0.0_f64;
            for axis in 0..d {
                let delta = centers[[i, axis]] - centers[[j, axis]];
                let b = (-2.0 * eta_log_scales[axis]).exp();
                acc += b * delta * delta;
            }
            let r = acc.sqrt();
            if r > 0.0 {
                lags.push(r);
            }
        }
    }
    if lags.is_empty() {
        return 1e-12;
    }
    lags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = lags[lags.len() / 2];
    (median * 1e-6).max(1e-12)
}


pub fn closed_form_operator_penalty_in_total_basis(
    centers: ArrayView2<'_, f64>,
    q: usize,
    p_order: usize,
    s_order: usize,
    kappa: f64,
    aniso_log_scales: Option<&[f64]>,
    kernel_nullspace: Option<&Array2<f64>>,
    polynomial_block_cols: usize,
    outer_identifiability: Option<&Array2<f64>>,
) -> Array2<f64> {
    // 1. Closed-form penalty in raw kernel basis (K×K).
    let g_raw =
        closed_form_anisotropic_pair_block(centers, q, p_order, s_order, kappa, aniso_log_scales);
    // 2. Apply kernel-constraint nullspace transform Z (K×kernel_cols).
    let g_kernel = if let Some(z) = kernel_nullspace {
        let zt_g = fast_atb(z, &g_raw);
        fast_ab(&zt_g, z)
    } else {
        g_raw
    };
    // 3. Block-diag pad polynomial nullspace (zero penalty in continuous form).
    let kernel_cols = g_kernel.nrows();
    let total_pre_cols = kernel_cols + polynomial_block_cols;
    let g_padded = if polynomial_block_cols == 0 {
        g_kernel
    } else {
        let mut padded = Array2::<f64>::zeros((total_pre_cols, total_pre_cols));
        padded
            .slice_mut(s![0..kernel_cols, 0..kernel_cols])
            .assign(&g_kernel);
        padded
    };
    // 4. Apply outer spatial identifiability transform if any.
    let g_total = if let Some(t) = outer_identifiability {
        let tt_g = fast_atb(t, &g_padded);
        fast_ab(&tt_g, t)
    } else {
        g_padded
    };
    symmetrize(&g_total)
}


/// Closed-form penalty value `S_q` and its log-κ derivatives `S_q_psi`,
/// `S_q_psi_psi` in the final (post-transform) basis space, computed via
/// `pair_block_radial_with_j_second_derivatives` from the closed_form_penalty
/// module. Bundle's `d_kappa` and `d2_kappa` are κ-derivatives; chain rule
/// converts to log-κ: `∂/∂ψ = κ·∂/∂κ`,
/// `∂²/∂ψ² = κ²·∂²/∂κ² + κ·∂/∂κ`.
///
/// All three matrices share the same Z + poly-pad + outer-T transform pipeline
/// as `closed_form_operator_penalty_in_total_basis` (the transforms are
/// linear and commute with parameter differentiation).
pub fn closed_form_psi_derivatives_in_total_basis(
    centers: ArrayView2<'_, f64>,
    q: usize,
    p_order: usize,
    s_order: usize,
    kappa: f64,
    aniso_log_scales: Option<&[f64]>,
    kernel_nullspace: Option<&Array2<f64>>,
    polynomial_block_cols: usize,
    outer_identifiability: Option<&Array2<f64>>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let k = centers.nrows();
    let d = centers.ncols();
    let zeros: Vec<f64>;
    let eta_raw: &[f64] = match aniso_log_scales {
        Some(eta) => eta,
        None => {
            zeros = vec![0.0_f64; d];
            &zeros
        }
    };
    let r_eps =
        if closed_form_penalty::analytic_self_pair_bundle(q, p_order, s_order, kappa, eta_raw)
            .is_some()
        {
            0.0
        } else {
            pure_duchon_diagonal_epsilon(centers, eta_raw)
        };
    let powers = closed_form_penalty::AnisoMetricPowers::new(eta_raw);

    let n_pairs = lower_triangular_len(k);
    let mut g_values = vec![0.0_f64; n_pairs];
    let mut g_psi_values = vec![0.0_f64; n_pairs];
    let mut g_psi_psi_values = vec![0.0_f64; n_pairs];
    let g_ptr = SendPtr(g_values.as_mut_ptr());
    let g_psi_ptr = SendPtr(g_psi_values.as_mut_ptr());
    let g_psi_psi_ptr = SendPtr(g_psi_psi_values.as_mut_ptr());
    (0..k).into_par_iter().for_each(|i| {
        let row_offset = lower_triangular_offset(i);
        let g_row = g_ptr.add(row_offset);
        let g_psi_row = g_psi_ptr.add(row_offset);
        let g_psi_psi_row = g_psi_psi_ptr.add(row_offset);
        let mut r_buf: SmallVec<[f64; 16]> = SmallVec::with_capacity(d);
        r_buf.resize(d, 0.0);
        let mut r_eps_buf: SmallVec<[f64; 16]> = SmallVec::with_capacity(d);
        r_eps_buf.resize(d, 0.0);
        if d > 0 {
            r_eps_buf[0] = r_eps * eta_raw[0].exp();
        }
        for j in 0..=i {
            for axis in 0..d {
                r_buf[axis] = centers[[i, axis]] - centers[[j, axis]];
            }
            let bundle = if i == j {
                closed_form_penalty::analytic_self_pair_bundle(q, p_order, s_order, kappa, eta_raw)
                    .unwrap_or_else(|| {
                        closed_form_penalty::pair_block_radial_with_j_second_derivatives_with_powers(
                            q, p_order, s_order, kappa, eta_raw, &powers, &r_eps_buf,
                        )
                    })
            } else {
                closed_form_penalty::pair_block_radial_with_j_second_derivatives_with_powers(
                    q, p_order, s_order, kappa, eta_raw, &powers, &r_buf,
                )
            };
            // SAFETY: each output has k(k+1)/2 slots; for i in 0..k and j ∈ 0..=i,
            // lower_triangular_offset(i)+j is in bounds. Each rayon iteration
            // owns that lower-triangular row in every output, so writes are disjoint.
            unsafe {
                *g_row.add(j) = bundle.value;
                *g_psi_row.add(j) = kappa * bundle.d_kappa;
                *g_psi_psi_row.add(j) = kappa * kappa * bundle.d2_kappa + kappa * bundle.d_kappa;
            }
        }
    });
    let g = symmetric_matrix_from_lower_values(k, &g_values);
    let g_psi = symmetric_matrix_from_lower_values(k, &g_psi_values);
    let g_psi_psi = symmetric_matrix_from_lower_values(k, &g_psi_psi_values);

    // Apply Z + poly-pad + T to each of g, g_psi, g_psi_psi identically.
    (
        transform_closed_form_raw_block(
            &g,
            kernel_nullspace,
            polynomial_block_cols,
            outer_identifiability,
        ),
        transform_closed_form_raw_block(
            &g_psi,
            kernel_nullspace,
            polynomial_block_cols,
            outer_identifiability,
        ),
        transform_closed_form_raw_block(
            &g_psi_psi,
            kernel_nullspace,
            polynomial_block_cols,
            outer_identifiability,
        ),
    )
}


/// Closed-form anisotropic penalty `S_q` and its raw-η derivatives — full
/// d×d Hessian materialized — in the final (post-transform) basis space.
/// Returns `(S_q, S_q_eta_a per axis, S_q_eta_a_a per axis, S_q_eta_a_b for
/// (a, b) with a < b)`. All derivatives are with respect to raw η components
/// directly per math team Letter A §9 — no centering, no apply_raw_psi_scaling.
///
/// Bundle is computed via `pair_block_radial_with_j_second_derivatives`,
/// which uses the radial analytic derivative chain in regular regimes and
/// the Schoenberg derivative bundle in convergent singular/log-Riesz regimes.
pub fn closed_form_aniso_psi_derivatives_in_total_basis(
    centers: ArrayView2<'_, f64>,
    q: usize,
    p_order: usize,
    s_order: usize,
    kappa: f64,
    aniso_log_scales: Option<&[f64]>,
    kernel_nullspace: Option<&Array2<f64>>,
    polynomial_block_cols: usize,
    outer_identifiability: Option<&Array2<f64>>,
) -> (
    Array2<f64>,
    Vec<Array2<f64>>,
    Vec<Array2<f64>>,
    Vec<Vec<Array2<f64>>>,
) {
    let k = centers.nrows();
    let d = centers.ncols();
    let zeros: Vec<f64>;
    let eta_raw: &[f64] = match aniso_log_scales {
        Some(eta) => eta,
        None => {
            zeros = vec![0.0_f64; d];
            &zeros
        }
    };
    let r_eps =
        if closed_form_penalty::analytic_self_pair_bundle(q, p_order, s_order, kappa, eta_raw)
            .is_some()
        {
            0.0
        } else {
            pure_duchon_diagonal_epsilon(centers, eta_raw)
        };
    let powers = closed_form_penalty::AnisoMetricPowers::new(eta_raw);

    let cross_pairs: Vec<(usize, usize)> =
        (0..d).flat_map(|a| (a..d).map(move |b| (a, b))).collect();
    let n_pairs = lower_triangular_len(k);
    let mut g_values = vec![0.0_f64; n_pairs];
    let mut g_eta_values: Vec<Vec<f64>> = (0..d).map(|_| vec![0.0_f64; n_pairs]).collect();
    let mut g_eta2_diag_values: Vec<Vec<f64>> = (0..d).map(|_| vec![0.0_f64; n_pairs]).collect();
    let mut g_eta2_cross_values: Vec<Vec<f64>> =
        cross_pairs.iter().map(|_| vec![0.0_f64; n_pairs]).collect();

    let g_ptr = SendPtr(g_values.as_mut_ptr());
    let g_eta_ptrs: Vec<SendPtr> = g_eta_values
        .iter_mut()
        .map(|values| SendPtr(values.as_mut_ptr()))
        .collect();
    let g_eta2_diag_ptrs: Vec<SendPtr> = g_eta2_diag_values
        .iter_mut()
        .map(|values| SendPtr(values.as_mut_ptr()))
        .collect();
    let g_eta2_cross_ptrs: Vec<SendPtr> = g_eta2_cross_values
        .iter_mut()
        .map(|values| SendPtr(values.as_mut_ptr()))
        .collect();

    (0..k).into_par_iter().for_each(|i| {
        let row_offset = lower_triangular_offset(i);
        let g_row = g_ptr.add(row_offset);
        let g_eta_rows: Vec<*mut f64> = g_eta_ptrs
            .iter()
            .map(|ptr| ptr.add(row_offset))
            .collect();
        let g_eta2_diag_rows: Vec<*mut f64> = g_eta2_diag_ptrs
            .iter()
            .map(|ptr| ptr.add(row_offset))
            .collect();
        let g_eta2_cross_rows: Vec<*mut f64> = g_eta2_cross_ptrs
            .iter()
            .map(|ptr| ptr.add(row_offset))
            .collect();
        let mut r_buf: SmallVec<[f64; 16]> = SmallVec::with_capacity(d);
        r_buf.resize(d, 0.0);
        let mut r_eps_buf: SmallVec<[f64; 16]> = SmallVec::with_capacity(d);
        r_eps_buf.resize(d, 0.0);
        if d > 0 {
            r_eps_buf[0] = r_eps * eta_raw[0].exp();
        }
        for j in 0..=i {
            for axis in 0..d {
                r_buf[axis] = centers[[i, axis]] - centers[[j, axis]];
            }
            let bundle = if i == j {
                closed_form_penalty::analytic_self_pair_bundle(q, p_order, s_order, kappa, eta_raw)
                    .unwrap_or_else(|| {
                        closed_form_penalty::pair_block_radial_with_j_second_derivatives_with_powers(
                            q, p_order, s_order, kappa, eta_raw, &powers, &r_eps_buf,
                        )
                    })
            } else {
                closed_form_penalty::pair_block_radial_with_j_second_derivatives_with_powers(
                    q, p_order, s_order, kappa, eta_raw, &powers, &r_buf,
                )
            };
            // SAFETY: every output has k(k+1)/2 slots; for i in 0..k and j ∈ 0..=i,
            // lower_triangular_offset(i)+j is in bounds. Each rayon iteration owns
            // that lower-triangular row in every output, so writes are disjoint.
            unsafe {
                *g_row.add(j) = bundle.value;
                for a in 0..d {
                    *g_eta_rows[a].add(j) = bundle.d_eta[a];
                    *g_eta2_diag_rows[a].add(j) = bundle.d2_eta[a][a];
                }
                for (idx, &(a, b)) in cross_pairs.iter().enumerate() {
                    *g_eta2_cross_rows[idx].add(j) = bundle.d2_eta[a][b];
                }
            }
        }
    });

    let g = symmetric_matrix_from_lower_values(k, &g_values);
    let g_eta: Vec<Array2<f64>> = g_eta_values
        .iter()
        .map(|values| symmetric_matrix_from_lower_values(k, values))
        .collect();
    let g_eta2_diag: Vec<Array2<f64>> = g_eta2_diag_values
        .iter()
        .map(|values| symmetric_matrix_from_lower_values(k, values))
        .collect();
    let g_eta2_cross_unique: Vec<Array2<f64>> = g_eta2_cross_values
        .iter()
        .map(|values| symmetric_matrix_from_lower_values(k, values))
        .collect();

    // Apply Z + poly-pad + T to raw K×K matrices.
    let s = transform_closed_form_raw_block(
        &g,
        kernel_nullspace,
        polynomial_block_cols,
        outer_identifiability,
    );
    let s_first: Vec<Array2<f64>> = g_eta
        .par_iter()
        .map(|raw| {
            transform_closed_form_raw_block(
                raw,
                kernel_nullspace,
                polynomial_block_cols,
                outer_identifiability,
            )
        })
        .collect();
    let s_second_diag: Vec<Array2<f64>> = g_eta2_diag
        .par_iter()
        .map(|raw| {
            transform_closed_form_raw_block(
                raw,
                kernel_nullspace,
                polynomial_block_cols,
                outer_identifiability,
            )
        })
        .collect();
    let transformed_cross_unique: Vec<Array2<f64>> = g_eta2_cross_unique
        .par_iter()
        .map(|raw| {
            transform_closed_form_raw_block(
                raw,
                kernel_nullspace,
                polynomial_block_cols,
                outer_identifiability,
            )
        })
        .collect();
    let out_dim = s.nrows();
    let mut s_second_cross: Vec<Vec<Array2<f64>>> = (0..d)
        .map(|_| {
            (0..d)
                .map(|_| Array2::<f64>::zeros((out_dim, out_dim)))
                .collect()
        })
        .collect();
    for (idx, &(a, b)) in cross_pairs.iter().enumerate() {
        let block = &transformed_cross_unique[idx];
        s_second_cross[a][b] = block.clone();
        if a != b {
            s_second_cross[b][a] = block.clone();
        }
    }
    (s, s_first, s_second_diag, s_second_cross)
}


#[inline(always)]
fn duchon_closed_form_operator_penalty_converges(
    q: usize,
    p_order: usize,
    s_order: f64,
    dimension: usize,
) -> bool {
    // Real-valued conditions so fractional `s_order` falls inside the
    // convergent regime when admissible. Integer values reduce to the
    // original strict inequalities exactly.
    let four_ms = 4.0 * (p_order as f64 + s_order);
    let dp2q = (dimension + 2 * q) as f64;
    let four_m = (4 * p_order) as f64;
    four_ms > dp2q && dp2q > four_m && 2 * p_order >= q + 1
}


/// CPD-adequacy gate for the *scale-free* (`length_scale = None`) Duchon
/// closed-form pair block. The pure-polyharmonic pair block is
/// `S_q[i, j] ∝ R_J^d(|c_i − c_j|)` with `J = 2(p+s) − q`, which is only
/// *conditionally* positive-definite — its projection onto the
/// polynomial-orthogonal complement is PSD iff the spec's polynomial
/// null-space order is at least the kernel's CPD order
/// (Wendland Thm 8.17 / 8.18).
///
/// For the *hybrid* Matérn-blended path (`length_scale = Some`) the kernel
/// is strictly positive definite (Matérn regularization at low
/// frequencies), so no CPD restriction is needed — the UV/IR convergence
/// check is sufficient. This function is therefore only consulted from
/// the pure-Duchon candidate factory.
///
/// Concrete tripwire: at `d=8, p_order=2, s_order=3.5` with the Linear
/// null space, the closed-form pair block for q ∈ {1, 2} has CPD order
/// `(2J − d)/2 + 1 ∈ {7, 6}` (log case fires because `2s = 7` makes
/// `2J − d` an even integer at even `d`). Without this gate the
/// closed-form matrix at centers was non-PSD (15 / 30 negative
/// eigenvalues); with this gate, both q's route to collocation
/// `D_qᵀD_q` (PSD by construction). TPS sanity: `d=2, p=2, s=0, q=2`
/// gives `2J − d = 2`, log case, CPD order = 2, matched exactly by
/// the Linear null space.
fn duchon_pure_closed_form_pair_block_cpd_adequate(
    q: usize,
    p_order: usize,
    s_order: f64,
    dimension: usize,
) -> bool {
    // β = 2J − d where J = 2(p+s) − q. Equivalently β = 4(p+s) − 2q − d.
    let beta = 4.0 * (p_order as f64 + s_order) - 2.0 * q as f64 - dimension as f64;
    if beta < 0.0 {
        return false;
    }
    const LOG_EPS: f64 = 1e-12;
    let n_f = (beta / 2.0).round();
    let is_log_case =
        dimension.is_multiple_of(2) && n_f >= 0.0 && (n_f * 2.0 - beta).abs() < LOG_EPS;
    let cpd_required = if is_log_case {
        // Log case: kernel `c · r^{2n}(ln r + A_n)` is CPD of order n + 1
        // (Wendland Thm 8.18).
        (n_f as usize).saturating_add(1)
    } else {
        // Non-log case: kernel `c · r^β` is CPD of order ⌈(β+1)/2⌉
        // (Wendland Thm 8.17). For odd β this is `(β+1)/2`; for
        // fractional β it rounds up.
        ((beta + 1.0) / 2.0).ceil() as usize
    };
    p_order >= cpd_required
}


pub fn operator_penalty_candidates_closed_form(
    centers: ArrayView2<'_, f64>,
    d0: &Array2<f64>,
    d1: &Array2<f64>,
    d2: &Array2<f64>,
    spec: &DuchonOperatorPenaltySpec,
    p_order: usize,
    s_order: usize,
    length_scale: f64,
    aniso_log_scales: Option<&[f64]>,
    kernel_nullspace: Option<&Array2<f64>>,
    polynomial_block_cols: usize,
    outer_identifiability: Option<&Array2<f64>>,
) -> Vec<PenaltyCandidate> {
    let kappa = 1.0 / length_scale.max(1e-300);

    // Per-q Duchon convergence regime: closed-form Lebesgue kernel matrix is
    // PSD only when both UV `4(m+s) > d + 2q` and IR `d + 2q > 4m` hold,
    // and the partial-fraction expansion in `isotropic_duchon_penalty`
    // requires `2m - q ≥ 1`. Even-dimensional log-Riesz terms are handled
    // analytically by the canonical finite-part shift in `riesz_kernel_value`.
    // Outside this convergence regime the continuous Lebesgue Gram does not
    // define the desired PSD operator block, so the finite-K collocation
    // `D_q^T D_q` Gram remains the mathematically different object.
    let d = centers.ncols();

    // Threshold for emitting an operator-form handle alongside the dense
    // matrix. Above this raw kernel size, the closed-form factory attaches
    // a `ClosedFormPenaltyOperator` so downstream consumers (PCG-against-
    // implicit-H, Hutchinson EDF) can reuse the operator's matvec without
    // rebuilding the dense Gram. Below threshold, dense-only is preserved
    // (Cholesky on the small materialized H is faster).
    let emit_operator = centers.nrows() > CLOSED_FORM_OPERATOR_THRESHOLD;

    let make_op = |q: usize,
                   c: f64|
     -> Option<std::sync::Arc<dyn crate::terms::penalty_op::PenaltyOp>> {
        if !emit_operator {
            return None;
        }
        if !duchon_closed_form_operator_penalty_converges(q, p_order, s_order as f64, d) {
            return None;
        }
        let raw_op = std::sync::Arc::new(
            crate::terms::closed_form_operator::ClosedFormPenaltyOperator::new(
                centers,
                q,
                p_order,
                s_order,
                kappa,
                aniso_log_scales,
                kernel_nullspace,
                polynomial_block_cols,
                outer_identifiability,
            ),
        );
        // The candidate's `matrix` is the closed-form Gram divided by its
        // Frobenius norm `c`. Wrap in `ScaledPenaltyOp` with factor `1/c`
        // so `op.as_dense()` matches the candidate's dense matrix.
        let scale = if c > 1e-12 { 1.0 / c } else { 1.0 };
        let scaled: std::sync::Arc<dyn crate::terms::penalty_op::PenaltyOp> = std::sync::Arc::new(
            crate::terms::penalty_op::ScaledPenaltyOp::new(raw_op, scale),
        );
        Some(scaled)
    };

    // Each order is materialized ONLY when its spec is active, so a disabled
    // order never touches its `d_q` operand (lets the caller build `D_q` with
    // `max_op = max active order` and skip the `O(d²)`-row Hessian — the
    // `D2`-skip). Mass is the *centered* collocation Gram `Σ(f−f̄)²`, identical
    // to the pure path, so the constant direction is genuinely unpenalized.
    let mut out = Vec::new();
    if matches!(spec.mass, OperatorPenaltySpec::Active { .. }) {
        let (s0, c0) = normalize_penalty(&symmetrize(&centered_design_gram(d0)));
        out.push(PenaltyCandidate {
            matrix: s0,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorMass,
            normalization_scale: c0,
            kronecker_factors: None,
            op: None,
        });
    }
    if matches!(spec.tension, OperatorPenaltySpec::Active { .. }) {
        let s1_raw = if duchon_closed_form_operator_penalty_converges(1, p_order, s_order as f64, d)
        {
            closed_form_operator_penalty_in_total_basis(
                centers,
                1,
                p_order,
                s_order,
                kappa,
                aniso_log_scales,
                kernel_nullspace,
                polynomial_block_cols,
                outer_identifiability,
            )
        } else {
            symmetrize(&fast_ata(d1))
        };
        let (s1, c1) = normalize_penalty(&s1_raw);
        let op = make_op(1, c1);
        out.push(PenaltyCandidate {
            matrix: s1,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorTension,
            normalization_scale: c1,
            kronecker_factors: None,
            op,
        });
    }
    if matches!(spec.stiffness, OperatorPenaltySpec::Active { .. }) {
        let s2_raw = if duchon_closed_form_operator_penalty_converges(2, p_order, s_order as f64, d)
        {
            closed_form_operator_penalty_in_total_basis(
                centers,
                2,
                p_order,
                s_order,
                kappa,
                aniso_log_scales,
                kernel_nullspace,
                polynomial_block_cols,
                outer_identifiability,
            )
        } else {
            symmetrize(&fast_ata(d2))
        };
        let (s2, c2) = normalize_penalty(&s2_raw);
        let op = make_op(2, c2);
        out.push(PenaltyCandidate {
            matrix: s2,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorStiffness,
            normalization_scale: c2,
            kronecker_factors: None,
            op,
        });
    }
    out
}


/// Threshold above which the closed-form factory emits an operator-form `op`
/// handle alongside the dense matrix. Above 1500 raw kernel rows, downstream
/// consumers (PCG-against-implicit-H, Hutchinson EDF) reuse the operator's
/// matvec without rebuilding the dense Gram. Below it, only the dense form is
/// emitted — direct Cholesky on the small materialized H is faster than
/// PCG-against-implicit-H. The crossover was measured by
/// `bench_hessian_solve_dense_vs_implicit` in `benches/closed_form_criterion.rs`
/// against the synthetic SPD-with-coupled-penalty fixture there.
const CLOSED_FORM_OPERATOR_THRESHOLD: usize = 1500;


/// Pure-Duchon (κ=0 / `length_scale = None`) counterpart of
/// [`closed_form_operator_penalty_in_total_basis`]. Uses
/// [`closed_form_anisotropic_pair_block_pure`] to evaluate the closed-form
/// penalty via analytic radial derivatives of the pure-Riesz kernel, which
/// is finite for R > 0 in any (m, s, d, q) regime where
/// `radial_derivatives_of_isotropic_duchon` is defined. Self-pair (R=0)
/// regularization is handled inside the pair-block routine.
pub fn closed_form_operator_penalty_in_total_basis_pure(
    centers: ArrayView2<'_, f64>,
    q: usize,
    p_order: usize,
    s_order: f64,
    aniso_log_scales: Option<&[f64]>,
    kernel_nullspace: Option<&Array2<f64>>,
    polynomial_block_cols: usize,
    outer_identifiability: Option<&Array2<f64>>,
) -> Array2<f64> {
    // The whole scale-free Duchon chain — pair block, anisotropic radial,
    // uniform-metric branch, isotropic radial derivatives, Riesz kernel —
    // is now `f64`-threaded for `kappa = 0`. The integer-only self-pair /
    // partial-fraction helpers are still consulted opportunistically
    // (`s_int` gating inside the pair block) when `s` happens to be
    // whole-valued, but fractional `s` falls through cleanly to the
    // ε-regularized analytic radial chain.
    assert!(
        s_order.is_finite() && s_order >= 0.0,
        "closed_form_operator_penalty_in_total_basis_pure: s_order must be finite and ≥ 0, got {s_order}"
    );
    let g_raw =
        closed_form_anisotropic_pair_block_pure(centers, q, p_order, s_order, aniso_log_scales);
    let g_kernel = if let Some(z) = kernel_nullspace {
        let zt_g = fast_atb(z, &g_raw);
        fast_ab(&zt_g, z)
    } else {
        g_raw
    };
    let kernel_cols = g_kernel.nrows();
    let total_pre_cols = kernel_cols + polynomial_block_cols;
    let g_padded = if polynomial_block_cols == 0 {
        g_kernel
    } else {
        let mut padded = Array2::<f64>::zeros((total_pre_cols, total_pre_cols));
        padded
            .slice_mut(s![0..kernel_cols, 0..kernel_cols])
            .assign(&g_kernel);
        padded
    };
    let g_total = if let Some(t) = outer_identifiability {
        let tt_g = fast_atb(t, &g_padded);
        fast_ab(&tt_g, t)
    } else {
        g_padded
    };
    symmetrize(&g_total)
}


/// Pure-Duchon (κ=0 / `length_scale = None`) counterpart of
/// [`operator_penalty_candidates_closed_form`].
///
/// Builds the three operator penalty candidates (mass, tension, stiffness)
/// from the pure-Duchon closed-form path, which is a polyharmonic of order
/// `m + s = p_order + s_order` with no Matérn factor. `q ∈ {0,1,2}` rolls
/// the penalty differential order — same as the hybrid path.
///
/// The closed-form Lebesgue penalty is finite only when the Duchon
/// convergence conditions hold:
///   - UV (smoothness at origin): `4(m+s) > d + 2q`
///   - IR (decay at infinity):    `d + 2q > 4m`
/// When either condition fails for a given `q`, the closed-form integrand
/// diverges (UV) or vanishes identically (IR — kernel is a finite-degree
/// polynomial that drops out of `Δ_B^q`); in those regimes we fall back to
/// the collocation Gram `D_q^T D_q`, which is the same regularization the
/// pre-closed-form path used.
pub fn operator_penalty_candidates_closed_form_pure(
    centers: ArrayView2<'_, f64>,
    d0: &Array2<f64>,
    d1: &Array2<f64>,
    d2: &Array2<f64>,
    spec: &DuchonOperatorPenaltySpec,
    p_order: usize,
    s_order: f64,
    aniso_log_scales: Option<&[f64]>,
    kernel_nullspace: Option<&Array2<f64>>,
    polynomial_block_cols: usize,
    outer_identifiability: Option<&Array2<f64>>,
) -> Vec<PenaltyCandidate> {
    // q=0 mass is the *centered* collocation Gram — the data-density-weighted
    // spring penalty on deviations from the function's mean over the
    // collocation sites. Centering each design column by its mean across rows
    // before forming the Gram puts the constant direction exactly into the
    // penalty's null space (intercept genuinely unpenalized): for the
    // constant basis column (all-ones), the column mean is one and the
    // centered column is identically zero, so the resulting Gram row/column
    // for that direction is zero. Algebraically this is
    // `(D_0 - 1 μ')^T (D_0 - 1 μ')` where `μ_j = (1/N) Σ_i D_0[i, j]`. This
    // expresses the "springs to a floating flat sheet" semantics — the level
    // is free and only deviations get the spring force — while staying inside
    // the standard quadratic-penalty machinery.
    let d = centers.ncols();
    // Convergence predicate also requires `isotropic_duchon_penalty`'s
    // partial-fraction precondition `2m ≥ q + 1`; without it, closed-form
    // panics on configs like m=1, q=2. Even-dimensional log-Riesz branches are
    // admitted because `riesz_kernel_value` now uses the canonical finite part.
    // Closed-form pair block is admitted only when both the UV/IR
    // convergence predicate AND Wendland's CPD-adequacy condition hold;
    // the second guards against silently-non-PSD pair blocks when the
    // polynomial null space is too small to absorb the kernel's CPD
    // order (e.g. d=8, p_order=2, s_order=3.5 log-case). Failing
    // either test routes to collocation `D_qᵀD_q`, which is PSD by
    // construction.
    let closed_form_ok = |q: usize| -> bool {
        duchon_closed_form_operator_penalty_converges(q, p_order, s_order, d)
            && duchon_pure_closed_form_pair_block_cpd_adequate(q, p_order, s_order, d)
    };
    // Each order is materialized ONLY when its spec is active: a disabled order
    // never touches its `d_q` operand, so the caller can build `D_q` with
    // `max_op = max active order` and leave the higher-order designs empty (the
    // `D2`-skip — decisive in high `d`, where the Hessian has `O(d²)` rows).
    let mut out = Vec::new();
    if matches!(spec.mass, OperatorPenaltySpec::Active { .. }) {
        // q=0 mass is the *centered* collocation Gram — the data-density-weighted
        // spring penalty on deviations from the function's mean over the
        // collocation sites; centering puts the constant direction exactly into
        // the penalty null space (intercept genuinely unpenalized).
        let (s0, c0) = normalize_penalty(&symmetrize(&centered_design_gram(d0)));
        out.push(PenaltyCandidate {
            matrix: s0,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorMass,
            normalization_scale: c0,
            kronecker_factors: None,
            op: None,
        });
    }
    if matches!(spec.tension, OperatorPenaltySpec::Active { .. }) {
        let s1_raw = if closed_form_ok(1) {
            closed_form_operator_penalty_in_total_basis_pure(
                centers,
                1,
                p_order,
                s_order,
                aniso_log_scales,
                kernel_nullspace,
                polynomial_block_cols,
                outer_identifiability,
            )
        } else {
            symmetrize(&fast_ata(d1))
        };
        let (s1, c1) = normalize_penalty(&s1_raw);
        out.push(PenaltyCandidate {
            matrix: s1,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorTension,
            normalization_scale: c1,
            kronecker_factors: None,
            op: None,
        });
    }
    if matches!(spec.stiffness, OperatorPenaltySpec::Active { .. }) {
        let s2_raw = if closed_form_ok(2) {
            closed_form_operator_penalty_in_total_basis_pure(
                centers,
                2,
                p_order,
                s_order,
                aniso_log_scales,
                kernel_nullspace,
                polynomial_block_cols,
                outer_identifiability,
            )
        } else {
            symmetrize(&fast_ata(d2))
        };
        let (s2, c2) = normalize_penalty(&s2_raw);
        out.push(PenaltyCandidate {
            matrix: s2,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorStiffness,
            normalization_scale: c2,
            kronecker_factors: None,
            op: None,
        });
    }
    out
}


fn operator_penalty_candidates_from_collocation(
    d0: &Array2<f64>,
    d1: &Array2<f64>,
    d2: &Array2<f64>,
    spec: &DuchonOperatorPenaltySpec,
) -> Vec<PenaltyCandidate> {
    let s0_raw = symmetrize(&fast_ata(d0));
    let (s0, c0) = normalize_penalty(&s0_raw);
    let (s1, c1) = normalize_penalty(&symmetrize(&fast_ata(d1)));
    let (s2, c2) = normalize_penalty(&symmetrize(&fast_ata(d2)));
    let mut out = Vec::new();
    if matches!(spec.mass, OperatorPenaltySpec::Active { .. }) {
        out.push(PenaltyCandidate {
            matrix: s0,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorMass,
            normalization_scale: c0,
            kronecker_factors: None,
            op: None,
        });
    }
    if matches!(spec.tension, OperatorPenaltySpec::Active { .. }) {
        out.push(PenaltyCandidate {
            matrix: s1,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorTension,
            normalization_scale: c1,
            kronecker_factors: None,
            op: None,
        });
    }
    if matches!(spec.stiffness, OperatorPenaltySpec::Active { .. }) {
        out.push(PenaltyCandidate {
            matrix: s2,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorStiffness,
            normalization_scale: c2,
            kronecker_factors: None,
            op: None,
        });
    }
    out
}


fn active_operator_penalty_derivatives(
    penaltyinfo: &[PenaltyInfo],
    operator_derivatives: &[Array2<f64>],
    label: &str,
) -> Result<Vec<Array2<f64>>, BasisError> {
    if operator_derivatives.len() != 3 {
        crate::bail_invalid_basis!(
            "{label} operator derivative path requires 3 canonical penalties; found {}",
            operator_derivatives.len()
        );
    }

    penaltyinfo
        .iter()
        .filter(|info| info.active)
        .map(|info| match &info.source {
            PenaltySource::OperatorMass => Ok(operator_derivatives[0].clone()),
            PenaltySource::OperatorTension => Ok(operator_derivatives[1].clone()),
            PenaltySource::OperatorStiffness => Ok(operator_derivatives[2].clone()),
            other => Err(BasisError::InvalidInput(format!(
                "unexpected {label} penalty source in canonical operator path: {other:?}"
            ))),
        })
        .collect()
}


fn frozen_spatial_identifiability_transform(
    identifiability: &SpatialIdentifiability,
    expectedrows: usize,
    label: &str,
) -> Result<Option<Array2<f64>>, BasisError> {
    match identifiability {
        SpatialIdentifiability::None | SpatialIdentifiability::OrthogonalToParametric => Ok(None),
        SpatialIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != expectedrows {
                crate::bail_dim_basis!(
                    "frozen {label} identifiability transform mismatch: rows={}, expected {expectedrows}",
                    transform.nrows()
                );
            }
            Ok(Some(transform.clone()))
        }
    }
}


/// Returns the parametric-constraint columns used by the standalone
/// `OrthogonalToParametric` spatial identifiability transform.
///
/// **What it contains:** a single all-ones column (the global intercept).
///
/// **What it consumes vs. what it does not:**
///
/// - *Consumes* (orthogonalises the smooth against): the intercept direction
///   only.  After this transform the smooth columns have zero unweighted mean,
///   so they cannot absorb a global additive constant that belongs to the
///   intercept parameter.
///
/// - *Does not consume*: the full polynomial null space of the Duchon kernel
///   (constants + linear + higher-order monomials).  The linear and higher
///   monomial directions in `[1, x₁, …, x_d]` are already handled by the
///   kernel side-condition projection inside `kernel_constraint_nullspace` —
///   that step compresses the radial kernel block from `k` columns down to
///   `k − C(d+r, r)` columns, so those directions never appear as free
///   smooth columns in the first place.  The spatial identifiability transform
///   only needs to remove the global-intercept residual left over after that
///   projection.
///
/// - *Does not consume*: cross-block aliases that arise when the same Duchon
///   smooth appears in multiple formula channels (e.g. marginal and logslope).
///   Two channels with identical raw bases have cosine-similarity 1.0 between
///   the corresponding columns; that aliasing is detected and resolved by the
///   joint cross-block identifiability audit (`audit_identifiability` /
///   `audit_identifiability_channel_aware`), not here.
fn spatial_parametric_constraint_block(data: ArrayView2<'_, f64>) -> Array2<f64> {
    let n = data.nrows();
    Array2::<f64>::ones((n, 1))
}


fn build_thin_plate_penalty_matrices(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    kernel_transform: &Array2<f64>,
    double_penalty: bool,
) -> Result<(Array2<f64>, Option<Array2<f64>>), BasisError> {
    let k = centers.nrows();
    let d = centers.ncols();
    let kernel_cols = kernel_transform.ncols();
    let poly_cols = thin_plate_polynomial_basis_dimension(d);
    let total_cols = kernel_cols + poly_cols;
    let mut omega = Array2::<f64>::zeros((k, k));
    let length_scale_sq = length_scale * length_scale;
    fill_symmetric_from_row_kernel(&mut omega, |i, j| {
        let mut dist2 = 0.0;
        for c in 0..d {
            let delta = centers[[i, c]] - centers[[j, c]];
            dist2 += delta * delta;
        }
        thin_plate_kernel_from_dist2(dist2 / length_scale_sq, d)
    })?;
    let omega_constrained = {
        let zt_o = fast_atb(kernel_transform, &omega);
        // `kernel_transform` spans the side-constraint nullspace, so the
        // congruence transform preserves the thin-plate PSD construction.
        // Symmetrize to remove roundoff asymmetry without paying for a full EVD
        // on the large lazy-path penalty.
        symmetrize_penalty(&fast_ab(&zt_o, kernel_transform))
    };
    let mut penalty_bending = Array2::<f64>::zeros((total_cols, total_cols));
    penalty_bending
        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&omega_constrained);
    let penalty_ridge = if double_penalty {
        build_nullspace_shrinkage_penalty(&penalty_bending)?.map(|block| block.sym_penalty)
    } else {
        None
    };
    Ok((penalty_bending, penalty_ridge))
}


/// Drop redundant Matérn centers when an over-specified `centers=K` exceeds the
/// kernel's numerical rank on the data cloud (#755).
///
/// The Matérn kernel has a fixed `length_scale` (default 1.0 on standardized
/// inputs), so packing too many centers into a tight data cloud produces
/// overlapping, near-identical radial basis functions. The realized kernel
/// design `K(data, centers)` then carries exactly linearly-dependent columns,
/// which the downstream identifiability audit hard-FATALs as intra-block rank
/// deficiency. (Duchon is scale-free and never hits this.)
///
/// We detect the deficiency on the *realized* kernel design block (the same
/// matrix the audit RRQRs, before the identifiability transform) via a
/// column-pivoted rank-revealing QR at the crate-standard rank tolerance, so
/// the reduction fires exactly when — and only when — the audit would have
/// FATAL'd. When `rank < K`, we keep the leading `rank` pivoted centers
/// (restored to ascending original order so the basis layout stays
/// deterministic) and drop the redundant remainder. Returning a full-rank
/// center subset keeps the design, penalty, and identifiability machinery
/// mutually consistent because they are all rebuilt from the same centers.
fn matern_rank_reduce_centers(
    data: ArrayView2<'_, f64>,
    centers: &Array2<f64>,
    length_scale: f64,
    nu: MaternNu,
    aniso_log_scales: Option<&[f64]>,
) -> Result<Array2<f64>, BasisError> {
    let k = centers.nrows();
    let n = data.nrows();
    // Need at least as many rows as columns for a column rank to be meaningful;
    // the kernel design is n × K, and a 0/1-center basis can never be deficient.
    if k <= 1 || n < k {
        return Ok(centers.clone());
    }
    let mut kernel_block = Array2::<f64>::zeros((n, k));
    let axis_scales = aniso_log_scales.map(aniso_axis_scales);
    let centers_view = centers.view();
    kernel_block
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| {
            for j in 0..k {
                let r = if let Some(scales) = axis_scales.as_deref() {
                    aniso_distance_rows_with_scales(data, i, centers_view, j, scales)
                } else {
                    euclidean_distance_rows(data, i, centers_view, j)
                };
                row[j] = matern_kernel_from_distance(r, length_scale, nu)?;
            }
            Ok::<(), BasisError>(())
        })?;
    let rrqr = rrqr_with_permutation(&kernel_block, default_rrqr_rank_alpha())
        .map_err(BasisError::LinalgError)?;
    if rrqr.rank >= k {
        return Ok(centers.clone());
    }
    // Rank 0 means the realized kernel design has no numerically independent
    // columns at all: every center collapses to the same (near-constant) radial
    // response on this data cloud at the chosen `length_scale`, so the Matérn
    // term carries no usable signal. Emitting a 0-center basis here would leave a
    // degenerate term whose 0-column design desyncs against its identifiability
    // transform and silently corrupts the fit (#1090). Fail loudly with an
    // actionable message instead — a length_scale this large relative to the data
    // spread (or a near-degenerate coordinate cloud) needs the user to widen the
    // domain, shrink the length scale, or drop the term.
    if rrqr.rank == 0 {
        crate::bail_invalid_basis!(
            "Matérn smooth has data-supported numerical rank 0: all {k} center(s) are \
             numerically collinear at length_scale={length_scale} on this data cloud, so the \
             kernel basis is degenerate (no independent columns). Reduce length_scale, spread \
             the coordinate cloud, or drop this term (#1090/#755)."
        );
    }
    let mut keep = rrqr.column_permutation[..rrqr.rank].to_vec();
    keep.sort_unstable();
    log::info!(
        "Matérn centers reduced from {k} to {} (data-supported numerical rank): \
         requested centers exceed the kernel's rank at length_scale={length_scale}, so \
         {} collinear basis column(s) were dropped to keep the basis full-rank (#755).",
        rrqr.rank,
        k - rrqr.rank,
    );
    let mut reduced = Array2::<f64>::zeros((keep.len(), centers.ncols()));
    for (new_row, &old_row) in keep.iter().enumerate() {
        reduced.row_mut(new_row).assign(&centers.row(old_row));
    }
    Ok(reduced)
}


fn build_matern_kernel_penalty(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    aniso_log_scales: Option<&[f64]>,
) -> Result<Array2<f64>, BasisError> {
    let k = centers.nrows();
    let total_cols = k + usize::from(include_intercept);
    let mut center_kernel = Array2::<f64>::zeros((k, k));
    let axis_scales = aniso_log_scales.map(aniso_axis_scales);
    fill_symmetric_from_row_kernel(&mut center_kernel, |i, j| {
        let r = if let Some(scales) = axis_scales.as_deref() {
            aniso_distance_rows_with_scales(centers, i, centers, j, scales)
        } else {
            euclidean_distance_rows(centers, i, centers, j)
        };
        matern_kernel_from_distance(r, length_scale, nu)
    })?;
    let mut penalty_kernel = Array2::<f64>::zeros((total_cols, total_cols));
    penalty_kernel
        .slice_mut(s![0..k, 0..k])
        .assign(&center_kernel);
    Ok(penalty_kernel)
}


/// Compute the spatial identifiability transform for a dense design matrix.
///
/// For the `OrthogonalToParametric` policy the transform orthogonalises `design`
/// against the **intercept only** (a column of ones built from `data`).  This
/// removes one direction from the smooth's column space, so a basis with
/// pre-transform width `p` yields a post-transform width of `p − 1`.
///
/// The polynomial null space of the Duchon kernel is consumed *upstream* by
/// `kernel_constraint_nullspace`, not here.  See
/// [`spatial_parametric_constraint_block`] for a precise description of what
/// this step does and does not consume.
fn spatial_identifiability_transform_from_design(
    data: ArrayView2<'_, f64>,
    design: ArrayView2<'_, f64>,
    identifiability: &SpatialIdentifiability,
    label: &str,
) -> Result<Option<Array2<f64>>, BasisError> {
    match identifiability {
        SpatialIdentifiability::None => Ok(None),
        SpatialIdentifiability::OrthogonalToParametric => {
            let c = spatial_parametric_constraint_block(data);
            let (_, z) = applyweighted_orthogonality_constraint(design, c.view(), None)?;
            Ok(Some(z))
        }
        SpatialIdentifiability::FrozenTransform { .. } => {
            frozen_spatial_identifiability_transform(identifiability, design.ncols(), label)
        }
    }
}


fn spatial_identifiability_transform_from_design_matrix(
    data: ArrayView2<'_, f64>,
    design: &DesignMatrix,
    identifiability: &SpatialIdentifiability,
    label: &str,
) -> Result<Option<Array2<f64>>, BasisError> {
    match identifiability {
        SpatialIdentifiability::None => Ok(None),
        SpatialIdentifiability::OrthogonalToParametric => {
            let c = spatial_parametric_constraint_block(data);
            let z = orthogonality_transform_for_design(design, c.view(), None)?;
            Ok(Some(z))
        }
        SpatialIdentifiability::FrozenTransform { .. } => {
            frozen_spatial_identifiability_transform(identifiability, design.ncols(), label)
        }
    }
}


fn thin_plate_intercept_transform_from_column_means(
    column_means: &Array1<f64>,
    kernel_cols: usize,
    poly_cols: usize,
) -> Result<Array2<f64>, BasisError> {
    let total_cols = kernel_cols + poly_cols;
    if column_means.len() != total_cols {
        crate::bail_dim_basis!(
            "thin-plate column-mean length mismatch: got {}, expected {total_cols}",
            column_means.len()
        );
    }
    if poly_cols == 0 {
        return Ok(Array2::<f64>::eye(total_cols));
    }
    let out_cols = total_cols
        .checked_sub(1)
        .ok_or_else(|| BasisError::InvalidInput("thin-plate basis has no columns".to_string()))?;
    let mut transform = Array2::<f64>::zeros((total_cols, out_cols));

    for j in 0..kernel_cols {
        transform[[j, j]] = 1.0;
        transform[[kernel_cols, j]] = -column_means[j];
    }
    for poly_j in 1..poly_cols {
        let src = kernel_cols + poly_j;
        let dst = kernel_cols + poly_j - 1;
        transform[[src, dst]] = 1.0;
        transform[[kernel_cols, dst]] = -column_means[src];
    }
    Ok(transform)
}


fn thin_plate_identifiability_transform_from_design(
    design: ArrayView2<'_, f64>,
    kernel_cols: usize,
    poly_cols: usize,
    identifiability: &SpatialIdentifiability,
) -> Result<Option<Array2<f64>>, BasisError> {
    match identifiability {
        SpatialIdentifiability::None => Ok(None),
        SpatialIdentifiability::OrthogonalToParametric => {
            let n = design.nrows();
            if n == 0 {
                crate::bail_invalid_basis!("thin-plate identifiability requires at least one row");
            }
            let means = design.sum_axis(Axis(0)).mapv(|v| v / n as f64);
            Ok(Some(thin_plate_intercept_transform_from_column_means(
                &means,
                kernel_cols,
                poly_cols,
            )?))
        }
        SpatialIdentifiability::FrozenTransform { .. } => {
            frozen_spatial_identifiability_transform(identifiability, design.ncols(), "ThinPlate")
        }
    }
}


fn thin_plate_identifiability_transform_from_design_matrix(
    design: &DesignMatrix,
    kernel_cols: usize,
    poly_cols: usize,
    identifiability: &SpatialIdentifiability,
) -> Result<Option<Array2<f64>>, BasisError> {
    match identifiability {
        SpatialIdentifiability::None => Ok(None),
        SpatialIdentifiability::OrthogonalToParametric => {
            let n = design.nrows();
            if n == 0 {
                crate::bail_invalid_basis!("thin-plate identifiability requires at least one row");
            }
            let ones = Array1::<f64>::ones(n);
            let means = design.apply_transpose(&ones).mapv(|v| v / n as f64);
            Ok(Some(thin_plate_intercept_transform_from_column_means(
                &means,
                kernel_cols,
                poly_cols,
            )?))
        }
        SpatialIdentifiability::FrozenTransform { .. } => {
            frozen_spatial_identifiability_transform(identifiability, design.ncols(), "ThinPlate")
        }
    }
}


fn append_intercept_to_transform(transform: &Array2<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((transform.nrows() + 1, transform.ncols() + 1));
    out.slice_mut(s![0..transform.nrows(), 0..transform.ncols()])
        .assign(transform);
    out[[transform.nrows(), transform.ncols()]] = 1.0;
    out
}


fn project_penalty_matrix(matrix: &Array2<f64>, transform: Option<&Array2<f64>>) -> Array2<f64> {
    let projected = if let Some(z) = transform {
        let zt_s = z.t().dot(matrix);
        zt_s.dot(z)
    } else {
        matrix.clone()
    };
    symmetrize(&projected)
}


fn normalize_penalty_candidate(
    matrix: Array2<f64>,
    nullspace_dim_hint: usize,
    source: PenaltySource,
) -> PenaltyCandidate {
    let (matrix, normalization_scale) = if matrix.iter().all(|v| v.abs() <= 1e-12) {
        (matrix, 1.0)
    } else {
        normalize_penalty(&matrix)
    };
    PenaltyCandidate {
        matrix,
        nullspace_dim_hint,
        source,
        normalization_scale,
        kronecker_factors: None,
        op: None,
    }
}


pub fn build_matern_collocation_operator_matrices(
    centers: ArrayView2<'_, f64>,
    collocationweights: Option<ArrayView1<'_, f64>>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    identifiability_transform: Option<ArrayView2<'_, f64>>,
    aniso_log_scales: Option<&[f64]>,
) -> Result<CollocationOperatorMatrices, BasisError> {
    // Specialized Matérn operator assembly using explicit half-integer formulas:
    // - one exp(-a) and small polynomials per pair,
    // - NaN-safe phi'(r)/r without dividing by r for nu>=3/2,
    // - exact Hessian rows for the stiffness operator, not just the Laplacian.
    let p = centers.nrows();
    let d = centers.ncols();
    let row_scales = if let Some(w) = collocationweights {
        if w.len() != p {
            crate::bail_dim_basis!(
                "collocation weight length mismatch: got {}, expected {p}",
                w.len()
            );
        }
        let mut out = Vec::with_capacity(p);
        for &wk in w {
            if !wk.is_finite() || wk < 0.0 {
                crate::bail_invalid_basis!(
                    "collocation weights must be finite and non-negative; got {wk}"
                );
            }
            out.push(wk.sqrt());
        }
        out
    } else {
        vec![1.0; p]
    };
    let mut d0_raw = Array2::<f64>::zeros((p, p));
    let mut d1_raw = Array2::<f64>::zeros((p * d, p));
    let mut d2_raw = Array2::<f64>::zeros((p * d * d, p));
    let metric_weights = aniso_log_scales.map(centered_aniso_metric_weights);
    const R_EPS: f64 = 1e-12;
    // Row blocks are independent: output rows [k] in d0, [k*d..(k+1)*d] in d1,
    // and [k*d*d..(k+1)*d*d] in d2 are disjoint for each collocation row k.
    // Keep small assemblies serial to avoid Rayon scheduling overhead.
    const MATERN_COLLOCATION_PAR_WORK_THRESHOLD: usize = 32_768;
    const MATERN_COLLOCATION_ROW_BLOCK: usize = 32;
    let assembly_work = p
        .saturating_mul(p)
        .saturating_mul(d.max(1))
        .saturating_mul(d.max(1));
    let row_block_size = MATERN_COLLOCATION_ROW_BLOCK.min(p.max(1));
    let assemble_chunk = |ci: usize,
                          mut d0_chunk: ArrayViewMut2<'_, f64>,
                          mut d1_chunk: ArrayViewMut2<'_, f64>,
                          mut d2_chunk: ArrayViewMut2<'_, f64>|
     -> Result<(), BasisError> {
        let chunk_start = ci * row_block_size;
        for local_k in 0..d0_chunk.nrows() {
            let k = chunk_start + local_k;
            let scale_k = row_scales[k];
            for j in 0..p {
                // Distance: anisotropic r = |Ah| when eta present, isotropic |h| otherwise.
                let r = if let Some(eta) = aniso_log_scales {
                    aniso_distance_and_components(
                        centers.row(k).as_slice().unwrap(),
                        centers.row(j).as_slice().unwrap(),
                        eta,
                    )
                    .0
                } else {
                    stable_euclidean_norm((0..d).map(|c| centers[[k, c]] - centers[[j, c]]))
                };
                if matches!(nu, MaternNu::Half) && r <= R_EPS && d > 1 {
                    crate::bail_invalid_basis!(
                        "Matérn nu=1/2 has singular Laplacian at center collisions for d>1; choose nu>=3/2 or avoid collocation at centers"
                    );
                }
                let (phi, _, phi_rr, phi_r_over_r) =
                    if matches!(nu, MaternNu::Half) && r <= R_EPS && d == 1 {
                        // In 1D: Delta phi = phi'' and the singular phi'/r term is absent.
                        let s = 1.0 / length_scale;
                        let e = 1.0;
                        (e, -s * e, s * s * e, 0.0)
                    } else {
                        matern_kernel_radial_tripletwith_safe_ratio(r, length_scale, nu)?
                    };
                d0_chunk[[local_k, j]] = scale_k * phi;
                if r > R_EPS {
                    for c in 0..d {
                        let delta = centers[[k, c]] - centers[[j, c]];
                        d1_chunk[[local_k * d + c, j]] = scale_k * phi_r_over_r * delta;
                    }
                } else {
                    // Symmetry at center-center coincidence.
                    for c in 0..d {
                        d1_chunk[[local_k * d + c, j]] = 0.0;
                    }
                }
                let t = if r > R_EPS {
                    (phi_rr - phi_r_over_r) / (r * r)
                } else {
                    0.0
                };
                for a in 0..d {
                    let h_a = centers[[k, a]] - centers[[j, a]];
                    let w_a = metric_weights.as_ref().map(|w| w[a]).unwrap_or(1.0);
                    for b in 0..d {
                        let h_b = centers[[k, b]] - centers[[j, b]];
                        let w_b = metric_weights.as_ref().map(|w| w[b]).unwrap_or(1.0);
                        let diagonal = if a == b { phi_r_over_r * w_a } else { 0.0 };
                        let mixed = if r > R_EPS {
                            t * w_a * h_a * w_b * h_b
                        } else {
                            0.0
                        };
                        let row = (local_k * d + a) * d + b;
                        d2_chunk[[row, j]] = scale_k * (diagonal + mixed);
                    }
                }
                if !d0_chunk[[local_k, j]].is_finite()
                    || ((local_k * d * d)..((local_k + 1) * d * d))
                        .any(|row| !d2_chunk[[row, j]].is_finite())
                {
                    crate::bail_invalid_basis!(
                        "non-finite Matérn collocation operator entry at row={k}, col={j}, r={r}, nu={nu:?}"
                    );
                }
            }
        }
        Ok(())
    };
    if d == 0 && p > 0 {
        for k in 0..p {
            let scale_k = row_scales[k];
            for j in 0..p {
                let (phi, _, _, _) =
                    matern_kernel_radial_tripletwith_safe_ratio(0.0, length_scale, nu)?;
                d0_raw[[k, j]] = scale_k * phi;
                if !d0_raw[[k, j]].is_finite() {
                    crate::bail_invalid_basis!(
                        "non-finite Matérn collocation operator entry at row={k}, col={j}, r=0, nu={nu:?}"
                    );
                }
            }
        }
    } else if assembly_work >= MATERN_COLLOCATION_PAR_WORK_THRESHOLD && p > 1 {
        d0_raw
            .axis_chunks_iter_mut(Axis(0), row_block_size)
            .into_par_iter()
            .zip(
                d1_raw
                    .axis_chunks_iter_mut(Axis(0), row_block_size * d)
                    .into_par_iter(),
            )
            .zip(
                d2_raw
                    .axis_chunks_iter_mut(Axis(0), row_block_size * d * d)
                    .into_par_iter(),
            )
            .enumerate()
            .try_for_each(|(ci, ((d0_chunk, d1_chunk), d2_chunk))| {
                assemble_chunk(ci, d0_chunk, d1_chunk, d2_chunk)
            })?;
    } else if p > 0 {
        d0_raw
            .axis_chunks_iter_mut(Axis(0), row_block_size)
            .zip(d1_raw.axis_chunks_iter_mut(Axis(0), row_block_size * d))
            .zip(d2_raw.axis_chunks_iter_mut(Axis(0), row_block_size * d * d))
            .enumerate()
            .try_for_each(|(ci, ((d0_chunk, d1_chunk), d2_chunk))| {
                assemble_chunk(ci, d0_chunk, d1_chunk, d2_chunk)
            })?;
    }
    let (d0_kernel, d1_kernel, d2_kernel) = if let Some(z) = identifiability_transform {
        let z = z.to_owned();
        (
            fast_ab(&d0_raw, &z),
            fast_ab(&d1_raw, &z),
            fast_ab(&d2_raw, &z),
        )
    } else {
        (d0_raw, d1_raw, d2_raw)
    };
    let p_colloc = centers.nrows();
    let dim = centers.ncols();
    let kernel_cols = d0_kernel.ncols();
    let total_cols = kernel_cols + usize::from(include_intercept);
    let mut d0 = Array2::<f64>::zeros((p_colloc, total_cols));
    let mut d1 = Array2::<f64>::zeros((p_colloc * dim, total_cols));
    let mut d2 = Array2::<f64>::zeros((p_colloc * dim * dim, total_cols));
    d0.slice_mut(s![.., 0..kernel_cols]).assign(&d0_kernel);
    d1.slice_mut(s![.., 0..kernel_cols]).assign(&d1_kernel);
    d2.slice_mut(s![.., 0..kernel_cols]).assign(&d2_kernel);
    if include_intercept {
        for (k, &scale_k) in row_scales.iter().enumerate() {
            d0[[k, kernel_cols]] = scale_k;
        }
    }
    Ok(CollocationOperatorMatrices {
        d0,
        d1,
        d2,
        collocation_points: centers.to_owned(),
        kernel_nullspace_transform: None,
        polynomial_block_cols: usize::from(include_intercept),
    })
}


pub fn build_duchon_collocation_operator_matrices(
    centers: ArrayView2<'_, f64>,
    collocationweights: Option<ArrayView1<'_, f64>>,
    length_scale: Option<f64>,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: Option<&[f64]>,
    identifiability_transform: Option<ArrayView2<'_, f64>>,
    max_operator_derivative_order: usize,
) -> Result<CollocationOperatorMatrices, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_collocation_operator_matriceswithworkspace(
        centers,
        centers,
        collocationweights,
        length_scale,
        power,
        nullspace_order,
        aniso_log_scales,
        identifiability_transform,
        max_operator_derivative_order,
        &mut workspace,
    )
}


pub fn build_duchon_operator_penalty_matrices(
    centers: ArrayView2<'_, f64>,
    collocationweights: Option<ArrayView1<'_, f64>>,
    length_scale: Option<f64>,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: Option<&[f64]>,
    identifiability_transform: Option<ArrayView2<'_, f64>>,
) -> Result<DuchonOperatorPenaltyMatrices, BasisError> {
    let ops = build_duchon_collocation_operator_matrices(
        centers,
        collocationweights,
        length_scale,
        power,
        nullspace_order,
        aniso_log_scales,
        identifiability_transform,
        2,
    )?;
    let (mass, _) = normalize_penalty(&symmetrize(&fast_ata(&ops.d0)));
    let (tension, _) = normalize_penalty(&symmetrize(&fast_ata(&ops.d1)));
    let (stiffness, _) = normalize_penalty(&symmetrize(&fast_ata(&ops.d2)));
    Ok(DuchonOperatorPenaltyMatrices {
        mass,
        tension,
        stiffness,
    })
}


pub fn build_thin_plate_penalty_matrix(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
) -> Result<ThinPlatePenaltyMatrix, BasisError> {
    let mut workspace = BasisWorkspace::default();
    let kernel_transform = thin_plate_kernel_constraint_nullspace(centers, &mut workspace.cache)?;
    let (penalty, _) =
        build_thin_plate_penalty_matrices(centers, length_scale, &kernel_transform, false)?;
    let (penalty, _) = normalize_penalty(&penalty);
    Ok(ThinPlatePenaltyMatrix { penalty })
}


pub fn build_duchon_collocation_operator_matriceswithworkspace(
    centers: ArrayView2<'_, f64>,
    collocation_points: ArrayView2<'_, f64>,
    collocationweights: Option<ArrayView1<'_, f64>>,
    length_scale: Option<f64>,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: Option<&[f64]>,
    identifiability_transform: Option<ArrayView2<'_, f64>>,
    max_operator_derivative_order: usize,
    workspace: &mut BasisWorkspace,
) -> Result<CollocationOperatorMatrices, BasisError> {
    // The operator design rows are the COLLOCATION points (a density-blind,
    // space-filling sample of the data support); the columns are the `k` basis
    // CENTERS. Decoupling them is what makes the operator penalty a faithful
    // quadrature of `∫‖Dᵠf‖²` (collocating at the `k` centers themselves — the
    // old `collocation_points == centers` special case — under-samples a
    // `k`-bump basis and is what made these penalties explode).
    let nullspace_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let p_order = duchon_p_from_nullspace_order(nullspace_order);
    let s_order: f64 = power;
    let p_colloc = collocation_points.nrows();
    let n_basis = centers.nrows();
    let dim = centers.ncols();
    if collocation_points.ncols() != dim {
        crate::bail_dim_basis!(
            "collocation points dim {} != centers dim {dim}",
            collocation_points.ncols()
        );
    }
    validate_duchon_collocation_orders(
        length_scale,
        p_order,
        s_order,
        dim,
        max_operator_derivative_order,
    )?;
    if let Some(eta) = aniso_log_scales
        && eta.len() != dim
    {
        crate::bail_dim_basis!(
            "Duchon anisotropy dimension mismatch: got {}, expected {dim}",
            eta.len()
        );
    }
    // Partial-fraction expansion only runs in the hybrid Matérn branch
    // (`length_scale = Some`). The scale-free path (`length_scale = None`)
    // skips it entirely and is fractional-clean down to the Riesz kernel.
    let coeffs = length_scale.map(|scale| {
        let s_int = duchon_power_to_usize(s_order);
        duchon_partial_fraction_coeffs(p_order, s_int, 1.0 / scale.max(1e-300))
    });
    let metric_weights: Option<Vec<f64>> = aniso_log_scales.map(centered_aniso_metric_weights);
    let row_scales = if let Some(w) = collocationweights {
        if w.len() != p_colloc {
            crate::bail_dim_basis!(
                "collocation weight length mismatch: got {}, expected {p_colloc}",
                w.len()
            );
        }
        let mut out = Vec::with_capacity(p_colloc);
        for &wk in w {
            if !wk.is_finite() || wk < 0.0 {
                crate::bail_invalid_basis!(
                    "collocation weights must be finite and non-negative; got {wk}"
                );
            }
            out.push(wk.sqrt());
        }
        out
    } else {
        vec![1.0; p_colloc]
    };
    let z = kernel_constraint_nullspace(centers, nullspace_order, &mut workspace.cache)?;
    // D0/D1/D2 rows = collocation points (`p_colloc`), columns = basis centers
    // (`n_basis`). Gradients/Hessians are taken w.r.t. the EVALUATION point
    // (the collocation row), so `delta = collocation - center`. No symmetry: the
    // two point sets differ in general.
    // Skip the costly higher-derivative designs the caller doesn't need: mass
    // (D0) + tension (D1) build with `max_op = 1`, so the `O(d²)`-row Hessian
    // (D2) is never allocated or filled — decisive in high `d`.
    let build_d1 = max_operator_derivative_order >= 1;
    let build_d2 = max_operator_derivative_order >= 2;
    let mut d0_raw = Array2::<f64>::zeros((p_colloc, n_basis));
    let mut d1_raw = Array2::<f64>::zeros((if build_d1 { p_colloc * dim } else { 0 }, n_basis));
    let mut d2_raw =
        Array2::<f64>::zeros((if build_d2 { p_colloc * dim * dim } else { 0 }, n_basis));
    const R_EPS: f64 = 1e-10;
    for i in 0..p_colloc {
        let scale_i = row_scales[i];
        for j in 0..n_basis {
            let r = if let Some(eta) = aniso_log_scales {
                let row_i: Vec<f64> = (0..dim).map(|a| collocation_points[[i, a]]).collect();
                let row_j: Vec<f64> = (0..dim).map(|a| centers[[j, a]]).collect();
                aniso_distance(&row_i, &row_j, eta)
            } else {
                stable_euclidean_norm(
                    (0..dim).map(|axis| collocation_points[[i, axis]] - centers[[j, axis]]),
                )
            };
            // Floor coincident collocation/center pairs off the kernel's origin
            // singularity: a farthest-point sample can land exactly on a center.
            // The gradient/Hessian limits at r→0 are the zeros the `r > R_EPS`
            // guards below already produce, so flooring only avoids the log-case
            // `r²·log r` second-derivative blow-up at exact r=0.
            let r = r.max(R_EPS);
            let (phi, q, t) = if let (Some(length_scale), Some(coeffs)) =
                (length_scale, coeffs.as_ref())
            {
                let jets =
                    duchon_radial_jets(r, length_scale, p_order, s_order as usize, dim, coeffs)?;
                (jets.phi, jets.q, jets.t)
            } else {
                let (phi, phi_r, phi_rr) = duchon_kernel_radial_triplet(
                    r,
                    length_scale,
                    p_order,
                    s_order,
                    dim,
                    coeffs.as_ref(),
                )?;
                let q = if r > R_EPS { phi_r / r } else { phi_rr };
                let t = if r > R_EPS {
                    (phi_rr - q) / (r * r)
                } else {
                    0.0
                };
                (phi, q, t)
            };
            if !phi.is_finite() || !q.is_finite() || !t.is_finite() {
                crate::bail_invalid_basis!(
                    "non-finite Duchon collocation operator derivative at (colloc {i}, center {j}), r={r}"
                );
            }
            d0_raw[[i, j]] = scale_i * phi;
            if build_d2 {
                for axis_a in 0..dim {
                    let h_a = collocation_points[[i, axis_a]] - centers[[j, axis_a]];
                    let w_a = metric_weights
                        .as_ref()
                        .map(|weights| weights[axis_a])
                        .unwrap_or(1.0);
                    for axis_b in 0..dim {
                        let h_b = collocation_points[[i, axis_b]] - centers[[j, axis_b]];
                        let w_b = metric_weights
                            .as_ref()
                            .map(|weights| weights[axis_b])
                            .unwrap_or(1.0);
                        let diagonal = if axis_a == axis_b { q * w_a } else { 0.0 };
                        let mixed = if r > R_EPS {
                            t * w_a * h_a * w_b * h_b
                        } else {
                            0.0
                        };
                        let value = diagonal + mixed;
                        let row_i = (i * dim + axis_a) * dim + axis_b;
                        d2_raw[[row_i, j]] = scale_i * value;
                    }
                }
            }
            if build_d1 && r > R_EPS {
                for axis in 0..dim {
                    let delta = collocation_points[[i, axis]] - centers[[j, axis]];
                    let axis_scale = metric_weights
                        .as_ref()
                        .map(|weights| weights[axis])
                        .unwrap_or(1.0);
                    d1_raw[[i * dim + axis, j]] = scale_i * q * axis_scale * delta;
                }
            }
        }
    }
    let d0_kernel = fast_ab(&d0_raw, &z);
    let poly = polynomial_block_from_order(centers, nullspace_order);
    let kernel_cols = d0_kernel.ncols();
    let poly_cols = poly.ncols();
    let total_cols = kernel_cols + poly_cols;
    // The polynomial block is the unpenalized Duchon null space, left zero before
    // the outer identifiability transform (these operators feed only penalty
    // construction). Orders the caller skipped stay empty (0 rows).
    let mut d0 = Array2::<f64>::zeros((p_colloc, total_cols));
    d0.slice_mut(s![.., 0..kernel_cols]).assign(&d0_kernel);
    let mut d1 = Array2::<f64>::zeros((if build_d1 { p_colloc * dim } else { 0 }, total_cols));
    if build_d1 {
        d1.slice_mut(s![.., 0..kernel_cols])
            .assign(&fast_ab(&d1_raw, &z));
    }
    let mut d2 =
        Array2::<f64>::zeros((if build_d2 { p_colloc * dim * dim } else { 0 }, total_cols));
    if build_d2 {
        d2.slice_mut(s![.., 0..kernel_cols])
            .assign(&fast_ab(&d2_raw, &z));
    }
    if let Some(z) = identifiability_transform {
        let z = z.to_owned();
        d0 = fast_ab(&d0, &z);
        d1 = fast_ab(&d1, &z);
        d2 = fast_ab(&d2, &z);
    }
    Ok(CollocationOperatorMatrices {
        d0,
        d1,
        d2,
        collocation_points: collocation_points.to_owned(),
        kernel_nullspace_transform: Some(z),
        polynomial_block_cols: poly_cols,
    })
}


#[inline(always)]
fn bessel_k0_stable(x: f64) -> f64 {
    let x_pos = x.max(1e-300);
    if x_pos <= 2.0 {
        return bessel_k0_small_series(x_pos);
    }
    let y = 2.0 / x_pos;
    (-x_pos).exp() / x_pos.sqrt()
        * (1.253_314_14
            + y * (-0.078_323_58
                + y * (0.021_895_68
                    + y * (-0.010_624_46
                        + y * (0.005_878_72 + y * (-0.002_515_40 + y * 0.000_532_08))))))
}


#[inline(always)]
fn bessel_k1_stable(x: f64) -> f64 {
    let x_pos = x.max(1e-300);
    if x_pos <= 2.0 {
        return bessel_k1_small_series(x_pos);
    }
    let y = 2.0 / x_pos;
    (-x_pos).exp() / x_pos.sqrt()
        * (1.253_314_14
            + y * (0.234_986_19
                + y * (-0.036_556_20
                    + y * (0.015_042_68
                        + y * (-0.007_803_53 + y * (0.003_256_14 + y * -0.000_682_45))))))
}


#[inline(always)]
fn bessel_k0_k1_small_series(x: f64) -> (f64, f64) {
    const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
    let y = 0.25 * x * x;
    let log_half_plus_gamma = 0.5 * y.ln() + EULER_GAMMA;
    let mut i0 = 1.0;
    let mut i1 = 0.5 * x;
    let mut harmonic = 0.0;
    let mut y_power_over_fact_sq = 1.0;
    let mut k0_series = 0.0;
    let mut k0_series_y_derivative_times_y = 0.0;
    for k in 1..=256 {
        let kf = k as f64;
        harmonic += 1.0 / kf;
        y_power_over_fact_sq *= y / (kf * kf);
        let k0_term = harmonic * y_power_over_fact_sq;
        k0_series += k0_term;
        k0_series_y_derivative_times_y += kf * k0_term;
        i0 += y_power_over_fact_sq;
        i1 += 0.5 * x * y_power_over_fact_sq / (kf + 1.0);
        if k0_term.abs() <= f64::EPSILON * i0.abs().max(k0_series.abs()).max(1.0) {
            break;
        }
    }

    let k0 = -log_half_plus_gamma * i0 + k0_series;
    let k1 = i0 / x + log_half_plus_gamma * i1 - (2.0 / x) * k0_series_y_derivative_times_y;
    (k0, k1)
}


#[inline(always)]
fn bessel_k0_small_series(x: f64) -> f64 {
    bessel_k0_k1_small_series(x).0
}


#[inline(always)]
fn bessel_k1_small_series(x: f64) -> f64 {
    bessel_k0_k1_small_series(x).1
}


const DUCHON_DERIVATIVE_R_FLOOR_REL: f64 = 1e-5;

const DUCHON_COLLISION_TAYLOR_REL: f64 = 1e-4;


/// Minimum `(row, center)` pair count before a radial design sweep builds a
/// certified [`radial_profile::RadialProfile`] instead of evaluating every
/// pair exactly. The profile build costs a few hundred exact jet
/// evaluations, so it only pays for itself when the sweep reuses it well
/// beyond that; below the threshold the exact path keeps small fits
/// bit-identical to the pre-profile behavior.
const RADIAL_PROFILE_MIN_PAIRS: usize = 16_384;


#[inline(always)]
fn duchon_p_from_nullspace_order(order: DuchonNullspaceOrder) -> usize {
    match order {
        // Duchon null spaces contain all polynomials of degree < m.
        // The public `order` knob chooses that polynomial degree cutoff:
        //   order=0 -> constants only  -> m=1
        //   order=1 -> constants+linear -> m=2
        DuchonNullspaceOrder::Zero => 1,
        DuchonNullspaceOrder::Linear => 2,
        DuchonNullspaceOrder::Degree(degree) => degree + 1,
    }
}


/// Returns the effective Duchon null-space order, auto-degrading when the
/// requested order leaves no radial kernel degrees of freedom.
///
/// The constrained kernel block has `centers.nrows() - rank(P)` columns, where
/// `P` is the polynomial null-space block. A valid polynomial block with
/// exactly as many centers as columns is still useless for smoothing: every
/// center is consumed by the side constraints and the design collapses to the
/// polynomial tail. Degrade to the highest lower null-space order with at
/// least one constrained kernel column.
fn duchon_effective_nullspace_order(
    centers: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
) -> DuchonNullspaceOrder {
    if order == DuchonNullspaceOrder::Zero {
        return order;
    }
    let mut effective = order;
    while effective != DuchonNullspaceOrder::Zero
        && centers.nrows() <= polynomial_block_from_order(centers, effective).ncols()
    {
        effective = duchon_previous_nullspace_order(effective);
    }
    if effective != order {
        // Dedup: warn only once per (rows, cols, requested_order) per process.
        // BFGS × P-IRLS × derivative callsites hit this path many times.
        static SEEN: std::sync::OnceLock<
            std::sync::Mutex<std::collections::HashSet<(usize, usize, DuchonNullspaceOrder)>>,
        > = std::sync::OnceLock::new();
        let seen = SEEN.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
        let key = (centers.nrows(), centers.ncols(), order);
        let fresh = seen.lock().map(|mut s| s.insert(key)).unwrap_or(true);
        if fresh {
            let requested_cols = polynomial_block_from_order(centers, order).ncols();
            let effective_cols = polynomial_block_from_order(centers, effective).ncols();
            log::warn!(
                "Duchon nullspace order={:?} in dim={} with {} centers leaves no radial kernel columns (polynomial_cols={}); degrading to {:?} (polynomial_cols={})",
                order,
                centers.ncols(),
                centers.nrows(),
                requested_cols,
                effective,
                effective_cols
            );
        }
    }
    effective
}


#[inline(always)]
fn gamma_lanczos(x: f64) -> f64 {
    // Numerical Recipes / Lanczos approximation with reflection formula.
    const G: f64 = 7.0;
    const P: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pix = std::f64::consts::PI * x;
        return std::f64::consts::PI / (pix.sin() * gamma_lanczos(1.0 - x));
    }
    let z = x - 1.0;
    let mut a = P[0];
    for (i, coeff) in P.iter().enumerate().skip(1) {
        a += coeff / (z + i as f64);
    }
    let t = z + G + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * a
}


#[inline(always)]
fn bessel_k_integer_order(n: usize, z: f64) -> f64 {
    let zz = z.max(1e-300);
    if n == 0 {
        return bessel_k0_stable(zz);
    }
    if n == 1 {
        return bessel_k1_stable(zz);
    }
    let mut km1 = bessel_k0_stable(zz);
    let mut k = bessel_k1_stable(zz);
    for m in 1..n {
        let kp1 = km1 + 2.0 * (m as f64) * k / zz;
        km1 = k;
        k = kp1;
    }
    k
}


#[inline(always)]
fn bessel_k_half_integer_order(l: usize, z: f64) -> f64 {
    // Exact closed-form seeds and the stable upward recurrence
    //   K_{1/2}(z) = sqrt(π/(2z))·e^{−z},
    //   K_{3/2}(z) = K_{1/2}(z)·(1 + 1/z),
    //   K_{ν+1}(z) = K_{ν−1}(z) + (2ν/z)·K_ν(z)   (ν = 1/2 + m, m ≥ 1).
    // Equivalent to the closed-form polynomial sum, but uses EXACT integer
    // coefficients via the recurrence instead of approximate Lanczos-gamma
    // values for `c_j = (l+j)!/(j!(l−j)!)`. The Lanczos approximation is
    // accurate to ~1 ULP at integer arguments; that error gets amplified
    // through catastrophic cancellation in derivative lattices of the
    // r^μ·K_μ(κr) family. Matching the [`BesselKLadder`] arithmetic byte-
    // for-byte also ensures the ladder/per-call paths agree exactly.
    let zz = z.max(1e-300);
    let k_half = (std::f64::consts::PI / (2.0 * zz)).sqrt() * (-zz).exp();
    if l == 0 {
        return k_half;
    }
    let mut km1 = k_half;
    let mut k = k_half * (1.0 + 1.0 / zz);
    for m in 1..l {
        let nu = m as f64 + 0.5;
        let kp1 = km1 + 2.0 * nu * k / zz;
        km1 = k;
        k = kp1;
    }
    k
}


#[inline(always)]
fn bessel_k_real_half_integer_or_integer(nu_abs: f64, z: f64) -> Result<f64, BasisError> {
    let two_nu = (2.0 * nu_abs).round();
    if (two_nu - 2.0 * nu_abs).abs() > 1e-12 {
        crate::bail_invalid_basis!(
            "unsupported Bessel-K order ν={nu_abs}; only integer/half-integer orders are supported"
        );
    }
    let two_nu_i = two_nu as i64;
    if two_nu_i % 2 == 0 {
        let n = (two_nu_i / 2).max(0) as usize;
        Ok(bessel_k_integer_order(n, z))
    } else {
        let l = ((two_nu_i - 1) / 2).max(0) as usize;
        Ok(bessel_k_half_integer_order(l, z))
    }
}


/// Precomputed coefficient for `polyharmonic_kernel` that depends only on
/// `m` and `k_dim`, not on `r`.  Avoids repeated gamma_lanczos calls in the
/// hot kernel evaluation loop (called n × k times per basis build).
#[derive(Clone, Copy)]
struct PolyharmonicBlockCoeff {
    c: f64,
    power: f64,
    is_log_case: bool,
}


impl PolyharmonicBlockCoeff {
    fn new(m: f64, k_dim: usize) -> Self {
        assert!(
            m.is_finite() && m > 0.0,
            "PolyharmonicBlockCoeff::new: m must be finite and > 0, got {m}"
        );
        let k_half = 0.5 * k_dim as f64;
        let power = 2.0 * m - k_dim as f64;
        // Log case: k_dim is even and `2m − k_dim` is a non-negative even
        // integer (within ε). For fractional `m` this never fires; for
        // integer `m` it matches the original integer modulo check exactly.
        const LOG_EPS: f64 = 1e-12;
        let two_m = 2.0 * m;
        let is_log_case = k_dim.is_multiple_of(2) && {
            let n_f = (power / 2.0).round();
            n_f >= 0.0 && (n_f * 2.0 - power).abs() < LOG_EPS
        };
        if is_log_case {
            let m_int = m.round() as i64;
            let m_minus_half_d_plus_one = (m - k_half + 1.0).round() as i64;
            let c = polyharmonic_log_sign(m_int as usize, k_dim)
                / (2.0_f64.powi((two_m.round() as i32) - 1)
                    * std::f64::consts::PI.powf(k_half)
                    * gamma_lanczos(m)
                    * gamma_lanczos(m_minus_half_d_plus_one as f64));
            Self {
                c,
                power,
                is_log_case: true,
            }
        } else {
            let c = gamma_lanczos(k_half - m)
                / (4.0_f64.powf(m) * std::f64::consts::PI.powf(k_half) * gamma_lanczos(m));
            Self {
                c,
                power,
                is_log_case: false,
            }
        }
    }

    #[inline(always)]
    fn eval(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return self.origin_limit();
        }
        if self.is_log_case {
            self.c * r.powf(self.power) * r.max(1e-300).ln()
        } else {
            self.c * r.powf(self.power)
        }
    }

    #[inline(always)]
    fn origin_limit(&self) -> f64 {
        if self.is_log_case {
            log_power_origin_limit(self.c, self.power, 1.0, 0.0)
        } else {
            log_power_origin_limit(self.c, self.power, 0.0, 1.0)
        }
    }
}


fn polyharmonic_kernel(r: f64, m: f64, k_dim: usize) -> f64 {
    PolyharmonicBlockCoeff::new(m, k_dim).eval(r)
}


#[inline(always)]
fn signed_infinity(sign: f64) -> f64 {
    if sign.is_sign_negative() {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    }
}


#[inline(always)]
fn log_power_origin_limit(coeff: f64, exponent: f64, log_coeff: f64, pure_coeff: f64) -> f64 {
    if log_coeff == 0.0 && pure_coeff == 0.0 {
        return 0.0;
    }
    if exponent > 0.0 {
        return 0.0;
    }
    if exponent == 0.0 {
        if log_coeff != 0.0 {
            signed_infinity(-coeff * log_coeff)
        } else {
            coeff * pure_coeff
        }
    } else if log_coeff != 0.0 {
        signed_infinity(-coeff * log_coeff)
    } else {
        signed_infinity(coeff * pure_coeff)
    }
}


#[inline(always)]
fn polyharmonic_log_sign(m: usize, k_dim: usize) -> f64 {
    assert!(
        k_dim.is_multiple_of(2),
        "polyharmonic_log_sign requires even kernel dimension: k_dim={k_dim}, m={m}"
    );
    (-1.0_f64).powi(m as i32 - (k_dim as i32 / 2) + 1)
}


#[inline(always)]
fn duchon_matern_block(
    r: f64,
    kappa: f64,
    n_order: usize,
    k_dim: usize,
) -> Result<f64, BasisError> {
    let n = n_order as f64;
    let k_half = 0.5 * k_dim as f64;
    let nu = n - k_half;
    let nu_abs = nu.abs();
    let c = kappa.powf(k_half - n)
        / ((2.0 * std::f64::consts::PI).powf(k_half) * 2.0_f64.powf(n - 1.0) * gamma_lanczos(n));
    if r <= 0.0 {
        if nu > 0.0 {
            // r^ν K_ν(κr) → 2^(ν−1) Γ(ν) κ^(−ν) as r→0+.
            return Ok(c * 2.0_f64.powf(nu - 1.0) * gamma_lanczos(nu) * kappa.powf(-nu));
        }
        // ν ≤ 0: c·r^ν·K_|ν|(κr) is divergent at r=0 (logarithmically for ν=0,
        // power-law for ν<0). The hybrid-kernel diagonal must be evaluated via
        // duchon_hybrid_kernel_collision_value, which sums the divergent
        // Matérn and polyharmonic blocks so the singularities cancel exactly
        // (guaranteed by the PFD identity when 2(p+s) > d).
        crate::bail_invalid_basis!(
            "Duchon Matérn block at r=0 with ν={nu} ≤ 0 is divergent; \
             evaluate the hybrid kernel diagonal via the collision routine"
        );
    }
    let z = (kappa * r).max(1e-300);
    let k_nu = bessel_k_real_half_integer_or_integer(nu_abs, z)?;
    Ok(c * r.powf(nu) * k_nu)
}


#[inline(always)]
fn polyharmonic_kernel_triplet(
    r: f64,
    m: f64,
    k_dim: usize,
) -> Result<(f64, f64, f64), BasisError> {
    let (value, first, second, _, _) = polyharmonic_block_jet4(r, m, k_dim)?;
    Ok((value, first, second))
}


#[inline(always)]
fn falling_factorial(alpha: f64, order: usize) -> f64 {
    (0..order).fold(1.0, |acc, idx| acc * (alpha - idx as f64))
}


#[inline(always)]
fn falling_factorial_derivative(alpha: f64, order: usize) -> f64 {
    if order == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for omit in 0..order {
        let mut term = 1.0;
        for idx in 0..order {
            if idx != omit {
                term *= alpha - idx as f64;
            }
        }
        total += term;
    }
    total
}


/// Unified radial jet for one polyharmonic partial-fraction block.
///
/// Returns (φ, φ', φ'', φ''', φ'''') from a single consistent evaluation,
/// sharing normalization constant, r_safe, and log_r. This eliminates the
/// possibility of numerical drift between the triplet and higher-order
/// derivative paths.
fn polyharmonic_block_jet4(
    r: f64,
    m: f64,
    k_dim: usize,
) -> Result<(f64, f64, f64, f64, f64), BasisError> {
    if !r.is_finite() || r < 0.0 {
        crate::bail_invalid_basis!("polyharmonic distance must be finite and non-negative");
    }
    assert!(
        m.is_finite() && m > 0.0,
        "polyharmonic_block_jet4: m must be finite and > 0, got {m}"
    );

    let k_half = 0.5 * k_dim as f64;
    let alpha = 2.0 * m - k_dim as f64;
    // Log case: k_dim even and `2m − k_dim` is a non-negative even integer
    // (within ε). For fractional `m` this never fires.
    const LOG_EPS: f64 = 1e-12;
    let is_log_case = k_dim.is_multiple_of(2) && {
        let n_f = (alpha / 2.0).round();
        n_f >= 0.0 && (n_f * 2.0 - alpha).abs() < LOG_EPS
    };
    if is_log_case {
        let m_int = m.round() as usize;
        let c = polyharmonic_log_sign(m_int, k_dim)
            / (2.0_f64.powi((2 * m_int - 1) as i32)
                * std::f64::consts::PI.powf(k_half)
                * gamma_lanczos(m)
                * gamma_lanczos((m_int - k_dim / 2 + 1) as f64));
        let mut out = [0.0; 5];
        for d in 0..5 {
            let e = alpha - d as f64;
            let ff = falling_factorial(alpha, d);
            let ff_d = falling_factorial_derivative(alpha, d);
            out[d] = if r <= 0.0 {
                log_power_origin_limit(c, e, ff, ff_d)
            } else {
                c * r.powf(e) * (ff * r.ln() + ff_d)
            };
        }
        return Ok((out[0], out[1], out[2], out[3], out[4]));
    }

    let c = gamma_lanczos(k_half - m)
        / (4.0_f64.powf(m) * std::f64::consts::PI.powf(k_half) * gamma_lanczos(m));
    let mut out = [0.0; 5];
    for d in 0..5 {
        let e = alpha - d as f64;
        let ff = falling_factorial(alpha, d);
        out[d] = if r <= 0.0 {
            log_power_origin_limit(c, e, 0.0, ff)
        } else {
            c * ff * r.powf(e)
        };
    }
    Ok((out[0], out[1], out[2], out[3], out[4]))
}


#[inline(always)]
fn log_power_family_derivative(exponent: f64, log_coeff: f64, pure_coeff: f64) -> (f64, f64, f64) {
    (
        exponent - 1.0,
        exponent * log_coeff,
        exponent * pure_coeff + log_coeff,
    )
}


#[inline(always)]
fn log_power_family_value(
    r: f64,
    coeff: f64,
    exponent: f64,
    log_coeff: f64,
    pure_coeff: f64,
) -> f64 {
    if r <= 0.0 {
        log_power_origin_limit(coeff, exponent, log_coeff, pure_coeff)
    } else {
        coeff * r.powf(exponent) * (log_coeff * r.ln() + pure_coeff)
    }
}


#[inline(always)]
fn duchon_polyharmonic_operator_block_jets(
    r: f64,
    m: f64,
    k_dim: usize,
) -> Result<(f64, f64, f64, f64), BasisError> {
    if !r.is_finite() || r < 0.0 {
        crate::bail_invalid_basis!("polyharmonic distance must be finite and non-negative");
    }
    assert!(
        m.is_finite() && m > 0.0,
        "duchon_polyharmonic_operator_block_jets: m must be finite and > 0, got {m}"
    );

    let k_half = 0.5 * k_dim as f64;
    let alpha = 2.0 * m - k_dim as f64;
    // Log case: k_dim even and `2m − k_dim` is a non-negative even integer
    // (within ε). For fractional `m` this never fires; for integer `m` it
    // matches the original `k_dim % 2 == 0 && m >= k_dim / 2` check.
    const LOG_EPS: f64 = 1e-12;
    let is_log_case = k_dim.is_multiple_of(2) && {
        let n_f = (alpha / 2.0).round();
        n_f >= 0.0 && (n_f * 2.0 - alpha).abs() < LOG_EPS
    };
    let (c, phi_log_coeff, phi_pure_coeff) = if is_log_case {
        let m_int = m.round() as usize;
        (
            polyharmonic_log_sign(m_int, k_dim)
                / (2.0_f64.powi((2 * m_int - 1) as i32)
                    * std::f64::consts::PI.powf(k_half)
                    * gamma_lanczos(m)
                    * gamma_lanczos((m_int - k_dim / 2 + 1) as f64)),
            1.0,
            0.0,
        )
    } else {
        (
            gamma_lanczos(k_half - m)
                / (4.0_f64.powf(m) * std::f64::consts::PI.powf(k_half) * gamma_lanczos(m)),
            0.0,
            1.0,
        )
    };

    let (phi_r_exp, phi_r_log, phi_r_pure) =
        log_power_family_derivative(alpha, phi_log_coeff, phi_pure_coeff);
    let q_exp = phi_r_exp - 1.0;
    let q = log_power_family_value(r, c, q_exp, phi_r_log, phi_r_pure);

    let (q_r_exp_raw, q_r_log, q_r_pure) =
        log_power_family_derivative(q_exp, phi_r_log, phi_r_pure);
    let t_exp = q_r_exp_raw - 1.0;
    let t = log_power_family_value(r, c, t_exp, q_r_log, q_r_pure);

    let (t_r_exp, t_r_log, t_r_pure) = log_power_family_derivative(t_exp, q_r_log, q_r_pure);
    let t_r = log_power_family_value(r, c, t_r_exp, t_r_log, t_r_pure);

    let (t_rr_exp, t_rr_log, t_rr_pure) = log_power_family_derivative(t_r_exp, t_r_log, t_r_pure);
    let t_rr = log_power_family_value(r, c, t_rr_exp, t_rr_log, t_rr_pure);

    Ok((q, t, t_r, t_rr))
}


/// Shared Bessel-K ladder for one evaluation point `z = κ·r`.
///
/// Every Matérn partial-fraction block and every term of its radial
/// derivative lattice consumes `K_ν(z)` at orders from ONE parity class
/// (integer when the covariate dimension is even, half-integer when odd),
/// differing by integers — and all at the SAME `z`. The previous code
/// restarted the `K₀/K₁` (or closed-form half-integer) seed evaluation and
/// the upward recurrence inside every per-term Bessel call: hundreds of
/// redundant seed+recurrence runs per `(row, center)` pair, which the #979
/// CTN stage-1 stack profile showed to be the dominant cost of every Duchon
/// κ-trial at scale. One ladder per point replaces all of them: two seed
/// evaluations plus the standard upward recurrence
/// `K_{ν+1}(z) = K_{ν−1}(z) + (2ν/z)·K_ν(z)`, which is the numerically
/// STABLE direction for `K` (it grows with ν). For integer orders this is
/// arithmetic-identical to the old per-call `bessel_k_integer_order`, which
/// ran the same seeds and recurrence internally; for half-integer orders the
/// recurrence is exact and replaces the per-order closed-form sum.
struct BesselKLadder {
    /// `values[i] = K_{base + i}(z)` with `base ∈ {0, ½}`.
    values: SmallVec<[f64; 16]>,
    half_integer: bool,
}


impl BesselKLadder {
    fn build(z: f64, half_integer: bool, max_order_steps: usize) -> Self {
        let zz = z.max(1e-300);
        let mut values: SmallVec<[f64; 16]> = SmallVec::with_capacity(max_order_steps + 2);
        if half_integer {
            // K_{1/2}(z) = √(π/(2z))·e^{−z};  K_{3/2}(z) = K_{1/2}(z)·(1 + 1/z).
            let k_half = (std::f64::consts::PI / (2.0 * zz)).sqrt() * (-zz).exp();
            values.push(k_half);
            values.push(k_half * (1.0 + 1.0 / zz));
        } else {
            values.push(bessel_k0_stable(zz));
            values.push(bessel_k1_stable(zz));
        }
        let base = if half_integer { 0.5 } else { 0.0 };
        for i in 1..max_order_steps {
            let nu = base + i as f64;
            let next = values[i - 1] + 2.0 * nu * values[i] / zz;
            values.push(next);
        }
        Self {
            values,
            half_integer,
        }
    }

    /// `K_{|order|}(z)` from the ladder (`K_{−ν} = K_ν`).
    #[inline]
    fn k_abs(&self, order_abs: f64) -> f64 {
        let base = if self.half_integer { 0.5 } else { 0.0 };
        let idx = (order_abs - base).round() as usize;
        self.values[idx]
    }
}


/// Radial-derivative jets of the Matérn family `coeff·r^μ·K_μ(κr)` up to
/// order `max_j ≤ 4`, evaluated against a shared [`BesselKLadder`].
///
/// Exact recurrence derived from `d/dr[r^ν K_ν(κr)]` and the Bessel identity
/// `dK_ν/dz = −K_{ν−1}(z) − (ν/z)K_ν(z)`:
///
///   g⁽⁰⁾ = c · r^ν · K_ν(z)
///   g⁽¹⁾ = −c · κ · r^ν · K_{ν−1}(z)
///   g⁽²⁾ = c·κ² r^ν K_{ν−2} − c·κ r^{ν−1} K_{ν−1}, ...
///
/// Same derivative lattice as the per-order reference implementation
/// `duchon_matern_family_radial_derivative_reference` (kept in the test
/// module as the equivalence oracle)
/// (term-for-term, in the same order), but: (a) the lattice is expanded
/// incrementally once instead of rebuilt from scratch per derivative order,
/// (b) terms live in a fixed-capacity stack buffer instead of per-call heap
/// `Vec`s (≤ 2^max_j ≤ 16 terms), and (c) every Bessel factor is an indexed
/// ladder read instead of a fresh seed+recurrence evaluation. Only orders
/// `0..=max_j` are computed — the q-family consumes order 0 only and the
/// t-family orders ≤ 2, where the old path always expanded to order 4 and
/// discarded the tail.
fn duchon_matern_family_jets_with_ladder(
    r: f64,
    kappa: f64,
    coeff: f64,
    mu: f64,
    max_j: usize,
    ladder: &BesselKLadder,
    out: &mut [f64],
) -> Result<(), BasisError> {
    if max_j > 4 || out.len() <= max_j {
        crate::bail_invalid_basis!(
            "Duchon Matérn-family ladder jets support derivative orders 0..=4 with an output slot per order"
        );
    }
    if r <= 0.0 {
        out[..=max_j].fill(0.0);
        if mu > 0.0 {
            out[0] = coeff * 2.0_f64.powf(mu - 1.0) * gamma_lanczos(mu) * kappa.powf(-mu);
        }
        return Ok(());
    }
    let mut terms: SmallVec<[DuchonMaternDerivativeTerm; 16]> =
        smallvec![DuchonMaternDerivativeTerm {
            coeff,
            kappa_power: 0,
            r_power: mu,
            bessel_order: mu,
        }];
    for (j, slot) in out.iter_mut().enumerate().take(max_j + 1) {
        if j > 0 {
            let mut next: SmallVec<[DuchonMaternDerivativeTerm; 16]> =
                SmallVec::with_capacity(terms.len() * 2);
            for term in &terms {
                let stay_coeff = term.coeff * (term.r_power - term.bessel_order);
                if stay_coeff != 0.0 {
                    next.push(DuchonMaternDerivativeTerm {
                        coeff: stay_coeff,
                        kappa_power: term.kappa_power,
                        r_power: term.r_power - 1.0,
                        bessel_order: term.bessel_order,
                    });
                }
                next.push(DuchonMaternDerivativeTerm {
                    coeff: -term.coeff,
                    kappa_power: term.kappa_power + 1,
                    r_power: term.r_power,
                    bessel_order: term.bessel_order - 1.0,
                });
            }
            terms = next;
        }
        let mut value = KahanSum::default();
        for term in &terms {
            if term.coeff == 0.0 {
                continue;
            }
            value.add(
                term.coeff
                    * kappa.powi(term.kappa_power as i32)
                    * r.powf(term.r_power)
                    * ladder.k_abs(term.bessel_order.abs()),
            );
        }
        *slot = value.sum();
    }
    Ok(())
}


/// Maximum ladder steps (`K_base ..= K_{base+steps}`) needed by the q/t
/// operator families of Matérn block `n` in dimension `k_dim`: the q-family
/// reads `K_{|ν−1|}` and the t-family `K_{|ν−2−j|}` for `j ≤ 2`, ν = n − d/2.
fn duchon_matern_block_max_ladder_steps(n_order: usize, k_dim: usize) -> usize {
    let nu = n_order as f64 - 0.5 * k_dim as f64;
    let candidates = [
        (nu - 1.0).abs(),
        (nu - 2.0).abs(),
        (nu - 3.0).abs(),
        (nu - 4.0).abs(),
    ];
    let max_abs = candidates.iter().copied().fold(0.0_f64, f64::max);
    max_abs.floor() as usize + 1
}


fn duchon_matern_operator_block_jets_with_ladder(
    r: f64,
    kappa: f64,
    n_order: usize,
    k_dim: usize,
    ladder: &BesselKLadder,
) -> Result<(f64, f64, f64, f64), BasisError> {
    if r <= 0.0 {
        return Ok((0.0, 0.0, 0.0, 0.0));
    }
    let n = n_order as f64;
    let k_half = 0.5 * k_dim as f64;
    let nu = n - k_half;
    let c = kappa.powf(k_half - n)
        / ((2.0 * std::f64::consts::PI).powf(k_half) * 2.0_f64.powf(n - 1.0) * gamma_lanczos(n));

    let mut q_out = [0.0_f64; 1];
    duchon_matern_family_jets_with_ladder(r, kappa, -c * kappa, nu - 1.0, 0, ladder, &mut q_out)?;
    let mut t_out = [0.0_f64; 3];
    duchon_matern_family_jets_with_ladder(
        r,
        kappa,
        c * kappa * kappa,
        nu - 2.0,
        2,
        ladder,
        &mut t_out,
    )?;
    Ok((q_out[0], t_out[0], t_out[1], t_out[2]))
}


#[inline(always)]
fn pure_duchon_block_order(p_order: usize, s_order: f64) -> f64 {
    p_order as f64 + s_order
}


fn validate_duchon_kernel_orders(
    length_scale: Option<f64>,
    p_order: usize,
    s_order: f64,
    k_dim: usize,
) -> Result<(), BasisError> {
    if k_dim == 0 {
        crate::bail_invalid_basis!("Duchon basis requires at least one covariate dimension");
    }
    if let Some(scale) = length_scale
        && (!scale.is_finite() || scale <= 0.0)
    {
        crate::bail_invalid_basis!("Duchon hybrid length_scale must be finite and positive");
    }
    // Two independent well-posedness conditions on (p, s, d) for pure Duchon.
    //
    // (1) CPD-vs-nullspace adequacy — gated below on `length_scale.is_none()`.
    //     The pure-polyharmonic kernel of effective order m = p+s in R^d is
    //     phi(r) = r^{2m-d}, or r^{2m-d}·log r when 2m-d is a non-negative
    //     even integer (the "log case", reached precisely when d is even
    //     and m >= d/2). Wendland's Theorem 8.17 / 8.18 give its
    //     conditional-positive-definiteness order:
    //
    //         d odd,  exponent half-integer:  ceil((2m-d)/2) = m - (d-1)/2
    //         d even, log case:               (2m-d)/2 + 1   = m - d/2 + 1
    //
    //     Duchon interpolation with polynomial nullspace P_p (polynomials
    //     of degree < p) is uniquely solvable iff the kernel's CPD order
    //     does not exceed p. Substituting m = p + s:
    //
    //         d odd:  s <= (d-1)/2     <=>  2s <= d - 1
    //         d even: s <= d/2 - 1     <=>  2s <= d - 2
    //
    //     Both branches collapse to `2s < d` once we use that s and d are
    //     integers and 2s is therefore even (so `2s = d - 1` is impossible
    //     for even d, and `2s <= d - 2` is just `2s < d`).
    //
    //     Counter-example admitted if this guard is dropped: d=2, p=1, s=1
    //     passes the spectral check (2(1+1)=4 > 2) and builds the TPS
    //     kernel c·r²·log r against a constants-only nullspace P_1; the
    //     interpolation form is not PD on lambda perp P_1 and the fitted
    //     penalty is meaningless.
    //
    //     The hybrid (Matérn-blended) Duchon kernel sidesteps this entirely:
    //     the Matérn remainder is strictly positive definite (CPD order 0),
    //     so any P_p suffices — hence the `length_scale.is_none()` gate.
    //
    // (2) Spectral kernel-existence — universal, gated below on the sum.
    //     The pointwise kernel comes from the inverse Fourier of
    //     1/|xi|^{2(p+s)}, which is a finite distribution at the origin
    //     iff `2(p+s) > d`. Below that threshold the radial kernel value
    //     diverges and there is nothing to evaluate.
    if !s_order.is_finite() || s_order < 0.0 {
        crate::bail_invalid_basis!("Duchon spectral power must be finite and ≥ 0; got s={s_order}");
    }
    if length_scale.is_none() && 2.0 * s_order >= k_dim as f64 {
        crate::bail_invalid_basis!(
            "pure Duchon requires power < dimension/2 for nullspace degree < {p_order}; got power={s_order}, dimension={k_dim}"
        );
    }
    let spectral_order = 2.0 * (p_order as f64 + s_order);
    if spectral_order <= k_dim as f64 {
        crate::bail_invalid_basis!(
            "Duchon pointwise kernel values require 2*(p+s) > dimension; got 2*(p+s)={spectral_order}, dimension={k_dim}, p={p_order}, s={s_order}"
        );
    }
    Ok(())
}


fn validate_duchon_collocation_orders(
    length_scale: Option<f64>,
    p_order: usize,
    s_order: f64,
    k_dim: usize,
    max_operator_derivative_order: usize,
) -> Result<(), BasisError> {
    // Kernel-level conditions (existence + CPD/nullspace adequacy) come first;
    // the operator-level conditions below build on a pointwise-valid kernel.
    validate_duchon_kernel_orders(length_scale, p_order, s_order, k_dim)?;
    // The spectral_order > k_dim + k checks below are C^k-at-origin
    // conditions: for the polyharmonic kernel r^{2(p+s)-d} (or the log
    // variant) to admit k-th radial derivatives in the distributional sense
    // — and therefore for k-th-order derivative *collocation* of the
    // kernel against centers to produce a finite operator — we need its
    // exponent to clear the next k orders of differentiation at the
    // origin. Equivalently: 2(p+s) - d > k.
    //
    // Note these are independent of the CPD/nullspace check. The penalty
    // matrices ultimately built from these collocation operators are of
    // the form S = D_k^T D_k and are PSD by construction; the discipline
    // here is purely about *existence* of D_k itself.
    let spectral_order = 2.0 * (p_order as f64 + s_order);
    if max_operator_derivative_order >= 1 && spectral_order <= k_dim as f64 + 1.0 {
        crate::bail_invalid_basis!(
            "Duchon D1 collocation requires 2*(p+s) > dimension+1; got 2*(p+s)={spectral_order}, dimension={k_dim}, p={p_order}, s={s_order}"
        );
    }
    if max_operator_derivative_order >= 2 && spectral_order <= k_dim as f64 + 2.0 {
        crate::bail_invalid_basis!(
            "Duchon D2 collocation requires 2*(p+s) > dimension+2; got 2*(p+s)={spectral_order}, dimension={k_dim}, p={p_order}, s={s_order}"
        );
    }
    Ok(())
}


#[derive(Debug, Clone)]
pub struct DuchonPartialFractionCoeffs {
    pub(crate) a: Vec<f64>,
    pub(crate) b: Vec<f64>,
}


#[inline(always)]
pub(crate) fn duchon_partial_fraction_coeffs(
    p_order: usize,
    s_order: usize,
    kappa: f64,
) -> DuchonPartialFractionCoeffs {
    // 1/(ρ^{2p}(κ²+ρ²)^s) = Σ a_m/ρ^{2m} + Σ b_n/(κ²+ρ²)^n
    let mut a = vec![0.0_f64; p_order + 1]; // 1-based m
    let mut b = vec![0.0_f64; s_order + 1]; // 1-based n
    if s_order == 0 {
        if p_order > 0 {
            // Pure intrinsic polyharmonic case: no Matérn tail remains, so the
            // spectrum is exactly 1 / ρ^(2p).
            a[p_order] = 1.0;
        }
        return DuchonPartialFractionCoeffs { a, b };
    }
    for m in 1..=p_order {
        let sign = if (p_order - m).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        let expo = -2.0 * (s_order + p_order - m) as f64;
        let comb = binomial_f64(s_order + p_order - m - 1, p_order - m);
        a[m] = sign * kappa.powf(expo) * comb;
    }
    for n in 1..=s_order {
        let sign = if p_order.is_multiple_of(2) { 1.0 } else { -1.0 };
        let expo = -2.0 * (p_order + s_order - n) as f64;
        let comb = if p_order == 0 && n == s_order {
            // p=0 reduces to the pure Matérn block 1/(κ²+ρ²)^s.
            1.0
        } else {
            let top = p_order + s_order - n - 1;
            binomial_f64(top, s_order - n)
        };
        b[n] = sign * kappa.powf(expo) * comb;
    }
    DuchonPartialFractionCoeffs { a, b }
}


fn duchon_matern_kernel_general_from_distance(
    r: f64,
    length_scale: Option<f64>,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: Option<&DuchonPartialFractionCoeffs>,
) -> Result<f64, BasisError> {
    if !r.is_finite() || r < 0.0 {
        crate::bail_invalid_basis!("Duchon kernel distance must be finite and non-negative");
    }
    let Some(length_scale) = length_scale else {
        return Ok(polyharmonic_kernel(
            r,
            pure_duchon_block_order(p_order, s_order as f64),
            k_dim,
        ));
    };
    if !length_scale.is_finite() || length_scale <= 0.0 {
        crate::bail_invalid_basis!("Duchon hybrid length_scale must be finite and positive");
    }
    let kappa = 1.0 / length_scale;

    let coeffs_local;
    let coeffs_ref = if let Some(c) = coeffs {
        c
    } else {
        coeffs_local = duchon_partial_fraction_coeffs(p_order, s_order, kappa);
        &coeffs_local
    };
    let collision_taylor_radius = DUCHON_COLLISION_TAYLOR_REL * length_scale.max(1e-8);
    // The near-collision Taylor expansion uses phi(0) plus even-order
    // derivative collision limits. Those limits only exist when the kernel
    // is finite at the origin, i.e. when 2(p+s) > d. Below that threshold
    // the partial-fraction blocks individually diverge at r=0 but their
    // sum is still a well-defined function for any r > 0 (each Bessel-K
    // and r^{2m-d}-type block is finite away from origin). Fall through
    // to the direct sum in that regime; r=0 itself remains an error.
    let kernel_finite_at_origin = 2 * (p_order + s_order) > k_dim;
    if r <= collision_taylor_radius && kernel_finite_at_origin {
        return duchon_hybrid_kernel_near_collision_value(
            r,
            length_scale,
            p_order,
            s_order,
            k_dim,
            coeffs_ref,
        );
    }
    let mut val = KahanSum::default();
    for (m, coeff) in coeffs_ref.a.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        val.add(coeff * polyharmonic_kernel(r, (m) as f64, k_dim));
    }
    for (n, coeff) in coeffs_ref.b.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        val.add(coeff * duchon_matern_block(r, kappa, n, k_dim)?);
    }
    Ok(val.sum())
}


fn duchon_hybrid_kernel_collision_value(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<f64, BasisError> {
    let spectral_order = 2 * (p_order + s_order);
    if spectral_order <= k_dim {
        crate::bail_invalid_basis!(
            "Duchon hybrid diagonal is not finite when 2*(p+s) <= dimension; got 2*(p+s)={spectral_order}, dimension={k_dim}, p={p_order}, s={s_order}"
        );
    }

    let kappa = 1.0 / length_scale.max(1e-300);
    let mut pure = KahanSum::default();
    let mut log_part = KahanSum::default();
    for (m, &a_m) in coeffs.a.iter().enumerate().skip(1) {
        if a_m == 0.0 {
            continue;
        }
        let (block_pure, block_log) = duchon_polyharmonic_block_taylor_r2j(m, k_dim, 0);
        pure.add(a_m * block_pure);
        log_part.add(a_m * block_log);
    }
    for (n, &b_n) in coeffs.b.iter().enumerate().skip(1) {
        if b_n == 0.0 {
            continue;
        }
        let (block_pure, block_log) = duchon_matern_block_taylor_r2j(kappa, n, k_dim, 0);
        pure.add(b_n * block_pure);
        log_part.add(b_n * block_log);
    }
    let value = pure.sum();
    let log_value = log_part.sum();
    if log_value.abs() > 1e-8 * value.abs().max(1e-30) {
        crate::bail_invalid_basis!(
            "Duchon hybrid diagonal log terms did not cancel: log={log_value:.6e}, value={value:.6e}; p={p_order}, s={s_order}, d={k_dim}"
        );
    }
    if !value.is_finite() {
        crate::bail_invalid_basis!(
            "non-finite Duchon hybrid diagonal value for p={p_order}, s={s_order}, d={k_dim}"
        );
    }
    Ok(value)
}


fn duchon_hybrid_kernel_near_collision_value(
    r: f64,
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<f64, BasisError> {
    let mut value =
        duchon_hybrid_kernel_collision_value(length_scale, p_order, s_order, k_dim, coeffs)?;
    if r == 0.0 {
        return Ok(value);
    }

    // Radial Taylor expansion about the center collision:
    //
    //   phi(r) = phi(0)
    //          + phi''(0) r^2 / 2
    //          + phi''''(0) r^4 / 24
    //          + phi''''''(0) r^6 / 720 + ...
    //
    // Odd terms vanish for an isotropic radial kernel. A finite 2q-th
    // derivative at zero requires spectral smoothness 2(p+s) > d + 2q.
    // Terms whose collision derivative does not exist are omitted; this is
    // still strictly better than evaluating the raw partial-fraction sum at a
    // tiny nonzero radius, where large singular components cancel only after
    // losing many digits.
    let smoothness_order = 2 * (p_order + s_order);
    let r2 = r * r;
    if smoothness_order > k_dim + 2 {
        let (phi_rr, _, _) =
            duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, coeffs)?;
        value += 0.5 * phi_rr * r2;
    }
    if smoothness_order > k_dim + 4 {
        let phi_rrrr = duchon_phi_rrrr_collision(length_scale, p_order, s_order, k_dim, coeffs)?;
        value += (1.0 / 24.0) * phi_rrrr * r2 * r2;
    }
    if smoothness_order > k_dim + 6 {
        let phi_rrrrrr =
            duchon_phi_rrrrrr_collision(length_scale, p_order, s_order, k_dim, coeffs)?;
        value += (1.0 / 720.0) * phi_rrrrrr * r2 * r2 * r2;
    }
    if !value.is_finite() {
        crate::bail_invalid_basis!(
            "non-finite Duchon hybrid near-collision value at r={r}, p={p_order}, s={s_order}, d={k_dim}"
        );
    }
    Ok(value)
}


#[inline(always)]
fn stable_euclidean_norm<I>(components: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut scale = 0.0_f64;
    let mut sumsq = 1.0_f64;
    let mut has_nonzero = false;
    for component in components {
        let abs = component.abs();
        if abs == 0.0 {
            continue;
        }
        if !abs.is_finite() {
            return f64::INFINITY;
        }
        if !has_nonzero {
            scale = abs;
            has_nonzero = true;
            continue;
        }
        if scale < abs {
            let ratio = scale / abs;
            sumsq = 1.0 + sumsq * ratio * ratio;
            scale = abs;
        } else {
            let ratio = abs / scale;
            sumsq += ratio * ratio;
        }
    }
    if has_nonzero {
        scale * sumsq.sqrt()
    } else {
        0.0
    }
}


#[inline]
fn centered_aniso_log_scale_mean(eta: &[f64]) -> f64 {
    if eta.len() <= 1 {
        0.0
    } else {
        eta.iter().sum::<f64>() / eta.len() as f64
    }
}


#[inline]
fn centered_aniso_log_scale(value: f64, mean: f64) -> f64 {
    (value - mean).clamp(-50.0, 50.0)
}


#[inline]
fn aniso_axis_scale(value: f64, mean: f64) -> f64 {
    centered_aniso_log_scale(value, mean).exp()
}


#[inline]
fn aniso_metric_weight(value: f64, mean: f64) -> f64 {
    (2.0 * centered_aniso_log_scale(value, mean)).exp()
}


fn centered_aniso_metric_weights(eta: &[f64]) -> Vec<f64> {
    let mean = centered_aniso_log_scale_mean(eta);
    eta.iter()
        .map(|&value| aniso_metric_weight(value, mean))
        .collect()
}


/// Compute anisotropic squared distance components and total distance.
///
/// This is the core of **geometric anisotropy**: a linear warp Λ = diag(κ_a)
/// turns ellipsoidal correlation contours into isotropic ones. Writing h = x − c,
/// z = Λh, the anisotropic distance is r = |z| = |Λh|.
///
/// We decompose Λ = κ · A where det(A) = 1, parameterized as
///   ψ_a = ψ̄ + η_a,   Σ η_a = 0
/// where ψ̄ is the global scale (existing scalar κ) and η_a are d−1 anisotropy
/// contrasts. This separates scale from shape and preserves the Duchon scaling
/// law φ(r;κ) = κ^δ H(κr) for the global part.
///
/// Given per-axis log-scales `eta`, the identifiable centered contrasts are
/// ψ_a = eta_a - mean(eta). The metric uses those contrasts so Σ_a ψ_a = 0
/// even when a caller passes an uncentered vector:
///
///   r = √( Σ_a exp(2·ψ_a) · (x_a - c_a)² )
///
/// Returns `(r, s_vec)` where `s_vec[a] = exp(2·ψ_a) · h_a²` is the
/// per-axis weighted squared displacement. These components are needed for
/// per-axis derivatives: `∂φ/∂ψ_a = q · s_a`.
///
/// The derivative chain through r gives:
///   ∇_ψ r      = s / r
///   ∇²_ψ r     = (2/r) Diag(s) − (1/r³) ss'
/// which is diagonal + rank-1, so Hessian-vector products are O(d).
#[inline]
fn aniso_distance_and_components(data_row: &[f64], center: &[f64], eta: &[f64]) -> (f64, Vec<f64>) {
    assert_eq!(data_row.len(), center.len());
    assert_eq!(data_row.len(), eta.len());
    let d = data_row.len();
    let eta_mean = centered_aniso_log_scale_mean(eta);
    let mut s_vec = Vec::with_capacity(d);
    let mut scaled_components = Vec::with_capacity(d);
    for a in 0..d {
        let h_a = data_row[a] - center[a];
        // Clamp exp(2ψ) to avoid overflow/underflow: ψ in [-50, 50].
        let scale_a = aniso_axis_scale(eta[a], eta_mean);
        let scaled_h_a = scale_a * h_a;
        let s_a = scaled_h_a * scaled_h_a;
        scaled_components.push(scaled_h_a);
        s_vec.push(s_a);
    }
    (stable_euclidean_norm(scaled_components), s_vec)
}


/// Compute anisotropic distance without returning per-axis components.
///
/// This is the lightweight version of [`aniso_distance_and_components`] for
/// call sites that only need the scalar distance `r`.
#[inline]
fn aniso_distance(data_row: &[f64], center: &[f64], eta: &[f64]) -> f64 {
    assert_eq!(data_row.len(), center.len());
    assert_eq!(data_row.len(), eta.len());
    let eta_mean = centered_aniso_log_scale_mean(eta);
    stable_euclidean_norm(
        (0..data_row.len()).map(|a| aniso_axis_scale(eta[a], eta_mean) * (data_row[a] - center[a])),
    )
}


#[inline(always)]
fn euclidean_distance_rows(
    lhs: ArrayView2<'_, f64>,
    lhs_row: usize,
    rhs: ArrayView2<'_, f64>,
    rhs_row: usize,
) -> f64 {
    assert_eq!(lhs.ncols(), rhs.ncols());
    stable_euclidean_norm((0..lhs.ncols()).map(|axis| lhs[[lhs_row, axis]] - rhs[[rhs_row, axis]]))
}


#[inline(always)]
fn aniso_axis_scales(eta: &[f64]) -> Vec<f64> {
    let eta_mean = centered_aniso_log_scale_mean(eta);
    eta.iter()
        .map(|&value| aniso_axis_scale(value, eta_mean))
        .collect()
}


#[inline(always)]
fn aniso_distance_rows_with_scales(
    lhs: ArrayView2<'_, f64>,
    lhs_row: usize,
    rhs: ArrayView2<'_, f64>,
    rhs_row: usize,
    axis_scales: &[f64],
) -> f64 {
    assert_eq!(lhs.ncols(), rhs.ncols());
    assert_eq!(lhs.ncols(), axis_scales.len());
    stable_euclidean_norm(
        (0..lhs.ncols())
            .map(|axis| axis_scales[axis] * (lhs[[lhs_row, axis]] - rhs[[rhs_row, axis]])),
    )
}


fn fill_symmetric_from_row_kernel<F>(matrix: &mut Array2<f64>, kernel: F) -> Result<(), BasisError>
where
    F: Fn(usize, usize) -> Result<f64, BasisError> + Sync,
{
    assert_eq!(matrix.nrows(), matrix.ncols());
    let k = matrix.nrows();
    // The kernels passed here are pure functions of the (symmetric) pairwise
    // center distance, so `kernel(i, j) == kernel(j, i)`. Evaluate only the
    // upper triangle (including the diagonal) in parallel — each row task
    // touches only its own `j >= i` cells, so the borrows stay disjoint — then
    // mirror into the lower triangle. This halves the (sqrt + special-function)
    // kernel evaluations relative to filling every cell independently, with no
    // change to the resulting matrix (still exactly symmetric).
    matrix
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| {
            for j in i..k {
                row[j] = kernel(i, j)?;
            }
            Ok::<(), BasisError>(())
        })?;
    for i in 1..k {
        for j in 0..i {
            matrix[[i, j]] = matrix[[j, i]];
        }
    }
    Ok(())
}


/// Return y-space points `y_{i,a} = exp(ψ_a) x_{i,a}` with
/// `ψ_a = η_a - mean(η)` so Euclidean pairwise
/// distances in y equal anisotropic kernel distances in x:
///   |y_i - y_j|² = Σ_a exp(2 ψ_a) (x_{i,a} - x_{j,a})² = aniso_distance²(x_i, x_j, η).
/// Use this before `pairwise_distance_bounds` whenever κ conditioning
/// bounds must match the kernel's actual metric (anisotropic case). For
/// isotropic terms, pass `None` and keep using the raw centers.
pub(crate) fn points_in_aniso_y_space(points: ArrayView2<'_, f64>, eta: &[f64]) -> Array2<f64> {
    assert_eq!(points.ncols(), eta.len());
    let mut y = points.to_owned();
    let eta_mean = centered_aniso_log_scale_mean(eta);
    let weights: Vec<f64> = eta.iter().map(|&e| aniso_axis_scale(e, eta_mean)).collect();
    for a in 0..eta.len() {
        let w_a = weights[a];
        y.column_mut(a).mapv_inplace(|v| v * w_a);
    }
    y
}


/// Compute per-axis standard deviations of knot center coordinates.
///
/// Returns σ_a for each axis column of `centers`. Axes with zero variance
/// (constant column) get σ_a = 1.0. All values are clamped to [1e-6, 1e6].
pub fn knot_cloud_axis_scales(centers: ArrayView2<'_, f64>) -> Vec<f64> {
    let k = centers.nrows();
    let d = centers.ncols();
    if k < 2 || d == 0 {
        return vec![1.0; d];
    }
    let n = k as f64;
    let mut scales = Vec::with_capacity(d);
    for a in 0..d {
        let col = centers.column(a);
        let mean = col.sum() / n;
        let var = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let sigma = var.sqrt();
        // If variance is zero (constant column), use 1.0 (no scaling).
        let sigma = if sigma < 1e-12 { 1.0 } else { sigma };
        scales.push(sigma.clamp(1e-6, 1e6));
    }
    scales
}


/// Compute initial anisotropy contrasts η_a from knot center geometry.
///
/// Returns η_a = −ln(σ_a) + (1/d) Σ_b ln(σ_b), which satisfies Ση_a = 0
/// by construction. Axes with more spread get negative η_a (smaller κ_a,
/// longer correlation range), axes with less spread get positive η_a.
///
/// If d ≤ 1, returns an empty vector (anisotropy is meaningless for 1-D).
pub fn initial_aniso_contrasts(centers: ArrayView2<'_, f64>) -> Vec<f64> {
    let d = centers.ncols();
    if d <= 1 {
        return Vec::new();
    }
    let scales = knot_cloud_axis_scales(centers);
    let mean_neg_log: f64 = scales.iter().map(|&s| -s.ln()).sum::<f64>() / d as f64;
    // η_a = −ln(σ_a) + (1/d) Σ_b ln(σ_b)
    //     = −ln(σ_a) − mean(−ln(σ_b))
    //     = neg_log_scales[a] − mean(neg_log_scales)
    scales
        .iter()
        .map(|&scale| -scale.ln() - mean_neg_log)
        .collect()
}


/// Pure forward transform of the supplied anisotropy log-scales: subtract the
/// mean (so Σ η = 0) and zero tiny residuals. `None` (or a 1-D problem, where
/// centering is a no-op) means *no* anisotropy.
///
/// This is a **continuous function of η with no hidden data dependence**: an
/// explicit all-zero vector centers to all-zero, i.e. the isotropic metric
/// (weights `exp(2·0) = 1`, Euclidean radius). It is therefore identical, as a
/// design, to the `None` path through `η = 0`, and is continuous across it —
/// `[1e-9, -1e-9]` and `[0, 0]` map to neighboring designs, not a jump.
///
/// The Matérn input-location jet/Hessian (`matern_metric_weights`, the public
/// `matern_input_location_first_jet`/`_hessian` FFI) and the `UserProvided`-center
/// forward design both apply *this* transform, so the jet differentiates exactly
/// the function the public design evaluates (#437), and an explicit isotropic
/// request reduces to the closed-form isotropic Matérn kernel rather than a
/// data-driven anisotropic one (#1042).
///
/// Auto-initialization of `η` from knot-cloud geometry is a *separate* concern
/// handled by [`auto_seed_aniso_contrasts`]; it is reserved for callers that
/// opt into data-derived geometry (the κ-optimizer's data-driven center
/// strategies and the pure-Duchon `scale_dims` path), selected by
/// [`resolve_matern_forward_aniso`].
fn centered_aniso_contrasts(aniso: Option<&[f64]>) -> Option<Vec<f64>> {
    use crate::terms::smooth::center_aniso_log_scales as center;

    match aniso {
        Some(v) if v.len() > 1 => Some(center(v)),
        Some(v) => Some(v.to_vec()),
        None => None,
    }
}


/// Auto-seed anisotropy contrasts from knot-cloud geometry for callers that use
/// an all-zero vector as the "initialize me" sentinel.
///
/// Used by (a) the pure-Duchon `scale_dims` path, where `η` is a FIXED,
/// geometry-derived basis parameter that is never enrolled as a REML hyper-axis
/// (see `spatial_term_supports_hyper_optimization`): "standardize the geometry,
/// then learn the smoothness"; and (b) the Matérn forward design when the term
/// uses a **data-driven** center strategy, i.e. the κ-optimizer's seeding
/// sentinel (the optimizer's analytic ψ-gradient is computed against the same
/// auto-seeded design, so the pair stays consistent). A non-zero (or absent)
/// vector is honored verbatim (centered, exactly like [`centered_aniso_contrasts`]);
/// only an *exactly* all-zero vector is replaced by `initial_aniso_contrasts(centers)`.
///
/// A `UserProvided`-center Matérn term does NOT use this — its geometry is fully
/// caller-specified, so an explicit all-zero η must be honored literally; folding
/// the geometry seed into that path made the public design discontinuous at
/// `η = 0` and hijacked explicit isotropic requests (#1042).
fn auto_seed_aniso_contrasts(
    centers: ArrayView2<'_, f64>,
    aniso: Option<&[f64]>,
) -> Option<Vec<f64>> {
    use crate::terms::smooth::center_aniso_log_scales as center;

    let eta = match aniso {
        Some(v) if v.len() > 1 => v,
        Some(v) => return Some(v.to_vec()),
        None => return None,
    };
    let all_zero = eta.iter().all(|&e| e == 0.0);
    if !all_zero {
        return Some(center(eta));
    }
    let contrasts = initial_aniso_contrasts(centers);
    if contrasts.is_empty() {
        Some(center(eta))
    } else {
        Some(center(&contrasts))
    }
}


/// How the Matérn forward design build interprets an *exactly all-zero*
/// `aniso_log_scales` vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnisoSeedMode {
    /// All-zero `η` is the κ-optimizer / `scale_dims` seeding sentinel: replace
    /// it with geometry-derived contrasts from the knot cloud
    /// ([`auto_seed_aniso_contrasts`]). This is the default for every internal
    /// build entry; the optimizer's analytic ψ-gradient is computed against the
    /// same auto-seeded design, so value/gradient stay consistent. Note that by
    /// the time the κ-optimizer rebuilds a frozen design the center strategy has
    /// usually been resolved to `UserProvided`, so center provenance cannot be
    /// used to distinguish this from a genuine literal request — the mode must
    /// be carried explicitly.
    AutoSeedFromGeometry,
    /// All-zero `η` is an explicit isotropic request and is honored literally
    /// ([`centered_aniso_contrasts`]): the design reduces to the closed-form
    /// isotropic Matérn and varies continuously through `η = 0`. The public
    /// `matern_basis` FFI (and its input-location jet/Hessian) selects this so a
    /// caller's explicit isotropic request is not hijacked into a data-driven
    /// anisotropic kernel (#1042).
    Literal,
}


/// Resolve the anisotropy contrasts the Matérn forward design build applies,
/// dispatching on the explicit [`AnisoSeedMode`].
fn resolve_matern_forward_aniso(
    mode: AnisoSeedMode,
    centers: ArrayView2<'_, f64>,
    aniso: Option<&[f64]>,
) -> Option<Vec<f64>> {
    match mode {
        AnisoSeedMode::Literal => centered_aniso_contrasts(aniso),
        AnisoSeedMode::AutoSeedFromGeometry => auto_seed_aniso_contrasts(centers, aniso),
    }
}


pub(crate) fn pairwise_distance_bounds(points: ArrayView2<'_, f64>) -> Option<(f64, f64)> {
    let n = points.nrows();
    let d = points.ncols();
    if n < 2 || d == 0 {
        return None;
    }
    let mut r_min = f64::INFINITY;
    let mut r_max = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let r = stable_euclidean_norm((0..d).map(|c| points[[i, c]] - points[[j, c]]));
            if r.is_finite() && r > 0.0 {
                r_min = r_min.min(r);
                r_max = r_max.max(r);
            }
        }
    }
    if r_min.is_finite() && r_max.is_finite() && r_min > 0.0 && r_max > 0.0 {
        Some((r_min, r_max))
    } else {
        None
    }
}


/// Capped-sample pairwise distance bounds for large point clouds.
///
/// Returns `(r_min_hat, r_max_hat)` such that:
/// - `r_max_hat <= true r_max`  (pairwise max over a sub-sample is monotone
///    in the sample, so the sampled max underestimates the true max).
/// - `r_min_hat >= true r_min`  (pairwise min over a sub-sample can only
///    exclude some pairs, so the sampled min overestimates the true min).
///
/// Both approximations are conservative for κ-bound derivation:
///   kappa_lo = 1e-2 / r_max_hat  >=  1e-2 / true r_max  (wider window, low κ)
///   kappa_hi = 1e2  / r_min_hat  <=  1e2  / true r_min  (tighter window, high κ)
/// so no feasible κ that the exact bound would include is excluded by the
/// approximation — it can only slightly shrink the high-κ tail, which is
/// exactly the regime (κ → ∞ ⇒ degenerate kernel) that we want the outer
/// optimizer to avoid anyway.
///
/// Sampling is deterministic stride (points indexed 0, stride, 2·stride, …).
/// For a cap of `K = 1024` and n up to ~10⁹ this yields O(K²·d) work per
/// call — a few hundred μs. For n < K the exact pairwise is used.
pub(crate) fn pairwise_distance_bounds_sampled(points: ArrayView2<'_, f64>) -> Option<(f64, f64)> {
    const K_CAP: usize = 1024;
    let n = points.nrows();
    let d = points.ncols();
    if n < 2 || d == 0 {
        return None;
    }
    if n <= K_CAP {
        return pairwise_distance_bounds(points);
    }
    // Deterministic stride sampling: pick K_CAP evenly spaced indices.
    // This preserves any spatial stratification already present in the
    // data ordering (large-scale data is typically in insertion order, not
    // spatially stratified, so stride sampling is effectively uniform).
    let stride = n / K_CAP;
    let k = K_CAP; // exactly K_CAP samples by construction (stride rounds down)
    let mut r_min = f64::INFINITY;
    let mut r_max = 0.0_f64;
    for i_idx in 0..k {
        let i = i_idx * stride;
        for j_idx in (i_idx + 1)..k {
            let j = j_idx * stride;
            let r = stable_euclidean_norm((0..d).map(|c| points[[i, c]] - points[[j, c]]));
            if r.is_finite() && r > 0.0 {
                r_min = r_min.min(r);
                r_max = r_max.max(r);
            }
        }
    }
    if r_min.is_finite() && r_max.is_finite() && r_min > 0.0 && r_max > 0.0 {
        Some((r_min, r_max))
    } else {
        None
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ConstraintNullspaceCacheKey {
    centersrows: usize,
    centers_cols: usize,
    centers_hash: u64,
    order: ConstraintNullspaceOrderKey,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ConstraintNullspaceOrderKey {
    Duchon(DuchonNullspaceOrder),
    ThinPlate,
}


#[derive(Default, Clone, Debug)]
struct ConstraintNullspaceCache {
    map: HashMap<ConstraintNullspaceCacheKey, Arc<Array2<f64>>>,
    order: Vec<ConstraintNullspaceCacheKey>,
}


const CONSTRAINT_NULLSPACE_CACHE_MAX_ENTRIES: usize = 32;


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct OwnedDataCacheKey {
    rows: usize,
    cols: usize,
    ptr: usize,
    stride0: isize,
    stride1: isize,
}


#[derive(Debug)]
struct BasisCacheContext {
    constraint_nullspace: ConstraintNullspaceCache,
    owned_data: crate::resource::ByteLruCache<OwnedDataCacheKey, Arc<Array2<f64>>>,
}


impl BasisCacheContext {
    fn with_policy(policy: &crate::resource::ResourcePolicy) -> Self {
        Self {
            constraint_nullspace: ConstraintNullspaceCache::default(),
            owned_data: crate::resource::ByteLruCache::with_max_entries(
                policy.max_owned_data_cache_bytes,
                crate::resource::OWNED_DATA_CACHE_MAX_ENTRIES,
            ),
        }
    }
}


impl Default for BasisCacheContext {
    fn default() -> Self {
        Self::with_policy(&crate::resource::ResourcePolicy::default_library())
    }
}


/// Explicit per-run workspace for reusable basis-construction caches.
///
/// Pass one workspace through repeated basis builds to avoid global mutable state
/// and to keep caching scoped to a caller-controlled lifecycle.
///
/// Owned-data cache entries are byte-limited via the
/// [`crate::resource::ResourcePolicy`] provided at construction; use
/// [`BasisWorkspace::with_policy`] for large-scale workloads where a single
/// entry can be multiple gigabytes.
#[derive(Debug)]
pub struct BasisWorkspace {
    cache: BasisCacheContext,
    policy: crate::resource::ResourcePolicy,
}


impl BasisWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_policy(policy: crate::resource::ResourcePolicy) -> Self {
        Self {
            cache: BasisCacheContext::with_policy(&policy),
            policy,
        }
    }

    pub fn default_library() -> Self {
        Self::with_policy(crate::resource::ResourcePolicy::default_library())
    }

    /// Returns the resource policy this workspace was configured with.
    pub fn policy(&self) -> &crate::resource::ResourcePolicy {
        &self.policy
    }
}


impl Default for BasisWorkspace {
    fn default() -> Self {
        Self::default_library()
    }
}


fn hash_arrayview2(values: ArrayView2<'_, f64>) -> u64 {
    let mut hasher = DefaultHasher::new();
    values.nrows().hash(&mut hasher);
    values.ncols().hash(&mut hasher);
    for v in values {
        v.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}


fn shared_owned_data_matrix(
    data: ArrayView2<'_, f64>,
    cache: &BasisCacheContext,
) -> Arc<Array2<f64>> {
    let key = OwnedDataCacheKey {
        rows: data.nrows(),
        cols: data.ncols(),
        ptr: data.as_ptr() as usize,
        stride0: data.strides()[0],
        stride1: data.strides()[1],
    };
    if let Some(hit) = cache.owned_data.get(&key) {
        return hit;
    }

    let owned = Arc::new(data.to_owned());
    if let Some(hit) = cache.owned_data.get(&key) {
        return hit;
    }

    cache.owned_data.insert(key, owned.clone());
    owned
}


/// Minimal cache-less intern: wraps an `ArrayView2` into an `Arc<Array2<f64>>`.
///
/// Used by derivative-operator builders that don't have a `BasisCacheContext`
/// in scope (e.g. `build_aniso_design_psi_derivatives_shared`). The goal is the
/// same as `shared_owned_data_matrix`: move the owned payload into an `Arc`
/// once so that downstream `StreamingRadialState` copies share it via
/// `Arc::clone` instead of materializing a fresh n×d `Array2<f64>` per axis.
#[inline]
fn shared_owned_data_matrix_from_view(data: ArrayView2<'_, f64>) -> Arc<Array2<f64>> {
    Arc::new(data.to_owned())
}


/// Minimal cache-less intern for knot centers; mirrors
/// `shared_owned_data_matrix_from_view`. Centers are typically k×d with k
/// much smaller than n, but the `Arc::clone` pattern still avoids a k×d
/// copy per axis when the same operator feeds multiple derivative paths.
#[inline]
fn shared_owned_centers_matrix_from_view(centers: ArrayView2<'_, f64>) -> Arc<Array2<f64>> {
    Arc::new(centers.to_owned())
}


/// Compute the kernel reparameterisation transform `Z = null(P_centers^T)`.
///
/// `Z` is a `(k, k − C(d+r, r))` orthonormal matrix whose columns span the
/// null space of the polynomial side-condition system.  Reparameterising the
/// radial kernel coefficients as `α = Z γ` enforces `P_centers^T α = 0` and
/// reduces the kernel column count from `k` to `k − C(d+r, r)`.
///
/// After this projection the polynomial block `P_data` is appended as separate
/// explicit unpenalized columns (see `build_duchon_basis_designwithworkspace`),
/// so the pre-identifiability total width is always `k` (equal to the center
/// count), regardless of the polynomial null-space dimension.
///
/// This is the step that absorbs the full `C(d+r, r)`-dimensional polynomial
/// null space.  The subsequent `spatial_parametric_constraint_block` step only
/// removes the intercept.
fn kernel_constraint_nullspace(
    centers: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
    cache: &mut BasisCacheContext,
) -> Result<Array2<f64>, BasisError> {
    let effective_order = duchon_effective_nullspace_order(centers, order);
    let degraded = effective_order != order;
    let key = ConstraintNullspaceCacheKey {
        centersrows: centers.nrows(),
        centers_cols: centers.ncols(),
        centers_hash: hash_arrayview2(centers),
        order: ConstraintNullspaceOrderKey::Duchon(effective_order),
    };

    if let Some(hit) = cache.constraint_nullspace.map.get(&key) {
        return Ok((**hit).clone());
    }

    let p_k = polynomial_block_from_order(centers, effective_order);
    let z = Arc::new(kernel_constraint_nullspace_from_matrix(p_k.view()).map_err(|err| {
        if degraded {
            BasisError::InvalidInput(format!(
                "Duchon degraded from order={:?} to order={:?} due to insufficient centers ({} in dim={}); order={:?} construction then failed: {err}",
                order,
                effective_order,
                centers.nrows(),
                centers.ncols(),
                effective_order,
            ))
        } else {
            err
        }
    })?);

    if let Some(hit) = cache.constraint_nullspace.map.get(&key) {
        return Ok((**hit).clone());
    }
    cache.constraint_nullspace.map.insert(key, z.clone());
    cache.constraint_nullspace.order.push(key);
    while cache.constraint_nullspace.map.len() > CONSTRAINT_NULLSPACE_CACHE_MAX_ENTRIES {
        if cache.constraint_nullspace.order.is_empty() {
            break;
        }
        let oldkey = cache.constraint_nullspace.order.remove(0);
        cache.constraint_nullspace.map.remove(&oldkey);
    }

    Ok((*z).clone())
}


fn thin_plate_kernel_constraint_nullspace(
    centers: ArrayView2<'_, f64>,
    cache: &mut BasisCacheContext,
) -> Result<Array2<f64>, BasisError> {
    let key = ConstraintNullspaceCacheKey {
        centersrows: centers.nrows(),
        centers_cols: centers.ncols(),
        centers_hash: hash_arrayview2(centers),
        order: ConstraintNullspaceOrderKey::ThinPlate,
    };

    if let Some(hit) = cache.constraint_nullspace.map.get(&key) {
        return Ok((**hit).clone());
    }

    let p_k = thin_plate_polynomial_block(centers);
    if centers.nrows() < p_k.ncols() {
        crate::bail_invalid_basis!(
            "thin-plate spline requires at least {} centers to span the degree-{} polynomial null space in dimension {}; got {}",
            p_k.ncols(),
            thin_plate_polynomial_degree(centers.ncols()),
            centers.ncols(),
            centers.nrows()
        );
    }
    let (z, rank) =
        rrqr_nullspace_basis(&p_k, default_rrqr_rank_alpha()).map_err(BasisError::LinalgError)?;
    if rank != p_k.ncols() {
        crate::bail_invalid_basis!(
            "thin-plate spline polynomial block is rank deficient at the selected centers: expected rank {}, got {}; choose geometrically independent centers for dimension {}",
            p_k.ncols(),
            rank,
            centers.ncols()
        );
    }
    let z = Arc::new(z);

    if let Some(hit) = cache.constraint_nullspace.map.get(&key) {
        return Ok((**hit).clone());
    }
    cache.constraint_nullspace.map.insert(key, z.clone());
    cache.constraint_nullspace.order.push(key);
    while cache.constraint_nullspace.map.len() > CONSTRAINT_NULLSPACE_CACHE_MAX_ENTRIES {
        if cache.constraint_nullspace.order.is_empty() {
            break;
        }
        let oldkey = cache.constraint_nullspace.order.remove(0);
        cache.constraint_nullspace.map.remove(&oldkey);
    }

    Ok((*z).clone())
}


fn matern_identifiability_transform(
    centers: ArrayView2<'_, f64>,
    identifiability: &MaternIdentifiability,
) -> Result<Option<Array2<f64>>, BasisError> {
    let k = centers.nrows();
    match identifiability {
        MaternIdentifiability::None => Ok(None),
        MaternIdentifiability::CenterSumToZero => {
            let q = Array2::<f64>::ones((k, 1));
            Ok(Some(kernel_constraint_nullspace_from_matrix(q.view())?))
        }
        MaternIdentifiability::CenterLinearOrthogonal => {
            // Mirror the Duchon path: auto-degrade to Zero (constant-only) when
            // there aren't enough centers to affinely span [1, x_1, ..., x_d].
            // kernel_constraint_nullspace_from_matrix would otherwise hard-error
            // via rrqr_nullspace_basis when centers.nrows() < d + 1.
            let effective_order =
                duchon_effective_nullspace_order(centers, DuchonNullspaceOrder::Linear);
            let q = polynomial_block_from_order(centers, effective_order);
            Ok(Some(kernel_constraint_nullspace_from_matrix(q.view())?))
        }
        MaternIdentifiability::FrozenTransform { transform, .. } => {
            if transform.nrows() != k {
                crate::bail_dim_basis!(
                    "frozen Matérn identifiability transform mismatch: centers={k}, transform rows={}",
                    transform.nrows()
                );
            }
            Ok(Some(transform.clone()))
        }
    }
}


fn build_matern_operator_penalty_candidates(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    z_opt: Option<&Array2<f64>>,
    aniso_log_scales: Option<&[f64]>,
) -> Result<Vec<PenaltyCandidate>, BasisError> {
    let ops = build_matern_collocation_operator_matrices(
        centers,
        None,
        length_scale,
        nu,
        include_intercept,
        z_opt.map(|z| z.view()),
        aniso_log_scales,
    )?;
    // Gate the operator dials on the Matérn-ν RKHS smoothness so a rough kernel
    // (e.g. ν=1/2) is not over-smoothed by a higher-order roughness penalty its
    // own RKHS norm does not control (#707).
    let matern_spec = DuchonOperatorPenaltySpec::matern_for_smoothness(nu, centers.ncols());
    Ok(operator_penalty_candidates_from_collocation(
        &ops.d0,
        &ops.d1,
        &ops.d2,
        &matern_spec,
    ))
}


/// Decide whether the matern double-penalty path emits the
/// `DoublePenaltyNullspace` shrinkage candidate, honoring a FROZEN bootstrap-κ
/// decision when one is present (gam#787/#860). `frozen` is
/// `MaternIdentifiability::FrozenTransform`'s `nullspace_shrinkage_survived`:
/// `Some(b)` forces the answer (so the learned-penalty count stays invariant as
/// the κ-optimizer rebuilds the design), `None` falls back to the κ-dependent
/// spectral test (the cold-build / non-frozen behavior). Returns the emitted
/// candidate list together with the realized decision so the caller can record
/// it into the basis metadata for the freeze step.
fn matern_double_penalty_candidates_with_decision(
    primary: &Array2<f64>,
    frozen: Option<bool>,
) -> Result<(Vec<PenaltyCandidate>, bool), BasisError> {
    let mut candidates = vec![normalize_penalty_candidate(
        primary.clone(),
        0,
        PenaltySource::Primary,
    )];
    let survived = match frozen {
        Some(forced) => {
            if forced && let Some(shrinkage) = build_nullspace_shrinkage_penalty(primary)? {
                candidates.push(normalize_penalty_candidate(
                    shrinkage.sym_penalty,
                    0,
                    PenaltySource::DoublePenaltyNullspace,
                ));
                true
            } else {
                // Forced ON but the projected kernel has no near-zero direction
                // at this κ (so there is literally no shrinkage subspace to
                // build), OR forced OFF: emit only the primary kernel penalty.
                // Forced-ON-without-a-subspace cannot manufacture a 7th penalty,
                // but the frozen path only sets `Some(true)` when the bootstrap κ
                // DID find a subspace, and the projected-kernel null space is a
                // geometric property of the centers/transform (κ rescales every
                // eigenvalue together), so the subspace persists across rebuilds.
                false
            }
        }
        None => {
            if let Some(shrinkage) = build_nullspace_shrinkage_penalty(primary)? {
                candidates.push(normalize_penalty_candidate(
                    shrinkage.sym_penalty,
                    0,
                    PenaltySource::DoublePenaltyNullspace,
                ));
                true
            } else {
                false
            }
        }
    };
    Ok((candidates, survived))
}


fn build_matern_double_penalty_candidates(
    spline: &MaternSplineBasis,
    full_transform: Option<&Array2<f64>>,
    frozen_nullspace_shrinkage_survived: Option<bool>,
) -> Result<(Vec<PenaltyCandidate>, bool), BasisError> {
    let primary = project_penalty_matrix(&spline.penalty_kernel, full_transform);
    matern_double_penalty_candidates_with_decision(&primary, frozen_nullspace_shrinkage_survived)
}


/// Creates a Matérn spline basis from data and centers.
///
/// The design is `[K | 1]` when `include_intercept=true` and `[K]` otherwise, where:
/// - `K_ij = k(||x_i - c_j||; length_scale, nu)` is the Matérn kernel block.
///
/// The default kernel penalty is `alpha' S alpha` with `S_jl = k(||c_j - c_l||)`, embedded
/// in the full coefficient space. With intercept included, that column is unpenalized by
/// `penalty_kernel`; optional `penalty_ridge` is a nullspace projector used for
/// double-penalty shrinkage of previously unpenalized directions.
///
/// NOTE: This follows the RKHS Gram construction S = K_CC (not K_CC^{-1}) in
/// coefficient space, with global scaling absorbed by the smoothing parameter λ.
pub fn create_matern_spline_basiswithworkspace(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    aniso_log_scales: Option<&[f64]>,
    workspace: &mut BasisWorkspace,
) -> Result<MaternSplineBasis, BasisError> {
    let n = data.nrows();
    let d = data.ncols();
    let k = centers.nrows();
    let total_cols = k + usize::from(include_intercept);
    let dense_bytes = dense_design_bytes(n, total_cols);
    if dense_bytes > workspace.policy().max_single_materialization_bytes {
        crate::bail_invalid_basis!(
            "Matérn basis dense design exceeds resource policy: n={n}, p={total_cols}, dense={:.1} MiB, cap={:.1} MiB",
            dense_bytes as f64 / (1024.0 * 1024.0),
            workspace.policy().max_single_materialization_bytes as f64 / (1024.0 * 1024.0),
        );
    }

    if d == 0 {
        crate::bail_invalid_basis!("Matérn basis requires at least one covariate dimension");
    }
    if k == 0 {
        crate::bail_invalid_basis!("Matérn basis requires at least one center");
    }
    if centers.ncols() != d {
        crate::bail_dim_basis!(
            "Matérn basis dimension mismatch: data has {d} columns, centers have {}",
            centers.ncols()
        );
    }
    if data.iter().any(|v| !v.is_finite()) || centers.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!("Matérn basis requires finite data and center values");
    }
    validate_matern_length_scale(length_scale)?;
    if let Some(eta) = aniso_log_scales {
        if eta.len() != d {
            crate::bail_dim_basis!(
                "aniso_log_scales length {} does not match data dimension {d}",
                eta.len()
            );
        }
        if eta.iter().any(|v| !v.is_finite()) {
            crate::bail_invalid_basis!("aniso_log_scales must contain finite values");
        }
    }

    // Practical safe operating range for κ from center geometry (document Eq. D.2):
    //   κ in [1e-2 / r_max, 1e2 / r_min], with κ = 1/length_scale.
    // Warn rather than silently clamp so callers keep explicit control.
    // Under anisotropy the kernel metric is y-space (y_a = exp(η_a) x_a), so
    // the relevant r_min/r_max are y-space pairwise distances, not raw.
    let warn_bounds = if let Some(eta) = aniso_log_scales {
        let y_centers = points_in_aniso_y_space(centers, eta);
        pairwise_distance_bounds(y_centers.view())
    } else {
        pairwise_distance_bounds(centers)
    };
    if let Some((r_min, r_max)) = warn_bounds {
        let kappa = 1.0 / length_scale.max(1e-300);
        let kappa_lo = 1e-2 / r_max;
        let kappa_hi = 1e2 / r_min;
        if kappa < kappa_lo || kappa > kappa_hi {
            log::debug!(
                "Matérn κ={} is outside recommended range [{}, {}] derived from centers (r_min={}, r_max={}); kernel conditioning may degrade",
                kappa,
                kappa_lo,
                kappa_hi,
                r_min,
                r_max
            );
        }
    }

    // Distance computation: anisotropic when eta is present, isotropic otherwise.
    // Under anisotropy we work in y-space (y = Ax), so r = |Ah| replaces |h|.
    let mut kernel_block = Array2::<f64>::zeros((n, k));
    let mut center_kernel = Array2::<f64>::zeros((k, k));
    let axis_scales = aniso_log_scales.map(aniso_axis_scales);
    let kernel_result: Result<(), BasisError> = kernel_block
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| {
            for j in 0..k {
                let r = if let Some(scales) = axis_scales.as_deref() {
                    aniso_distance_rows_with_scales(data, i, centers, j, scales)
                } else {
                    euclidean_distance_rows(data, i, centers, j)
                };
                row[j] = matern_kernel_from_distance(r, length_scale, nu)?;
            }
            Ok(())
        });
    kernel_result?;
    // Center-center Gram matrix K_CC. In RKHS form, the kernel penalty on
    // radial coefficients is alpha^T K_CC alpha.
    fill_symmetric_from_row_kernel(&mut center_kernel, |i, j| {
        let r = if let Some(scales) = axis_scales.as_deref() {
            aniso_distance_rows_with_scales(centers, i, centers, j, scales)
        } else {
            euclidean_distance_rows(centers, i, centers, j)
        };
        matern_kernel_from_distance(r, length_scale, nu)
    })?;

    let mut basis = Array2::<f64>::zeros((n, total_cols));
    basis.slice_mut(s![.., 0..k]).assign(&kernel_block);
    if include_intercept {
        basis.column_mut(k).fill(1.0);
    }

    let mut penalty_kernel = Array2::<f64>::zeros((total_cols, total_cols));
    // RKHS coefficient penalty uses the center Gram matrix directly:
    //   S = K_CC  (not K_CC^{-1}).
    // This matches Duchon/Matérn spline theory where alpha^T K_CC alpha is the
    // native-space quadratic form up to a global scaling absorbed by lambda.
    penalty_kernel
        .slice_mut(s![0..k, 0..k])
        .assign(&center_kernel);
    let penalty_ridge = build_nullspace_shrinkage_penalty(&penalty_kernel)?
        .map(|block| block.sym_penalty)
        .unwrap_or_else(|| Array2::<f64>::zeros((total_cols, total_cols)));

    Ok(MaternSplineBasis {
        basis,
        penalty_kernel,
        penalty_ridge,
        num_kernel_basis: k,
        num_polynomial_basis: usize::from(include_intercept),
        dimension: d,
    })
}


#[inline]
fn validate_lat_lon_matrix(
    data: ArrayView2<'_, f64>,
    context: &str,
    radians: bool,
) -> Result<(), BasisError> {
    if data.ncols() != 2 {
        crate::bail_dim_basis!(
            "{context} requires exactly two columns: latitude and longitude; got {}",
            data.ncols()
        );
    }
    if data.nrows() == 0 {
        crate::bail_invalid_basis!("{context} requires at least one row");
    }
    let (lat_lo, lat_hi, unit) = if radians {
        (
            -std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
            "radians",
        )
    } else {
        (-90.0, 90.0, "degrees")
    };
    for (i, row) in data.outer_iter().enumerate() {
        let lat = row[0];
        let lon = row[1];
        if !lat.is_finite() || !lon.is_finite() {
            crate::bail_invalid_basis!(
                "{context} requires finite latitude/longitude; row {i} has ({lat}, {lon})"
            );
        }
        if !(lat_lo..=lat_hi).contains(&lat) {
            crate::bail_invalid_basis!(
                "{context} latitude must be in [{lat_lo}, {lat_hi}] {unit}; row {i} has {lat}"
            );
        }
    }
    Ok(())
}


pub fn spherical_wahba_kernel_matrix(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    penalty_order: usize,
    radians: bool,
) -> Result<Array2<f64>, BasisError> {
    spherical_wahba_kernel_matrix_with_kind(
        data,
        centers,
        penalty_order,
        radians,
        SphereWahbaKernel::Sobolev,
    )
}


pub fn spherical_wahba_kernel_matrix_with_kind(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    penalty_order: usize,
    radians: bool,
    kernel: SphereWahbaKernel,
) -> Result<Array2<f64>, BasisError> {
    validate_lat_lon_matrix(data, "spherical spline data", radians)?;
    validate_lat_lon_matrix(centers, "spherical spline centers", radians)?;
    let n = data.nrows();
    let k = centers.nrows();
    let deg = if radians {
        1.0
    } else {
        std::f64::consts::PI / 180.0
    };
    // Precompute (sin_lat, cos_lat, sin_lon, cos_lon) for each center once.
    // Using cos(lon - lon_c) = cos(lon)·cos(lon_c) + sin(lon)·sin(lon_c)
    // collapses the inner-loop trig from one `.cos()` per (i, j) down to
    // four multiplies and an add — a ~10x speedup on the inner body at
    // large-scale N·K.
    let mut sin_lat_c = Vec::<f64>::with_capacity(k);
    let mut cos_lat_c = Vec::<f64>::with_capacity(k);
    let mut sin_lon_c = Vec::<f64>::with_capacity(k);
    let mut cos_lon_c = Vec::<f64>::with_capacity(k);
    for c in centers.outer_iter() {
        let lat = c[0] * deg;
        let lon = c[1] * deg;
        let (s_lat, c_lat) = lat.sin_cos();
        let (s_lon, c_lon) = lon.sin_cos();
        sin_lat_c.push(s_lat);
        cos_lat_c.push(c_lat);
        sin_lon_c.push(s_lon);
        cos_lon_c.push(c_lon);
    }
    let mut out = Array2::<f64>::zeros((n, k));
    let err_flag = std::sync::atomic::AtomicBool::new(false);
    out.axis_chunks_iter_mut(ndarray::Axis(0), 256)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut block)| {
            use wide::f64x4;
            let row_offset = chunk_idx * 256;
            let chunks = k / 4;
            let tail = k % 4;
            for (local_i, mut out_row) in block.outer_iter_mut().enumerate() {
                let i = row_offset + local_i;
                let lat = data[(i, 0)] * deg;
                let lon = data[(i, 1)] * deg;
                let (sin_lat, cos_lat) = lat.sin_cos();
                let (sin_lon, cos_lon) = lon.sin_cos();
                let sin_lat_v = f64x4::from(sin_lat);
                let cos_lat_v = f64x4::from(cos_lat);
                let sin_lon_v = f64x4::from(sin_lon);
                let cos_lon_v = f64x4::from(cos_lon);
                // SIMD over 4 centers at a time.
                for cidx in 0..chunks {
                    let base = cidx * 4;
                    let sl_c = f64x4::from([
                        sin_lat_c[base],
                        sin_lat_c[base + 1],
                        sin_lat_c[base + 2],
                        sin_lat_c[base + 3],
                    ]);
                    let cl_c = f64x4::from([
                        cos_lat_c[base],
                        cos_lat_c[base + 1],
                        cos_lat_c[base + 2],
                        cos_lat_c[base + 3],
                    ]);
                    let sn_c = f64x4::from([
                        sin_lon_c[base],
                        sin_lon_c[base + 1],
                        sin_lon_c[base + 2],
                        sin_lon_c[base + 3],
                    ]);
                    let cn_c = f64x4::from([
                        cos_lon_c[base],
                        cos_lon_c[base + 1],
                        cos_lon_c[base + 2],
                        cos_lon_c[base + 3],
                    ]);
                    let dlon_cos = cos_lon_v * cn_c + sin_lon_v * sn_c;
                    let cos_gamma = sin_lat_v * sl_c + cos_lat_v * cl_c * dlon_cos;
                    let vals =
                        wahba_sphere_kernel_from_cos_simd_kind(cos_gamma, penalty_order, kernel);
                    let arr = vals.to_array();
                    for lane in 0..4 {
                        if !arr[lane].is_finite() {
                            err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                            return;
                        }
                        out_row[base + lane] = arr[lane];
                    }
                }
                // Scalar tail (0..3 elements).
                let tail_start = chunks * 4;
                for t in 0..tail {
                    let j = tail_start + t;
                    let dlon_cos = cos_lon * cos_lon_c[j] + sin_lon * sin_lon_c[j];
                    let cos_gamma = sin_lat * sin_lat_c[j] + cos_lat * cos_lat_c[j] * dlon_cos;
                    match wahba_sphere_kernel_from_cos_kind(cos_gamma, penalty_order, kernel) {
                        Ok(v) => out_row[j] = v,
                        Err(_) => {
                            err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                            return;
                        }
                    }
                }
            }
        });
    if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
        crate::bail_invalid_basis!("spherical spline kernel produced a non-finite value");
    }
    Ok(out)
}


pub(crate) fn weighted_coefficient_sum_to_zero_transform(
    weights: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, BasisError> {
    let k = weights.len();
    if k < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: k });
    }
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        crate::bail_invalid_basis!(
            "sphere coefficient constraint weights must be finite and non-negative"
        );
    }
    let norm = weights.iter().map(|w| w * w).sum::<f64>().sqrt();
    if norm <= 0.0 {
        crate::bail_invalid_basis!("sphere coefficient constraint weights cannot all be zero");
    }
    let c = Array2::from_shape_vec((k, 1), weights.iter().map(|w| *w / norm).collect())
        .map_err(|e| BasisError::InvalidInput(format!("invalid sphere constraint weights: {e}")))?;
    let (z, rank) =
        rrqr_nullspace_basis(&c, default_rrqr_rank_alpha()).map_err(BasisError::LinalgError)?;
    if rank >= k {
        return Err(BasisError::ConstraintNullspaceCollapsed {
            site: "weighted_coefficient_sum_to_zero_transform",
            cross_rank: rank,
            coeff_dim: k,
            cross_frobenius: 1.0,
            constrained_gram_max_eigenvalue: f64::NAN,
            constrained_gram_min_eigenvalue: f64::NAN,
            spectral_tolerance: f64::NAN,
        });
    }
    Ok(z)
}


pub(crate) fn sphere_area_weights(centers: ArrayView2<'_, f64>, radians: bool) -> Array1<f64> {
    let to_rad = if radians {
        1.0
    } else {
        std::f64::consts::PI / 180.0
    };
    Array1::from_iter(
        centers
            .outer_iter()
            .map(|row| (row[0] * to_rad).cos().max(0.0)),
    )
}


#[inline]
fn spherical_chord_distance2(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>, radians: bool) -> f64 {
    let to_rad = if radians {
        1.0
    } else {
        std::f64::consts::PI / 180.0
    };
    let lat_a = a[0] * to_rad;
    let lon_a = a[1] * to_rad;
    let lat_b = b[0] * to_rad;
    let lon_b = b[1] * to_rad;
    let cos_gamma = lat_a.sin() * lat_b.sin() + lat_a.cos() * lat_b.cos() * (lon_a - lon_b).cos();
    2.0 * (1.0 - cos_gamma.clamp(-1.0, 1.0))
}


pub fn select_spherical_farthest_point_centers(
    data: ArrayView2<'_, f64>,
    num_centers: usize,
    radians: bool,
) -> Result<Array2<f64>, BasisError> {
    validate_lat_lon_matrix(data, "spherical farthest-point centers", radians)?;
    let n = data.nrows();
    if num_centers == 0 {
        crate::bail_invalid_basis!("spherical farthest-point center count must be positive");
    }
    if num_centers > n {
        crate::bail_invalid_basis!(
            "requested {} spherical centers but only {} rows are available",
            num_centers,
            n
        );
    }

    let mut seed_idx = 0usize;
    for i in 1..n {
        let lat_i = data[[i, 0]];
        let lon_i = data[[i, 1]];
        let lat_s = data[[seed_idx, 0]];
        let lon_s = data[[seed_idx, 1]];
        if lat_i < lat_s || (lat_i == lat_s && lon_i < lon_s) {
            seed_idx = i;
        }
    }

    let mut selected = Vec::with_capacity(num_centers);
    let mut chosen = vec![false; n];
    let mut min_dist2 = vec![f64::INFINITY; n];
    selected.push(seed_idx);
    chosen[seed_idx] = true;

    min_dist2.par_iter_mut().enumerate().for_each(|(i, slot)| {
        *slot = spherical_chord_distance2(data.row(i), data.row(seed_idx), radians);
    });
    min_dist2[seed_idx] = 0.0;

    while selected.len() < num_centers {
        let best_idx = min_dist2
            .par_iter()
            .enumerate()
            .filter(|(i, _)| !chosen[*i])
            .map(|(i, &cand)| (i, cand))
            .reduce_with(|a, b| {
                if b.1 > a.1 || (b.1 == a.1 && b.0 < a.0) {
                    b
                } else {
                    a
                }
            })
            .map(|(i, _)| i);
        let Some(next_idx) = best_idx else {
            break;
        };
        selected.push(next_idx);
        chosen[next_idx] = true;
        min_dist2.par_iter_mut().enumerate().for_each(|(i, slot)| {
            if chosen[i] {
                return;
            }
            let d2 = spherical_chord_distance2(data.row(i), data.row(next_idx), radians);
            if d2 < *slot {
                *slot = d2;
            }
        });
    }

    let mut centers = Array2::<f64>::zeros((selected.len(), 2));
    for (r, &idx) in selected.iter().enumerate() {
        centers.row_mut(r).assign(&data.row(idx));
    }
    Ok(centers)
}


/// Auto-derive a streaming row chunk size for dense basis evaluation.
///
/// The opt-in `streaming_chunk_size` knob has been removed from public specs:
/// streaming activates automatically when the would-be dense buffer
/// `n_rows * n_basis_cols * 8 bytes` exceeds 1 GiB. When streaming is
/// active, the chunk size is sized so each resident chunk holds ~256 MiB
/// of `f64` (`chunk = (256 MiB) / (n_basis_cols * 8)`), clamped to
/// `[1024, n_rows]`. Returning `None` means "do not stream, materialize
/// densely".
pub fn auto_streaming_chunk_size_for_dense(n_rows: usize, n_basis_cols: usize) -> Option<usize> {
    if n_rows == 0 || n_basis_cols == 0 {
        return None;
    }
    const DENSE_THRESHOLD_BYTES: usize = 1024 * 1024 * 1024;
    const TARGET_CHUNK_BYTES: usize = 256 * 1024 * 1024;
    const MIN_CHUNK_ROWS: usize = 1024;
    let dense_bytes = n_rows.saturating_mul(n_basis_cols).saturating_mul(8);
    if dense_bytes <= DENSE_THRESHOLD_BYTES {
        return None;
    }
    let row_bytes = n_basis_cols.saturating_mul(8).max(1);
    let raw_chunk = TARGET_CHUNK_BYTES / row_bytes;
    let clamped = raw_chunk.max(MIN_CHUNK_ROWS).min(n_rows);
    Some(clamped)
}


/// GPU dispatch for the raw `(n × m)` Wahba truncated-spectral kernel
/// matrix. Returns `None` when the kernel variant is not one of the
/// `Truncated` forms (the GPU path matches only those exactly), when the
/// `(n, m, lmax)` workload is too small to amortise H2D/D2H, or when the
/// GPU runtime is unavailable. A transient GPU failure falls back to the
/// CPU path (logged at warn level).
fn try_build_truncated_sphere_design_gpu(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    kernel: SphereWahbaKernel,
    penalty_order: usize,
    radians: bool,
) -> Option<Array2<f64>> {
    let (lmax_u16, kind) = match kernel {
        SphereWahbaKernel::SobolevTruncated { lmax } => (
            lmax,
            crate::terms::sphere_gpu::SphereSpectralKernelKind::Sobolev,
        ),
        SphereWahbaKernel::PseudoTruncated { lmax } => (
            lmax,
            crate::terms::sphere_gpu::SphereSpectralKernelKind::Pseudo,
        ),
        SphereWahbaKernel::Sobolev | SphereWahbaKernel::Pseudo => return None,
    };
    let lmax = lmax_u16 as usize;
    if lmax == 0 {
        return None;
    }
    let n = data.nrows();
    let m = centers.nrows();
    let decision = crate::terms::sphere_gpu::sphere_kernel_decision(n, m, lmax);
    if !decision.use_gpu {
        return None;
    }
    let data_xyz = crate::terms::sphere_gpu::latlon_to_xyz_host(data, radians).ok()?;
    let centers_xyz = crate::terms::sphere_gpu::latlon_to_xyz_host(centers, radians).ok()?;
    let coeffs = kind.coefficients(lmax, penalty_order);
    let inputs = crate::terms::sphere_gpu::S2KernelBuildInputs {
        n,
        m,
        lmax,
        data_xyz: &data_xyz,
        centers_xyz: &centers_xyz,
        coeffs: &coeffs,
        kind,
        layout: crate::terms::sphere_gpu::DeviceMatrixLayout::ColumnMajor,
    };
    let dev = match crate::terms::sphere_gpu::build_kernel_matrix_device(inputs) {
        Ok(d) => d,
        Err(err) => {
            log::warn!(
                "sphere GPU kernel build fell back to CPU (n={n}, m={m}, lmax={lmax}): {err}"
            );
            return None;
        }
    };
    match dev.to_host_array() {
        Ok(arr) => {
            log::info!(
                "sphere GPU kernel matrix: n={n} m={m} lmax={lmax} kind={}",
                kind.tag()
            );
            Some(arr)
        }
        Err(err) => {
            log::warn!("sphere GPU dtoh fell back to CPU (n={n}, m={m}, lmax={lmax}): {err}");
            None
        }
    }
}
