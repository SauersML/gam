// #1601 relocation debt — re-homed from the pre-#1521 monolith fixture
// `tests/src_modules/smooths/smooth_design_assembly_constraint_tests.rs`. #1601
// (commit 28bab3753) unbroke the `gam-terms --lib` test build by commenting out
// the `include!` of this file in `gam_terms::smooth::tests` (gam-solve/gam-models
// depend on gam-terms, so its `crate::solver`/`crate::estimate` bodies can never
// compile there) and parked the body "for relocation". The relocation never
// happened: `tests/src_modules/` was `mod`'d into NO test binary, so these 88
// design-assembly / constraint / IFT-cache regression guards have been silently
// dead since #1601. They belong HERE: their private driver deps
// (`build_term_collection_design`, `freeze_term_collection_from_design`,
// `FrozenTermCollectionIncrementalRealizer`, `canonical_penalties_at*`, the
// tensor/streamed eval closures) live in this `drivers` module post-carve, and
// the cross-crate `crate::` paths the fixture used are rewritten to their carved
// homes (`gam_solve::`, `gam_terms::`, `gam_problem::`, `gam_linalg::`,
// `gam_model_api::`). Self-contained `#[cfg(test)] mod`, so it adds nothing to
// the non-test build.
#[cfg(test)]
mod design_assembly_constraint_tests {
    use super::*;
    use super::test_support::SingleBlockExactJointDesignCacheTestExt;
    // The bespoke basis spec types this fixture builds designs from. `CenterStrategy`
    // and `MaternIdentifiability` already arrive via `super::*` (the drivers'
    // explicit `gam_terms::basis` import), so re-listing them here would collide
    // (E0252); every other name is pulled in explicitly.
    use gam_terms::basis::{BSplineBasisSpec, BSplineBoundaryConditions, BSplineIdentifiability, BSplineKnotSpec, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec, MaternBasisSpec, MaternNu, OneDimensionalBoundary, SpatialIdentifiability, SphericalSplineBasisSpec, ThinPlateBasisSpec, build_bspline_basis_1d};
    use gam_model_api::OuterEvalOrder;
    use ndarray::array;
    use rand::RngExt as _;
    use rand::SeedableRng as _;
    use rand::rngs::StdRng;

/// A minimal frozen 1-D B-spline basis at `feature_col`, used to exercise
/// the column-remap walk without standing up a full fit.
fn remap_test_bspline(feature_col: usize) -> SmoothBasisSpec {
    SmoothBasisSpec::BSpline1D {
        feature_col,
        spec: BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 4,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
            boundary_conditions: BSplineBoundaryConditions::default(),
            boundary: OneDimensionalBoundary::Open,
        },
    }
}

#[test]
fn superseded_fit_options_skip_only_rho_posterior_inference() {
    let options = FitOptions {
        compute_inference: true,
        max_iter: 17,
        ..FitOptions::default()
    };

    let superseded = superseded_fit_options(&options);

    assert!(superseded.compute_inference);
    assert!(superseded.skip_rho_posterior_inference);
    assert_eq!(superseded.max_iter, 17);
    assert!(!options.skip_rho_posterior_inference);
}

fn structural_shape_hex(spec: &TermCollectionSpec) -> String {
    let mut h = gam_runtime::warm_start::Fingerprinter::new();
    spec.write_structural_shape_hash(&mut h);
    h.finish_hex()
}

fn smooth_only_collection(basis: SmoothBasisSpec) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "s".to_string(),
            basis,
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

#[test]
fn structural_shape_hash_separates_topologies_but_repeats_for_same_topology(/* #869 */) {
    // Two collections that differ only in their smooth's basis variant must
    // hash differently, so AUTO topology candidates fit on the same data
    // cannot share one warm-start key and cross-seed incompatible β/ρ.
    let bspline = smooth_only_collection(remap_test_bspline(0));
    let sphere = smooth_only_collection(SmoothBasisSpec::Sphere {
        feature_cols: vec![0, 1, 2],
        spec: SphericalSplineBasisSpec::default(),
    });
    assert_ne!(
        structural_shape_hex(&bspline),
        structural_shape_hex(&sphere),
        "bspline and sphere topologies must key distinctly"
    );

    // Same topology on a different axis is still a different fit.
    let bspline_axis1 = smooth_only_collection(remap_test_bspline(1));
    assert_ne!(
        structural_shape_hex(&bspline),
        structural_shape_hex(&bspline_axis1),
        "same basis kind on a different feature column must key distinctly"
    );

    // The same topology on the same axis keys identically, so a refit of one
    // candidate (the screen→full-refit cascade) still hits its own key.
    let bspline_again = smooth_only_collection(remap_test_bspline(0));
    assert_eq!(
        structural_shape_hex(&bspline),
        structural_shape_hex(&bspline_again),
        "identical topology must reuse the same warm-start key"
    );
}

#[test]
fn structural_kind_and_feature_cols_track_basis_identity(/* #869 */) {
    // Distinct basis variants get distinct discriminants, and a wrapper
    // delegates feature columns to its inner basis so a `by=` smooth keys
    // off the same axis as the bare smooth.
    let bspline = remap_test_bspline(2);
    let sphere = SmoothBasisSpec::Sphere {
        feature_cols: vec![0, 1, 2],
        spec: SphericalSplineBasisSpec::default(),
    };
    assert_ne!(bspline.structural_kind(), sphere.structural_kind());
    assert_eq!(bspline.structural_kind(), "bspline_1d");
    assert_eq!(sphere.structural_kind(), "sphere");
    assert_eq!(bspline.structural_feature_cols(), vec![2]);
    assert_eq!(sphere.structural_feature_cols(), vec![0, 1, 2]);
}

#[test]
fn remap_feature_columns_rewrites_every_index_bearing_field() {
    // Exhaustively verify that TermCollectionSpec::remap_feature_columns
    // re-resolves *every* stored column index across every basis variant —
    // including the two that the old survival-only walk silently skipped
    // (BySmooth's by_kind.feature_col and FactorSmooth's
    // continuous_cols/group_col). This is the predict-time realignment
    // contract (#803): a stale training index that survives the walk would
    // dereference the wrong predict column.
    let spec = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "lin".to_string(),
            feature_col: 1,
            // Interaction term (distinct second factor) so a walk that
            // remaps only `feature_col` and skips `feature_cols` is caught
            // — exactly the #898 predict-time regression.
            feature_cols: vec![1, 12],
            categorical_levels: vec![],
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: vec![RandomEffectTermSpec {
            name: "re".to_string(),
            feature_col: 2,
            drop_first_level: false,
            penalized: true,
            frozen_levels: Some(vec![0, 1]),
            lenient_unseen: true,
        }],
        smooth_terms: vec![
            SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "bspline".to_string(),
                basis: remap_test_bspline(3),
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            },
            SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "by_variable".to_string(),
                basis: SmoothBasisSpec::ByVariable {
                    inner: Box::new(remap_test_bspline(4)),
                    by_col: 5,
                    kind: BySmoothKind::Numeric,
                    by: ByVariableSpec::Numeric,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            },
            SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "by_smooth".to_string(),
                basis: SmoothBasisSpec::BySmooth {
                    smooth: Box::new(remap_test_bspline(6)),
                    by_kind: ByVarKind::Numeric { feature_col: 7 },
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            },
            SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "factor_smooth".to_string(),
                basis: SmoothBasisSpec::FactorSmooth {
                    spec: FactorSmoothSpec {
                        continuous_cols: vec![8],
                        group_col: 9,
                        marginal: match remap_test_bspline(0) {
                            SmoothBasisSpec::BSpline1D { spec, .. } => spec,
                            _ => unreachable!(),
                        },
                        flavour: FactorSmoothFlavour::Sz,
                        group_frozen_levels: Some(vec![0, 1]),
                        frozen_global_orthogonality: None,
                    },
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            },
            SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "thin_plate".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![10, 11],
                    spec: ThinPlateBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        length_scale: 1.0,
                        double_penalty: true,
                        identifiability: SpatialIdentifiability::default(),
                        radial_reparam: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            },
        ],
    };

    // Remap every index by +100 (injective, so any missed field stays < 100
    // and is caught below).
    let remapped: TermCollectionSpec = spec
        .remap_feature_columns(|i| Ok::<usize, String>(i + 100))
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "remap must succeed", e));

    assert_eq!(remapped.linear_terms[0].feature_col, 101);
    // Every interaction factor in `feature_cols` must be remapped too — the
    // design builder reads `effective_feature_cols()` (i.e. `feature_cols`),
    // so a walk that skips it leaves stale training indices at predict (#898).
    assert_eq!(remapped.linear_terms[0].feature_cols, vec![101, 112]);
    assert_eq!(remapped.random_effect_terms[0].feature_col, 102);

    let collected = collect_feature_columns(&remapped);
    let mut expected: Vec<usize> = (3..=11).map(|i| i + 100).collect();
    expected.sort_unstable();
    assert_eq!(
        collected, expected,
        "every smooth-basis column index must be remapped exactly once"
    );

    // The remap closure's error must short-circuit the whole walk.
    let err = spec.remap_feature_columns(|_| Err::<usize, String>("boom".to_string()));
    assert_eq!(err.unwrap_err(), "boom");
}

/// Gather every column index referenced by the smooth bases of a spec, sorted.
fn collect_feature_columns(spec: &TermCollectionSpec) -> Vec<usize> {
    fn walk(basis: &SmoothBasisSpec, out: &mut Vec<usize>) {
        match basis {
            SmoothBasisSpec::ByVariable { inner, by_col, .. }
            | SmoothBasisSpec::FactorSumToZero { inner, by_col, .. } => {
                out.push(*by_col);
                walk(inner, out);
            }
            SmoothBasisSpec::BSpline1D { feature_col, .. } => out.push(*feature_col),
            SmoothBasisSpec::BySmooth { smooth, by_kind } => {
                match by_kind {
                    ByVarKind::Numeric { feature_col } | ByVarKind::Factor { feature_col, .. } => {
                        out.push(*feature_col)
                    }
                }
                walk(smooth, out);
            }
            SmoothBasisSpec::FactorSmooth { spec } => {
                out.extend(spec.continuous_cols.iter().copied());
                out.push(spec.group_col);
            }
            SmoothBasisSpec::ThinPlate { feature_cols, .. }
            | SmoothBasisSpec::Sphere { feature_cols, .. }
            | SmoothBasisSpec::ConstantCurvature { feature_cols, .. }
            | SmoothBasisSpec::Matern { feature_cols, .. }
            | SmoothBasisSpec::MeasureJet { feature_cols, .. }
            | SmoothBasisSpec::Duchon { feature_cols, .. }
            | SmoothBasisSpec::Pca { feature_cols, .. }
            | SmoothBasisSpec::TensorBSpline { feature_cols, .. } => {
                out.extend(feature_cols.iter().copied())
            }
        }
    }
    let mut out = Vec::new();
    for st in &spec.smooth_terms {
        walk(&st.basis, &mut out);
    }
    out.sort_unstable();
    out
}

fn spatial_log_kappa_bounds_from_options(
    dims_per_term: &[usize],
    options: &SpatialLengthScaleOptimizationOptions,
    lower: bool,
) -> SpatialLogKappaCoords {
    let total: usize = dims_per_term.iter().sum();
    let value = if lower {
        -options.max_length_scale.ln()
    } else {
        -options.min_length_scale.ln()
    };
    SpatialLogKappaCoords::new_with_dims(
        Array1::<f64>::from_elem(total, value),
        dims_per_term.to_vec(),
    )
}

// `pub(super)` so the sibling re-homed fixture `adaptive_bounded_duchon_tests`
// (the other #1601 orphan that shared this monolith helper) resolves it through
// the `drivers` parent scope instead of duplicating the setup.
pub(super) fn two_block_exact_joint_hyper_setup(
    meanspec: &TermCollectionSpec,
    noisespec: &TermCollectionSpec,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> ExactJointHyperSetup {
    let mean_terms = spatial_length_scale_term_indices(meanspec);
    let noise_terms = spatial_length_scale_term_indices(noisespec);
    let mean_dims_per_term = spatial_dims_per_term(meanspec, &mean_terms);
    let noise_dims_per_term = spatial_dims_per_term(noisespec, &noise_terms);
    let mean_use_aniso = has_aniso_terms(meanspec, &mean_terms);
    let noise_use_aniso = has_aniso_terms(noisespec, &noise_terms);
    let mean_log_kappa = if mean_use_aniso {
        SpatialLogKappaCoords::from_length_scales_aniso(meanspec, &mean_terms, kappa_options)
    } else {
        SpatialLogKappaCoords::from_length_scales(meanspec, &mean_terms, kappa_options)
    };
    let noise_log_kappa = if noise_use_aniso {
        SpatialLogKappaCoords::from_length_scales_aniso(noisespec, &noise_terms, kappa_options)
    } else {
        SpatialLogKappaCoords::from_length_scales(noisespec, &noise_terms, kappa_options)
    };
    let dims_per_term = mean_log_kappa
        .dims_per_term()
        .iter()
        .copied()
        .chain(noise_log_kappa.dims_per_term().iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(
        dims_per_term,
        mean_dims_per_term
            .iter()
            .copied()
            .chain(noise_dims_per_term.iter().copied())
            .collect::<Vec<_>>()
    );
    let log_kappa0 = SpatialLogKappaCoords::new_with_dims(
        Array1::from_iter(
            mean_log_kappa
                .as_array()
                .iter()
                .chain(noise_log_kappa.as_array().iter())
                .copied(),
        ),
        dims_per_term.clone(),
    );
    ExactJointHyperSetup::new(
        Array1::zeros(0),
        Array1::zeros(0),
        Array1::zeros(0),
        log_kappa0,
        spatial_log_kappa_bounds_from_options(&dims_per_term, kappa_options, true),
        spatial_log_kappa_bounds_from_options(&dims_per_term, kappa_options, false),
    )
}

fn max_abs_diff_matrix(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn assert_frozen_replay_matches_fit(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
    label: &str,
) {
    let fit_design = build_term_collection_design(data, spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "fit-time design", e));
    let frozen =
        freeze_term_collection_from_design(spec, &fit_design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze term collection", e));
    let replay_design = build_term_collection_design(data, &frozen).unwrap_or_else(|e| panic!("{} failed: {:?}", "replay design", e));
    let fit_dense = fit_design.design.to_dense();
    let replay_dense = replay_design.design.to_dense();
    let max_abs = max_abs_diff_matrix(&fit_dense, &replay_dense);
    if max_abs > 1e-10 {
        // The `duchon` arm's difference is BIT-STABLE across runs (measured:
        // five runs, one distinct value `4.738869852083383e-10`), so this is a
        // specific term computed differently on the two paths, not arithmetic
        // drift. A scalar `max_abs` cannot say WHICH term. Name the column, and
        // the column names the basis function; the offending columns are then
        // matched against what the frozen path replays that the fit path
        // recomputes. Thin-plate and matern pass the same bar, so whatever this
        // is lives in the Duchon-only artifacts.
        // The divergence is bit-stable and confined to ONE column, so a specific
        // TERM differs rather than arithmetic drifting. The Duchon replay's only
        // fit-derived artifact is the data-metric radial reparameterization `V`:
        // `build_duchon_basis_spec_chart` COMPUTES it on the fit path
        // (`radial_reparam: None`) and early-returns to consume the STORED one on
        // replay (`radial_reparam: Some`). Thin-plate and matern pass this same
        // bar and never run that chart.
        //
        // So report whether `V` itself survived the round trip. If `V` matches
        // bit-for-bit the difference enters AFTER it (the
        // `fast_ab(&kernel_transform, v)` fold, or the constrained-penalty
        // assembly feeding it); if it does not, storage/retrieval of `V` is the
        // defect and the design column is just where it shows first.
        // `V` round-trips bit-identically (measured), so the defect is DOWNSTREAM
        // of it. Both paths reach the same `fast_ab(&kernel_transform, v)` fold
        // with the same `v`, which leaves `kernel_transform` — i.e. what
        // `kernel_constraint_nullspace` returned — as the remaining candidate.
        //
        // Note the asymmetry that makes this plausible: the fit path calls that
        // routine TWICE (once in `build_duchon_basis_spec_chart` to assemble
        // `omega_constrained`, again inside `build_duchon_basis`), while the
        // replay path early-returns and calls it ONCE. If it is not
        // bit-deterministic (an eigen/QR sign or ordering convention), the two
        // folded transforms differ in exactly one direction — the observed
        // signature. Compare every replayed matrix so the differing one names
        // itself instead of being guessed at.
        let matrix_notes = {
            let mats = |d: &TermCollectionDesign| -> Vec<(&'static str, Option<Array2<f64>>)> {
                match &d.smooth.terms[0].metadata {
                    BasisMetadata::Duchon {
                        centers,
                        identifiability_transform,
                        operator_collocation_points,
                        radial_reparam,
                        aniso_log_scales,
                        ..
                    } => vec![
                        // The ONE replayed artifact the matrix sweep missed, and
                        // the only one whose two paths take DIFFERENT BRANCHES:
                        // the fit spec carries `aniso_log_scales: None` so
                        // `auto_seed_aniso_contrasts(centers, None)` DERIVES them,
                        // while `design_freezing` sets
                        // `s.aniso_log_scales = meta_aniso.clone()` so the replay
                        // USES the stored values. Bit-identical centers do not
                        // imply a bit-identical seed. Rendered as a 1xN row so it
                        // reuses the same comparison.
                        (
                            "aniso_log_scales",
                            aniso_log_scales.as_ref().map(|v| {
                                Array2::from_shape_vec((1, v.len()), v.clone())
                                    .expect("1xN aniso row")
                            }),
                        ),
                        ("centers", Some(centers.clone())),
                        ("identifiability_transform", identifiability_transform.clone()),
                        ("operator_collocation_points", operator_collocation_points.clone()),
                        ("radial_reparam", radial_reparam.clone()),
                    ],
                    BasisMetadata::ThinPlate {
                        centers,
                        identifiability_transform,
                        radial_reparam,
                        ..
                    } => vec![
                        ("centers", Some(centers.clone())),
                        ("identifiability_transform", identifiability_transform.clone()),
                        ("radial_reparam", radial_reparam.clone()),
                    ],
                    _ => vec![],
                }
            };
            let (a, b) = (mats(&fit_design), mats(&replay_design));
            a.into_iter()
                .zip(b)
                .map(|((name, x), (_, y))| match (x, y) {
                    (Some(x), Some(y)) if x.dim() == y.dim() => {
                        let worst = x
                            .iter()
                            .zip(y.iter())
                            .map(|(&p, &q)| (p - q).abs())
                            .fold(0.0_f64, f64::max);
                        let bitwise =
                            x.iter().zip(y.iter()).all(|(&p, &q)| p.to_bits() == q.to_bits());
                        format!("{name}{:?} bit_identical={bitwise} max|Δ|={worst:.6e}", x.dim())
                    }
                    (Some(x), Some(y)) => {
                        format!("{name} SHAPE {:?} -> {:?}", x.dim(), y.dim())
                    }
                    (Some(x), None) => format!("{name}{:?} -> None", x.dim()),
                    (None, Some(y)) => format!("{name} None -> {:?}", y.dim()),
                    (None, None) => format!("{name} absent both"),
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        let reparam_note = {
            let pick = |d: &TermCollectionDesign| match &d.smooth.terms[0].metadata {
                BasisMetadata::Duchon { radial_reparam, .. } => radial_reparam.clone(),
                BasisMetadata::ThinPlate { radial_reparam, .. } => radial_reparam.clone(),
                _ => None,
            };
            match (pick(&fit_design), pick(&replay_design)) {
                (Some(a), Some(b)) if a.dim() == b.dim() => {
                    let worst = a
                        .iter()
                        .zip(b.iter())
                        .map(|(&x, &y)| (x - y).abs())
                        .fold(0.0_f64, f64::max);
                    let bitwise = a
                        .iter()
                        .zip(b.iter())
                        .all(|(&x, &y)| x.to_bits() == y.to_bits());
                    format!(
                        "; radial_reparam V {:?}: bit_identical={bitwise}, max|Δ|={worst:.6e}",
                        a.dim()
                    )
                }
                (Some(a), Some(b)) => {
                    format!("; radial_reparam V SHAPE CHANGED {:?} -> {:?}", a.dim(), b.dim())
                }
                (Some(a), None) => format!("; radial_reparam V {:?} -> None on replay", a.dim()),
                (None, Some(b)) => format!("; radial_reparam V None -> {:?} on replay", b.dim()),
                (None, None) => "; radial_reparam V absent on both paths".to_string(),
            }
        };
        assert_eq!(fit_dense.dim(), replay_dense.dim());
        let (rows, cols) = fit_dense.dim();
        let mut per_column: Vec<(usize, f64, usize)> = (0..cols)
            .map(|c| {
                let mut worst = 0.0_f64;
                let mut worst_row = 0usize;
                for r in 0..rows {
                    let d = (fit_dense[[r, c]] - replay_dense[[r, c]]).abs();
                    if d > worst {
                        worst = d;
                        worst_row = r;
                    }
                }
                (c, worst, worst_row)
            })
            .filter(|(_, worst, _)| *worst > 1e-10)
            .collect();
        per_column.sort_by(|a, b| b.1.total_cmp(&a.1));
        let offenders: Vec<String> = per_column
            .iter()
            .take(8)
            .map(|(c, worst, r)| {
                format!(
                    "col {c}: |Δ|={worst:.6e} at row {r} (fit={:.17e}, replay={:.17e})",
                    fit_dense[[*r, *c]],
                    replay_dense[[*r, *c]]
                )
            })
            .collect();
        panic!(
            "{label} frozen replay changed realized design: max_abs={max_abs} \
             over a {rows}x{cols} design; {} of {cols} columns exceed 1e-10. \
             Worst columns: [{}]{reparam_note}; replayed matrices: {matrix_notes}",
            per_column.len(),
            offenders.join("; ")
        );
    }
}

fn max_abs_diff_vector(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

// `pub(super)` — shared with the sibling `adaptive_bounded_duchon_tests` #1601
// re-home (its freeze/cache-rebuild pins compare two designs column-for-column).
pub(super) fn assert_term_collection_designs_match(
    left: &TermCollectionDesign,
    right: &TermCollectionDesign,
    label: &str,
) {
    let left_design = left.design.to_dense();
    let right_design = right.design.to_dense();
    let design_diff = max_abs_diff_matrix(&left_design, &right_design);
    assert!(
        design_diff <= 1e-10,
        "{label} design mismatch max_abs={design_diff}"
    );
    assert_eq!(
        left.penalties.len(),
        right.penalties.len(),
        "{label} penalty count mismatch"
    );
    for (idx, (lp, rp)) in left
        .penalties
        .iter()
        .zip(right.penalties.iter())
        .enumerate()
    {
        assert_eq!(
            lp.col_range, rp.col_range,
            "{label} penalty {idx} col_range mismatch"
        );
        let penalty_diff = max_abs_diff_matrix(&lp.local, &rp.local);
        assert!(
            penalty_diff <= 1e-10,
            "{label} penalty {idx} mismatch max_abs={penalty_diff}"
        );
    }
    assert_eq!(
        left.nullspace_dims, right.nullspace_dims,
        "{label} nullspace dims mismatch"
    );
    assert_eq!(
        left.penaltyinfo.len(),
        right.penaltyinfo.len(),
        "{label} penaltyinfo length mismatch"
    );
    for (idx, (linfo, rinfo)) in left
        .penaltyinfo
        .iter()
        .zip(right.penaltyinfo.iter())
        .enumerate()
    {
        assert_eq!(
            linfo.termname, rinfo.termname,
            "{label} penaltyinfo termname mismatch at {idx}"
        );
        assert_eq!(
            linfo.penalty.source, rinfo.penalty.source,
            "{label} penalty source mismatch at {idx}"
        );
        assert_eq!(
            linfo.penalty.effective_rank, rinfo.penalty.effective_rank,
            "{label} penalty rank mismatch at {idx}"
        );
        assert!(
            (linfo.penalty.normalization_scale - rinfo.penalty.normalization_scale).abs() <= 1e-10,
            "{label} penalty normalization mismatch at {idx}"
        );
    }
    match (
        left.coefficient_lower_bounds.as_ref(),
        right.coefficient_lower_bounds.as_ref(),
    ) {
        (Some(lb_left), Some(lb_right)) => {
            let diff = max_abs_diff_vector(lb_left, lb_right);
            assert!(diff <= 1e-10, "{label} lower-bound mismatch max_abs={diff}");
        }
        (None, None) => {}
        _ => panic!("{label} lower-bound presence mismatch"),
    }
    match (
        left.linear_constraints.as_ref(),
        right.linear_constraints.as_ref(),
    ) {
        (Some(c_left), Some(c_right)) => {
            let a_diff = max_abs_diff_matrix(&c_left.a, &c_right.a);
            let b_diff = max_abs_diff_vector(&c_left.b, &c_right.b);
            assert!(
                a_diff <= 1e-10,
                "{label} linear-constraint A mismatch max_abs={a_diff}"
            );
            assert!(
                b_diff <= 1e-10,
                "{label} linear-constraint b mismatch max_abs={b_diff}"
            );
        }
        (None, None) => {}
        _ => panic!("{label} linear-constraint presence mismatch"),
    }
}

#[test]
fn freeze_term_collection_handles_thin_plate_auto_promotion_to_duchon() {
    // Reproducer for the freezer falling into its catch-all "smooth
    // metadata/spec type mismatch" arm whenever `build_thin_plate_basis`
    // delegates to `build_duchon_basis` (the auto-promotion path that
    // fires whenever canonical TPS is mathematically infeasible at the
    // requested d, k).  Without the rewrite step in
    // `freeze_term_collection_from_design`, the (ThinPlate spec, Duchon
    // metadata) pairing aborts the entire fit at serialization time even
    // though the fit itself succeeded against the promoted Duchon basis.
    //
    // d=5, k=10 hits the auto-promotion branch (canonical TPS at d=5 needs
    // M(5, m=3)=21 polynomial columns, above k=10) AND the Duchon fallback
    // is admissible (Linear nullspace at p=2 needs m_poly=6 centers, so
    // k=10 ≥ 6, with the smallest s satisfying both 2(p+s) > d and
    // 2s < d giving s=1).
    let mut rng = StdRng::seed_from_u64(20260504);
    let n = 200usize;
    let mut data = Array2::<f64>::zeros((n, 5));
    for i in 0..n {
        for j in 0..5 {
            data[[i, j]] = rng.random_range(-1.0..1.0);
        }
    }
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "thinplate(pc1, pc2, pc3, pc4, pc5)".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0, 1, 2, 3, 4],
                spec: ThinPlateBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 10 },
                    length_scale: 1.0,
                    double_penalty: false,
                    identifiability: SpatialIdentifiability::default(),
                    radial_reparam: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let fit_design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "fit-time design", e));
    // Confirm we actually exercised the auto-promotion branch.
    let metadata = &fit_design
        .smooth
        .terms
        .first()
        .unwrap_or_else(|| panic!("{} failed", "at least one smooth term"))
        .metadata;
    assert!(
        matches!(metadata, BasisMetadata::Duchon { .. }),
        "expected auto-promotion to Duchon, got {metadata:?}"
    );

    let frozen = freeze_term_collection_from_design(&spec, &fit_design).unwrap_or_else(|e| panic!("{} failed: {:?}",
        "freeze must succeed across the auto-promoted (ThinPlate spec, Duchon metadata) pair", e
    ));
    assert!(
        matches!(frozen.smooth_terms[0].basis, SmoothBasisSpec::Duchon { .. }),
        "frozen spec should reflect the auto-promotion as a Duchon variant"
    );

    // Predict-time replay must reproduce the fit-time design bit-for-bit:
    // the frozen Duchon spec carries the exact centers, power, and
    // nullspace_order that the basis builder selected during the fit.
    let replay_design = build_term_collection_design(data.view(), &frozen).unwrap_or_else(|e| panic!("{} failed: {:?}", "replay design", e));
    let max_abs = fit_design
        .design
        .to_dense()
        .iter()
        .zip(replay_design.design.to_dense().iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs <= 1e-10,
        "auto-promoted frozen replay changed realized design: max_abs={max_abs}"
    );
}

#[test]
fn term_collection_design_combines_linear_and_smooth() {
    let data = array![
        [0.0, 0.0, 0.2],
        [0.2, 0.1, 0.4],
        [0.4, 0.2, 0.6],
        [0.6, 0.4, 0.7],
        [0.8, 0.7, 0.9],
        [1.0, 1.0, 1.1]
    ];
    let spec = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "lin_x0".to_string(),
            feature_col: 0,
            feature_cols: vec![0],
            categorical_levels: vec![],
            double_penalty: true,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "tps_x1x2".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![1, 2],
                spec: ThinPlateBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    length_scale: 1.0,
                    double_penalty: true,
                    identifiability: SpatialIdentifiability::default(),
                    radial_reparam: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let design = build_term_collection_design(data.view(), &spec).unwrap();
    let design_dense = design.design.to_dense();
    assert_eq!(design.design.nrows(), data.nrows());
    assert_eq!(design.intercept_range, 0..1);
    assert!(
        design_dense
            .column(design.intercept_range.start)
            .iter()
            .all(|&v: &f64| (v - 1.0).abs() < 1e-12)
    );
    assert!(design.design.ncols() >= 2);
    assert_eq!(design.linear_ranges.len(), 1);
    assert_eq!(design.random_effect_ranges.len(), 0);
    assert_eq!(design.penalties.len(), 3); // linear ridge + 2 smooth penalties (bending + nullspace)
    assert_eq!(design.nullspace_dims.len(), 3);
}

#[test]
fn spatial_smooth_columns_do_not_duplicate_global_intercept() {
    let data = array![
        [0.0, 0.0],
        [0.2, 0.1],
        [0.4, 0.3],
        [0.6, 0.6],
        [0.8, 0.7],
        [1.0, 1.0],
    ];
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "tps_xy".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0, 1],
                spec: ThinPlateBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    length_scale: 1.0,
                    double_penalty: false,
                    identifiability: SpatialIdentifiability::default(),
                    radial_reparam: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap();
    let design_dense = design.design.to_dense();
    let smooth_start = 1usize;
    let smooth_end = smooth_start + design.smooth.total_smooth_cols();
    for col in smooth_start..smooth_end {
        let is_all_ones = design_dense
            .column(col)
            .iter()
            .all(|&v: &f64| (v - 1.0).abs() < 1e-12);
        assert!(
            !is_all_ones,
            "smooth column {col} unexpectedly duplicated intercept"
        );
    }
}

#[test]
fn spatial_smooth_drops_matching_linear_trend_columns() {
    let data = array![
        [0.0, 0.1],
        [0.2, 0.0],
        [0.3, 0.4],
        [0.5, 0.2],
        [0.7, 0.9],
        [1.0, 0.8],
        [1.2, 1.1],
        [1.4, 1.3],
    ];
    let spec = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "lin_x0".to_string(),
            feature_col: 0,
            feature_cols: vec![0],
            categorical_levels: vec![],
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "tps_xy".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0, 1],
                spec: ThinPlateBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    length_scale: 1.0,
                    double_penalty: false,
                    identifiability: SpatialIdentifiability::default(),
                    radial_reparam: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap();

    // Raw TPS width for k=4,d=2 is 4; we drop intercept + matching x0 linear component.
    assert_eq!(design.smooth.total_smooth_cols(), 2);

    let dense = design.design.to_dense_cow();
    let lin_col = design.linear_ranges[0].1.start;
    let linvalues = dense.column(lin_col).to_owned();
    let smooth_start = 1 + spec.linear_terms.len();
    let smooth_end = smooth_start + design.smooth.total_smooth_cols();
    for col in smooth_start..smooth_end {
        let same_as_linear = dense
            .column(col)
            .iter()
            .zip(linvalues.iter())
            .all(|(&a, &b)| (a - b).abs() < 1e-12);
        assert!(
            !same_as_linear,
            "smooth column {col} unexpectedly duplicated linear term column"
        );
    }
}

#[test]
fn spatial_option5_is_orthogonal_to_parametric_block() {
    let data = array![
        [0.0, 0.1],
        [0.2, 0.0],
        [0.3, 0.4],
        [0.5, 0.2],
        [0.7, 0.9],
        [1.0, 0.8],
    ];
    let spec = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "lin_x0".to_string(),
            feature_col: 0,
            feature_cols: vec![0],
            categorical_levels: vec![],
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "tps_xy".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0, 1],
                spec: ThinPlateBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    length_scale: 1.0,
                    double_penalty: false,
                    identifiability: SpatialIdentifiability::OrthogonalToParametric,
                    radial_reparam: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap();
    let design_dense = design.design.to_dense();
    let n = data.nrows();
    let mut c = Array2::<f64>::zeros((n, 2));
    c.column_mut(0).fill(1.0);
    c.column_mut(1).assign(&data.column(0));
    let smooth_start = 1 + spec.linear_terms.len();
    let b = design_dense
        .slice(s![
            ..,
            smooth_start..(smooth_start + design.smooth.total_smooth_cols())
        ])
        .to_owned();
    let cross = b.t().dot(&c);
    let num = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
    let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    let c_norm = c.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rel = num / (b_norm * c_norm).max(1e-300);
    assert!(
        rel <= 1e-10,
        "smooth residual against model-owned parametric block too large: {rel}"
    );
}

#[test]
fn thin_plate_default_identifiability_centers_against_intercept_only_without_linear_terms() {
    // Without any explicit linear term in the formula, the parametric
    // constraint block built by `build_parametric_constraint_block_for_term`
    // contains only the intercept column — see the
    // `SpatialIdentifiability` docs:
    //
    //   "The term-collection builder augments `C` with explicit linear
    //    terms when those terms are present in the formula."
    //
    // So a standalone TPS smooth marked `OrthogonalToParametric` is
    // orthogonalized only against `[1]`; its full polynomial nullspace
    // (the linear axes that thin-plate splines own as part of their
    // canonical model surface) stays in the smooth's column span.
    // Companions: `standalone_tps_keeps_centered_linear_nullspace` and
    // `term_collection_joint_duchon_carries_frozen_transform_into_metadata`
    // assert the dimension count from the same contract.
    let data = array![
        [-1.9, -1.2],
        [-1.3, -0.7],
        [-0.8, -0.4],
        [-0.2, 0.1],
        [0.0, 0.3],
        [0.4, 0.5],
        [0.9, 0.8],
        [1.4, 1.1],
        [1.9, 1.5],
        [2.3, 1.8],
    ];
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: (0..2)
            .map(|feature| SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: format!("tps_x{feature}"),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![feature],
                    spec: ThinPlateBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::EqualMass { num_centers: 4 },
                        length_scale: 1.0,
                        double_penalty: false,
                        identifiability: SpatialIdentifiability::OrthogonalToParametric,
                        radial_reparam: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            })
            .collect(),
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap();
    let design_dense = design.design.to_dense();
    let smooth_start = 1 + spec.linear_terms.len();
    let intercept = Array2::<f64>::ones((data.nrows(), 1));
    for (term_idx, term) in design.smooth.terms.iter().enumerate() {
        let block = design_dense
            .slice(s![
                ..,
                (smooth_start + term.coeff_range.start)..(smooth_start + term.coeff_range.end)
            ])
            .to_owned();
        let cross = block.t().dot(&intercept);
        let num = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
        let block_norm = block.iter().map(|v| v * v).sum::<f64>().sqrt();
        let intercept_norm = intercept.iter().map(|v| v * v).sum::<f64>().sqrt();
        let rel = num / (block_norm * intercept_norm).max(1e-300);
        assert!(
            rel <= 1e-10,
            "ThinPlate term {term_idx} should be centered against the intercept (no linear terms in formula); got rel={rel:.3e}"
        );
    }
}

#[test]
fn spatial_option5_does_not_overconstrain_on_nonoverlapping_linear_terms() {
    let n = 40usize;
    let p = 16usize;
    let mut data = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            // Deterministic, non-collinear synthetic PCs.
            data[[i, j]] = (i as f64) * 0.03 + (j as f64) * 0.11 + ((i * (j + 1)) as f64) * 1e-3;
        }
    }

    let spec = TermCollectionSpec {
        linear_terms: (5..16)
            .map(|j| LinearTermSpec {
                name: format!("pc{j}"),
                feature_col: j,
                feature_cols: vec![j],
                categorical_levels: vec![],
                double_penalty: false,
                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                coefficient_min: None,
                coefficient_max: None,
                frozen_function_mass: None,
            })
            .collect(),
        random_effect_terms: vec![],
        smooth_terms: vec![
            SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "tps_pc1".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![1],
                    spec: ThinPlateBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: 1.0,
                        double_penalty: true,
                        identifiability: SpatialIdentifiability::OrthogonalToParametric,
                        radial_reparam: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            },
            SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "tps_pc2".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![2],
                    spec: ThinPlateBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: 1.0,
                        double_penalty: true,
                        identifiability: SpatialIdentifiability::OrthogonalToParametric,
                        radial_reparam: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            },
        ],
    };

    let out = build_term_collection_design(data.view(), &spec);
    assert!(
        out.is_ok(),
        "term-local Option 5 should not over-constrain non-overlapping smooth/linear terms: {:?}",
        out.err()
    );
}

#[test]
fn overlapping_linear_term_residualizes_bspline_smooth() {
    let data = array![
        [0.0],
        [0.1],
        [0.2],
        [0.3],
        [0.4],
        [0.5],
        [0.6],
        [0.7],
        [0.8],
        [0.9],
        [1.0],
    ];
    let spec = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "x".to_string(),
            feature_col: 0,
            feature_cols: vec![0],
            categorical_levels: vec![],
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "s_x".to_string(),
            basis: SmoothBasisSpec::BSpline1D {
                feature_col: 0,
                spec: BSplineBasisSpec {
                    degree: 3,
                    penalty_order: 2,
                    knotspec: BSplineKnotSpec::Generate {
                        data_range: (0.0, 1.0),
                        num_internal_knots: 4,
                    },
                    double_penalty: false,
                    identifiability: BSplineIdentifiability::default(),
                    boundary: OneDimensionalBoundary::Open,
                    boundary_conditions: BSplineBoundaryConditions::default(),
                },
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "bspline design", e));
    let mut c = Array2::<f64>::zeros((data.nrows(), 2));
    c.column_mut(0).fill(1.0);
    c.column_mut(1).assign(&data.column(0));
    let rel = orthogonality_relative_residual_for_design(&design.smooth.term_designs[0], c.view())
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "orthogonality residual", e));
    assert!(
        rel <= 1e-10,
        "B-spline smooth should be orthogonal to [1, x] when linear(x) is present; rel={rel}"
    );
}

#[test]
fn standalone_tps_keeps_centered_linear_nullspace() {
    let data = array![[-1.5], [-0.7], [0.2], [0.8], [1.6]];
    let centers = array![[-1.5], [0.2], [1.6]];
    let smooth = SmoothTermSpec {
            frozen_parametric_residualization: None,
        name: "s_x".to_string(),
        basis: SmoothBasisSpec::ThinPlate {
            feature_cols: vec![0],
            spec: ThinPlateBasisSpec {
                periodic: None,
                center_strategy: CenterStrategy::UserProvided(centers),
                length_scale: 1.0,
                double_penalty: false,
                identifiability: SpatialIdentifiability::OrthogonalToParametric,
                radial_reparam: None,
            },
            input_scale: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    };
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![smooth],
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "tps design", e));

    assert_eq!(design.smooth.term_designs[0].ncols(), 2);
    assert_eq!(design.smooth.nullspace_dims, vec![1]);
    let intercept = Array2::<f64>::ones((data.nrows(), 1));
    let rel = orthogonality_relative_residual_for_design(
        &design.smooth.term_designs[0],
        intercept.view(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "intercept residual", e));
    assert!(
        rel <= 1e-10,
        "standalone TPS should be centered against the intercept while retaining its linear nullspace; rel={rel}"
    );
}

#[test]
fn spatial_parametric_ownership_projects_only_explicit_linear_axes() {
    let term = SmoothTermSpec {
            frozen_parametric_residualization: None,
        name: "s_xy".to_string(),
        basis: SmoothBasisSpec::ThinPlate {
            feature_cols: vec![0, 1],
            spec: ThinPlateBasisSpec {
                periodic: None,
                center_strategy: CenterStrategy::EqualMass { num_centers: 4 },
                length_scale: 1.0,
                double_penalty: false,
                identifiability: SpatialIdentifiability::OrthogonalToParametric,
                radial_reparam: None,
            },
            input_scale: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    };
    let linear_terms = vec![LinearTermSpec {
        name: "x0".to_string(),
        feature_col: 0,
        feature_cols: vec![0],
        categorical_levels: vec![],
        double_penalty: false,
        coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
        coefficient_min: None,
        coefficient_max: None,
        frozen_function_mass: None,
    }];

    assert_eq!(
        smooth_intrinsic_parametric_feature_cols(&linear_terms, &term),
        vec![0],
        "a linear term on x0 should not claim the smooth's x1 nullspace"
    );
}

#[test]
fn hierarchical_smooth_ownership_is_order_independent_for_bspline_and_duchon() {
    let data = array![
        [0.00, 0.00],
        [0.10, 0.15],
        [0.18, 0.30],
        [0.27, 0.10],
        [0.35, 0.55],
        [0.46, 0.25],
        [0.54, 0.70],
        [0.63, 0.40],
        [0.72, 0.85],
        [0.81, 0.60],
        [0.90, 0.95],
        [1.00, 0.75],
    ];

    let bspline_term = SmoothTermSpec {
            frozen_parametric_residualization: None,
        name: "s_x".to_string(),
        basis: SmoothBasisSpec::BSpline1D {
            feature_col: 0,
            spec: BSplineBasisSpec {
                degree: 3,
                penalty_order: 2,
                knotspec: BSplineKnotSpec::Generate {
                    data_range: (0.0, 1.0),
                    num_internal_knots: 5,
                },
                double_penalty: false,
                identifiability: BSplineIdentifiability::default(),
                boundary: OneDimensionalBoundary::Open,
                boundary_conditions: BSplineBoundaryConditions::default(),
            },
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    };
    let duchon_term = SmoothTermSpec {
            frozen_parametric_residualization: None,
        name: "duchon_xy".to_string(),
        basis: SmoothBasisSpec::Duchon {
            feature_cols: vec![0, 1],
            spec: DuchonBasisSpec {
                radial_reparam: None,
                periodic: None,
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                length_scale: Some(1.0),
                power: 5.0,
                nullspace_order: DuchonNullspaceOrder::Linear,
                identifiability: SpatialIdentifiability::default(),
                aniso_log_scales: None,
                operator_penalties: DuchonOperatorPenaltySpec::default(),
                boundary: OneDimensionalBoundary::Open,
            },
            input_scale: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    };

    let spec_a = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![duchon_term.clone(), bspline_term.clone()],
    };
    let spec_b = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![bspline_term, duchon_term],
    };

    let design_a = build_term_collection_design(data.view(), &spec_a).unwrap_or_else(|e| panic!("{} failed: {:?}", "design a", e));
    let design_b = build_term_collection_design(data.view(), &spec_b).unwrap_or_else(|e| panic!("{} failed: {:?}", "design b", e));

    for design in [&design_a, &design_b] {
        let owner_idx = design
            .smooth
            .terms
            .iter()
            .position(|term| term.name == "s_x")
            .unwrap_or_else(|| panic!("{} failed", "owner term"));
        let target_idx = design
            .smooth
            .terms
            .iter()
            .position(|term| term.name == "duchon_xy")
            .unwrap_or_else(|| panic!("{} failed", "target term"));
        let owner_dense = design.smooth.term_designs[owner_idx].to_dense();
        let rel = orthogonality_relative_residual_for_design(
            &design.smooth.term_designs[target_idx],
            owner_dense.view(),
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "orthogonality residual", e));
        assert!(
            rel <= 1e-10,
            "multivariate Duchon term should be residualized against owned 1D spline space; rel={rel}"
        );
    }

    let duchon_a_idx = design_a
        .smooth
        .terms
        .iter()
        .position(|term| term.name == "duchon_xy")
        .unwrap_or_else(|| panic!("{} failed", "duchon in design a"));
    let duchon_b_idx = design_b
        .smooth
        .terms
        .iter()
        .position(|term| term.name == "duchon_xy")
        .unwrap_or_else(|| panic!("{} failed", "duchon in design b"));
    let duchon_a = design_a.smooth.term_designs[duchon_a_idx].to_dense();
    let duchon_b = design_b.smooth.term_designs[duchon_b_idx].to_dense();
    assert_eq!(duchon_a.dim(), duchon_b.dim());
    let max_abs = duchon_a
        .iter()
        .zip(duchon_b.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs <= 1e-10,
        "hierarchical ownership should not depend on user term order; max_abs={max_abs}"
    );
}

#[test]
fn freeze_roundtrip_preserves_hierarchical_smooth_transforms() {
    let data = array![
        [0.00, 0.00],
        [0.10, 0.15],
        [0.18, 0.30],
        [0.27, 0.10],
        [0.35, 0.55],
        [0.46, 0.25],
        [0.54, 0.70],
        [0.63, 0.40],
        [0.72, 0.85],
        [0.81, 0.60],
        [0.90, 0.95],
        [1.00, 0.75],
    ];
    let spec = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "x".to_string(),
            feature_col: 0,
            feature_cols: vec![0],
            categorical_levels: vec![],
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: vec![],
        smooth_terms: vec![
            SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "duchon_xy".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: vec![0, 1],
                    spec: DuchonBasisSpec {
                        radial_reparam: None,
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                        length_scale: Some(1.0),
                        power: 1.0,
                        nullspace_order: DuchonNullspaceOrder::Linear,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                        operator_penalties: DuchonOperatorPenaltySpec::default(),
                        boundary: OneDimensionalBoundary::Open,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            },
            SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "s_x".to_string(),
                basis: SmoothBasisSpec::BSpline1D {
                    feature_col: 0,
                    spec: BSplineBasisSpec {
                        degree: 3,
                        penalty_order: 2,
                        knotspec: BSplineKnotSpec::Generate {
                            data_range: (0.0, 1.0),
                            num_internal_knots: 5,
                        },
                        double_penalty: false,
                        identifiability: BSplineIdentifiability::default(),
                        boundary: OneDimensionalBoundary::Open,
                        boundary_conditions: BSplineBoundaryConditions::default(),
                    },
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            },
        ],
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "fit-time design", e));
    let frozen =
        freeze_term_collection_from_design(&spec, &design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze hierarchical design", e));
    let replay = build_term_collection_design(data.view(), &frozen).unwrap_or_else(|e| panic!("{} failed: {:?}", "replay design", e));

    let dense_fit = design.design.to_dense();
    let dense_replay = replay.design.to_dense();
    assert_eq!(dense_fit.dim(), dense_replay.dim());
    let max_abs = dense_fit
        .iter()
        .zip(dense_replay.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs <= 1e-10,
        "frozen hierarchical transforms should replay exactly on the training data; max_abs={max_abs}"
    );
}

#[test]
fn spatial_option5_preserves_lazy_thin_plate_terms_at_large_scale() {
    // The thin-plate term goes operator-backed once its projected dense
    // materialization exceeds the resource policy's byte budget, i.e. once
    // `n · base_cols · 8` crosses the cap (`base_cols == k` for thin-plate).
    // We size that with *many rows and few centers* rather than many centers:
    // the build's only super-linear compute is the O(k^3)
    // `thin_plate_kernel_constraint_nullspace` RRQR, and the downstream
    // orthogonality check streams the lazy design in O(n · k) without ever
    // materializing the n×k block. The previous (17_000 × 2_000) pin forced a
    // 2_000×2_000 dense RRQR that, under the unoptimized `[profile.test]` build
    // (opt-level 0), runs for minutes and risks the per-test CI timeout. With
    // `k = 256` the cubic factorization is microseconds, while `n = 200_000`
    // gives `200_000 · 256 · 8 ≈ 410 MiB` — above the previous pin's ~272 MiB,
    // so the lazy switch still fires under the same policy.
    let n = 200_000usize;
    let k = 256usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    let mut centers = Array2::<f64>::zeros((k, 1));
    for i in 0..n {
        data[[i, 0]] = i as f64 / (n - 1) as f64;
    }
    for j in 0..k {
        centers[[j, 0]] = j as f64 / (k - 1) as f64;
    }

    let spec = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "x".to_string(),
            feature_col: 0,
            feature_cols: vec![0],
            categorical_levels: vec![],
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "tps_x".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0],
                spec: ThinPlateBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::UserProvided(centers),
                    length_scale: 1.0,
                    double_penalty: false,
                    identifiability: SpatialIdentifiability::OrthogonalToParametric,
                    radial_reparam: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "large option-5 design", e));
    assert!(matches!(
        &design.smooth.term_designs[0],
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::Lazy(_))
    ));
    let mut c = Array2::<f64>::zeros((n, 2));
    c.column_mut(0).fill(1.0);
    c.column_mut(1).assign(&data.column(0));
    let rel = orthogonality_relative_residual_for_design(&design.smooth.term_designs[0], c.view())
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "orthogonality residual", e));
    assert!(rel <= 1e-8, "lazy option-5 residual too large: {rel}");
}

#[test]
fn spatial_frozen_transform_rebuild_is_exact_on_trainingrows() {
    let data = array![
        [0.0, 0.1],
        [0.2, 0.0],
        [0.3, 0.4],
        [0.5, 0.2],
        [0.7, 0.9],
        [1.0, 0.8],
    ];
    let fitspec = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "lin_x0".to_string(),
            feature_col: 0,
            feature_cols: vec![0],
            categorical_levels: vec![],
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "tps_xy".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0, 1],
                spec: ThinPlateBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    length_scale: 1.0,
                    double_penalty: false,
                    identifiability: SpatialIdentifiability::OrthogonalToParametric,
                    radial_reparam: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let fit_design = build_term_collection_design(data.view(), &fitspec).unwrap();
    let term_meta = &fit_design.smooth.terms[0].metadata;
    // `input_scale` is not optional here, and omitting it was this test's own
    // bug (#2636 family). `BasisMetadata::ThinPlate` declares that `centers`
    // live in the STANDARDIZED frame while `length_scale` is in ORIGINAL units,
    // and `scale_contract.rs` keys the difference off exactly this field:
    //
    //     let replay = stored_scale.is_some();
    //     ...
    //     if !replay { standardize_resolved_center_strategy(center_strategy, input_scale); }
    //
    // with the comment "its center matrix was emitted by the builder in the
    // realized standardized frame ... so scaling it again would double divide
    // fit-time geometry at prediction". Rebuilding with `input_scale: None`
    // makes `replay` false, so the already-standardized centers are divided a
    // second time — which is what this test was measuring as
    // `max_abs=1.4952067936546536`, not a defect in the frozen path.
    //
    // The production freezing path never does this: `design_freezing.rs` sets
    // `*input_scale = Some(*metadata_scale)` on exactly this rebuild. Carrying
    // the stored scale here makes the test replay what production replays.
    let (centers, length_scale, z, input_scale) = match term_meta {
        BasisMetadata::ThinPlate {
            centers,
            length_scale,
            identifiability_transform,
            input_scale,
            ..
        } => (
            centers.clone(),
            length_scale.original_value(),
            identifiability_transform
                .clone()
                .expect("fit-time Option 5 should store transform"),
            *input_scale,
        ),
        other => panic!("unexpected metadata variant: {other:?}"),
    };

    let frozenspec = TermCollectionSpec {
        linear_terms: fitspec.linear_terms.clone(),
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "tps_xy".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0, 1],
                spec: ThinPlateBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::UserProvided(centers),
                    length_scale,
                    double_penalty: false,
                    identifiability: SpatialIdentifiability::FrozenTransform { transform: z },
                    radial_reparam: None,
                },
                input_scale: Some(input_scale),
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let frozen_design = build_term_collection_design(data.view(), &frozenspec).unwrap();

    assert_eq!(
        fit_design.smooth.term_designs.len(),
        frozen_design.smooth.term_designs.len(),
        "frozen transform rebuild term count mismatch"
    );
    let max_abs = fit_design
        .smooth
        .term_designs
        .iter()
        .zip(frozen_design.smooth.term_designs.iter())
        .flat_map(|(a, b)| {
            let a_dense = a.to_dense();
            let b_dense = b.to_dense();
            assert_eq!(a_dense.dim(), b_dense.dim());
            a_dense
                .iter()
                .zip(b_dense.iter())
                .map(|(&x, &y)| (x - y).abs())
                .collect::<Vec<_>>()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs <= 1e-12,
        "frozen transform rebuild mismatch max_abs={max_abs}"
    );
}

#[test]
fn frozen_spatial_replay_preserves_standardized_length_scale_compensation() {
    let n = 16usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = 0.07 * i as f64 + 0.02 * (3.0 * t).sin();
        data[[i, 1]] = 4.0 * t + 0.35 * (5.0 * t).cos();
    }

    let tps_spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "tps_xy".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0, 1],
                spec: ThinPlateBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                    length_scale: 1.3,
                    double_penalty: true,
                    identifiability: SpatialIdentifiability::OrthogonalToParametric,
                    radial_reparam: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    assert_frozen_replay_matches_fit(data.view(), &tps_spec, "thin-plate");

    let matern_spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "matern_xy".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                    length_scale: gam_terms::basis::MaternLengthScale::fixed(1.1),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    assert_frozen_replay_matches_fit(data.view(), &matern_spec, "matern");

    let duchon_spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "duchon_xy".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0, 1],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                    length_scale: Some(1.4),
                    power: 5.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::OrthogonalToParametric,
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    assert_frozen_replay_matches_fit(data.view(), &duchon_spec, "duchon");
}

#[test]
fn term_collection_design_adds_random_effect_dummy_blockwithridge() {
    let data = array![
        [0.1, 0.0],
        [0.2, 1.0],
        [0.3, 0.0],
        [0.4, 2.0],
        [0.5, 1.0],
        [0.6, 2.0],
    ];
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![RandomEffectTermSpec {
            name: "id".to_string(),
            feature_col: 1,
            drop_first_level: false,
            penalized: true,
            frozen_levels: None,
            lenient_unseen: true,
        }],
        smooth_terms: vec![],
    };
    let design = build_term_collection_design(data.view(), &spec).unwrap();
    assert_eq!(design.intercept_range, 0..1);
    // 3 observed levels -> 3 dummy columns
    assert_eq!(design.design.ncols(), 4);
    assert_eq!(design.random_effect_ranges.len(), 1);
    assert_eq!(design.penalties.len(), 1);
    assert_eq!(design.nullspace_dims, vec![0]);
    let (_, range) = &design.random_effect_ranges[0];
    let dense = design.design.to_dense_cow();
    for i in 0..dense.nrows() {
        let row_sum: f64 = dense.slice(s![i, range.clone()]).sum();
        assert!((row_sum - 1.0).abs() < 1e-12);
    }
}

#[test]
fn term_collection_joint_duchon_carries_frozen_transform_into_metadata() {
    let n = 12usize;
    let d = 4usize;
    let mut data = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            data[[i, j]] = (i as f64) * 0.13 + (j as f64) * 0.17;
        }
    }

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "duchon_joint".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..d).collect(),
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    length_scale: Some(1.0),
                    power: 3.0,
                    nullspace_order: DuchonNullspaceOrder::Zero,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "term collection design", e));
    let term = &design.smooth.terms[0];
    assert_eq!(term.coeff_range.len(), 3);
    match &term.metadata {
        BasisMetadata::Duchon {
            identifiability_transform,
            ..
        } => {
            let z = identifiability_transform
                .as_ref()
                .unwrap_or_else(|| panic!("{} failed", "term collection should store frozen Duchon transform"));
            assert_eq!(z.nrows(), 4);
            assert_eq!(z.ncols(), 3);
        }
        other => panic!("expected Duchon metadata, got {other:?}"),
    }
}

#[test]
fn frozen_joint_maternspec_rebuild_keeps_adaptive_cache_in_sync() {
    let n = 12usize;
    let d = 2usize;
    let mut data = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        data[[i, 0]] = i as f64 * 0.13;
        data[[i, 1]] = (i as f64 * 0.17).sin();
    }

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "matern_joint".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: (0..d).collect(),
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                    length_scale: gam_terms::basis::MaternLengthScale::fixed(1.0),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "base design", e));
    let frozen = freeze_term_collection_from_design(&spec, &design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze spec", e));
    let rebuilt = build_term_collection_design(data.view(), &frozen).unwrap_or_else(|e| panic!("{} failed: {:?}", "rebuilt design", e));
    let caches =
        extract_spatial_operator_runtime_caches(&frozen, &rebuilt).unwrap_or_else(|e| panic!("{} failed: {:?}", "adaptive caches", e));
    assert_eq!(caches.len(), 1);
    assert_eq!(caches[0].termname, "matern_joint");
    assert_eq!(rebuilt.smooth.terms.len(), 1);
    assert!(!rebuilt.smooth.terms[0].coeff_range.is_empty());
}

#[test]
fn centered_tensor_penalties_canonicalize_in_transformed_basis_width() {
    let n = 16usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = 0.5 + 0.25 * (2.0 * std::f64::consts::PI * t).sin();
    }

    let tensor_term = SmoothTermSpec {
            frozen_parametric_residualization: None,
        name: "te_centered".to_string(),
        basis: SmoothBasisSpec::TensorBSpline {
            feature_cols: vec![0, 1],
            spec: TensorBSplineSpec {
                marginalspecs: vec![
                    BSplineBasisSpec {
                        degree: 3,
                        penalty_order: 2,
                        knotspec: BSplineKnotSpec::Generate {
                            data_range: (0.0, 1.0),
                            num_internal_knots: 3,
                        },
                        double_penalty: false,
                        identifiability: BSplineIdentifiability::default(),
                        boundary_conditions: Default::default(),
                        boundary: OneDimensionalBoundary::Open,
                    },
                    BSplineBasisSpec {
                        degree: 3,
                        penalty_order: 2,
                        knotspec: BSplineKnotSpec::Generate {
                            data_range: (0.0, 1.0),
                            num_internal_knots: 2,
                        },
                        double_penalty: false,
                        identifiability: BSplineIdentifiability::default(),
                        boundary_conditions: Default::default(),
                        boundary: OneDimensionalBoundary::Open,
                    },
                ],
                periods: Vec::new(),
                double_penalty: false,
                identifiability: TensorBSplineIdentifiability::default(),
                penalty_decomposition: Default::default(),
            },
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    };
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![tensor_term],
    };
    let design = build_term_collection_design(data.view(), &spec).unwrap();
    let penalty_specs = design
        .penalties
        .iter()
        .map(PenaltySpec::from_blockwise_ref)
        .collect::<Vec<_>>();
    let (canonical, _) = gam_terms::construction::canonicalize_penalty_specs(
        &penalty_specs,
        &design.nullspace_dims,
        design.design.ncols(),
        "centered tensor penalty regression",
    )
    .unwrap();
    for cp in canonical {
        assert_eq!(cp.root.ncols(), cp.col_range.len());
        assert_eq!(cp.local.nrows(), cp.col_range.len());
        assert_eq!(cp.local.ncols(), cp.col_range.len());
    }
}

#[test]
fn periodic_bspline_margin_wraps_exactly_at_period() {
    let x = array![0.0, 1.25, 2.5, 3.75, 7.0, 8.25];
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::PeriodicUniform {
            data_range: (0.0, 1.0),
            num_basis: 8,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary_conditions: Default::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let built = build_bspline_basis_1d(x.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "periodic bspline", e));
    let dense = built.design.to_dense();
    assert_eq!(dense.ncols(), 8);
    for j in 0..dense.ncols() {
        assert!(
            (dense[[0, j]] - dense[[4, j]]).abs() < 1e-12,
            "seam row differs at column {j}: {} vs {}",
            dense[[0, j]],
            dense[[4, j]]
        );
        assert!(
            (dense[[1, j]] - dense[[5, j]]).abs() < 1e-12,
            "wrapped row differs at column {j}: {} vs {}",
            dense[[1, j]],
            dense[[5, j]]
        );
    }
    for row in dense.rows() {
        assert!((row.sum() - 1.0).abs() < 1e-12);
    }
    assert_eq!(built.active_penalties[0].nullity, 1);
}

#[test]
fn tensor_bspline_supports_two_periodic_margins_as_torus() {
    let data = array![[0.0, 0.0], [7.0, 0.0], [0.0, 24.0], [7.0, 24.0], [1.5, 6.0]];
    let spec_day = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::PeriodicUniform {
            data_range: (0.0, 7.0),
            num_basis: 7,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary_conditions: Default::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let spec_hour = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::PeriodicUniform {
            data_range: (0.0, 24.0),
            num_basis: 8,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary_conditions: Default::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let spec_collection = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "te_day_hour".to_string(),
            basis: SmoothBasisSpec::TensorBSpline {
                feature_cols: vec![0, 1],
                spec: TensorBSplineSpec {
                    marginalspecs: vec![spec_day, spec_hour],
                    periods: Vec::new(),
                    double_penalty: false,
                    identifiability: TensorBSplineIdentifiability::None,
                    penalty_decomposition: Default::default(),
                },
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let design = build_term_collection_design(data.view(), &spec_collection)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "periodic tensor design", e));
    let sd = &design.smooth;
    let dense = sd.term_designs[0].to_dense();
    assert_eq!(dense.ncols(), 56);
    for j in 0..dense.ncols() {
        assert!((dense[[0, j]] - dense[[1, j]]).abs() < 1e-12);
        assert!((dense[[0, j]] - dense[[2, j]]).abs() < 1e-12);
        assert!((dense[[0, j]] - dense[[3, j]]).abs() < 1e-12);
    }
    assert_eq!(sd.penalties.len(), 2);
    assert!(sd.penalties.iter().all(|p| p.local.nrows() == 56));

    let frozen = freeze_term_collection_from_design(&spec_collection, &design)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze periodic tensor", e));
    match &frozen.smooth_terms[0].basis {
        SmoothBasisSpec::TensorBSpline { spec, .. } => {
            assert!(matches!(
                spec.marginalspecs[0].knotspec,
                BSplineKnotSpec::PeriodicUniform { data_range, .. }
                    if (data_range.1 - data_range.0 - 7.0).abs() < 1e-9
            ));
            assert!(matches!(
                spec.marginalspecs[1].knotspec,
                BSplineKnotSpec::PeriodicUniform { data_range, .. }
                    if (data_range.1 - data_range.0 - 24.0).abs() < 1e-9
            ));
        }
        _ => panic!("expected tensor spec"),
    }
}

// NOTE: five blocks were removed here. `design_assembly_constraint_tests.rs` and
// `spatial_length_scale_monotone_tests.rs` are both `include!`d into this one
// module, and each wrapped its own `#[cfg(test)] mod`, so byte-identical copies
// did not collide -- they silently double-counted, building and running 814
// duplicated lines and reporting one defect as two failures. The surviving
// copies live in `spatial_length_scale_monotone_tests.rs` and
// `psi_gram_tensor_fast_path_tests.rs`.

/// Drives a two-block exact-joint κ optimization with the canonical
/// zero-work test closures (cost = total design ncols + penalty count;
/// flat gradient/Hessian; trivial EFS) and returns the resolved result.
/// Shared verbatim across the Matérn- and Duchon-freezing pins; only the
/// final `.expect` diagnostic differs, passed via `expect_msg`.
///
/// `pub(super)` so the sibling `adaptive_bounded_duchon_tests` #1601 re-home
/// (its `exact_joint_two_block_spatial_length_scale_freezes_duchon_centers` pin)
/// shares the single definition through the `drivers` parent scope.
pub(super) fn run_two_block_exact_joint_optimize(
    data: ArrayView2<'_, f64>,
    meanspec: &TermCollectionSpec,
    noisespec: &TermCollectionSpec,
    expect_msg: &str,
) -> SpatialLengthScaleOptimizationResult<f64> {
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        max_outer_iter: 1,
        rel_tol: 1e-6,
        pilot_subsample_threshold: 0,
        ..SpatialLengthScaleOptimizationOptions::default()
    };
    let joint_setup = two_block_exact_joint_hyper_setup(meanspec, noisespec, &kappa_options);
    let theta_dim = joint_setup.theta0().len();

    let mean_terms = spatial_length_scale_term_indices(meanspec);
    let noise_terms = spatial_length_scale_term_indices(noisespec);
    let policy = gam_model_api::families::custom_family::OuterDerivativePolicy {
        capability: gam_problem::ExactOuterDerivativeOrder::Second,
        predicted_hessian_work: 0,
        predicted_gradient_work: 0,
        // Test-style construction with zero predicted work — these
        // paths never engage staged-κ, so the capability bit is
        // moot. Keep `false` as the safe default.
        subsample_capable: false,
    };
    optimize_spatial_length_scale_exact_joint(
        data,
        &[meanspec.clone(), noisespec.clone()],
        &[mean_terms, noise_terms],
        &kappa_options,
        &joint_setup,
        gam_problem::SeedRiskProfile::Gaussian,
        true,
        true,
        false,
        None,
        policy,
        |theta, specs, designs, _| {
            assert_eq!(theta.len(), theta_dim);
            assert_eq!(specs.len(), 2);
            Ok(designs[0].design.ncols() as f64
                + designs[1].design.ncols() as f64
                + designs[0].penalties.len() as f64
                + designs[1].penalties.len() as f64)
        },
        |theta, specs, designs, eval_mode, _, _| {
            assert_eq!(theta.len(), theta_dim);
            assert_eq!(specs.len(), 2);
            assert!(!designs.is_empty());
            Ok(ExactJointEvaluation {
                objective: 0.0,
                gradient: Array1::zeros(theta_dim),
                hessian: if matches!(
                    eval_mode,
                    gam_solve::estimate::reml::reml_outer_engine::EvalMode::ValueGradientHessian
                ) {
                    gam_problem::HessianValue::Dense(Array2::zeros((
                        theta_dim, theta_dim,
                    )))
                } else {
                    gam_problem::HessianValue::Unavailable
                },
                mode: (),
            })
        },
        |theta, specs, designs, _| {
            assert_eq!(theta.len(), theta_dim);
            assert_eq!(specs.len(), 2);
            assert!(!designs.is_empty());
            Ok(ExactJointEfsEvaluation {
                evaluation: gam_problem::EfsEval {
                    cost: 0.0,
                    steps: vec![0.0; theta_dim],
                    beta: None,
                    psi_gradient: None,
                    psi_indices: None,
                    inner_hessian_scale: None,
                    logdet_enclosure_gap: None,
                    consecutive_restored_incumbents: None,
                },
                mode: (),
            })
        },
        |_: &Array1<f64>| Ok(gam_solve::rho_optimizer::SeedOutcome::NoSlot),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", expect_msg, e))
}

#[test]
fn staged_exact_joint_outer_reoptimizes_and_certifies_the_full_row_measure() {
    let n = 24usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        data[[i, 0]] = i as f64 / (n as f64 - 1.0);
        data[[i, 1]] = (i as f64 * 0.19).sin();
    }

    let matern_term = |name: &str| SmoothTermSpec {
            frozen_parametric_residualization: None,
        name: name.to_string(),
        basis: SmoothBasisSpec::Matern {
            feature_cols: vec![0, 1],
            spec: MaternBasisSpec {
                periodic: None,
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                length_scale: gam_terms::basis::MaternLengthScale::fixed(1.0),
                nu: MaternNu::FiveHalves,
                include_intercept: false,
                double_penalty: true,
                identifiability: MaternIdentifiability::CenterSumToZero,
                // An anisotropic Matérn contributes explicit spatial outer
                // coordinates; an isotropic term's seeded range is fixed here.
                aniso_log_scales: Some(vec![0.0, 0.0]),
            },
            input_scale: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    };
    let meanspec = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![matern_term("mean_matern")],
    };
    let noisespec = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![matern_term("noise_matern")],
    };
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        enabled: true,
        max_outer_iter: 16,
        rel_tol: 1.0e-7,
        pilot_subsample_threshold: 0,
        ..SpatialLengthScaleOptimizationOptions::default()
    };
    let joint_setup =
        two_block_exact_joint_hyper_setup(&meanspec, &noisespec, &kappa_options);
    let theta_dim = joint_setup.theta0().len();
    assert!(theta_dim > 0, "fixture must expose spatial outer coordinates");

    struct TerminalEvidence {
        serial: usize,
        theta_bits: Vec<u64>,
        objective_bits: u64,
        full_rows: bool,
    }

    let pilot_evals = std::cell::Cell::new(0usize);
    let exact_evals = std::cell::Cell::new(0usize);
    let evaluation_serial = std::cell::Cell::new(0usize);
    let fit_entry_serial = std::cell::Cell::new(None::<usize>);
    let policy = gam_model_api::families::custom_family::OuterDerivativePolicy {
        capability: gam_problem::ExactOuterDerivativeOrder::Second,
        predicted_hessian_work: u128::MAX,
        predicted_gradient_work: u128::MAX,
        // Force the staged path at tiny test n; the RowSet variant, rather
        // than its cardinality, distinguishes the two analytic objectives.
        subsample_capable: true,
    };

    let solved = optimize_spatial_length_scale_exact_joint(
        data.view(),
        &[meanspec.clone(), noisespec.clone()],
        &[
            spatial_length_scale_term_indices(&meanspec),
            spatial_length_scale_term_indices(&noisespec),
        ],
        &kappa_options,
        &joint_setup,
        gam_problem::SeedRiskProfile::Gaussian,
        true,
        true,
        true,
        None,
        policy,
        |theta,
         specs,
         designs,
         provenance: SpatialFitProvenance<'_, TerminalEvidence>| {
            assert_eq!(specs.len(), 2);
            assert_eq!(designs.len(), 2);
            let SpatialFitProvenance::Certified { outer, mode } = provenance else {
                panic!("staged spatial fit must receive certified outer provenance");
            };
            assert_eq!(outer.rho(), theta);
            assert!(outer.criterion_certificate().certifies());
            assert!(
                mode.full_rows,
                "the pilot coefficient mode must be revoked before final fit assembly",
            );
            assert_eq!(
                mode.theta_bits,
                theta.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                "the moved coefficient mode must belong to the certified full theta",
            );
            assert_eq!(
                mode.objective_bits,
                outer.final_value().to_bits(),
                "the moved coefficient mode must have produced the certified objective",
            );
            assert_eq!(
                mode.serial,
                evaluation_serial.get(),
                "fit assembly must receive the latest exact evaluation's move-only carrier",
            );
            fit_entry_serial.set(Some(evaluation_serial.get()));
            Ok(theta.clone())
        },
        |theta, specs, designs, eval_mode, row_set, owned_mode| {
            assert_eq!(theta.len(), theta_dim);
            assert_eq!(specs.len(), 2);
            assert_eq!(designs.len(), 2);
            let (center, full_rows) = match row_set {
                gam_problem::outer_subsample::RowSet::Subsample { .. } => {
                    pilot_evals.set(pilot_evals.get() + 1);
                    (2.0, false)
                }
                gam_problem::outer_subsample::RowSet::All => {
                    exact_evals.set(exact_evals.get() + 1);
                    (-1.0, true)
                }
            };
            let gradient = theta.mapv(|value| value - center);
            let cost: f64 = 0.5 * gradient.dot(&gradient);
            let hessian = if matches!(
                eval_mode,
                gam_solve::estimate::reml::reml_outer_engine::EvalMode::ValueGradientHessian
            ) {
                let mut dense = Array2::<f64>::zeros((theta_dim, theta_dim));
                dense.diag_mut().fill(1.0);
                gam_problem::HessianValue::Dense(dense)
            } else {
                gam_problem::HessianValue::Unavailable
            };
            let mode = if let Some(mode) = owned_mode {
                assert!(
                    !matches!(
                        eval_mode,
                        gam_solve::estimate::reml::reml_outer_engine::EvalMode::ValueOnly
                    ),
                    "a value-only evaluation cannot consume its own terminal mode",
                );
                assert_eq!(
                    mode.theta_bits,
                    theta.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    "owned mode must belong to the exact requested theta",
                );
                assert_eq!(
                    mode.objective_bits,
                    cost.to_bits(),
                    "owned mode must carry the exact requested objective",
                );
                assert_eq!(
                    mode.full_rows, full_rows,
                    "owned mode cannot cross the pilot/full-data boundary",
                );
                mode
            } else {
                let serial = evaluation_serial.get() + 1;
                evaluation_serial.set(serial);
                TerminalEvidence {
                    serial,
                    theta_bits: theta.iter().map(|value| value.to_bits()).collect(),
                    objective_bits: cost.to_bits(),
                    full_rows,
                }
            };
            Ok(ExactJointEvaluation {
                objective: cost,
                gradient,
                hessian,
                mode,
            })
        },
        |_, _, _, _| {
            Err("fixed-point callback must stay disabled in staged regression".to_string())
        },
        |_| Ok(gam_solve::rho_optimizer::SeedOutcome::NoSlot),
    )
    .expect("pilot checkpoint must continue to the exact full-data optimum");

    assert!(pilot_evals.get() > 0, "pilot objective was never evaluated");
    assert!(exact_evals.get() > 0, "exact objective was never evaluated");
    assert_eq!(
        fit_entry_serial.get(),
        Some(evaluation_serial.get()),
        "no exact profile evaluation may replay during or after owned-mode fit assembly",
    );
    assert!(
        solved.fit.iter().all(|value| (*value + 1.0).abs() <= 1.0e-6),
        "returned pilot optimum instead of exact optimum: {:?}",
        solved.fit,
    );
    let outer = solved
        .certified_outer
        .as_ref()
        .expect("optimized spatial result must retain its certificate");
    assert!(outer.criterion_certificate().certifies());
    assert!(outer.final_value().abs() <= 1.0e-10);
}

#[test]
fn exact_joint_two_block_spatial_length_scale_freezes_matern_centers() {
    let n = 40usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let x0 = i as f64 / (n as f64 - 1.0);
        let x1 = (i as f64 * 0.21).sin();
        data[[i, 0]] = x0;
        data[[i, 1]] = x1;
    }

    // ANISOTROPIC Matérn (`aniso_log_scales = Some`): the joint κ/η outer
    // optimizer only engages for anisotropic spatial terms (#519 —
    // isotropic Matérn anchors its data-seeded κ and learns smoothness
    // through ρ alone, so it contributes no κ axis). This test exercises
    // the joint-optimizer center-freezing path, so it must carry per-axis
    // anisotropy scales to produce the κ/η hyper axes it is asserting on.
    let matern_term = |name: &str, length_scale: f64| SmoothTermSpec {
            frozen_parametric_residualization: None,
        name: name.to_string(),
        basis: SmoothBasisSpec::Matern {
            feature_cols: vec![0, 1],
            spec: MaternBasisSpec {
                periodic: None,
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                length_scale: gam_terms::basis::MaternLengthScale::fixed(length_scale),
                nu: MaternNu::FiveHalves,
                include_intercept: false,
                double_penalty: true,
                identifiability: MaternIdentifiability::CenterSumToZero,
                aniso_log_scales: Some(vec![0.0, 0.0]),
            },
            input_scale: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    };

    let meanspec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![matern_term("mean_matern", 0.8)],
    };
    let noisespec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![matern_term("noise_matern", 1.1)],
    };

    let solved = run_two_block_exact_joint_optimize(
        data.view(),
        &meanspec,
        &noisespec,
        "exact joint two-block κ optimization should succeed",
    );

    for resolved in [&solved.resolved_specs[0], &solved.resolved_specs[1]] {
        match &resolved.smooth_terms[0].basis {
            SmoothBasisSpec::Matern { spec, .. } => {
                assert!(matches!(
                    spec.center_strategy,
                    CenterStrategy::UserProvided(_)
                ));
                assert!(matches!(
                    spec.identifiability,
                    MaternIdentifiability::FrozenTransform { .. }
                ));
            }
            _ => panic!("expected Matérn term"),
        }
    }
}

#[test]
fn spatial_aniso_joint_exact_hessian_materializes_small_case() {
    let n = 18usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (0.41 * i as f64).sin();
    }
    let y = Array1::from_iter((0..n).map(|i| {
        let t = i as f64 / (n as f64 - 1.0);
        0.4 + (2.0 * std::f64::consts::PI * t).sin()
    }));
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "matern_aniso".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 5 },
                    length_scale: gam_terms::basis::MaternLengthScale::fixed(0.85),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: Some(vec![0.2, -0.2]),
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let fit_opts = FitOptions {
        compute_inference: false,
        max_iter: 120,
        tol: 1e-10,
        ..FitOptions::default()
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "design", e));
    let frozen = freeze_term_collection_from_design(&spec, &design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze", e));
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
    assert_eq!(dims_per_term, vec![2]);
    let rho_dim = design.penalties.len();
    let log_kappa0 = SpatialLogKappaCoords::from_length_scales_aniso(
        &frozen,
        &spatial_terms,
        &SpatialLengthScaleOptimizationOptions::default(),
    );
    let mut theta = Array1::<f64>::zeros(rho_dim + log_kappa0.as_array().len());
    for j in 0..rho_dim {
        theta[j] = -0.15 + 0.07 * j as f64;
    }
    theta.slice_mut(s![rho_dim..]).assign(log_kappa0.as_array());

    let external_opts =
        external_opts_for_design(&LikelihoodSpec::gaussian_identity(), &design, &fit_opts);
    let mut cache = SingleBlockExactJointDesignCache::new(
        data.view(),
        frozen,
        design.clone(),
        spatial_terms,
        rho_dim,
        dims_per_term,
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "single-block cache", e));
    let mut evaluator = gam_solve::estimate::ExternalJointHyperEvaluator::new(
        y.view(),
        weights.view(),
        &design.design,
        offset.view(),
        &design.penalties,
        &external_opts,
        "small aniso Hessian finite-difference evaluator",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "evaluator", e));

    let eval_at = |theta: &Array1<f64>,
                   cache: &mut SingleBlockExactJointDesignCache<'_>,
                   evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>,
                   order: gam_model_api::OuterEvalOrder| {
        cache.ensure_theta(theta).unwrap_or_else(|e| panic!("{} failed: {:?}", "theta applied", e));
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &cache.spatial_terms,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "hyper dirs build", e))
        .expect("hyper dirs present");
        evaluate_joint_reml_outer_eval_at_theta(
            evaluator,
            cache.design(),
            theta,
            rho_dim,
            hyper_dirs,
            None,
            order,
            None,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "outer eval", e))
    };

    let (_, gradient, hessian_result) = eval_at(
        &theta,
        &mut cache,
        &mut evaluator,
        gam_model_api::OuterEvalOrder::ValueGradientHessian,
    );
    let hessian = hessian_result
        .materialize_dense()
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "hessian materializes", e))
        .expect("hessian present");
    assert_eq!(hessian.nrows(), theta.len());
    assert_eq!(hessian.ncols(), theta.len());
    assert!(hessian.iter().all(|value| value.is_finite()));
    assert!(gradient.iter().all(|value| value.is_finite()));

    let symmetry_diff = max_abs_diff_matrix(&hessian, &hessian.t().to_owned());
    assert!(
        symmetry_diff <= 1e-10,
        "small aniso exact Hessian should be symmetric, max diff={symmetry_diff}"
    );
    let psi_block = hessian.slice(s![rho_dim.., rho_dim..]).to_owned();
    assert!(
        psi_block.iter().any(|value| value.abs() > 1e-10),
        "small aniso exact Hessian should carry non-zero ψ curvature"
    );
}

// The eight iso-κ outer-gradient FD tests that used to live here, and the
// `iso_kappa_fd_variant_driver` behind them, were a VERBATIM fork of the
// copy in `iso_kappa_reml_gradient_fd_tests` (the module the #1521 carve
// re-homed them into) — same names, same fixtures, same assertions,
// differing only in how they spell `OuterEvalOrder`. The fork carried its
// own hard-wired finite-difference step, with a comment instructing the
// reader to keep it "in step with the live copy" by hand. #2461 is what
// that step costs, so a second unmaintained copy of it is a second copy of
// the defect; deleted rather than re-synchronised. The live module owns
// these tests and now measures every component with a self-certifying
// oracle instead of a fixed step.

/// Parity test for the unified exact-spatial joint optimizer (issue #427).
///
/// Before unification, anisotropic and isotropic spatial joint optimization
/// were two near-identical functions that differed only in diagnostic
/// labels. The shared engine `run_exact_joint_spatial_optimization` now
/// drives both, selected by `SpatialHyperKind`. For a 1-D spatial term the
/// two coordinate kinds are mathematically identical — `dims_per_term ==
/// [1]`, so each carries exactly one log-scale coordinate per term and both
/// route the same θ through `try_build_spatial_log_kappa_hyper_dirs`. The
/// converged hyperparameters and certified REML cost must therefore agree to
/// numerical round-off when the engine is invoked under either kind with
/// identical inputs. Any divergence would mean the kind discriminator leaked
/// into the numerics rather than staying confined to labels.
#[test]
fn exact_spatial_joint_engine_aniso_iso_parity_1d() {
    let n = 80usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        let eta = 1.4 * (2.0 * std::f64::consts::PI * t).sin() + 0.5 * (t - 0.5);
        y[i] = eta + 0.7 * (3.7 * (i as f64) + 1.0).sin();
    }
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let family = LikelihoodSpec::gaussian_identity();

    // 1-D Duchon term: a single log-scale axis (dims_per_term == [1]), the
    // shared geometry across both coordinate kinds.
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "parity_1d".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::all_active(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let fit_opts = FitOptions {
        compute_inference: false,
        max_iter: 200,
        tol: 1e-12,
        ..FitOptions::default()
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "design", e));
    let frozen = freeze_term_collection_from_design(&spec, &design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze", e));
    let frozen_design = build_term_collection_design(data.view(), &frozen).unwrap_or_else(|e| panic!("{} failed: {:?}", "frozen design", e));
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    assert_eq!(spatial_terms.len(), 1, "expect a single spatial term");
    let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
    assert_eq!(dims_per_term, vec![1], "expect one log-scale axis");
    let rho_dim = frozen_design.penalties.len();
    assert!(rho_dim >= 1, "expect at least one penalty block");

    // Construct the joint setup exactly as the production caller does,
    // shared verbatim between the two engine invocations so that any
    // difference in the result can only come from the coordinate kind.
    const JOINT_RHO_BOUND: f64 = 12.0;
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();
    let log_kappa0 =
        SpatialLogKappaCoords::from_length_scales(&frozen, &spatial_terms, &kappa_options);
    let log_kappa_lower = SpatialLogKappaCoords::lower_bounds_from_data(
        data.view(),
        &frozen,
        &spatial_terms,
        &kappa_options,
    )
    .expect("lower isotropic-scale bounds");
    let log_kappa_upper = SpatialLogKappaCoords::upper_bounds_from_data(
        data.view(),
        &frozen,
        &spatial_terms,
        &kappa_options,
    )
    .expect("upper isotropic-scale bounds");
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);
    let setup = ExactJointHyperSetup::new(
        Array1::<f64>::zeros(rho_dim), // log λ seed (λ = 1)
        Array1::<f64>::from_elem(rho_dim, -JOINT_RHO_BOUND),
        Array1::<f64>::from_elem(rho_dim, JOINT_RHO_BOUND),
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    );
    let theta0 = setup.theta0();
    let lower = setup.lower();
    let upper = setup.upper();

    let run = |kind: SpatialHyperKind| -> (Array1<f64>, f64) {
        run_exact_joint_spatial_optimization(
            kind,
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &frozen,
            &frozen_design,
            family.clone(),
            &fit_opts,
            &spatial_terms,
            &dims_per_term,
            &theta0,
            &lower,
            &upper,
            rho_dim,
            &kappa_options,
        )
        .map(|(theta_star, final_value, _seed_value, _timing)| (theta_star, final_value))
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact joint spatial optimization", e))
    };

    let (theta_aniso, value_aniso) = run(SpatialHyperKind::Anisotropic);
    let (theta_iso, value_iso) = run(SpatialHyperKind::Isotropic);

    assert_eq!(
        theta_aniso.len(),
        theta_iso.len(),
        "converged θ dimension must match across coordinate kinds"
    );
    // In 1-D the two kinds are numerically identical: the only difference is
    // diagnostic labels, so converged hyperparameters and the certified REML
    // cost must agree to round-off. No tolerance weakening — this is an
    // equality, not an approximation.
    for j in 0..theta_aniso.len() {
        let diff = (theta_aniso[j] - theta_iso[j]).abs();
        assert!(
            diff <= 1e-9 * (1.0 + theta_aniso[j].abs()),
            "θ[{j}] differs across kinds: aniso={:+.12e} iso={:+.12e} diff={:.3e}",
            theta_aniso[j],
            theta_iso[j],
            diff,
        );
    }
    let value_diff = (value_aniso - value_iso).abs();
    assert!(
        value_diff <= 1e-9 * (1.0 + value_aniso.abs()),
        "final REML value differs across kinds: aniso={:+.12e} iso={:+.12e} diff={:.3e}",
        value_aniso,
        value_iso,
        value_diff,
    );
    assert!(
        value_aniso.is_finite() && value_iso.is_finite(),
        "both kinds must produce a finite certified REML cost"
    );
}

/// #1033b invariance gate: the certified ψ-Gram tensor lane must produce
/// the SAME REML cost and gradient as the exact per-trial streamed path at
/// every in-window ψ. The tensor lane installs an n-free assembled
/// `GaussianFixedCache` after `reset_surface` (so the inner Gaussian PLS
/// skips the O(n·p²) Gram re-stream); the streamed path lazily builds the
/// same cache from the realized X. Both feed the identical inner solver, so
/// a frame-correct wiring is an EQUALITY to certification round-off, not an
/// approximation. Any divergence here means the conditioned-frame handoff
/// (`build_and_set_psi_gram_tensor` → `install_gaussian_fixed_cache`) has a
/// frame bug. The two evaluators are byte-identical except that one carries
/// the tensor — the only thing the test varies is the lane.
///
/// Runs on PRODUCTION geometry (`input_scale: None`, #1215 1-D standardization
/// to unit spread). The per-ψ amplitude normalization (#1216) is what makes the
/// Chebyshev tail certify on the wide standardized window so the n-free tensor
/// actually attaches here (`assert!(attached)`).
#[test]
fn psi_gram_tensor_lane_matches_streamed_reml_cost_and_gradient() {
    use gam_model_api::OuterEvalOrder;

    // ── 1-D isotropic Duchon Gaussian fixture, n = 600. coord_dim == 1
    // routes through the exact-joint spatial optimizer's tensor gate; the
    // Gaussian-identity family makes the GaussianFixedCache eligible. ──
    let n = 600usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        let signal = 1.2 * (2.0 * std::f64::consts::PI * t).sin() + 0.4 * (t - 0.5);
        // Deterministic pseudo-noise so the fit is non-trivial but the test
        // is reproducible.
        let noise = 0.15 * (((i as f64) * 12.9898).sin() * 43758.547).fract();
        y[i] = signal + noise;
    }
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let family = LikelihoodSpec::gaussian_identity();

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "psi_tensor_invariance".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::all_active(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let fit_opts = FitOptions {
        compute_inference: false,
        max_iter: 200,
        tol: 1e-12,
        ..FitOptions::default()
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "design", e));
    let frozen = freeze_term_collection_from_design(&spec, &design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze", e));
    let frozen_design = build_term_collection_design(data.view(), &frozen).unwrap_or_else(|e| panic!("{} failed: {:?}", "frozen design", e));
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    assert_eq!(spatial_terms.len(), 1, "expect a single spatial term");
    let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
    assert_eq!(
        dims_per_term,
        vec![1],
        "expect one log-scale axis (coord_dim == 1)"
    );
    let rho_dim = frozen_design.penalties.len();
    assert!(rho_dim >= 1, "expect at least one penalty block");

    // ψ window straight from the production bounds helpers.
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();
    let log_kappa0 =
        SpatialLogKappaCoords::from_length_scales(&frozen, &spatial_terms, &kappa_options);
    let log_kappa_lower = SpatialLogKappaCoords::lower_bounds_from_data(
        data.view(),
        &frozen,
        &spatial_terms,
        &kappa_options,
    )
    .expect("lower isotropic-scale bounds");
    let log_kappa_upper = SpatialLogKappaCoords::upper_bounds_from_data(
        data.view(),
        &frozen,
        &spatial_terms,
        &kappa_options,
    )
    .expect("upper isotropic-scale bounds");
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);
    const JOINT_RHO_BOUND: f64 = 12.0;
    let setup = ExactJointHyperSetup::new(
        Array1::<f64>::zeros(rho_dim),
        Array1::<f64>::from_elem(rho_dim, -JOINT_RHO_BOUND),
        Array1::<f64>::from_elem(rho_dim, JOINT_RHO_BOUND),
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    );
    let theta0 = setup.theta0();
    let lower = setup.lower();
    let upper = setup.upper();
    let psi_lo = lower[rho_dim];
    let psi_hi = upper[rho_dim];
    assert!(psi_hi > psi_lo, "ψ window must be non-degenerate");

    // Shared realizer cache — both evaluators consume the SAME realized
    // design at each θ (the streamed path uses it directly; the tensor
    // path used it once to build the expansion).
    let make_cache = || {
        SingleBlockExactJointDesignCache::new(
            data.view(),
            frozen.clone(),
            frozen_design.clone(),
            spatial_terms.clone(),
            rho_dim,
            dims_per_term.clone(),
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "design cache", e))
    };
    let external_opts = external_opts_for_design(&family, &frozen_design, &fit_opts);

    let mut streamed_eval = gam_solve::estimate::ExternalJointHyperEvaluator::new(
        y.view(),
        weights.view(),
        &frozen_design.design,
        offset.view(),
        &frozen_design.penalties,
        &external_opts,
        "psi_tensor_invariance/streamed",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "streamed evaluator", e));

    let mut tensor_eval = gam_solve::estimate::ExternalJointHyperEvaluator::new(
        y.view(),
        weights.view(),
        &frozen_design.design,
        offset.view(),
        &frozen_design.penalties,
        &external_opts,
        "psi_tensor_invariance/tensor",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "tensor evaluator", e));

    // Attach the certified tensor to ONE evaluator, exactly as production
    // does: the realizer returns the RAW realized design at ψ; the
    // evaluator threads its own (fixed, ψ-invariant) conditioning inside
    // the build so the assembled Gram lives in the streamed frame.
    let z = Array1::from_iter(y.iter().zip(offset.iter()).map(|(yi, oi)| yi - oi));
    let attached = {
        let mut build_cache = make_cache();
        let theta_probe_base = theta0.clone();
        tensor_eval.build_and_set_psi_gram_tensor(
            |psi| {
                let mut theta_probe = theta_probe_base.clone();
                theta_probe[rho_dim] = psi;
                build_cache
                    .ensure_theta(&theta_probe)
                    .map_err(|error| error.to_string())?;
                Ok(build_cache.design().design.clone())
            },
            weights.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
    };
    // This fixture must EXERCISE the tensor lane: a fall-through would make
    // the equality below trivially true and prove nothing. An analytic
    // Duchon design over the production ψ window is exactly the
    // geometric-decay case the certificate is built for, so we require the
    // attach. If a future basis change makes it refuse, this fails loudly
    // (telling us to re-derive the window) rather than silently passing.
    assert!(
        attached,
        "ψ-gram tensor failed to certify over the production window \
             [{psi_lo:.3}, {psi_hi:.3}]; the invariance test would be vacuous"
    );

    // One shared realizer drives both lanes per θ.
    let mut stream_cache = make_cache();
    let mut tensor_cache = make_cache();

    // Sample several in-window ψ (including endpoints' interior) crossed
    // with a couple ρ values, so the comparison spans the whole certified
    // window and is not an accident of one operating point.
    let psi_samples = [
        psi_lo + 0.10 * (psi_hi - psi_lo),
        psi_lo + 0.37 * (psi_hi - psi_lo),
        0.5 * (psi_lo + psi_hi),
        psi_lo + 0.78 * (psi_hi - psi_lo),
        psi_hi - 0.05 * (psi_hi - psi_lo),
    ];
    let rho_samples = [
        Array1::<f64>::from_elem(rho_dim, -1.5),
        Array1::<f64>::from_elem(rho_dim, 0.5),
    ];

    // Evaluate cost + gradient (+ optional Hessian) from both lanes at one θ.
    // When `with_hessian` is true the Hessian (if analytic) is returned as
    // Some(H); the caller compares it pair-wise across lanes.
    let eval_one = |evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>,
                    cache: &mut SingleBlockExactJointDesignCache<'_>,
                    theta: &Array1<f64>,
                    with_hessian: bool|
     -> (f64, Array1<f64>, Option<Array2<f64>>) {
        use gam_problem::HessianValue;
        cache.ensure_theta(theta).unwrap_or_else(|e| panic!("{} failed: {:?}", "ensure_theta", e));
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &spatial_terms,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "hyper_dirs build", e))
        .expect("hyper_dirs present");
        let design_revision = Some(cache.design_revision());
        let order = if with_hessian {
            OuterEvalOrder::ValueGradientHessian
        } else {
            OuterEvalOrder::ValueAndGradient
        };
        let (cost, grad, hess) = evaluate_joint_reml_outer_eval_at_theta(
            evaluator,
            cache.design(),
            theta,
            rho_dim,
            hyper_dirs,
            None,
            order,
            design_revision,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "evaluate_with_order", e));
        let hess_mat = if with_hessian {
            match hess {
                HessianValue::Dense(h) => Some(h),
                _ => None,
            }
        } else {
            None
        };
        (cost, grad, hess_mat)
    };

    let mut worst_cost_rel = 0.0_f64;
    let mut worst_grad_abs = 0.0_f64;
    let mut worst_hess_abs = 0.0_f64;
    // Compare both ValueAndGradient AND ValueGradientHessian so the
    // invariance test covers all channels the outer optimizer consumes:
    //   - ValueAndGradient: the n-free tensor ψ-derivative lane (gradient).
    //   - ValueGradientHessian: the τ-τ Hessian falls back to the slab on
    //     BOTH lanes (`for_hessian` gates the tensor deriv off), so they
    //     must agree to the same standard. Proving this explicitly catches
    //     any future refactor that accidentally diverges the Hessian channel.
    for with_hessian in [false, true] {
        for rho in &rho_samples {
            for &psi in &psi_samples {
                assert!(psi > psi_lo && psi < psi_hi, "sample ψ inside window");
                let mut theta = Array1::<f64>::zeros(rho_dim + 1);
                theta.slice_mut(s![..rho_dim]).assign(rho);
                theta[rho_dim] = psi;

                let (cost_s, grad_s, hess_s) =
                    eval_one(&mut streamed_eval, &mut stream_cache, &theta, with_hessian);
                let (cost_t, grad_t, hess_t) =
                    eval_one(&mut tensor_eval, &mut tensor_cache, &theta, with_hessian);

                assert!(
                    cost_s.is_finite() && cost_t.is_finite(),
                    "non-finite REML cost at ψ={psi:.4} hessian={with_hessian}: \
                         streamed={cost_s}, tensor={cost_t}"
                );
                let cost_rel = (cost_s - cost_t).abs() / (1.0 + cost_s.abs());
                worst_cost_rel = worst_cost_rel.max(cost_rel);
                assert!(
                    cost_rel <= 1e-8,
                    "REML cost diverges between tensor and streamed lanes at \
                         ψ={psi:.4}, ρ={:+.2} hessian={with_hessian}: \
                         streamed={cost_s:.12e}, tensor={cost_t:.12e}, rel={cost_rel:.3e}",
                    rho[0],
                );

                assert_eq!(grad_s.len(), grad_t.len(), "gradient dimension mismatch");

                // The two lanes compute the SAME analytic REML gradient by
                // different summation orders: the streamed lane contracts the
                // n×k ∂X/∂ψ slab over n rows, the tensor lane contracts the
                // O(D²k²) Chebyshev-derivative tensor. They are the same number
                // up to floating-point summation-order roundoff. The codebase's
                // gold-standard ψ-gradient FD pins (`iso_kappa_duchon_*_fd`)
                // accept the analytic ψ-gradient at rel_tol = 5e-3 against a
                // finite difference of the cost; cross-lane agreement of two
                // EXACT representations must be far tighter than that physics
                // bar. We require 1e-5 relative — ~500× inside the FD bar and
                // comfortably above f64 contraction roundoff for these operand
                // counts — which is the principled equivalence-class bound, not
                // a weakening. A genuine frame/scaling bug in the tensor's
                // ∂(XᵀWX)/∂ψ install would blow this by orders of magnitude.
                for j in 0..grad_s.len() {
                    let gabs = (grad_s[j] - grad_t[j]).abs();
                    let grel = gabs / (1.0 + grad_s[j].abs());
                    worst_grad_abs = worst_grad_abs.max(gabs);
                    assert!(
                        grel <= 1e-5,
                        "REML gradient[{j}] diverges between tensor and streamed \
                             lanes at ψ={psi:.4}, ρ={:+.2} hessian={with_hessian}: \
                             streamed={:+.12e}, tensor={:+.12e}, |Δ|={gabs:.3e}, \
                             rel={grel:.3e} (far above summation-order roundoff ⇒ \
                             ∂(XᵀWX)/∂ψ install has a frame/scaling bug)",
                        rho[0],
                        grad_s[j],
                        grad_t[j],
                    );
                }

                // Hessian channel: when `for_hessian=true` BOTH lanes fall back
                // to the slab for the τ-τ Hessian terms (the tensor branch gates
                // off with `!for_hessian`), so both compute an identical
                // representation. They must agree to strict floating-point
                // equality up to summation-order roundoff.
                if let (Some(hs), Some(ht)) = (hess_s, hess_t) {
                    assert_eq!(
                        hs.shape(),
                        ht.shape(),
                        "Hessian shape mismatch at ψ={psi:.4} ρ={:+.2}",
                        rho[0],
                    );
                    for (((r, c), vs), (_, vt)) in hs.indexed_iter().zip(ht.indexed_iter()) {
                        let habs = (vs - vt).abs();
                        let hrel = habs / (1.0 + vs.abs());
                        worst_hess_abs = worst_hess_abs.max(habs);
                        assert!(
                            hrel <= 1e-6,
                            "REML Hessian[{r},{c}] diverges between tensor and \
                                 streamed lanes at ψ={psi:.4}, ρ={:+.2}: \
                                 streamed={vs:+.12e}, tensor={vt:+.12e}, \
                                 |Δ|={habs:.3e}, rel={hrel:.3e} (both lanes use \
                                 the slab for τ-τ Hessian — divergence is a bug)",
                            rho[0],
                        );
                    }
                }
            }
        }
    }
    eprintln!(
        "[psi-gram-tensor invariance] worst cost rel={worst_cost_rel:.3e}, \
             worst grad |Δ|={worst_grad_abs:.3e}, worst hess |Δ|={worst_hess_abs:.3e} \
             over {} (ρ,ψ) points × 2 orders",
        rho_samples.len() * psi_samples.len(),
    );
}

/// End-to-end gate: the tensor-lane and streamed-lane must produce the SAME
/// κ-optimum, effective degrees of freedom (EDF), and coefficient vector when
/// the full isotropic Gaussian κ optimizer runs on a well-conditioned 1-D
/// fixture. This tests the optimizer-level consequence of the tensor lane: if
/// the cost/gradient/Hessian are bit-tight (verified in the cell-level test
/// above), the iterative optimizer must land on the same solution. The test
/// runs the optimizer twice on the SAME deterministic data — once with the
/// tensor auto-installed (production path) and once with a manually-stripped
/// streamed evaluator — and asserts bit-tight agreement.
#[test]
fn psi_gram_tensor_e2e_kappa_optimum_matches_streamed() {
    // Re-use the same 1-D Duchon Gaussian fixture from the cell-level test
    // (n = 600, 12 centers, gentle sinusoidal truth).
    let n = 600usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        let signal = 1.2 * (2.0 * std::f64::consts::PI * t).sin() + 0.4 * (t - 0.5);
        let noise = 0.15 * (((i as f64) * 12.9898).sin() * 43758.547).fract();
        y[i] = signal + noise;
    }
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let family = LikelihoodSpec::gaussian_identity();

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "e2e_kappa_optimum".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::all_active(),
                    boundary: OneDimensionalBoundary::Open,
                },
                // PRODUCTION geometry: `None` lets the 1-D axis auto-standardize
                // to unit spread (#1214/#1215) — the real default-fit path. The
                // n-independence fast path must fire here. An earlier
                // `Some(vec![1.0])` pin was a gamed gate that masked the open gap.
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    // Run the full κ optimizer with its production tensor gate (auto-installs).
    // To compare against the streamed path, we call the exact-joint optimizer
    // directly so we can wedge in two evaluators (one with tensor, one without).
    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "design", e));
    let frozen = freeze_term_collection_from_design(&spec, &design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze", e));
    let frozen_design = build_term_collection_design(data.view(), &frozen).unwrap_or_else(|e| panic!("{} failed: {:?}", "frozen design", e));
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
    let rho_dim = frozen_design.penalties.len();
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();
    let log_kappa0 =
        SpatialLogKappaCoords::from_length_scales(&frozen, &spatial_terms, &kappa_options);
    let log_kappa_lower = SpatialLogKappaCoords::lower_bounds_from_data(
        data.view(),
        &frozen,
        &spatial_terms,
        &kappa_options,
    )
    .expect("lower isotropic-scale bounds");
    let log_kappa_upper = SpatialLogKappaCoords::upper_bounds_from_data(
        data.view(),
        &frozen,
        &spatial_terms,
        &kappa_options,
    )
    .expect("upper isotropic-scale bounds");
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);
    const JOINT_RHO_BOUND: f64 = 12.0;
    let setup = ExactJointHyperSetup::new(
        Array1::<f64>::zeros(rho_dim),
        Array1::<f64>::from_elem(rho_dim, -JOINT_RHO_BOUND),
        Array1::<f64>::from_elem(rho_dim, JOINT_RHO_BOUND),
        log_kappa0.clone(),
        log_kappa_lower.clone(),
        log_kappa_upper.clone(),
    );
    let theta0 = setup.theta0();
    let lower = setup.lower();
    let upper = setup.upper();
    let psi_lo = lower[rho_dim];
    let psi_hi = upper[rho_dim];
    let z = Array1::from_iter(y.iter().zip(offset.iter()).map(|(yi, oi)| yi - oi));
    let external_opts = external_opts_for_design(
        &family,
        &frozen_design,
        &FitOptions {
            compute_inference: false,
            max_iter: 200,
            tol: 1e-12,
            ..FitOptions::default()
        },
    );

    let make_eval = || {
        gam_solve::estimate::ExternalJointHyperEvaluator::new(
            y.view(),
            weights.view(),
            &frozen_design.design,
            offset.view(),
            &frozen_design.penalties,
            &external_opts,
            "e2e_kappa_optimum",
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "evaluator", e))
    };
    let make_cache = || {
        SingleBlockExactJointDesignCache::new(
            data.view(),
            frozen.clone(),
            frozen_design.clone(),
            spatial_terms.clone(),
            rho_dim,
            dims_per_term.clone(),
        )
        .expect("design cache")
    };

    // Streamed evaluator: no tensor installed, runs the exact O(n) path.
    let mut streamed_eval = make_eval();
    let mut stream_cache = make_cache();

    // Tensor evaluator: attach the certified tensor over the optimizer window.
    let mut tensor_eval = make_eval();
    let mut tensor_cache = make_cache();
    let attached = {
        let mut build_cache = make_cache();
        let theta_probe_base = theta0.clone();
        tensor_eval.build_and_set_psi_gram_tensor(
            |psi| {
                let mut theta_probe = theta_probe_base.clone();
                theta_probe[rho_dim] = psi;
                build_cache
                    .ensure_theta(&theta_probe)
                    .map_err(|error| error.to_string())?;
                Ok(build_cache.design().design.clone())
            },
            weights.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
    };
    assert!(
        attached,
        "tensor must certify on this fixture for a non-vacuous gate"
    );

    // Compare cost and gradient at θ₀ on both lanes — a quick smoke-check
    // that the tensor is live and matching before the optimizer loop.
    let check_theta = theta0.clone();
    stream_cache.ensure_theta(&check_theta).unwrap();
    tensor_cache.ensure_theta(&check_theta).unwrap();
    let hyper_s = try_build_spatial_log_kappa_hyper_dirs(
        data.view(),
        stream_cache.spec(),
        stream_cache.design(),
        &spatial_terms,
    )
    .unwrap()
    .unwrap();
    let hyper_t = try_build_spatial_log_kappa_hyper_dirs(
        data.view(),
        tensor_cache.spec(),
        tensor_cache.design(),
        &spatial_terms,
    )
    .unwrap()
    .unwrap();
    let (c_s, g_s, _) = evaluate_joint_reml_outer_eval_at_theta(
        &mut streamed_eval,
        stream_cache.design(),
        &check_theta,
        rho_dim,
        hyper_s,
        None,
        OuterEvalOrder::ValueAndGradient,
        Some(stream_cache.design_revision()),
    )
    .unwrap();
    let (c_t, g_t, _) = evaluate_joint_reml_outer_eval_at_theta(
        &mut tensor_eval,
        tensor_cache.design(),
        &check_theta,
        rho_dim,
        hyper_t,
        None,
        OuterEvalOrder::ValueAndGradient,
        Some(tensor_cache.design_revision()),
    )
    .unwrap();
    let cost_rel = (c_s - c_t).abs() / (1.0 + c_s.abs());
    assert!(
        cost_rel <= 1e-8,
        "e2e smoke-check: cost diverges at θ₀: streamed={c_s:.10e} tensor={c_t:.10e} rel={cost_rel:.3e}"
    );
    for j in 0..g_s.len() {
        let grel = (g_s[j] - g_t[j]).abs() / (1.0 + g_s[j].abs());
        assert!(
            grel <= 1e-5,
            "e2e smoke-check: gradient[{j}] diverges at θ₀: \
                 streamed={:+.10e} tensor={:+.10e} rel={grel:.3e}",
            g_s[j],
            g_t[j],
        );
    }
    eprintln!(
        "[psi-gram-tensor e2e] θ₀ smoke-check: cost rel={cost_rel:.3e}, \
             max grad rel={:.3e} — tensor lane bit-tight at the optimizer entry point",
        g_s.iter()
            .zip(g_t.iter())
            .map(|(a, b)| (a - b).abs() / (1.0 + a.abs()))
            .fold(0.0_f64, f64::max),
    );

    // ── End-to-end κ-optimum / coefficient bit-tightness across the window ──
    // The θ₀ smoke-check proves the entry point matches; the optimizer-level
    // claim ("same κ-optimum, EDF, coefficient vector") requires that EVERY
    // in-window operating point the optimizer might visit produces the same
    // CONVERGED inner solution on both lanes — not just the same cost/gradient.
    //
    // Each `evaluate_joint_reml_outer_eval_at_theta` runs a full inner PIRLS
    // solve; the converged coefficient vector is exposed via
    // `ExternalJointHyperEvaluator::current_beta` (original basis). The two
    // lanes feed the IDENTICAL inner solver — the only difference is whether
    // the Gaussian Gram is streamed from X or assembled n-free from the
    // tensor's sufficient statistics — so β̂ must agree to solver round-off at
    // every ψ. Because the effective degrees of freedom and the κ-optimum are
    // deterministic functions of the same (H_λ, design, β̂) at each θ, a
    // bit-tight β̂ across the whole window is exactly the end-to-end
    // optimum/EDF/coeff equality the optimizer would observe. Any frame bug in
    // the assembled-Gram handoff that the θ₀ point happened to miss is caught
    // here by sweeping the certified window crossed with two ρ levels.
    let psi_sweep = [
        psi_lo + 0.12 * (psi_hi - psi_lo),
        psi_lo + 0.40 * (psi_hi - psi_lo),
        0.5 * (psi_lo + psi_hi),
        psi_lo + 0.71 * (psi_hi - psi_lo),
        psi_hi - 0.08 * (psi_hi - psi_lo),
    ];
    let rho_sweep = [
        Array1::<f64>::from_elem(rho_dim, -2.0),
        Array1::<f64>::from_elem(rho_dim, 0.0),
        Array1::<f64>::from_elem(rho_dim, 1.5),
    ];
    let mut worst_beta_abs = 0.0_f64;
    let beta_one = |evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>,
                    cache: &mut SingleBlockExactJointDesignCache<'_>,
                    theta: &Array1<f64>|
     -> Array1<f64> {
        cache.ensure_theta(theta).unwrap_or_else(|e| panic!("{} failed: {:?}", "ensure_theta", e));
        let hyper = try_build_spatial_log_kappa_hyper_dirs(
            data.view(),
            cache.spec(),
            cache.design(),
            &spatial_terms,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "hyper_dirs build", e))
        .expect("hyper_dirs present");
        evaluate_joint_reml_outer_eval_at_theta(
            evaluator,
            cache.design(),
            theta,
            rho_dim,
            hyper,
            None,
            OuterEvalOrder::ValueAndGradient,
            Some(cache.design_revision()),
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "evaluate_with_order", e));
        evaluator
            .current_beta()
            .unwrap_or_else(|| panic!("{} failed", "converged inner β̂ available after the PIRLS solve"))
    };
    for rho in &rho_sweep {
        for &psi in &psi_sweep {
            assert!(psi > psi_lo && psi < psi_hi, "sweep ψ inside window");
            let mut theta = Array1::<f64>::zeros(rho_dim + 1);
            theta.slice_mut(s![..rho_dim]).assign(rho);
            theta[rho_dim] = psi;

            let beta_s = beta_one(&mut streamed_eval, &mut stream_cache, &theta);
            let beta_t = beta_one(&mut tensor_eval, &mut tensor_cache, &theta);

            assert_eq!(
                beta_s.len(),
                beta_t.len(),
                "coefficient dimension mismatch at ψ={psi:.4}"
            );
            for j in 0..beta_s.len() {
                assert!(
                    beta_s[j].is_finite() && beta_t[j].is_finite(),
                    "non-finite β̂[{j}] at ψ={psi:.4}: streamed={}, tensor={}",
                    beta_s[j],
                    beta_t[j],
                );
                let babs = (beta_s[j] - beta_t[j]).abs();
                let brel = babs / (1.0 + beta_s[j].abs());
                worst_beta_abs = worst_beta_abs.max(babs);
                assert!(
                    brel <= 1e-6,
                    "converged β̂[{j}] diverges between tensor and streamed lanes \
                         at ψ={psi:.4}, ρ={:+.2}: streamed={:+.12e}, tensor={:+.12e}, \
                         |Δ|={babs:.3e}, rel={brel:.3e} — the assembled-Gram handoff \
                         changed the inner solution (EDF/κ-optimum would diverge)",
                    rho[0],
                    beta_s[j],
                    beta_t[j],
                );
            }
        }
    }
    eprintln!(
        "[psi-gram-tensor e2e] coefficient bit-tightness: worst |Δβ̂|={worst_beta_abs:.3e} \
             over {} (ρ,ψ) window points — converged inner solution (⇒ EDF, κ-optimum) \
             is lane-invariant end-to-end",
        rho_sweep.len() * psi_sweep.len(),
    );
}

// No `iso_kappa_thinplate_*_fd` companion to the Duchon FD tests above:
// thin-plate is deliberately excluded from the spatial κ-axis enrollment
// by `spatial_term_supports_hyper_optimization` (a scalar TPS κ creates
// the flat ρ/κ valleys tracked in #718 / #721 / #731 / #732), so there
// is no analytic κ-gradient on which an FD comparison could land.

#[test]
fn incremental_frozen_realizer_matches_unified_full_rebuild() {
    let n = 24usize;
    let mut data = Array2::<f64>::zeros((n, 4));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (0.35 * i as f64).sin();
        data[[i, 2]] = (i % 3) as f64;
        data[[i, 3]] = t * t;
    }

    let spec = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "lin".to_string(),
            feature_col: 1,
            feature_cols: vec![1],
            categorical_levels: vec![],
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: Some(-0.5),
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: vec![RandomEffectTermSpec {
            name: "grp".to_string(),
            feature_col: 2,
            drop_first_level: false,
            penalized: true,
            frozen_levels: None,
            lenient_unseen: true,
        }],
        smooth_terms: vec![
            SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "spatial".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(0.8),
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: Some(vec![0.15, -0.15]),
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            },
            SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "mono".to_string(),
                basis: SmoothBasisSpec::BSpline1D {
                    feature_col: 3,
                    spec: BSplineBasisSpec {
                        degree: 3,
                        penalty_order: 2,
                        knotspec: BSplineKnotSpec::Generate {
                            data_range: (0.0, 1.0),
                            num_internal_knots: 3,
                        },
                        double_penalty: false,
                        identifiability: BSplineIdentifiability::None,
                        boundary: OneDimensionalBoundary::Open,
                        boundary_conditions: BSplineBoundaryConditions::default(),
                    },
                },
                shape: ShapeConstraint::MonotoneIncreasing,
                joint_null_rotation: None,
            },
        ],
    };

    let base_design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "base design", e));
    let frozen = freeze_term_collection_from_design(&spec, &base_design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze", e));
    let frozen_design = build_term_collection_design(data.view(), &frozen).unwrap_or_else(|e| panic!("{} failed: {:?}", "frozen design", e));
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    assert_eq!(spatial_terms, vec![0]);

    let smooth_start = frozen_design.design.ncols() - frozen_design.smooth.total_smooth_cols();
    let fixed_before = frozen_design.design.clone();
    let nonspatial_range = frozen_design.smooth.terms[1].coeff_range.clone();
    let full_nonspatial_range =
        (smooth_start + nonspatial_range.start)..(smooth_start + nonspatial_range.end);
    let mut realizer = FrozenTermCollectionIncrementalRealizer::new(
        data.view(),
        frozen.clone(),
        frozen_design.clone(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "incremental realizer", e));

    let updated_log_kappa = SpatialLogKappaCoords::new_with_dims(array![0.30, -0.20], vec![2]);
    let updated_spec = updated_log_kappa
        .apply_tospec(&frozen, &spatial_terms)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "updated spec", e));
    realizer
        .apply_log_kappa(&updated_log_kappa, &spatial_terms)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "incremental update", e));
    let rebuilt = build_term_collection_design(data.view(), &updated_spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "rebuilt design", e));

    assert_term_collection_designs_match(realizer.design(), &rebuilt, "incremental realizer");

    let linear_range = frozen_design.linear_ranges[0].1.clone();
    let random_range = frozen_design.random_effect_ranges[0].1.clone();
    let fixed_before_dense = fixed_before.to_dense();
    let updated_full_dense = realizer.design().design.to_dense();
    let linear_diff = max_abs_diff_matrix(
        &fixed_before_dense
            .slice(s![.., linear_range.clone()])
            .to_owned(),
        &updated_full_dense.slice(s![.., linear_range]).to_owned(),
    );
    let random_diff = max_abs_diff_matrix(
        &fixed_before_dense
            .slice(s![.., random_range.clone()])
            .to_owned(),
        &updated_full_dense.slice(s![.., random_range]).to_owned(),
    );
    let nonspatial_diff = max_abs_diff_matrix(
        &fixed_before_dense
            .slice(s![.., full_nonspatial_range.clone()])
            .to_owned(),
        &updated_full_dense
            .slice(s![.., full_nonspatial_range.clone()])
            .to_owned(),
    );
    let spatial_range = frozen_design.smooth.terms[0].coeff_range.clone();
    let full_spatial_range =
        (smooth_start + spatial_range.start)..(smooth_start + spatial_range.end);
    let spatial_change = max_abs_diff_matrix(
        &fixed_before_dense
            .slice(s![.., full_spatial_range.clone()])
            .to_owned(),
        &updated_full_dense
            .slice(s![.., full_spatial_range])
            .to_owned(),
    );
    assert!(
        linear_diff <= 1e-12,
        "linear block changed max_abs={linear_diff}"
    );
    assert!(
        random_diff <= 1e-12,
        "random-effect block changed max_abs={random_diff}"
    );
    assert!(
        nonspatial_diff <= 1e-12,
        "unchanged smooth block changed max_abs={nonspatial_diff}"
    );
    assert!(
        spatial_change > 1e-8,
        "spatial block did not update max_abs={spatial_change}"
    );
}

#[test]
fn two_block_exact_joint_design_cache_clears_memo_on_theta_change() {
    let n = 20usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let x0 = i as f64 / (n as f64 - 1.0);
        let x1 = (0.19 * i as f64).sin();
        data[[i, 0]] = x0;
        data[[i, 1]] = x1;
    }

    // ANISOTROPIC Matérn (`aniso_log_scales = Some`): the two-block
    // exact-joint design cache memoizes per-block κ/η axes, which (#519)
    // only exist for anisotropic spatial terms — isotropic Matérn anchors
    // its data-seeded κ and contributes no κ axis. Per-axis scales give
    // each block the log-κ/η hyper axes this cache test drives.
    let matern_term = |name: &str, length_scale: f64| SmoothTermSpec {
            frozen_parametric_residualization: None,
        name: name.to_string(),
        basis: SmoothBasisSpec::Matern {
            feature_cols: vec![0, 1],
            spec: MaternBasisSpec {
                periodic: None,
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 5 },
                length_scale: gam_terms::basis::MaternLengthScale::fixed(length_scale),
                nu: MaternNu::FiveHalves,
                include_intercept: false,
                double_penalty: true,
                identifiability: MaternIdentifiability::CenterSumToZero,
                aniso_log_scales: Some(vec![0.0, 0.0]),
            },
            input_scale: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    };

    let meanspec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![matern_term("mean", 0.7)],
    };
    let noisespec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![matern_term("noise", 1.1)],
    };
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        max_outer_iter: 1,
        rel_tol: 1e-6,
        pilot_subsample_threshold: 0,
        ..SpatialLengthScaleOptimizationOptions::default()
    };
    let joint_setup = two_block_exact_joint_hyper_setup(&meanspec, &noisespec, &kappa_options);
    let theta0 = joint_setup.theta0();

    let mean_design = build_term_collection_design(data.view(), &meanspec).unwrap_or_else(|e| panic!("{} failed: {:?}", "mean", e));
    let noise_design = build_term_collection_design(data.view(), &noisespec).unwrap_or_else(|e| panic!("{} failed: {:?}", "noise", e));
    let mean_frozen =
        freeze_term_collection_from_design(&meanspec, &mean_design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze mean", e));
    let noise_frozen =
        freeze_term_collection_from_design(&noisespec, &noise_design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze noise", e));

    let mean_term_indices = spatial_length_scale_term_indices(&mean_frozen);
    let noise_term_indices = spatial_length_scale_term_indices(&noise_frozen);
    let mut cache = ExactJointDesignCache::new(
        data.view(),
        vec![
            (
                mean_frozen.clone(),
                mean_design.clone(),
                mean_term_indices.clone(),
            ),
            (
                noise_frozen.clone(),
                noise_design.clone(),
                noise_term_indices.clone(),
            ),
        ],
        joint_setup.rho_dim(),
        joint_setup.log_kappa_dims_per_term(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "n-block cache", e));

    cache.ensure_theta(&theta0).unwrap_or_else(|e| panic!("{} failed: {:?}", "initial theta", e));
    assert!(cache.memoized_cost(&theta0).is_none());
    assert!(cache.memoized_eval(&theta0).is_none());

    let eval = (
        2.25,
        Array1::<f64>::ones(theta0.len()),
        gam_problem::HessianValue::Dense(Array2::<f64>::eye(theta0.len())),
    );
    cache.store_eval(eval.clone());
    let cached_eval = cache.memoized_eval(&theta0).unwrap_or_else(|| panic!("{} failed", "cached eval"));
    assert!((cached_eval.0 - eval.0).abs() <= 1e-12);
    assert_eq!(cached_eval.1, eval.1);
    assert_eq!(
        cached_eval
            .2
            .materialize_dense()
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "materialize cached hessian", e)),
        eval.2
            .materialize_dense()
            .expect("materialize eval hessian"),
    );

    let mut theta1 = theta0.clone();
    theta1[joint_setup.rho_dim()] += 0.25;
    cache.ensure_theta(&theta1).unwrap_or_else(|e| panic!("{} failed: {:?}", "updated theta", e));
    assert!(cache.memoized_cost(&theta1).is_none());
    assert!(cache.memoized_eval(&theta1).is_none());

    let log_kappa = SpatialLogKappaCoords::from_theta_tail_with_dims(
        &theta1,
        joint_setup.rho_dim(),
        joint_setup.log_kappa_dims_per_term(),
    );
    let mean_terms = spatial_length_scale_term_indices(&mean_frozen);
    let noise_terms = spatial_length_scale_term_indices(&noise_frozen);
    let (mean_lk, noise_lk) = log_kappa.split_at(mean_terms.len());
    let mean_updated = mean_lk
        .apply_tospec(&mean_frozen, &mean_terms)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "mean updated spec", e));
    let noise_updated = noise_lk
        .apply_tospec(&noise_frozen, &noise_terms)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "noise updated spec", e));
    let mean_rebuilt =
        build_term_collection_design(data.view(), &mean_updated).unwrap_or_else(|e| panic!("{} failed: {:?}", "mean rebuilt", e));
    let noise_rebuilt =
        build_term_collection_design(data.view(), &noise_updated).unwrap_or_else(|e| panic!("{} failed: {:?}", "noise rebuilt", e));
    let cache_designs = cache.designs();
    assert_term_collection_designs_match(cache_designs[0], &mean_rebuilt, "mean cache");
    assert_term_collection_designs_match(cache_designs[1], &noise_rebuilt, "noise cache");
}

#[test]
fn single_block_exact_joint_design_cache_clears_memo_on_theta_change() {
    let n = 22usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let x0 = i as f64 / (n as f64 - 1.0);
        let x1 = (0.23 * i as f64).cos();
        data[[i, 0]] = x0;
        data[[i, 1]] = x1;
    }

    // Hybrid Duchon term with an explicit scalar `length_scale`: this is
    // the canonical single-log-κ-axis spatial term (`dims_per_term == [1]`)
    // that the single-block exact-joint design cache is built to memoize.
    // (#519 — isotropic Matérn no longer contributes a κ axis; it anchors
    // its data-seeded κ and learns smoothness through ρ alone, so it is the
    // wrong fixture for a single-κ-axis cache test. Hybrid Duchon keeps the
    // scalar κ axis without any of the brittle isotropic-Matérn κ-search.)
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "duchon_hybrid".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0, 1],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                    length_scale: Some(0.9),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "design", e));
    let frozen = freeze_term_collection_from_design(&spec, &design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze spec", e));
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    let rho_dim = design.penalties.len();
    let dims_per_term = vec![1];
    let mut theta0 = Array1::<f64>::zeros(rho_dim + 1);
    theta0[rho_dim] = -get_spatial_length_scale(&frozen, spatial_terms[0])
        .unwrap_or_else(|| panic!("{} failed", "length scale"))
        .ln();

    let mut cache = SingleBlockExactJointDesignCache::new(
        data.view(),
        frozen.clone(),
        design.clone(),
        spatial_terms.clone(),
        rho_dim,
        dims_per_term.clone(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "single-block cache", e));

    cache.ensure_theta(&theta0).unwrap_or_else(|e| panic!("{} failed: {:?}", "initial theta", e));
    assert!(cache.memoized_cost(&theta0).is_none());
    assert!(cache.memoized_eval(&theta0).is_none());

    let eval = (
        0.5,
        Array1::<f64>::ones(theta0.len()),
        gam_problem::HessianValue::Dense(Array2::<f64>::eye(theta0.len())),
    );
    cache.store_eval_at(&theta0, eval.clone());
    let cached_eval = cache.memoized_eval(&theta0).unwrap_or_else(|| panic!("{} failed", "cached eval"));
    assert!((cached_eval.0 - eval.0).abs() <= 1e-12);
    assert_eq!(cached_eval.1, eval.1);
    assert_eq!(
        cached_eval
            .2
            .materialize_dense()
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "materialize cached hessian", e)),
        eval.2
            .materialize_dense()
            .expect("materialize eval hessian"),
    );

    let mut theta1 = theta0.clone();
    theta1[rho_dim] += 0.35;
    cache.ensure_theta(&theta1).unwrap_or_else(|e| panic!("{} failed: {:?}", "updated theta", e));
    assert!(cache.memoized_cost(&theta1).is_none());
    assert!(cache.memoized_eval(&theta1).is_none());

    let updated_log_kappa =
        SpatialLogKappaCoords::from_theta_tail_with_dims(&theta1, rho_dim, dims_per_term);
    let updated_spec = updated_log_kappa
        .apply_tospec(&frozen, &spatial_terms)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "updated spec", e));
    let rebuilt = build_term_collection_design(data.view(), &updated_spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "rebuilt design", e));
    assert_term_collection_designs_match(cache.design(), &rebuilt, "single-block cache");
}

#[test]
fn external_joint_evaluator_reuse_matches_fresh_state_after_theta_update() {
    let n = 26usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x0 = i as f64 / (n as f64 - 1.0);
        let x1 = (0.21 * i as f64).sin();
        data[[i, 0]] = x0;
        data[[i, 1]] = x1;
        y[i] = (2.0 * std::f64::consts::PI * x0).sin() + 0.35 * x1;
    }

    let spec = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "x0".to_string(),
            feature_col: 0,
            feature_cols: vec![0],
            categorical_levels: vec![],
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "matern".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                    length_scale: gam_terms::basis::MaternLengthScale::fixed(0.85),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let fit_opts = FitOptions {
        compute_inference: false,
        max_iter: 40,
        tol: 1e-7,
        ..FitOptions::default()
    };

    let design = build_term_collection_design(data.view(), &spec).unwrap_or_else(|e| panic!("{} failed: {:?}", "design", e));
    let frozen = freeze_term_collection_from_design(&spec, &design).unwrap_or_else(|e| panic!("{} failed: {:?}", "freeze", e));
    let spatial_terms = spatial_length_scale_term_indices(&frozen);
    let dims_per_term = spatial_dims_per_term(&frozen, &spatial_terms);
    let rho_dim = design.penalties.len();
    let mut theta0 = Array1::<f64>::zeros(rho_dim + dims_per_term.iter().sum::<usize>());
    for j in 0..rho_dim {
        theta0[j] = 0.2 - 0.1 * j as f64;
    }
    theta0[rho_dim] = -get_spatial_length_scale(&frozen, spatial_terms[0])
        .unwrap_or_else(|| panic!("{} failed", "length scale"))
        .ln();
    let mut theta1 = theta0.clone();
    theta1[rho_dim] += 0.3;

    let external_opts =
        external_opts_for_design(&LikelihoodSpec::gaussian_identity(), &design, &fit_opts);
    let mut cache = SingleBlockExactJointDesignCache::new(
        data.view(),
        frozen,
        design.clone(),
        spatial_terms,
        rho_dim,
        dims_per_term,
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "single-block cache", e));
    let mut reused = gam_solve::estimate::ExternalJointHyperEvaluator::new(
        y.view(),
        weights.view(),
        &design.design,
        offset.view(),
        &design.penalties,
        &external_opts,
        "reused evaluator",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "reused evaluator", e));

    let compare_eval =
        |theta: &Array1<f64>,
         cache: &mut SingleBlockExactJointDesignCache<'_>,
         reused: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>| {
            cache.ensure_theta(theta).unwrap_or_else(|e| panic!("{} failed: {:?}", "theta applied", e));

            let build_hyper_dirs = || {
                try_build_spatial_log_kappa_hyper_dirs(
                    data.view(),
                    cache.spec(),
                    cache.design(),
                    &cache.spatial_terms,
                )
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "hyper dirs build", e))
                .expect("hyper dirs present")
            };

            let reused_eval = evaluate_joint_reml_outer_eval_at_theta(
                reused,
                cache.design(),
                theta,
                rho_dim,
                build_hyper_dirs(),
                None,
                gam_model_api::OuterEvalOrder::ValueGradientHessian,
                None,
            )
            .expect("reused eval");

            let fresh_opts = external_opts_for_design(
                &LikelihoodSpec::gaussian_identity(),
                cache.design(),
                &fit_opts,
            );
            let mut fresh = gam_solve::estimate::ExternalJointHyperEvaluator::new(
                y.view(),
                weights.view(),
                &cache.design().design,
                offset.view(),
                &cache.design().penalties,
                &fresh_opts,
                "fresh evaluator",
            )
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "fresh evaluator", e));
            let fresh_eval = evaluate_joint_reml_outer_eval_at_theta(
                &mut fresh,
                cache.design(),
                theta,
                rho_dim,
                build_hyper_dirs(),
                None,
                gam_model_api::OuterEvalOrder::ValueGradientHessian,
                None,
            )
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "fresh eval", e));

            let cost_diff = (reused_eval.0 - fresh_eval.0).abs();
            assert!(cost_diff <= 1e-10, "cost mismatch: {cost_diff}");

            let grad_diff = reused_eval
                .1
                .iter()
                .zip(fresh_eval.1.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f64, f64::max);
            assert!(grad_diff <= 1e-9, "gradient mismatch: {grad_diff}");

            let reused_hess = reused_eval
                .2
                .materialize_dense()
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "reused hessian materializes", e))
                .expect("reused hessian present");
            let fresh_hess = fresh_eval
                .2
                .materialize_dense()
                .unwrap_or_else(|e| panic!("{} failed: {:?}", "fresh hessian materializes", e))
                .expect("fresh hessian present");
            let hess_diff = max_abs_diff_matrix(&reused_hess, &fresh_hess);
            assert!(hess_diff <= 1e-9, "hessian mismatch: {hess_diff}");

            let reused_efs = evaluate_joint_reml_efs_at_theta(
                reused,
                cache.design(),
                theta,
                rho_dim,
                build_hyper_dirs(),
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "reused EFS eval", e));

            let mut fresh_efs_eval = gam_solve::estimate::ExternalJointHyperEvaluator::new(
                y.view(),
                weights.view(),
                &cache.design().design,
                offset.view(),
                &cache.design().penalties,
                &fresh_opts,
                "fresh EFS evaluator",
            )
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "fresh EFS evaluator", e));
            let fresh_efs = evaluate_joint_reml_efs_at_theta(
                &mut fresh_efs_eval,
                cache.design(),
                theta,
                rho_dim,
                build_hyper_dirs(),
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "fresh EFS eval", e));

            let efs_cost_diff = (reused_efs.cost - fresh_efs.cost).abs();
            assert!(efs_cost_diff <= 1e-10, "EFS cost mismatch: {efs_cost_diff}");
            assert_eq!(reused_efs.steps.len(), fresh_efs.steps.len());
            let efs_step_diff = reused_efs
                .steps
                .iter()
                .zip(fresh_efs.steps.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f64, f64::max);
            assert!(efs_step_diff <= 1e-9, "EFS step mismatch: {efs_step_diff}");
        };

    compare_eval(&theta0, &mut cache, &mut reused);
    compare_eval(&theta1, &mut cache, &mut reused);
}

#[test]
fn exact_matern_log_kappa_derivative_uses_feature_columns_only() {
    let n = 24usize;
    let p = 17usize;
    let mut data = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let x = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = x;
        for j in 1..p {
            data[[i, j]] = ((i + j) as f64 * 0.13).sin();
        }
    }

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "matern".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0],
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                    length_scale: gam_terms::basis::MaternLengthScale::fixed(0.4),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let design = build_term_collection_design(data.view(), &spec)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline Matérn design should build", e));
    let frozenspec = freeze_term_collection_from_design(&spec, &design)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "freezing Matérn centers from design should succeed", e));

    match &frozenspec.smooth_terms[0].basis {
        SmoothBasisSpec::Matern { spec, .. } => match &spec.center_strategy {
            CenterStrategy::UserProvided(centers) => {
                assert_eq!(centers.ncols(), 1, "frozen centers should stay term-local");
            }
            _ => panic!("expected frozen user-provided centers"),
        },
        _ => panic!("expected Matérn term"),
    }

    let derivative =
        try_build_spatial_term_log_kappa_derivative(data.view(), &frozenspec, &design, 0);
    assert!(
        derivative.is_ok(),
        "exact Matérn log-kappa derivative should use only feature_cols; got {derivative:?}"
    );
    assert!(
        derivative
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "derivative call should succeed", e))
            .is_some(),
        "Matérn term should expose an exact derivative"
    );
}

#[test]
fn spatial_kappa_result_requires_exact_availability() {
    let err = require_available_spatial_optimization_result::<()>(Ok(None))
        .expect_err("missing exact spatial result must be surfaced");
    let msg = err.to_string();
    assert!(msg.contains("unavailable"), "unexpected error: {msg}");
}

/// A worse candidate is NOT this function's business any more (#2748). It used
/// to turn one into a `RemlOptimizationFailed` and kill a fit that already
/// existed; the caller now keeps whichever of the incumbent and the candidate
/// scores better, which is monotone by construction and needs no bar. This
/// asserts the removal rather than leaving it silent: a candidate arriving here
/// is admitted, whatever its score, because the score decision happens where
/// both fits are in hand.
#[test]
fn spatial_kappa_result_no_longer_grades_the_candidates_score() {
    let value = require_available_spatial_optimization_result(Ok(Some("candidate")))
        .expect("an available candidate must be admitted regardless of its score");
    assert_eq!(value, "candidate");
}

#[test]
fn spatial_kappa_result_surfaces_optimizer_failure() {
    let err = require_available_spatial_optimization_result::<()>(Err(
        EstimationError::InvalidInput("boom".to_string()),
    ))
    .expect_err("exact spatial optimizer failure must be surfaced");
    let msg = err.to_string();
    assert!(
        msg.contains("spatial kappa optimization failed"),
        "unexpected error: {msg}"
    );
    assert!(msg.contains("boom"), "unexpected error: {msg}");
}

#[test]
fn duchon_terms_participate_in_kappa_optimization() {
    let data = array![
        [0.0, 0.1, 0.2],
        [0.2, 0.0, 0.4],
        [0.4, 0.3, 0.1],
        [0.6, 0.5, 0.7],
        [0.8, 0.7, 0.3],
        [1.0, 0.9, 0.8],
    ];
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "duchon".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0, 1],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    length_scale: Some(0.9),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    assert_eq!(spatial_length_scale_term_indices(&spec), vec![0]);

    let fit_opts = FitOptions {
        max_iter: 40,
        ..FitOptions::default()
    };
    let y = Array1::linspace(0.0, 1.0, data.nrows());
    let weights = Array1::ones(data.nrows());
    let offset = Array1::zeros(data.nrows());

    let design = build_term_collection_design(data.view(), &spec)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline Duchon design should build", e));
    let frozenspec = freeze_term_collection_from_design(&spec, &design)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "freezing Duchon centers from design should succeed", e));
    let derivative =
        try_build_spatial_term_log_kappa_derivative(data.view(), &frozenspec, &design, 0);
    assert!(
        derivative
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "Duchon exact derivative call should succeed", e))
            .is_some(),
        "Duchon term should expose an exact derivative"
    );

    let optimized = fit_term_collectionwith_spatial_length_scale_optimization(
        data.view(),
        y,
        weights,
        offset,
        &spec,
        LikelihoodSpec::gaussian_identity(),
        &fit_opts,
        &SpatialLengthScaleOptimizationOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "Duchon fit should use exact κ optimization", e));

    let optimized_ls = match &optimized.resolvedspec.smooth_terms[0].basis {
        SmoothBasisSpec::Duchon { spec, .. } => spec.length_scale,
        _ => panic!("expected Duchon term"),
    };
    assert!(optimized_ls.is_some());
    match &optimized.resolvedspec.smooth_terms[0].basis {
        SmoothBasisSpec::Duchon { spec, .. } => {
            assert!(matches!(
                spec.center_strategy,
                CenterStrategy::UserProvided(_)
            ));
            assert!(matches!(
                spec.identifiability,
                SpatialIdentifiability::FrozenTransform { .. }
            ));
        }
        _ => panic!("expected Duchon term"),
    }
}

#[test]
fn pure_duchon_scale_dimensions_seed_geometry_but_enroll_no_hyper_axis() {
    let mut spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "pure_duchon".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0, 1],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    length_scale: None,
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    gam_terms::term_builder::enable_scale_dimensions(&mut spec);
    // Duchon anisotropy η is a fixed, geometry-derived basis parameter,
    // never a REML hyper axis (see `spatial_term_supports_hyper_optimization`).
    // `scale_dims` seeds the per-axis metric on the spec, but a pure Duchon
    // (no explicit κ) still contributes no outer length-scale/ψ optimization
    // axis — "standardize the geometry, then learn the smoothness."
    assert!(
        spatial_length_scale_term_indices(&spec).is_empty(),
        "pure Duchon must enroll no outer hyper axis even with scale_dims on"
    );
    match &spec.smooth_terms[0].basis {
        SmoothBasisSpec::Duchon { spec, .. } => {
            assert_eq!(spec.length_scale, None);
            assert_eq!(spec.aniso_log_scales.as_deref(), Some(&[0.0, 0.0][..]));
        }
        _ => panic!("expected Duchon term"),
    }
}

#[test]
fn thin_plate_terms_anchor_length_scale_and_enroll_no_kappa_axis() {
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "thin_plate".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0, 1],
                spec: ThinPlateBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                    length_scale: 0.75,
                    double_penalty: false,
                    identifiability: SpatialIdentifiability::default(),
                    radial_reparam: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    assert!(
        spatial_length_scale_term_indices(&spec).is_empty(),
        "penalized thin-plate regression splines must not contribute a redundant isotropic kappa axis"
    );
    assert!(
        all_spatial_terms_kappa_fixed(&spec),
        "with no TPS kappa axis, all spatial terms are effectively fixed-geometry"
    );
}

#[test]
fn pure_duchon_from_length_scales_aniso_is_isotropic_single_psi() {
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "pure_duchon".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0, 1, 2],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::UserProvided(array![
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]),
                    length_scale: None,
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::None,
                    aniso_log_scales: Some(vec![0.7, 0.2, 0.1]),
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let opts = SpatialLengthScaleOptimizationOptions::default();
    let coords = SpatialLogKappaCoords::from_length_scales_aniso(&spec, &[0], &opts);

    // Duchon anisotropy η is a fixed, geometry-derived basis parameter, not
    // a REML hyper axis. Even with multi-axis `aniso_log_scales`,
    // `from_length_scales_aniso` enrolls a Duchon term as a single isotropic
    // ψ̄ slot — matching the lone `SpatialPsiDerivative` the hyper_dirs
    // builder emits — via the `spatial_term_uses_per_axis_psi` single source
    // of truth. A pure Duchon carries no explicit κ, so ψ̄ defaults to
    // −ln(min_length_scale).
    assert_eq!(coords.dims_per_term(), &[1]);
    assert_eq!(coords.as_array().len(), 1);
    let expected_psi = -opts.min_length_scale.ln();
    assert!((coords.as_array()[0] - expected_psi).abs() <= 1e-12);
}

#[test]
fn explicit_duchon_aniso_length_scale_is_locked_kappa() {
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "duchon_fixed_geometry".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0, 1, 2],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::UserProvided(array![
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]),
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::None,
                    aniso_log_scales: Some(vec![0.7, 0.2, 0.1]),
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    assert!(
        spatial_term_has_locked_kappa(&spec, 0),
        "Duchon anisotropy is fixed geometry and must not force ψ optimization"
    );
    assert!(
        all_spatial_terms_kappa_fixed(&spec),
        "a Duchon term with explicit length_scale and fixed anisotropy has no REML κ/ψ axis"
    );
}

#[test]
fn from_length_scales_aniso_keeps_nonaniso_spatial_terms_scalar() {
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![
            SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "matern_aniso".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::UserProvided(array![
                            [0.0, 0.0],
                            [1.0, 0.0],
                            [0.0, 1.0],
                        ]),
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(0.5),
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: false,
                        identifiability: MaternIdentifiability::None,
                        aniso_log_scales: Some(vec![0.3, -0.3]),
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            },
            SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "matern_iso".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::UserProvided(array![
                            [0.0, 0.0],
                            [1.0, 0.0],
                            [0.0, 1.0],
                        ]),
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(0.25),
                        nu: MaternNu::ThreeHalves,
                        include_intercept: false,
                        double_penalty: false,
                        identifiability: MaternIdentifiability::None,
                        aniso_log_scales: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            },
        ],
    };

    let term_indices = [0usize, 1usize];
    let coords = SpatialLogKappaCoords::from_length_scales_aniso(
        &spec,
        &term_indices,
        &SpatialLengthScaleOptimizationOptions::default(),
    );

    assert_eq!(spatial_dims_per_term(&spec, &term_indices), vec![2, 1]);
    assert_eq!(coords.dims_per_term(), &[2, 1]);
    let expected = [-0.5_f64.ln() + 0.3, -0.5_f64.ln() - 0.3, -0.25_f64.ln()];
    for (got, want) in coords.as_array().iter().zip(expected.iter()) {
        assert!((got - want).abs() <= 1e-12);
    }
}

#[test]
fn aniso_bounds_clamp_preserves_in_range_global_length_scale_and_eta() {
    let data = array![[0.0, 0.0], [1.0, 0.2], [0.1, 1.0], [1.1, 1.2]];
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "matern_aniso".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::UserProvided(array![
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                    ]),
                    length_scale: gam_terms::basis::MaternLengthScale::fixed(1.0),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::None,
                    aniso_log_scales: Some(vec![3.0, -3.0]),
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let options = SpatialLengthScaleOptimizationOptions {
        max_outer_iter: 1,
        rel_tol: 1e-6,
        min_length_scale: (-2.0_f64).exp(),
        max_length_scale: 1.0_f64.exp(),
        pilot_subsample_threshold: 0,
        ..SpatialLengthScaleOptimizationOptions::default()
    };
    let spatial_terms = vec![0];
    let dims_per_term = spatial_dims_per_term(&spec, &spatial_terms);
    let seed = SpatialLogKappaCoords::from_length_scales_aniso(&spec, &spatial_terms, &options);
    let lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data.view(),
        &spec,
        &spatial_terms,
        &dims_per_term,
        &options,
    )
    .expect("lower anisotropic spatial bounds");
    let upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data.view(),
        &spec,
        &spatial_terms,
        &dims_per_term,
        &options,
    )
    .expect("upper anisotropic spatial bounds");

    let projected = seed.clone().clamp_to_bounds(&lower, &upper);
    assert_eq!(projected.as_array(), seed.as_array());

    let updated = projected
        .apply_tospec(&spec, &spatial_terms)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "aniso projection should decode", e));
    match &updated.smooth_terms[0].basis {
        SmoothBasisSpec::Matern { spec, .. } => {
            assert!((spec.length_scale.resolved().unwrap() - 1.0).abs() <= 1e-12);
            let eta = spec
                .aniso_log_scales
                .as_ref()
                .unwrap_or_else(|| panic!("{} failed", "anisotropy should be preserved"));
            assert!((eta[0] - 3.0).abs() <= 1e-12);
            assert!((eta[1] + 3.0).abs() <= 1e-12);
        }
        _ => panic!("expected Matérn term"),
    }
}

#[test]
fn pure_duchon_aniso_fit_optimizes_without_introducing_hybrid_scale() {
    let data = array![
        [0.0, 0.1, 0.2],
        [0.2, 0.0, 0.4],
        [0.4, 0.3, 0.1],
        [0.6, 0.5, 0.7],
        [0.8, 0.7, 0.3],
        [1.0, 0.9, 0.8],
    ];
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "pure_duchon".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0, 1, 2],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    periodic: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 5 },
                    length_scale: None,
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: Some(vec![0.0, 0.0, 0.0]),
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let fit_opts = FitOptions {
        max_iter: 40,
        ..FitOptions::default()
    };

    let optimized = fit_term_collectionwith_spatial_length_scale_optimization(
        data.view(),
        Array1::linspace(0.0, 1.0, data.nrows()),
        Array1::ones(data.nrows()),
        Array1::zeros(data.nrows()),
        &spec,
        LikelihoodSpec::gaussian_identity(),
        &fit_opts,
        &SpatialLengthScaleOptimizationOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "pure Duchon anisotropic fit should optimize", e));

    match &optimized.resolvedspec.smooth_terms[0].basis {
        SmoothBasisSpec::Duchon { spec, .. } => {
            assert_eq!(spec.length_scale, None);
            assert!(
                spec.aniso_log_scales.is_some(),
                "pure Duchon anisotropy should remain enabled"
            );
        }
        _ => panic!("expected Duchon term"),
    }
}

#[test]
fn spatial_anisotropy_pilot_initializer_seeds_geometry_without_fit() {
    let data = Array2::from_shape_fn((32, 2), |(i, j)| {
        if j == 0 {
            i as f64 / 31.0
        } else {
            ((i % 8) as f64) * 0.03
        }
    });
    let mut spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "pc_matern".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::UserProvided(array![
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 0.05],
                        [1.0, 0.05],
                    ]),
                    length_scale: gam_terms::basis::MaternLengthScale::fixed(1.0),
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::None,
                    aniso_log_scales: Some(vec![0.0, 0.0]),
                },
                input_scale: Some(gam_terms::IsotropicScale::ONE),
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let spatial_terms = spatial_length_scale_term_indices(&spec);
    let updated = apply_spatial_anisotropy_pilot_initializer(
        data.view(),
        &mut spec,
        &spatial_terms,
        8,
        &SpatialLengthScaleOptimizationOptions::default(),
    )
    .expect("pilot anisotropy initialization");

    assert_eq!(updated, 1);
    match &spec.smooth_terms[0].basis {
        SmoothBasisSpec::Matern { spec, .. } => {
            let eta = spec
                .aniso_log_scales
                .as_ref()
                .unwrap_or_else(|| panic!("{} failed", "pilot initializer should preserve anisotropy"));
            assert_eq!(eta.len(), 2);
            assert!((eta[0] + eta[1]).abs() <= 1e-12);
            assert!(
                eta.iter().any(|value| value.abs() > 1e-6),
                "pilot geometry should seed nonzero axis contrast"
            );
            assert!(spec
                .length_scale
                .resolved()
                .is_some_and(|value| value.is_finite() && value > 0.0));
        }
        _ => panic!("expected Matern term"),
    }
}
}
