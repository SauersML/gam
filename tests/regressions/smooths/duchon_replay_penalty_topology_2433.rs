//! gam#2433: the collection design and the frozen single-term replay must
//! agree, exactly and at every κ, on how many penalty blocks a Duchon smooth
//! owns.
//!
//! The spatial length-scale (κ) optimizer caches ONE authoritative
//! `TermCollectionDesign` and then re-realizes the single spatial term on every
//! trial κ through `build_single_local_smooth_term`, driven by the spec frozen
//! out of that design. `replace_term_realization` hard-refuses a realization
//! whose penalty count differs from the cached range — correctly, because the
//! optimizer would otherwise differentiate a penalty set the design does not
//! have. So a disagreement between the two builders is not a cosmetic
//! mismatch; it makes an isotropic-κ Duchon fit unfittable.
//!
//! What made them disagree was not the gauge algebra: both builders produced
//! the *same* `Primary` matrix, with the same normalization scale and the same
//! spectrum, and `analyze_penalty_block` reported `nullity = 1` for both. The
//! split came from `constructive_nullspace_basis`, which decided whether the
//! Marra & Wood trend ridge had a null space to shrink using a
//! machine-precision RRQR cutoff (`≈ 100·ε·n·|R₀₀|`) rather than the canonical
//! penalty-spectrum convention (`spectral_tolerance = p·1e-10·λ_max`) that
//! every other consumer uses. Whether a `6.9e-13·λ_max` mode survived in the
//! energy factor then depended purely on whether a `try_from_dense_psd` — which
//! applies the loose cutoff and drops sub-tolerance modes — was interposed
//! before or after the identifiability restriction. The term collection
//! restricts an already-factored RAW penalty; the frozen replay factors the
//! dense CONSTRAINED penalty. Same object, opposite answers.
//!
//! These gates pin the fixed contract from three independent angles: the two
//! builders agree, they agree across the whole κ window the optimizer searches,
//! and the surviving ridge is a genuine double-penalty coordinate (it shrinks
//! only the constrained curvature seminorm's null space).

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec, OperatorPenaltySpec,
    OneDimensionalBoundary, PenaltySource, SpatialIdentifiability,
};
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
    build_term_collection_design, freeze_term_collection_from_design,
};
use ndarray::{Array2, ArrayView2};

/// The `perf_scale::misc::kappa_loop_n_scaling` fixture: a 1-D hybrid Duchon
/// (`length_scale = Some`) with every operator penalty active, on the default
/// `OrthogonalToParametric` identifiability. That combination is what puts a
/// `DoublePenaltyNullspace` trend ridge and a non-trivial global gauge in the
/// same term, which is the configuration the two builders disagreed on.
fn spec_1d(length_scale: f64) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "duchon_1d".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    periodic: None,
                    length_scale: Some(length_scale),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec {
                        mass: OperatorPenaltySpec::Active {
                            initial_log_lambda: 0.0,
                            prior: None,
                        },
                        tension: OperatorPenaltySpec::Active {
                            initial_log_lambda: 0.0,
                            prior: None,
                        },
                        stiffness: OperatorPenaltySpec::Active {
                            initial_log_lambda: 0.0,
                            prior: None,
                        },
                    },
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

fn covariate(n: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        x[[i, 0]] = (i as f64) / (n as f64 - 1.0) * 6.0 - 3.0;
    }
    x
}

fn sources(penalties: &[gam::basis::ActivePenalty]) -> Vec<PenaltySource> {
    penalties.iter().map(|p| p.info.source.clone()).collect()
}

/// Replay one smooth term exactly the way `FrozenTermCollectionIncrementalRealizer`
/// does on a κ proposal: from the spec frozen out of the authoritative design,
/// through the single-term local builder, with the length scale moved.
fn frozen_replay(
    data: ArrayView2<'_, f64>,
    frozen: &TermCollectionSpec,
    length_scale: Option<f64>,
) -> gam_terms::smooth::LocalSmoothTermBuild {
    let mut term = frozen.smooth_terms[0].clone();
    if let (Some(ls), SmoothBasisSpec::Duchon { spec, .. }) = (length_scale, &mut term.basis) {
        spec.length_scale = Some(ls);
    }
    let mut workspace = gam_terms::basis::BasisWorkspace::default();
    gam_terms::smooth::build_single_local_smooth_term(data, &term, &mut workspace)
        .expect("frozen single-term Duchon replay must build")
}

#[test]
fn duchon_frozen_replay_reproduces_the_collection_penalty_topology_2433() {
    let data = covariate(1000);
    let spec = spec_1d(1.0);
    let design = build_term_collection_design(data.view(), &spec).expect("collection design");
    let term = &design.smooth.terms[0];
    let frozen = freeze_term_collection_from_design(&spec, &design).expect("freeze");
    let replay = frozen_replay(data.view(), &frozen, None);

    assert_eq!(
        sources(&replay.active_penalties),
        sources(&term.active_penalties),
        "the frozen single-term replay must reproduce the collection's penalty \
         topology exactly; the κ optimizer refuses the fit otherwise (#2433)"
    );
    assert_eq!(
        replay.dim,
        term.coeff_range.len(),
        "the frozen replay must reproduce the collection's coefficient width"
    );

    // The topology the two now agree on is the CORRECT one, not merely a shared
    // one. The constrained curvature seminorm's smallest eigenvalue sits ~1600×
    // below `spectral_tolerance`, so it is unpenalized under the canonical
    // convention and the Marra & Wood trend ridge belongs in the emitted set.
    // Dropping it silently disabled the Duchon double penalty.
    assert!(
        term.active_penalties
            .iter()
            .any(|p| matches!(p.info.source, PenaltySource::DoublePenaltyNullspace)),
        "the collection must keep the trend ridge: the constrained Primary reports \
         nullity = 1, so there IS a null space for the double penalty to shrink \
         (emitted set was {:?})",
        sources(&term.active_penalties)
    );
}

/// The topology guard fires on EVERY κ proposal, not just the first, so a
/// penalty count that is merely correct at the realized length scale is not
/// enough — it has to be invariant across the whole ψ window the optimizer
/// searches. A count that depends on a rank test at the scale of a conditioning
/// term can flip mid-optimization; one derived from the canonical convention
/// cannot, because the affine trend is structurally unpenalized at every κ.
#[test]
fn duchon_replay_penalty_topology_is_invariant_across_the_kappa_window_2433() {
    let data = covariate(1000);
    let spec = spec_1d(1.0);
    let design = build_term_collection_design(data.view(), &spec).expect("collection design");
    let certified = sources(&design.smooth.terms[0].active_penalties);
    let frozen = freeze_term_collection_from_design(&spec, &design).expect("freeze");

    // Spans the fixture's own (0.01, 100.0) length-scale bounds.
    for length_scale in [0.01_f64, 0.1, 0.5, 1.0, 4.0, 25.0, 100.0] {
        let replay = frozen_replay(data.view(), &frozen, Some(length_scale));
        assert_eq!(
            sources(&replay.active_penalties),
            certified,
            "κ replay at length_scale={length_scale} changed the realized penalty \
             topology; the incremental realizer hard-refuses this (#2433)"
        );
    }
}

/// A surviving `DoublePenaltyNullspace` block must be a genuine second REML
/// coordinate: its range is the constrained `Primary`'s null space, so it
/// shrinks only unpenalized directions and never re-penalizes curvature
/// (#1476/#2372). Keeping the ridge is only right if it keeps this property.
#[test]
fn duchon_trend_ridge_annihilates_the_constrained_curvature_seminorm_2433() {
    let data = covariate(1000);
    let spec = spec_1d(1.0);
    let design = build_term_collection_design(data.view(), &spec).expect("collection design");
    let term = &design.smooth.terms[0];

    let primary = term
        .active_penalties
        .iter()
        .find(|p| matches!(p.info.source, PenaltySource::Primary))
        .expect("Duchon emits a Primary curvature block");
    let ridge = term
        .active_penalties
        .iter()
        .find(|p| matches!(p.info.source, PenaltySource::DoublePenaltyNullspace))
        .expect("Duchon emits a trend ridge");

    let frob = |m: &Array2<f64>| m.iter().map(|v| v * v).sum::<f64>().sqrt();
    let product = ridge.matrix.dot(&primary.matrix);
    let rel = frob(&product) / (frob(&ridge.matrix) * frob(&primary.matrix));
    assert!(
        rel < 1e-8,
        "the constrained trend ridge must annihilate the constrained curvature \
         seminorm (‖R·S‖/(‖R‖‖S‖) = {rel:.3e} ≥ 1e-8): it is a separate REML \
         coordinate over null(S_c), not a second roughness penalty"
    );
    assert_eq!(
        ridge.info.effective_rank, 1,
        "the 1-D affine trend leaves exactly one unpenalized direction once the \
         parametric orthogonalization has removed the intercept"
    );
}
