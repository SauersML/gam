//! Structural regression guard for #1718: a `duchon(x, z)` cold build that
//! computes a fresh data-metric radial reparameterization must materialize its
//! `n×k` kernel design ONCE, not twice.
//!
//! #1718 reported a 2-D `duchon(x, z)` fit running 1.5–2.5x slower than its own
//! `thinplate(x, z)` on identical data with essentially identical accuracy. Both
//! are isotropic polyharmonic splines with the same REML loop. Part of the
//! avoidable work: on the branch that computes a FRESH mgcv-TPRS-style radial
//! reparam `V` (`#1355`), the builder materialized the full un-rotated design
//! `[K·Z | P]` (`build_duchon_basis_designwithworkspace` — a full kernel
//! evaluation over every (data-row, center) pair) ONLY to read its realized
//! Gram `G_c = (K·Z)ᵀ(K·Z)`, then threw that design away and materialized the
//! design a SECOND time with `V` folded in. `thinplate(x, z)` already reuses its
//! single materialized `K·Z` block (`fast_ab(&kernel_constrained, &V)`) and
//! never re-evaluates the kernel.
//!
//! The fix reuses the already-materialized un-rotated design: the fit-time
//! kernel block is `(K·Z)·V` (a cheap `n×k` × `k×k` GEMM on the block already in
//! hand) and the polynomial block is reparam-independent, so `[K·Z·V | P]` is
//! assembled by rotating the existing design instead of re-evaluating the
//! kernel. `(K·Z)·V = K·(Z·V)`, so the model space and the numeric design are
//! unchanged — the second test below pins that equivalence bit-for-bit against
//! the un-fused rebuild that the predict/replay path still uses.
//!
//! NOTE ON SCOPE: this branch (`frozen_radial_reparam.is_none() &&
//! !operators_active`) is taken by an operators-OFF Duchon smooth. The DEFAULT
//! `duchon(x, z)` from the formula front-end keeps the mass + tension operator
//! penalties ACTIVE (the Hilbert scale), so it takes the operators-active branch
//! which already builds the design exactly once; its remaining gap versus
//! `thinplate(x, z)` is the collocation quadrature for those extra penalties,
//! which is intended modeling work rather than redundancy. The redundant SECOND
//! full kernel materialization this fix removes is on the operators-OFF path
//! (e.g. `duchon(..., operators=off)` and scale-block Duchon smooths per #1561).

use gam_terms::basis::{
    build_duchon_basis, duchon_design_build_count, BasisMetadata, CenterStrategy, DuchonBasisSpec,
    DuchonNullspaceOrder, DuchonOperatorPenaltySpec, OneDimensionalBoundary, SpatialIdentifiability,
};
use ndarray::Array2;
use std::sync::Mutex;

/// `duchon_design_build_count` is a PROCESS-WIDE counter, so any Duchon build on
/// another test thread would pollute a `before`/`after` delta. Serialize every
/// build in this file behind one mutex so the counted region is never racing a
/// concurrent build. (`cargo test` runs test fns in parallel by default.)
static DESIGN_BUILD_COUNT_GUARD: Mutex<()> = Mutex::new(());

/// A deterministic 2-D scatter, large enough to take the dense cold-build path
/// (not the lazy chunked path) and to select a non-trivial number of centers.
fn spatial_data_2d(n: usize) -> Array2<f64> {
    let mut state = 0x1718_2024_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        data[[i, 0]] = next();
        data[[i, 1]] = next();
    }
    data
}

/// A `duchon(x, z)` spec that exercises the fresh data-metric radial
/// reparameterization branch: native scale-free Gram (`length_scale=None`),
/// operator penalties OFF, no frozen reparam, affine (`Linear`) null space, and
/// the cubic spectral power `s = (d-1)/2 = 0.5` (`duchon_cubic_default` for
/// `d=2`). This is the configuration that computes a FRESH reparam `V` on the
/// dense cold path — the branch that used to build the design twice.
fn reparam_branch_duchon_spec_2d() -> DuchonBasisSpec {
    DuchonBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 40 },
        periodic: None,
        length_scale: None,
        power: 0.5,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::all_disabled(),
        boundary: OneDimensionalBoundary::Open,
        radial_reparam: None,
    }
}

#[test]
fn duchon_fresh_reparam_cold_build_materializes_design_exactly_once() {
    // Fresh data each run so the process-wide `(data, spec)` basis cache misses
    // and the cold build actually executes (a cache hit would perform zero
    // design builds and vacuously pass).
    let data = spatial_data_2d(600);
    let spec = reparam_branch_duchon_spec_2d();

    // Hold the guard across the measured build so no concurrent test build lands
    // between the two counter reads. `.unwrap_or_else(|e| e.into_inner())`
    // tolerates a poisoned lock from an unrelated test panic.
    let _guard = DESIGN_BUILD_COUNT_GUARD
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let before = duchon_design_build_count();
    let result = build_duchon_basis(data.view(), &spec).expect("duchon(x, z) cold build");
    let builds = duchon_design_build_count() - before;

    // Precondition: the fresh-reparam branch must actually be the one exercised
    // (a real radial reparam frozen into the basis metadata), otherwise this
    // test would not be guarding the doubled-build path at all.
    match &result.metadata {
        BasisMetadata::Duchon { radial_reparam, .. } => assert!(
            radial_reparam.is_some(),
            "#1718 guard precondition: expected the dense duchon(x, z) build to compute a \
             data-metric radial reparam (the branch that used to build the design twice); \
             got None"
        ),
        other => panic!("expected Duchon basis metadata, got {other:?}"),
    }

    // Core structural assertion: the full `n×k` kernel design is materialized
    // exactly ONCE. Before the #1718 fix this was 2 (once to read `G_c`, once to
    // fold in `V`).
    assert_eq!(
        builds, 1,
        "#1718 regression: duchon(x, z) cold build materialized the kernel design {builds} \
         times (expected exactly 1). The reparam rotation must reuse the already-built \
         un-rotated design, not re-evaluate the kernel."
    );
}

#[test]
fn duchon_fused_reparam_design_equals_unfused_rebuild() {
    // Numerical-correctness guard for the fusion: `(K·Z)·V` (the reused,
    // rotated block) must equal `K·(Z·V)` (the un-fused rebuild the predict /
    // replay path still performs when handed a FROZEN reparam). We build the
    // cold (fused) design, then re-drive the SAME spec with its reparam frozen
    // in — which forces the un-fused `build_duchon_basis_designwithworkspace`
    // rebuild path — and compare the two dense designs element-wise.
    let data = spatial_data_2d(400);
    let spec = reparam_branch_duchon_spec_2d();

    // Share the counter guard so this test's builds never overlap the counted
    // region of the sibling count test.
    let _guard = DESIGN_BUILD_COUNT_GUARD
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let fused = build_duchon_basis(data.view(), &spec).expect("cold fused duchon build");
    let fused_design = fused.design.to_dense();

    // Extract the reparam frozen at the cold build and replay it: with a frozen
    // reparam present the builder skips the fresh-reparam branch and takes the
    // un-fused rebuild (`K·(Z·V)`).
    let frozen_reparam = match &fused.metadata {
        BasisMetadata::Duchon { radial_reparam, .. } => radial_reparam
            .clone()
            .expect("cold build should have produced a data-metric reparam"),
        other => panic!("expected Duchon basis metadata, got {other:?}"),
    };
    let mut replay_spec = reparam_branch_duchon_spec_2d();
    replay_spec.radial_reparam = Some(frozen_reparam);
    let unfused = build_duchon_basis(data.view(), &replay_spec).expect("frozen unfused duchon build");
    let unfused_design = unfused.design.to_dense();

    assert_eq!(
        fused_design.dim(),
        unfused_design.dim(),
        "fused and un-fused Duchon designs must have identical shape"
    );

    // `(K·Z)·V` and `K·(Z·V)` are the same product in exact arithmetic; in
    // floating point they differ only by benign reassociation. Require agreement
    // to a tight relative tolerance scaled by the design magnitude.
    let max_abs = fused_design
        .iter()
        .chain(unfused_design.iter())
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let tol = 1e-9 * max_abs.max(1.0);
    let max_diff = fused_design
        .iter()
        .zip(unfused_design.iter())
        .fold(0.0_f64, |acc, (&a, &b)| acc.max((a - b).abs()));
    assert!(
        max_diff <= tol,
        "#1718 fusion changed the Duchon design: max element-wise diff {max_diff:.3e} \
         exceeds tolerance {tol:.3e} (design max-abs {max_abs:.3e}). The fused `(K·Z)·V` \
         path must reproduce the un-fused `K·(Z·V)` design."
    );
}
