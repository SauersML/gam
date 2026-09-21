//! Measurement probe for #2445 / #2444 — the `DoublePenaltyNullspace` rebuild
//! frame, before and after the structural-frame fix.
//!
//! PRE-FIX MODEL (confirmed to every printed digit before the fix landed):
//! `rebuild_metric_consistent_ridge` built `R = N (NᵀR_cN) Nᵀ` with
//! `N = null(S_c)` from a numerical rank test; `dim N = 1` here, so the
//! shipped block was exactly `u uᵀ` and its whole FD residue
//! (`fd_norm = 2.3142e-6`) was the frame motion `√2‖u′‖` — the `√ε` affine
//! conditioning ridge tilting `null(S_c)` as ψ moves (measured principal
//! angle `3.2728e-10` at `ε = 1e-4`, agreeing with the prediction from
//! `fd_norm` to every digit).
//!
//! POST-FIX PINS (what this probe now gates):
//!  1. the rank-test frame STILL moves with ψ — and under an orthonormal
//!     collection chart it is empty, the ridge reading as range — which is
//!     why a rank test may not decide topology; printed, not asserted;
//!  2. the SHIPPED trend ridge no longer moves: its central FD across ±ε is
//!     rebuild roundoff, orders below the pre-fix 2.3e-6;
//!  3. the shipped ridge's range is the STRUCTURAL frame
//!     `{γ : Tγ ∈ span(poly block)} = null(T[..kernel_cols, :])`, which is
//!     ψ-invariant under the frozen chart by construction.

#![cfg(test)]

use ndarray::{Array2, ArrayView2, s};

use super::*;

/// The `_frozen` / `_linear` fixture from
/// `test_duchon_log_kappa_derivative_matchesfd_dim1_power1_*`, chart frozen.
fn frozen_fixture() -> (Array2<f64>, DuchonBasisSpec) {
    let n = 80usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = i as f64 / (n as f64 - 1.0);
    }
    let mut spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
        length_scale: Some(1.0),
        power: 1.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::default(),
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let base = build_duchon_basis(data.view(), &spec).expect("cold base build");
    if let BasisMetadata::Duchon {
        centers,
        identifiability_transform,
        radial_reparam,
        ..
    } = &base.metadata
    {
        spec.center_strategy = CenterStrategy::UserProvided(centers.clone());
        spec.radial_reparam = radial_reparam.clone();
        spec.identifiability = match identifiability_transform {
            Some(t) => SpatialIdentifiability::FrozenTransform {
                transform: t.clone(),
            },
            None => SpatialIdentifiability::None,
        };
    } else {
        panic!("expected Duchon metadata");
    }
    (data, spec)
}

/// The `(kernel_transform, outer_identifiability)` pair the forward penalty
/// builder consumes, replicated exactly from
/// `build_duchon_native_penalty_psi_derivatives`.
fn frozen_chart_transforms(
    centers: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> (Array2<f64>, Option<Array2<f64>>, usize, usize) {
    let mut workspace = BasisWorkspace::default();
    let order = duchon_effective_nullspace_order(centers, spec.nullspace_order);
    let mut z = kernel_constraint_nullspace(centers, order, &mut workspace.cache)
        .expect("kernel constraint nullspace");
    if let Some(v) = spec.radial_reparam.as_ref() {
        z = z.dot(v);
    }
    let kernel_cols = z.ncols();
    let poly_cols = polynomial_block_from_order(centers, order).ncols();
    let transform = match &spec.identifiability {
        SpatialIdentifiability::FrozenTransform { transform } => Some(transform.clone()),
        _ => None,
    };
    (z, transform, kernel_cols, poly_cols)
}

/// `sin θ_max` between two orthonormal frames, computed as
/// `‖(I − BBᵀ)A‖₂` rather than `acos(σ_min(AᵀB))`.
///
/// The cosine form cannot resolve this measurement at all: the angle the model
/// predicts is `3.3e-10`, whose cosine is `1 − 5.4e-20`, which rounds to
/// exactly `1.0` in double precision and reports an angle of zero. The
/// projection residual loses no digits.
fn frame_sin_angle(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.ncols(), b.ncols(), "frames must have equal dimension");
    let residual = a - &b.dot(&b.t().dot(a));
    let (_, singular, _) = gam_linalg::faer_ndarray::FaerSvd::svd(&residual, false, false)
        .expect("svd of projection residual");
    singular.iter().copied().fold(0.0_f64, f64::max)
}

#[test]
fn zz_measure_2445_rank_test_frame_moves_with_psi() {
    let (_, spec) = frozen_fixture();
    let centers = match &spec.center_strategy {
        CenterStrategy::UserProvided(c) => c.clone(),
        _ => unreachable!("fixture freezes the centers"),
    };
    let (z, transform, kernel_cols, poly_cols) = frozen_chart_transforms(centers.view(), &spec);
    eprintln!(
        "[2445] chart: kernel_cols={kernel_cols} poly_cols={poly_cols} \
         transform={:?}",
        transform.as_ref().map(|t| t.dim())
    );

    let eps = 1e-4_f64;
    let mut frames: Vec<Array2<f64>> = Vec::new();
    let mut ridges: Vec<Array2<f64>> = Vec::new();
    for (label, psi) in [("minus", -eps), ("zero", 0.0), ("plus", eps)] {
        let length_scale = 1.0 / psi.exp();
        let candidates = duchon_native_penalty_candidates(
            centers.view(),
            Some(length_scale),
            spec.power,
            duchon_effective_nullspace_order(centers.view(), spec.nullspace_order),
            None,
            &z,
            transform.as_ref(),
        )
        .expect("native penalty candidates");
        let primary = candidates
            .iter()
            .find(|c| matches!(c.source, PenaltySource::Primary))
            .expect("primary");
        let ridge = candidates
            .iter()
            .find(|c| matches!(c.source, PenaltySource::DoublePenaltyNullspace))
            .expect("trend ridge");
        let physical_primary = primary
            .matrix
            .scaled(primary.normalization_scale, "physical primary")
            .expect("scale primary");
        let null = constructive_nullspace_basis(&physical_primary)
            .expect("nullspace basis")
            .unwrap_or_else(|| Array2::<f64>::zeros((physical_primary.nrows(), 0)));
        eprintln!(
            "[2445] psi={psi:+.1e} ({label}): rank-test null dim={} ridge_norm={:.6e}",
            null.ncols(),
            ridge
                .matrix
                .dense()
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt(),
        );
        frames.push(null);
        ridges.push(ridge.matrix.dense().to_owned());
    }

    let angle = frame_sin_angle(&frames[0], &frames[2]);
    let fd = (&ridges[2] - &ridges[0]) / (2.0 * eps);
    let fd_norm = fd.iter().map(|v| v * v).sum::<f64>().sqrt();
    eprintln!(
        "[2445] principal angle between null(S_c)(-eps) and null(S_c)(+eps) = {angle:.6e}\n\
         [2445] fd_norm(shipped trend ridge) = {fd_norm:.6e}\n\
         [2445] pre-fix fd_norm was 2.3142e-6 (= the frame motion, #2444)",
    );

    // Pin 2 — the SHIPPED block is ψ-invariant under the frozen chart: its
    // central FD is rebuild roundoff (≈1e-14/entry eigendecomposition noise
    // divided by 2ε), not the 2.3e-6 frame motion. Same floor derivation as
    // the `_frozen`/`_linear` gates.
    let entries = ridges[1].len() as f64;
    let fd_roundoff_floor = 1e2 * 1e-14 * entries.sqrt() / (2.0 * eps);
    assert!(
        fd_norm <= fd_roundoff_floor,
        "shipped trend ridge must not move with ψ under a frozen chart: \
         fd_norm={fd_norm:.6e} floor={fd_roundoff_floor:.6e}"
    );

    // Pin 3 — the shipped ridge's range is the STRUCTURAL frame. With a 1-D
    // frame the normalized block is exactly `u uᵀ`, so compare `u` (leading
    // eigenvector of the shipped block) against the structural transport.
    if let Some(t) = transform.as_ref() {
        let kernel_rows_t = t.slice(s![..kernel_cols, ..]).t().to_owned();
        let (structural, rank) =
            gam_linalg::faer_ndarray::rrqr_nullspace_basis(&kernel_rows_t)
                .expect("structural transport nullspace");
        eprintln!(
            "[2445] structural transported frame: rank(T_kernel)={rank} dim={} (expected {})",
            structural.ncols(),
            poly_cols - 1
        );
        // Whether the rank test finds the structural direction at all is
        // itself a function of the chart: the `√ε` ridge sits within a decade
        // of the rank cutoff, so it reads as null under a data-whitened chart
        // and as range under an orthonormal one. Either way it is printed,
        // never asserted — that is pin 1's point.
        if structural.ncols() == frames[1].ncols() {
            let overlap = frame_sin_angle(&structural, &frames[1]);
            eprintln!(
                "[2445] angle(structural frame, rank-test frame at psi=0) = {overlap:.6e} \
                 (pre-fix: 4.32e-6 — the conditioning-ridge tilt)"
            );
        } else {
            eprintln!(
                "[2445] rank-test null dim {} vs structural dim {}: the rank test reads \
                 the conditioning ridge as range in this chart",
                frames[1].ncols(),
                structural.ncols()
            );
        }
        if structural.ncols() == 1 {
            let u = structural.column(0);
            let ru = ridges[1].dot(&u);
            let ru_norm = ru.iter().map(|v| v * v).sum::<f64>().sqrt();
            // `R = u uᵀ` on the structural direction ⇒ `‖R u‖ = 1` after unit
            // Frobenius normalization; a residual off `span(u)` means the
            // shipped range is NOT the structural frame.
            let residual = &ru - &(u.to_owned() * ru.dot(&u));
            let off = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
            eprintln!(
                "[2445] shipped-ridge action on structural u: ‖Ru‖={ru_norm:.6e} \
                 off-span residual={off:.6e}"
            );
            assert!(
                off <= 1e-8 * ru_norm.max(1e-12),
                "shipped trend ridge range must be the structural frame: \
                 off-span residual {off:.3e} vs ‖Ru‖ {ru_norm:.3e}"
            );
        }
    }
}

/// #2445 symptom 2 — the double-penalty TOPOLOGY must not depend on the
/// kernel Gram's conditioning. Under one frozen chart, sweep κ across five
/// octaves: the trend ridge must be emitted, nonzero, and retained by the
/// candidate filter in EVERY arm, and the emitted penalty count must be
/// κ-invariant (this is also what #2433's per-κ-trial guard needs: a topology
/// that can flip with κ is a latent hard refusal on every trial move).
#[test]
fn duchon_trend_ridge_topology_is_kappa_invariant_2445() {
    let (_, spec) = frozen_fixture();
    let centers = match &spec.center_strategy {
        CenterStrategy::UserProvided(c) => c.clone(),
        _ => unreachable!("fixture freezes the centers"),
    };
    let (z, transform, _, _) = frozen_chart_transforms(centers.view(), &spec);
    let mut counts = Vec::new();
    for octave in 0..6 {
        let length_scale = f64::from(1u32 << octave);
        let candidates = duchon_native_penalty_candidates(
            centers.view(),
            Some(length_scale),
            spec.power,
            duchon_effective_nullspace_order(centers.view(), spec.nullspace_order),
            None,
            &z,
            transform.as_ref(),
        )
        .expect("native penalty candidates");
        let filtered = filter_penalty_candidates(candidates).expect("filter penalty candidates");
        let trend_retained = filtered
            .active
            .iter()
            .any(|penalty| matches!(penalty.info.source, PenaltySource::DoublePenaltyNullspace));
        eprintln!(
            "[2445-topology] length_scale={length_scale}: active={} trend_retained={trend_retained}",
            filtered.active.len(),
        );
        assert!(
            trend_retained,
            "trend ridge must be emitted at length_scale={length_scale}: \
             the null space of a curvature seminorm is a theorem, not a \
             property of the Gram's conditioning"
        );
        counts.push(filtered.active.len());
    }
    assert!(
        counts.windows(2).all(|pair| pair[0] == pair[1]),
        "emitted penalty topology must be κ-invariant under a frozen chart, got {counts:?}"
    );
}
