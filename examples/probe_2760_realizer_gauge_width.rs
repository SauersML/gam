//! gam#2760 blocker probe: WHERE does the incremental realizer lose a column?
//!
//! Every arm of `tests/perf_scale/misc/kappa_loop_n_scaling.rs` dies in 0.2 s at
//! current `main` with
//!
//! ```text
//!   incremental realizer width mismatch for term 0: rebuilt_cols=10, cached_cols=11
//! ```
//!
//! so the issue's own acceptance arms cannot be measured at all. The mismatch is
//! between the width the COLLECTION build stored (`coeff_range`, 11) and the
//! width a TERM-LOCAL rebuild has after `place_term_in_collection_gauge` (10).
//! Three widths decide which of the two halves moved, and this prints all three
//! on the fixture's own spec:
//!
//!   1. the local build's own `design.ncols()` at the seed length scale,
//!   2. the collection design's `coeff_range` for that term, plus the gauge it
//!      exported (arm, constraint-block shape),
//!   3. the same local build at a MOVED length scale, put back through
//!      `place_term_in_collection_gauge` — the realizer's own path.
//!
//! Report-only.

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, SpatialIdentifiability,
};
use gam::smooth::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec};
use ndarray::{Array1, Array2};

fn simulate_1d_gaussian(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = (i as f64) / (n as f64 - 1.0) * 6.0 - 3.0;
        x[[i, 0]] = t;
        y[i] = t.sin();
    }
    (x, y)
}

fn term_spec(length_scale: f64) -> SmoothTermSpec {
    SmoothTermSpec {
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
                operator_penalties: DuchonOperatorPenaltySpec::all_active(),
                boundary: OneDimensionalBoundary::Open,
            },
            input_scale: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    }
}

fn spec_1d(length_scale: f64) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![term_spec(length_scale)],
    }
}

fn main() {
    let n = 600usize;
    let (x, _y) = simulate_1d_gaussian(n);

    // 1. The term-local build, exactly as the realizer drives it.
    let mut ws = gam_terms::basis::BasisWorkspace::new();
    let local = gam_terms::smooth::build_single_local_smooth_term(x.view(), &term_spec(1.0), &mut ws)
        .expect("local build at the seed length scale");
    println!(
        "[probe2760] local build @ ell=1.0: design {}x{} dim={} joint_null_rotation={:?}",
        local.design.nrows(),
        local.design.ncols(),
        local.dim,
        local
            .joint_null_rotation
            .as_ref()
            .map(|rotation| (rotation.rotation.nrows(), rotation.rotation.ncols())),
    );

    // 2. The collection build's own width and the gauge it exported.
    let collection = gam_terms::smooth::build_term_collection_design(x.view(), &spec_1d(1.0))
        .expect("collection build at the seed length scale");
    let term = &collection.smooth.terms[0];
    println!(
        "[probe2760] collection @ ell=1.0: design {}x{} term.coeff_range={:?} (width {})",
        collection.design.nrows(),
        collection.design.ncols(),
        term.coeff_range,
        term.coeff_range.len(),
    );
    match term.collection_gauge.as_ref() {
        Some(gauge) => println!(
            "[probe2760] gauge: arm={:?} constraint_block={}x{} owner_terms={:?} has_parametric_block={}",
            gauge.arm,
            gauge.constraint_block.nrows(),
            gauge.constraint_block.ncols(),
            gauge.owner_terms,
            gauge.has_parametric_block,
        ),
        None => println!("[probe2760] gauge: NONE exported"),
    }

    // 3. A MOVED local build put back through the collection's gauge — the
    //    realizer's own path, minus the splice.
    let Some(gauge) = term.collection_gauge.as_ref() else {
        println!("[probe2760] no gauge to place into; nothing more to measure");
        return;
    };
    // The realizer does not rebuild from the caller's spec: it rebuilds from a
    // FROZEN one (`freeze_geometry_from_metadata`), whose centers are
    // `UserProvided` and whose `input_scale` is pinned. Mirror that here, since
    // it is the only difference between this probe and the failing path.
    let frozen_spec = |ell: f64| -> SmoothTermSpec {
        let mut spec = term_spec(ell);
        if let (
            SmoothBasisSpec::Duchon {
                spec: duchon,
                input_scale,
                ..
            },
            gam_terms::basis::BasisMetadata::Duchon {
                centers,
                input_scale: metadata_scale,
                ..
            },
        ) = (&mut spec.basis, &local.metadata)
        {
            duchon.center_strategy = CenterStrategy::UserProvided(centers.clone());
            *input_scale = Some(*metadata_scale);
        }
        spec
    };
    // The same freeze PLUS the #1355 data-metric radial reparameterization `V`,
    // whose own doc says it is stored "so predict-time and κ-trial rebuilds
    // replay the exact fit-time rotated radial basis" — and which the κ-trial
    // freeze does not carry.
    let frozen_spec_with_reparam = |ell: f64| -> SmoothTermSpec {
        let mut spec = frozen_spec(ell);
        if let (
            SmoothBasisSpec::Duchon { spec: duchon, .. },
            gam_terms::basis::BasisMetadata::Duchon { radial_reparam, .. },
        ) = (&mut spec.basis, &local.metadata)
        {
            duchon.radial_reparam = radial_reparam.clone();
        }
        spec
    };
    // Is the frozen rebuild even the SAME geometry? The metadata's centers are
    // documented as living in the standardized frame; a spec that carries them
    // as `UserProvided` while also pinning `input_scale` can standardize them a
    // second time. Read both builds' realized centers and compare.
    {
        let frozen_at_seed =
            gam_terms::smooth::build_single_local_smooth_term(x.view(), &frozen_spec(1.0), &mut ws)
                .expect("frozen local build at the seed");
        if let (
            gam_terms::basis::BasisMetadata::Duchon {
                centers: cold,
                input_scale: cold_scale,
                ..
            },
            gam_terms::basis::BasisMetadata::Duchon {
                centers: warm,
                input_scale: warm_scale,
                ..
            },
        ) = (&local.metadata, &frozen_at_seed.metadata)
        {
            let ratio = if cold.is_empty() || warm.is_empty() {
                f64::NAN
            } else {
                let cold_rms =
                    (cold.iter().map(|v| v * v).sum::<f64>() / cold.len() as f64).sqrt();
                let warm_rms =
                    (warm.iter().map(|v| v * v).sum::<f64>() / warm.len() as f64).sqrt();
                cold_rms / warm_rms
            };
            println!(
                "[probe2760] centers @ ell=1.0: cold {}x{} rms-ratio(cold/frozen)={ratio:.6} \
                 cold_input_scale={cold_scale:?} frozen_input_scale={warm_scale:?}",
                cold.nrows(),
                cold.ncols(),
            );
        }
    }
    for ell in [
        1.0_f64, 0.5, 2.0, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1, 3.0, 10.0, 30.0, 1.0e2,
    ] {
        let frozen = frozen_spec(ell);
        let frozen_build =
            gam_terms::smooth::build_single_local_smooth_term(x.view(), &frozen, &mut ws)
                .expect("frozen-geometry local build");
        let with_reparam = gam_terms::smooth::build_single_local_smooth_term(
            x.view(),
            &frozen_spec_with_reparam(ell),
            &mut ws,
        );
        println!(
            "[probe2760] ell={ell}: FROZEN-geometry local build -> {} cols (unfrozen gives 12); \
             frozen+radial_reparam -> {}",
            frozen_build.design.ncols(),
            match &with_reparam {
                Ok(build) => format!("{} cols", build.design.ncols()),
                Err(error) => format!("FAILED: {error}"),
            },
        );
        let moved = gam_terms::smooth::build_single_local_smooth_term(
            x.view(),
            &term_spec(ell),
            &mut ws,
        )
        .expect("local build at the moved length scale");
        let local_cols = moved.design.ncols();
        let placed = gam_terms::smooth::place_term_in_collection_gauge(
            gauge,
            gam_terms::smooth::LocalTermRealization {
                design: moved.design,
                metadata: &moved.metadata,
                active_penalties: &moved.active_penalties,
                dropped_penalties: moved.dropped_penalties.clone(),
                linear_constraints_local: moved.linear_constraints.as_ref(),
                joint_null_rotation: moved.joint_null_rotation.as_ref(),
                termname: "duchon_1d",
            },
        );
        match placed {
            Ok(placed) => println!(
                "[probe2760] ell={ell}: local_cols={local_cols} -> placed_cols={} (collection cached {})",
                placed.design.ncols(),
                term.coeff_range.len(),
            ),
            Err(error) => println!("[probe2760] ell={ell}: local_cols={local_cols} -> PLACEMENT FAILED: {error}"),
        }
    }
}
