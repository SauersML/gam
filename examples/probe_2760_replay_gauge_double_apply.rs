//! gam#2760 blocker probe, second stage: replay the incremental realizer's OWN
//! sequence outside a fit, and read the orthogonality residual the collection
//! gauge decides on.
//!
//! Stage one (`probe_2760_realizer_gauge_width`) showed the local rebuild width
//! is ψ-dependent when the radial chart is re-derived. The fit's refusal reports
//! something narrower: at the SEED length scale, with the geometry cached and the
//! chart frozen `(10, 10)`, the local rebuild produces **11** columns and the
//! collection gauge then deletes one more, against a cached width of 11.
//!
//! Eleven is the FIT's width, so the local rebuild is already carrying the
//! collection's transform — the replay spec froze it — and the gauge is applying
//! that step a second time. The question this probe answers is whether the
//! doubly-charted design is still orthogonal to the constraint block (in which
//! case the Delete arm should be a no-op and is not) or genuinely is not.
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

fn chart_shape(spec: &SmoothTermSpec) -> String {
    match &spec.basis {
        SmoothBasisSpec::Duchon { spec, .. } => match &spec.identifiability {
            SpatialIdentifiability::FrozenTransform { transform } => {
                format!("FrozenTransform{:?}", transform.dim())
            }
            other => format!("{other:?}"),
        },
        _ => "not a Duchon term".to_string(),
    }
}

fn main() {
    let n = 600usize;
    let (x, _y) = simulate_1d_gaussian(n);
    let spec = spec_1d(1.0);

    let collection = gam_terms::smooth::build_term_collection_design(x.view(), &spec)
        .expect("collection build");
    let term = &collection.smooth.terms[0];
    println!(
        "[probe2760b] collection: width={} gauge={:?}",
        term.coeff_range.len(),
        term.collection_gauge.as_ref().map(|gauge| (
            gauge.arm,
            gauge.constraint_block.nrows(),
            gauge.constraint_block.ncols()
        )),
    );

    // The realizer's replay specification, verbatim.
    let frozen = gam_terms::smooth::freeze_term_collection_from_design(&spec, &collection)
        .expect("freeze the replay specification");
    println!(
        "[probe2760b] replay spec identifiability: {}",
        chart_shape(&frozen.smooth_terms[0]),
    );

    let mut ws = gam_terms::basis::BasisWorkspace::new();
    for ell in [
        1.0_f64, 0.5, 2.0, 1.0e-2, 2.6e-2, 5.0e-2, 1.0e-1, 1.0e1, 1.0e2,
    ] {
        let mut trial = frozen.smooth_terms[0].clone();
        if let SmoothBasisSpec::Duchon { spec, .. } = &mut trial.basis {
            spec.length_scale = Some(ell);
        }
        let local = gam_terms::smooth::build_single_local_smooth_term(x.view(), &trial, &mut ws)
            .expect("local rebuild from the replay spec");
        let residual = term.collection_gauge.as_ref().map(|gauge| {
            gam_terms::smooth::orthogonality_relative_residual_for_design(
                &local.design,
                gauge.constraint_block.view(),
            )
        });
        // What the gauge's realization does to the width, and the conditioning
        // of the local Gram it decides on.
        let placed = term.collection_gauge.as_ref().map(|gauge| {
            gam_terms::smooth::place_term_in_collection_gauge(
                gauge,
                gam_terms::smooth::LocalTermRealization {
                    design: local.design.clone(),
                    metadata: &local.metadata,
                    active_penalties: &local.active_penalties,
                    dropped_penalties: local.dropped_penalties.clone(),
                    linear_constraints_local: local.linear_constraints.as_ref(),
                    joint_null_rotation: local.joint_null_rotation.as_ref(),
                    termname: "duchon_1d",
                },
            )
            .map(|placed| placed.design.ncols())
        });
        let dense = local.design.to_dense();
        let gram = gam_linalg::faer_ndarray::fast_atb(&dense, &dense);
        let spectrum = gam_linalg::faer_ndarray::FaerEigh::eigh(&gram, faer::Side::Lower)
            .ok()
            .map(|(values, _)| {
                let max = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
                let min = values.iter().fold(f64::INFINITY, |acc, v| acc.min(v.abs()));
                (min, max, min / max)
            });
        println!(
            "[probe2760b] ell={ell:>8.3e}: local {} cols -> placed {:?}, residual {:?}, \
             local-Gram (min,max,ratio) {:?}",
            local.design.ncols(),
            placed,
            residual,
            spectrum,
        );
    }
}
