//! The analytic magnitude (mass) penalty is OPT-IN and reachable.
//!
//! The non-periodic Euclidean Duchon default penalty is the native
//! reproducing-norm roughness Gram (`Primary`) + null-space ridge — it shrinks
//! WIGGLE, not amplitude. The redesign dropped the per-order operator dials; this
//! restores the magnitude/mass one ANALYTICALLY: a closed-form L2 ridge on the
//! smooth's coefficients (`‖β_kernel‖²`, an `OperatorMass` block with its own REML
//! λ) that shrinks the smooth's amplitude toward zero. No quadrature, no sparse
//! collocation. It is OFF by default and turned on with `duchon(..., magnitude=true)`.

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, PenaltySource, SpatialIdentifiability, build_duchon_basis,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn synthetic_data(n: usize, d: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Uniform::new(-1.0_f64, 1.0).expect("uniform");
    let mut data = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            data[[i, j]] = dist.sample(&mut rng);
        }
    }
    data
}

fn duchon_spec(k: usize, operator_penalties: DuchonOperatorPenaltySpec) -> DuchonBasisSpec {
    DuchonBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: k },
        periodic: None,
        length_scale: None,
        power: 0.5,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties,
        boundary: OneDimensionalBoundary::default(),
    }
}

fn penalty_sources(spec: &DuchonBasisSpec, data: &Array2<f64>) -> Vec<PenaltySource> {
    let built = build_duchon_basis(data.view(), spec).expect("build_duchon_basis");
    built
        .penaltyinfo
        .iter()
        .filter(|info| info.active)
        .map(|info| info.source.clone())
        .collect()
}

/// Wiring: the magnitude (`OperatorMass`) penalty is emitted ONLY when opted in,
/// and the default penalty set is unchanged (`Primary` + null-space ridge only).
#[test]
fn magnitude_penalty_is_emitted_only_when_opted_in() {
    let data = synthetic_data(200, 2, 7);

    let default_sources = penalty_sources(&duchon_spec(20, DuchonOperatorPenaltySpec::default()), &data);
    assert!(
        default_sources
            .iter()
            .any(|s| matches!(s, PenaltySource::Primary)),
        "default Duchon must still emit the native roughness Gram (Primary); got {default_sources:?}"
    );
    assert!(
        !default_sources
            .iter()
            .any(|s| matches!(s, PenaltySource::OperatorMass)),
        "default Duchon must NOT include a magnitude (OperatorMass) penalty; got {default_sources:?}"
    );

    let mag_sources =
        penalty_sources(&duchon_spec(20, DuchonOperatorPenaltySpec::magnitude_only()), &data);
    assert!(
        mag_sources
            .iter()
            .any(|s| matches!(s, PenaltySource::Primary)),
        "magnitude opt-in must KEEP the native roughness Gram (Primary); got {mag_sources:?}"
    );
    assert!(
        mag_sources
            .iter()
            .any(|s| matches!(s, PenaltySource::OperatorMass)),
        "magnitude opt-in must add the analytic magnitude (OperatorMass) penalty; got {mag_sources:?}"
    );
}

/// End-to-end: `duchon(x, magnitude=true)` parses, fits, and recovers the signal
/// (the opt-in is reachable through the formula and does not break the fit).
#[test]
fn magnitude_formula_optin_fits_and_recovers() {
    init_parallelism();

    let n = 200usize;
    let mut rng = StdRng::seed_from_u64(31);
    let noise = Normal::new(0.0, 0.08).expect("normal");
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let two_pi_f = 2.0 * std::f64::consts::PI * 3.0;
    let y: Vec<f64> = x
        .iter()
        .map(|&t| (two_pi_f * t).sin() + noise.sample(&mut rng))
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ duchon(x, k=20, magnitude=true)", &ds, &cfg).expect("magnitude fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };

    // Fitted values at the training rows must be finite and recover the signal
    // (RMSE well below the trivial predictor, RMS(sin) ≈ 0.707). Rebuild the
    // design at the training x from the frozen spec (identity link ⇒ Xβ = mean).
    let x_idx = ds.column_map()["x"];
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for (i, &t) in x.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training x");
    let fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert!(
        fitted.iter().all(|v| v.is_finite()),
        "magnitude opt-in produced non-finite fitted values"
    );
    let truth: Vec<f64> = x.iter().map(|&t| (two_pi_f * t).sin()).collect();
    let rmse = (fitted
        .iter()
        .zip(truth.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    assert!(
        rmse < 0.3,
        "magnitude opt-in failed to recover sin: rmse={rmse:.4} (trivial ≈ 0.707)"
    );
}
