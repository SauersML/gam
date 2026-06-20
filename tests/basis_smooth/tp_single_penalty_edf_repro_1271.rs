//! #1271 localization repro: `s(x, bs="tp")` single-penalty OVER-FITS linear
//! data (gam EDF ≈ 4.87 vs mgcv 2.10). This is ridge-INDEPENDENT (tp single ==
//! tp double here), so distinct from #1266 (the ps double-penalty ridge).
//!
//! DGP: y = 2 + 3x + N(0,0.15), x = linspace(0,1,800), k = 20, Gaussian, REML.
//! mgcv `s(x,bs="tp",k=20)` REML → EDF ≈ 2.10. The truth lives entirely in the
//! {1,x} polynomial null space, so the correct optimum suppresses ALL kernel
//! (wiggle) EDF, leaving only the 2-D null → EDF ≈ 2.
//!
//! This DIAGNOSTIC prints the converged per-block λ + EDF AND the optimized
//! length-scale, so we can see WHICH lever inflates: a collapsed λ_bend
//! (under-penalized wiggle) or a runaway optimized length-scale (an extra
//! hyper-axis mgcv's pure-r³ tp does NOT have).

use gam::estimate::FitOptions;
use gam::smooth::{
    SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec, fit_term_collection_forspec,
};
use gam::terms::basis::{CenterStrategy, SpatialIdentifiability, ThinPlateBasisSpec};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, RhoPrior, StandardLink};
use ndarray::{Array1, Array2};

fn fit_options() -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 200,
        tol: 1e-10,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: RhoPrior::Flat,
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    }
}

fn linear_dgp(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut data = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x = i as f64 / (n as f64 - 1.0);
        let u1 = next().max(1e-12);
        let u2 = next();
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        data[[i, 0]] = x;
        y[i] = 2.0 + 3.0 * x + 0.15 * z;
    }
    (data, y)
}

/// k=20 thin-plate, matching mgcv's `s(x,bs="tp",k=20)`. `length_scale = 0.0`
/// is the auto-init sentinel (the term-builder default) — the data-derived
/// length-scale is then OPTIMIZED as a spatial hyper-axis.
fn tp_spec(double_penalty: bool) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "s_x".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0],
                spec: ThinPlateBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 20 },
                    periodic: None,
                    length_scale: 0.0,
                    double_penalty,
                    identifiability: SpatialIdentifiability::OrthogonalToParametric,
                    radial_reparam: None,
                },
                input_scales: None,
            },
            shape: gam::terms::smooth::ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

#[test]
fn tp_single_penalty_edf_inflation_1271() {
    let n = 800usize;
    let likelihood = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );
    let opts = fit_options();

    eprintln!(
        "[1271-repro] mgcv s(x,bs=tp,k=20) REML EDF ≈ 2.10. Truth is in {{1,x}} null → optimum EDF≈2."
    );
    eprintln!(
        "[1271-repro] {:>4}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}",
        "seed", "edf_1pen", "edf_2pen", "lambda_bend", "lambda_null", "edf_by_block"
    );

    let mut single_vals = Vec::new();
    let mut double_vals = Vec::new();
    for seed in 0..5u64 {
        let (data, y) = linear_dgp(n, seed);
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);

        let fit_single = fit_term_collection_forspec(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &tp_spec(false),
            likelihood.clone(),
            &opts,
        )
        .expect("tp single-penalty fit");
        let fit_double = fit_term_collection_forspec(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &tp_spec(true),
            likelihood.clone(),
            &opts,
        );

        let edf_single = fit_single.fit.edf_total().unwrap_or(f64::NAN);
        let edf_double = fit_double
            .as_ref()
            .ok()
            .and_then(|f| f.fit.edf_total())
            .unwrap_or(f64::NAN);
        let lam = fit_single.fit.lambdas.to_vec();
        let lambda_bend = lam.first().copied().unwrap_or(f64::NAN);
        let lambda_null = lam.get(1).copied().unwrap_or(f64::NAN);
        let edf_blocks = fit_single
            .fit
            .edf_by_block()
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join(",");

        eprintln!(
            "[1271-repro] {seed:>4}  {edf_single:>10.4}  {edf_double:>10.4}  {lambda_bend:>14.4e}  \
             {lambda_null:>14.4e}  [{edf_blocks}]"
        );
        single_vals.push(edf_single);
        double_vals.push(edf_double);
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    eprintln!(
        "[1271-repro] MEAN edf_single={:.4}  edf_double={:.4}  (mgcv target ≈ 2.10)",
        mean(&single_vals),
        mean(&double_vals)
    );

    assert!(
        single_vals.iter().all(|v| v.is_finite()),
        "tp single-penalty fits must converge to finite EDF"
    );
    assert!(
        mean(&single_vals) < 3.0,
        "tp single-penalty over-fits linear data: mean EDF={:.4}, per-seed={single_vals:?}",
        mean(&single_vals)
    );
}
