//! #1266 localization repro: the default double penalty (Marra–Wood nullspace
//! shrinkage) INFLATES a B-spline smooth's EDF on linear data instead of leaving
//! it at the {1,x} null (EDF≈2). The mgcv reference (`s(x,bs="ps",select=TRUE)`,
//! REML) keeps EDF≈2 on the identical DGP — so the correct REML optimum IS
//! EDF≈2 and this is a gam λ-selection bug, not an inherent property.
//!
//! Construction-side levers were proven to be no-ops (per-block normalization is
//! absorbed by the free per-block ρ; an orthogonal [range|null] reparam leaves
//! the fit β̂ basis-invariant). The remaining locus is gam's λ-SELECTION — the
//! REML score that picks (λ_bend, λ_null) or the optimizer that minimizes it.
//!
//! This is a DIAGNOSTIC (report-only, no hard assertion on the inflation): it
//! prints the converged per-block λ's and EDF for the double-penalty fit so we
//! can see WHICH parameter is mis-selected. mgcv's correct optimum is
//! λ_bend HIGH (suppress wiggle, range-EDF≈0) and λ_null LOW (retain the linear
//! trend, null-EDF≈2). If gam instead converges with λ_bend LOW, the wiggle
//! (range) block is under-penalized — the inflation is the range EDF, and the
//! coupling that lowered λ_bend is the bug.

use gam::estimate::FitOptions;
use gam::smooth::{
    SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec, fit_term_collection_forspec,
};
use gam::terms::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineIdentifiability, BSplineKnotSpec,
    OneDimensionalBoundary,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, RhoPrior, StandardLink};
use ndarray::{Array1, Array2};

fn fit_options() -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true, // need edf_total()/edf_by_block() (read from inference)
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

/// The exact issue DGP: y = 2 + 3x + N(0, 0.15) on a uniform x∈[0,1], so the
/// truth lives entirely in the {1,x} null space of the 2nd-difference penalty.
fn linear_dgp(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    // Deterministic LCG noise so the repro is seed-stable without an RNG dep.
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let u = (state.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64;
        u
    };
    let mut data = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x = i as f64 / (n as f64 - 1.0);
        // Box–Muller for a standard normal from two uniforms.
        let u1 = next().max(1e-12);
        let u2 = next();
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        data[[i, 0]] = x;
        y[i] = 2.0 + 3.0 * x + 0.15 * z;
    }
    (data, y)
}

fn bspline_spec(double_penalty: bool) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "s_x".to_string(),
            basis: SmoothBasisSpec::BSpline1D {
                feature_col: 0,
                spec: BSplineBasisSpec {
                    degree: 3,
                    penalty_order: 2,
                    knotspec: BSplineKnotSpec::Generate {
                        data_range: (0.0, 1.0),
                        // 16 internal knots ≈ k=20 basis fns — the issue's config,
                        // where the ps double penalty inflates EDF (2.56→5.28).
                        // 10 knots did NOT reproduce the inflation (λ_bend stayed
                        // large/correct, EDF≈1.2) → the bug is knot-count
                        // dependent; probe at the config that actually inflates.
                        num_internal_knots: 16,
                    },
                    double_penalty,
                    // Sum-to-zero centering matches gamfit's `s(x)` (the mgcv
                    // convention the issue's reproducer uses). `None` left the
                    // unpenalized constant aliased → the single-penalty fit was
                    // pre-fit rank deficient (NaN), and the raw frame did not
                    // reproduce the gamfit-path inflation.
                    identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
                    boundary: OneDimensionalBoundary::Open,
                    boundary_conditions: BSplineBoundaryConditions::default(),
                },
            },
            shape: gam::terms::smooth::ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

#[test]
fn double_penalty_edf_inflation_localization_1266() {
    let n = 800usize;
    let likelihood = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );
    let opts = fit_options();

    eprintln!(
        "[1266-repro] mgcv reference on this DGP: select=TRUE EDF≈2.10, select=FALSE EDF≈2.10 \
         (contract EDF_on≤EDF_off HOLDS). Correct REML optimum is EDF≈2."
    );
    eprintln!(
        "[1266-repro] {:>4}  {:>10}  {:>10}  {:>14}  {:>14}  {:>12}",
        "seed", "edf_ON", "edf_OFF", "lambda_bend", "lambda_null", "reml_ON"
    );

    let mut on_vals = Vec::new();
    let mut off_vals = Vec::new();
    for seed in 0..5u64 {
        let (data, y) = linear_dgp(n, seed);
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);

        let fit_on = fit_term_collection_forspec(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &bspline_spec(true),
            likelihood.clone(),
            &opts,
        )
        .expect("double-penalty fit");
        // Single-penalty fit is the in-gam control (≈ mgcv ≈ 2.10). It can hit a
        // pre-fit rank deficiency on some knot counts (the boundary columns the
        // double penalty's ridge would otherwise regularize); keep it non-fatal
        // so the load-bearing double-penalty trace always prints.
        let fit_off = fit_term_collection_forspec(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &bspline_spec(false),
            likelihood.clone(),
            &opts,
        );

        let edf_on = fit_on.fit.edf_total().unwrap_or(f64::NAN);
        let edf_off = fit_off
            .as_ref()
            .ok()
            .and_then(|f| f.fit.edf_total())
            .unwrap_or(f64::NAN);
        // Double penalty ships [primary(bend), DoublePenaltyNullspace(ridge)] in
        // that order; lambdas align with the canonical penalty list.
        let lam = fit_on.fit.lambdas.to_vec();
        let lambda_bend = lam.first().copied().unwrap_or(f64::NAN);
        let lambda_null = lam.get(1).copied().unwrap_or(f64::NAN);
        // Per-block EDF: with one smooth term the double penalty's two blocks
        // (range/bend and null/ridge) are reported as edf_by_block. The split
        // says whether the inflation is the RANGE (wiggle) or NULL block.
        let edf_blocks = fit_on.fit.edf_by_block();
        let edf_block_str = edf_blocks
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join(",");

        eprintln!(
            "[1266-repro] {seed:>4}  {edf_on:>10.4}  {edf_off:>10.4}  {lambda_bend:>14.4e}  \
             {lambda_null:>14.4e}  {:>12.4}  edf_by_block=[{edf_block_str}]",
            fit_on.fit.reml_score
        );
        on_vals.push(edf_on);
        off_vals.push(edf_off);
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    eprintln!(
        "[1266-repro] MEAN edf_ON={:.4}  edf_OFF={:.4}  (target ≈ mgcv 2.10; contract edf_ON≤edf_OFF)",
        mean(&on_vals),
        mean(&off_vals)
    );
    eprintln!(
        "[1266-repro] INTERPRETATION: if lambda_bend is SMALL at the double-penalty optimum, the \
         wiggle (range) block is under-penalized — gam's λ-selection collapsed λ_bend (the bug). \
         mgcv keeps λ_bend large. Report-only; the fix lands against the mgcv EDF≈2 target."
    );

    // No hard assertion on the inflation: this is the localization probe. The
    // committed CONTRACT test owns the gate; this prints the per-block λ's that
    // say WHY it fails. Only require the load-bearing double-penalty arm to
    // converge (the single-penalty arm is allowed to be a non-fatal NaN if it
    // hits a pre-fit rank deficiency on this knot count).
    assert!(
        on_vals.iter().all(|v| v.is_finite()),
        "double-penalty fits must converge to finite EDF"
    );
}

/// Noise-only DGP: `y = N(0, 0.3)` and an irrelevant uniform covariate `z`.
/// The truth has NO dependence on `z`, so an `s(z)` smooth is unsupported and a
/// correct double penalty must shrink it toward `EDF → 0` — and certainly never
/// ABOVE the single-penalty EDF.
fn irrelevant_dgp(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
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
        let z = next(); // irrelevant covariate
        let u1 = next().max(1e-12);
        let u2 = next();
        let g = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        data[[i, 0]] = z;
        y[i] = 0.3 * g; // pure noise — no dependence on z
    }
    (data, y)
}

/// #1266 CONTRACT GATE. Enabling the Marra–Wood double penalty can only ADD
/// shrinkage capacity, so an UNSUPPORTED smooth's effective d.o.f. under the
/// double penalty must never EXCEED its single-penalty d.o.f.:
/// `mean(EDF_on) ≤ mean(EDF_off)`.
///
/// The bug: the default `Normal(0, sd=3)` ρ-prior (the prior the gamfit formula
/// / orchestration path resolves to via `canonical_standard_fit_options`) caps
/// each log-λ. For a double-penalty term that cap pulls BOTH the wiggliness and
/// null-space log-λ back toward 0, so REML settles at a point that leaves the
/// term under-shrunk — its EDF lands ABOVE the single-penalty EDF (the #1266
/// inflation, ~4.6 vs ~2.0 on this DGP). The fix lifts the cap off
/// Gaussian-identity B-spline double-penalty selection coordinates (flat prior,
/// matching mgcv `select=TRUE`), restoring the contract.
///
/// Runs the same `Normal(0, sd=3)` prior the formula path uses (NOT the flat
/// prior of the localisation probe above), so it actually exercises the
/// regression — a flat-prior fit would pass even without the fix.
#[test]
fn double_penalty_does_not_inflate_unsupported_edf_1266() {
    let n = 800usize;
    let likelihood = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );
    let opts = FitOptions {
        rho_prior: RhoPrior::Normal { mean: 0.0, sd: 3.0 },
        penalty_shrinkage_floor: Some(1e-6),
        ..fit_options()
    };
    let mut on_vals = Vec::new();
    let mut off_vals = Vec::new();
    for seed in 100..105u64 {
        let (data, y) = irrelevant_dgp(n, seed);
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);
        let edf = |double_penalty: bool| -> f64 {
            fit_term_collection_forspec(
                data.view(),
                y.view(),
                weights.view(),
                offset.view(),
                &bspline_spec(double_penalty),
                likelihood.clone(),
                &opts,
            )
            .expect("fit")
            .fit
            .edf_total()
            .expect("edf")
        };
        on_vals.push(edf(true));
        off_vals.push(edf(false));
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (on, off) = (mean(&on_vals), mean(&off_vals));
    assert!(
        on <= off + 1e-8,
        "double penalty must not inflate EDF on an unsupported smooth: \
         mean(EDF_on)={on:.4} > mean(EDF_off)={off:.4} (on={on_vals:?}, off={off_vals:?})"
    );
}
