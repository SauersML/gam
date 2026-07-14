//! Regression for #500: `s(x, bc=clamped)` must not ABORT the whole REML
//! fit on a zero-offset sine.
//!
//! The clamped endpoint-derivative equality is split into two opposing
//! inequalities (`row·β ≥ 0` and `−row·β ≥ 0`), so the constraint KKT
//! primal-feasibility residual is exactly `|row·β|` — the magnitude of the
//! smooth's first derivative at the endpoint. Those B-spline derivative
//! rows carry a large norm (‖a_i‖ ≫ 1), so the diagnostic measured a RAW
//! slack inflated by ‖a_i‖: a geometrically-feasible iterate (distance
//! ~2e-8 to the hyperplane) reported primal ≈ 2e-5 and was refused against
//! the 1e-7 gate. Every candidate seed was disqualified by this spurious
//! pre-warm refusal ("no candidate seeds passed outer startup validation").
//! The fix (active_set.rs) measures primal feasibility in per-row-scaled
//! (geometric) coordinates, consistent with the solver's contract, so the
//! refusal no longer fires and the fit proceeds.
//!
//! SCOPE: this test guards two things, both of which the #500 reopen confirmed
//! are now genuinely fixed (the original close left the SECOND one unguarded):
//!   1. the ABORT regression — the clamped fit must COMPLETE, not refuse every
//!      seed during outer startup validation; and
//!   2. interior fit QUALITY — the returned clamped smooth must actually track
//!      the signal, not collapse to a near-constant.
//!
//! When this issue was first closed, only (1) was fixed and (2) was explicitly
//! deferred with the claim that "REML over-smooths to the λ-ceiling whenever a
//! linear inequality constraint is active, so the returned clamped fit is
//! near-constant (interior RMSE ≈ 0.77 vs ≈ 0.02 unconstrained)". That claim
//! is no longer true: `bc=clamped` no longer lowers to a runtime active-set
//! inequality at all — since #1238 it is a *structural nullspace
//! reparameterization* (the pinned endpoint-derivative DOF is removed from the
//! basis before the solver sees it), so the smooth fits on the ordinary
//! unconstrained REML path and recovers the interior at ≈ unconstrained
//! quality. The deferred degeneracy was fixed but left UNGUARDED — a future
//! regression in the constrained-chart penalty geometry would silently
//! re-collapse the clamped fit with nothing red. This test now pins the
//! quality so the fix cannot rot. (The sibling `bc_fit_quality_sanity.rs`
//! pins the same contract on the `+0.3`-offset dataset.)

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn truth(x: f64) -> f64 {
    // Zero vertical offset — the only difference from the passing
    // `bc_fit_quality_sanity.rs` dataset, and the trigger for #500.
    (2.0 * std::f64::consts::PI * x).sin()
}

fn make_data(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|t| truth(*t) + noise.sample(&mut rng))
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_and_probe(formula: &str, data: &gam::data::EncodedDataset, eval_xs: &[f64]) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg)
        .unwrap_or_else(|e| panic!("fit failed for `{formula}`: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit for `{formula}`")
    };
    let n = eval_xs.len();
    let mut m = Array2::<f64>::zeros((n, 2));
    for (i, &x) in eval_xs.iter().enumerate() {
        m[[i, 0]] = x;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .unwrap_or_else(|e| panic!("predict design failed for `{formula}`: {e:?}"));
    design.design.apply(&fit.fit.beta).to_vec()
}

fn rmse(pred: &[f64], eval_xs: &[f64]) -> f64 {
    let mut sumsq = 0.0_f64;
    for (p, &x) in pred.iter().zip(eval_xs.iter()) {
        let d = p - truth(x);
        sumsq += d * d;
    }
    (sumsq / pred.len() as f64).sqrt()
}

#[test]
fn clamped_zero_offset_sine_does_not_abort_startup_validation() {
    init_parallelism();
    let data = make_data(300, 0.1, 7);
    let eval_xs: Vec<f64> = (0..101).map(|i| 0.10 + 0.80 * (i as f64) / 100.0).collect();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    // The #500 contract: the clamped fit must COMPLETE — not abort during
    // outer startup validation. Before the fix this returned
    // `Err(RemlOptimizationFailed("... no candidate seeds passed outer
    // startup validation ..."))` because every seed was refused over a
    // spuriously-inflated raw primal-KKT residual.
    let result = fit_from_formula("y ~ s(x, k=12, bc=clamped)", &data, &cfg);
    let result = result.unwrap_or_else(|e| {
        panic!(
            "#500 regression: clamped fit aborted instead of completing: {e}\n\
             (the abort signature is \"no candidate seeds passed outer startup \
             validation\"; the fix lives in active_set.rs's per-row-scaled \
             primal feasibility)"
        )
    });
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard fit for the clamped model");
    };

    // The returned fit must be usable: finite coefficients and finite
    // predictions across the interior probe grid.
    assert!(
        fit.fit.beta.iter().all(|b| b.is_finite()),
        "clamped fit returned non-finite coefficients"
    );
    let pred_clamped = fit_and_probe("y ~ s(x, k=12, bc=clamped)", &data, &eval_xs);
    assert_eq!(pred_clamped.len(), eval_xs.len());
    assert!(
        pred_clamped.iter().all(|p| p.is_finite()),
        "clamped fit produced non-finite predictions"
    );

    // QUALITY GATE (the part the original close left unguarded): the clamped
    // smooth must recover the interior signal at ≈ unconstrained quality, NOT
    // collapse to the λ-ceiling near-constant the original triage feared.
    // Since #1238 `bc=clamped` is a structural nullspace reparameterization —
    // the pinned endpoint-derivative DOF is removed from the basis before the
    // solver runs — so the fit takes the ordinary unconstrained REML path and
    // the interior RMSE tracks the free fit within a small multiple.
    //
    // Concrete numbers at this seed/n: free ≈ 0.018, clamped ≈ 0.022. We assert
    // a generous bar (clamped ≤ 5× free, and an absolute ceiling of 0.10) so the
    // test is robust to platform FP drift yet still RED if the fit ever
    // re-collapses toward the ≈0.77 near-constant degeneracy.
    let rmse_free = rmse(&fit_and_probe("y ~ s(x, k=12)", &data, &eval_xs), &eval_xs);
    let rmse_clamped = rmse(&pred_clamped, &eval_xs);
    eprintln!("[bc-500] free={rmse_free:.4} clamped={rmse_clamped:.4}");
    assert!(
        rmse_clamped <= 0.10,
        "#500 quality regression: clamped interior RMSE={rmse_clamped:.4} exceeds the 0.10 \
         absolute ceiling — the structural bc=clamped reparameterization appears to have \
         re-collapsed toward the old λ-ceiling near-constant (≈0.77) degeneracy"
    );
    assert!(
        rmse_clamped <= 5.0 * rmse_free,
        "#500 quality regression: clamped RMSE={rmse_clamped:.4} is more than 5× the \
         unconstrained reference RMSE={rmse_free:.4}; the clamped fit is no longer tracking \
         the interior signal at unconstrained quality"
    );
}
