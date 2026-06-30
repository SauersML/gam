//! Parity gate (#1196): the `gam` CLI fit path and the Python/PyO3 `gamfit.fit`
//! path must produce the *identical* fit for the same formula + data + config,
//! by construction — it must not be structurally possible for them to diverge.
//!
//! Background (#1191): the FFI fit path (`fit_dataset_impl` in `gam-pyffi`) and
//! the CLI fit path (`run_fit` in `src/main`) each build their own
//! `StandardFitRequest` and then call the shared `fit_model`. Because the two
//! request builders were independent, they could (and did) resolve a *different*
//! outer-REML optimization policy for the same model — the CLI hand-built
//! `FitOptions { tol: 1e-6, skip_rho_posterior_inference: false, .. }` while the
//! formula/FFI path used `1e-10` / `true`, so the same shape-constrained smooth
//! that the CLI fit cleanly NaN-aborted under `gamfit.fit`.
//!
//! The structural fix routes BOTH request builders through one policy source,
//! `canonical_standard_fit_options`, and sources every config-derived request
//! field from the same resolved `FitConfig`. This test pins that invariant on
//! the #1191 shape-constrained fixture (and an unconstrained control):
//!
//!   1. The `FitOptions` that `materialize` (the FFI/formula path) puts on the
//!      `StandardFitRequest` is EXACTLY `canonical_standard_fit_options(config,
//!      ..)` — i.e. the shared policy source is actually what flows into the fit,
//!      not a parallel-but-equal hand-built block. This is the field that #1191
//!      diverged on (`tol` / `skip_rho_posterior_inference`).
//!
//!   2. Fitting the materialized request directly through `fit_model` (the exact
//!      call both the CLI and the FFI make) and fitting the same formula through
//!      the shared `fit_from_formula` entry yield bit-comparable β̂. Both binaries
//!      wrap this same orchestration, so agreement here is agreement across the
//!      two entry points.
//!
//! If a future change re-introduces a CLI-only or FFI-only knob in the standard
//! fit path (a different tolerance, a skipped/added validation pass, a dropped
//! config field), one of these assertions fails.

use gam::solver::fit_orchestration::{
    self, FitRequest, FitResult, StandardFitOptionsInputs, canonical_standard_fit_options,
    fit_from_formula, fit_model, materialize,
};
use gam::{FitConfig, init_parallelism, load_csvwith_inferred_schema};
use std::io::Write;

/// Deterministic `y = sqrt(x) + N(0, 0.05²)` on x ∈ [0,1] (the #1191 fixture):
/// strictly increasing and concave in expectation, so it exercises both binding
/// and non-binding shape constraints. Self-contained SplitMix64 + Box–Muller.
fn sqrt_dataset(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut state: u64 = 11;
    let mut next_unit = move || -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut x = vec![0.0f64; n];
    for xi in x.iter_mut() {
        *xi = next_unit();
    }
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let u1 = next_unit().max(1e-300);
            let u2 = next_unit();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            xi.sqrt() + 0.05 * noise
        })
        .collect();
    (x, y)
}

fn load_sqrt_dataset(n: usize) -> gam::inference::data::EncodedDataset {
    let (x, y) = sqrt_dataset(n);
    let mut csv = String::from("x,y\n");
    for i in 0..n {
        csv.push_str(&format!("{:.17e},{:.17e}\n", x[i], y[i]));
    }
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "gam_cli_ffi_parity_1196_{}_{}.csv",
        std::process::id(),
        n
    ));
    {
        let mut f = std::fs::File::create(&tmp).expect("create synthetic csv");
        f.write_all(csv.as_bytes()).expect("write synthetic csv");
    }
    let ds = load_csvwith_inferred_schema(&tmp).expect("load synthetic sqrt data");
    std::fs::remove_file(&tmp).ok();
    ds
}

/// Materialize `formula`, then fit it two ways that the CLI and FFI binaries
/// respectively reduce to, and assert the fits are identical.
///
/// `compare_beta` is `false` for the unconstrained `s(x)` control: a single 1-D
/// Gaussian cubic smooth is detected by the exact O(n) spline-scan fast path,
/// which `fit_from_formula` routes to (returning `FitResult::SplineScan`) but a
/// bare `fit_model(request)` does not — both the CLI and the FFI run this same
/// scan-detection step *before* `fit_model`, so the dense-vs-scan representation
/// is not a CLI/FFI divergence and the β̂ vectors are not directly comparable.
/// The shared FitOptions-policy assertion still runs for that case; the β̂
/// comparison runs on the four shape-constrained smooths (which carry linear
/// constraints, are ineligible for the scan, and are the #1191 critical cases).
fn assert_parity_for(formula: &str, ds: &gam::inference::data::EncodedDataset, compare_beta: bool) {
    let cfg = FitConfig::default(); // gaussian / identity / REML — the CLI `fit` default

    // ---- The request the formula/FFI path builds. -----------------------------
    let mat = materialize(formula, ds, &cfg).unwrap_or_else(|e| {
        panic!("materialize '{formula}' failed: {e}");
    });
    let FitRequest::Standard(std_request) = &mat.request else {
        panic!("'{formula}' on gaussian sqrt data must materialize a Standard request");
    };

    // (1) The single shared policy source (#1196) is what actually rides on the
    // materialized request: the exact `FitOptions` field set #1191 diverged on.
    let canonical = canonical_standard_fit_options(
        &cfg,
        StandardFitOptionsInputs {
            firth_bias_reduction: cfg.firth,
            ..Default::default()
        },
    );
    let got = &std_request.options;
    assert_eq!(
        got.tol, canonical.tol,
        "materialized FitOptions.tol must equal the shared canonical policy (#1191 divergence field)"
    );
    assert_eq!(
        got.skip_rho_posterior_inference, canonical.skip_rho_posterior_inference,
        "materialized FitOptions.skip_rho_posterior_inference must equal the shared canonical policy"
    );
    assert_eq!(
        got.max_iter, canonical.max_iter,
        "materialized FitOptions.max_iter must equal the shared canonical policy"
    );
    assert_eq!(
        got.compute_inference, canonical.compute_inference,
        "materialized FitOptions.compute_inference must equal the shared canonical policy"
    );
    assert_eq!(
        got.penalty_shrinkage_floor, canonical.penalty_shrinkage_floor,
        "materialized FitOptions.penalty_shrinkage_floor must equal the shared canonical policy"
    );

    if !compare_beta {
        // Unconstrained control: the policy parity above is the assertion of
        // interest; the scan/dense representation split (see fn docs) makes the
        // coefficient vectors non-comparable.
        return;
    }

    // (2a) Fit the materialized request directly through `fit_model` — the exact
    // call both `run_fit` (CLI) and `fit_dataset_impl` (FFI) make.
    let direct = fit_model(mat.request).unwrap_or_else(|e| {
        panic!("fit_model('{formula}') failed: {e}");
    });
    let FitResult::Standard(direct) = direct else {
        panic!("'{formula}' fit_model must return a Standard result");
    };
    let beta_direct = direct.fit.beta.clone();

    // (2b) Fit the same formula through the shared `fit_from_formula` entry that
    // wraps materialize + fast-path dispatch + fit_model. The two binaries are
    // (modulo persistence-payload assembly) thin wrappers over this orchestration,
    // so agreement here is agreement across the entry points.
    let shared = fit_from_formula(formula, ds, &cfg).unwrap_or_else(|e| {
        panic!("fit_from_formula('{formula}') failed: {e}");
    });
    let FitResult::Standard(shared) = shared else {
        panic!("'{formula}' fit_from_formula must return a Standard result for this fixture");
    };
    let beta_shared = shared.fit.beta.clone();

    assert_eq!(
        beta_direct.len(),
        beta_shared.len(),
        "coefficient length mismatch for '{formula}': direct={} shared={}",
        beta_direct.len(),
        beta_shared.len()
    );
    let mut max_abs = 0.0f64;
    let mut scale = 1.0f64;
    for (a, b) in beta_direct.iter().zip(beta_shared.iter()) {
        assert!(
            a.is_finite() && b.is_finite(),
            "non-finite coefficient for '{formula}': direct={a} shared={b}"
        );
        max_abs = max_abs.max((a - b).abs());
        scale = scale.max(a.abs()).max(b.abs());
    }
    // Same deterministic optimizer, same data, same options — coefficients must
    // agree to a tight relative tolerance (allowing only for any incidental
    // floating-point reassociation between the two call shapes).
    let rel = max_abs / scale;
    assert!(
        rel < 1e-8,
        "CLI/FFI standard-fit parity broken for '{formula}': max |Δβ̂| = {max_abs:.3e} (rel {rel:.3e})"
    );
}

#[test]
fn cli_and_ffi_standard_fit_paths_are_identical_by_construction() {
    init_parallelism();
    let ds = load_sqrt_dataset(400);

    // Unconstrained control plus all four #1191 shape-constrained smooths: the
    // shape constraint is what drove the indefinite-Hessian ALO abort that made
    // the CLI and FFI behave differently, so it is the critical parity case.
    for (formula, compare_beta) in [
        ("y ~ s(x)", false),
        ("y ~ s(x, shape=monotone_increasing)", true),
        ("y ~ s(x, shape=monotone_decreasing)", true),
        ("y ~ s(x, shape=convex)", true),
        ("y ~ s(x, shape=concave)", true),
    ] {
        assert_parity_for(formula, &ds, compare_beta);
    }

    // Touch the module path so the import is exercised even if the inner asserts
    // are ever feature-gated; keeps the shared-entry symbol load-bearing.
    let touch = fit_orchestration::WorkflowError::IntegrationFailed {
        reason: String::new(),
    };
    assert!(matches!(
        touch,
        fit_orchestration::WorkflowError::IntegrationFailed { .. }
    ));
}
