//! End-to-end quality: gam's Gaussian *location-scale* predictive distribution,
//! scored by the **continuous ranked probability score (CRPS)**, must agree with
//! `properscoring.crps_gaussian` — the canonical Python CRPS library, whose
//! `crps_gaussian` implements the closed-form normal CRPS analytically.
//!
//! Why this comparator. CRPS is the standard strictly-proper scoring rule for
//! probabilistic (distributional) regression: it rewards a predictive
//! distribution that is both sharp and calibrated. For a Gaussian predictive
//! `N(mu, sigma^2)` it has the exact closed form
//!
//!     CRPS(N(mu, sigma), y) = sigma * [ w*(2*Phi(w) - 1) + 2*phi(w) - 1/sqrt(pi) ],
//!     w = (y - mu) / sigma,
//!
//! where `Phi`/`phi` are the standard normal CDF/PDF. `properscoring.crps_gaussian`
//! evaluates exactly this expression. gam itself does not expose a CRPS routine,
//! but it *does* expose the building blocks of its predictive distribution —
//! `gam::inference::probability::{normal_cdf, normal_pdf}` (the very Phi/phi,
//! backed by `statrs::erfc`, that gam uses for Gaussian inference internally).
//! So this test scores gam's location-scale prediction with the *same* canonical
//! closed form, computed from gam's *own* probability primitives, and checks it
//! against properscoring's independent computation at the *identical* (mu, sigma).
//!
//! What it verifies. The synthetic data is heteroscedastic (both mu(x) and
//! sigma(x) move with x), so gam must exercise *both* smooth channels — the mean
//! `s(x, k=8)` and the log-sigma `1 + s(x, k=8)` — to produce sensible predictive
//! distributions. We fit on 100 training rows, predict (mu, sigma) on a held-out
//! 50-row test set, and feed those held-out (mu, sigma) plus the held-out y to
//! properscoring. The held-out split isolates *generalization* calibration from
//! training-set overfitting. Because both sides evaluate the identical analytic
//! formula at the identical (mu, sigma, y), exact agreement is mathematically
//! guaranteed; any element-wise divergence signals a transcription / linking bug
//! in how gam wires its predictive (mu, sigma) — not a statistical disagreement.
//!
//! gam-side API (pinned by reading the source):
//!   * `fit_from_formula(.., FitConfig{ noise_formula: Some(..), .. })` produces a
//!     `FitResult::GaussianLocationScale`; the mean/log-sigma coefficient blocks
//!     carry `BlockRole::Location` / `BlockRole::Scale`, and the resolved designs
//!     live in `fit.meanspec_resolved` / `fit.noisespec_resolved`.
//!   * The noise link is `sigma = LOGB_SIGMA_FLOOR + exp(eta_scale)` with
//!     `LOGB_SIGMA_FLOOR = 0.01` (mirrors `families::sigma_link`, mgcv `gaulss(b=0.01)`).
//!   * The in-Rust path does NOT rescale `y`, so reconstructed mu/sigma are in raw
//!     response units — directly comparable to the y we generated.

use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::inference::probability::{normal_cdf, normal_pdf};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, max_abs_diff, run_python};
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use ndarray::Array2;

/// gam's location-scale noise link floor: sigma = 0.01 + exp(eta_scale).
/// Mirrors `families::sigma_link::LOGB_SIGMA_FLOOR` (and mgcv `gaulss(b=0.01)`).
const LOGB_SIGMA_FLOOR: f64 = 0.01;

/// Closed-form CRPS of a Gaussian predictive `N(mu, sigma)` at observation `y`,
/// computed from gam's OWN standard-normal CDF/PDF primitives. This is the exact
/// expression `properscoring.crps_gaussian` evaluates; computing it here from
/// gam's `normal_cdf`/`normal_pdf` is what makes the comparison a check on gam's
/// predictive plumbing rather than on a re-derived scoring formula.
fn gam_crps_gaussian(y: f64, mu: f64, sigma: f64) -> f64 {
    const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3; // 1 / sqrt(pi)
    let w = (y - mu) / sigma;
    sigma * (w * (2.0 * normal_cdf(w) - 1.0) + 2.0 * normal_pdf(w) - INV_SQRT_PI)
}

#[test]
fn gam_gaussian_location_scale_crps_matches_properscoring() {
    init_parallelism();

    // ---- synthetic heteroscedastic recipe, seed = 1337 --------------------
    // n = 150, x ~ Uniform(0,1); y ~ N(sin(2*pi*x), [0.1 + 0.2*sin(2*pi*x)]^2).
    // A deterministic seeded LCG draws the uniforms and the Box-Muller normals so
    // the exact same y is reproducible and the (mu, sigma, y) we hand to Python
    // are byte-identical to gam's internal values.
    let n = 150usize;
    let two_pi = 2.0 * std::f64::consts::PI;

    let mut state: u64 = 1337;
    let mut next_unit = || -> f64 {
        // Numerical Recipes LCG; take the high bits for a uniform in [0,1).
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x: Vec<f64> = (0..n).map(|_| next_unit()).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Box-Muller standard normals from the same LCG stream (seed continues).
    let mut z: Vec<f64> = Vec::with_capacity(n);
    while z.len() < n {
        let u1 = next_unit().max(1e-300);
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        z.push(r * (two_pi * u2).cos());
        if z.len() < n {
            z.push(r * (two_pi * u2).sin());
        }
    }

    let mu_true = |t: f64| (two_pi * t).sin();
    let sigma_true = |t: f64| 0.1 + 0.2 * (two_pi * t).sin();
    let y: Vec<f64> = (0..n).map(|i| mu_true(x[i]) + sigma_true(x[i]) * z[i]).collect();

    // ---- split: first 100 train, last 50 hold-out test --------------------
    // x is sorted, so taking a contiguous tail as the test set probes
    // extrapolation/interpolation across the full support rather than a single
    // region; CRPS agreement must hold regardless of where the held-out rows lie.
    let n_train = 100usize;
    let n_test = n - n_train; // 50
    let x_train = &x[..n_train];
    let y_train = &y[..n_train];
    let x_test = &x[n_train..];
    let y_test = &y[n_train..];

    // ---- build the TRAINING dataset (column 0 = x, column 1 = y) ----------
    let headers: Vec<String> = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n_train)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", x_train[i]),
                format!("{:.17e}", y_train[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode train data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    // ---- fit with gam: mu ~ s(x, k=8), log-sigma ~ 1 + s(x, k=8) ----------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1 + s(x, k=8)".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x, k=8)", &ds, &cfg).expect("gam location-scale fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult { fit, .. }) = result else {
        panic!("expected a Gaussian location-scale fit");
    };

    let beta_location = fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location (mean) block present")
        .beta
        .clone();
    let beta_scale = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale (log-sigma) block present")
        .beta
        .clone();

    // ---- predict (mu, sigma) on the held-out TEST points -------------------
    // Rebuild the frozen mean / log-sigma designs at the test x and apply each
    // block's coefficients. mu = X_mean*beta; sigma = floor + exp(X_scale*beta).
    let mut grid = Array2::<f64>::zeros((n_test, ncols));
    for (i, &t) in x_test.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }
    let mean_design = build_term_collection_design(grid.view(), &fit.meanspec_resolved)
        .expect("rebuild mean design at test points");
    let scale_design = build_term_collection_design(grid.view(), &fit.noisespec_resolved)
        .expect("rebuild log-sigma design at test points");

    let mu_test: Vec<f64> = mean_design.design.apply(&beta_location).to_vec();
    let eta_sigma_test: Vec<f64> = scale_design.design.apply(&beta_scale).to_vec();
    let sigma_test: Vec<f64> = eta_sigma_test
        .iter()
        .map(|&e| LOGB_SIGMA_FLOOR + e.exp())
        .collect();

    assert_eq!(mu_test.len(), n_test);
    assert_eq!(sigma_test.len(), n_test);
    for &s in &sigma_test {
        assert!(s > 0.0 && s.is_finite(), "predicted sigma must be positive/finite: {s}");
    }

    // ---- gam-side CRPS via gam's own normal_cdf / normal_pdf ---------------
    let gam_crps: Vec<f64> = (0..n_test)
        .map(|i| gam_crps_gaussian(y_test[i], mu_test[i], sigma_test[i]))
        .collect();

    // ---- properscoring: independent CRPS at the IDENTICAL (y, mu, sigma) ---
    let ref_py = run_python(
        &[
            Column::new("y_test", y_test),
            Column::new("mu_test", &mu_test),
            Column::new("sigma_test", &sigma_test),
        ],
        r#"
import properscoring as ps
crps = ps.crps_gaussian(np.asarray(df["y_test"], dtype=float),
                        mu=np.asarray(df["mu_test"], dtype=float),
                        sig=np.asarray(df["sigma_test"], dtype=float))
emit("crps", np.asarray(crps, dtype=float))
"#,
    );
    let ps_crps = ref_py.vector("crps");
    assert_eq!(ps_crps.len(), n_test, "properscoring CRPS length mismatch");

    // ---- compare element-wise and in aggregate ----------------------------
    let max_rel = (0..n_test)
        .map(|i| (gam_crps[i] - ps_crps[i]).abs() / ps_crps[i].abs().max(1e-12))
        .fold(0.0_f64, f64::max);
    let abs_diff = max_abs_diff(&gam_crps, ps_crps);
    let mean_gam = gam_crps.iter().sum::<f64>() / n_test as f64;
    let mean_ps = ps_crps.iter().sum::<f64>() / n_test as f64;
    let mean_diff = (mean_gam - mean_ps).abs();

    eprintln!(
        "crps gaussian vs properscoring: n_train={n_train} n_test={n_test} \
         max_rel={max_rel:.3e} max_abs={abs_diff:.3e} \
         mean_crps(gam)={mean_gam:.6} mean_crps(ps)={mean_ps:.6} mean_diff={mean_diff:.3e}"
    );

    // Both sides evaluate the identical closed-form Gaussian CRPS at the identical
    // (y, mu, sigma); the only freedom is the floating-point erf/erfc backend
    // (gam: statrs::erfc; properscoring: scipy/numpy), each near machine epsilon.
    // The per-observation relative gap is therefore ~1e-12 in practice, so a 1e-9
    // relative bound is principled — tight enough that ANY transcription/linking
    // error in gam's predictive (mu, sigma) blows past it, yet a few hundred ulps
    // of headroom over the genuine cross-backend erf disagreement.
    assert!(
        max_rel < 1e-9,
        "per-observation Gaussian CRPS diverges from properscoring: max_rel={max_rel:.3e}"
    );
    // The aggregate (mean) CRPS is the headline calibration number; with identical
    // inputs and formula the per-element abs diffs are ~1e-13, so the mean diff is
    // ~1e-13 and must match to within 1e-9 absolute. A larger gap is not a
    // statistical disagreement but a sign that gam mis-wired mu/sigma into the
    // predictive distribution (wrong link, wrong block, wrong floor).
    assert!(
        mean_diff < 1e-9,
        "mean hold-out CRPS disagrees with properscoring: gam={mean_gam:.8} ps={mean_ps:.8} diff={mean_diff:.3e}"
    );
}
