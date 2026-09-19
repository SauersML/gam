//! #1464 regression: the constant-curvature `curv(...)` smooth, fitted through
//! the FULL `fit_from_formula` pipeline, must IDENTIFY THE SIGN of the true
//! curvature — a positive κ̂ for genuinely spherical data and a negative κ̂ for
//! genuinely hyperbolic data — instead of railing κ̂ to the positive chart bound
//! for every dataset (the reported bug: hyperbolic truth recovered as spherical,
//! with the SAME κ̂ returned for the mirror spherical/hyperbolic datasets).
//!
//! The evidence is a sign-symmetry argument that needs no absolute scale: two
//! mirror datasets, one spherical (κ⋆ = +2) and one hyperbolic (κ⋆ = −2), each
//! a member of the fitted term's OWN span at its κ⋆ (see `curved_dataset`), so
//! the planted signal's geometry is gam's own truth, never another tool's
//! output. A correct estimator MUST distinguish them.
//!
//! κ̂ is read back from the FITTED resolved term spec (the same κ̂ that
//! `model.curvature()` surfaces), so this drives exactly the user-visible
//! full-fit path the issue reports on.

use gam::inference::formula_dsl::parse_formula;
use gam::smooth::{SmoothBasisSpec, get_constant_curvature_kappa};
use gam::terms::basis::build_constant_curvature_basis;
use gam::terms::term_builder::build_termspec;
use ndarray::{Array1, Array2};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

use csv::StringRecord;

/// Forwards the κ route's own trace records (`[#1464-trace]`, `[spatial-kappa]`)
/// to stderr, so a run names the route that produced κ̂ whichever way the
/// assertions go.
struct KappaTraceLogger;

impl log::Log for KappaTraceLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record<'_>) {
        let message = record.args().to_string();
        if message.starts_with("[#1464-trace]") || message.starts_with("[spatial-kappa]") {
            eprintln!("{message}");
        }
    }
    fn flush(&self) {}
}

static KAPPA_TRACE_LOGGER: KappaTraceLogger = KappaTraceLogger;

fn install_kappa_trace_logger() {
    // Losing the race to an already-installed logger leaves that logger in place.
    if log::set_logger(&KAPPA_TRACE_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
}

// --- deterministic RNG (splitmix64 → unit / gaussian), no external deps -------
use gam::utils::splitmix64;
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// The one formula the plant and the fit share, so the planted span is the
/// span the fit estimates.
const FIT_FORMULA: &str = "y ~ curv(x1, x2, centers=10)";

/// `n` chart points uniformly in a disk of radius `radius`, with a Gaussian
/// response that is a member of the κ⋆ span of the fitted `curv(...)` term: the
/// term's own columns at the TRUE κ (its realized centers and auto range),
/// weighted `w_j = 1/(1+j)` and standardized to unit SD, plus noise.
///
/// This fixture used to plant `μ = 2·exp(−d_{κ⋆}(x, 0)) − 1`, a function of
/// the geodesic distance to the chart ORIGIN. That plant is curvature-BLIND as
/// a function class: `d_κ(x, 0)` is a strictly monotone reparametrization of
/// the chart radius at every κ (the argument
/// `constant_curvature_kappa_inference_e2e` gives for its own generator), so
/// the geometry carried no signal and both mirror fits railed at the positive
/// chart bound: κ̂ = +2.1589 (ℓ̂ = 0.954) and +2.1607 (ℓ̂ = 2.6e7) in sw4l job
/// 1244874 at 95115c8a1f. On an in-span plant the full formula pipeline
/// recovers κ⋆ = −2, −0.75, +0.75 and +2 as −1.999, −0.729, +0.801 and +2.001
/// (lane probe 1252018).
fn curved_dataset(kappa_star: f64, seed: u64) -> gam::data::EncodedDataset {
    let radius = 0.68_f64;
    let noise = 0.02_f64;
    let n = 600usize;
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    let mut st = seed;
    let mut points = Array2::<f64>::zeros((n, 2));
    let mut filled = 0usize;
    while filled < n {
        let a = 2.0 * next_unit(&mut st) - 1.0;
        let b = 2.0 * next_unit(&mut st) - 1.0;
        if a * a + b * b > 1.0 {
            continue;
        }
        points[(filled, 0)] = a * radius;
        points[(filled, 1)] = b * radius;
        filled += 1;
    }
    let encode = |y: &Array1<f64>| {
        let records = (0..n)
            .map(|i| {
                StringRecord::from(vec![
                    y[i].to_string(),
                    points[(i, 0)].to_string(),
                    points[(i, 1)].to_string(),
                ])
            })
            .collect();
        encode_recordswith_inferred_schema(headers.clone(), records)
            .expect("encode curved dataset")
    };

    // Resolve the fitted term on the design columns alone (the response does not
    // enter the spec), then build its columns at κ⋆.
    let design_only = encode(&Array1::<f64>::zeros(n));
    let parsed = parse_formula(FIT_FORMULA).expect("the fixture formula parses");
    let col_map = design_only.column_map();
    let mut notes = Vec::new();
    let fitspec = build_termspec(&parsed.terms, &design_only, &col_map, &mut notes)
        .expect("the fixture formula resolves to a term spec");
    let SmoothBasisSpec::ConstantCurvature { spec, .. } = &fitspec.smooth_terms[0].basis else {
        panic!("the fixture formula must resolve to a constant-curvature term");
    };
    let mut truth = spec.clone();
    truth.kappa = kappa_star;
    truth.kappa_fixed = true;
    truth.double_penalty = false;
    let basis = build_constant_curvature_basis(points.view(), &truth)
        .expect("the planted κ⋆ geometry must be inside its own chart");
    let design = basis.design.to_dense();
    let mut y = Array1::<f64>::zeros(n);
    for j in 0..design.ncols() {
        let w = 1.0 / (1.0 + j as f64);
        for i in 0..n {
            y[i] += w * design[(i, j)];
        }
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let sd = (y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
    assert!(
        sd > 0.0,
        "the planted κ⋆ = {kappa_star} signal collapsed to a constant"
    );
    for i in 0..n {
        y[i] = (y[i] - mean) / sd + noise * next_gauss(&mut st);
    }
    encode(&y)
}

/// Fit `curv(x1, x2, centers=10)` through the full formula pipeline and return
/// the fitted curvature κ̂ read back from the resolved term spec.
fn fit_kappa_hat(kappa_star: f64, seed: u64) -> f64 {
    let data = curved_dataset(kappa_star, seed);
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(FIT_FORMULA, &data, &config)
        .expect("curv formula fit should succeed");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian fit");
    };
    // The single smooth term is the constant-curvature term (index 0); κ̂ is the
    // outer-optimized curvature persisted in the resolved spec.
    get_constant_curvature_kappa(&fit.resolvedspec, 0)
        .expect("fitted spec must carry a constant-curvature κ̂")
}

// #1464 production route: before the baseline fit, each free
// constant-curvature coordinate is selected once by the continuously optimized,
// analytically differentiated Gaussian REML profile `V_p(κ; y)`. The subsequent nuisance-ρ/spatial
// solve holds that selected κ fixed, and curvature inference evaluates the same
// likelihood profile for its interval and flatness statistic. This contract therefore
// guards the actual user-visible estimand route rather than a deleted
// basis-local RSS profiler or the raw pinned-fit diagnostic.
//
// It runs unconditionally in CI: re-`#[ignore]`ing it would hide the exact
// hyperbolic-as-spherical regression the issue reports.
#[test]
fn curv_full_fit_identifies_curvature_sign_on_mirror_datasets() {
    init_parallelism();
    install_kappa_trace_logger();

    // Control: genuinely spherical data must recover POSITIVE curvature.
    let kappa_spherical = fit_kappa_hat(2.0, 0x5151_0001);
    // The headline failure: genuinely hyperbolic data must recover NEGATIVE
    // curvature, NOT rail to the positive chart bound.
    let kappa_hyperbolic = fit_kappa_hat(-2.0, 0x5151_0003);

    eprintln!(
        "[#1464] full-fit κ̂: spherical(κ⋆=+2)={kappa_spherical:+.4}  hyperbolic(κ⋆=−2)={kappa_hyperbolic:+.4}"
    );

    // (a) Control — spherical truth recovers positive curvature.
    assert!(
        kappa_spherical > 0.0,
        "spherical truth (κ⋆=+2) must recover POSITIVE curvature through the full fit; got κ̂={kappa_spherical}"
    );

    // (b) The two mirror datasets must be GENUINELY DISTINGUISHED — not the
    // bit-identical κ̂ the bug returns for both signs.
    assert!(
        (kappa_spherical - kappa_hyperbolic).abs() > 0.1,
        "spherical and hyperbolic mirror datasets must yield materially DIFFERENT κ̂ \
         (the #1464 bug returns the same chart-bound value for both): \
         spherical κ̂={kappa_spherical}, hyperbolic κ̂={kappa_hyperbolic}"
    );

    // (c) The headline: hyperbolic truth recovers NEGATIVE curvature.
    assert!(
        kappa_hyperbolic < 0.0,
        "hyperbolic truth (κ⋆=−2) must recover NEGATIVE curvature through the full fit; \
         got κ̂={kappa_hyperbolic} (the bug rails this to the +chart bound, calling it spherical)"
    );
}
