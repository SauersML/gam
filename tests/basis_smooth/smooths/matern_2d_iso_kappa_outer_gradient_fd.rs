//! #1122 / #901 iso-κ gradient class — the FULL joint-REML objective gradient
//! w.r.t. the Matérn isotropic length-scale coordinate `ψ = log κ` must match a
//! central finite difference of the COMPLETE outer criterion, not just the
//! penalty block.
//!
//! The penalty-only basis test
//! (`basis_matern_double_penalty_log_kappa_derivative_fd`) checks only
//! `∂(penalty matrices)/∂log_κ`. The full profiled REML objective
//!   V(κ) = data-fit(deviance) + ½log|H+Sλ| − ½log|Sλ|₊ + penalty-quad
//! also depends on κ through the Matérn DESIGN columns (and hence the deviance
//! and `H = XᵀWX`) and through the H-side of the `½log|H+Sλ|` term. A stall with
//! a large residual gradient at the iteration cap is the signature of an
//! objective↔gradient DESYNC: the optimizer follows a gradient inconsistent with
//! the objective it is minimizing.
//!
//! The original root cause was κ-dependent numerical null classification
//! combined with an identifiability chart `Z` that was not frozen across
//! per-trial rebuilds. Matérn topology is now structural (only an explicitly
//! appended intercept is a null direction), and `Z` is frozen at the first
//! rebuild and mirrored onto the spec read by the analytic ψ-gradient. Value
//! and gradient therefore share one fixed chart and topology at every trial.
//!
//! The generic outer runner exposes an opt-in structured finite-difference
//! record at its real bounded seed. This test fits an ordinary Gaussian 2-D
//! surface with a single `matern(x1, x2)` smooth (default double-penalty), then
//! asserts that the typed Matérn log-κ component matches the finite difference
//! of the full criterion.
//!
//! Reference-as-truth: every assertion is against the analytic FD of gam's own
//! profiled REML criterion — never another tool's output.

use gam::{FitConfig, encode_recordswith_inferred_schema};

fn init() {
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);
    gam::init_parallelism();
}

use gam::utils::splitmix64;
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn truth(a: f64, b: f64) -> f64 {
    (2.0 * std::f64::consts::PI * a).sin() * (2.0 * std::f64::consts::PI * b).sin()
}

fn build_dataset(n: usize, sigma: f64, seed: u64) -> gam::inference::data::EncodedDataset {
    let mut st = seed;
    let mut header = String::from("y,x1,x2\n");
    let mut body = String::new();
    for _ in 0..n {
        let a = next_unit(&mut st);
        let b = next_unit(&mut st);
        let y = truth(a, b) + sigma * next_gauss(&mut st);
        body.push_str(&format!("{y:.6},{a:.6},{b:.6}\n"));
    }
    header.push_str(&body);
    let mut rdr = csv::ReaderBuilder::new().from_reader(header.as_bytes());
    let records: Vec<csv::StringRecord> = rdr.records().map(|r| r.unwrap()).collect();
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    encode_recordswith_inferred_schema(headers, records).expect("encode dataset")
}

/// #1270 regression: a single `matern(x1, x2)` 2-D smooth must fit an ordinary
/// Gaussian surface to convergence over the FULL (uncapped) κ-optimizer loop,
/// exactly like the `duchon` control below.
///
/// Root cause: the single-spatial-term "n-free penalty re-key" fast path
/// declared itself supported for Matérn, so the design-revision skip path was
/// taken. But the realized Matérn design carries the collocation operator
/// triplet (mass/tension/stiffness, #1259) while the n-free re-key rebuilds the
/// projected-kernel double-penalty — a different block topology. The block-count
/// guard rejected the rebuild, cleared the staged surface, and the next
/// skip-path eval converted "no exact S(ψ) staged" into a HARD ERROR
/// (IntegrationError), aborting the fit. `duchon`/`thinplate`/`te` were
/// unaffected because their re-key reproduces the frozen topology exactly.
///
/// The fix drops `Matern` from `supports_nfree_penalty_rekey`, routing it
/// through the slow path that re-realizes the design every trial (re-deriving
/// the correct operator triplet). This test caps NOTHING on the outer loop, so
/// it reaches the skip-window evals that armed the bug; pre-fix it aborts with
/// IntegrationError, post-fix it converges.
#[test]
fn matern_2d_smooth_fits_ordinary_surface_full_outer_loop() {
    init();
    let data = build_dataset(160, 0.05, 0x1270_0001_2D5Eu64);
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        // No outer_max_iter cap: run the FULL κ loop so the design-revision
        // skip path (the bug's trigger) is actually reached.
        gpu_policy: if cfg!(target_os = "macos") {
            gam::gpu::GpuPolicy::Off
        } else {
            gam::gpu::GpuPolicy::Auto
        },
        ..FitConfig::default()
    };

    // matern: must fit without the IntegrationError abort (#1270).
    let matern = gam::fit_from_formula("y ~ matern(x1, x2)", &data, &config);
    assert!(
        matern.is_ok(),
        "matern(x1,x2) 2-D smooth must fit an ordinary surface, but the fit \
         returned an error (#1270 regression): {:?}",
        matern.err()
    );

    // duchon control: the sibling spatial smooth that was always healthy.
    let duchon = gam::fit_from_formula("y ~ duchon(x1, x2)", &data, &config);
    assert!(
        duchon.is_ok(),
        "duchon(x1,x2) control must fit (it always did): {:?}",
        duchon.err()
    );
}
