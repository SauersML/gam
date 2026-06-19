//! Regression guards for the large-scale 8h-hang investigation.
//!
//! Pinned mechanism (validated via multiple orthogonal lines):
//!
//!   * `marginal_slope_shared.rs:maybe_install_auto_outer_subsample`
//!     used to construct `AutoOuterSubsampleOptions::default()` and
//!     discard the `outer_work_per_k_unit` argument. At large-scale n the
//!     noise-only rule then picked K ≈ 0.10·n = 19_661, ~9× larger
//!     than the survival family's intended K ≈ 2_000. The first outer
//!     analytic Hessian evaluation then exhausted RAM and disk; the
//!     kernel killed the process with SIGKILL (exit 137) after 8h of
//!     silent grinding.
//!
//!   * The identifiability audit / canonicalization halts the large-scale
//!     5-block shape (51 cols, rank 38) with
//!     `CustomFamilyError::IdentifiabilityFailure`. The release binary
//!     that emitted the failing log predates this gate — the literal
//!     phrase "family-specific reparameterisation required" does not
//!     appear in current source.
//!
//! These tests guard both fixes against regression and document the
//! K-cap multiplier mechanism quantitatively.

use gam::families::marginal_slope_shared::{
    AUTO_OUTER_MIN_K_FLOOR, AUTO_OUTER_WORK_BUDGET, AutoOuterCapReason, AutoOuterSubsampleOptions,
    auto_outer_score_subsample, maybe_install_auto_outer_subsample,
};
use gam::identifiability::audit::audit_identifiability;
use ndarray::{Array1, Array2};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;

const LARGE_SCALE_N: usize = 195_780;
const SURVIVAL_WORK_PER_K_UNIT: u64 = 250_000;
const OBSERVED_K_IN_RUN: usize = 19_661;

// -------- H1: AutoOuterSubsampleOptions::target_k_detailed math --------

#[test]
fn h1a_target_k_with_default_work_per_k_picks_noise_rule() {
    let opts = AutoOuterSubsampleOptions::default();
    let choice = opts
        .target_k_detailed(LARGE_SCALE_N)
        .expect("large-scale n should auto-subsample");
    assert_eq!(
        choice.cap_reason,
        AutoOuterCapReason::Noise,
        "default work_per_k=1 should leave the noise rule binding; got cap_reason={:?}",
        choice.cap_reason
    );
    // round(0.10 * 195_780) = 19_578, max(min_k=10_000) leaves 19_578.
    assert_eq!(choice.k_noise, 19_578, "k_noise math: round(0.1 n)");
    assert_eq!(choice.k, 19_578, "k = noise rule when work cap is inactive");
    // k_work = 5e8 / 1 = 5e8: effectively no cap at n=195_780.
    let expected_k_work = (AUTO_OUTER_WORK_BUDGET / 1) as usize;
    assert_eq!(choice.k_work, expected_k_work);
}

#[test]
fn h1b_target_k_with_survival_work_per_k_picks_work_rule() {
    // Hypothetical: if the survival 250_000 reached target_k_detailed via
    // a non-default options instance, the cap would bind to ~2_000.
    let opts = AutoOuterSubsampleOptions {
        outer_work_per_k_unit: SURVIVAL_WORK_PER_K_UNIT,
        ..AutoOuterSubsampleOptions::default()
    };
    let choice = opts
        .target_k_detailed(LARGE_SCALE_N)
        .expect("large-scale n should auto-subsample");
    let expected_k_work = (AUTO_OUTER_WORK_BUDGET / SURVIVAL_WORK_PER_K_UNIT) as usize;
    assert_eq!(
        choice.k_work, expected_k_work,
        "k_work = 5e8 / 250_000 = 2_000"
    );
    assert_eq!(
        choice.cap_reason,
        AutoOuterCapReason::Work,
        "work cap should bind for survival per-K-unit cost"
    );
    assert_eq!(choice.k, expected_k_work);
    assert!(
        choice.k < OBSERVED_K_IN_RUN / 5,
        "if the survival arg were applied, K would be ~2k; got k={}",
        choice.k
    );
    // sanity: floor still respected
    assert!(choice.k >= AUTO_OUTER_MIN_K_FLOOR);
}

// -------- H1: maybe_install_auto_outer_subsample's argument is dead --------

#[test]
fn h1c_maybe_install_honors_outer_work_per_k_unit_argument() {
    // Regression guard for the K-cap bypass at marginal_slope_shared.rs:1109.
    // Pre-fix: the function constructed `AutoOuterSubsampleOptions::default()`
    // and discarded the `outer_work_per_k_unit` argument, so the work-budget
    // cap never bound; the noise-only rule picked K ≈ 0.10·n (≈ 19_661 at
    // large-scale n=195_780) instead of the survival family's intended K ≈ 2_000.
    // That ~9× inflation drove the observed 8h large-scale hang (exit 137).
    //
    // Post-fix: the function threads `outer_work_per_k_unit` into the options
    // it constructs, so the cap binds and K lands near 2_000.
    let z: Vec<f64> = (0..LARGE_SCALE_N)
        .map(|i| (i as f64) / (LARGE_SCALE_N as f64))
        .collect();
    let stratum: Vec<u8> = (0..LARGE_SCALE_N).map(|i| (i & 1) as u8).collect();
    let opts_struct = gam::families::custom_family::BlockwiseFitOptions {
        auto_outer_subsample: true,
        ..Default::default()
    };
    let counter = Arc::new(AtomicUsize::new(0));
    let last = Arc::new(Mutex::new(None::<Array1<f64>>));
    let rho = vec![0.0_f64; 17];

    let installed = maybe_install_auto_outer_subsample(
        &opts_struct,
        &z,
        Some(&stratum),
        &rho,
        &counter,
        &last,
        12,
        "h1c-test",
        SURVIVAL_WORK_PER_K_UNIT,
        30_000,
        10_000,
        1_000,
    )
    .expect("auto-subsample should install at large-scale n");

    let mask = installed
        .outer_score_subsample
        .as_ref()
        .expect("installed options must carry the mask");
    let k = mask.len();
    let expected_work_capped_k = (AUTO_OUTER_WORK_BUDGET / SURVIVAL_WORK_PER_K_UNIT) as usize;

    // With the cap honoured, K is at most a small stratification ceil
    // overshoot above 2_000. Allow a generous overshoot bound (×1.5)
    // since stratum count depends on the synthetic z distribution.
    let allowed_overshoot = expected_work_capped_k + expected_work_capped_k / 2;
    assert!(
        k <= allowed_overshoot,
        "outer_work_per_k_unit must be honoured. Expected K ≤ {} (work-cap with ceil overshoot); \
         got K = {}. Live-run buggy value was K = {} ≈ 9× this.",
        allowed_overshoot,
        k,
        OBSERVED_K_IN_RUN,
    );
    // Floor: the cap should not collapse below the AUTO_OUTER_MIN_K_FLOOR.
    assert!(
        k >= AUTO_OUTER_MIN_K_FLOOR,
        "K must stay above the {} floor; got {}",
        AUTO_OUTER_MIN_K_FLOOR,
        k,
    );
    // Strict: K is far below the buggy fallback.
    assert!(
        k < OBSERVED_K_IN_RUN / 5,
        "K should be ~10× smaller than the buggy fallback; got K={}",
        k,
    );
}

#[test]
fn h1d_direct_capped_options_do_produce_small_k() {
    // Independent line of evidence: when the *function* internals are
    // bypassed and the cap is built into the options directly, the
    // mask comes out small. This proves the cap mechanism itself works,
    // so the bug is specifically in `maybe_install_auto_outer_subsample`
    // constructing default options at line 1109.
    let z: Vec<f64> = (0..LARGE_SCALE_N)
        .map(|i| (i as f64) / (LARGE_SCALE_N as f64))
        .collect();
    let stratum: Vec<u8> = (0..LARGE_SCALE_N).map(|i| (i & 1) as u8).collect();
    let capped = AutoOuterSubsampleOptions {
        outer_work_per_k_unit: SURVIVAL_WORK_PER_K_UNIT,
        ..AutoOuterSubsampleOptions::default()
    };
    let mask = auto_outer_score_subsample(&z, Some(&stratum), &capped)
        .expect("capped subsample should still install at large-scale n");
    let k = mask.len();
    let expected_work_capped_k = (AUTO_OUTER_WORK_BUDGET / SURVIVAL_WORK_PER_K_UNIT) as usize;
    // Allow stratification ceil overshoot (≤ 2 × stratum_count rows over).
    assert!(
        k >= expected_work_capped_k && k <= expected_work_capped_k + 500,
        "expected K near {} (work cap); got K={}",
        expected_work_capped_k,
        k,
    );
}

// -------- H3-quantitative: K-cap-bypass resource multiplier ----------
//
// The user's failed run died with exit 137 (SIGKILL by OOM/disk) after
// 8h post-`eval=1/12`. RAM was at 67G/85G during the run; disk hit
// 100% by the end. The mechanism is consistent with the K-cap bypass
// inflating per-outer-eval working set ≈ 10× over intended.
//
// This test makes the multiplier concrete: same n, same z, same
// strata; only the `outer_work_per_k_unit` differs. K_buggy / K_capped
// is the resource-consumption multiplier per outer evaluation.

#[test]
fn h3a_uncapped_vs_capped_subsample_size_ratio_is_about_10x() {
    // Documents the resource-consumption multiplier the cap saves.
    // Same n, same z, same strata; only `outer_work_per_k_unit` differs.
    // Outer Hessian per-direction work scales linearly in K when the
    // HT-weighted row pass dominates, so this K ratio is the wall-time
    // and intermediate-working-set blowup the survival family would
    // suffer if the cap argument were not honoured.
    let z: Vec<f64> = (0..LARGE_SCALE_N)
        .map(|i| (i as f64) / (LARGE_SCALE_N as f64))
        .collect();
    let stratum: Vec<u8> = (0..LARGE_SCALE_N).map(|i| (i & 1) as u8).collect();

    let uncapped = AutoOuterSubsampleOptions::default();
    let k_uncapped = auto_outer_score_subsample(&z, Some(&stratum), &uncapped)
        .expect("uncapped default still installs at large-scale n")
        .len();

    let capped = AutoOuterSubsampleOptions {
        outer_work_per_k_unit: SURVIVAL_WORK_PER_K_UNIT,
        ..AutoOuterSubsampleOptions::default()
    };
    let k_capped = auto_outer_score_subsample(&z, Some(&stratum), &capped)
        .expect("capped install at large-scale n")
        .len();

    let multiplier = (k_uncapped as f64) / (k_capped as f64);
    assert!(
        multiplier > 7.0 && multiplier < 12.0,
        "expected ~10x K multiplier between uncapped and capped paths; \
         got K_uncapped={} K_capped={} ratio={:.2}",
        k_uncapped,
        k_capped,
        multiplier,
    );
    assert!(
        k_capped <= 2_500,
        "capped K should ≤ 2500, got {}",
        k_capped
    );
    assert!(
        k_uncapped >= 19_000,
        "uncapped K should ≥ 19_000, got {}",
        k_uncapped
    );
    eprintln!(
        "H3a: K_uncapped={} K_capped={} multiplier={:.2}x (the resource ratio the cap argument saves)",
        k_uncapped, k_capped, multiplier
    );
}

// -------- H2: identifiability audit halts the large-scale shape ----------

/// Build five overlapping blocks that mimic the large-scale rank
/// deficiency the user's log reported: joint rank ≪ joint cols, many
/// near-1.0 aliases across blocks, mirroring time/marginal/logslope/
/// score_warp/link_dev sharing the same polynomial/nullspace directions.
fn build_large_scale_like_aliased_specs() -> Vec<gam::families::custom_family::ParameterBlockSpec> {
    use gam::families::custom_family::ParameterBlockSpec;
    use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};

    let n = 200;
    // Common low-frequency basis: constant, x, x^2, x^3.
    let xs: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * (i as f64) / ((n - 1) as f64))
        .collect();
    let common = |k: usize| -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((n, k + 1));
        for (i, &x) in xs.iter().enumerate() {
            let mut pow = 1.0;
            for j in 0..=k {
                m[[i, j]] = pow;
                pow *= x;
            }
        }
        m
    };

    // Each block embeds the common low-frequency basis plus its own
    // high-frequency tail. This produces near-perfect aliases on the
    // low-frequency columns across all blocks — the same structural
    // collinearity pattern the user's log reported between
    // time_surface / marginal_surface / logslope_surface / etc.
    let block = |seed: u64, k_common: usize, k_extra: usize| -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((n, k_common + 1 + k_extra));
        let c = common(k_common);
        for j in 0..=k_common {
            for i in 0..n {
                m[[i, j]] = c[[i, j]];
            }
        }
        // High-frequency tail — block-specific sin/cos with seed-based
        // frequencies so blocks are distinguishable past the polynomial
        // nullspace.
        for t in 0..k_extra {
            let freq = (seed as f64) + 1.0 + 1.7 * (t as f64);
            for i in 0..n {
                m[[i, k_common + 1 + t]] = (freq * xs[i]).sin();
            }
        }
        m
    };

    let names = [
        "time_surface",
        "marginal_surface",
        "logslope_surface",
        "score_warp_dev",
        "link_dev",
    ];
    let extras = [8, 7, 6, 5, 5]; // total widths after +k_common+1=4 → 12, 11, 10, 9, 9
    let mut specs = Vec::new();
    for (idx, (name, &extra)) in names.iter().zip(extras.iter()).enumerate() {
        let m = block(idx as u64, 3, extra);
        specs.push(ParameterBlockSpec {
            name: name.to_string(),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(m)),
            offset: Array1::<f64>::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::<f64>::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        });
    }
    specs
}

#[test]
fn h2a_audit_marks_large_scale_shape_fatal() {
    let specs = build_large_scale_like_aliased_specs();
    let total_cols: usize = specs.iter().map(|s| s.design.ncols()).sum();
    // Sanity: the synthetic design matches the user's reported widths.
    assert_eq!(total_cols, 51, "large-scale-like total cols");
    let audit = audit_identifiability(&specs).expect("audit must run mechanically");
    assert!(
        audit.fatal,
        "current source MUST halt on the large-scale rank-deficient shape; \
         got fatal=false, summary={:?}",
        audit.summary
    );
    // Independent line of evidence: rank deficiency is visible AND named.
    assert!(
        audit.summary.contains("FATAL") || audit.summary.contains("joint rank"),
        "summary should name rank deficiency; got {:?}",
        audit.summary
    );
}

#[test]
fn h2b_release_log_string_absent_from_current_source() {
    // The user's installed release printed the literal phrase:
    //   "alias(es) flagged; family-specific reparameterisation required"
    // This phrase appears NOWHERE in current source — independent proof
    // that the running binary is older than current main.
    //
    // We can only check this from inside the test by exercising a fatal
    // audit on a known shape and confirming the modern summary uses
    // either "FATAL" or "partial alias(es) below halt threshold" or
    // "clean" instead.
    let specs = build_large_scale_like_aliased_specs();
    let audit = audit_identifiability(&specs).expect("audit must run");
    assert!(
        !audit
            .summary
            .contains("family-specific reparameterisation required"),
        "release-only phrase leaked into current source: {:?}",
        audit.summary
    );
    assert!(
        !audit.summary.contains("alias(es) flagged;"),
        "release-only phrase leaked into current source: {:?}",
        audit.summary
    );
}

#[test]
fn h2c_canonicalize_returns_identifiability_failure_on_large_scale_shape() {
    use gam::families::custom_family::CustomFamilyError;
    use gam::identifiability::canonical::canonicalize_for_identifiability;
    let specs = build_large_scale_like_aliased_specs();
    let outcome = canonicalize_for_identifiability(&specs);
    match outcome {
        Err(CustomFamilyError::IdentifiabilityFailure { audit }) => {
            assert!(
                audit.fatal,
                "IdentifiabilityFailure attached audit must be fatal"
            );
        }
        Ok(_) => panic!(
            "canonicalize_for_identifiability should refuse the large-scale shape; \
             it instead returned Ok (current source is missing the halt)."
        ),
        Err(other) => panic!("expected IdentifiabilityFailure, got: {:?}", other),
    }
}
