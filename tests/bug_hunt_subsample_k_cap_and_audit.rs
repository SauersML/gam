//! Falsification tests for the biobank 8-hour-hang investigation.
//!
//! Two hypotheses are tested here against the current local source:
//!
//!   H1: `maybe_install_auto_outer_subsample` ignores its
//!       `outer_work_per_k_unit` argument because line 1109 constructs
//!       `AutoOuterSubsampleOptions::default()` (work_per_k=1). Survival
//!       passes 250_000 expecting K_work ≈ 2_000; the run logs K = 19_661.
//!
//!   H2: The current `audit_identifiability` sets `fatal=true` and
//!       `canonicalize_for_identifiability` returns `IdentifiabilityFailure`
//!       on the biobank shape (5 blocks, joint rank ≪ joint cols). The
//!       installed release's audit log shows "alias(es) flagged;
//!       family-specific reparameterisation required" — a phrase absent
//!       from current source — and the run proceeds, so the release lacks
//!       the gate. The test below proves current source halts.
//!
//! No tests are required to run on the live biobank host. These run
//! locally and either PASS (hypothesis confirmed) or FAIL (hypothesis
//! falsified), with `assert!` messages diagnosing the actual values.

use gam::families::marginal_slope_shared::{
    AUTO_OUTER_MIN_K_FLOOR, AUTO_OUTER_WORK_BUDGET, AutoOuterCapReason,
    AutoOuterSubsampleOptions, auto_outer_score_subsample,
    maybe_install_auto_outer_subsample,
};
use gam::solver::identifiability_audit::audit_identifiability;
use ndarray::{Array1, Array2};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;

const BIOBANK_N: usize = 195_780;
const SURVIVAL_WORK_PER_K_UNIT: u64 = 250_000;
const OBSERVED_K_IN_RUN: usize = 19_661;

// -------- H1: AutoOuterSubsampleOptions::target_k_detailed math --------

#[test]
fn h1a_target_k_with_default_work_per_k_picks_noise_rule() {
    let opts = AutoOuterSubsampleOptions::default();
    let choice = opts
        .target_k_detailed(BIOBANK_N)
        .expect("biobank n should auto-subsample");
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
        .target_k_detailed(BIOBANK_N)
        .expect("biobank n should auto-subsample");
    let expected_k_work = (AUTO_OUTER_WORK_BUDGET / SURVIVAL_WORK_PER_K_UNIT) as usize;
    assert_eq!(choice.k_work, expected_k_work, "k_work = 5e8 / 250_000 = 2_000");
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
fn h1c_maybe_install_ignores_outer_work_per_k_unit_argument() {
    // Build a small synthetic outer-score input. The function only uses
    // z + stratum for stratification; the K it picks depends solely on
    // the `AutoOuterSubsampleOptions` it constructs internally.
    let z: Vec<f64> = (0..BIOBANK_N).map(|i| (i as f64) / (BIOBANK_N as f64)).collect();
    let stratum: Vec<u8> = (0..BIOBANK_N).map(|i| (i & 1) as u8).collect();
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
    )
    .expect("auto-subsample should install at biobank n");

    let mask = installed
        .outer_score_subsample
        .as_ref()
        .expect("installed options must carry the mask");
    let k = mask.len();
    let expected_work_capped_k = (AUTO_OUTER_WORK_BUDGET / SURVIVAL_WORK_PER_K_UNIT) as usize;

    // If the argument were threaded through, k should be ~2_000 (with a small
    // stratification ceil overshoot). The bug at marginal_slope_shared.rs:1109
    // throws it away and the function picks ~19_578-19_661 instead.
    //
    // Falsification: this assert FAILS on current source iff the argument
    // is correctly threaded. We expect it to PASS (confirming the bug).
    assert!(
        k > expected_work_capped_k * 3,
        "If outer_work_per_k_unit were honored, mask K would be ~{}; got K={}. \
         Bug at marginal_slope_shared.rs:1109 is FIXED (test should be inverted).",
        expected_work_capped_k,
        k,
    );
    // The observed live-run K=19_661 should be reproduced quantitatively.
    let lo = 19_500;
    let hi = 20_000;
    assert!(
        k >= lo && k <= hi,
        "Buggy path predicts K in [{}, {}] (noise-rule + stratification ceil overshoot); \
         got K={}. Live run logged K={}.",
        lo,
        hi,
        k,
        OBSERVED_K_IN_RUN,
    );
}

#[test]
fn h1d_direct_capped_options_do_produce_small_k() {
    // Independent line of evidence: when the *function* internals are
    // bypassed and the cap is built into the options directly, the
    // mask comes out small. This proves the cap mechanism itself works,
    // so the bug is specifically in `maybe_install_auto_outer_subsample`
    // constructing default options at line 1109.
    let z: Vec<f64> = (0..BIOBANK_N).map(|i| (i as f64) / (BIOBANK_N as f64)).collect();
    let stratum: Vec<u8> = (0..BIOBANK_N).map(|i| (i & 1) as u8).collect();
    let capped = AutoOuterSubsampleOptions {
        outer_work_per_k_unit: SURVIVAL_WORK_PER_K_UNIT,
        ..AutoOuterSubsampleOptions::default()
    };
    let mask = auto_outer_score_subsample(&z, Some(&stratum), &capped)
        .expect("capped subsample should still install at biobank n");
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

// -------- H2: identifiability audit halts the biobank shape ----------

/// Build five overlapping blocks that mimic the biobank-shape rank
/// deficiency the user's log reported: joint rank ≪ joint cols, many
/// near-1.0 aliases across blocks, mirroring time/marginal/logslope/
/// score_warp/link_dev sharing the same polynomial/nullspace directions.
fn build_biobank_like_aliased_specs() -> Vec<gam::families::custom_family::ParameterBlockSpec> {
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
        });
    }
    specs
}

#[test]
fn h2a_audit_marks_biobank_shape_fatal() {
    let specs = build_biobank_like_aliased_specs();
    let total_cols: usize = specs.iter().map(|s| s.design.ncols()).sum();
    // Sanity: the synthetic design matches the user's reported widths.
    assert_eq!(total_cols, 51, "biobank-like total cols");
    let audit = audit_identifiability(&specs).expect("audit must run mechanically");
    assert!(
        audit.fatal,
        "current source MUST halt on the biobank rank-deficient shape; \
         got fatal=false, summary={:?}",
        audit.summary
    );
    // Independent line of evidence: rank deficiency is visible AND named.
    assert!(
        audit.summary.contains("FATAL")
            || audit.summary.contains("joint rank"),
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
    let specs = build_biobank_like_aliased_specs();
    let audit = audit_identifiability(&specs).expect("audit must run");
    assert!(
        !audit.summary.contains("family-specific reparameterisation required"),
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
fn h2c_canonicalize_returns_identifiability_failure_on_biobank_shape() {
    use gam::families::custom_family::CustomFamilyError;
    use gam::solver::identifiability_canonical::canonicalize_for_identifiability;
    let specs = build_biobank_like_aliased_specs();
    let outcome = canonicalize_for_identifiability(&specs);
    match outcome {
        Err(CustomFamilyError::IdentifiabilityFailure { audit }) => {
            assert!(
                audit.fatal,
                "IdentifiabilityFailure attached audit must be fatal"
            );
        }
        Ok(_) => panic!(
            "canonicalize_for_identifiability should refuse the biobank shape; \
             it instead returned Ok (current source is missing the halt)."
        ),
        Err(other) => panic!(
            "expected IdentifiabilityFailure, got: {:?}",
            other
        ),
    }
}
