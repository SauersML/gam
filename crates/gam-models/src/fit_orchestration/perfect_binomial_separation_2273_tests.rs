//! End-to-end CLI-equivalent n-sweep regression for #2273 at the real
//! `fit_from_formula` pipeline layer.
//!
//! #2273's repro is a bare **exact** (not near-, not quasi-) separation: two
//! classes with a genuine gap between them (`y=0` at `x∈[1,·]`, `y=1` at
//! `x∈[10,·]`, step 0.1), reproducing down to n=6 with the default `y ~ x`
//! formula. Before the two landed fixes, the fit hard-failed with NO model
//! at n∈{6,10,20,30,40,100,150} (and a *different* failure signature —
//! `hessian_psd=NO` — at every n tried for `y ~ smooth(x)`), while n∈{60,80,
//! 400} happened to converge: a non-monotonic pass/fail-by-n pattern that was
//! itself the tell that this was a fragile-outer-search bug, not a genuine
//! capacity/identifiability boundary (see the issue's root-cause comments).
//!
//! Two independent fixes landed:
//!   1. `FitConvergenceEvidence::try_from_parts` now mints a
//!      `StalledAtValidMinimum` fit when the analytic outer criterion
//!      certificate certifies (measurement over status-enum taxonomy) — this
//!      is what lets the automatic Firth retry actually rescue a fit instead
//!      of reaching a certified-but-refused inner state.
//!   2. `run_outer`'s stale-tolerance desync fix (7e6af6e): a solver
//!      convergence claim that fails analytic certification is retried once,
//!      re-seeded at the refused checkpoint, so the retry's tolerance anchor
//!      matches the certificate's bound by construction — this is what fixes
//!      the *non*-Firth path's premature 1-outer-iteration give-up.
//!
//! This file is the CLI n-sweep verification the issue explicitly calls out
//! as owed: every n across the reported failure/pass boundary must now
//! return a MODEL (mint), not a hard error, through the production
//! `fit_from_formula` entry point (the same one `gam fit ... --out m.json`
//! resolves to) — the top-level `gam` crate itself cannot build in this
//! environment (a `build.rs` author tripwire, see the #1762 sibling test
//! file), so this exercises the identical formula-fit path one layer down,
//! in `gam-models`, which builds standalone.

use super::entry::fit_from_formula;
use super::request::{FitConfig, FitResult, StandardFitResult};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;

/// Build the issue's EXACT (not statistically near-) separation fixture:
/// `n/2` rows of class 0 at `x = 1.0, 1.1, 1.2, ...` and `n/2` rows of class
/// 1 at `x = 10.0, 10.1, 10.2, ...` — a genuine gap between the two support
/// intervals `[1, 1+0.1·(n/2−1)]` and `[10, 10+0.1·(n/2−1)]`, deterministic
/// (no RNG), mirroring the issue's `sep_n6.csv`/n-sweep table verbatim. `n`
/// must be even (every n the issue reports, 6..400, is).
fn perfectly_separated_binomial(n: usize) -> gam_data::EncodedDataset {
    assert_eq!(n % 2, 0, "perfectly_separated_binomial requires an even n");
    let half = n / 2;
    let headers: Vec<String> = ["x", "y"].iter().map(|s| s.to_string()).collect();
    let mut rows = Vec::with_capacity(n);
    for i in 0..half {
        let x = 1.0 + 0.1 * i as f64;
        rows.push(StringRecord::from(vec![x.to_string(), "0".to_string()]));
    }
    for i in 0..half {
        let x = 10.0 + 0.1 * i as f64;
        rows.push(StringRecord::from(vec![x.to_string(), "1".to_string()]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// Fit `formula` on the exact-separation fixture at `n` through the
/// production formula-fit entry point and assert a MODEL is minted (not a
/// hard error) — the #2273 contract. `firth` mirrors the CLI's `--firth`
/// flag; `false` is the default CLI path (automatic Firth retry still
/// engages internally when the base error is Firth-retryable, per
/// `firth_can_rescue` in `fit_orchestration/fit.rs`).
fn assert_exact_separation_mints(n: usize, formula: &str, firth: bool) {
    let ds = perfectly_separated_binomial(n);
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        firth,
        ..FitConfig::default()
    };

    let result = fit_from_formula(formula, &ds, &cfg).unwrap_or_else(|err| {
        panic!(
            "#2273: exact-separation fit (n={n}, formula={formula:?}, firth={firth}) \
             must mint a model, not hard-fail with no fit at all: {err}"
        )
    });

    let StandardFitResult { fit, .. } = match result {
        FitResult::Standard(s) => s,
        _ => panic!("expected Standard fit for n={n}, formula={formula:?}"),
    };

    // The fit existing at all is the mint proof under the sealed
    // `FitConvergenceEvidence` contract (mirroring the #1762 sibling test):
    // a refused non-converged state surfaces as a typed `Err` from
    // `fit_from_formula` and trips the `unwrap_or_else` panic above.
    let edf = fit.edf_total().unwrap_or(f64::NAN);
    let gnorm = fit.outer_gradient_norm.unwrap_or(f64::NAN);
    eprintln!(
        "#2273 exact-separation n-sweep: n={n} formula={formula:?} firth={firth} \
         minted=yes edf={edf:.3} |g|={gnorm:.3e}"
    );
    assert!(
        edf.is_finite() && edf > 0.0,
        "#2273: minted fit (n={n}, formula={formula:?}) must report a finite \
         positive edf, got {edf}"
    );
}

/// The issue's exact n-sweep table for the default `y ~ x` formula
/// (automatic-retry path, no explicit `--firth`): every listed n — both the
/// ones that hard-failed pre-fix (6, 10, 20, 30, 40, 100, 150) and the ones
/// that happened to already converge (60, 80, 400) — must mint a model.
/// The non-monotonic fail/pass-by-n pattern in the original report is
/// exactly why every one of these is asserted individually rather than
/// spot-checking a single n.
#[test]
fn exact_separation_linear_n_sweep_mints_2273() {
    for &n in &[6usize, 10, 20, 30, 40, 60, 80, 100, 150, 400] {
        assert_exact_separation_mints(n, "y ~ x", false);
    }
}

/// The `smooth(x)` variant the issue reports failing at every n tried
/// (40/60/80/100) with a *different* signature (`hessian_psd=NO`, a
/// genuinely indefinite outer REML Hessian rather than a gradient-tolerance
/// miss) — a separate code path from the linear-term sweep above, so it gets
/// its own regression rather than riding the linear assertion.
#[test]
fn exact_separation_smooth_n_sweep_mints_2273() {
    for &n in &[40usize, 60, 80, 100] {
        assert_exact_separation_mints(n, "y ~ smooth(x)", false);
    }
}

/// The issue's explicit `--firth` repro at n=40 (`gam fit ... --firth`,
/// which pre-fix reached exactly `StalledAtValidMinimum` and was refused by
/// the strict fit-assembly gate even though the certified stationarity
/// residual was five orders of magnitude inside its own bound). Explicit
/// Firth must mint here too, independent of the automatic-retry path
/// exercised by the two sweeps above.
#[test]
fn exact_separation_explicit_firth_mints_2273() {
    assert_exact_separation_mints(40, "y ~ x", true);
}
