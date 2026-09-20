#![cfg(test)]
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
//! ## What makes each arm pass
//!
//! Two fixes landed first and are what the `y ~ x` logit sweep and the explicit
//! `--firth` arm rest on:
//!   1. `FitConvergenceEvidence::try_from_parts` mints a
//!      `StalledAtValidMinimum` fit when the analytic outer criterion
//!      certificate certifies (measurement over status-enum taxonomy) — this
//!      is what lets the automatically armed Jeffreys-prior fit certify
//!      instead of reaching a certified-but-refused inner state.
//!   2. `run_outer`'s stale-tolerance desync fix (7e6af6e): a solver
//!      convergence claim that fails analytic certification is retried once,
//!      re-seeded at the refused checkpoint, so the retry's tolerance anchor
//!      matches the certificate's bound by construction — this is what fixes
//!      the *non*-Firth path's premature 1-outer-iteration give-up.
//!
//! They were not enough, and the issue was reopened. The `smooth(x)` and
//! non-logit arms needed four more, each a separate defect:
//!   3. `detect_logit_instability` turned a converged, finite, well-penalized
//!      logit fit into `PirlsStatus::Unstable` whenever its fitted η separated
//!      the classes — which on separable data is what the CORRECT fit does. The
//!      criterion was `+∞` over the whole region containing its own optimum, so
//!      `y ~ smooth(x)` could not be fitted at any `n`. Under a penalty that
//!      covers the separating direction the penalized objective is coercive, so
//!      `β̂(λ)` is finite and unique even under exact separation; the saturation
//!      heuristics are gone from the penalized branch and the genuinely unbounded
//!      λ are refused by measurement instead.
//!   4. The Jeffreys coefficient Hessian `HΦ`, which `WorkingState.hessian`
//!      deliberately omits, reached only ONE of the four places that invert that
//!      Hessian. The augmented-square-root direction solve folded it in by
//!      congruence, but that route is gated on Fisher curvature — true only for
//!      the canonical logit link — so every non-canonical binomial link solved a
//!      system with the Jeffreys score in the gradient and no Jeffreys curvature
//!      in the matrix, and contracted linearly (0.4937/iteration, measured)
//!      instead of converging. The dense direction solves, the undamped Newton
//!      polish and the exact-decrement certificate now all use the objective's
//!      own curvature.
//!   5. The Bernoulli variance was rebuilt from `μ` alone, so once `μ` rounds to
//!      exactly `1.0` (cloglog at η ≈ 3.62, probit at η ≈ 8.29) `V = μ(1−μ)`
//!      collapsed to zero and every observed-information quantity divided by it.
//!      The link's cancellation-free tail complement now reaches the variance and
//!      the working residual.
//!   6. The observed weight's closed forms divided by `φV²`, `φV³` and `φV⁴`,
//!      and `V⁴` underflows at `V = 2.9e-84`. They are now the Leibniz recurrence
//!      the third derivative already used, which divides by `φV` once per order.
//!
//! This file is the CLI n-sweep verification the issue explicitly calls out
//! as owed: every n across the reported failure/pass boundary must now
//! return a MODEL (mint), not a hard error, through the production
//! `fit_from_formula` entry point (the same one `gam fit ... --out m.json`
//! resolves to) — the top-level `gam` crate itself cannot build in this
//! environment (a `build.rs` author tripwire, see the #1762 sibling test
//! file), so this exercises the identical formula-fit path one layer down,
//! in `gam-models`, which builds standalone.
//!
//! The single declaration of this module in `fit_orchestration.rs` is
//! `#[cfg(test)] mod ...;`, so this file is test-only. Declaring that
//! scope here with an inner attribute makes it a claim the compiler
//! enforces rather than a claim in a filename.
#![cfg(test)]

use super::entry::fit_from_formula;
use super::request::{FitConfig, FitResult, StandardFitResult};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;

/// Build the issue's separation fixture: `n/2` rows of class 0 at
/// `x = 1.0, 1.1, 1.2, ...` and `n/2` rows of class 1 at
/// `x = 10.0, 10.1, 10.2, ...`, deterministic (no RNG), mirroring the issue's
/// `sep_n6.csv`/n-sweep table verbatim. `n` must be even (every n the issue
/// reports, 6..400, is).
///
/// The support intervals `[1, 1+0.1·(n/2−1)]` and `[10, 10+0.1·(n/2−1)]` have
/// a genuine gap only while `1+0.1·(n/2−1) < 10`, i.e. `n < 182`
/// ([`fixture_is_separated`]). The issue's n=400 row overlaps: its classes
/// share `x ∈ [10, 20.9]` and it has a finite maximum likelihood.
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

/// Whether [`perfectly_separated_binomial`]'s two class supports are
/// disjoint, i.e. the largest class-0 `x` lies below the smallest class-1 `x`.
fn fixture_is_separated(n: usize) -> bool {
    1.0 + 0.1 * ((n / 2 - 1) as f64) < 10.0
}

/// Fit `formula` on the exact-separation fixture at `n` through the
/// production formula-fit entry point and assert a MODEL is minted (not a
/// hard error) — the #2273 contract. `firth` mirrors the CLI's `--firth`
/// flag; `false` is the default CLI path, where the solver arms the Jeffreys
/// prior before its first solve when the realized design certifies separation
/// (`arm_jeffreys_on_prefit_binomial_separation`, #3129).
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
    // #3129: the estimator is a function of the data. A separated design is
    // fitted under the Jeffreys prior from the first solve: without `--firth`
    // the prefit certificate armed it and is the recorded reason; with
    // `--firth` the caller asked for it and no certificate was needed. An
    // overlapping design keeps the flat prior unless the caller asked.
    let separated = fixture_is_separated(n);
    assert_eq!(
        fit.artifacts.firth_bias_reduction,
        firth || separated,
        "#3129: fit (n={n}, formula={formula:?}, firth={firth}, separated={separated}) \
         chose the wrong prior"
    );
    assert_eq!(
        fit.artifacts.jeffreys_arming_evidence.is_some(),
        !firth && separated,
        "#3129: fit (n={n}, formula={formula:?}, firth={firth}, separated={separated}) \
         recorded arming evidence {:?}",
        fit.artifacts.jeffreys_arming_evidence
    );
}

/// The issue's exact n-sweep table for the default `y ~ x` formula
/// (automatic arming, no explicit `--firth`): every listed n — both the
/// ones that hard-failed pre-fix (6, 10, 20, 30, 40, 100, 150) and the ones
/// that happened to already converge (60, 80, 400) — must mint a model.
/// The non-monotonic fail/pass-by-n pattern in the original report is
/// exactly why every one of these is asserted individually rather than
/// spot-checking a single n. The n=400 row overlaps, so it is also the
/// control for arming: it is fitted under the flat prior.
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
/// Firth must mint here too, independent of the automatic arming
/// exercised by the two sweeps above.
#[test]
fn exact_separation_explicit_firth_mints_2273() {
    assert_exact_separation_mints(40, "y ~ x", true);
}

/// The coefficient-runaway / Fisher-weight-collapse separation pathology is
/// NOT logit-specific — it afflicts every binomial inverse link. Before the
/// link-general arming, the Firth path in `fit_from_formula` was
/// gated on `is_binomial_logit`, so a NON-logit binomial fit
/// (`link(type=probit)`, `link(type=cloglog)`) on exactly-separated data
/// produced a `RemlDidNotConverge`/PrefitSeparation error that the
/// link gate then refused to act on — a hard failure with no model, off the
/// default link, despite the README promising Firth handles binomial
/// separation.
///
/// The headline case is probit at n=6: measured to halt on a flat-valley
/// outer stall (|g|≈1.9e2 ≫ bound=1.0) and mint only under Firth, with the
/// Jeffreys prior never engaging pre-fix. With arming gated on
/// `LikelihoodSpec::supports_firth()` (Binomial + Fisher-weight-jet link),
/// every one of these mints through the same no-explicit-`--firth`
/// automatic arming the default-link sweeps use. This test goes red if
/// arming is ever narrowed back to the logit special case.
#[test]
fn exact_separation_nonlogit_links_rescued_by_firth_2273() {
    for link in ["probit", "cloglog"] {
        for &n in &[6usize, 10, 40, 100] {
            assert_exact_separation_mints(n, &format!("y ~ x + link(type={link})"), false);
        }
    }
}
